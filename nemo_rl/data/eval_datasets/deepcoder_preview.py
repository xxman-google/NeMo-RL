"""DeepCoder preview code dataset.

Original dataset: https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset
"""

import json
import re

from datasets import load_dataset, disable_caching
from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec
from typing import Any, Literal, Optional

disable_caching()

_TEST_LENGTH_LIMIT = 10000000

_FORMATTED_PROMPT_WITH_STARTER_CODE = """
{}

You will use the following starter code to write the solution to the problem and enclose your code within delimiters. Also, remember to import any missing modules in the code.
```python
{}
```
"""

_FORMATTED_PROMPT_GENERAL = """
{}

Remember to import any missing modules in the code, and enclose your code within delimiters like this:
```python
```
"""

_FORMATTED_PROMPT_WITH_FN_NAME = """
{}

You will use the function name `{}` as the code entry point for your solution. Remember to import any missing modules in the code, and enclose your code within delimiters like this:
```python
```
"""

class DeepCoderPreviewDataset:
    def __init__(
        self,
        subset: Literal["lcbv5", "primeintellect", "taco"],
        test_type: Literal["stdio", "functional"],
        code_exe_dir: str,
        prompt_file: Optional[str],
        system_prompt_file: Optional[str] = None,
    ):
        self.subset = subset
        self.test_type = test_type
        self.code_exe_dir = code_exe_dir
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", 
                          name=subset,
                          split="train")
        ds = ds.rename_column("tests", "tests_str")
        ds = ds.filter(lambda x: len(x["tests_str"]) < _TEST_LENGTH_LIMIT)
        ds = ds.filter(lambda x: self._filter_based_on_test_type(x, test_type))
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="DeepCoderPreview",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_processor
        
    def _filter_based_on_test_type(
        self, example: dict[str, Any], test_type: str
    ) -> bool:
        tests = json.loads(example["tests_str"])
        if test_type == "stdio":
            if self.subset == "lcbv5":
                # tests is a list of dict
                return len(tests) > 0 and tests[0]["testtype"] == "stdin"
            elif self.subset == "primeintellect":
                # tests is a list of dict
                return len(tests) > 0 and tests[0]["type"] == "stdin_stdout"
            elif self.subset == "taco":
                # tests is a dict
                return len(tests) > 0 and "fn_name" not in tests
        elif test_type == "functional":
            if self.subset == "lcbv5":
                return len(tests) > 0 and tests[0]["testtype"] == "functional"
            elif self.subset == "primeintellect":
                return len(tests) > 0 and tests[0]["type"] == "function_call"
            elif self.subset == "taco":
                return len(tests) > 0 and "fn_name" in tests
    
    def _rekey(self, example: dict[str, Any]):
        tests = json.loads(example["tests_str"]) 
        if self.subset == "taco":
            # Convert tests from dict to list of dict.
            if self.test_type == "stdio":
                converted_tests = [{"input": str(i), "output": str(o)} for i, o in zip(tests["inputs"], tests["outputs"])]
            elif self.test_type == "functional":
                converted_tests = [{"input": i, "output": o} for i, o in zip(tests["inputs"], tests["outputs"])]         
            tests = converted_tests
        
        if self.test_type == "stdio":
            return {
                "question": example["problem"],
                "tests": tests,
                "code_exe_dir": self.code_exe_dir,
            }
        elif self.test_type == "functional":
            return {
                "question": example["problem"],
                "prompt": self._create_custom_user_prompt(example),
                "tests": self._create_functional_tests(example, tests),
                "code_exe_dir": self.code_exe_dir,
            }
        
    def _create_custom_user_prompt(self, example: dict[str, Any]):
        if self.subset == "lcbv5":
            return _FORMATTED_PROMPT_WITH_STARTER_CODE.format(example["problem"], example["starter_code"]).strip()
        elif self.subset == "primeintellect":
            return _FORMATTED_PROMPT_GENERAL.format(example["problem"]).strip()
        elif self.subset == "taco":
            fn_name = json.loads(example["tests_str"])["fn_name"]
            return _FORMATTED_PROMPT_WITH_FN_NAME.format(example["problem"], fn_name).strip()

    def _create_functional_tests(self, example: dict[str, Any], tests: list[dict]):
        if self.subset == "lcbv5":
            starter_code = example["starter_code"]
            pattern = r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
            match = re.search(pattern, starter_code, re.MULTILINE)
            fn_name = match.group(1)
            all_tests = []
            for test_case in tests:
                inputs = ",".join(test_case["input"].split("\n"))
                output = test_case["output"]
                all_tests.append(f"assert s.{fn_name}({inputs}) == {output}")
            return "\n".join(["s = Solution()"] + all_tests)
        elif self.subset == "primeintellect":
            all_tests = []
            for test_case in tests:
                fn_name = test_case["fn_name"]
                inputs = ",".join(str(x) for x in test_case["input"])
                output = str(test_case["output"][0])
                all_tests.append(f"assert {fn_name}({inputs}) == {output}")
            return "\n".join(all_tests)
        elif self.subset == "taco":
            all_tests = []
            for test_case in tests:
                fn_name = json.loads(example["tests_str"])["fn_name"]
                inputs = ",".join(str(x) for x in test_case["input"])
                output = str(test_case["output"][0])
                all_tests.append(f"assert {fn_name}({inputs}) == {output}")
            return "\n".join(all_tests)
