"""DeepCoder preview code dataset.

Original dataset: https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset
"""

import json

from datasets import load_dataset, Features, Value, Sequence, List
from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec
from typing import Any, Literal, Optional

_TEST_LENGTH_LIMIT = 10000000

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
            converted_tests = [{"input": str(i), "output": str(o)} for i, o in zip(tests["inputs"], tests["outputs"])]            
            tests = converted_tests
        
        return {
            "question": example["problem"],
            "tests": tests,
            "code_exe_dir": self.code_exe_dir,
        }
