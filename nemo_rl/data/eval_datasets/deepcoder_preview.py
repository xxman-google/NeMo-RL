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
        code_exe_dir: str,
        prompt_file: Optional[str],
        system_prompt_file: Optional[str] = None,
    ):
        self.subset = subset
        self.code_exe_dir = code_exe_dir
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", 
                          name=subset,
                          split="train")
        ds = ds.filter(lambda x: len(x["tests"]) < _TEST_LENGTH_LIMIT)
        ds = ds.filter(self._filter_keep_stdio)
        ds = ds.map(self._transform_tests)
        ds = ds.map(self._rekey)
        self.rekeyed_ds = ds.select_columns(["question", "tests", "code_exe_dir"])
        self.task_spec = TaskDataSpec(
            task_name="DeepCoderPreview",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_processor

    def _filter_keep_stdio(self, example: dict[str, Any]):
        tests = json.loads(example["tests"])
        if self.subset == "lcbv5":
            # tests is a list of dict
            return len(tests) > 0 and tests[0]["testtype"] == "stdin"
        elif self.subset == "primeintellect":
            # tests is a list of dict
            return len(tests) > 0 and tests[0]["type"] == "stdin_stdout"
        elif self.subset == "taco":
            # tests is a dict
            return len(tests) > 0 and "fn_name" not in tests
        
    def _transform_tests(self, example: dict[str, Any]):
        tests = json.loads(example["tests"])
        if self.subset == "taco":
            # Convert tests from dict to list of dict.
            converted_tests = [{"input": i, "output": o} for i, o in zip(tests["inputs"], tests["outputs"])]
            tests = converted_tests

        return {"tests": tests}
    
    def _rekey(self, example: dict[str, Any]):
        return {
            "question": example["problem"],
            "tests": example["tests"],
            "code_exe_dir": self.code_exe_dir,
        }
