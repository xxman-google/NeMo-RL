"""DeepCoder preview code dataset.

Original dataset: https://huggingface.co/datasets/agentica-org/DeepCoder-Preview-Dataset
"""

import json

from datasets import load_dataset, Features, Value, Sequence, List
from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec
from typing import Any, Literal, Optional

_TEST_LENGTH_LIMIT = 50000000

class DeepCoderPreviewDataset:
    def __init__(
        self,
        subset: Literal["lcbv5", "primeintellect"],
        code_exe_dir: str,
        prompt_file: Optional[str],
        system_prompt_file: Optional[str] = None,
    ):
        self.code_exe_dir = code_exe_dir
        ds = load_dataset("agentica-org/DeepCoder-Preview-Dataset", 
                          name=subset,
                          split="train")
        ds = ds.filter(lambda x: len(x["tests"]) < _TEST_LENGTH_LIMIT)
        if subset == "lcbv5":
            ds = ds.filter(self._filter_lcbv5)
        elif subset == "primeintellect":
            ds = ds.filter(self._filter_primeintellect)

        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="DeepCoderPreview",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.code_processor


    def _rekey(self, example: dict[str, Any]):
        return {
            "question": example["problem"],
            "tests": json.loads(example["tests"]),
            "code_exe_dir": self.code_exe_dir,
        }
    
    def _filter_lcbv5(self, example: dict[str, Any]):
        tests = json.loads(example["tests"])
        return len(tests) > 0 and tests[0]["testtype"] == "stdin"
    
    def _filter_primeintellect(self, example: dict[str, Any]):
        tests = json.loads(example["tests"])
        return len(tests) > 0 and tests[0]["type"] == "stdin_stdout"