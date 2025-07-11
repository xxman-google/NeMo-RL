"""SimpleQA dataset."""

import ast
import random
from typing import Any, Optional
from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class SimpleQADataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset("csv", data_files=f"https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv", split="train")
        self.rekeyed_ds = ds.map(
            self._rekey, remove_columns=ds.column_names
        )
        self.task_spec = TaskDataSpec(
            task_name="SimpleQAEval",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.data_processor

    def _rekey(self, data: dict[str, Any]):
        metadata_json = ast.literal_eval(data["metadata"])
        return {
            "problem": data["problem"],
            "expected_answer": data["answer"],
            "subject": metadata_json["topic"],
            "category": metadata_json["answer_type"]
        }