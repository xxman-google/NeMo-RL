"""Math dataset and its variants."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec
import ast


class IFEvalLikeDataset:
    def __init__(
        self,
        source: Optional[str] = "filtered",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset("argilla/ifeval-like-data", source, split="train[:100]")
        if source == "default":
            ds = ds.rename_column("instruction", "prompt")
        ds = ds.filter(self.is_valid_text)
        # for row in ds:
        #     row["kwargs"] = ast.literal_eval(row["kwargs"])
        # ds = ds.map(self.convert_str_to_list)
        self.rekeyed_ds = ds.map(
            self._rekey, remove_columns=["kwargs", "instruction_id_list"]
        )
        self.task_spec = TaskDataSpec(
            task_name="IFEval",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.if_processor

    def _rekey(self, data: dict[str, Any]):
        data["checker_info"] = {
            "instruction_id_list": data["instruction_id_list"],
            "checker_kwargs": data["kwargs"],
            "prompt": data["prompt"],
        }
        return data
    
    def is_valid_text(self, example):
        return "null" not in example["kwargs"]

    def convert_str_to_list(self, example):
        example["kwargs"] = ast.literal_eval(example["kwargs"])
        return example