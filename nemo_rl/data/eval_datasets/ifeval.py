"""Math dataset and its variants."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class IFEvalDataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset("google/IFEval", split="train")
        ds = ds.map(self._append_postfix_for_prompt)
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
    
    def _append_postfix_for_prompt(self, example, postfix="/no_think"): # /no_think /think
        example["prompt"] += f" {postfix}"
        return example
