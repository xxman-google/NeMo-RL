"""HumanEval dataset."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class HumanEvalDataset:
    def __init__(self, prompt_file: str, system_prompt_file: Optional[str] = None):
        ds = load_dataset("openai/openai_humaneval", split="test")
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="HumanEval",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.coding_processor

    def _rekey(self, data: dict[str, Any]):
        test = "\n".join([data["test"], f"check({data['entry_point']})"])
        return {
            "question": data["prompt"],
            "tests": test,
        }
