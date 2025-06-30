"""MMLU dataset and its variants."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class MMLUDataset:
    def __init__(
        self,
        language: Literal[
            "AR-XY",
            "BN-BD",
            "DE-DE",
            "EN-US",
            "ES-LA",
            "FR-FR",
            "HI-IN",
            "ID-ID",
            "IT-IT",
            "JA-JP",
            "KO-KR",
            "PT-BR",
            "ZH-CN",
            "SW-KE",
            "YO-NG",
        ] = "EN-US",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset(
            "csv",
            data_files=f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv",
            split="train",
        )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)

        self.task_spec = TaskDataSpec(
            task_name=f"MMLU_{language}",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.multichoice_qa_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "question": data["Question"],
            "options": dict(
                A=data["A"],
                B=data["B"],
                C=data["C"],
                D=data["D"],
            ),
            "answer": data["Answer"],
            "subject": data["Subject"],
        }
