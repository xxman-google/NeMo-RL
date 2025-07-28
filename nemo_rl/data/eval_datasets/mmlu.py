# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MMLU dataset and its variants."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

_SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "stem",
    "anatomy": "other",
    "astronomy": "stem",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "other",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "econometrics": "social_sciences",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "stem",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "stem",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "stem",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "stem",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


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
        if language != "EN-US":
            data_files = f"https://openaipublic.blob.core.windows.net/simple-evals/mmlu_{language}.csv"
        else:
            data_files = (
                "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
            )
        ds = load_dataset(
            "csv",
            data_files=data_files,
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
        subject = data["Subject"]
        return {
            "question": data["Question"],
            "options": dict(
                A=data["A"],
                B=data["B"],
                C=data["C"],
                D=data["D"],
            ),
            "answer": data["Answer"],
            "category": _SUBJECT_TO_CATEGORY[subject],
            "subject": subject,
        }
