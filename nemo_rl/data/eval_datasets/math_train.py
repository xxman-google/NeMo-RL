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

"""Math dataset train split."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec

MATH_TRAIN_JSONS = [
    "https://huggingface.co/datasets/HuggingFaceTB/MATH/raw/main/data/algebra_train.jsonl",
    "https://huggingface.co/datasets/HuggingFaceTB/MATH/raw/main/data/counting_and_probability_train.jsonl",
    "https://huggingface.co/datasets/HuggingFaceTB/MATH/raw/main/data/geometry_train.jsonl",
    "https://huggingface.co/datasets/HuggingFaceTB/MATH/raw/main/data/intermediate_algebra_train.jsonl",
    "https://huggingface.co/datasets/HuggingFaceTB/MATH/raw/main/data/number_theory_train.jsonl",
    "https://huggingface.co/datasets/HuggingFaceTB/MATH/raw/main/data/prealgebra_train.jsonl",
    "https://huggingface.co/datasets/HuggingFaceTB/MATH/raw/main/data/precalculus_train.jsonl",
]


class MathTrainDataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset("json", data_files=MATH_TRAIN_JSONS, split="train")
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="MathTrain",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.data_processor

    def _extract_answer(self, solution: str):
        start_idx = solution.find("\\boxed{") + len("\\boxed")
        s = []
        for idx in range(start_idx, len(solution)):
            c = solution[idx]
            if c == "{":
                s.append("{")
            elif c == "}":
                s.pop()
            if not s:
                break
        return solution[start_idx + 1 : idx]

    def _rekey(self, data: dict[str, Any]):
        solution = data["solution"]
        answer = self._extract_answer(solution)
        return {
            "problem": data["problem"],
            "expected_answer": answer,
            "solution": solution,
        }
