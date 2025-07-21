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

"""NuminaMath dataset."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class NuminaMathDataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        """Initialize the NuminaMath subset.

        Args:
            prompt_file: Path to the prompt file to use.
        """
        ds = load_dataset(
            "parquet",
            data_files="/tmp/run_outputs/tulu3_sft_math/train-00000-of-00001.parquet",
            split="train",
        )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=["messages"])
        self.task_spec = TaskDataSpec(
            task_name="NuminaMath",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.data_processor

    def _rekey(self, data: dict[str, Any]):
        questions = []
        answers = []
        for msg in data["messages"]:
            if msg["role"] == "user":
                questions.append(msg["content"])
            elif msg["role"] == "assistant":
                answers.append(msg["content"])
        return {
            "problem": questions[0],
            "expected_answer": answers[0],
        }
