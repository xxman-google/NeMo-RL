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

"""OpenR1 Math dataset."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class OpenR1MathDataset:
    def __init__(
        self,
        source: Literal[
            "amc_aime",
            "aops_forum",
            "cn_contest",
            "inequalities",
            "number_theory",
            "olympiads",
            "olympiads_ref",
            "all",
        ] = "amc_aime",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset("open-r1/OpenR1-Math-220k", split="train")
        if source != "all":
            ds = ds.filter(
                lambda example: example["source"] == source
                and example["question_type"] == "math-word-problem"
            )
        else:
            ds = ds.filter(
                lambda example: example["question_type"] == "math-word-problem"
            )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="openr1_math",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_rejection_sampling_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "problem": data["problem"],
            "expected_answer": data["answer"],
            "source": data["source"],
        }
