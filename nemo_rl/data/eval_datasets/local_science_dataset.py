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

"""Local science dataset."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class LocalScienceDataset:
    def __init__(
        self,
        data_paths: str | list[str],
        problem_key: str,
        answer_key: str,
        name: str,
        split: Optional[str] = None,
        file_format: Literal["csv", "json", "parquet"] = "parquet",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset(file_format, data_files=data_paths)
        if split is not None:
            ds = ds[split]
        else:
            ds = ds["train"]
        self._problem_key = problem_key
        self._answer_key = answer_key
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=name,
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.math_rejection_sampling_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "problem": data[self._problem_key],
            "expected_answer": data[self._answer_key],
        }
