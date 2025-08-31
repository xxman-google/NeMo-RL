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

"""SWE-Bench_Verified dataset with 'oracle' retrieval and style-3 prompt."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


_MAX_PROMPT_LENGTH = 262144  # Maximum length of the prompt in characters
_FILTER = False  # Whether to filter instances based on prompt length
_TRUNCATE = True  # Whether to truncate prompts that exceed the maximum length


class SweBenchVerifiedOracleDataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset(
            "jcpagadora/SWE-bench_Verified__style-3__fs-oracle",
            split="test",
        )
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        if _FILTER:
            print(f"Filtering out instances with prompt len > {_MAX_PROMPT_LENGTH} characters.")
            self.rekeyed_ds = self.rekeyed_ds.filter(self._filter)
        if _TRUNCATE:
            print(f"Truncating instances with prompt len > {_MAX_PROMPT_LENGTH} characters.")
            self.rekeyed_ds = self.rekeyed_ds.map(self._truncate)
        self.task_spec = TaskDataSpec(
            task_name="swebench_verified_oracle",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.swe_bench_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "prompt": data["text"],
            "ground_truth": data["patch"],
            "instance": data,
        }

    def _filter(self, data: dict[str, Any]) -> bool:
        """Filter function to remove instances with empty ground truth."""
        return len(data["prompt"]) <= _MAX_PROMPT_LENGTH

    def _truncate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Truncate the prompt to a maximum length."""
        if len(data["prompt"]) > _MAX_PROMPT_LENGTH:
            data["prompt"] = data["prompt"][:_MAX_PROMPT_LENGTH]
        return data
