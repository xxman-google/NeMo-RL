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

"""Alpaca Eval 2.0 dataset."""

from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


ALPACA2_JSON = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/raw/main/alpaca_eval_gpt4_baseline.json" 


class Alpaca2Dataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset("json", data_files=ALPACA2_JSON, split="train")
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name="alpaca2",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.alpaca2_processor

    def _rekey(self, data: dict[str, Any]):
        return {
            "prompt": data["instruction"],
            "baseline_model_response": data["output"],
            "baseline_model": data["generator"],
            "dataset": data["dataset"],
        }
