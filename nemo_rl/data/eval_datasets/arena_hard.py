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

"""Arena-Hard dataset."""

import json
import requests
from typing import Any, Optional, List


from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


_QUESTIONS_PATH = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/refs/heads/main/data/arena-hard-v2.0/question.jsonl"
_O3_MINI_ANSWERS = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/refs/heads/main/data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl"
_GEMINI_2_0_FLASH_001_ANSWERS = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/refs/heads/main/data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl"


class ArenaHardDataset:
    def __init__(
        self,
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        ds = load_dataset(
            "json",
            data_files=_QUESTIONS_PATH,
            split="train",
        )
        self.hard_prompt_baseline_model_answers = self._get_baseline_answers(_O3_MINI_ANSWERS)
        self.creative_writing_baseline_model_answers = self._get_baseline_answers(_GEMINI_2_0_FLASH_001_ANSWERS)

        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)

        self.task_spec = TaskDataSpec(
            task_name="arena_hard",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.arena_hard_processor

    def _rekey(self, data: dict[str, Any]):
        if data["category"] == "hard_prompt":
            baseline = self.hard_prompt_baseline_model_answers[data["uid"]]
        elif data["category"] == "creative_writing":
            baseline = self.creative_writing_baseline_model_answers[data["uid"]]
        else:
            raise ValueError(f"Unknown category: {data['category']}")
        return {
            "uid": data["uid"],
            "category": data["category"],
            "subcategory": data["subcategory"],
            "prompt": data["prompt"],
            "baseline_model": baseline["model"],
            "baseline_model_response": baseline["messages"][-1]["content"]['answer']
        }

    def _get_baseline_answers(self, url: str):
        response = requests.get(url)
        if response.status_code == 200:
            lines = response.text.splitlines()
            data = {}
            buffer = ""
            for line in lines:
                try:
                    buffer += line
                    example = json.loads(buffer)
                    data[example["uid"]] = example
                    buffer = ""
                except json.JSONDecodeError:
                    continue
            return data
        else:
            raise ValueError(f"Failed to retrieve baseline answers from {url}")
