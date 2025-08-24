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

"""SciQ dataset.

Original dataset: https://huggingface.co/datasets/nvidia/Llama-Nemotron-Post-Training-Datase
"""

import random
from typing import Any, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class NemotronDataset:
    def __init__(
        self,
        split: str = "science",
        prompt_file: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
    ):
        print("Loading NemotronDataset from huggingface...")
        ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset", split=split)
        print("Loaded NemotronDataset from huggingface.")
        self._rng = random.Random()
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=ds.column_names)
        self.task_spec = TaskDataSpec(
            task_name=f"Nemotron_{split}",
            prompt_file=prompt_file,
            system_prompt_file=system_prompt_file,
        )
        self.processor = processors.science_rejection_sampling_processor

    def _rekey(self, data: dict[str, Any]):
        """Rekey the data to match the expected format.       
        """
        problem = None
        if isinstance(data["input"], list) and len(data["input"]) > 0 and "content" in data["input"][0]:
            problem = data["input"][0]["content"]
        else:
            problem = str(data["input"])
        return {
            "problem": problem,
            "answer": data["output"],
        }
