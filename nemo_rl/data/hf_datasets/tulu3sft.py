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


from datasets import load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class Tulu3SftDataset:
    def __init__(
        self,
        seed: int = 42,
        val_size: float = 0.05,
    ):
        """Initialize the Tulu3 SFT dataset with train/validation split.

        Args:
            seed: Random seed for reproducible splitting
            test_size: Proportion of data to use for validation (0.0-1.0)
        """
        ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
        split_ds = ds.train_test_split(test_size=val_size, seed=seed)
        self.formatted_ds = {
            "train": split_ds.pop("train"),
            "validation": split_ds.pop("test"),
        }

        self.task_spec = TaskDataSpec(
            task_name="Tulu3-SFT",
            prompt_file=None,
        )
