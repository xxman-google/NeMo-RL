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


from typing import Any, Literal, Optional, TypedDict

from datasets import concatenate_datasets, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


class WeightedDataset(TypedDict):
    file_format: Literal["json", "parquet", "huggingface", "csv"]
    name_or_paths: str | list[str] | dict[str, str]
    samples: int
    additional_kwargs: dict[str, Any]
    user_msg_postfix: Optional[str]


def _append_postfix_for_user_msg(example, postfix):
    for msg in example["messages"]:
        if msg["role"] == "user":
            msg["content"] += f" {postfix}"
    return example


class DatasetMixture:

    def __init__(
        self,
        mixture: list[WeightedDataset],
        val_size: float = 0.05,
        seed: int = 42,
    ):
        """Initialize a mixture of datasets.

        Args:
            mixture: Configuration of the mixture.
            val_size: Percentage of the data used for validation.
            seed: Random seed.
        """
        datasets = []
        for weighted_ds in mixture:
            file_format = weighted_ds["file_format"]
            if weighted_ds["file_format"] != "huggingface":
                ds = load_dataset(
                    file_format,
                    data_files=weighted_ds["name_or_paths"],
                    **weighted_ds["additional_kwargs"],
                )
            else:
                ds = load_dataset(
                    weighted_ds["name_or_paths"], **weighted_ds["additional_kwargs"]
                )
            if weighted_ds.get("user_msg_postfix", None):
                ds = ds.map(lambda example: _append_postfix_for_user_msg(example, postfix=weighted_ds["user_msg_postfix"]))
            target_samples = weighted_ds["samples"]
            ds.shuffle(seed=seed)
            if target_samples > len(ds):
                print(
                    f"Requested samples: {target_samples} is greater than the actual dataset size: {len(ds)}."
                )
            datasets.append(ds.select(range(min(target_samples, len(ds)))))

        combined_dataset = concatenate_datasets(datasets)
        # shuffle by default inside train_test_split()
        split_ds = combined_dataset.train_test_split(test_size=val_size, seed=seed)
        self.formatted_ds = {
            "train": split_ds.pop("train"),
            "validation": split_ds.pop("test"),
        }

        self.task_spec = TaskDataSpec(
            task_name="DatasetMixture",
            prompt_file=None,
            system_prompt_file=None,
        )
