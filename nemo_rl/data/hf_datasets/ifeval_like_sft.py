from typing import Any

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_data(data: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {
        "messages": [
            {
                "role": "user",
                "content": data["prompt"],
            },
            {
                "role": "assistant",
                "content": data["response"],
            },
        ]
    }


def prepare_dataset(
    seed: int = 42,
    val_size: float = 0.05,
    train_sample_ratio: float = 1.0,
) -> dict[str, Dataset | None]:
    
    # Load the original dataset
    original_ds = load_dataset("argilla/ifeval-like-data", "filtered")

    # Split into train and validation sets using HF's train_test_split
    split_ds = original_ds['train'].train_test_split(test_size=val_size, seed=seed)

    # Format the examples
    if train_sample_ratio == 1.0:
        train_formatted = split_ds["train"].map(format_data)
    else:
        train_formatted = split_ds["train"].train_test_split(test_size=1 - train_sample_ratio, seed=seed)["train"].map(format_data)
    val_formatted = split_ds["test"].map(format_data)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class IFEvalLikeSFTDataset:
    def __init__(self, train_sample_ratio: float = 1.0) -> None:
        self.formatted_ds = prepare_dataset(train_sample_ratio=train_sample_ratio)
        self.task_spec = TaskDataSpec(
            task_name="ifeval-like-sft",
        )