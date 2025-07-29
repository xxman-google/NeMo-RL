from typing import Any

from datasets import Dataset, load_dataset

from nemo_rl.data.interfaces import TaskDataSpec


def format_data(data: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {"messages": data["messages"]}


def prepare_dataset(
    seed: int = 42,
    val_size: float = 0.05,
) -> dict[str, Dataset | None]:
    
    # Load the original dataset
    original_ds = load_dataset("allenai/tulu-3-sft-mixture")

    # Split into train and validation sets using HF's train_test_split
    split_ds = original_ds['train'].train_test_split(test_size=val_size, seed=seed)

    # Format the examples
    train_formatted = split_ds["train"].map(format_data)
    val_formatted = split_ds["test"].map(format_data)

    return {
        "train": train_formatted,
        "validation": val_formatted,
    }


class Tulu3SftMixtureDataset:
    def __init__(self) -> None:
        self.formatted_ds = prepare_dataset()
        self.task_spec = TaskDataSpec(
            task_name="tulu-3-sft-mixture",
        )