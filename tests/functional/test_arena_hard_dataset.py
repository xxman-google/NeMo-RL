from datasets import load_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.eval_datasets.arena_hard import ArenaHardDataset


_BASELINE_MODEL_BY_KEY = {
    "hard_prompt": "o3-mini-2025-01-31",
    "creative_writing": "gemini-2.0-flash-001",
}


def main():
    dataset = ArenaHardDataset()

    # Show task metadata
    print("== Task Spec ==")
    print(f"Task name: {dataset.task_spec.task_name}")
    print(f"Prompt file: {dataset.task_spec.prompt_file}")
    print(f"System prompt file: {dataset.task_spec.system_prompt_file}")

    # Show a few examples
    print("\n== Sample Rekeyed Data ==")
    for i in range(5):
        example = dataset.rekeyed_ds[i]
        print(f"\nExample {i + 1}:")
        print(f"  Category: {example['category']}")
        print(f"  Subcategory: {example['subcategory']}")
        print(f"  Prompt: {example['prompt']}")
        print(f"  Baseline model: {example['baseline_model']}")
        print(f"  Baseline response: {example['baseline_model_response']}")

    for example in dataset.rekeyed_ds:
        assert example["category"] in _BASELINE_MODEL_BY_KEY.keys()
        assert example["baseline_model"] == _BASELINE_MODEL_BY_KEY[example["category"]]

if __name__ == "__main__":
    main()