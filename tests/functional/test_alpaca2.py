import random
from datasets import load_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.eval_datasets.alpaca2 import Alpaca2Dataset


def main():
    dataset = Alpaca2Dataset()

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
        print(f"  Prompt: {example['prompt']}")
        print(f"  Baseline model: {example['baseline_model']}")
        print(f"  Baseline model response: {example['baseline_model_response']}")
        print(f"  Dataset/category: {example['dataset']}")

if __name__ == "__main__":
    main()
