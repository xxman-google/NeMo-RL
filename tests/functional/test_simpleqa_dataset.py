import random
from datasets import load_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.eval_datasets.simpleqa import SimpleQADataset


def main():
    dataset = SimpleQADataset()

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
        print(f"  Problem: {example['problem']}")
        print(f"  Expected Answer: {example['expected_answer']}")
        print(f"  Subject: {example['subject']}")
        print(f"  Category: {example['category']}")

if __name__ == "__main__":
    main()