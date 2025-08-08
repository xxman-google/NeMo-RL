from datasets import load_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.eval_datasets.arc_agi import ArcAgiDataset


def main():
    dataset = ArcAgiDataset()

    # Show task metadata
    print("== Task Spec ==")
    print(f"Task name: {dataset.task_spec.task_name}")
    print(f"Prompt file: {dataset.task_spec.prompt_file}")
    print(f"System prompt file: {dataset.task_spec.system_prompt_file}")

    # Show a few examples
    print("\n== Sample Rekeyed Data ==")
    for i in range(4):
        example = dataset.rekeyed_ds[i]
        print(f"\nExample {i + 1}:")
        print(f"  Training examples: {example['training_examples']}")
        print(f"  Test input: {example['test_input']}")
        print(f"  Expected output: {example['ground_truth']}")

if __name__ == "__main__":
    main()