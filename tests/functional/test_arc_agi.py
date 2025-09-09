from datasets import load_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.eval_datasets.arc_agi import ArcAgiDataset


ARC_AGI_TRAIN_NUM_EXAMPLES = 400
ARC_AGI_EVAL_NUM_EXAMPLES = 400

ARC_AGI2_TRAIN_NUM_EXAMPLES = 1000
ARC_AGI2_EVAL_NUM_EXAMPLES = 120


def run_arc_agi_tests(version: str):
    if version == "v1":
        num_eval_examples = ARC_AGI_EVAL_NUM_EXAMPLES
        num_train_examples = ARC_AGI_TRAIN_NUM_EXAMPLES
        print("Running ARC-AGI-1 Dataset Test...")
    elif version == "v2":
        num_eval_examples = ARC_AGI2_EVAL_NUM_EXAMPLES
        num_train_examples = ARC_AGI2_TRAIN_NUM_EXAMPLES
        print("Running ARC-AGI-2 Dataset Test...")

    print("===================================")
    dataset = ArcAgiDataset(version=version)

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

    # Check dataset stats.
    train = ArcAgiDataset(split="training", version=version)
    assert len(train.rekeyed_ds) == num_train_examples, (
        f"Expected {num_train_examples} training examples, but got {len(train.rekeyed_ds)}"
    )

    eval = ArcAgiDataset(split="evaluation", version=version)
    assert len(eval.rekeyed_ds) == num_eval_examples, (
        f"Expected {num_eval_examples} evaluation examples, but got {len(eval.rekeyed_ds)}"
    )

    for example in train.rekeyed_ds:
        assert "training_examples" in example
        assert isinstance(example["training_examples"], list)
        assert "test_input" in example
        assert isinstance(example["test_input"], list)
        assert "ground_truth" in example
        assert isinstance(example["ground_truth"], list)

    for example in eval.rekeyed_ds:
        assert "training_examples" in example
        assert isinstance(example["training_examples"], list)
        assert "test_input" in example
        assert isinstance(example["test_input"], list)
        assert "ground_truth" in example
        assert isinstance(example["ground_truth"], list)

    print("✅ All checks passed.")
    print("===================================\n")


def main():
    run_arc_agi_tests(version="v1")
    run_arc_agi_tests(version="v2")

if __name__ == "__main__":
    main()