import random
from datasets import load_dataset
from torch.utils.data import DataLoader
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.eval_datasets import load_eval_dataset
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.eval_datasets.local_science_dataset import LocalScienceDataset


def main():
    dataset = LocalScienceDataset(
        data_paths="logs/nemotron/Qwen/consensus_examples.parquet",
        problem_key="problems",
        answer_key="expected_answers",
        name="local_science",
        file_format="parquet",
    )

    # Show task metadata
    print("== Task Spec ==")
    print(f"Task name: {dataset.task_spec.task_name}")
    print(f"Prompt file: {dataset.task_spec.prompt_file}")
    print(f"System prompt file: {dataset.task_spec.system_prompt_file}")

    # Show a few examples
    print("\n== Sample Rekeyed Data ==")
    for i in range(2):
        example = dataset.rekeyed_ds[i]
        print(f"\nExample {i + 1}:")
        print(f"  Problem: {example['problem']}")
        print(f"  Answer: {example['expected_answer']}")

    # Test dataset loading
    print("\n== Testing data loading")

    data_config = DataConfig(
        data_paths="logs/nemotron/Qwen/consensus_examples.parquet",
        problem_key="problems",
        answer_key="expected_answers",
        dataset_name="local_science",
        split=None,
        file_format="parquet",
        prompt_file=None,
        system_prompt_file=None,
        max_input_seq_length=2048,
    )
    tokenizer_config = {
        "name": "Qwen/Qwen3-8B",
        "chat_template": "default",
    }
    base_dataset = load_eval_dataset(data_config)
    rekeyed_ds = base_dataset.rekeyed_ds
    tokenizer = get_tokenizer(tokenizer_config)


    dataset = AllTaskProcessedDataset(
        dataset=rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=base_dataset.processor,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    
    for batch in dataloader:
        # get input prompt from message_log
        prompts = []
        for message_log in batch["message_log"]:
            content = [message["content"] for message in message_log]
            content = "\n".join(content)
            prompts.append(content)
        # problems are prompts without chat template
        problems = []
        expected_answers = []
        for info in batch["extra_env_info"]:
            problem = info["problem"]
            problems.append(problem)
            expected_answer = info["ground_truth"]
            expected_answers.append(expected_answer)
        print("\n== Processed Sample 0==")
        print("Prompt: ", prompts[0])
        print("Problem: ", problems[0])
        print("Expected Answer: ", expected_answers[0])
        print("\n== Processed Sample 1==")
        print("Prompt: ", prompts[1])
        print("Problem: ", problems[1])
        print("Expected Answer: ", expected_answers[1])
        break

if __name__ == "__main__":
    main()