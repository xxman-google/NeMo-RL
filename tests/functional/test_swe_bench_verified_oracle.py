from nemo_rl.data.eval_datasets.swe_bench_verified_oracle import SweBenchVerifiedOracleDataset

def main():
    dataset = SweBenchVerifiedOracleDataset()

    # Show task metadata
    print("== Task Spec ==")
    print(f"Task name: {dataset.task_spec.task_name}")
    print(f"Prompt file: {dataset.task_spec.prompt_file}")
    print(f"System prompt file: {dataset.task_spec.system_prompt_file}")

    # Show first example.
    print("\n== Sample Rekeyed Data ==")
    example = dataset.rekeyed_ds[0]
    print(f"  Instance ID: {example['instance_id']}")
    print(f"  Problem: {example['problem']}")
    print(f"  Golden Patch: {example['ground_truth']}")

if __name__ == "__main__":
    main()