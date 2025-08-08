"""Creates a new version of Tulu3 SFT mixture by removing some math subsets."""

import os

from absl import app, flags
from datasets import load_dataset
from tqdm import tqdm

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Output directory to store the parquet files."
)
_NUM_SHARDS = flags.DEFINE_integer("num_shards", 6, "Number of shards.")

MATH_DATASETS = [
    "ai2-adapt-dev/numinamath_tir_math_decontaminated",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
    "allenai/tulu-3-sft-personas-math-filtered",
    "allenai/tulu-3-sft-personas-math-grade-filtered",
]


def filter_code_use(example):
    if example["source"] not in MATH_DATASETS:
        return True
    questions = []
    answers = []
    for msg in example["messages"]:
        if msg["role"] == "user":
            questions.append(msg["content"])
        elif msg["role"] == "assistant":
            answers.append(msg["content"])
    if "```python" in answers[0]:
        return False
    return True


def main(_):
    ds = load_dataset("allenai/tulu-3-sft-mixture-0225", split="train")
    ds = ds.filter(filter_code_use)

    if not os.path.exists(_OUTPUT_DIR.value):
        os.makedirs(_OUTPUT_DIR.value)

    num_shards = _NUM_SHARDS.value
    for i in tqdm(range(num_shards), desc="Sharding"):
        shard = ds.shard(num_shards=num_shards, index=i)
        output_path = os.path.join(
            _OUTPUT_DIR.value, f"train-{i:05d}-of-{num_shards:05d}.parquet"
        )
        print(f"Writing shard {i} to {output_path}...")
        shard.to_parquet(output_path)


if __name__ == "__main__":
    flags.mark_flag_as_required("output_dir")
    app.run(main)
