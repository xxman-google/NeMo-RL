"""Add instructions to the SFT training data.

The program randomly selects an instruction from a JSON file and adds it to the question.
"""

import json
import os
import random

from absl import app, flags
from datasets import load_dataset
from tqdm import tqdm

_INPUT_PARQUET_FILES = flags.DEFINE_string(
    "input_parquet_files", None, "Path to the input parquet files."
)
_PROMPT_LIST_JSON = flags.DEFINE_string(
    "prompt_list_json", None, "Path to the JSON file containing a list of prompts."
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Output directory to store the parquet files."
)
_NUM_SHARDS = flags.DEFINE_integer("num_shards", 1, "Number of shards.")


def add_prompt(
    messages: list[dict[str, str]], instructions: list[str]
) -> list[dict[str, str]]:
    responses = [msg["content"] for msg in messages if msg["role"] == "assistant"]
    questions = [msg["content"] for msg in messages if msg["role"] == "user"]
    response = responses[0]
    prompt = random.choice(instructions).format(questions[0])
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def main(_):
    with open(_PROMPT_LIST_JSON.value, "r") as fid:
        instructions = json.load(fid)
    ds = load_dataset("parquet", data_files=_INPUT_PARQUET_FILES.value, split="train")
    ds = ds.map(
        lambda example: {"messages": add_prompt(example["messages"], instructions)}
    )

    if not os.path.exists(_OUTPUT_DIR.value):
        os.makedirs(_OUTPUT_DIR.value)

    num_shards = _NUM_SHARDS.value
    for i in tqdm(range(num_shards), desc="Sharding"):
        shard = ds.shard(num_shards=num_shards, index=i)
        output_path = os.path.join(
            _OUTPUT_DIR.value, f"train-augmented-{i:05d}-of-{num_shards:05d}.parquet"
        )
        print(f"Writing shard {i} to {output_path}...")
        shard.to_parquet(output_path)


if __name__ == "__main__":
    flags.mark_flags_as_required(
        ["input_parquet_files", "prompt_list_json", "output_dir"]
    )
    app.run(main)
