"""Filters out thinking data that exist in non-thinking dataset."""

import functools
import numpy as np
import os
import tqdm
from absl import app, flags
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from nemo_rl.evals import visualization as vis_lib
from nemo_rl.utils.logger import Logger, LoggerConfig, WandbConfig

_NONTHINKING_PARQUET_FILES = flags.DEFINE_string(
    "nonthinking_parquet_files", None, "Path to the parquet files that contain non-thinking data."
)
_THINKING_PARQUET_FILES = flags.DEFINE_string(
    "thinking_parquet_files", None, "Path to the parquet files that contain thinking data."
)
_NUM_SHARDS = flags.DEFINE_integer("num_shards", 1, "Number of shards.")


def build_key_set(ds: Dataset) -> set[str]:
    keys = set()
    for msg in tqdm.tqdm(ds["messages"]):
        prompt, _ = get_prompt_and_response(msg)
        keys.add(prompt)
    return keys


def get_prompt_and_response(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompts = []
    responses = []
    for msg in messages:
        if msg["role"] == "user":
            prompts.append(msg["content"])
        elif msg["role"] == "assistant":
            responses.append(msg["content"])
    return prompts[0], responses[0]


def not_in_nonthinking_data(example, keys):
    prompt, _ = get_prompt_and_response(example["messages"])
    return prompt not in keys


def main(_):
    nonthinking_ds = load_dataset("parquet", data_files=_NONTHINKING_PARQUET_FILES.value, split="train")
    keys = build_key_set(nonthinking_ds)
    thinking_ds = load_dataset("parquet", data_files=_THINKING_PARQUET_FILES.value, split="train")
    thinking_ds = thinking_ds.filter(functools.partial(not_in_nonthinking_data, keys=keys))

    num_shards = _NUM_SHARDS.value
    output_dir = os.path.dirname(_THINKING_PARQUET_FILES.value)
    for i in tqdm.tqdm(range(num_shards), desc="Sharding"):
        shard = thinking_ds.shard(num_shards=num_shards, index=i)
        output_path = os.path.join(
            output_dir, f"train-nonthink-removed-{i:05d}-of-{num_shards:05d}.parquet"
        )
        print(f"Writing shard {i} to {output_path}...")
        shard.to_parquet(output_path)


if __name__ == "__main__":
    flags.mark_flags_as_required(["thinking_parquet_files", "nonthinking_parquet_files"])
    app.run(main)
