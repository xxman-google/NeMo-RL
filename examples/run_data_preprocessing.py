"""Creates HTMLs for data visualization."""

from absl import app, flags
from datasets import load_dataset
from transformers import AutoTokenizer
import tqdm
from datasets import Dataset
import os

from nemo_rl.evals import visualization as vis_lib
from nemo_rl.utils.logger import Logger, LoggerConfig, WandbConfig

_INPUT_PARQUET_FILES = flags.DEFINE_string(
    "input_parquet_files", "/home/tangyoubao_google_com/nemo-rl/data/if_rejection_sampling/qwen3_8b_thinking/train-00000-of-00001.parquet", "Path to the input parquet files."
)
_TOKENIZER = flags.DEFINE_string("tokenizer", "Qwen/Qwen3-8B", "Tokenizer name.")


def get_prompt_and_response(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompts = []
    responses = []
    for msg in messages:
        if msg["role"] == "user":
            prompts.append(msg["content"])
        elif msg["role"] == "assistant":
            responses.append(msg["content"])
    return prompts[0], responses[0]

def write_to_parquet(
    dataset: Dataset, num_shards: int, output_dir: str, basename: str = "train"
):
    for i in tqdm.tqdm(range(num_shards), desc="Sharding"):
        shard = dataset.shard(num_shards=num_shards, index=i)
        output_path = os.path.join(
            output_dir, f"{basename}-{i:05d}-of-{num_shards:05d}.parquet"
        )
        print(f"Writing shard {i} to {output_path}...")
        shard.to_parquet(output_path)

def main(_):
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER.value)
    ds = load_dataset("parquet", data_files=_INPUT_PARQUET_FILES.value, split="train")
    messages = ds["messages"]
    results = []
    generation_lengths = []
    i = 0
    for msg in messages:
        i += 1
        print(f"{i}/{len(messages)}")
        prompt, response = get_prompt_and_response(msg)
        token_ids = tokenizer(response, return_tensors="np")["input_ids"][0]
        if len(token_ids) < 2000:
            results.append({"messages": msg})
            generation_lengths.append(len(token_ids))

    print("average token length: ", sum(generation_lengths) / len(generation_lengths))
    write_to_parquet(
        Dataset.from_list(results),
        num_shards=1,
        output_dir="/home/tangyoubao_google_com/nemo-rl/data/if_rejection_sampling/qwen3_8b_thinking_filtered",
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("input_parquet_files")
    app.run(main)
