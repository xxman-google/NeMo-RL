"""Filters samples that have repetition patterns."""

import os
from collections import Counter

import numpy as np
import tqdm
from absl import app, flags
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.evals import visualization as vis_lib
from nemo_rl.rejection_sampling import rejection_sampling
from nemo_rl.utils.logger import Logger, LoggerConfig, WandbConfig

_INPUT_PARQUET_FILES = flags.DEFINE_string(
    "input_parquet_files", None, "Path to the input parquet files."
)
_WANDB_LOG_DIR = flags.DEFINE_string(
    "wandb_log_dir", None, "Path to the log directory."
)
_DATASET_NAME = flags.DEFINE_string("dataset_name", None, "Dataset name.")
_N_TOKENS = flags.DEFINE_integer(
    "n_tokens", 100, "Number of tokens used to detect repetitions."
)
_TOKENIZER = flags.DEFINE_string("tokenizer", "Qwen/Qwen3-14B", "Tokenizer name.")
_REPETITION_ALLOWED = flags.DEFINE_integer(
    "repetition_allowed",
    5,
    "Number of repetitions allowed. If n-gram repetition is greater than this number, the sample is filtered.",
)
_MAX_HTMLS_TO_DISPLAY = flags.DEFINE_integer(
    "max_htmls_to_display",
    10,
    "Maximum HTMLs to display.",
)


def has_repeated_ngrams(
    token_ids: np.ndarray, n: int, max_repetition_allowed: int
) -> bool:
    ngrams = [
        tuple(token_ids[i : i + n].tolist()) for i in range(len(token_ids) - n + 1)
    ]
    counts = Counter(ngrams)
    if not counts:
        return False
    return max(counts.values()) > max_repetition_allowed


def get_prompt_and_response(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompts = []
    responses = []
    for msg in messages:
        if msg["role"] == "user":
            prompts.append(msg["content"])
        elif msg["role"] == "assistant":
            responses.append(msg["content"])
    return prompts[0], responses[0]


def filter_messages(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    n_tokens: int,
    max_repetition_allowed: int,
):
    prompt, response = get_prompt_and_response(messages)
    token_ids = tokenizer(response, return_tensors="np")["input_ids"][0]
    skip = has_repeated_ngrams(
        token_ids,
        n=n_tokens,
        max_repetition_allowed=max_repetition_allowed,
    )
    if skip:
        html = vis_lib.MathRenderTemplate().render(
            prompt=prompt,
            response=response,
            score=None,
            correct_answer=None,
            extracted_answer=None,
        )
        return None, html
    return messages, None


def main(_):
    logger_config = LoggerConfig(
        log_dir=_WANDB_LOG_DIR.value,
        wandb_enabled=True,
        tensorboard_enabled=False,
        wandb=WandbConfig(project="data_filtering", name=_DATASET_NAME.value),
        monitor_gpus=False,
    )
    logger = Logger(logger_config)
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER.value)
    ds = load_dataset("parquet", data_files=_INPUT_PARQUET_FILES.value, split="train")
    messages = ds["messages"]
    num_total_samples = len(messages)
    filtered_messages, htmls = [], []
    for msg in tqdm.tqdm(messages):
        filtered_msg, html = filter_messages(
            msg,
            tokenizer,
            n_tokens=_N_TOKENS.value,
            max_repetition_allowed=_REPETITION_ALLOWED.value,
        )
        if html is not None:
            htmls.append(html)
        if filtered_msg is not None:
            filtered_messages.append({"messages": filtered_msg})
    num_filtered_samples = len(filtered_messages)
    max_htmls_to_display = min(len(htmls), _MAX_HTMLS_TO_DISPLAY.value)
    all_htmls = vis_lib.make_report_from_example_htmls(htmls[:max_htmls_to_display])
    logger.log_html("Repetition Samples", all_htmls)
    print("\n" + "=" * 60)
    print(f"{num_total_samples=} {num_filtered_samples=}")
    print("=" * 60 + "\n")
    rejection_sampling.write_to_parquet(
        Dataset.from_list(filtered_messages),
        num_shards=1,
        output_dir=os.path.dirname(_INPUT_PARQUET_FILES.value),
        basename="train-filtered",
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("input_parquet_files")
    app.run(main)
