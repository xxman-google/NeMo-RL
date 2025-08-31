"""Filters examples with image links."""

import numpy as np
import os
import re
import tqdm
from absl import app, flags
from datasets import load_dataset
from transformers import AutoTokenizer

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

_MAX_NUM_TO_SHOW = 10
_IMG_LINK_PATTERN = r'https?:\/\/[^\s]+?\.(?:jpg|jpeg|png|gif|bmp|webp|svg)(?:\?[^\s]*)?'


def get_prompt_and_response(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompts = []
    responses = []
    for msg in messages:
        if msg["role"] == "user":
            prompts.append(msg["content"])
        elif msg["role"] == "assistant":
            responses.append(msg["content"])
    return prompts[0], responses[0]


def filter_images(example) -> bool:
    prompt, _ = get_prompt_and_response(example["messages"])
    image_urls = re.findall(_IMG_LINK_PATTERN, prompt, flags=re.IGNORECASE)
    return len(image_urls) == 0


def main(_):
    logger_config = LoggerConfig(
        log_dir=_WANDB_LOG_DIR.value,
        wandb_enabled=True,
        mlflow_enabled=False,
        tensorboard_enabled=False,
        wandb=WandbConfig(project="data", name=_DATASET_NAME.value),
        monitor_gpus=False,
    )
    logger = Logger(logger_config)
    ds = load_dataset("parquet", data_files=_INPUT_PARQUET_FILES.value, split="train")
    htmls = []
    pattern = r'https?:\/\/[^\s]+?\.(?:jpg|jpeg|png|gif|bmp|webp|svg)(?:\?[^\s]*)?'

    for msg in tqdm.tqdm(ds["messages"]):
        prompt, response = get_prompt_and_response(msg)
        image_urls = re.findall(pattern, prompt, flags=re.IGNORECASE)
        if image_urls:
            html = vis_lib.MathRenderTemplate().render(
                prompt=prompt,
                response=response,
                score=None,
                correct_answer=None,
                extracted_answer=None,
            )
            htmls.append(html)
    print("Samples with images: ", len(htmls))

    selected_htmls = htmls[:_MAX_NUM_TO_SHOW]
    all_htmls = vis_lib.make_report_from_example_htmls(selected_htmls)
    logger.log_html("Filtered data", all_htmls)

    ds = ds.filter(filter_images)
    rejection_sampling.write_to_parquet(
        ds,
        num_shards=1,
        output_dir=os.path.dirname(_INPUT_PARQUET_FILES.value),
        basename="train-img-filtered",
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("input_parquet_files")
    app.run(main)
