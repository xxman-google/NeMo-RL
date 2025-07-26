"""Creates HTMLs for data visualization."""

from absl import app, flags
from datasets import load_dataset

from nemo_rl.evals import visualization as vis_lib
from nemo_rl.utils.logger import Logger, LoggerConfig, WandbConfig

_INPUT_PARQUET_FILES = flags.DEFINE_string(
    "input_parquet_files", None, "Path to the input parquet files."
)
_WANDB_LOG_DIR = flags.DEFINE_string(
    "wandb_log_dir", None, "Path to the log directory."
)
_DATASET_NAME = flags.DEFINE_string("dataset_name", None, "Dataset name.")
_WHICH_SAMPLES = flags.DEFINE_string(
    "which_samples", "first_30", "`first_n`, `last_n`, or `random_n`"
)


def get_prompt_and_response(messages: list[dict[str, str]]) -> tuple[str, str]:
    prompts = []
    responses = []
    for msg in messages:
        if msg["role"] == "user":
            prompts.append(msg["content"])
        elif msg["role"] == "assistant":
            responses.append(msg["content"])
    return prompts[0], responses[0]


def main(_):
    logger_config = LoggerConfig(
        log_dir=_WANDB_LOG_DIR.value,
        wandb_enabled=True,
        tensorboard_enabled=False,
        wandb=WandbConfig(project="data", name=_DATASET_NAME.name),
        monitor_gpus=False,
    )
    logger = Logger(logger_config)
    ds = load_dataset("parquet", data_files=_INPUT_PARQUET_FILES.value, split="train")
    location, num_samples = _WHICH_SAMPLES.value.split("_")
    num_samples = int(num_samples)
    if location == "first":
        messages = ds["messages"][:num_samples]
    elif location == "last":
        messages = ds["messages"][-num_samples:]
    elif location == "random":
        ds.shuffle()
        messages = ds["messages"][:num_samples]
    else:
        raise ValueError(f"Invalid arg for `which_samples` {_WHICH_SAMPLES.value}.")
    htmls = []
    for msg in messages:
        prompt, response = get_prompt_and_response(msg)
        html = vis_lib.MathRenderTemplate().render(
            prompt=prompt,
            response=response,
            score=None,
            correct_answer=None,
            extracted_answer=None,
        )
        htmls.append(html)

    all_htmls = vis_lib.make_report_from_example_htmls(htmls)
    logger.log_html("Data", all_htmls)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_parquet_files")
    app.run(main)
