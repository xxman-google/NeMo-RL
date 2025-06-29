# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import pprint
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.eval_datasets import (
    aime2024,
    gpqa,
    humaneval,
    ifeval,
    math,
    mbpp,
    mmlu,
    mmlu_pro,
)
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.math_environment import MathEnvironment
from nemo_rl.evals.eval import MasterConfig, run_env_eval, setup
from nemo_rl.models.generation import configure_generation_config

TokenizerType = PreTrainedTokenizerBase


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Evaluation with configuration")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Parse known args for the script
    args, remaining = parser.parse_known_args()

    # Convert remaining args to OmegaConf format
    overrides = OmegaConf.from_dotlist(remaining)

    return args, overrides


def setup_data(tokenizer: AutoTokenizer, data_config, env_configs):
    print("Setting up data...")

    # load dataset
    dataset_name = data_config["dataset_name"]
    if dataset_name == "mmlu":
        base_dataset = mmlu.MMLUDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "aime2024":
        base_dataset = aime2024.AIME2024Dataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gpqa":
        base_dataset = gpqa.GPQADataset(
            variant="main",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "humaneval":
        base_dataset = humaneval.HumanEvalDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "ifeval":
        base_dataset = ifeval.IFEvalDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gpqa_diamond":
        base_dataset = gpqa.GPQADataset(
            variant="diamond",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mmlu_pro":
        base_dataset = mmlu_pro.MMLUProDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math":
        base_dataset = math.MathDataset(
            variant="math_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math500":
        base_dataset = math.MathDataset(
            variant="math_500_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mbpp":
        base_dataset = mbpp.MBPPDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mbpp_sanitized":
        base_dataset = mbpp.MBPPDataset(
            variant="sanitized",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    rekeyed_ds = base_dataset.rekeyed_ds

    env = MathEnvironment.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.math_environment.MathEnvironment"
            )
        }
    ).remote(env_configs["math"])

    dataset = AllTaskProcessedDataset(
        dataset=rekeyed_ds,
        tokenizer=tokenizer,
        default_task_data_spec=base_dataset.task_spec,
        task_data_processors=base_dataset.processor,
        max_seq_length=data_config["max_input_seq_length"],
    )

    return dataset, env, tokenizer


def main():
    """Main entry point."""
    # Parse arguments
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "evals", "eval.yaml"
        )

    config = OmegaConf.load(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        override_conf = OmegaConf.from_cli()
        print(f"Overrides: {override_conf}")
        config = OmegaConf.merge(config, override_conf)

    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    print("Applied CLI overrides")

    # Print config
    print("Final config:")
    pprint.pprint(config)

    # Init ray
    init_ray()

    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )

    # Setup data
    (
        dataset,
        env,
        tokenizer,
    ) = setup_data(tokenizer, config["data"], config["env"])

    # Setup
    (
        vllm_generation,
        dataloader,
        master_config,
        logger,
    ) = setup(config, tokenizer, dataset)

    # Run evaluation
    run_env_eval(
        vllm_generation,
        dataloader,
        env,
        master_config,
        logger,
    )


if __name__ == "__main__":
    main()
