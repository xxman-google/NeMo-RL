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

import asyncio
import itertools
import os
import re
from typing import TypedDict

import numpy as np
import ray
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import MathDataConfig, processors
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.evals import eval as eval_lib
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.logger import Logger, LoggerConfig

# ===============================================================================
# Configuration
# ===============================================================================


class RejectionSamplingConfig(TypedDict):
    num_tests_per_prompt: int # Number of tests to run per prompt when rejection = true
    seed: int
    rejection: bool # If false, only sample once and do not reject


class MasterConfig(TypedDict):
    rejection_sampling: RejectionSamplingConfig
    generate: GenerationConfig
    data: MathDataConfig
    env: MathEnvConfig
    cluster: ClusterConfig
    logger: LoggerConfig


# ===============================================================================
# Setup & Initialization
# ===============================================================================


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    dataset: AllTaskProcessedDataset,
) -> tuple[
    VllmGeneration,
    DataLoader,
    MasterConfig,
    Logger,
]:
    """Set up components for rejection sampling.

    Initializes the VLLM model and data loader.

    Args:
        master_config: Configuration settings.
        dataset: Dataset to evaluate on.

    Returns:
        VLLM model, data loader, config, and logger.
    """
    # Extract individual configs for easier access
    rejection_sampling_config = master_config["rejection_sampling"]
    generation_config = master_config["generation"]
    cluster_config = master_config["cluster"]
    logger_config = master_config["logger"]

    # Set seed for reproducibility
    set_seed(rejection_sampling_config["seed"])

    # Check settings
    num_tests_per_prompt = rejection_sampling_config["num_tests_per_prompt"]
    temperature = generation_config["temperature"]
    top_k = generation_config["top_k"]

    if num_tests_per_prompt > 1:
        assert temperature > 0 and top_k != 1, (
            "temperature > 0 and top_k != 1 are required for multiple samples"
        )
    # ==========================
    #           Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #           Data
    # ==========================
    if generation_config["num_prompts_per_step"] == -1:
        generation_config["num_prompts_per_step"] = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=generation_config["num_prompts_per_step"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    print(f"  ✓ Evaluation dataset loaded with {len(dataset)} samples")

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="rejection_sampling_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #           Model
    # ==========================
    print("\n▶ Setting up model...")
    # check backend
    backend = generation_config["backend"]
    assert backend == "vllm", "Only vLLM backend is supported for rejection sampling"

    # initialize vllm generation
    vllm_generation = VllmGeneration(cluster=cluster, config=generation_config)
    print(
        f"  ✓ Using vLLM backend for generation with {generation_config['model_name']}"
    )

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        vllm_generation,
        dataloader,
        master_config,
        logger,
    )


# ===============================================================================
# Rejection sampling
# ===============================================================================


def run_env_rejection_sampling(vllm_generation, dataloader, env, master_config, logger):
    """Main entry point for running rejection sampling using environment.

    Generates model responses and evaluates them by env.

    Args:
        vllm_generation: Model for generating responses.
        dataloader: Data loader with evaluation samples.
        env: Environment that scores responses.
        master_config: Configuration settings.
        logger: Logger for recording evaluation results.
    """
    # Check if async engine is enabled and run appropriate version
    if master_config["generation"]["vllm_cfg"]["async_engine"]:
        if master_config["rejection_sampling"]["rejection"]:
            print("Run async rejection sampling")
            asyncio.run(
                _run_env_rejection_sampling_impl(
                    vllm_generation, dataloader, env, master_config, logger, use_async=True
                )
            )
        else:
            print("Run async sampling")
            asyncio.run(
                _run_env_sampling_impl(
                    vllm_generation, dataloader, env, master_config, logger, use_async=True
                )
            )
    else:
        if master_config["rejection_sampling"]["rejection"]:
            print("Run sync rejection sampling")
            asyncio.run(
                _run_env_rejection_sampling_impl(
                    vllm_generation, dataloader, env, master_config, logger, use_async=False
                )
            )
        else:
            print("Run sync sampling")
            asyncio.run(
                _run_env_sampling_impl(
                    vllm_generation, dataloader, env, master_config, logger, use_async=False
                )
            )


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


def parse_pass_at_k_values(pass_at_ks: str) -> list[int]:
    pattern = r"pass@\d+(?:,\d+)*"
    if re.match(pattern, pass_at_ks) is None:
        raise ValueError(f"{pass_at_ks} is not a valid metric.")
    return list(map(int, pass_at_ks.split("@")[1].split(",")))


async def _run_env_rejection_sampling_impl(
    vllm_generation, dataloader, env, master_config, logger, use_async=False
):
    """Unified implementation for both sync and async rejection sampling."""
    # Extract for easier access
    generation_config = master_config["generation"]
    rejection_sampling_config = master_config["rejection_sampling"]
    metric = rejection_sampling_config["metric"]
    ks = parse_pass_at_k_values(metric)
    logger_config = master_config["logger"]
    num_tests_per_prompt = rejection_sampling_config["num_tests_per_prompt"]
    assert num_tests_per_prompt >= max(ks), (
        "num_tests_per_prompt must be greater than or equal to pass_k_value for pass@k metric"
    )

    # Run rejection sampling loop
    generation_lengths = []
    data = []
    scores = {f"pass@{k}": 0 for k in ks}

    for batch in dataloader:
        # measure multiple samples
        if num_tests_per_prompt > 1:
            batch = batch.repeat_interleave(num_tests_per_prompt)

        # get input prompt from message_log
        prompts = []
        for message_log in batch["message_log"]:
            content = [message["content"] for message in message_log]
            content = "\n".join(content)
            prompts.append(content)
        # problems are prompts without chat template
        problems = []
        for info in batch["extra_env_info"]:
            problem = info["problem"]
            if "options" in info:
                problem = processors.construct_multichoice_prompt(
                    prompt="", question=problem, options=info["options"]
                )
            problems.append(problem)

        # generate by vllm
        inputs = BatchedDataDict({"prompts": prompts})
        output_texts, batch_generation_lengths = await _generate_texts(
            vllm_generation, inputs, use_async
        )
        generation_lengths.extend(batch_generation_lengths)

        # append to message_log
        for idx, output in enumerate(output_texts):
            batch["message_log"][idx].append(
                {
                    "role": "assistant",
                    "content": output,
                }
            )

        # evaluate generations with the environment
        to_env = [
            get_keys_from_message_log(batch["message_log"][i], ["role", "content"])
            for i in range(len(batch["message_log"]))
        ]
        env_return = ray.get(env.step.remote(to_env, batch["extra_env_info"]))
        rewards = itertools.batched(env_return.rewards.tolist(), num_tests_per_prompt)
        output_texts = itertools.batched(output_texts, num_tests_per_prompt)
        problems = itertools.batched(problems, num_tests_per_prompt)
        for chunk_rewards, chunk_outputs, chunk_problems in zip(
            rewards, output_texts, problems
        ):
            sorted_indices = np.argsort(chunk_rewards)
            last_idx = sorted_indices[-1]
            if chunk_rewards[last_idx] == 0.0:
                continue
            messages = [
                {"role": "user", "content": chunk_problems[last_idx]},
                {"role": "assistant", "content": chunk_outputs[last_idx]},
            ]
            data.append({"messages": messages})

        for k in ks:
            cur_score = eval_lib.eval_pass_k(
                env_return.rewards,
                num_tests_per_prompt,
                k,
            )
            scores[f"pass@{k}"] += cur_score

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    vllm_generation.shutdown()
    _print_results(
        master_config, generation_config, scores, len(dataloader.dataset), logger
    )
    write_to_parquet(
        Dataset.from_list(data),
        num_shards=logger_config["num_output_shards"],
        output_dir=logger_config["output_dir"],
    )


async def _run_env_sampling_impl(
    vllm_generation, dataloader, env, master_config, logger, use_async=False
):
    """Unified implementation for both sync and async sampling without rejection."""
    # Extract for easier access
    generation_config = master_config["generation"]
    model_name = generation_config["model_name"]
    rejection_sampling_config = master_config["rejection_sampling"]
    logger_config = master_config["logger"]

    # Run rejection sampling loop
    generation_lengths = []
    data = []

    for batch in dataloader:
        # get input prompt from message_log
        prompts = []
        for message_log in batch["message_log"]:
            content = [message["content"] for message in message_log]
            content = "\n".join(content)
            prompts.append(content)
        # problems are prompts without chat template
        problems = []
        for info in batch["extra_env_info"]:
            problem = info["problem"]
            if "options" in info:
                problem = processors.construct_multichoice_prompt(
                    prompt="", question=problem, options=info["options"]
                )
            problems.append(problem)

        # generate by vllm
        inputs = BatchedDataDict({"prompts": prompts})
        output_texts, batch_generation_lengths = await _generate_texts(
            vllm_generation, inputs, use_async
        )
        generation_lengths.extend(batch_generation_lengths)

        for output, problem in zip(output_texts, problems):
            messages = [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": output},
            ]
            data.append({"messages": messages})

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    vllm_generation.shutdown()
    write_to_parquet(
        Dataset.from_list(data),
        num_shards=logger_config["num_output_shards"],
        output_dir=logger_config["output_dir"],
        basename=model_name,
    )


async def _generate_texts(vllm_generation, inputs, use_async):
    """Generate texts using either sync or async method."""
    if use_async:
        # Use async generation - collect all results
        output_texts, generation_lengths = [], []
        async for idx, result in vllm_generation.generate_text_async(inputs):
            output_texts.append((idx, result["texts"][0]))
            generation_lengths.append((idx, result["generation_lengths"][0]))

        # Sort by index to maintain order
        output_texts.sort(key=lambda x: x[0])
        generation_lengths.sort(key=lambda x: x[0])
        return [text for _, text in output_texts], [l for _, l in generation_lengths]
    else:
        # Use sync generation
        outputs = vllm_generation.generate_text(inputs)
        output_texts = outputs["texts"]
        generation_lengths = outputs["generation_lengths"]
        return output_texts, generation_lengths


def _print_results(
    master_config,
    generation_config,
    scores,
    dataset_size,
    logger,
):
    """Print evaluation results."""
    dataset_name = os.path.basename(master_config["data"]["dataset_name"])
    model_name = os.path.basename(generation_config["model_name"])
    max_new_tokens = generation_config["vllm_cfg"]["max_model_len"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    top_k = generation_config["top_k"]

    print("\n" + "=" * 60)
    print(f"{model_name=} {dataset_name=}")
    print(f"{max_new_tokens=} {temperature=} {top_p=} {top_k=}\n")
    for metric, score in scores.items():
        print(f"{metric}={score / dataset_size:.4f}")
    print("=" * 60 + "\n")

    columns = ["Metric", "Scores"]
    rows = [[metric, score / dataset_size] for metric, score in scores.items()]
    logger.log_table("Overall Results", columns, rows)
