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
import collections
import json
import os
from typing import Optional, TypedDict

import ray
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import MathDataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.llm_message_utils import get_keys_from_message_log
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.environments.code_environment import CodeEnvConfig
from nemo_rl.environments.math_environment import MathEnvConfig
from nemo_rl.evals import visualization as vis_lib
from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy import TokenizerConfig
from nemo_rl.utils.logger import Logger, LoggerConfig

# ===============================================================================
# Configuration
# ===============================================================================


class EvalConfig(TypedDict):
    metric: str
    num_tests_per_prompt: int
    seed: int
    pass_k_value: int
    save_path: str | None


class MasterConfig(TypedDict):
    eval: EvalConfig
    generation: GenerationConfig  # Fixed: was 'generate'
    tokenizer: TokenizerConfig  # Added missing tokenizer key
    data: MathDataConfig
    env: CodeEnvConfig | MathEnvConfig
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
    """Set up components for model evaluation.

    Initializes the VLLM model and data loader.

    Args:
        master_config: Configuration settings.
        dataset: Dataset to evaluate on.

    Returns:
        VLLM model, data loader, config, and logger.
    """
    # Extract individual configs for easier access
    eval_config = master_config["eval"]
    generation_config = master_config["generation"]
    cluster_config = master_config["cluster"]
    logger_config = master_config["logger"]

    # Set seed for reproducibility
    set_seed(eval_config["seed"])

    # Check settings
    metric = eval_config["metric"]
    pass_k_value = eval_config["pass_k_value"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]
    temperature = generation_config["temperature"]
    top_k = generation_config["top_k"]

    # TODO @yukih: support cons@k
    # Validate metrics
    assert metric in ["pass@k"], f"Invalid metric: {metric}"
    if num_tests_per_prompt > 1:
        assert temperature > 0 and top_k != 1, (
            "temperature > 0 and top_k != 1 are required for multiple samples"
        )
    # ==========================
    #           Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    assert pass_k_value >= 1, (
        "pass_k_value must be greater than or equal to 1 for pass@k metric"
    )
    assert num_tests_per_prompt >= pass_k_value, (
        "num_tests_per_prompt must be greater than or equal to pass_k_value for pass@k metric"
    )

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
        name="eval_cluster",
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
    assert backend == "vllm", "Only vLLM backend is supported for evaluation"

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
# Evaluation
# ===============================================================================


def eval_pass_k(
    rewards: torch.Tensor,
    num_tests_per_prompt: int,
    k: int,
    subjects: Optional[list[str]] = None,
    subject_counts: Optional[dict[str, int]] = None,
    subject_scores: Optional[dict[str, float]] = None,
) -> float:
    """Evaluate pass@k score using an unbiased estimator.

    Reference: https://github.com/huggingface/evaluate/blob/32546aafec25cdc2a5d7dd9f941fc5be56ba122f/metrics/code_eval/code_eval.py#L198-L213
    Args:
        rewards: Tensor of shape (batch_size * num_tests_per_prompt)
        k: int (pass@k value)

    Returns:
        pass_k_score: float
    """

    def eval_single_chunk(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return float(1.0 - torch.prod(1.0 - k / torch.arange(n - c + 1, n + 1)).item())

    # rewards is a 1d tensor of size (batch_size * num_tests_per_prompt)
    group_rewards = rewards.split(num_tests_per_prompt)
    if subjects is not None:
        group_subjects = [
            subjects[i : i + num_tests_per_prompt]
            for i in range(0, len(subjects), num_tests_per_prompt)
        ]

    pass_k_score = 0.0
    for group_idx, group_reward in enumerate(group_rewards):
        num_correct = group_reward.sum().item()
        cur_score = eval_single_chunk(num_tests_per_prompt, num_correct, k)
        pass_k_score += cur_score
        if subjects is not None:
            subject = group_subjects[group_idx][0]
            subject_counts[subject] += 1
            subject_scores[subject] += cur_score

    return pass_k_score


def run_env_eval(vllm_generation, dataloader, env, master_config, logger):
    """Main entry point for running evaluation using environment.

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
        asyncio.run(
            _run_env_eval_impl(
                vllm_generation, dataloader, env, master_config, logger, use_async=True
            )
        )
    else:
        asyncio.run(
            _run_env_eval_impl(
                vllm_generation, dataloader, env, master_config, logger, use_async=False
            )
        )


async def _run_env_eval_impl(
    vllm_generation, dataloader, env, master_config, logger, use_async=False
):
    """Unified implementation for both sync and async evaluation."""
    # Extract for easier access
    generation_config = master_config["generation"]
    eval_config = master_config["eval"]
    env_type = master_config["env"].get("env_type", "math")
    env_config = master_config["env"].get(env_type)
    logger_config = master_config["logger"]
    metric = eval_config["metric"]
    num_tests_per_prompt = eval_config["num_tests_per_prompt"]
    pass_k_value = eval_config["pass_k_value"]

    # List to collect evaluation data for parquet file
    evaluation_data = []
    metric_group_key = env_config.get("metric_group_key", None)
    render_template = {
        "arc_agi": vis_lib.BaseRenderTemplate,
        "math": vis_lib.MathRenderTemplate,
        "mgsm": vis_lib.MathRenderTemplate,
        "python_code_verify": vis_lib.CodeRenderTemplate,
        "english_multichoice": vis_lib.MathRenderTemplate,
        "multilingual_multichoice": vis_lib.MathRenderTemplate,
        "instruction_following": vis_lib.BaseRenderTemplate,
    }[env_config["worker_type"]]()

    # Run evaluation loop
    score = 0.0
    htmls = []
    generation_lengths = []
    pass_k_value = eval_config["pass_k_value"]
    subject_scores = collections.defaultdict(float)
    subject_counts = collections.defaultdict(int)

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
        rewards = env_return.rewards

        # Collect data for JSON file
        for i, (prompt, output, message_log, reward, extra_info) in enumerate(
            zip(
                prompts,
                output_texts,
                batch["message_log"],
                rewards.tolist(),
                batch["extra_env_info"],
            )
        ):
            evaluation_data.append(
                {
                    "prompt": prompt,
                    "response": output,
                    "reward": reward,
                    "message_log": message_log,
                    "extra_env_info": extra_info,
                    "sample_index": len(evaluation_data),
                }
            )

        for prompt, response, reward, observation in zip(
            prompts,
            output_texts,
            rewards,
            env_return.observations,
        ):
            html = render_template.render(
                prompt=prompt,
                response=response,
                score=reward.item(),
                correct_answer=observation.get("correct_answer", None),
                extracted_answer=observation.get("extracted_answer", None),
            )
            htmls.append(html)

        # update stats
        if metric == "pass@k":
            if metric_group_key is not None:
                subjects = [info[metric_group_key] for info in batch["extra_env_info"]]
            else:
                subjects = None
            cur_score = eval_pass_k(
                rewards,
                num_tests_per_prompt,
                pass_k_value,
                subjects,
                subject_counts,
                subject_scores,
            )
            score += cur_score
        else:
            raise ValueError(f"Invalid metric: {metric}")

    # Cleanup before printing results
    ray.get(env.shutdown.remote())
    vllm_generation.shutdown()

    # Save evaluation data to JSON file if save_path is specified
    save_path = eval_config.get("save_path")
    if evaluation_data and save_path is not None:
        _save_evaluation_data_to_json(evaluation_data, master_config, save_path)

    # Print results
    _print_results(
        master_config,
        generation_config,
        score,
        len(dataloader.dataset),
        metric,
        pass_k_value,
        num_tests_per_prompt,
        logger,
        subject_counts,
        subject_scores,
        htmls,
        generation_lengths,
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


def _save_evaluation_data_to_json(evaluation_data, master_config, save_path):
    """Save evaluation data to a JSON file.

    Args:
        evaluation_data: List of evaluation samples
        master_config: Configuration dictionary
        save_path: Path to save evaluation results. Set to null to disable saving.
                  Example: "results/eval_output" or "/path/to/evaluation_results"
    """
    # Extract configuration information
    config_data = {
        "model_name": master_config["generation"]["model_name"],
        "dataset_name": master_config["data"]["dataset_name"],
        "metric": master_config["eval"]["metric"],
        "pass_k_value": master_config["eval"]["pass_k_value"],
        "num_tests_per_prompt": master_config["eval"]["num_tests_per_prompt"],
        "temperature": master_config["generation"]["temperature"],
        "top_p": master_config["generation"]["top_p"],
        "top_k": master_config["generation"]["top_k"],
    }

    # Create directory if it doesn't exist
    save_dir = save_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Generate file paths within the directory
    eval_data_path = os.path.join(save_dir, "evaluation_data.json")
    config_path = os.path.join(save_dir, "config.json")

    # Prepare the data to save
    data_to_save = {"evaluation_data": evaluation_data}

    # Save configuration to separate JSON file
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"\n✓ Configuration saved to: {config_path}")

    # Process data to make it JSON serializable
    processed_data = []
    for sample in evaluation_data:
        processed_sample = sample.copy()
        # Convert non-serializable objects to strings
        processed_sample["message_log"] = str(sample["message_log"])
        processed_sample["extra_env_info"] = str(sample["extra_env_info"])
        processed_data.append(processed_sample)

    # Update data to save with processed version
    data_to_save["evaluation_data"] = processed_data

    # Save to JSON file
    with open(eval_data_path, "w") as f:
        json.dump(data_to_save, f, indent=2)

    print(f"\n✓ Evaluation data saved to: {eval_data_path}")
    print(f"  Total samples: {len(evaluation_data)}")
    print(f"  File size: {os.path.getsize(eval_data_path) / 1024 / 1024:.2f} MB")


def _print_results(
    master_config,
    generation_config,
    score,
    dataset_size,
    metric,
    pass_k_value,
    num_tests_per_prompt,
    logger,
    subject_counts,
    subject_scores,
    htmls,
    generation_lengths,
):
    """Print evaluation results."""
    dataset_name = os.path.basename(master_config["data"]["dataset_name"])
    model_name = os.path.basename(generation_config["model_name"])
    max_new_tokens = generation_config["vllm_cfg"]["max_model_len"]
    temperature = generation_config["temperature"]
    top_p = generation_config["top_p"]
    top_k = generation_config["top_k"]
    average_score = score / dataset_size

    print("\n" + "=" * 60)
    print(f"{model_name=} {dataset_name=}")
    print(f"{max_new_tokens=} {temperature=} {top_p=} {top_k=}\n")
    print(f"{metric=} {pass_k_value=} {num_tests_per_prompt=}\n")
    print(f"score={average_score:.4f} ({score}/{dataset_size})")
    print("=" * 60 + "\n")
    logger.log_histogram("generation length", generation_lengths, num_bins=10)
    max_samples_to_html = master_config["logger"].get("max_samples_to_html", 30)
    all_htmls = vis_lib.make_report_from_example_htmls(htmls[:max_samples_to_html])
    logger.log_html("Decoding Results", all_htmls)

    if subject_scores:
        columns = ["Subjects", "Scores", "Counts"]
        subject_names = sorted(subject_scores.keys())
        subject_data = [["Overall", average_score, dataset_size]]
        for name in subject_names:
            subject_data.append(
                [
                    name,
                    subject_scores[name] / subject_counts[name],
                    subject_counts[name],
                ]
            )
        logger.log_table("Subject Results", columns, subject_data)
    else:
        columns = ["Name", "Scores", "Counts"]
        logger.log_table(
            "Overall Results", columns, [[dataset_name, average_score, dataset_size]]
        )
