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
import os
import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import NotRequired, Optional, TypedDict, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import (
    DPOLossFn,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, dpo_collate_fn
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer


class DPOSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: NotRequired[float]  # Optional field - may not be present during training
    consumed_samples: int


def _default_dpo_save_state() -> DPOSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class DPOConfig(TypedDict):
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int

    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    ## TODO(@ashors) support other loss functions
    ## https://github.com/NVIDIA-NeMo/RL/issues/193
    # preference_loss: str
    # gt_reward_scale: float
    preference_loss_weight: float
    sft_loss_weight: float


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    dpo: DPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: AllTaskProcessedDataset,
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader,
    DPOLossFn,
    Logger,
    CheckpointManager,
    DPOSaveState,
    MasterConfig,
]:
    """Main entry point for running DPO algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    # Make sure we are not using dynamic batching or sequence packing.
    # Anything that changes the order of data within a batch is currently incompatible with DPO.
    assert not master_config["policy"]["dynamic_batching"]["enabled"], (
        "Dynamic batching is currently not supported with DPO. "
        "See https://github.com/NVIDIA-NeMo/RL/issues/719"
    )
    assert not master_config["policy"]["sequence_packing"]["enabled"], (
        "Sequence packing is currently not supported with DPO. "
        "See https://github.com/NVIDIA-NeMo/RL/issues/719"
    )

    set_seed(master_config["dpo"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    dpo_config = master_config["dpo"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    dpo_save_state: Optional[DPOSaveState] = cast(
        Optional[DPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )

    # ==========================
    #           Data
    # ==========================
    ## TODO(@ashors) reduce boilerplate and move reused code into utils
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=partial(
            dpo_collate_fn,
            tokenizer=tokenizer,
            make_sequence_length_divisible_by=policy_config[
                "make_sequence_length_divisible_by"
            ],
        ),
        drop_last=True,
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=dpo_config["val_global_batch_size"],
        shuffle=False,
        collate_fn=partial(
            dpo_collate_fn,
            tokenizer=tokenizer,
            make_sequence_length_divisible_by=policy_config[
                "make_sequence_length_divisible_by"
            ],
        ),
        drop_last=True,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="dpo_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training
    # ==========================
    print("\n▶ Setting up model...")
    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
        init_reference_model=True,
    )
    loss_fn = DPOLossFn(master_config["dpo"])
    print("  ✓ Model initialized")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        dpo_save_state,
        master_config,
    )


def add_ref_logprobs_to_data(dataloader, policy, master_config, is_val=False):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)

            micro_batch_size = (
                master_config["dpo"]["val_micro_batch_size"] * 2
                if is_val
                else master_config["policy"]["train_micro_batch_size"] * 2
            )

            ## append ref policy logprobs to batch
            logprobs = policy.get_reference_policy_logprobs(
                batch,
                micro_batch_size=micro_batch_size,
            )["reference_logprobs"]
            ## want logprobs for batch to correspond to the log probabilities of the next tokens
            ## so we roll the logprobs to the left by one
            batch["reference_policy_logprobs"] = torch.roll(logprobs, -1, dims=-1)

            yield batch

        except StopIteration:
            break


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
):
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        val_metrics = defaultdict(lambda: 0.0)
        num_valid_batches = 0
        for batch_idx, val_batch in enumerate(
            add_ref_logprobs_to_data(val_dataloader, policy, master_config, is_val=True)
        ):
            ## just run model fwd
            val_results = policy.train(
                val_batch,
                loss_fn,
                eval_mode=True,
                gbs=val_batch_size * 2,
                mbs=val_mbs * 2,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )

            else:
                for k, v in val_results["all_mb_metrics"].items():
                    if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        val_metrics[k] += np.mean(v).item()
                    else:
                        val_metrics[k] += np.sum(v).item()
                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        for k, v in val_metrics.items():
            if k == "num_valid_samples":
                continue
            val_metrics[k] /= num_valid_batches

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if len(val_metrics) == 0:
        warnings.warn(
            "No validation metrics were collected."
            " This is likely because there were no valid samples in the validation set."
        )

    else:
        # Print summary of validation results
        print("\n📊 Validation Results:")
        print(f"    • Validation loss: {float(val_metrics['loss']):.4f}")
        print(f"    • Validation accuracy: {float(val_metrics['accuracy']):.4f}")

        # Print timing information
        print("\n  ⏱️  Validation Timing:")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


def dpo_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    dpo_save_state: DPOSaveState,
) -> None:
    # Run dpo training
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    if dpo_save_state is None:
        dpo_save_state = _default_dpo_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
    else:
        current_epoch = dpo_save_state["epoch"]
        current_step = dpo_save_state["step"]
        total_steps = dpo_save_state["total_steps"]

    dpo_config = master_config["dpo"]
    # Validation configuration
    val_period = dpo_config["val_period"]
    val_at_start = dpo_config["val_at_start"]
    max_num_epochs = dpo_config["max_num_epochs"]

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...")
        validation_result = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            val_batches=dpo_config["val_batches"],
            val_batch_size=dpo_config["val_global_batch_size"],
            val_mbs=dpo_config["val_micro_batch_size"],
        )
        if validation_result is not None:
            val_metrics, validation_timings = validation_result
        else:
            val_metrics, validation_timings = None, None

        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config["dpo"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in add_ref_logprobs_to_data(train_dataloader, policy, master_config):
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['dpo']['max_num_steps'])} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                print("▶ Taking a training step...")
                with timer.time("policy_training"):
                    train_results = policy.train(
                        batch,
                        loss_fn,
                        eval_mode=False,
                        ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                        ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                        gbs=master_config["policy"]["train_global_batch_size"] * 2,
                        mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                    )

                is_last_step = total_steps + 1 >= master_config["dpo"][
                    "max_num_steps"
                ] or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    validation_result = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=dpo_config["val_batches"],
                        val_batch_size=dpo_config["val_global_batch_size"],
                        val_mbs=dpo_config["val_micro_batch_size"],
                    )
                    if validation_result is not None:
                        val_metrics, validation_timings = validation_result
                    else:
                        val_metrics, validation_timings = None, None
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                ## Checkpointing
                dpo_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    dpo_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    dpo_save_state["total_steps"] = total_steps + 1
                    dpo_save_state["epoch"] = current_epoch
                    if val_metrics is not None:
                        dpo_save_state["val_loss"] = val_metrics["loss"]
                    elif "val_loss" in dpo_save_state:
                        del dpo_save_state["val_loss"]

                    if master_config["checkpointing"]["metric_name"] is not None:
                        if (
                            master_config["checkpointing"]["metric_name"]
                            not in dpo_save_state
                        ):
                            warnings.warn(
                                f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                                "Saving most recent k checkpoints instead."
                            )
                            master_config["checkpointing"]["metric_name"] = None

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, dpo_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            losses = train_results["loss"]
            metrics = {
                "loss": train_results["loss"].numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
            }
            metrics.update(train_results["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops
            print("\n⏱️  Timing:")
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")

            # Display all other timing metrics (if any)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if total_steps >= master_config["dpo"]["max_num_steps"]:
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
