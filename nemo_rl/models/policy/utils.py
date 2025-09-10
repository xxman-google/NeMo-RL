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

import importlib
import os
from typing import Any

import torch
from transformers import AutoConfig

from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches


def is_vllm_v1_engine_enabled() -> bool:
    """Check if vLLM V1 engine is enabled.

    Returns:
        bool: True if V1 engine is enabled, False otherwise (defaults to True if not set)
    """
    return os.environ.get("NRL_VLLM_USE_V1", "1") == "1"


def import_class_from_path(name: str) -> Any:
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


def get_gpu_info(model: torch.nn.Module) -> dict[str, Any]:
    """Return information about the GPU being used by this worker."""
    import torch

    # Get distributed training info
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Get device info from CUDA
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    device_count = torch.cuda.device_count()
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # in MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # in MB
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # in MB
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)  # in MB

    # Try to get the real global device ID (not the local one)
    # In distributed training, each process only sees its assigned GPU as device 0
    local_device_id = device
    global_device_id = local_device_id

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if local_rank < len(cuda_visible_devices):
            global_device_id = int(cuda_visible_devices[local_rank])

    # Get a parameter from the model to verify CUDA device placement
    # This confirms tensors are actually on the appropriate device
    param_info = {}
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if param is not None and param.requires_grad:
                full_name = f"{module_name}.{param_name}"
                param_info[full_name] = {
                    "device": str(param.device),
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                }
                # Just grab one parameter for verification
                break
        if param_info:
            break

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
        "local_device_id": local_device_id,
        "global_device_id": global_device_id,
        "device_count": device_count,
        "device_name": device_name,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
        "peak_memory_allocated_mb": peak_memory,
        "peak_memory_reserved_mb": peak_reserved,
        "parameter_sample": param_info,
        "env_vars": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CUDA") or k in ["LOCAL_RANK", "RANK", "WORLD_SIZE"]
        },
    }


def sliding_window_overwrite(model_name: str) -> dict[str, Any]:
    """Returns configuration overrides to handle sliding window settings based on model rules.

    Args:
        model_name: The HuggingFace model name or path to load configuration from

    Returns:
        dict: Dictionary with overwrite values, or empty dict if no overwrites needed
    """
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    overwrite_dict = {}

    # Override sliding_window setting to address a HF mismatch relevant to use_sliding_window
    # TODO(@zhiyul): remove this once the bug is fixed https://github.com/huggingface/transformers/issues/38002
    if (
        hasattr(hf_config, "use_sliding_window")
        and hf_config.use_sliding_window == False
    ):
        assert hasattr(hf_config, "sliding_window")
        overwrite_dict = {
            "sliding_window": None,
        }
        print(
            f"use_sliding_window=False in config - overriding sliding_window parameter to None: {overwrite_dict}"
        )

    return overwrite_dict


def configure_expandable_segments() -> None:
    """Configure expandable_segments on Hopper and newer architectures (compute capability 9.x+).

    This helps with memory allocation but causes crashes on Ampere GPUs, so we only enable it
    on newer architectures. If PYTORCH_CUDA_ALLOC_CONF is already set, preserves existing values.
    """
    compute_capability = torch.cuda.get_device_properties(0).major

    if compute_capability >= 9:  # Hopper+
        existing_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")

        # Check if expandable_segments is already configured
        if "expandable_segments" in existing_conf:
            print(f"expandable_segments already configured: {existing_conf}")
            # Already configured, don't override
            return

        # Add expandable_segments to existing configuration
        if existing_conf:
            # Append to existing configuration
            new_conf = f"{existing_conf},expandable_segments:True"
        else:
            # Set new configuration
            new_conf = "expandable_segments:True"

        print(f"Setting PYTORCH_CUDA_ALLOC_CONF to {new_conf}")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf

    else:
        ## make sure that expandable_segments is not set to True
        if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            conf_items = os.environ["PYTORCH_CUDA_ALLOC_CONF"].split(",")
            for item in conf_items:
                if item.strip().startswith("expandable_segments"):
                    key_value = item.split(":")
                    if len(key_value) == 2 and key_value[1].strip().lower() == "true":
                        raise RuntimeError(
                            "expandable_segments is enabled in PYTORCH_CUDA_ALLOC_CONF, "
                            "but this is not supported on architectures older than Hopper (compute capability < 9). "
                            "Please set expandable_segments to False."
                        )


def configure_dynamo_cache() -> None:
    """Disable dynamo autotune_local_cache.

    Dynamo may fail at cached_autotune when there's already a cache with different order of node_bundles.
    Disable autotune_local_cache as a workaround.
    See https://github.com/pytorch/pytorch/issues/153791 for more details.
    """
    torch._inductor.config.autotune_local_cache = False


def get_runtime_env_for_policy_worker(policy_worker_name: str) -> dict[str, Any]:
    """Get runtime environment configuration for policy workers.

    Note: expandable_segments configuration is handled directly in the worker init methods
    to ensure proper GPU detection after CUDA initialization.
    """
    runtime_env = {
        **get_nsight_config_if_pattern_matches(policy_worker_name),
    }

    return runtime_env


def get_megatron_checkpoint_dir() -> str:
    """Gets the default megatron checkpoint directory for initial HF -> Mcore conversion.

    Megatron initial checkpoint should be saved to a path available on all nodes. The directory used will take this order of precendence:
    1. $NRL_MEGATRON_CHECKPOINT_DIR (if set)
    2. $HF_HOME/nemo_rl (if HF_HOME is set)
    3. ~/.cache/huggingface/nemo_rl

    HF_HOME is preferred since many users will also have that path mounted and it means one less directory
    to mount into your runtime environment.
    """
    nrl_checkpoint_dir = os.environ.get("NRL_MEGATRON_CHECKPOINT_DIR")
    if nrl_checkpoint_dir is not None and nrl_checkpoint_dir.strip():
        checkpoint_dir = nrl_checkpoint_dir
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home is not None and hf_home.strip():
            checkpoint_dir = os.path.join(hf_home, "nemo_rl")
        else:
            checkpoint_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "nemo_rl"
            )
    print(f"Using default megatron checkpoint dir: {checkpoint_dir}")
    return checkpoint_dir


def get_handle_from_tensor(tensor: torch.Tensor) -> tuple[Any]:
    """Get IPC handle from a tensor."""
    from torch.multiprocessing.reductions import reduce_tensor

    # skip serializing the function for better refit performance
    return reduce_tensor(tensor.detach())[1:]
