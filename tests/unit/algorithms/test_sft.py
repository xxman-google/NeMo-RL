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

from unittest.mock import MagicMock

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.loss_functions import NLLLoss
from nemo_rl.algorithms.sft import _default_sft_save_state, sft_train


@pytest.fixture
def mock_components():
    # Create mock components
    policy = MagicMock()
    policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {},
    }

    # Create a proper message log structure with token_ids
    mock_batch = {
        "message_log": [[{"token_ids": torch.tensor([1, 2, 3]), "role": "assistant"}]],
        "loss_multiplier": torch.tensor(1.0),
    }

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    loss_fn = NLLLoss()
    logger = MagicMock()
    checkpointer = MagicMock()
    sft_task_spec = MagicMock()

    # Create mock master config
    master_config = {
        "sft": {
            "max_num_steps": 5,
            "max_num_epochs": 2,
            "val_period": 100,
            "val_batches": 1,
            "val_global_batch_size": 1,
            "val_micro_batch_size": 1,
            "val_at_start": False,
        },
        "policy": {
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
        },
    }

    return {
        "policy": policy,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "sft_task_spec": sft_task_spec,
        "master_config": master_config,
    }


def test_exit_on_max_steps(mock_components):
    """Test that training loop exits when max_num_steps is reached"""
    # Set max steps to 12, which is less than len(train_dataloader) * max_num_epochs
    mock_components["master_config"]["sft"]["max_num_steps"] = 12

    sft_save_state = _default_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["sft_task_spec"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we only trained for 12 steps.
    assert mock_components["policy"].train.call_count == 12


def test_exit_on_max_epochs(mock_components):
    """Test that training loop exits when max_num_epochs is reached"""
    # Set max epochs to 2 and max steps to a large number
    mock_components["master_config"]["sft"]["max_num_epochs"] = 2
    mock_components["master_config"]["sft"]["max_num_steps"] = 100

    sft_save_state = _default_sft_save_state()

    # Run training
    sft_train(
        mock_components["policy"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["master_config"],
        mock_components["logger"],
        mock_components["sft_task_spec"],
        mock_components["checkpointer"],
        sft_save_state,
    )

    # Verify we trained for exactly two epochs (20 batches).
    assert mock_components["policy"].train.call_count == 20
