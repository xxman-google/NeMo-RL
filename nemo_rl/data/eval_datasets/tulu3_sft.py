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

"""Tulu3 SFT dataset."""

from typing import Any, Literal, Optional

from datasets import load_dataset

from nemo_rl.data import processors
from nemo_rl.data.interfaces import TaskDataSpec


class Tulu3SftDataset:
    def __init__(
        self,
        source: Literal[
            "ai2-adapt-dev/coconot_converted",
            "ai2-adapt-dev/evol_codealpaca_heval_decontaminated",
            "ai2-adapt-dev/flan_v2_converted",
            "ai2-adapt-dev/no_robots_converted",
            "ai2-adapt-dev/numinamath_tir_math_decontaminated",
            "ai2-adapt-dev/oasst1_converted",
            "ai2-adapt-dev/personahub_code_v2_34999",
            "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980",
            "ai2-adapt-dev/tulu_hard_coded_repeated_10",
            "ai2-adapt-dev/tulu_v3.9_aya_100k",
            "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
            "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k",
            "ai2-adapt-dev/tulu_v3.9_sciriff_10k",
            "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k",
            "ai2-adapt-dev/tulu_v3.9_table_gpt_5k",
            "ai2-adapt-dev/tulu_v3.9_wildchat_100k",
            "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k",
            "allenai/tulu-3-sft-personas-math-filtered",
            "allenai/tulu-3-sft-personas-math-grade-filtered",
        ] = "ai2-adapt-dev/numinamath_tir_math_decontaminated",
        prompt_file: Optional[str] = None,
    ):
        """Initialize the Tulu3 SFT subset.

        Args:
            source: Name of the subset to use.
            prompt_file: Path to the prompt file to use.
        """
        ds = load_dataset("allenai/tulu-3-sft-mixture-0225", split="train")
        ds = ds.filter(lambda example: example["source"] == source)
        self.rekeyed_ds = ds.map(self._rekey, remove_columns=["messages"])
        name = source.split("/")[1]
        self.task_spec = TaskDataSpec(
            task_name=f"{name}",
            prompt_file=prompt_file,
            system_prompt_file=None,
        )
        self.processor = processors.data_processor

    def _rekey(self, data: dict[str, Any]):
        questions = []
        answers = []
        for msg in data["messages"]:
            if msg["role"] == "user":
                questions.append(msg["content"])
            elif msg["role"] == "assistant":
                answers.append(msg["content"])
        return {
            "question": questions[0],
            "answer": answers[0],
        }
