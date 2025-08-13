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

from nemo_rl.data.eval_datasets.ai2_arc import ArcDataset
from nemo_rl.data.eval_datasets.aime2024 import AIME2024Dataset
from nemo_rl.data.eval_datasets.aime2025 import AIME2025Dataset
from nemo_rl.data.eval_datasets.arc_agi import ArcAgiDataset
from nemo_rl.data.eval_datasets.beyond_aime import BeyondAIMEDataset
from nemo_rl.data.eval_datasets.deepscaler import DeepScaleRDataset
from nemo_rl.data.eval_datasets.gpqa import GPQADataset
from nemo_rl.data.eval_datasets.gsm8k import Gsm8kDataset
from nemo_rl.data.eval_datasets.humaneval import HumanEvalDataset
from nemo_rl.data.eval_datasets.ifeval import IFEvalDataset
from nemo_rl.data.eval_datasets.livecodebench import LiveCodeBenchDataset
from nemo_rl.data.eval_datasets.local_math_dataset import LocalMathDataset
from nemo_rl.data.eval_datasets.math import MathDataset
from nemo_rl.data.eval_datasets.math_train import MathTrainDataset
from nemo_rl.data.eval_datasets.mbpp import MBPPDataset
from nemo_rl.data.eval_datasets.mgsm import MGSMDataset
from nemo_rl.data.eval_datasets.mmlu import MMLUDataset
from nemo_rl.data.eval_datasets.mmlu_pro import MMLUProDataset
from nemo_rl.data.eval_datasets.numina_math import NuminaMathDataset
from nemo_rl.data.eval_datasets.openr1_math import OpenR1MathDataset
from nemo_rl.data.eval_datasets.simpleqa import SimpleQADataset
from nemo_rl.data.eval_datasets.openr1_verifiable_code import (
    OpenR1VerifiableCodeDataset,
)
from nemo_rl.data.eval_datasets.sciq import SciQDataset
from nemo_rl.data.eval_datasets.tulu3_sft import Tulu3SftDataset


def load_eval_dataset(data_config):
    """Loads evaluation dataset."""
    dataset_name = data_config["dataset_name"]
    if dataset_name.startswith("mmlu") and dataset_name != "mmlu_pro":
        if dataset_name == "mmlu":
            base_dataset = MMLUDataset(
                prompt_file=data_config["prompt_file"],
                system_prompt_file=data_config["system_prompt_file"],
            )
        else:
            language = dataset_name.split("_")[1]
            base_dataset = MMLUDataset(
                language=language,
                prompt_file=data_config["prompt_file"],
                system_prompt_file=data_config["system_prompt_file"],
            )
    elif dataset_name == "ai2_arc":
        base_dataset = ArcDataset(
            subset=data_config["subset"],
            split=data_config["split"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "deepscaler":
        base_dataset = DeepScaleRDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mbpp":
        base_dataset = MBPPDataset(
            code_exe_dir=data_config["code_exe_dir"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mbpp_sanitized":
        base_dataset = MBPPDataset(
            code_exe_dir=data_config["code_exe_dir"],
            variant="sanitized",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mgsm":
        base_dataset = MGSMDataset(
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "humaneval":
        base_dataset = HumanEvalDataset(
            code_exe_dir=data_config["code_exe_dir"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "livecodebench":
        base_dataset = LiveCodeBenchDataset(
            code_exe_dir=data_config["code_exe_dir"],
            version=data_config["version"],
            test_type=data_config["test_type"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "ifeval":
        base_dataset = IFEvalDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "aime2024":
        base_dataset = AIME2024Dataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "aime2025":
        base_dataset = AIME2025Dataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "beyond_aime":
        base_dataset = BeyondAIMEDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gpqa":
        base_dataset = GPQADataset(
            variant="main",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gsm8k_train":
        base_dataset = Gsm8kDataset(
            split="train",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gsm8k_test":
        base_dataset = Gsm8kDataset(
            split="test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "gpqa_diamond":
        base_dataset = GPQADataset(
            variant="diamond",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "mmlu_pro":
        base_dataset = MMLUProDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math":
        base_dataset = MathDataset(
            variant="math_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math_train":
        base_dataset = MathTrainDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "math500":
        base_dataset = MathDataset(
            variant="math_500_test",
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "simpleqa":
        base_dataset = SimpleQADataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "local":
        base_dataset = LocalMathDataset(
            name=dataset_name,
            data_paths=data_config["data_paths"],
            problem_key=data_config["problem_key"],
            solution_key=data_config["solution_key"],
            file_format=data_config["file_format"],
            split=data_config["split"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "tulu3_sft":
        base_dataset = Tulu3SftDataset(
            source=data_config["subset"],
            prompt_file=data_config["prompt_file"],
        )
    elif dataset_name == "numina_math":
        base_dataset = NuminaMathDataset(
            source=data_config["source"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "openr1_math":
        base_dataset = OpenR1MathDataset(
            source=data_config["source"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "openr1_verifiable_code":
        base_dataset = OpenR1VerifiableCodeDataset(
            source=data_config["source"],
            code_exe_dir=data_config["code_exe_dir"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "arc_agi":
        base_dataset = ArcAgiDataset(
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    elif dataset_name == "sciq":
        base_dataset = SciQDataset(
            split=data_config["split"],
            prompt_file=data_config["prompt_file"],
            system_prompt_file=data_config["system_prompt_file"],
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")
    return base_dataset


__all__ = [
    "ArcDataset",
    "AIME2024Dataset",
    "AIME2025Dataset",
    "BeyondAIMEDataset",
    "DeepScaleRDataset",
    "GPQADataset",
    "Gsm8kDataset",
    "HumanEvalDataset",
    "IFEvalDataset",
    "LiveCodeBenchDataset",
    "LocalMathDataset",
    "MathDataset",
    "MathTrainDataset",
    "MBPPDataset",
    "MGSMDataset",
    "MMLUDataset",
    "MMLUProDataset",
    "NuminaMathDataset",
    "OpenR1MathDataset",
    "OpenR1VerifiableCodeDataset",
    "SciQDataset",
    "Tulu3SftDataset",
]
