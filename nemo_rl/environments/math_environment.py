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
import contextlib
import io
import logging
import os
import re
from typing import Any, Optional, TypedDict

import evaluate as hf_evaluate
import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.environments.metrics import (
    calculate_pass_rate_per_prompt,
)
from nemo_rl.environments.utils import chunk_list_to_workers
from nemo_rl.evals import answer_parsing
from nemo_rl.evals.ifeval import instructions_registry

# This is needed for running code evaluation
os.environ["HF_ALLOW_CODE_EVAL"] = "1"


class MathEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    verifier_type: Optional[str]


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


@ray.remote
class MathVerifyWorker:
    def __init__(self) -> None:
        logging.getLogger("math_verify").setLevel(logging.CRITICAL)

        # Use Latex and plain math extraction from predictions
        # https://github.com/huggingface/Math-Verify?tab=readme-ov-file#extraction-targets
        self.verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(),
            ),
        )

    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and the extracted answer for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            try:
                ground_truth_parsable = "\\boxed{" + ground_truth + "}"
                with _mute_output():
                    try:
                        ret_score, (_, extracted_answer) = self.verify_func(
                            [ground_truth_parsable], [response]
                        )
                        extracted_answer = extracted_answer[-1]
                    # It's possible to emit a TimeoutException and that wouldn't be caught since
                    # it actually subclasses from BaseException and math-verify itself does not
                    # to catch it.
                    except (Exception, TimeoutException):
                        ret_score = 0.0
                        extracted_answer = None

                results.append((float(ret_score), ground_truth, extracted_answer))
            except Exception:
                results.append((0.0, ground_truth, extracted_answer))
        return results

@ray.remote
class MultichoiceVerifyWorker:
    def verify(
        self, pred_responses: list[str], ground_truths: list[str]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answers for each predicted response.
        """
        results = []
        for response, ground_truth in zip(pred_responses, ground_truths):
            response = normalize_response(response)
            extracted_answer = None
            for answer_regex in answer_parsing.MULTILINGUAL_ANSWER_REGEXES:
                regex = answer_parsing.MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(
                    answer_regex
                )
                match = re.search(regex, response)
                if match:
                    extracted_answer = answer_parsing.normalize_extracted_answer(
                        match.group(1)
                    )
                    break
            score = 1.0 if extracted_answer == ground_truth else 0.0
            results.append((score, ground_truth, extracted_answer))
        return results


@ray.remote
class CodeVerifyWorker:
    def __init__(self) -> None:
        self._pass_at_k = hf_evaluate.load("code_eval")

    def _find_code(self, response: str) -> str:
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(response)
        extracted_answer = matches[0] if len(matches) >= 1 else response
        return extracted_answer

    def verify(
        self, pred_responses: list[str], tests_list: list[str]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            tests_list: list[str]. The unit tests.

        Returns:
            list[tuple[float, str, str]]. The rewards, unit tests, and extracted code segment for each predicted response.
        """
        outputs = []
        for response, tests in zip(pred_responses, tests_list):
            code = self._find_code(response)
            predictions = [[code]]
            results = self._pass_at_k.compute(
                references=[tests], predictions=predictions, k=[1]
            )
            score = float(results[0]["pass@1"] == 1.0)
            outputs.append((score, tests, code))
        return outputs


@ray.remote
class IFVerifyWorker:
    """Response verifier worker for instruction following problems."""

    def _remove_kwargs_none(self, kwargs) -> dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v is not None}

    def _is_following(
        self, response: str, checker_info: dict[str, Any]
    ) -> tuple[bool, list[str], list[str]]:
        instruction_list = checker_info["instruction_id_list"]
        checker_kwargs = checker_info["checker_kwargs"]
        prompt = checker_info["prompt"]
        is_following_list = []
        descriptions = []
        results = []
        for index, instruction_id in enumerate(instruction_list):
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)

            description = instruction.build_description(
                **self._remove_kwargs_none(checker_kwargs[index])
            )
            descriptions.append(description)
            args = instruction.get_instruction_args()
            if args and "prompt" in args:
                instruction.build_description(prompt=prompt)

            if response.strip() and instruction.check_following(response):
                is_following_list.append(True)
                results.append(f"{instruction_id}: True")
            else:
                is_following_list.append(False)
                results.append(f"{instruction_id}: False")
        return all(is_following_list), descriptions, results

    def verify(
        self, pred_responses: list[str], checker_info_list: list[dict[str, Any]]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            checker_info_list: list[dict[str, Any]]. The instruction lists and constraints.

        Returns:
            list[tuple[float, str, str]]. The rewards, instruction descriptions, and predicted responses.
        """
        outputs = []
        for response, checker_info in zip(pred_responses, checker_info_list):
            score, descriptions, results = self._is_following(response, checker_info)
            description = "\n".join(descriptions)
            results = "\n".join(results)
            outputs.append((float(score), description, results))
        return outputs


class MathEnvironmentMetadata(TypedDict):
    ground_truth: Optional[str]
    tests: Optional[str]
    checker_info: Optional[dict[str, Any]]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class MathEnvironment(EnvironmentInterface):
    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        verifier_type = cfg.get("verifier_type", "math")
        worker_cls = {
            "math": MathVerifyWorker,
            "code": CodeVerifyWorker,
            "multichoice": MultichoiceVerifyWorker,
            "instruction_following": IFVerifyWorker,
        }[verifier_type]
        self.workers = [
            worker_cls.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote()
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(  # type: ignore[override]
        self,
        message_log_batch: list[list[dict[str, str]]],
        metadata: list[MathEnvironmentMetadata],
    ) -> EnvironmentReturn:
        """Runs a step in the math environment.

        Args:
            message_log: list[list[dict[str, str]]]. A batch of OpenAI-API-like message logs that represent interactions with the LLM.
            metadata: list[MathEnvironmentMetadata]. The grader will use the 'ground_truth' key to evaluate correctness.

        Returns:
            EnvironmentReturn: A tuple containing:
                - list[dict[str, str]]: Observations/responses batch
                - list[dict]: Updated metadata
                - list[str]: Next stop strings for the next turn
                - Tensor: Rewards tensor
                - Tensor: Done flags tensor
        """
        # Extract the assistant's responses from the message history
        # Each message list should have at least one assistant response
        assistant_response_batch = []
        for conversation in message_log_batch:
            assistant_responses = [
                interaction["content"]
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        verifier_metadata_key = self.cfg.get("verifier_metadata_key", "ground_truth")
        verifier_metadata = [g[verifier_metadata_key] for g in metadata]

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_verifier_metadata = chunk_list_to_workers(
            verifier_metadata, self.num_workers
        )

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, metadata_chunk)
            for i, (chunk, metadata_chunk) in enumerate(
                zip(chunked_assistant_response_batch, chunked_verifier_metadata)
            )
        ]

        results = ray.get(futures)

        # flatten the results
        results = [
            (score, correct_answer, extracted_answer)
            for sublist in results
            for (score, correct_answer, extracted_answer) in sublist
        ]
        observations = [
            {
                "role": "environment",
                "content": "Environment: correct"
                if score
                else "Environment: incorrect",
                "extracted_answer": extracted_answer,
                "correct_answer": correct_answer,
            }
            for score, correct_answer, extracted_answer in results
        ]

        # create a tensor of rewards and done flags
        rewards = torch.tensor([score for score, _, _ in results]).cpu()
        done = torch.ones_like(rewards).cpu()

        next_stop_strings = [None] * len(message_log_batch)

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards,
            terminateds=done,
        )

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict[Any]
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Computes metrics for this environment given a global rollout batch.

        Every rank will run this function, so you're free to use distributed
        calculations if you'd prefer for heavy metrics.
        """
        batch["rewards"] = (
            batch["rewards"] * batch["is_end"]
        )  # set a reward of 0 for any incorrectly ended sequences
        if (batch["rewards"] == 1).float().sum() > 0:
            correct_solution_generation_lengths = (
                (batch["generation_lengths"] - batch["prompt_lengths"])[
                    batch["rewards"] == 1
                ]
                .float()
                .mean()
                .item()
            )
        else:
            correct_solution_generation_lengths = 0

        metrics = {
            # "table": table, TODO @sahilj WIP
            "accuracy": batch["rewards"].mean().item(),
            "pass@samples_per_prompt": calculate_pass_rate_per_prompt(
                batch["text"], batch["rewards"]
            ),
            "fraction_of_samples_properly_ended": batch["is_end"].float().mean().item(),
            "num_problems_in_batch": batch["is_end"].shape[0],
            "generation_lengths": batch["generation_lengths"].float().mean().item(),
            "prompt_lengths": batch["prompt_lengths"].float().mean().item(),
            "correct_solution_generation_lengths": correct_solution_generation_lengths,
        }

        return batch, metrics
