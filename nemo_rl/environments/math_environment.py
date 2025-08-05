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
import ast
import contextlib
import io
import json
import logging
import os
import re
from typing import Any, Optional, TypedDict

# import evaluate as hf_evaluate
import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from swebench.harness.run_evaluation import run_instances
from swebench.harness.reporting import make_run_report
from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)

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

# from nemo_rl.evals.ifeval import instructions_registry

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


class MathEnvironmentMetadata(TypedDict):
    ground_truth: Optional[str]
    tests: Optional[str]
    checker_info: Optional[dict[str, Any]]


@ray.remote  # pragma: no cover
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
        self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and the extracted answer for each predicted response.
        """
        results = []
        for response, metadata in zip(pred_responses, metadata_list):
            ground_truth = metadata["ground_truth"]
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


@ray.remote  # pragma: no cover
class MGSMVerifyWorker:
    def _score_mgsm(self, target: str, prediction: str) -> bool:
        if "." in prediction:
            prediction = prediction.rstrip("0").rstrip(".")

        target = target.replace(",", "")
        prediction = prediction.replace(",", "")

        return target == prediction

    def verify(
        self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and the extracted answer for each predicted response.
        """
        results = []
        for response, metadata in zip(pred_responses, metadata_list):
            lang = metadata["lang"]
            correct_answer = metadata["ground_truth"]
            answer_prefix = answer_parsing.LANG_TO_ANSWER_PREFIX[lang]
            extracted_answer = answer_parsing.mgsm_parse_answer(response, answer_prefix)
            score = self._score_mgsm(correct_answer, extracted_answer)
            results.append((score, correct_answer, extracted_answer))
        return results


@ray.remote  # pragma: no cover
class MultilingualMultichoiceVerifyWorker:
    def verify(
        self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answers for each predicted response.
        """
        results = []
        for response, metadata in zip(pred_responses, metadata_list):
            ground_truth = answer_parsing.normalize_response(metadata["ground_truth"])
            response = answer_parsing.normalize_response(response)
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


@ray.remote  # pragma: no cover
class EnglishMultichoiceVerifyWorker:
    def verify(
        self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answers for each predicted response.
        """
        results = []
        for response, metadata in zip(pred_responses, metadata_list):
            ground_truth = answer_parsing.normalize_response(metadata["ground_truth"])
            response = answer_parsing.normalize_response(response)
            extracted_answer = None
            match = re.search(
                "(?i)(?:Answer\s*:|answer is)[ \t]*([A-Z])[.]*\s*$", response
            )
            # match = re.search("(?i)Answer\s*:[ \t]*([A-Z])\s*$", response)
            if match:
                extracted_answer = answer_parsing.normalize_extracted_answer(
                    match.group(1)
                )
            score = 1.0 if extracted_answer == ground_truth else 0.0
            results.append((score, ground_truth, extracted_answer))
        return results


# @ray.remote
# class CodeVerifyWorker:
#     def __init__(self) -> None:
#         self._pass_at_k = hf_evaluate.load("code_eval")

#     def _find_code(self, response: str) -> str:
#         pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
#         matches = pattern.findall(response)
#         extracted_answer = matches[0] if len(matches) >= 1 else response
#         return extracted_answer

#     def verify(
#         self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
#     ) -> list[tuple[float, str, str]]:
#         """Verify the correctness of the predicted responses against the ground truth.

#         Args:
#             pred_responses: list[str]. The predicted responses from the LLM.
#             tests_list: list[str]. The unit tests.

#         Returns:
#             list[tuple[float, str, str]]. The rewards, unit tests, and extracted code segment for each predicted response.
#         """
#         outputs = []
#         for response, metadata in zip(pred_responses, metadata_list):
#             tests = metadata["tests"]
#             code = self._find_code(response)
#             predictions = [[code]]
#             results = self._pass_at_k.compute(
#                 references=[tests], predictions=predictions, k=[1]
#             )
#             score = float(results[0]["pass@1"] == 1.0)
#             outputs.append((score, tests, code))
#         return outputs


# @ray.remote
# class IFVerifyWorker:
#     """Response verifier worker for instruction following problems."""

#     def _remove_kwargs_none(self, kwargs) -> dict[str, Any]:
#         return {k: v for k, v in kwargs.items() if v is not None}

#     def _is_following(
#         self, response: str, checker_info: dict[str, Any]
#     ) -> tuple[bool, list[str], list[str]]:
#         instruction_list = checker_info["instruction_id_list"]
#         checker_kwargs = checker_info["checker_kwargs"]
#         prompt = checker_info["prompt"]
#         is_following_list = []
#         descriptions = []
#         results = []
#         for index, instruction_id in enumerate(instruction_list):
#             instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
#             instruction = instruction_cls(instruction_id)

#             description = instruction.build_description(
#                 **self._remove_kwargs_none(checker_kwargs[index])
#             )
#             descriptions.append(description)
#             args = instruction.get_instruction_args()
#             if args and "prompt" in args:
#                 instruction.build_description(prompt=prompt)

#             if response.strip() and instruction.check_following(response):
#                 is_following_list.append(True)
#                 results.append(f"{instruction_id}: True")
#             else:
#                 is_following_list.append(False)
#                 results.append(f"{instruction_id}: False")
#         return all(is_following_list), descriptions, results

#     def verify(
#         self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
#     ) -> list[tuple[float, str, str]]:
#         """Verify the correctness of the predicted responses against the ground truth.

#         Args:
#             pred_responses: list[str]. The predicted responses from the LLM.
#             checker_info_list: list[dict[str, Any]]. The instruction lists and constraints.

#         Returns:
#             list[tuple[float, str, str]]. The rewards, instruction descriptions, and predicted responses.
#         """
#         outputs = []
#         for response, metadata in zip(pred_responses, metadata_list):
#             checker_info = metadata["checker_info"]
#             score, descriptions, results = self._is_following(response, checker_info)
#             description = "\n".join(descriptions)
#             results = "\n".join(results)
#             outputs.append((float(score), description, results))
#         return outputs


@ray.remote
class ArcAgiVerifyWorker:
    """Response verifier worker for ARC-AGI problems."""

    def _extract_response_grid(self, s: str) -> Optional[list[list[int]]]:
        # Regex for a 2D grid of integers (optionally with whitespace)
        pattern = r"<output>\s*(\[[^\]]*(?:\][^\[]*\[?[^\]]*)*)\s*</output>"
        match = re.search(pattern, s, re.DOTALL)
        if not match:
            return None
        grid_str = match.group(1)
        try:
            return ast.literal_eval(grid_str)
        except (SyntaxError, ValueError):
            return None

    def verify(
        self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            metadata_list: list[MathEnvironmentMetadata]. The metadata containing ground truth and other info.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answer for each predicted response.
        """
        results = []
        for response, metadata in zip(pred_responses, metadata_list):
            extracted_answer = self._extract_response_grid(response)
            score = 1.0 if extracted_answer == metadata["ground_truth"] else 0.0
            results.append((score, metadata["ground_truth"], extracted_answer))
        return results


@ray.remote
class SweBenchVerifyWorker:
    """Response verifier worker for SweBench problems."""

    def verify(
        self, pred_responses: list[str], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Run swebench evaluation on the model-generated patches.

        Args:
            pred_responses: list[str]. The predicted responses from the LLM.
            metadata_list: list[MathEnvironmentMetadata]. The metadata containing ground truth and other info.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answer for each predicted response.
        """
        predictions = {}
        instances = []
        model_name = "model_name"  # TODO: Placeholder for the model name, should be set appropriately.
        for response, metadata in zip(pred_responses, metadata_list):
            instance = metadata["instance"]
            prediction = {
                KEY_INSTANCE_ID: instance[KEY_INSTANCE_ID],
                KEY_MODEL: model_name,
                KEY_PREDICTION: response,
                "golden_patch": instance["patch"],
            }
            predictions[instance[KEY_INSTANCE_ID]] = prediction
            instances.append(instance)

        run_id = "swebench_verified_oracle_eval"
        run_instances(
            predictions=predictions,
            instances=instances,
            cache_level="env",
            clean=False,
            force_rebuild=False,
            max_workers=4,
            run_id=run_id,
            timeout=600,
        )

        results = []
        # Read results from instance results files.
        eval_dir = f"logs/run_evaluation/{run_id}/{model_name}"
        if not os.path.exists(eval_dir):
            raise FileNotFoundError(f"Evaluation directory {eval_dir} does not exist.")
        verified_issues = []
        for instance_id, prediction in predictions.items():
            instance_result_file = os.path.join(
                eval_dir, instance_id, "report.json"
            )
            if not os.path.exists(instance_result_file):
                continue
            with open(instance_result_file, "r") as f:
                instance_report = f.read()
            score = self._get_score_from_report(instance_id, instance_report)
            golden_patch = prediction["golden_patch"]
            model_patch = prediction[KEY_PREDICTION]
            results.append((score, golden_patch, model_patch))
            if score == 1.0:
                verified_issues.append(instance_id)
        with open(f"{eval_dir}/verified_issues.txt", "w") as f:
            f.write(f"{instance_id}: {model_patch}\n\n")
        return results

    def _get_score_from_report(self, instance_id: str, instance_report: str) -> float:
        """Parses the report and returns whether the patch resolved the issue."""
        instance_report = json.loads(instance_report)
        return float(instance_report[instance_id]["resolved"])


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class MathEnvironment(EnvironmentInterface):
    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        verifier_type = cfg.get("verifier_type", "math")
        worker_cls = {
            # "code": CodeVerifyWorker,
            "english_multichoice": EnglishMultichoiceVerifyWorker,
            # "instruction_following": IFVerifyWorker,
            "math": MathVerifyWorker,
            "mgsm": MGSMVerifyWorker,
            "multilingual_multichoice": MultilingualMultichoiceVerifyWorker,
            "arc_agi": ArcAgiVerifyWorker,
            "swebench_verified": SweBenchVerifyWorker,
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

        chunked_assistant_response_batch = chunk_list_to_workers(
            assistant_response_batch, self.num_workers
        )
        chunked_verifier_metadata = chunk_list_to_workers(metadata, self.num_workers)

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
