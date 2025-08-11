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
import logging
import os
import re
from typing import Any, Optional, TypedDict

import ray
import torch
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

from nemo_rl.data.interfaces import LLMMessageLogType
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
from nemo_rl.evals.grader_model import (
    GptGraderModel,
    GeminiGraderModel,
    ALPACA2_GRADER_TEMPLATE,
    ALPACA2_SYSTEM_MESSAGE,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    QA_GRADER_TEMPLATE
)

# from nemo_rl.evals.ifeval import instructions_registry


class MathEnvConfig(TypedDict):
    num_workers: int
    stop_strings: Optional[list[str]]  # Default stop strings for this env
    worker_type: Optional[str]
    grader_model_name: Optional[str]  # Model to use for grading, e.g., "gpt-4o"
    grader_api_key: Optional[str]  # API key
    grader_system_message: Optional[str]  # System message for the grader model
    grader_temperature: Optional[float]  # Temperature for the grader model
    grader_max_tokens: Optional[int]  # Max tokens for the grader model


@contextlib.contextmanager
def _mute_output():
    devnull_out, devnull_err = io.StringIO(), io.StringIO()
    with (
        contextlib.redirect_stdout(devnull_out),
        contextlib.redirect_stderr(devnull_err),
    ):
        yield


class MathEnvironmentMetadata(TypedDict):
    ground_truth: str


class MultilingualMathEnvironmentMetadata(TypedDict):
    ground_truth: str
    lang: str


@ray.remote  # pragma: no cover
class MathVerifyWorker:
    def __init__(self, cfg: MathEnvConfig) -> None:
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
        self, pred_data: list[dict[str, str]], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and the extracted answer for each predicted response.
        """
        results = []
        for data, metadata in zip(pred_data, metadata_list):
            response = data["response"]
            ground_truth = str(metadata["ground_truth"])
            extracted_answer = None
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

                results.append((float(ret_score), ground_truth, extracted_answer))
            except Exception as e:
                print(e)
                results.append((0.0, ground_truth, extracted_answer))
        return results


@ray.remote  # pragma: no cover
class QAVerifyWorker:
    def __init__(self, cfg: MathEnvConfig) -> None:
        model=cfg.get("grader_model", "gemini-2.5-flash")
        self.grader_model = None
        logger = logging.getLogger("qa_verify_worker")
        logger.setLevel(logging.INFO)
        logger.info(f"Initialized Grader Mmodel: {model})")
        if model.startswith("gpt"):
            self.grader_model = GptGraderModel(
                model=model,
                api_key=cfg.get("grader_api_key", os.getenv("OPENAI_API_KEY")),
                system_message=cfg.get("grader_system_message",OPENAI_SYSTEM_MESSAGE_CHATGPT),
                temperature=cfg.get("grader_temperature", 0.5),
                max_tokens=cfg.get("grader_max_tokens", 1024),
            )
        else:
           self.grader_model = GeminiGraderModel(
               model=model,
               api_key=cfg.get("grader_api_key", os.getenv("GEMINI_API_KEY")),
               system_message=cfg.get("grader_system_message", OPENAI_SYSTEM_MESSAGE_CHATGPT),
               temperature=cfg.get("grader_temperature", 0.5),
               max_tokens=cfg.get("grader_max_tokens", 1024),
           )
        
    
    def _grade_sample(self, question: str, ground_truth: str, predicted_answer: str) -> str:
        grader_prompt = QA_GRADER_TEMPLATE.format(
            question=question,
            target=ground_truth,
            predicted_answer=predicted_answer,
        )
        prompt_messages = [
            self.grader_model.pack_message(content=grader_prompt, role="user")
        ]
        grader_response = self.grader_model(prompt_messages)
        grading_response = grader_response.response_text
        # Extract the grading letter (A, B, C) from the response
        match = re.search(r"(A|B|C)", grading_response)
        return match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match
        
    def verify(
        self, pred_data: list[dict[str, str]], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and the extracted answer for each predicted response.
        """
        results = []
        for data, metadata in zip(pred_data, metadata_list):
            question = data["prompt"]
            model_response = data["response"]
            ground_truth = str(metadata["ground_truth"])
            grade_letter = self._grade_sample(question, ground_truth, model_response)
            is_correct = grade_letter == "A"
            is_incorrect = grade_letter == "B"
            is_not_attempted = grade_letter == "C"
            score = is_correct
            results.append((score, ground_truth, data["response"]))
        return results
        

@ray.remote  # pragma: no cover
class MGSMVerifyWorker:
    def __init__(self, cfg: MathEnvConfig) -> None:
        pass

    def _score_mgsm(self, target: str, prediction: str) -> bool:
        if "." in prediction:
            prediction = prediction.rstrip("0").rstrip(".")

        target = target.replace(",", "")
        prediction = prediction.replace(",", "")

        return target == prediction

    def verify(
        self,
        pred_data: list[dict[str, str]],
        metadata_list: list[MultilingualMathEnvironmentMetadata],
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and the extracted answer for each predicted response.
        """
        results = []
        for data, metadata in zip(pred_data, metadata_list):
            response = data["response"]
            lang = metadata["lang"]
            correct_answer = metadata["ground_truth"]
            answer_prefix = answer_parsing.LANG_TO_ANSWER_PREFIX[lang]
            extracted_answer = answer_parsing.mgsm_parse_answer(response, answer_prefix)
            score = self._score_mgsm(correct_answer, extracted_answer)
            results.append((score, correct_answer, extracted_answer))
        return results


@ray.remote  # pragma: no cover
class MultilingualMultichoiceVerifyWorker:
    def __init__(self, cfg: MathEnvConfig) -> None:
        pass

    def verify(
        self, pred_data: list[dict[str, str]], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answers for each predicted response.
        """
        results = []
        for data, metadata in zip(pred_data, metadata_list):
            response = data["response"]
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
    def __init__(self, cfg: MathEnvConfig) -> None:
        pass

    def verify(
        self, pred_data: list[dict[str, str]], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
            ground_truths: list[str]. The ground truth responses.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answers for each predicted response.
        """
        results = []
        for data, metadata in zip(pred_data, metadata_list):
            response = data["response"]
            ground_truth = answer_parsing.normalize_response(metadata["ground_truth"])
            response = answer_parsing.normalize_response(response)
            extracted_answer = None
            match = re.search(
                r"(?i)(?:Answer\s*:|answer is)[ \t]*([A-Z])[.]*\s*$", response
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
#         self, pred_data: list[dict[str, str]], metadata_list: list[MathEnvironmentMetadata]
#     ) -> list[tuple[float, str, str]]:
#         """Verify the correctness of the predicted responses against the ground truth.

#         Args:
#             pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
#             checker_info_list: list[dict[str, Any]]. The instruction lists and constraints.

#         Returns:
#             list[tuple[float, str, str]]. The rewards, instruction descriptions, and predicted responses.
#         """
#         outputs = []
#         for data, metadata in zip(pred_data, metadata_list):
#             response = data["response"]
#             checker_info = metadata["checker_info"]
#             score, descriptions, results = self._is_following(response, checker_info)
#             description = "\n".join(descriptions)
#             results = "\n".join(results)
#             outputs.append((float(score), description, results))
#         return outputs


@ray.remote
class ArcAgiVerifyWorker:
    """Response verifier worker for ARC-AGI problems."""

    def __init__(self, cfg: MathEnvConfig) -> None:
        pass

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
        self, pred_data: list[dict[str, str]], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
            metadata_list: list[MathEnvironmentMetadata]. The metadata containing ground truth and other info.

        Returns:
            list[tuple[float, str, str]]. The rewards, correct answer, and extracted answer for each predicted response.
        """
        results = []
        for data, metadata in zip(pred_data, metadata_list):
            response = data["response"]
            extracted_answer = self._extract_response_grid(response)
            score = 1.0 if extracted_answer == metadata["ground_truth"] else 0.0
            results.append((score, metadata["ground_truth"], extracted_answer))
        return results


@ray.remote  # pragma: no cover
class Alpaca2VerifyWorker:
    def __init__(self, cfg: MathEnvConfig) -> None:
        model = cfg.get("grader_model_name", "gpt-4-1106-preview")
        self.grader_model = None
        logger = logging.getLogger("alpaca2_verify_worker")
        logger.setLevel(logging.INFO)
        logger.info(f"Initialized Grader Model: {model})")
        assert model.startswith("gpt"), (
            "Alpaca Eval 2.0 requires 'gpt-4-1106-preview' to be the grader model."
        )
        self.grader_model = GptGraderModel(
            model=model,
            api_key=cfg.get("grader_api_key", os.getenv("OPENAI_API_KEY")),
            system_message=cfg.get("grader_system_message",ALPACA2_SYSTEM_MESSAGE),
            temperature=cfg.get("grader_temperature", 1.0),
            max_tokens=cfg.get("grader_max_tokens", 1),
            logprobs=cfg.get("grader_logprobs", 1),
            top_logprobs=cfg.get("grader_top_logprobs", 5),
        )

    def _grade_sample(self, prompt: str, golden_response: str, sample_response: str) -> str:
        grader_prompt = ALPACA2_GRADER_TEMPLATE.format(
            instruction=prompt,
            output_1=golden_response,
            output_2=sample_response,
        )
        prompt_messages = [
            self.grader_model.pack_message(content=grader_prompt, role="user")
        ]
        grader_response = self.grader_model(prompt_messages)
        return grader_response.response_text
        
    def verify(
        self, pred_data: list[dict[str, str]], metadata_list: list[MathEnvironmentMetadata]
    ) -> list[tuple[float, str, str]]:
        """Verify the correctness of the predicted responses against the ground truth.

        Args:
            pred_data: list[dict[str, str]]. The predicted data including prompt and response from the LLM.
            metadata_list: list[MathEnvironmentMetadata]. The metadata containing golden responses and other info.

        Returns:
            list[tuple[float, str, str]]. The rewards, golden response, and model sample response for each instance.
        """
        results = []
        for data, metadata in zip(pred_data, metadata_list):
            prompt = data["prompt"]
            sample_response = data["response"]
            golden_response = str(metadata["golden_response"])
            verdict = self._grade_sample(prompt, golden_response, sample_response)
            assert verdict in ["m", "M"], (
                f"Judge model expected to respond with 'm' or 'M', but responded with {verdict}"
            )
            score = verdict == "M"
            results.append((score, golden_response, sample_response))
        return results


@ray.remote(max_restarts=-1, max_task_retries=-1)  # pragma: no cover
class MathEnvironment(EnvironmentInterface[MathEnvironmentMetadata]):
    def __init__(self, cfg: MathEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]
        # TODO: split out this environment since it's doing more than just math
        worker_type = cfg.get("worker_type", "math")
        assert isinstance(worker_type, str), (
            f"{worker_type=} must be a string but was {type(worker_type)}"
        )
        worker_cls = {
            "arc_agi": ArcAgiVerifyWorker,
            "english_multichoice": EnglishMultichoiceVerifyWorker,
            # "instruction_following": IFVerifyWorker,
            "math": MathVerifyWorker,
            "mgsm": MGSMVerifyWorker,
            "multilingual_multichoice": MultilingualMultichoiceVerifyWorker,
            "simpleqa": QAVerifyWorker,
            "alpaca2": Alpaca2VerifyWorker,
        }[worker_type]
        self.workers = [
            worker_cls.options(  # type: ignore # (decorated with @ray.remote)
                runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
            ).remote(cfg=self.cfg)
            for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        # shutdown all workers
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[MathEnvironmentMetadata],
    ) -> EnvironmentReturn[MathEnvironmentMetadata]:
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
        user_prompt_batch = []
        assistant_response_batch = []
        for conversation in message_log_batch:
            user_prompts = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "user"
            ]
            user_prompt_batch.append("".join(user_prompts))
            assistant_responses = [
                str(interaction["content"])
                for interaction in conversation
                if interaction["role"] == "assistant"
            ]
            assistant_response_batch.append("".join(assistant_responses))

        chunk = [{"prompt": p, "response": r} for p, r in zip(user_prompt_batch, assistant_response_batch)]
        chunked_batch = chunk_list_to_workers(chunk, self.num_workers)
        chunked_verifier_metadata = chunk_list_to_workers(metadata, self.num_workers)

        # # Process each chunk in parallel
        futures = [
            self.workers[i].verify.remote(chunk, metadata_chunk)
            for i, (chunk, metadata_chunk) in enumerate(
                zip(chunked_batch, chunked_verifier_metadata)
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
