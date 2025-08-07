r"""Removes `\\boxed{}` from model responses."""

import os
import random
from typing import Any

import regex
from absl import app, flags
from datasets import load_dataset
from tqdm import tqdm

_FRACTION_TO_KEEP = flags.DEFINE_float(
    "fraction_to_keep",
    0.05,
    "Fraction of the data to keep as it is without removing the \\boxed{}.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", None, "Output directory to store the parquet files."
)
_SOURCE = flags.DEFINE_string("source", "math", "Dataset source: math or science.")
_NUM_SHARDS = flags.DEFINE_integer("num_shards", 4, "Number of shards.")

_MCQ_PROMPTS = [
    "Choose the correct answer for the following multiple-choice question.",
    "Solve the following problem by selecting the best option from the choices below.",
    "Answer the multiple-choice question that follows.",
    "For the question below, pick the correct choice from the options given.",
    "Respond to the following question by selecting one of the options.",
    "Read the following multiple-choice question and provide your answer.",
    "For the problem below, indicate the correct answer based on the options listed.",
    "From the choices provided, determine the correct response for the question.",
    "Evaluate the options and select the most accurate answer.",
    "Carefully consider the question and choose the best answer.",
    "Use your reasoning to select the correct multiple-choice answer for the problem below.",
    "Analyze the following question and pick the appropriate option.",
    "Think carefully and choose the correct answer from the list for the following problem.",
    "Answer the question below by identifying the most suitable choice.",
    "Determine the right answer by comparing all available options of the following question.",
]
_COT_PROMPTS = [
    "Solve the following problem step by step.",
    "Break down the following problem and solve it one step at a time.",
    "Approach the following problem using a step-by-step process.",
    "Go through the following problem carefully, explaining each step.",
    "Use logical reasoning to solve the problem below in stages.",
    "Think through each part of the following problem before answering.",
    "Solve the following problem and explain your thought process step by step.",
    "Solve the problem below and reason through the solution one step at a time.",
    "Take it slow and solve the problem below piece by piece.",
    "First, understand the problem below, then solve it step by step.",
]
_MATH_ANSWER_PROMPTS = [
    "Please put your final answer inside \\boxed{}.",
    "Enclose your answer within \\boxed{}.",
    "Your answer should be wrapped in \\boxed{} tags.",
    "Format the answer using \\boxed{}.",
    "Output the result in the format: \\boxed{your_answer}.",
    "Use \\boxed{} to indicate your final answer.",
    "Make sure your answer appears inside \\boxed{}.",
    "Display the answer inside a LaTeX box using \\boxed{}.",
    "Place the answer within \\boxed{} to highlight it.",
    "Return your answer wrapped in \\boxed{}.",
    "Present your answer using LaTeX \\boxed{} notation.",
    "Surround your answer with \\boxed{} brackets.",
    "The answer must appear between \\boxed{}.",
    "Wrap the solution in \\boxed{} for clarity.",
    "Provide your answer enclosed in a LaTeX box: \\boxed{}.",
]
_MCQ_ANSWER_PROMPTS = [
    "Please enclose your final answer in \\boxed{$LETTER}, where LETTER is one of A, B, C, or D.",
    "Format your final answer like this: \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.",
    "Wrap your final choice in LaTeX \\boxed{} syntax using one of A, B, C, or D.",
    "Indicate your answer using \\boxed{LETTER}, choosing from A, B, C, or D.",
    "Your response should be in the form \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.",
    "Please express your answer using LaTeX notation: \\boxed{$LETTER} (A, B, C, or D).",
    "Submit your final answer enclosed in a box, like so: \\boxed{A}.",
    "For your final answer, use the format \\boxed{} with one of the valid options: A--D.",
    "Select a letter from A to D and display it inside \\boxed{}.",
    "Your answer must appear inside \\boxed{} and should be a single uppercase letter from A to D.",
    "Choose one of the letters A, B, C, or D, and place it inside \\boxed{} to indicate your answer.",
    "Once you have your answer, wrap it in \\boxed{} -- for example, \\boxed{C}.",
    "Put your selected option (A--D) inside a LaTeX box like \\boxed{D}.",
    "Answer format: one capital letter (A--D) inside \\boxed{}.",
    "Your final answer should look like this: \\boxed{B}.",
]


def unbox_latex(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    responses = [msg["content"] for msg in messages if msg["role"] == "assistant"]
    prompts = [msg["content"] for msg in messages if msg["role"] == "user"]
    response = responses[0]
    prompt = "\n\n".join([random.choice(_COT_PROMPTS), prompts[0]])
    # Recursive pattern to match \(\boxed{...}\) with nested braces
    pattern = r"(?:\\\()?\s*\\boxed\{((?:[^{}]+|(?R))*)\}\s*(?:\\\))?"
    # Replace all \(\boxed{...}\) with the inner content
    while True:
        matches = regex.search(pattern, response)
        if not matches:
            break
        response = response.replace(matches.group(0), matches.group(1))
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def remove_boxed(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    responses = [msg["content"] for msg in messages if msg["role"] == "assistant"]
    questions = [msg["content"] for msg in messages if msg["role"] == "user"]
    response = responses[0]
    matches = regex.search(r"\n\n\\boxed\{[A-D]\}", response)
    if matches:
        response = response.replace(matches.group(0), "")
        prompt = "\n\n".join([random.choice(_MCQ_PROMPTS), questions[0]])
    else:
        prompt = " ".join(
            [random.choice(_MCQ_PROMPTS), random.choice(_MCQ_ANSWER_PROMPTS)]
        )
        prompt = "\n\n".join([prompt, questions[0]])
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def add_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    responses = [msg["content"] for msg in messages if msg["role"] == "assistant"]
    questions = [msg["content"] for msg in messages if msg["role"] == "user"]
    response = responses[0]
    prompt = " ".join(
        [random.choice(_COT_PROMPTS), random.choice(_MATH_ANSWER_PROMPTS)]
    )
    prompt = "\n\n".join([prompt, questions[0]])
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def map_math_data(data: dict[str, Any]):
    keep = random.random() < _FRACTION_TO_KEEP.value
    messages = add_prompt(data["messages"]) if keep else unbox_latex(data["messages"])
    return {
        "num_tokens": data["num_tokens"],
        "source": data["source"],
        "messages": messages,
    }


def map_science_data(data: dict[str, Any]):
    return {
        "num_tokens": data["num_tokens"],
        "source": data["source"],
        "messages": remove_boxed(data["messages"]),
    }


def main(_):
    ds = load_dataset("open-r1/Mixture-of-Thoughts", name=_SOURCE.value, split="train")
    if _SOURCE.value == "math":
        ds = ds.map(map_math_data)
    else:
        ds = ds.map(map_science_data)

    if not os.path.exists(_OUTPUT_DIR.value):
        os.makedirs(_OUTPUT_DIR.value)

    num_shards = _NUM_SHARDS.value
    for i in tqdm(range(num_shards), desc="Sharding"):
        shard = ds.shard(num_shards=num_shards, index=i)
        output_path = os.path.join(
            _OUTPUT_DIR.value, f"train-{i:05d}-of-{num_shards:05d}.parquet"
        )
        print(f"Writing shard {i} to {output_path}...")
        shard.to_parquet(output_path)


if __name__ == "__main__":
    flags.mark_flag_as_required("output_dir")
    app.run(main)
