# Tool for debugging parquet files
# Example command to run:
# uv run examples/run_read_parquet_files.py --input_parquet_files logs/nemotron/Qwen/Qwen3-8B.parquet

import os
import re

from absl import app, flags
from datasets import load_dataset

_INPUT_PARQUET_FILES = flags.DEFINE_string(
    "input_parquet_files", None, "Path to the input parquet files."
)

def extract_prompt_and_answer(messages):
    """
    Extract the user prompt and assistant answer from a conversation.

    Args:
        messages (list of dict): List of messages with 'content' and 'role' keys.

    Returns:
        tuple: (user_prompt, assistant_answer, extracted_answer)
    """
    user_prompt = None
    assistant_answer = None
    extracted_answer = None

    # Regex: matches "Answer" + optional space/punct + optional space + [A-J]
    answer_pattern = re.compile(r'(?i)Answer[\s:,\-=]*\s*([A-J])')
    
    for msg in messages:
        if msg["role"] == "user" and user_prompt is None:
            user_prompt = msg["content"]
        elif msg["role"] == "assistant" and assistant_answer is None:
            assistant_answer = msg["content"]
            # Find all possible matches
            matches = answer_pattern.findall(assistant_answer)
            if matches:
                # Take the last match, just the letter
                extracted_answer = matches[-1]

        # Stop once both are found
        if user_prompt and assistant_answer and extracted_answer:
            break

    return user_prompt, assistant_answer, extracted_answer

def main(_):
    # Load the dataset
    print("Load dataset")
    ds = load_dataset("parquet", data_files=_INPUT_PARQUET_FILES.value, split="train")
    print(type(_INPUT_PARQUET_FILES.value))
    
    # Print dataset info
    print("\nDataset size:", len(ds))
    
    # Print first few examples
    print("\nFirst few examples:")
    for i, example in enumerate(ds):
        if i >= 3:  # Print first 3 example
            break
        print("-" * 20)
        print(f"\nExample {i + 1}:")
        print(example)
        # messages = example['messages']
        # print(messages)
        # print("*" * 10)
        # user_prompt, assistant_answer, extracted_answer = extract_prompt_and_answer(messages)
        # print("User: ", user_prompt)
        # print("Assistant: ", assistant_answer)
        # print("Extracted Answer: ", extracted_answer)

if __name__ == "__main__":
    flags.mark_flags_as_required(["input_parquet_files"])
    app.run(main)