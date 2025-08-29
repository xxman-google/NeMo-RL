"""
This script processes parquet files containing conversation data and identifies consistent responses across multiple models.

The script:
1. Reads parquet files containing conversations with 'messages' column
2. Extracts user prompts and assistant answers from each conversation
3. Identifies prompts where all models gave the same answer
4. Saves the consistent conversations to a new parquet file

Usage:
    uv run examples/run_consensus_check.py --input_parquet_files logs/nemotron/Qwen/fullset --output_file logs/nemotron/Qwen/fullset/8B_14B_consensus_examples.parquet
"""


import os
from absl import app, flags
from datasets import load_dataset
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
FLAGS = flags.FLAGS
flags.DEFINE_string('input_parquet_files', None, 'Path to directory containing parquet files')
flags.DEFINE_string('output_file', None, 'Output parquet file path')

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

def process_parquet_files(input_path):
    # Get all parquet files in the directory
    parquet_files = [str(p) for p in Path(input_path).glob('*.parquet')]
    len_files = len(parquet_files)

    if not parquet_files:
        print(f"No parquet files found in {input_path}")
        return
    print(f"Found {len_files} parquet files: {parquet_files}")
    
    # Dictionary to store user content and corresponding assistant responses
    content_map = defaultdict(set)
    
    # Load all parquet files as a dataset
    print("Loading dataset...")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    print(f"Loaded dataset with {len(ds)} entries.")
    # Process each row in the dataset
    problems = []
    expected_answers = []

    print("Extracting prompts and answers...")
    for row in ds:
        messages = row.get('messages', [])
        prompt, answer, extracted = extract_prompt_and_answer(messages)
        
        if extracted:
            # If this is a new prompt
            if prompt not in content_map:
                content_map[prompt] = [extracted]
            else:
                # Add this answer to existing answers for this prompt
                content_map[prompt].append(extracted)
    print("Extracted prompts and answers...")
    
    # Find prompts where there are all assistants gave the same answer
    for prompt, answers in content_map.items():
        # There are less answers than the number of model responses or the answer is not consistent
        if len(answers) < len_files or len(set(answers)) != 1:
            continue
        problems.append(prompt)
        expected_answers.append(answers[0])

    print(f"Filtered problems: {len(problems)}")
    print(f"Filtered expected_answers: {len(expected_answers)}")

    return problems, expected_answers

def main(_):
    problems, expected_answers = process_parquet_files(FLAGS.input_parquet_files)
    # Save filtered messages to parquet file
    df = pd.DataFrame({"problems": problems, "expected_answers": expected_answers})
    df.to_parquet(FLAGS.output_file)

if __name__ == "__main__":
    flags.mark_flags_as_required(["input_parquet_files", "output_file"])
    app.run(main)
    