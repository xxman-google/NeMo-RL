#!/bin/bash

dataset_name="openr1_math"

model_names=("qwen3_8b")
sources=("amc_aime" "aops_forum" "cn_contest" "inequalities" "number_theory" "olympiads" "olympiads_ref")

for model_name in "${model_names[@]}"; do

  for source in "${sources[@]}"; do
    nonthinking_input_dir="${LOCAL_LOG_DIR_IN_CONTAINER}/${dataset_name}_${source}_${model_name}_no_thinking"
    thinking_input_dir="${LOCAL_LOG_DIR_IN_CONTAINER}/${dataset_name}_${source}_${model_name}_thinking"

    uv run examples/run_separate_thinking_data.py \
    --nonthinking_parquet_files="${nonthinking_input_dir}/train-filtered-00000-of-00001.parquet" \
    --thinking_parquet_files="${thinking_input_dir}/train-filtered-00000-of-00001.parquet"
  done

done
