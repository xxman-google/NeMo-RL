#!/bin/bash

dataset_name="openr1_math"

model_names=("qwen3_14b_no_thinking" "qwen3_14b_thinking")
sources=("amc_aime" "aops_forum" "cn_contest" "inequalities" "number_theory" "olympiads" "olympiads_ref")

for model_name in "${model_names[@]}"; do

  for source in "${sources[@]}"; do
    input_dir="${LOCAL_LOG_DIR_IN_CONTAINER}/${dataset_name}_${source}_${model_name}"
    echo "Reading parquet files from $input_dir"
    uv run examples/run_repetition_filtering.py \
    --input_parquet_files="${input_dir}/train-00000-of-00001.parquet" \
    --dataset_name="${dataset_name}_${source}" \
    --tokenizer="Qwen/Qwen3-14B" \
    --wandb_log_dir=$LOCAL_LOG_DIR_IN_CONTAINER/
  done

done
