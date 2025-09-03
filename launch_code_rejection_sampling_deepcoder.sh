#!/bin/bash

dataset_name="deepcoder_preview"

# model_names=("qwen3_14b_code_no_thinking" "qwen3_14b_code_thinking")
model_names=("qwen3_8b_code_no_thinking")
subsets=("lcbv5" "primeintellect")

for model_name in "${model_names[@]}"; do
  config_path="examples/configs/rejection_sampling/${model_name}.yaml"
  echo "Reading from ${config_path}"
  if [[ "$model_name" == *"no_thinking" ]]; then
    max_model_len=8192
  else
    max_model_len=38912
  fi
  if [[ "$model_name" == "qwen3_32b_code_thinking" ]]; then
    tp=2
  else
    tp=1
  fi

  for subset in "${subsets[@]}"; do
    echo "subset: $subset"
    output_dir="${LOCAL_LOG_DIR_IN_CONTAINER}/${dataset_name}_${subset}_${model_name}"
    echo "writing results to $output_dir"
    uv run examples/run_rejection_sampling.py \
    --config $config_path \
    generation.num_prompts_per_step=-1 \
    generation.vllm_cfg.max_model_len=$max_model_len \
    generation.vllm_cfg.tensor_parallel_size=$tp \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NNODES \
    logger.log_dir=$LOCAL_LOG_DIR_IN_CONTAINER/ \
    logger.output_dir=$output_dir \
    logger.wandb.name="${model_name}-${dataset_name}-${subset}" \
    data.prompt_file="examples/prompts/code_stdin_stdout.txt" \
    data.dataset_name=$dataset_name \
    data.subset=$subset
    sleep 20
  done

done
