#!/bin/bash

model_names=("google/gemma-3-12b-it")

max_model_len=8192
temperature=0.0
top_p=1.0
top_k=-1
num_tests_per_prompt=1
benchmarks=("aime2024" "aime2025" "beyond_aime" "gpqa" "math" "math500" "mgsm" "mmlu" "mmlu_pro" "humaneval" "livecodebench_functional" "livecodebench_stdin" "ifeval")

for model_name in "${model_names[@]}"; do

  for benchmark_name in "${benchmarks[@]}"; do
    if [ $benchmark_name = "math500" ]; then
      config_file="examples/configs/evals/math.yaml"  
    else
      config_file="examples/configs/evals/${benchmark_name}.yaml"
    fi
    if [[ $benchmark_name == "livecodebench"* ]]; then
      dataset_name="livecodebench"
    else
      dataset_name=$benchmark_name
    fi

    echo "Reading from config: ${config_file}"
    wandb_name="$model_name-$benchmark_name"
    uv run examples/run_eval.py --config $config_file \
    eval.num_tests_per_prompt=$num_tests_per_prompt \
    data.dataset_name=$dataset_name \
    generation.vllm_cfg.max_model_len=$max_model_len \
    generation.temperature=$temperature \
    generation.top_p=$top_p \
    generation.top_k=$top_k \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NNODES \
    generation.model_name=$model_name \
    logger.wandb.name=$wandb_name
    sleep 20
  done

done
