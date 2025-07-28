#!/bin/bash

model_names=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B")
enable_thinking=true
if $enable_thinking; then
  max_model_len=38912
  temperature=0.6
  top_p=0.95
  num_tests_per_prompt=3
  metric="pass@1,3"
else
  max_model_len=8192
  temperature=0.7
  top_p=0.8
  num_tests_per_prompt=5
  metric="pass@1,5"
fi

top_k=20
benchmarks=("aime2024" "aime2025" "gpqa" "math" "math500" "mgsm" "mmlu" "mmlu_pro")

for model_name in "${model_names[@]}"; do

  if [[ "$model_name" == "Qwen/Qwen3-32B" && $enable_thinking]]; then
    tp=2
  else
    tp=1
  fi

  for benchmark_name in "${benchmarks[@]}"; do
    if [ $benchmark_name = "math500" ]; then
      config_file="examples/configs/evals/math.yaml"  
    else
      config_file="examples/configs/evals/${benchmark_name}.yaml"
    fi

    echo "Reading from config: ${config_file}"
    wandb_name="$model_name-thinking-$benchmark_name"
    uv run examples/run_eval.py --config $config_file \
    eval.metric=$metric \
    eval.num_tests_per_prompt=$num_tests_per_prompt \
    data.dataset_name=$benchmark_name \
    generation.vllm_cfg.max_model_len=$max_model_len \
    generation.vllm_cfg.tensor_parallel_size=$tp \
    generation.temperature=$temperature \
    generation.top_p=$top_p \
    generation.top_k=$top_k \
    generation.enable_thinking=$enable_thinking \
    cluster.gpus_per_node=$GPUS_PER_NODE \
    cluster.num_nodes=$NNODES \
    generation.model_name=$model_name \
    logger.wandb.name=$wandb_name
    sleep 20
  done

done
