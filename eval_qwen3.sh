#!/bin/bash

# model_names=("Qwen/Qwen3-8B" "Qwen/Qwen3-14B" "Qwen/Qwen3-32B")
# model_names=("Qwen/Qwen3-235B-A22B-Instruct-2507")
model_names=("Qwen/Qwen3-235B-A22B-Thinking-2507")
enable_thinking=true

if $enable_thinking; then
  max_model_len=38912
  temperature=0.6
  top_p=0.95
else
  max_model_len=8192
  temperature=0.7
  top_p=0.8
fi
if [[ "$model_name" == "Qwen/Qwen3-235B-A22B-Instruct-2507" ]]; then
  max_model_len=16384
elif [[ "$model_name" == "Qwen/Qwen3-235B-A22B-Thinking-2507" ]]; then
  max_model_len=81920
fi

top_k=20
benchmarks=("aime2024" "aime2025" "beyond_aime" "math500" "mgsm" "gpqa" "mmlu" "mmlu_pro" "humaneval" "livecodebench_functional" "livecodebench_stdin" "arc_agi")
num_tests_per_prompt=(5 5 5 1 1 5 1 1 5 5 5 5)
len=${#benchmarks[@]}

for model_name in "${model_names[@]}"; do

  if [[ "$model_name" == "Qwen/Qwen3-235B-A22B-Instruct-2507" && $enable_thinking == "true" ]]; then
    echo "enable_thinking is set to true but Qwen/Qwen3-235B-A22B-Instruct-2507 is a non-thinking model." >&2
    exit 1
  fi
  if [[ "$model_name" == "Qwen/Qwen3-235B-A22B-Thinking-2507" && $enable_thinking == "false" ]]; then
    echo "enable_thinking is set to false but Qwen/Qwen3-235B-A22B-Thinking-2507 is a thinking model." >&2
    exit 1
  fi

  if [[ "$model_name" == "Qwen/Qwen3-32B" && $enable_thinking == "true" ]]; then
    tp=2
    enable_expert_parallel=false
  elif [[ "$model_name" == "Qwen/Qwen3-235B-A22B-Instruct-2507" || "$model_name" == "Qwen/Qwen3-235B-A22B-Thinking-2507" ]]; then
    tp=8
    enable_expert_parallel=true
  else
    tp=1
    enable_expert_parallel=false
  fi

  for ((i=0; i<$len; i++)); do

    benchmark_name=${benchmarks[$i]}
    repeats=${num_tests_per_prompt[$i]}

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
    if [[ $enable_thinking == "true" ]]; then
      wandb_name="$model_name-thinking-$benchmark_name"
    else
      wandb_name="$model_name-non-thinking-$benchmark_name"
    fi
    uv run examples/run_eval.py --config $config_file \
    eval.num_tests_per_prompt=$repeats \
    data.dataset_name=$dataset_name \
    generation.vllm_cfg.gpu_memory_utilization=0.95 \
    generation.vllm_cfg.max_model_len=$max_model_len \
    generation.vllm_cfg.tensor_parallel_size=$tp \
    generation.vllm_cfg.enable_expert_parallel=$enable_expert_parallel \
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
