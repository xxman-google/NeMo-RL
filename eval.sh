#!/bin/bash

# Check for exactly 2 arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ckpt_path> <exp_name>"
    exit 1
fi

enable_thinking=true
if [[ $enable_thinking == "true" ]]; then
  max_model_len=32768
  temperature=0.6
  top_p=0.95
  num_tests_per_prompt=1
else
  max_model_len=8192
  temperature=0.7
  top_p=0.8
  num_tests_per_prompt=1
fi

top_k=20

ckpt_path=$1
exp_name=$2
hf_ckpt_path=$ckpt_path/hf

# uv run python examples/converters/convert_dcp_to_hf.py --config $ckpt_path/config.yaml --dcp-ckpt-path $ckpt_path/policy/weights/ --hf-ckpt-path $hf_ckpt_path

benchmarks=("aime2024" "aime2025" "beyond_aime" "math" "math500" "mgsm" "gpqa" "mmlu" "mmlu_pro" "humaneval" "livecodebench_functional" "livecodebench_stdin")
num_tests_per_prompt=(5 5 5 1 1 1 5 1 1 5 5 5)
len=${#benchmarks[@]}

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
  wandb_name="$exp_name-$benchmark_name"

  uv run examples/run_eval.py --config $config_file \
  eval.num_tests_per_prompt=$repeats \
  data.dataset_name=$dataset_name \
  generation.stop_token_ids=\[151643,151645\] \
  generation.enable_thinking=$enable_thinking \
  generation.vllm_cfg.max_model_len=$max_model_len \
  generation.temperature=$temperature \
  generation.top_p=$top_p \
  generation.top_k=$top_k \
  cluster.gpus_per_node=$GPUS_PER_NODE \
  cluster.num_nodes=$NNODES \
  generation.model_name=$hf_ckpt_path \
  logger.wandb.name=$wandb_name
done
