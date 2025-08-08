#!/bin/bash

# Check for exactly 2 arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ckpt_path> <exp_name>"
    exit 1
fi

enable_thinking=true
if $enable_thinking; then
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
#hf_ckpt_path=$ckpt_path
hf_ckpt_path=$ckpt_path/hf

uv run python examples/converters/convert_dcp_to_hf.py --config $ckpt_path/config.yaml --dcp-ckpt-path $ckpt_path/policy/weights/ --hf-ckpt-path $hf_ckpt_path

benchmarks=("aime2024" "aime2025" "gpqa" "math" "math500" "mgsm" "mmlu" "mmlu_pro")
#benchmarks=("mmlu")

for benchmark_name in "${benchmarks[@]}"; do
  if [ $benchmark_name = "math500" ]; then
    config_file="examples/configs/evals/math.yaml"  
  else
    config_file="examples/configs/evals/${benchmark_name}.yaml"
  fi

  echo "Reading from config: ${config_file}"
  wandb_name="$exp_name-$benchmark_name"
  if [ $benchmark_name = "math500" ]; then
    uv run examples/run_eval.py --config $config_file \
      cluster.gpus_per_node=8 \
      cluster.num_nodes=${NNODES} \
      data.dataset_name="math500" \
      eval.num_tests_per_prompt=$num_tests_per_prompt \
      generation.top_p=$top_p \
      generation.top_k=$top_k \
      generation.model_name=$hf_ckpt_path \
      generation.temperature=$temperature \
      generation.stop_token_ids=\[151643,151645\] \
      generation.enable_thinking=$enable_thinking \
      generation.vllm_cfg.max_model_len=$max_model_len \
      logger.wandb.name=$wandb_name
  else
    uv run examples/run_eval.py --config $config_file \
      cluster.gpus_per_node=8 \
      cluster.num_nodes=${NNODES} \
      data.dataset_name=$benchmark_name \
      eval.num_tests_per_prompt=$num_tests_per_prompt \
      generation.top_p=$top_p \
      generation.top_k=$top_k \
      generation.model_name=$hf_ckpt_path \
      generation.temperature=$temperature \
      generation.stop_token_ids=\[151643,151645\] \
      generation.enable_thinking=$enable_thinking \
      generation.vllm_cfg.max_model_len=$max_model_len \
      logger.wandb.name=$wandb_name
  fi
done
