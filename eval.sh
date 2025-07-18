#!/bin/bash

# Check for exactly 2 arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <ckpt_path> <exp_name>"
    exit 1
fi

ckpt_path=$1
exp_name=$2
hf_ckpt_path=$ckpt_path/hf

uv run python examples/converters/convert_dcp_to_hf.py --config $ckpt_path/config.yaml --dcp-ckpt-path $ckpt_path/policy/weights/ --hf-ckpt-path $hf_ckpt_path

benchmarks=("aime2024" "gpqa" "math" "math500" "mgsm" "mmlu" "mmlu_pro")

for benchmark_name in "${benchmarks[@]}"; do
  if [ $benchmark_name = "math500" ]; then
    config_file="examples/configs/evals/math.yaml"  
  else
    config_file="examples/configs/evals/${benchmark_name}.yaml"
  fi

  echo "Reading from config: ${config_file}"
  wandb_name="$exp_name-$benchmark_name"
  if [ $benchmark_name = "math500" ]; then
    uv run examples/run_eval.py --config $config_file data.dataset_name="math500" generation.stop_token_ids=\[151643,151645\] cluster.gpus_per_node=8 generation.model_name=$hf_ckpt_path logger.wandb.name=$wandb_name
  else
    uv run examples/run_eval.py --config $config_file generation.stop_token_ids=\[151643,151645\] cluster.gpus_per_node=8 generation.model_name=$hf_ckpt_path logger.wandb.name=$wandb_name
  fi
done
