#!/bin/bash

# Check for exactly 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <ckpt_path> <exp_name> <is_megatron>"
    exit 1
fi

enable_thinking=true
append_think_token_to_user_msg=false
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
is_megatron=$3
hf_ckpt_path=$ckpt_path/hf

if [ ! -d "$hf_ckpt_path" ]; then
  if [ $is_megatron = "false" ]; then
    uv run python examples/converters/convert_dcp_to_hf.py \
    --config $ckpt_path/config.yaml \
    --dcp-ckpt-path $ckpt_path/policy/weights/ \
    --hf-ckpt-path $hf_ckpt_path
  else
    uv run --extra mcore python examples/converters/convert_megatron_to_hf.py \
    --config $ckpt_path/config.yaml \
    --megatron-ckpt-path $ckpt_path/policy/weights/iter_0000000 \
    --hf-ckpt-path $hf_ckpt_path
  fi
fi

benchmarks=("aime2024" "aime2025" "beyond_aime" "math500" "mgsm" "gpqa" "mmlu" "mmlu_pro" "humaneval" "livecodebench_functional" "livecodebench_stdin" "ifeval" "arc_agi")
num_tests_per_prompt=(5 5 5 1 1 5 1 1 5 5 5 1 5)

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
  data.append_think_token_to_user_msg=$append_think_token_to_user_msg \
  env.math.end_thinking_token="</think>" \
  eval.num_tests_per_prompt=$repeats \
  data.dataset_name=$dataset_name \
  generation.stop_token_ids=\[151643,151645\] \
  generation.enable_thinking=$enable_thinking \
  generation.vllm_cfg.max_model_len=$max_model_len \
  generation.vllm_cfg.tensor_parallel_size=1 \
  generation.vllm_cfg.gpu_memory_utilization=0.9 \
  generation.temperature=$temperature \
  generation.top_p=$top_p \
  generation.top_k=$top_k \
  cluster.gpus_per_node=$GPUS_PER_NODE \
  cluster.num_nodes=$NNODES \
  generation.model_name=$hf_ckpt_path \
  logger.wandb.name=$wandb_name
done
