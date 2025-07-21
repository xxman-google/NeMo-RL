#!/bin/bash


configs=("examples/configs/cirrus/official/sft-qwen2.5-32b-tulu3-238m.yaml")
exp_names=("sft-qwen2.5-32b-tulu3-238m")
len=${#configs[@]}

for ((i=0; i<$len; i++)); do
  config=${configs[$i]}
  exp_name=${exp_names[$i]}

  ckpt_dir="${CKPT_LOCAL_DIR}/${exp_name}"
  echo "writing checkpoints to $ckpt_dir"
  uv run examples/run_sft.py --config ${config} cluster.num_nodes=4 logger.log_dir=$LOCAL_LOG_DIR_IN_CONTAINER checkpointing.checkpoint_dir=$ckpt_dir logger.wandb.name=${exp_name}
done
