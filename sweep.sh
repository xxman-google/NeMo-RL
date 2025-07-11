#!/bin/bash

lrs=(1e-5 7e-6 5e-6)
config="examples/configs/cirrus/sft-qwen2.5-7b-tulu3-174m.yaml"

for lr in "${lrs[@]}"; do
  lr_float=$(awk "BEGIN {printf \"%f\", $lr}")
  end_lr=$(echo "$lr_float * 0.1" | bc -l)
  echo "lr: $lr, end_lr: $end_lr"
  ckpt_dir="${CKPT_LOCAL_DIR}/sft_qwen2p5_7b_tulu3_174m_lr${lr}"
  echo "writing checkpoints to $ckpt_dir"
  uv run examples/run_sft.py --config ${config} cluster.num_nodes=2 logger.log_dir=$LOCAL_LOG_DIR_IN_CONTAINER checkpointing.checkpoint_dir=$ckpt_dir policy.optimizer.kwargs.lr=$lr policy.scheduler.1.kwargs.eta_min=$end_lr logger.wandb.name=sft_qwen2p5_7b_tulu3_174m_lr$lr
done
