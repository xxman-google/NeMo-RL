#!/bin/bash
#
# Check for exactly 2 arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <config_path> <exp_name>"
    exit 1
fi

config_path=$1
exp_name=$2

lrs=(5.67e-6 7e-6)

for lr in "${lrs[@]}"; do
  lr_float=$(awk "BEGIN {printf \"%.10f\", $lr}")
  end_lr=$(echo "$lr_float * 0.1" | bc -l)
  echo "lr: $lr, end_lr: $end_lr"
  ckpt_dir="${CKPT_LOCAL_DIR}/${exp_name}_lr${lr}"
  echo "writing checkpoints to $ckpt_dir"
  uv run examples/run_sft.py --config ${config_path} cluster.num_nodes=2 logger.log_dir=$LOCAL_LOG_DIR_IN_CONTAINER checkpointing.checkpoint_dir=$ckpt_dir policy.optimizer.kwargs.lr=$lr policy.scheduler.1.kwargs.eta_min=$end_lr logger.wandb.name=${exp_name}_lr$lr
done
