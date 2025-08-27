set -e
cd .
ls examples/
pwd

uv run examples/run_rejection_sampling.py --config examples/configs/sampling/qwen3_14b_science_thinking_sampling.yaml cluster.num_nodes=8
