# if certain packages are not installed, run:
# pip install sentence-transformers scikit-learn matplotlib pandas seaborn

uv run examples/run_dedup.py \
  --filelist_path_base /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_aops_forum_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet \
  --filelist_path_candidate /gcs/cloud-nas-hns-data/cirrus0.2/openr1_math_amc_aime_qwen3_8b_thinking/train-unboxed-00000-of-00001.parquet /gcs/cloud-nas-hns-data/cirrus0.2/openr1_math_aops_forum_qwen3_8b_thinking/train-unboxed-00000-of-00001.parquet \
  --type cross_list --threshold 0.95 --output_dir ./output_dedup --output_suffix usecase5_test
