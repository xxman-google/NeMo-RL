# if certain packages are not installed, run:
# pip install sentence-transformers scikit-learn matplotlib pandas seaborn

uv run /examples/dedup_run_pip.py \
--data_path /data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet \
--output_dir ./dedup \
--model_name all-MiniLM-L6-v2 \
--metric cosine \
--threshold 0.8 