# Dataset Deduplication and Analysis Pipeline

Enabled a Python pipeline for analyzing and deduplicating a dataset based on semantic similarity. The pipeline uses Sentence-BERT embeddings and cosine similarity to identify and remove redundant data points, ensuring a high-quality, non-repetitive dataset for model training.

***

## Features

- **Deduplication**: Identifies and removes redundant samples based on a configurable similarity threshold.
- **Visualizations**: Generates and saves a histogram, box plot, and t-SNE plot to provide insights into the dataset's redundancy and diversity.
- **Showcase**: Saves a detailed log of all redundant pairs to a text file for manual review.

***

## Getting Started

If running with docker in interative mode, see an example command in dedup.sh
```
uv run examples/dedup_run_pip.py
```
install missing packages by uv pip install MISSING_PACKAGE.

If not running with docker, to run the pipeline, execute the run_dedup.py script from your terminal. You can configure the pipeline's behavior using various command-line arguments.

Example Command
```
python examples/run_dedup.py
```


You can customize the pipeline's behavior with the following options:

```
 --filelist_path_candidate YOUR_LIST_OF_FILES_TO_DEDUP --filelist_path_base YOUR_LIST_OF_FILES_TO_DEDUP_AGAINST --type YOUR_DEDUP_TYPE --output_dir YOUR_OUTPUT_DIR --model_name YOUR_ENCODER --metric YOUR_SIMILARITY_METRIC --threshold YOUR_THRESHOLD
```


# Example usage:
# usecase1: intra-set deduplication
```
python3 examples/run_dedup.py --filelist_path_base /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet \
  --filelist_path_candidate /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet \
  --type intra_list --threshold 0.95 --output_dir ./output_dedup --output_suffix usecase1_test
```

# usecase2: cross-set deduplication
```
python3 examples/run_dedup.py --filelist_path_base /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet \
  --filelist_path_candidate /gcs/cloud-nas-hns-data/cirrus0.2/openr1_math_amc_aime_qwen3_8b_thinking/train-unboxed-00000-of-00001.parquet \
  --type cross_list --threshold 0.95 --output_dir ./output_dedup --output_suffix usecase2_test
```

# usecase3: single file dedup against a list of files
```
python3 examples/run_dedup.py --filelist_path_base /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_aops_forum_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet\
  --filelist_path_candidate /gcs/cloud-nas-hns-data/cirrus0.2/openr1_math_amc_aime_qwen3_8b_thinking/train-unboxed-00000-of-00001.parquet \
  --type cross_list --threshold 0.95 --output_dir ./output_dedup --output_suffix usecase3_test
```

# usecase4: a list of files dedup against themselves
```
python3 examples/run_dedup.py --filelist_path_base /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_aops_forum_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet\
  --filelist_path_candidate /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_aops_forum_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet \
  --type intra_list --threshold 0.95 --output_dir ./output_dedup --output_suffix usecase4_test
```

# usecase5: a list of files dedup against another list of files
```
python3 examples/run_dedup.py --filelist_path_base /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet /gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_aops_forum_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet\
  --filelist_path_candidate /gcs/cloud-nas-hns-data/cirrus0.2/openr1_math_amc_aime_qwen3_8b_thinking/train-unboxed-00000-of-00001.parquet /gcs/cloud-nas-hns-data/cirrus0.2/openr1_math_aops_forum_qwen3_8b_thinking/train-unboxed-00000-of-00001.parquet \
  --type cross_list --threshold 0.95 --output_dir ./output_dedup --output_suffix usecase5_test
```
