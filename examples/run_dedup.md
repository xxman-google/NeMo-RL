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
uv run /examples/dedup_run_pip.py \
--data_path /data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet \
--output_dir ./dedup \
--model_name all-MiniLM-L6-v2 \
--metric cosine \
--threshold 0.8 
```
install missing packages by uv pip install MISSING_PACKAGE.

If not running with docker, to run the pipeline, execute the run_dedup.py script from your terminal. You can configure the pipeline's behavior using various command-line arguments.

Example Command
```
python run_dedup.py
```
This will perform the following actions:
- Load the default dataset.
- Generate embeddings and a similarity matrix.
- Save statistical analysis, visualization plots (histogram, box plot, t-SNE), and a showcase of redundant pairs to the ./output_dedup/ directory.
- Save the deduplicated dataset as dedup.parquet inside the same output directory.

You can customize the pipeline's behavior with the following options:

```
 --data_path YOUR_DATAPATH --output_dir YOUR_OUTPUT_DIR --model_name YOUR_ENCODER --metric YOUR_SIMILARITY_METRIC --threshold YOUR_THRESHOLD
```