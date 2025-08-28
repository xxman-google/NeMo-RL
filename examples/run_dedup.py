import argparse
from utils_dedup.dedup_utils import DeduplicationPipeline


def main():
    """
    Main function to parse arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="Analyze dataset redundancy using semantic similarity.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet",
        help="Path to the Parquet data file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dedup/",
        help="Path to save deduplicated dataset and visualizations."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='all-MiniLM-L6-v2',
        help="The name of the Sentence-BERT model to use for embeddings. Currently only 'all-MiniLM-L6-v2' is supported."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='cosine',
        choices=['cosine'],
        help="The similarity metric to use. Currently only 'cosine' is supported."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="The cosine similarity threshold for identifying redundant samples. Must be between 0 and 1."
    )

    args = parser.parse_args()

    # Validate threshold
    if not (0 <= args.threshold <= 1):
        parser.error("Threshold must be a value between 0 and 1.")

    # Instantiate and run the pipeline
    pipeline = DeduplicationPipeline(
        model_name=args.model_name,
        metric=args.metric,
        threshold=args.threshold,
    )
    pipeline.run(data_path=args.data_path, output_dir=args.output_dir)

if __name__ == '__main__':
    main()