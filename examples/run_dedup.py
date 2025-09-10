import argparse
from utils_dedup.dedup_single_set import Deduplication_Pipeline, Gen_Embeddings_Similarity_Pipeline, Dedup_Visualize
import os
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

def load_data_to_df(path):
    """
    Loads data from a Parquet file, preprocesses it, generates embeddings.
    [optional] Saves embeddings to output_path/embeddings.pt if not already present.
    Returns the embeddings, preprocessed questions, and the original DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at: {path}")
    try:
        parquet_file = pq.ParquetFile(path)
        df = parquet_file.read().to_pandas()
        print(f"Loaded dataset at {path} with {len(df)} records.")
    except Exception as e:
        raise IOError(f"Failed to read parquet file at {path}: {e}")
    return df


def extract_embeddings_questions(df, output_path, gen_embed_sim_pipeline, flag_save):
    questions = gen_embed_sim_pipeline._preprocess_data(df)
    if not os.path.exists(f"{output_path}/embeddings.pt"):
        embeddings = gen_embed_sim_pipeline._generate_embeddings(samples=questions, output_path=output_path, flag_save=flag_save)
    else:
        embeddings = torch.load(f"{output_path}/embeddings.pt")
        print(f"Loaded existing embeddings from {output_path}/embeddings.pt")
    return embeddings, questions




def dedup_cross_set(args, df_base, df_candidate, base_suffix, candidate_suffix, flag_save, flag_visualize, type_dedup):
    """
    Performs cross-set deduplication on candiate dataset against base dataset.
    save embeddings, similarity matrix, cleaned dataset.
    save visualizations (optional): box/hist plots of similarity scores, t-SNE of embeddings, similarity score stats
    save analysis: redundancy score, showcase redundant pairs.
    
    """
    
    base_output_dir, candidate_output_dir = args.output_dir + "/" + base_suffix, args.output_dir + "/" + candidate_suffix
    output_dir = args.output_dir + "/" + candidate_suffix + "/dedup_against/" + base_suffix
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(candidate_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)  
    
    # Preprocess data, generate & save embeddings
    gen_embed_sim_pipeline = Gen_Embeddings_Similarity_Pipeline(args.model_name, args.metric)
    gen_embed_sim_pipeline._load_model()
    embeddings_base, questions_base = extract_embeddings_questions(df_base, base_output_dir, gen_embed_sim_pipeline, flag_save)
    if type_dedup == 'intra_set':
        embeddings_candidate, questions_candidate = embeddings_base, questions_base
    else:
        embeddings_candidate, questions_candidate = extract_embeddings_questions(df_candidate, candidate_output_dir, gen_embed_sim_pipeline, flag_save)

    # Calculate & save similarity matrix
    # analysis & visualization: 
    #  -- similarity: stats, box/hist plots, 
    #  -- embeddings: t-SNE,
    
    if not os.path.exists(f"{output_dir}/similarity_matrix.pt"):
        similarity_matrix = gen_embed_sim_pipeline._calculate_similarity_matrix(embeddings1=embeddings_base, embeddings2=embeddings_candidate, output_dir=output_dir, flag_save=flag_save)
    else:
        print(f"Loading existing similarity matrix from {output_dir}/similarity_matrix.pt")
        similarity_matrix = torch.load(f"{output_dir}/similarity_matrix.pt", weights_only=False)
        print(f"Loaded.")
    print("=====================similarity matrix=====================")
    print(similarity_matrix.shape)
    print(type(similarity_matrix))

    dedup_visualize = Dedup_Visualize(flag_intra_set=False)
    
    if flag_visualize:
        if not os.path.exists(f"{output_dir}/similarity_stats_{args.threshold}.txt"):
            dedup_visualize._analyze_scores(similarity_matrix, output_dir)
        if not os.path.exists(f"{output_dir}/cosine_score_boxplot.png") or not os.path.exists(f"{output_dir}/cosine_score_histogram.png") or not os.path.exists(f"{output_dir}/tsne_clusters.png"):
            dedup_visualize._plot_box_and_hist(similarity_matrix, output_dir)
            # dedup_visualize._plot_tsne_clusters(embeddings, output_dir)
    

    dedup_single_set_pipeline = Deduplication_Pipeline(
        threshold=args.threshold,
    )
    flag_intra_set = True if type_dedup=='intra_set' else False
    # set flag_save_cleaned to False for ablation study; to True for normal runs
    indices_scores, indices_to_remove, cleaned_df = dedup_single_set_pipeline._deduplicate_dataset(flag_intra_set=flag_intra_set, df1=df_base, df2=df_candidate, similarity_matrix=similarity_matrix, output_path=output_dir, flag_save_cleaned=False)
    # always calculate redundancy score and showcase pairs for cross-set deduplication
    dedup_single_set_pipeline._calculate_redundancy_score(indices_to_remove, len(questions_candidate), output_dir)
    dedup_visualize._showcase_pairs(df1 = df_base, df2 = df_candidate, indices_scores = indices_scores, threshold = args.threshold, output_dir = output_dir)
    
    return indices_scores, cleaned_df



def dedup_cross_filelist(args):
    """
    Performs cross-set deduplication for a list of candidate files against a list of base files.
    Saves cleaned files at output_dir.
    """
    num_files_base, num_files_candidate = len(args.filelist_path_base), len(args.filelist_path_candidate)
    print(f"Starting cross-set deduplication for {num_files_candidate} candidate files against {num_files_base} base files.")
    gen_embed_sim_pipeline = Gen_Embeddings_Similarity_Pipeline(args.model_name, args.metric)
    gen_embed_sim_pipeline._load_model()
    
    for i in tqdm(range(num_files_candidate), desc="Processing candidate files"):
        if not os.path.exists(args.filelist_path_candidate[i]):
            raise FileNotFoundError(f"Candidate data file not found at: {args.filelist_path_candidate[i]}")
        df_candidate = load_data_to_df(args.filelist_path_candidate[i])
        indices_to_remove = set()
        
        for j in tqdm(range(num_files_base), desc=f"Deduplicating against base files for candidate file {i+1}/{num_files_candidate}"):
            if args.type == 'intra_list' and j < i:
                continue
            if not os.path.exists(args.filelist_path_base[j]):
                raise FileNotFoundError(f"Base data file not found at: {args.filelist_path_base[j]}")
            
            print(f"\nDeduplicating candidate file: {args.filelist_path_candidate[i]} against base file: {args.filelist_path_base[j]}")
            
            # Create unique output directory for this cross-deduplication run
            file_path_base, file_path_candidate = args.filelist_path_base[j], args.filelist_path_candidate[i]
            base_suffix, candidate_suffix = "_".join(file_path_base.split("/")[-2:]).replace(".","_"), "_".join(file_path_candidate.split("/")[-2:]).replace(".","_")
            
            # Preprocess data, generate & save embeddings
            df_base = load_data_to_df(file_path_base)
            type_dedup = 'cross_set'
            if args.type == 'intra_list' and file_path_base == file_path_candidate:
                print("Intra-list deduplication.")
                type_dedup = 'intra_set'
                
            indices_scores, _ = dedup_cross_set(args=args, df_base=df_base, df_candidate=df_candidate, base_suffix=base_suffix, \
                                                candidate_suffix=candidate_suffix, flag_save=True, flag_visualize= False, type_dedup=type_dedup)
            indices_to_remove.update({idx_candidate for (idx_base, idx_candidate), score in indices_scores})
            
            print(f"Completed deduplication for candidate file: {file_path_candidate} against base file: {file_path_base}")
        
         
        print(f"\nOriginal dataset size of {candidate_suffix}: {len(df_candidate)}")
        cleaned_df = df_candidate.drop(index=list(indices_to_remove)).reset_index(drop=True)  
        print(f"Number of samples removed: {len(indices_to_remove)}")
        print(f"Cleaned dataset size: {len(cleaned_df)}")
        os.makedirs(f"{args.output_dir}/{candidate_suffix}/{args.output_suffix}", exist_ok=True)
        cleaned_df.to_parquet(f"{args.output_dir}/{candidate_suffix}/{args.output_suffix}/{args.threshold}.parquet", engine='pyarrow')
        print(f"Cleaned dataset saved to {args.output_dir}/{candidate_suffix}/{args.output_suffix}/{args.threshold}.parquet")
            
           



def main():
    """
    Main function to parse arguments and run the pipeline.
    """
    parser = argparse.ArgumentParser(description="Analyze dataset redundancy using semantic similarity.")
    parser.add_argument(
        "--filelist_path_base",
        type=str,
        nargs='+',
        default="/gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet",
        help="Paths to the files to dedup against."
    )
    parser.add_argument(
        "--filelist_path_candidate",
        type=str,
        nargs='+',
        default="/gcs/cloud-nas-hns-data/cirrus0.0/openr1_math_amc_aime_qwen3_8b_no_thinking/train-unboxed-00000-of-00001.parquet",
        help="Paths to the files to dedup."
    )
    parser.add_argument(
        "--type",
        type=str,
        default="cross_set",
        choices=['intra_list', 'cross_list']
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dedup/",
        help="prefix for Path to save deduplicated dataset and visualizations."
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="train-unboxed-00000-of-00001",
        help="suffix for Path to save deduplicated dataset and visualizations."
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
        default=1.00,
        help="The cosine similarity threshold for identifying redundant samples. Must be between 0 and 1."
    )

    args = parser.parse_args()

    # Validate threshold
    if not (0 <= args.threshold <= 1):
        parser.error("Threshold must be a value between 0 and 1.")
    
    dedup_cross_filelist(args)
    

if __name__ == '__main__':
    main()