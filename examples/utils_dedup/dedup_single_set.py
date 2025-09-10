import pandas as pd
import torch
import os
import tqdm
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, Union, List
from sentence_transformers import SentenceTransformer, util



class Deduplication_Pipeline:
    """
    A pipeline for analyzing dataset redundancy using semantic similarity,
    with options for statistical analysis, visualization, a showcase feature,
    and a final deduplication step.
    """

    def __init__(
        self,
        threshold: float = 0.8,
    ):
        """
        Initializes the pipeline with configurable parameters.

        Args:
            model_name (str): The name of the Sentence-BERT model to use.
            metric (str): The similarity metric to use. Currently supports 'cosine'.
            threshold (float): The similarity score threshold for redundancy.
            visualize (bool): Whether to generate and save visualization plots.
            showcase (bool): Whether to display the most and least similar sample pairs.
            deduplicate_and_save (bool): If True, removes redundant samples and saves the
                                         cleaned dataset to a new file.
        """
        self.threshold = threshold


   
    

    def _calculate_redundancy_score(self, redundant_indices: set, total_samples: int, output_dir: str) -> float:
        """
        Calculates the final redundancy score as a percentage and saves the results to a file.
        """
        num_redundant_samples = len(redundant_indices)
        redundancy_score = (num_redundant_samples / total_samples) * 100 if total_samples > 0 else 0
        
        output_file_path = os.path.join(output_dir, "dedup")
        
        with open(f"{output_file_path}redundancy_results_{self.threshold}.txt", 'w') as f:
            f.write("--- Redundancy Analysis Results ---\n")
            f.write(f"Total samples processed: {total_samples}\n")
            f.write(f"Number of unique samples involved in redundancy: {num_redundant_samples}\n")
            f.write(f"Redundancy Score: {redundancy_score:.2f}%\n")
            f.write("---\n")
            
            if num_redundant_samples > 0:
                f.write("Note: This score represents the percentage of your dataset that is part of a redundant pair.\n")
                
        print(f"\nRedundancy analysis results saved to: {output_file_path}redundancy_results_{self.threshold}.txt")
        
        return redundancy_score    


    def _deduplicate_dataset(self, flag_intra_set: bool, df1: pd.DataFrame, df2:pd.DataFrame, similarity_matrix: torch.Tensor, output_path: str, flag_save_cleaned: bool):
        """
        Removes redundant samples from the DataFrame df2 and saves the cleaned dataset.
        This optimized version avoids the nested loop for faster execution.
        """
        total_samples = len(df2)
        
        print(f"\nIdentifying redundant pairs with similarity > {self.threshold}...")
        
        # Use torch.triu to get the upper triangle of the similarity matrix, excluding the diagonal
        # This avoids redundant comparisons (e.g., (1, 2) and (2, 1)) and self-comparisons
        # upper_triangular_indices = torch.triu(torch.ones(total_samples, total_samples), diagonal=1).bool()
        
        # Get the indices of all pairs where the score is above the threshold
        redundant_indices = torch.where(similarity_matrix >= self.threshold)
        
        # Filter the redundant_indices to only consider the upper triangle
        if flag_intra_set:
            valid_redundant_indices = [
                (i, j) for i, j in zip(redundant_indices[0].tolist(), redundant_indices[1].tolist())
                if i < j # Ensure we only consider one of each pair
            ]
        else:
            valid_redundant_indices = [
                (i, j) for i, j in zip(redundant_indices[0].tolist(), redundant_indices[1].tolist())
            ]

        # Create a set of indices to remove
        indices_to_remove = set()
        pairs_indice_score = set()
        
        for i, j in valid_redundant_indices:
            # A simple strategy: remove the sample with the higher index in each redundant pair.
            # This ensures one of the duplicates is kept.
            indices_to_remove.add(j)
            pairs_indice_score.add(((i, j), similarity_matrix[i][j].item()))

        print(f"\nRemoving {len(indices_to_remove)} samples from a total of {total_samples}...")
        
        cleaned_df = df2.drop(index=list(indices_to_remove)).reset_index(drop=True)
        
        print(f"\nOriginal dataset size: {total_samples}")
        print(f"Number of samples removed: {len(indices_to_remove)}")
        print(f"Cleaned dataset size: {len(cleaned_df)}")
        
        print("flag_save_cleaned should be set to false", flag_save_cleaned)
        if flag_save_cleaned:
            os.makedirs(os.path.dirname(f"{output_path}/dedup/"), exist_ok=True)
            print(f"=====================dir made for {output_path}/dedup/=====================")
            cleaned_df.to_parquet(f"{output_path}/dedup/{self.threshold}.parquet", engine='pyarrow')
            print(f"Cleaned dataset saved to {output_path}/dedup/{self.threshold}.parquet")
        return pairs_indice_score, indices_to_remove, cleaned_df
    
    




class Dedup_Visualize:
    """
    A class for visualizing and analyzing similarity scores and embeddings.
    """

    def __init__(self, flag_intra_set: bool):
        self.flag_intra_set = flag_intra_set
    
    
    def _analyze_scores(self, similarity_matrix: torch.Tensor, output_dir: str) -> Dict[str, Union[int, float]]:
        """
        Calculates and saves descriptive statistics for the similarity scores to a file.
        """
        # Use torch's triu_indices to get the upper triangle values directly.
        # This avoids creating a full copy of the entire matrix.
        # The 
        if self.flag_intra_set:
            upper_triangle = similarity_matrix[torch.triu_indices(similarity_matrix.shape[0], similarity_matrix.shape[1], offset=1)]
            upper_triangle_np = upper_triangle.cpu().numpy()
        else:
            upper_triangle = similarity_matrix.flatten()
            upper_triangle_np = upper_triangle.cpu().numpy()
        print("=====================similarity matrix upper triangle=====================")
        print(similarity_matrix.shape)
        print(similarity_matrix)
        # Move the result to a small NumPy array for calculation if needed, but it's small.

        stats = {
            "number_of_unique_pairs": len(upper_triangle_np),
            "mean": upper_triangle_np.mean(),
            "std": upper_triangle_np.std(),
            "min_score": upper_triangle_np.min(),
            "max_score": upper_triangle_np.max(),
            "median": np.median(upper_triangle_np),
            "25th_percentile": np.percentile(upper_triangle_np, 25),
            "75th_percentile": np.percentile(upper_triangle_np, 75)
        }

        output_file_path = os.path.join(output_dir, "similarity_stats.txt")
        with open(output_file_path, 'w') as f:
            f.write("--- Similarity Score Statistics ---\n")
            for key, value in stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            f.write("---\n")

        print(f"\nSimilarity statistics saved to: {output_file_path}")
        return stats
    
    
    # def _analyze_scores(self, similarity_matrix: torch.Tensor, output_dir: str) -> Dict[str, Union[int, float]]:
    #     """
    #     Calculates and saves descriptive statistics for the similarity scores to a file.
    #     """
    #     scores_np = similarity_matrix.cpu().numpy()
    #     upper_triangle = scores_np[np.triu_indices(scores_np.shape[0], k=1)]

    #     stats = {
    #         "number_of_unique_pairs": len(upper_triangle),
    #         "mean": np.mean(upper_triangle),
    #         "std": np.std(upper_triangle),
    #         "min_score": np.min(upper_triangle),
    #         "max_score": np.max(upper_triangle),
    #         "median": np.median(upper_triangle),
    #         "25th_percentile": np.percentile(upper_triangle, 25),
    #         "75th_percentile": np.percentile(upper_triangle, 75)
    #     }

    #     output_file_path = os.path.join(output_dir, "similarity_stats.txt")
    #     with open(output_file_path, 'w') as f:
    #         f.write("--- Similarity Score Statistics ---\n")
    #         for key, value in stats.items():
    #             if isinstance(value, float):
    #                 f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
    #             else:
    #                 f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    #         f.write("---\n")

    #     print(f"\nSimilarity statistics saved to: {output_file_path}")
    #     return stats



    def _plot_histgram(self, cosine_scores: np.ndarray, output_dir: str):
            """Create and save a histogram of cosine similarity scores."""
            # Extract the upper triangle of the cosine similarity matrix, excluding the diagonal

            print("Creating and saving the histogram...")
            plt.figure(figsize=(10, 6))
            plt.hist(cosine_scores, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribution of Cosine Similarity Scores')
            plt.xlabel('Cosine Similarity Score')
            plt.ylabel('Density')
            plt.grid(axis='y', alpha=0.75)

            # Add vertical lines for common thresholds
            plt.axvline(x=0.8, color='red', linestyle='--', linewidth=1, label='Similarity Threshold (0.8)')
            plt.axvline(x=0.9, color='orange', linestyle='--', linewidth=1, label='Similarity Threshold (0.9)')

            plt.legend()

            # Save the plot to a file
            plt.savefig(output_dir + '/cosine_score_histogram.png')
            plt.close()
            print(f"\nHistogram saved as {output_dir}/cosine_score_histogram.png")





    def _plot_boxplot(self, cosine_scores: np.ndarray, output_dir: str):
        """Create and save a box plot of cosine similarity scores."""

        # Step 4: Create and save the box plot
        print("\nCreating and saving the box plot...")
        plt.figure(figsize=(8, 6))
        plt.boxplot(cosine_scores, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
        plt.title('Distribution of Cosine Similarity Scores', fontsize=16)
        plt.xlabel('Cosine Similarity Score', fontsize=12)
        plt.yticks([])  # Hide y-axis labels as there's only one box plot
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Step 5: Add statistical annotations
        median = np.median(cosine_scores)
        plt.axvline(median, color='red', linestyle='--', linewidth=1.5, label=f'Median: {median:.4f}')
        plt.legend()
        
        # Save the plot to a file
        plt.savefig(output_dir + '/cosine_score_boxplot.png')
        plt.close()
        print(f"Box plot saved as {output_dir}/cosine_score_boxplot.png")



    def _plot_box_and_hist(self, similarity_matrix: torch.Tensor, output_dir: str):
        """
        Generates and saves a box plot and histogram of the similarity scores.
        This is a wrapper function that calls the individual plotting methods.
        """
        # Get the upper triangle of the matrix to avoid redundant pairs and self-comparisons
        if self.flag_intra_set:
            scores = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)].cpu().numpy()
        else:
            scores = similarity_matrix.flatten().cpu().numpy()
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        self._plot_histgram(scores, output_dir)
        self._plot_boxplot(scores, output_dir)


    def _plot_tsne_clusters(self, embeddings: torch.Tensor, output_dir: str):
        """Generates and saves a t-SNE plot to visualize clusters of embeddings."""
        print("Performing t-SNE dimensionality reduction...")
        subset_size = min(len(embeddings), 2000)
        indices = np.random.choice(len(embeddings), subset_size, replace=False)
        subset_embeddings = embeddings[indices]
        
        try:
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            tsne_results = tsne.fit_transform(subset_embeddings.cpu().numpy())

            plt.figure(figsize=(10, 8))
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=10, alpha=0.6)
            plt.title(f't-SNE Visualization of Sample Embeddings (Subset of {subset_size})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.savefig(os.path.join(output_dir, 'tsne_clusters.png'))
            plt.close()
            print(f"Saved t-SNE plot to {output_dir}/tsne_clusters.png")
        except TypeError as e:
            print(f"Warning: Could not perform t-SNE visualization due to a potential API change. Error: {e}")
            print("The pipeline will continue without this plot.")
    
    def _showcase_pairs(self, df1: pd.DataFrame, df2: pd.DataFrame, indices_scores: set, threshold: float, output_dir: str):
        """
        Finds and saves all pairs whose cosine similarity is larger than the threshold
        to a text file.
        """
        output_file_path = os.path.join(output_dir, 'dedup')

        with open(f"{output_file_path}/{threshold}_showcase_pairs.txt", 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write(f"--- Showcase: Redundant Pairs with Similarity > {threshold:.4f} ---\n\n")
            found_pairs = False
            sorted_indices_scores = sorted(list(indices_scores), key=lambda x: x[1], reverse=True)
            
            for index,score in sorted_indices_scores:
                found_pairs = True
                i,j = index
                f.write(f"Score: {score:.4f} (Pair: Indices {i} and {j})\n")
                
                sample1_text = df1.iloc[i]['messages']
                sample2_text = df2.iloc[j]['messages']
                
                f.write("Sample 1:\n")
                f.write(f"Question: {sample1_text[0]['content']}\n")
                # f.write(f"Answer: {sample1_text[1]['content']}\n\n")

                f.write("Sample 2:\n")
                f.write(f"Question: {sample2_text[0]['content']}\n")
                # f.write(f"Answer: {sample2_text[1]['content']}\n")
                f.write("="*50 + "\n\n\n\n")
            
            if not found_pairs:
                f.write("No pairs found above the specified threshold.\n")
                
            f.write("="*50 + "\n")
            
        print(f"Showcase of redundant pairs saved to: {output_file_path}/{threshold}_showcase_pairs.txt")

    def _print_sample_pair(self, df: pd.DataFrame, indices: tuple):
        """Helper function to print a pair of samples."""
        for i, idx in enumerate(indices):
            sample_text = df.iloc[idx]['messages']
            print(f"\nSample {i + 1} (Index {idx}):")
            print(f"Question: {sample_text[0]['content']}")
            print("---")
            print(f"Answer: {sample_text[1]['content']}")
    




class Gen_Embeddings_Similarity_Pipeline:
    
    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        metric: str = 'cosine',
    ):
        """
        Initializes the pipeline with configurable parameters.

        Args:
            model_name (str): The name of the Sentence-BERT model to use.
            metric (str): The similarity metric to use. Currently supports 'cosine'.
        """
        self.model_name = model_name
        self.metric = metric
        self.model = None
    
    def _load_model(self):
        """Loads the Sentence-BERT model."""
        if self.model is None:
            print(f"\nLoading Sentence-BERT model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def _preprocess_data(self, df: pd.DataFrame) -> List[str]:
        """Combines question and answer text from the DataFrame."""
        questions = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            messages = row['messages']
            question = messages[0]['content']
            questions.append(f"{question}")
        print(f"{len(questions)} samples preprocessed.")
        return questions

    def _generate_embeddings(self, samples: List[str], output_path:str, flag_save:bool) -> torch.Tensor:
        """Generates embeddings for the given text samples."""
        print("Generating embeddings for samples...")
        embeddings = self.model.encode(samples, convert_to_tensor=True, show_progress_bar=True)
        if flag_save:
            torch.save(embeddings, f'{output_path}/embeddings.pt')
        return embeddings

    def _calculate_similarity_matrix(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, output_dir:str, flag_save:bool) -> torch.Tensor:
        """Calculates the similarity matrix based on the chosen metric."""
        print(f"\nCalculating {self.metric} similarity matrix...")
        if self.metric == 'cosine':
            cosine_score = util.cos_sim(embeddings1, embeddings2)
            if flag_save:
                print("similarity matrix calculated")
                print(f"{cosine_score.shape} cosine similarity matrix")
                # simple_matrix = cosine_score.cpu().numpy()
                # print("=====================similarity matrix=====================")
                # print(simple_matrix[0][0])
                # print(simple_matrix)
                # torch.save(simple_matrix, f'{output_dir}/similarity_matrix.pt')
                # print("similarity matrix saved")
                torch.save(cosine_score, f'{output_dir}/similarity_matrix.pt')
            return cosine_score
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")