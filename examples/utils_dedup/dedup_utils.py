import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import os
import tqdm
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import List, Dict, Any, Union

class DeduplicationPipeline:
    """
    A pipeline for analyzing dataset redundancy using semantic similarity,
    with options for statistical analysis, visualization, a showcase feature,
    and a final deduplication step.
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        metric: str = 'cosine',
        threshold: float = 0.8,
    ):
        """
        Initializes the pipeline with configurable parameters.

        Args:
            model_name (str): The name of the Sentence-BERT model to use.
            metric (str): The similarity metric to use. Currently supports 'cosine'.
            threshold (float): The similarity score threshold for redundancy.
        """
        self.model_name = model_name
        self.metric = metric
        self.threshold = threshold
        self.model = None

    def _load_model(self):
        """Loads the Sentence-BERT model."""
        if self.model is None:
            print(f"\nLoading Sentence-BERT model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def _preprocess_data(self, df: pd.DataFrame) -> List[str]:
        """Combines question and answer text from the DataFrame."""
        combined_samples = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            messages = row['messages']
            question = messages[0]['content']
            answer = messages[1]['content']
            combined_samples.append(f"Question: {question}\nAnswer: {answer}")
        print(f"{len(combined_samples)} samples preprocessed.")
        return combined_samples

    def _generate_embeddings(self, samples: List[str], output_dir:str) -> torch.Tensor:
        """Generates embeddings for the given text samples."""
        print("Generating embeddings for samples...")
        embeddings = self.model.encode(samples, convert_to_tensor=True, show_progress_bar=True)
        torch.save(embeddings, f'{output_dir}/embeddings.pt')
        return embeddings

    def _calculate_similarity_matrix(self, embeddings: torch.Tensor, output_dir:str) -> torch.Tensor:
        """Calculates the similarity matrix based on the chosen metric."""
        print(f"\nCalculating {self.metric} similarity matrix...")
        if self.metric == 'cosine':
            cosine_score = util.cos_sim(embeddings, embeddings)
            torch.save(cosine_score, f'{output_dir}/similarity_matrix.pt')
            return cosine_score
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def _analyze_scores(self, similarity_matrix: torch.Tensor, total_samples: int) -> Dict[str, Union[int, float]]:
        """Calculates and prints descriptive statistics for the similarity scores."""
        scores_np = similarity_matrix.cpu().numpy()
        upper_triangle = scores_np[np.triu_indices(scores_np.shape[0], k=1)]

        stats = {
            "number_of_unique_pairs": len(upper_triangle),
            "mean": np.mean(upper_triangle),
            "std": np.std(upper_triangle),
            "min_score": np.min(upper_triangle),
            "max_score": np.max(upper_triangle),
            "median": np.median(upper_triangle),
            "25th_percentile": np.percentile(upper_triangle, 25),
            "75th_percentile": np.percentile(upper_triangle, 75)
        }
        
        print("\n--- Similarity Score Statistics ---")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")
        print("---")
        return stats

    def _find_redundant_samples(self, similarity_matrix: torch.Tensor, total_samples: int) -> set:
        """Identifies and returns the indices of redundant samples."""
        redundant_indices = set()
        print(f"\nIdentifying redundant pairs with a similarity score > {self.threshold}...")
        for i in tqdm.tqdm(range(total_samples), desc="Finding Redundancy"):
            for j in range(i + 1, total_samples):
                if similarity_matrix[i][j] > self.threshold:
                    redundant_indices.add(i)
                    redundant_indices.add(j)
        return redundant_indices

    def _calculate_redundancy_score(self, redundant_indices: set, total_samples: int) -> float:
        """Calculates the final redundancy score as a percentage."""
        num_redundant_samples = len(redundant_indices)
        redundancy_score = (num_redundant_samples / total_samples) * 100 if total_samples > 0 else 0
        
        print("\n--- Redundancy Analysis Results ---")
        print(f"Total samples processed: {total_samples}")
        print(f"Number of unique samples involved in redundancy: {num_redundant_samples}")
        print(f"Redundancy Score: {redundancy_score:.2f}%")
        print("---")
        
        if num_redundant_samples > 0:
            print("Note: This score represents the percentage of your dataset that is part of a redundant pair.")
            
        return redundancy_score

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
        scores = similarity_matrix[np.triu_indices(len(similarity_matrix), k=1)].cpu().numpy()
        
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

    def _showcase_pairs(self, df: pd.DataFrame, similarity_matrix: torch.Tensor, output_dir: str):
        """
        Finds and saves all pairs whose cosine similarity is larger than the threshold
        to a text file.
        """
        output_file_path = os.path.join(output_dir, 'showcase_pairs.txt')

        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write(f"--- Showcase: Redundant Pairs with Similarity > {self.threshold:.4f} ---\n\n")

            total_samples = len(df)
            found_pairs = False
            
            for i in range(total_samples):
                for j in range(i + 1, total_samples):
                    score = similarity_matrix[i][j].item()
                    if score > self.threshold:
                        found_pairs = True
                        f.write(f"Score: {score:.4f} (Pair: Indices {i} and {j})\n")
                        
                        sample1_text = df.iloc[i]['messages']
                        sample2_text = df.iloc[j]['messages']
                        
                        f.write("Sample 1:\n")
                        f.write(f"Question: {sample1_text[0]['content']}\n")
                        f.write(f"Answer: {sample1_text[1]['content']}\n\n")

                        f.write("Sample 2:\n")
                        f.write(f"Question: {sample2_text[0]['content']}\n")
                        f.write(f"Answer: {sample2_text[1]['content']}\n")
                        f.write("#"*50 + "\n\n\n\n")
            
            if not found_pairs:
                f.write("No pairs found above the specified threshold.\n")
                
            f.write("="*50 + "\n")
            
        print(f"Showcase of redundant pairs saved to: {output_file_path}")

    def _print_sample_pair(self, df: pd.DataFrame, indices: tuple):
        """Helper function to print a pair of samples."""
        for i, idx in enumerate(indices):
            sample_text = df.iloc[idx]['messages']
            print(f"\nSample {i + 1} (Index {idx}):")
            print(f"Question: {sample_text[0]['content']}")
            print("---")
            print(f"Answer: {sample_text[1]['content']}")
    
    def _deduplicate_dataset(self, df: pd.DataFrame, similarity_matrix: torch.Tensor, output_path: str):
        """
        Removes redundant samples from the DataFrame and saves the cleaned dataset.
        """
        total_samples = len(df)
        indices_to_remove = set()
        
        print(f"\nRemoving redundant samples with similarity > {self.threshold}...")
        for i in tqdm.tqdm(range(total_samples), desc="Deduplicating"):
            for j in range(i + 1, total_samples):
                if similarity_matrix[i][j] > self.threshold:
                    # A simple strategy: remove the sample with the higher index in each redundant pair.
                    # This ensures one of the duplicates is kept.
                    indices_to_remove.add(j)

        cleaned_df = df.drop(index=list(indices_to_remove)).reset_index(drop=True)
        
        print(f"\nOriginal dataset size: {total_samples}")
        print(f"Number of samples removed: {len(indices_to_remove)}")
        print(f"Cleaned dataset size: {len(cleaned_df)}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cleaned_df.to_parquet(output_path + "/dedup.parquet", engine='pyarrow')
        print(f"Cleaned dataset saved to {output_path}/dedup.parquet")
        return cleaned_df

    def run(self, data_path: str, output_dir: str):
        """
        Runs the full deduplication pipeline.

        Args:
            data_path (str): Path to the parquet data file.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at: {data_path}")

        # Read the dataset
        try:
            parquet_file = pq.ParquetFile(data_path)
            df = parquet_file.read().to_pandas()
            print("Loaded dataset at {data_path} with {len(df)} records.")
        except Exception as e:
            raise IOError(f"Failed to read parquet file at {data_path}: {e}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Preprocess data
        combined_samples = self._preprocess_data(df)
        total_samples = len(combined_samples)

        # Load model and generate embeddings
        self._load_model()
        embeddings = self._generate_embeddings(samples=combined_samples, output_dir=output_dir)

        # Calculate similarity matrix, analyze scores, and find redundant samples
        similarity_matrix = self._calculate_similarity_matrix(embeddings=embeddings, output_dir=output_dir)
        self._analyze_scores(similarity_matrix, total_samples)
        redundant_indices = self._find_redundant_samples(similarity_matrix, total_samples)
        self._calculate_redundancy_score(redundant_indices, total_samples)

        # Run visualizations and showcase features
        self._plot_box_and_hist(similarity_matrix, output_dir)
        self._plot_tsne_clusters(embeddings, output_dir)
        self._showcase_pairs(df, similarity_matrix, output_dir)

        # Deduplicate and save the final dataset
        self._deduplicate_dataset(df, similarity_matrix, output_dir)
