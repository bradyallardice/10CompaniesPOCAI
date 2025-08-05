#!/usr/bin/env python3
"""
Step 4: Deduplicate AI-application strings → Embed with BGE-large → Match to O*NET tasks → Keep top n%

This step links each AI-application string (output from Step 3) to the most semantically similar
O*NET task statements, using high-quality embeddings (1024-D BGE-large-en-v1.5) and plain NumPy 
cosine similarity.

Phases:
A. Deduplicate AI-application strings  
B. Embed texts using BGE-large model
C. Similarity computation and top n% filter
D. Output formatting with validation checks
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from typing import List, Tuple, Optional
import hashlib
import re
from pathlib import Path
import argparse
import pickle

# Embedding and ML libraries
try:
    from sentence_transformers import SentenceTransformer
    import torch
except ImportError:
    print("ERROR: sentence-transformers and/or torch not installed.")
    print("Please install with: pip install sentence-transformers torch")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONETSimilarityMatcher:
    """
    Main class for matching AI applications to O*NET tasks via semantic similarity.
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 top_n_percent: float = 95.0,
                 similarity_threshold: float = 0.99,
                 batch_size: int = 256,
                 embeddings_dir: str = "Data/embeddings"):
        """
        Initialize the matcher.
        
        Args:
            model_name: Hugging Face model name for embeddings
            top_n_percent: Percentage of top similar tasks to keep (e.g., 10.0 for top 10%)
            similarity_threshold: Threshold for fuzzy deduplication clustering
            batch_size: Batch size for embedding computation
            embeddings_dir: Directory to store cached embeddings
        """
        self.model_name = model_name
        self.top_n_percent = top_n_percent
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cpu":
            self.batch_size = min(self.batch_size, 64)  # Reduce batch size for CPU
        
        self.model = self.model.to(self.device)
    
    def clear_embeddings_cache(self, cache_type: str = "all") -> None:
        """
        Clear cached embeddings.
        
        Args:
            cache_type: Type of cache to clear ('all', 'onet', 'apps')
        """
        if not os.path.exists(self.embeddings_dir):
            logger.info("No embeddings cache directory found")
            return
        
        cache_files = []
        
        if cache_type in ["all", "onet"]:
            cache_files.extend(Path(self.embeddings_dir).glob("onet_*.pkl"))
        
        if cache_type in ["all", "apps"]:
            cache_files.extend(Path(self.embeddings_dir).glob("apps_*.pkl"))
        
        if cache_type == "all":
            cache_files.extend(Path(self.embeddings_dir).glob("*.pkl"))
        
        removed_count = 0
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                removed_count += 1
                logger.info(f"Removed cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Cleared {removed_count} cache files")
    
    def list_cached_embeddings(self) -> dict:
        """
        List all cached embeddings with metadata.
        
        Returns:
            Dictionary with cache information
        """
        if not os.path.exists(self.embeddings_dir):
            return {}
        
        cache_info = {}
        cache_files = list(Path(self.embeddings_dir).glob("*.pkl"))
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                cache_info[str(cache_file)] = {
                    'model_name': cache_data.get('model_name', 'unknown'),
                    'created_at': cache_data.get('created_at', 'unknown'),
                    'num_texts': len(cache_data.get('texts', [])),
                    'embedding_shape': cache_data.get('embeddings', np.array([])).shape if 'embeddings' in cache_data else 'unknown'
                }
            except Exception as e:
                cache_info[str(cache_file)] = {'error': str(e)}
        
        return cache_info
    
    def _get_embedding_cache_path(self, texts_hash: str, prefix: str) -> str:
        """
        Get the cache file path for embeddings.
        
        Args:
            texts_hash: Hash of the texts being embedded
            prefix: Prefix for the cache file (e.g., 'onet', 'apps')
            
        Returns:
            Path to the cache file
        """
        model_safe = self.model_name.replace("/", "_")
        filename = f"{prefix}_{model_safe}_{texts_hash[:8]}.pkl"
        return os.path.join(self.embeddings_dir, filename)
    
    def _compute_texts_hash(self, texts: List[str]) -> str:
        """
        Compute a hash for a list of texts to use for caching.
        
        Args:
            texts: List of text strings
            
        Returns:
            Hash string
        """
        combined_text = "\n".join(sorted(texts))
        return hashlib.sha256(combined_text.encode('utf-8')).hexdigest()
    
    def _save_embeddings(self, embeddings: np.ndarray, texts: List[str], prefix: str) -> str:
        """
        Save embeddings to cache.
        
        Args:
            embeddings: Numpy array of embeddings
            texts: List of original texts
            prefix: Prefix for cache file
            
        Returns:
            Path to saved file
        """
        texts_hash = self._compute_texts_hash(texts)
        cache_path = self._get_embedding_cache_path(texts_hash, prefix)
        
        cache_data = {
            'embeddings': embeddings,
            'texts': texts,
            'model_name': self.model_name,
            'created_at': datetime.now().isoformat()
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        logger.info(f"Saved embeddings to cache: {cache_path}")
        return cache_path
    
    def _load_embeddings(self, texts: List[str], prefix: str) -> Optional[np.ndarray]:
        """
        Load embeddings from cache if available.
        
        Args:
            texts: List of texts to find embeddings for
            prefix: Prefix for cache file
            
        Returns:
            Embeddings array if found, None otherwise
        """
        texts_hash = self._compute_texts_hash(texts)
        cache_path = self._get_embedding_cache_path(texts_hash, prefix)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Validate cache
            if (cache_data['model_name'] != self.model_name or 
                cache_data['texts'] != texts):
                logger.warning(f"Cache validation failed for {cache_path}")
                return None
            
            logger.info(f"Loaded embeddings from cache: {cache_path}")
            return cache_data['embeddings']
            
        except Exception as e:
            logger.warning(f"Failed to load embeddings from cache {cache_path}: {e}")
            return None
        
    def canonicalize_text(self, text: str) -> str:
        """
        Canonicalize text for deduplication: lowercase, normalize quotes/whitespace, trim.
        
        Args:
            text: Input text string
            
        Returns:
            Canonicalized text string
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower().strip()
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def deduplicate_applications(self, df: pd.DataFrame, 
                               text_column: str = 'step3_output',
                               enable_fuzzy: bool = False) -> pd.DataFrame:
        """
        Phase A: Deduplicate AI-application strings.
        
        Args:
            df: Input dataframe with AI applications
            text_column: Column name containing the application text
            enable_fuzzy: Whether to enable fuzzy clustering deduplication
            
        Returns:
            Deduplicated dataframe
        """
        logger.info("Phase A: Deduplicating AI-application strings")
        
        # Canonicalize text
        df = df.copy()
        df['canonicalized'] = df[text_column].apply(self.canonicalize_text)
        
        # Remove empty strings
        df = df[df['canonicalized'] != ''].copy()
        
        initial_count = len(df)
        logger.info(f"Initial count: {initial_count}")
        
        # Exact hash deduplication
        df['text_hash'] = df['canonicalized'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        
        # Keep first occurrence of each hash, but maintain all job_uids
        dedup_exact = df.groupby('text_hash').agg({
            'job_uid': lambda x: list(x),  # Keep all job_uids
            'canonicalized': 'first',
            text_column: 'first'
        }).reset_index()
        
        exact_count = len(dedup_exact)
        logger.info(f"After exact deduplication: {exact_count} unique strings")
        
        # Expand job_uids back to individual rows for processing
        expanded_rows = []
        for _, row in dedup_exact.iterrows():
            for job_uid in row['job_uid']:
                expanded_rows.append({
                    'job_uid': job_uid,
                    'app_text': row['canonicalized'],
                    'original_text': row[text_column]
                })
        
        result_df = pd.DataFrame(expanded_rows)
        
        # Optional fuzzy clustering deduplication
        if enable_fuzzy and exact_count > 1:
            logger.info("Applying fuzzy clustering deduplication")
            result_df = self._fuzzy_deduplicate(result_df)
        
        final_count = len(result_df['app_text'].unique())
        logger.info(f"Final unique application strings: {final_count}")
        
        return result_df
    
    def _fuzzy_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional fuzzy clustering deduplication using the same embedding model.
        
        Args:
            df: Dataframe with deduplicated exact matches
            
        Returns:
            Dataframe with fuzzy deduplication applied
        """
        unique_texts = df['app_text'].unique()
        
        if len(unique_texts) <= 1:
            return df
        
        # Embed unique texts
        logger.info(f"Computing embeddings for {len(unique_texts)} unique texts")
        embeddings = self.model.encode(unique_texts, 
                                     batch_size=self.batch_size,
                                     normalize_embeddings=True,
                                     show_progress_bar=True)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(embeddings, embeddings.T)
        
        # Simple clustering: group texts above threshold
        clusters = {}
        cluster_id = 0
        assigned = set()
        
        for i, text in enumerate(unique_texts):
            if i in assigned:
                continue
                
            cluster_members = [i]
            assigned.add(i)
            
            for j in range(i + 1, len(unique_texts)):
                if j not in assigned and similarity_matrix[i, j] >= self.similarity_threshold:
                    cluster_members.append(j)
                    assigned.add(j)
            
            # Use the first text as the representative
            representative_text = unique_texts[cluster_members[0]]
            for member_idx in cluster_members:
                clusters[unique_texts[member_idx]] = representative_text
            
            cluster_id += 1
        
        # Apply clustering to dataframe
        df = df.copy()
        df['app_text'] = df['app_text'].map(lambda x: clusters.get(x, x))
        
        unique_after_fuzzy = len(df['app_text'].unique())
        logger.info(f"After fuzzy clustering: {unique_after_fuzzy} unique strings")
        
        return df
    
    def load_onet_tasks(self, onet_file_path: str) -> pd.DataFrame:
        """
        Load O*NET task statements from Excel file.
        
        Args:
            onet_file_path: Path to O*NET Task Statements.xlsx file
            
        Returns:
            DataFrame with task_id and Task columns
        """
        logger.info(f"Loading O*NET tasks from: {onet_file_path}")
        
        try:
            # Try to read the Excel file
            onet_df = pd.read_excel(onet_file_path)
            
            # Check for expected columns
            if 'Task' not in onet_df.columns:
                raise ValueError("Expected 'Task' column not found in O*NET file")
            
            # Create task_id as row index
            onet_df = onet_df.reset_index()
            onet_df['task_id'] = onet_df.index
            
            # Clean and validate tasks
            onet_df = onet_df[onet_df['Task'].notna()].copy()
            onet_df['Task'] = onet_df['Task'].astype(str).str.strip()
            onet_df = onet_df[onet_df['Task'] != ''].copy()
            
            logger.info(f"Loaded {len(onet_df)} O*NET task statements")
            return onet_df[['task_id', 'Task']]
            
        except Exception as e:
            logger.error(f"Error loading O*NET file: {e}")
            raise
    
    def embed_texts(self, texts: List[str], description: str = "texts", 
                   cache_prefix: str = "texts", use_cache: bool = True) -> np.ndarray:
        """
        Phase B: Embed texts using BGE-large model with caching support.
        
        Args:
            texts: List of text strings to embed
            description: Description for logging
            cache_prefix: Prefix for cache files
            use_cache: Whether to use cached embeddings
            
        Returns:
            Normalized embeddings array
        """
        logger.info(f"Phase B: Embedding {len(texts)} {description}")
        
        # Try to load from cache first
        if use_cache:
            cached_embeddings = self._load_embeddings(texts, cache_prefix)
            if cached_embeddings is not None:
                logger.info(f"Using cached embeddings for {description}")
                return cached_embeddings
        
        # Compute embeddings if not cached
        logger.info(f"Computing new embeddings for {description}")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # L2 normalize
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Save to cache
        if use_cache:
            self._save_embeddings(embeddings, texts, cache_prefix)
        
        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings
    
    def compute_similarities_and_filter(self, 
                                      apps_df: pd.DataFrame,
                                      apps_embeddings: np.ndarray,
                                      onet_df: pd.DataFrame,
                                      onet_embeddings: np.ndarray,
                                      save_all_similarities: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Phase C: Compute similarities and apply top n% filter.
        
        Args:
            apps_df: DataFrame with application strings
            apps_embeddings: Embeddings for application strings
            onet_df: DataFrame with O*NET tasks
            onet_embeddings: Embeddings for O*NET tasks
            save_all_similarities: Whether to also compute and return all similarities
            
        Returns:
            Tuple of (filtered_results_df, all_similarities_df)
        """
        logger.info("Phase C: Computing similarities and applying top n% filter")
        
        # Get unique application texts and their embeddings
        unique_apps = apps_df['app_text'].unique()
        
        filtered_results = []
        all_results = [] if save_all_similarities else None
        
        logger.info(f"Processing {len(unique_apps)} unique application strings")
        
        for i, app_text in enumerate(unique_apps):
            if i % 100 == 0:
                logger.info(f"Processing application {i+1}/{len(unique_apps)}")
            
            # Compute cosine similarities to all O*NET tasks
            app_embedding = apps_embeddings[i:i+1]  # Shape: (1, embedding_dim)
            similarities = np.dot(app_embedding, onet_embeddings.T)[0]  # Shape: (n_onet_tasks,)
            
            # Find all job_uids that have this application text
            matching_jobs = apps_df[apps_df['app_text'] == app_text]['job_uid'].tolist()
            
            # Save all similarities if requested
            if save_all_similarities:
                for onet_idx, similarity in enumerate(similarities):
                    for job_uid in matching_jobs:
                        all_results.append({
                            'job_uid': job_uid,
                            'app_text': app_text,
                            'onet_task_id': onet_df.iloc[onet_idx]['task_id'],
                            'onet_task': onet_df.iloc[onet_idx]['Task'],
                            'similarity': float(similarity)
                        })
            
            # Apply percentile cutoff for filtered results
            cutoff_percentile = 100 - self.top_n_percent
            cutoff_score = np.percentile(similarities, cutoff_percentile)
            
            # Always retain at least the single highest-similarity task
            max_score = np.max(similarities)
            cutoff_score = min(cutoff_score, max_score)
            
            # Get indices of tasks above cutoff
            above_cutoff = similarities >= cutoff_score
            selected_indices = np.where(above_cutoff)[0]
            
            # Create filtered results for this application
            for onet_idx in selected_indices:
                for job_uid in matching_jobs:
                    filtered_results.append({
                        'job_uid': job_uid,
                        'app_text': app_text,
                        'onet_task_id': onet_df.iloc[onet_idx]['task_id'],
                        'onet_task': onet_df.iloc[onet_idx]['Task'],
                        'similarity': float(similarities[onet_idx])
                    })
        
        filtered_results_df = pd.DataFrame(filtered_results)
        all_results_df = pd.DataFrame(all_results) if save_all_similarities else None
        
        logger.info(f"Generated {len(filtered_results_df)} filtered similarity matches")
        if save_all_similarities:
            logger.info(f"Generated {len(all_results_df)} total similarity matches")
        
        return filtered_results_df, all_results_df
    
    def validate_results(self, results_df: pd.DataFrame) -> dict:
        """
        Phase D: Validation and quality checks.
        
        Args:
            results_df: Results dataframe to validate
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Phase D: Running validation and quality checks")
        
        # Check coverage - % of apps with >= 1 retained task
        unique_apps = results_df['app_text'].nunique()
        total_apps_with_matches = results_df.groupby('app_text').size().count()
        coverage = (total_apps_with_matches / unique_apps * 100) if unique_apps > 0 else 0
        
        # Histogram of max similarity per app
        max_similarities = results_df.groupby('app_text')['similarity'].max()
        
        # Statistics
        validation_metrics = {
            'total_unique_applications': unique_apps,
            'total_matches': len(results_df),
            'coverage_percent': coverage,
            'avg_matches_per_app': len(results_df) / unique_apps if unique_apps > 0 else 0,
            'similarity_stats': {
                'min': float(results_df['similarity'].min()),
                'max': float(results_df['similarity'].max()),
                'mean': float(results_df['similarity'].mean()),
                'median': float(results_df['similarity'].median()),
                'std': float(results_df['similarity'].std())
            },
            'max_similarity_stats': {
                'min': float(max_similarities.min()),
                'max': float(max_similarities.max()),
                'mean': float(max_similarities.mean()),
                'median': float(max_similarities.median())
            }
        }
        
        # Log key metrics
        logger.info(f"Coverage: {coverage:.1f}% of applications have matches")
        logger.info(f"Average matches per application: {validation_metrics['avg_matches_per_app']:.1f}")
        logger.info(f"Similarity range: {validation_metrics['similarity_stats']['min']:.3f} - {validation_metrics['similarity_stats']['max']:.3f}")
        logger.info(f"Mean similarity: {validation_metrics['similarity_stats']['mean']:.3f}")
        
        return validation_metrics
    
    def run_full_pipeline(self, 
                         step3_file_path: str,
                         onet_file_path: str,
                         output_dir: str = "Data",
                         enable_fuzzy_dedup: bool = False,
                         use_onet_cache: bool = True,
                         use_apps_cache: bool = True,
                         save_all_similarities: bool = False) -> Tuple[pd.DataFrame, dict]:
        """
        Run the complete pipeline from Step 3 output to O*NET similarity matches.
        
        Args:
            step3_file_path: Path to Step 3 filtered tasks CSV
            onet_file_path: Path to O*NET Task Statements Excel file
            output_dir: Directory to save output files
            enable_fuzzy_dedup: Whether to enable fuzzy deduplication
            use_onet_cache: Whether to use cached O*NET embeddings
            use_apps_cache: Whether to use cached application embeddings
            save_all_similarities: Whether to save all similarity scores (creates large file)
            
        Returns:
            Tuple of (results_dataframe, validation_metrics)
        """
        logger.info("Starting Step 4: O*NET Similarity Matching Pipeline")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Top N%: {self.top_n_percent}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"O*NET cache: {'enabled' if use_onet_cache else 'disabled'}")
        logger.info(f"Apps cache: {'enabled' if use_apps_cache else 'disabled'}")
        
        # Load Step 3 output
        logger.info(f"Loading Step 3 output from: {step3_file_path}")
        step3_df = pd.read_csv(step3_file_path)
        logger.info(f"Loaded {len(step3_df)} records from Step 3")
        
        # Phase A: Deduplicate applications
        dedup_df = self.deduplicate_applications(step3_df, enable_fuzzy=enable_fuzzy_dedup)
        
        # Save deduplicated applications
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dedup_file = os.path.join(output_dir, f"dedup_apps_{timestamp}.parquet")
        dedup_df.to_parquet(dedup_file, index=False)
        logger.info(f"Saved deduplicated applications to: {dedup_file}")
        
        # Load O*NET tasks
        onet_df = self.load_onet_tasks(onet_file_path)
        
        # Phase B: Embed texts
        unique_app_texts = dedup_df['app_text'].unique().tolist()
        apps_embeddings = self.embed_texts(
            unique_app_texts, 
            "application strings", 
            cache_prefix="apps", 
            use_cache=use_apps_cache
        )
        
        onet_tasks = onet_df['Task'].tolist()
        onet_embeddings = self.embed_texts(
            onet_tasks, 
            "O*NET tasks", 
            cache_prefix="onet", 
            use_cache=use_onet_cache
        )
        
        # Phase C: Compute similarities and filter
        results_df, all_similarities_df = self.compute_similarities_and_filter(
            dedup_df, apps_embeddings, onet_df, onet_embeddings, save_all_similarities=save_all_similarities
        )
        
        # Phase D: Validate results
        validation_metrics = self.validate_results(results_df)
        
        # Create deduplicated similarity scores and job mapping
        logger.info("Creating deduplicated similarity scores and job mapping")
        
        # Create deduplicated similarity scores (unique app_text + onet_task combinations)
        deduplicated_similarities = results_df.drop_duplicates(
            subset=['app_text', 'onet_task_id']
        )[['app_text', 'onet_task_id', 'onet_task', 'similarity']].copy()
        
        # Create job mapping (app_text -> list of job_uids)
        job_mapping = results_df.groupby('app_text')['job_uid'].apply(list).reset_index()
        job_mapping['job_uids'] = job_mapping['job_uid'].apply(lambda x: '|'.join(sorted(set(x))))
        job_mapping = job_mapping[['app_text', 'job_uids']].copy()
        
        logger.info(f"Deduplicated to {len(deduplicated_similarities)} unique app-task pairs")
        logger.info(f"Created job mapping for {len(job_mapping)} unique applications")
        
        # Save final results in multiple formats
        # 1. Parquet format (for programmatic use) - original format with duplicates
        output_file_parquet = os.path.join(output_dir, f"ai_app_onet_top{int(self.top_n_percent)}_{timestamp}.parquet")
        results_df.to_parquet(output_file_parquet, index=False)
        logger.info(f"Saved final results (parquet) to: {output_file_parquet}")
        
        # 2. CSV format for top n% matches - DEDUPLICATED (for manual review)
        output_file_csv = os.path.join(output_dir, f"top_{int(self.top_n_percent)}_matches.csv")
        deduplicated_similarities.to_csv(output_file_csv, index=False)
        logger.info(f"Saved deduplicated top {int(self.top_n_percent)}% matches (CSV) to: {output_file_csv}")
        
        # 3. Job mapping file (to link back to original job ads)
        job_mapping_file = os.path.join(output_dir, "job_app_mapping.csv")
        job_mapping.to_csv(job_mapping_file, index=False)
        logger.info(f"Saved job-application mapping to: {job_mapping_file}")
        
        # 4. CSV format for ALL similarity scores (comprehensive analysis)
        if all_similarities_df is not None:
            # Create deduplicated version of all similarities too
            all_deduplicated = all_similarities_df.drop_duplicates(
                subset=['app_text', 'onet_task_id']
            )[['app_text', 'onet_task_id', 'onet_task', 'similarity']].copy()
            
            all_similarities_file = os.path.join(output_dir, f"all_similarity_scores_{timestamp}.csv")
            all_deduplicated.to_csv(all_similarities_file, index=False)
            logger.info(f"Saved deduplicated all similarity scores (CSV) to: {all_similarities_file}")
        
        # 5. Save validation metrics
        metrics_file = os.path.join(output_dir, f"validation_metrics_{timestamp}.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(validation_metrics, f, indent=2)
        logger.info(f"Saved validation metrics to: {metrics_file}")
        
        logger.info("Step 4 pipeline completed successfully!")
        
        return results_df, validation_metrics


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Step 4: AI Applications to O*NET Task Similarity Matching")
    
    parser.add_argument("--top-n-percent", type=float, default=10.0,
                       help="Percentage of top similar tasks to keep (default: 10.0)")
    
    parser.add_argument("--enable-fuzzy-dedup", action="store_true",
                       help="Enable fuzzy deduplication clustering")
    
    parser.add_argument("--no-onet-cache", action="store_true",
                       help="Disable O*NET embeddings cache (force re-embedding)")
    
    parser.add_argument("--no-apps-cache", action="store_true", 
                       help="Disable applications embeddings cache (force re-embedding)")
    
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for embedding computation (auto-detect if not specified)")
    
    parser.add_argument("--model", type=str, default="BAAI/bge-large-en-v1.5",
                       help="Embedding model name (default: BAAI/bge-large-en-v1.5)")
    
    parser.add_argument("--step3-file", type=str, default=None,
                       help="Path to Step 3 output file (auto-detect latest if not specified)")
    
    parser.add_argument("--onet-file", type=str, default="Data/Task Statements.xlsx",
                       help="Path to O*NET Task Statements file")
    
    parser.add_argument("--output-dir", type=str, default="Data",
                       help="Output directory for results")
    
    parser.add_argument("--embeddings-dir", type=str, default="Data/embeddings",
                       help="Directory for cached embeddings")
    
    parser.add_argument("--clear-cache", type=str, choices=["all", "onet", "apps"], default=None,
                       help="Clear cached embeddings before running")
    
    parser.add_argument("--list-cache", action="store_true",
                       help="List cached embeddings and exit")
    
    parser.add_argument("--save-all-similarities", action="store_true",
                       help="Save all similarity scores (creates large CSV file)")
    
    return parser.parse_args()


def main():
    """
    Main execution function for Step 4.
    """
    args = parse_arguments()
    
    # Configuration from arguments
    TOP_N_PERCENT = args.top_n_percent
    ENABLE_FUZZY_DEDUP = args.enable_fuzzy_dedup
    USE_ONET_CACHE = not args.no_onet_cache
    USE_APPS_CACHE = not args.no_apps_cache
    
    # File paths
    if args.step3_file:
        step3_file = args.step3_file
    else:
        # Find the most recent Step 3 output file
        step3_files = list(Path(args.output_dir).glob("step3_filtered_tasks_train_*.csv"))
        
        if not step3_files:
            logger.error("No Step 3 output files found matching pattern: step3_filtered_tasks_train_*.csv")
            logger.error("Please ensure Step 3 has been run successfully or specify --step3-file")
            return
        
        # Use the most recent file
        step3_file = str(sorted(step3_files)[-1])
    
    logger.info(f"Using Step 3 file: {step3_file}")
    onet_file = args.onet_file
    
    # Validate input files exist
    if not os.path.exists(step3_file):
        logger.error(f"Step 3 file not found: {step3_file}")
        return
    
    if not os.path.exists(onet_file):
        logger.error(f"O*NET file not found: {onet_file}")
        logger.error("Please ensure 'Task Statements.xlsx' is in the Data directory")
        return
    
    # Initialize matcher
    try:
        batch_size = args.batch_size
        if batch_size is None:
            batch_size = 256 if torch.cuda.is_available() else 64
            
        matcher = ONETSimilarityMatcher(
            model_name=args.model,
            top_n_percent=TOP_N_PERCENT,
            batch_size=batch_size,
            embeddings_dir=args.embeddings_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize matcher: {e}")
        return
    
    # Handle cache operations
    if args.list_cache:
        cache_info = matcher.list_cached_embeddings()
        if not cache_info:
            print("No cached embeddings found")
        else:
            print("\nCached Embeddings:")
            print("=" * 60)
            for file_path, info in cache_info.items():
                print(f"File: {os.path.basename(file_path)}")
                if 'error' in info:
                    print(f"  Error: {info['error']}")
                else:
                    print(f"  Model: {info['model_name']}")
                    print(f"  Created: {info['created_at']}")
                    print(f"  Texts: {info['num_texts']}")
                    print(f"  Shape: {info['embedding_shape']}")
                print()
        return
    
    if args.clear_cache:
        matcher.clear_embeddings_cache(args.clear_cache)
        if not (USE_ONET_CACHE or USE_APPS_CACHE):
            logger.info("Cache cleared. Exiting since caching is disabled.")
            return
    
    # Run pipeline
    try:
        results_df, validation_metrics = matcher.run_full_pipeline(
            step3_file_path=step3_file,
            onet_file_path=onet_file,
            output_dir=args.output_dir,
            enable_fuzzy_dedup=ENABLE_FUZZY_DEDUP,
            use_onet_cache=USE_ONET_CACHE,
            use_apps_cache=USE_APPS_CACHE,
            save_all_similarities=args.save_all_similarities
        )
        
        # Print summary
        print("\n" + "="*60)
        print("STEP 4 SUMMARY")
        print("="*60)
        print(f"Processed {validation_metrics['total_unique_applications']} unique AI applications")
        print(f"Generated {validation_metrics['total_matches']} similarity matches")
        print(f"Coverage: {validation_metrics['coverage_percent']:.1f}% of applications matched")
        print(f"Average matches per application: {validation_metrics['avg_matches_per_app']:.1f}")
        print(f"Similarity range: {validation_metrics['similarity_stats']['min']:.3f} - {validation_metrics['similarity_stats']['max']:.3f}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()