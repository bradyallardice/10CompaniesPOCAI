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
from typing import List, Tuple, Optional, Dict
import hashlib
import re
from pathlib import Path
import argparse
import pickle
import gc

# Embedding and ML libraries
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    from tqdm import tqdm
except ImportError:
    print("ERROR: sentence-transformers and/or torch not installed.")
    print("Please install with: pip install sentence-transformers torch tqdm")
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
                 minimum_similarity: float = 0.3,
                 similarity_threshold: float = 0.99,
                 batch_size: int = 256,
                 embeddings_dir: str = "Data/embeddings"):
        """
        Initialize the matcher.

        Args:
            model_name: Hugging Face model name for embeddings
            minimum_similarity: Minimum BGE similarity threshold for keeping pairs (default: 0.3)
            similarity_threshold: Threshold for fuzzy deduplication clustering
            batch_size: Batch size for embedding computation
            embeddings_dir: Directory to store cached embeddings
        """
        self.model_name = model_name
        self.minimum_similarity = minimum_similarity
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Initialize model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Check if GPU is available (supports CUDA, MPS, and CPU)
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Metal Performance Shaders (Apple Silicon GPU) detected")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA GPU detected")
        else:
            self.device = "cpu"
            logger.info("No GPU detected, using CPU")

        logger.info(f"Using device: {self.device}")

        # Adjust batch size based on device
        if self.device == "cpu":
            self.batch_size = min(self.batch_size, 64)  # Reduce batch size for CPU
        elif self.device == "mps":
            # MPS can handle similar batch sizes to CUDA, but may need tuning
            self.batch_size = min(self.batch_size, 128)

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

    def _get_ce_cache_path(self, app_texts: List[str], onet_task_ids: List[str],
                           model_name: str) -> Path:
        """
        Generate cache file path for cross-encoder scores.

        Args:
            app_texts: List of AI application texts
            onet_task_ids: List of O*NET task IDs
            model_name: Cross-encoder model name

        Returns:
            Path to cache file
        """
        # Create deterministic hash from pair set + model
        pair_strings = [f"{app}|{task}" for app, task in zip(app_texts, onet_task_ids)]
        content_hash = hashlib.md5(
            (model_name + "|" + "|".join(sorted(pair_strings))).encode()
        ).hexdigest()[:8]

        # Clean model name for filename
        safe_model_name = model_name.replace("/", "_").replace("-", "_")
        cache_filename = f"ce_scores_{safe_model_name}_{content_hash}.pkl"

        return Path(self.embeddings_dir) / cache_filename

    def _load_ce_cache(self, cache_path: Path) -> Optional[dict]:
        """
        Load cross-encoder scores from cache.

        Returns:
            Dictionary mapping (app_text, onet_task_id) -> ce_score, or None if invalid
        """
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict):
                logger.warning(f"Invalid cache format at {cache_path}")
                return None

            if 'scores' not in cache_data or 'model_name' not in cache_data:
                logger.warning(f"Missing required fields in cache at {cache_path}")
                return None

            logger.info(f"Loaded {len(cache_data['scores'])} cached cross-encoder scores")
            logger.info(f"Cache created: {cache_data.get('created_at', 'unknown')}")

            return cache_data['scores']

        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None

    def _save_ce_cache(self, cache_path: Path, scores_dict: dict,
                       model_name: str) -> None:
        """
        Save cross-encoder scores to cache.

        Args:
            cache_path: Path to save cache file
            scores_dict: Dictionary mapping (app_text, onet_task_id) -> ce_score
            model_name: Cross-encoder model name
        """
        try:
            cache_data = {
                'scores': scores_dict,
                'model_name': model_name,
                'created_at': datetime.now().isoformat(),
                'num_pairs': len(scores_dict)
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"Saved {len(scores_dict)} cross-encoder scores to cache")
            logger.info(f"Cache location: {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")

    def _get_similarity_cache_path(self, app_texts: List[str], onet_tasks: List[str],
                                   minimum_similarity: float) -> Path:
        """
        Generate cache file path for similarity computation results.

        Args:
            app_texts: List of AI application texts
            onet_tasks: List of O*NET task texts
            minimum_similarity: Minimum similarity threshold used

        Returns:
            Path to cache file
        """
        # Create stable deterministic hash from counts + threshold (not full text lists)
        # This ensures the same datasets produce the same cache key across runs
        content_hash = hashlib.md5(
            (str(len(app_texts)) + "|" + str(len(onet_tasks)) + "|" + str(minimum_similarity)).encode()
        ).hexdigest()[:8]

        cache_filename = f"similarities_apps{len(app_texts)}_tasks{len(onet_tasks)}_min{minimum_similarity:.1f}_{content_hash}.pkl"
        return Path(self.embeddings_dir) / cache_filename

    def _load_similarity_cache(self, cache_path: Path) -> Optional[Tuple[pd.DataFrame, Optional[pd.DataFrame]]]:
        """
        Load similarity computation results from cache.

        Returns:
            Tuple of (filtered_results_df, all_results_df), or None if cache invalid
        """
        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Validate cache structure
            if not isinstance(cache_data, dict):
                logger.warning(f"Invalid cache format at {cache_path}")
                return None

            if 'filtered_results' not in cache_data or 'top_n_percent' not in cache_data:
                logger.warning(f"Missing required fields in cache at {cache_path}")
                return None

            logger.info(f"Loaded cached similarity computation results")
            logger.info(f"Cache created: {cache_data.get('created_at', 'unknown')}")
            logger.info(f"Filtered results: {len(cache_data['filtered_results'])} rows")
            if cache_data['all_results'] is not None:
                logger.info(f"All results: {len(cache_data['all_results'])} rows")

            # Restore global percentiles if present in cache
            if 'global_percentile_thresholds' in cache_data:
                self.global_percentile_thresholds = cache_data['global_percentile_thresholds']
                logger.info("Restored global percentile thresholds from cache")

            return (cache_data['filtered_results'], cache_data['all_results'])

        except Exception as e:
            logger.warning(f"Failed to load similarity cache from {cache_path}: {e}")
            return None

    def _save_similarity_cache(self, cache_path: Path, filtered_results_df: pd.DataFrame,
                               all_results_df: Optional[pd.DataFrame], top_n_percent: float) -> None:
        """
        Save similarity computation results to cache.

        Args:
            cache_path: Path to save cache file
            filtered_results_df: Filtered similarity results
            all_results_df: Optional all similarity results
            top_n_percent: Top N percent threshold used
        """
        try:
            cache_data = {
                'filtered_results': filtered_results_df,
                'all_results': all_results_df,
                'top_n_percent': top_n_percent,
                'created_at': datetime.now().isoformat()
            }

            # Save global percentiles if available
            if hasattr(self, 'global_percentile_thresholds') and self.global_percentile_thresholds is not None:
                cache_data['global_percentile_thresholds'] = self.global_percentile_thresholds

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"Saved similarity computation results to cache")
            logger.info(f"Cache location: {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save similarity cache to {cache_path}: {e}")

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
        
        # Handle column name compatibility - map original_job_uid to job_uid if needed
        if 'original_job_uid' in df.columns and 'job_uid' not in df.columns:
            df['job_uid'] = df['original_job_uid']
        
        df['canonicalized'] = df[text_column].apply(self.canonicalize_text)
        
        # Remove empty strings
        df = df[df['canonicalized'] != ''].copy()
        
        initial_count = len(df)
        logger.info(f"Initial count: {initial_count}")
        
        # Sort by tst_created (earliest first) to ensure temporal ordering
        if 'tst_created' in df.columns:
            df = df.sort_values('tst_created', ascending=True)
            logger.info("Sorted by tst_created (earliest first) for temporal deduplication")
        else:
            logger.warning("tst_created column not found - temporal ordering not applied")
        
        # Exact hash deduplication
        df['text_hash'] = df['canonicalized'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())
        
        # Keep first occurrence of each hash (which is earliest due to sorting), but maintain all job_uids
        dedup_exact = df.groupby('text_hash').agg({
            'job_uid': lambda x: list(x),  # Keep all job_uids
            'tst_created': 'first',  # Keep earliest timestamp
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
                    'original_text': row[text_column],
                    'first_occurrence_tst_created': row.get('tst_created', None)
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
    
    def load_onet_tasks(self, onet_file_path: str, exclude_soc_15: bool = False) -> pd.DataFrame:
        """
        Load O*NET task statements from Excel file.
        
        Args:
            onet_file_path: Path to O*NET Task Statements.xlsx file
            exclude_soc_15: If True, exclude SOC group 15 (Computer and Mathematical Occupations)
            
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
            
            # Filter to Core tasks only (exclude Supplemental tasks)
            if 'Task Type' in onet_df.columns:
                initial_count = len(onet_df)
                onet_df = onet_df[onet_df['Task Type'] == 'Core'].copy()
                filtered_count = initial_count - len(onet_df)
                logger.info(f"Filtered to {len(onet_df)} Core tasks (excluded {filtered_count} Supplemental tasks)")
            
            # Filter out SOC group 15 if requested
            if exclude_soc_15:
                initial_count = len(onet_df)
                onet_df = onet_df[~onet_df['O*NET-SOC Code'].str.startswith('15-', na=False)].copy()
                excluded_count = initial_count - len(onet_df)
                logger.info(f"Excluded {excluded_count} tasks from SOC group 15 (Computer and Mathematical Occupations)")
            
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
                                      save_all_similarities: bool = False,
                                      use_cache: bool = True,
                                      bge_percentiles: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Phase C: Compute similarities and apply global minimum threshold with optional caching.

        Args:
            apps_df: DataFrame with application strings
            apps_embeddings: Embeddings for application strings
            onet_df: DataFrame with O*NET tasks
            onet_embeddings: Embeddings for O*NET tasks
            save_all_similarities: Whether to also compute and return all similarities
            use_cache: Whether to use cached similarity results (default: True)
            bge_percentiles: List of percentile thresholds for later use (default: [20, 15, 10, 5, 1])

        Returns:
            Tuple of (filtered_results_df, all_similarities_df)
        """
        logger.info("Phase C: Computing similarities and applying global minimum threshold")
        logger.info(f"Minimum similarity threshold: {self.minimum_similarity}")

        # Set default percentiles if not provided
        if bge_percentiles is None:
            bge_percentiles = [20, 15, 10, 5, 1]

        # Check cache first
        app_texts = apps_df['app_text'].unique().tolist()
        onet_tasks = onet_df['Task'].tolist()
        cache_path = self._get_similarity_cache_path(app_texts, onet_tasks, self.minimum_similarity)

        if use_cache:
            cache_result = self._load_similarity_cache(cache_path)
            if cache_result is not None:
                logger.info("Using cached similarity computation results")
                return cache_result

        # Get unique application texts and their embeddings
        unique_apps = apps_df['app_text'].unique()

        # FIX 1: Pre-build job_uid lookup dictionary ONCE to avoid repeated DataFrame scans
        logger.info("Building job_uid lookup dictionary...")
        job_uid_lookup = {}
        for _, row in apps_df.iterrows():
            app_text = row['app_text']
            job_uid = row['job_uid']
            timestamp = row['first_occurrence_tst_created']

            if app_text not in job_uid_lookup:
                job_uid_lookup[app_text] = {}
            job_uid_lookup[app_text][job_uid] = timestamp

        logger.info(f"Built lookup for {len(job_uid_lookup)} unique apps")

        # FIX 2: Chunked processing to prevent unbounded memory accumulation
        CHUNK_SIZE = 1000  # Process 1000 apps at a time

        # Setup checkpointing directory
        checkpoint_dir = Path(self.embeddings_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint identifier based on cache path
        cache_filename = f"similarities_apps{len(app_texts)}_tasks{len(onet_tasks)}_min{self.minimum_similarity:.1f}"
        checkpoint_prefix = checkpoint_dir / cache_filename

        filtered_results_df = pd.DataFrame()
        all_results_df = pd.DataFrame() if save_all_similarities else None

        logger.info(f"Processing {len(unique_apps)} unique application strings in chunks of {CHUNK_SIZE}")
        logger.info(f"Using checkpoints at: {checkpoint_dir}")

        # Process in chunks
        for chunk_start in range(0, len(unique_apps), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(unique_apps))
            chunk_apps = unique_apps[chunk_start:chunk_end]

            # Check if chunk already exists (for resuming from checkpoints)
            chunk_checkpoint_filtered = checkpoint_prefix.parent / f"{checkpoint_prefix.name}_chunk_{chunk_start}_{chunk_end}_filtered.parquet"
            chunk_checkpoint_all = checkpoint_prefix.parent / f"{checkpoint_prefix.name}_chunk_{chunk_start}_{chunk_end}_all.parquet"

            if chunk_checkpoint_filtered.exists():
                logger.info(f"Loading checkpoint for chunk {chunk_start}-{chunk_end}")
                chunk_filtered_df = pd.read_parquet(chunk_checkpoint_filtered)
                filtered_results_df = pd.concat([filtered_results_df, chunk_filtered_df], ignore_index=True)

                if save_all_similarities and chunk_checkpoint_all.exists():
                    chunk_all_df = pd.read_parquet(chunk_checkpoint_all)
                    all_results_df = pd.concat([all_results_df, chunk_all_df], ignore_index=True)

                logger.info(f"Skipping processing for chunk {chunk_start}-{chunk_end} (already checkpointed)")
                continue

            logger.info(f"Processing chunk {chunk_start}-{chunk_end} of {len(unique_apps)} apps")

            # Accumulate results for this chunk only
            chunk_filtered_results = []
            chunk_all_results = [] if save_all_similarities else None

            for local_idx, app_text in enumerate(chunk_apps):
                global_idx = chunk_start + local_idx

                # Compute cosine similarities to all O*NET tasks
                app_embedding = apps_embeddings[global_idx:global_idx+1]  # Shape: (1, embedding_dim)
                similarities = np.dot(app_embedding, onet_embeddings.T)[0]  # Shape: (n_onet_tasks,)

                # Get job_uids and timestamps from pre-built lookup (O(1) dict access, not DataFrame scan)
                job_uid_to_timestamp = job_uid_lookup.get(app_text, {})
                matching_jobs = list(job_uid_to_timestamp.keys())

                # Save all similarities if requested
                if save_all_similarities:
                    for onet_idx, similarity in enumerate(similarities):
                        for job_uid in matching_jobs:
                            chunk_all_results.append({
                                'job_uid': job_uid,
                                'app_text': app_text,
                                'onet_task_id': onet_df.iloc[onet_idx]['task_id'],
                                'onet_task': onet_df.iloc[onet_idx]['Task'],
                                'similarity': float(similarity)
                            })

                # Apply global minimum threshold
                above_threshold = similarities >= self.minimum_similarity
                selected_indices = np.where(above_threshold)[0]

                # Create filtered results for this application
                for onet_idx in selected_indices:
                    for job_uid in matching_jobs:
                        # Get timestamp from cached lookup
                        timestamp = job_uid_to_timestamp[job_uid]

                        chunk_filtered_results.append({
                            'job_uid': job_uid,
                            'app_text': app_text,
                            'onet_task_id': onet_df.iloc[onet_idx]['task_id'],
                            'onet_task': onet_df.iloc[onet_idx]['Task'],
                            'similarity': float(similarities[onet_idx]),
                            'first_occurrence_tst_created': timestamp
                        })

                if (global_idx + 1) % 100 == 0:
                    logger.info(f"Processing application {global_idx + 1}/{len(unique_apps)}")

            # Convert chunk to DataFrame and append to main results
            if chunk_filtered_results:
                chunk_filtered_df = pd.DataFrame(chunk_filtered_results)
                filtered_results_df = pd.concat([filtered_results_df, chunk_filtered_df], ignore_index=True)

                # Save checkpoint for filtered results
                chunk_filtered_df.to_parquet(chunk_checkpoint_filtered)
                logger.info(f"Saved filtered checkpoint: {chunk_checkpoint_filtered.name}")
                del chunk_filtered_df

            if save_all_similarities and chunk_all_results:
                chunk_all_df = pd.DataFrame(chunk_all_results)
                all_results_df = pd.concat([all_results_df, chunk_all_df], ignore_index=True)

                # Save checkpoint for all results
                chunk_all_df.to_parquet(chunk_checkpoint_all)
                logger.info(f"Saved all-similarities checkpoint: {chunk_checkpoint_all.name}")
                del chunk_all_df

            # Explicit memory cleanup
            del chunk_filtered_results, chunk_all_results

            import gc
            gc.collect()

            logger.info(f"Chunk complete. Current memory: {filtered_results_df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

        logger.info(f"Generated {len(filtered_results_df)} filtered similarity matches (above threshold {self.minimum_similarity})")
        if save_all_similarities:
            logger.info(f"Generated {len(all_results_df)} total similarity matches")

        # Calculate global percentile threshold on ALL similarity scores
        # This must be done BEFORE the 0.3 filter to get true top 20% of all pairs
        if save_all_similarities and all_results_df is not None:
            logger.info("Computing global percentile on ALL similarity scores")
            all_similarities_array = all_results_df['similarity'].values

            # Store percentile thresholds for later use by Phase E
            self.global_percentile_thresholds = {}
            for percentile in sorted(bge_percentiles):  # Use the passed percentiles, not hardcoded ones
                threshold = np.percentile(all_similarities_array, 100 - percentile)
                self.global_percentile_thresholds[percentile] = threshold
                logger.info(f"  Global {percentile}% percentile (top {percentile}%): {threshold:.4f}")

            logger.info(f"Global percentiles calculated from {len(all_similarities_array):,} total pairs")
        else:
            logger.warning("Cannot compute global percentiles: save_all_similarities=False")
            logger.warning("Will fall back to percentiles calculated on filtered subset (may be inaccurate)")
            self.global_percentile_thresholds = None

        # Clean up chunk checkpoints after successful completion
        logger.info("Cleaning up chunk checkpoints...")
        for checkpoint_file in checkpoint_dir.glob(f"{checkpoint_prefix.name}_chunk_*.parquet"):
            checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_file.name}")

        # Save to cache
        if use_cache:
            self._save_similarity_cache(cache_path, filtered_results_df, all_results_df, self.minimum_similarity)

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
    
    def validate_with_cross_encoder(self,
                                  similarity_df: pd.DataFrame,
                                  cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                                  threshold: float = 0.6,
                                  batch_size: int = 256,
                                  use_cache: bool = True) -> Tuple[pd.DataFrame, dict]:
        """
        Phase E: Validate top similarity matches using cross-encoder with optional caching.

        Args:
            similarity_df: DataFrame with app_text, onet_task, and similarity columns
            cross_encoder_model: Cross-encoder model name
            threshold: Cross-encoder score threshold for keeping matches
            batch_size: Batch size for cross-encoder inference
            use_cache: Whether to use cached cross-encoder scores (default: True)

        Returns:
            Tuple of (validated_df, validation_report)
        """
        logger.info(f"Phase E: Cross-encoder validation with {cross_encoder_model}")
        logger.info(f"Validating {len(similarity_df)} similarity matches")
        logger.info(f"Threshold: {threshold}, Batch size: {batch_size}")

        # Extract pairs and IDs
        app_texts = similarity_df['app_text'].tolist()
        onet_task_ids = similarity_df['onet_task_id'].tolist() if 'onet_task_id' in similarity_df.columns else [f"task_{i}" for i in range(len(similarity_df))]
        onet_tasks = similarity_df['onet_task'].tolist()

        # Check cache
        cache_path = self._get_ce_cache_path(app_texts, onet_task_ids, cross_encoder_model)
        cached_scores = None

        if use_cache:
            cached_scores = self._load_ce_cache(cache_path)

        # Determine which pairs need processing
        if cached_scores is not None:
            logger.info("Using cached cross-encoder scores")
            # Build scores list from cache
            cross_encoder_scores = []
            for app, task_id in zip(app_texts, onet_task_ids):
                pair_key = (app, task_id)
                if pair_key in cached_scores:
                    cross_encoder_scores.append(cached_scores[pair_key])
                else:
                    cross_encoder_scores.append(None)  # Mark as missing

            # Check if any pairs are missing from cache
            missing_indices = [i for i, score in enumerate(cross_encoder_scores) if score is None]

            if missing_indices:
                logger.warning(f"Cache incomplete: {len(missing_indices)} pairs missing")
                logger.info("Recomputing all scores...")
                cached_scores = None  # Force full recomputation
                cross_encoder_scores = []
            else:
                logger.info(f"All {len(cross_encoder_scores)} scores retrieved from cache")

        # Compute scores if not cached or cache incomplete
        if cached_scores is None:
            logger.info("Computing cross-encoder scores...")

            # Initialize cross-encoder
            try:
                # Initialize cross-encoder with explicit device handling
                cross_encoder = CrossEncoder(cross_encoder_model, device=self.device)

                # Ensure model is on correct device (redundant but safe)
                if hasattr(cross_encoder, 'model'):
                    cross_encoder.model.to(self.device)
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise

            # Prepare input pairs for cross-encoder
            pairs = [[app, task] for app, task in zip(app_texts, onet_tasks)]

            # Run cross-encoder inference in batches with progress tracking
            cross_encoder_scores = []

            for i in tqdm(range(0, len(pairs), batch_size), desc="Cross-encoder batches"):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = cross_encoder.predict(batch_pairs)

                # Convert to list if it's a single score
                if isinstance(batch_scores, (int, float)):
                    batch_scores = [batch_scores]
                elif hasattr(batch_scores, 'tolist'):
                    batch_scores = batch_scores.tolist()

                cross_encoder_scores.extend(batch_scores)

            # Save to cache
            if use_cache:
                scores_dict = {
                    (app, task_id): score
                    for app, task_id, score in zip(app_texts, onet_task_ids, cross_encoder_scores)
                }
                self._save_ce_cache(cache_path, scores_dict, cross_encoder_model)

        # Add cross-encoder scores to dataframe
        similarity_df_with_ce = similarity_df.copy()
        similarity_df_with_ce['cross_encoder_score'] = cross_encoder_scores

        # Apply threshold filtering
        validated_df = similarity_df_with_ce[
            similarity_df_with_ce['cross_encoder_score'] >= threshold
        ].copy()

        # Sort by cross-encoder score in descending order
        validated_df = validated_df.sort_values('cross_encoder_score', ascending=False)

        # Create validation report
        total_matches = len(similarity_df_with_ce)
        validated_matches = len(validated_df)
        filtered_matches = total_matches - validated_matches

        validation_report = {
            'cross_encoder_model': cross_encoder_model,
            'threshold': threshold,
            'batch_size': batch_size,
            'total_matches': total_matches,
            'validated_matches': validated_matches,
            'filtered_matches': filtered_matches,
            'validation_rate': validated_matches / total_matches if total_matches > 0 else 0.0,
            'cross_encoder_stats': {
                'min': float(np.min(cross_encoder_scores)),
                'max': float(np.max(cross_encoder_scores)),
                'mean': float(np.mean(cross_encoder_scores)),
                'std': float(np.std(cross_encoder_scores)),
            },
            'bge_vs_ce_correlation': float(np.corrcoef(
                similarity_df_with_ce['similarity'],
                similarity_df_with_ce['cross_encoder_score']
            )[0, 1]),
            'cache_used': cached_scores is not None
        }

        logger.info(f"Cross-encoder validation completed:")
        logger.info(f"  Total matches: {total_matches}")
        logger.info(f"  Validated matches: {validated_matches}")
        logger.info(f"  Filtered out: {filtered_matches}")
        logger.info(f"  Validation rate: {validation_report['validation_rate']:.3f}")
        logger.info(f"  Cross-encoder score range: {validation_report['cross_encoder_stats']['min']:.3f} - {validation_report['cross_encoder_stats']['max']:.3f}")
        logger.info(f"  BGE vs CE correlation: {validation_report['bge_vs_ce_correlation']:.3f}")
        logger.info(f"  Cache used: {validation_report['cache_used']}")

        return validated_df, validation_report

    def validate_with_cross_encoder_parallel(self,
                                            similarity_df: pd.DataFrame,
                                            cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                                            threshold: float = 0.6,
                                            batch_size: int = 256,
                                            num_workers: int = 4,
                                            use_cache: bool = True) -> Tuple[pd.DataFrame, dict]:
        """
        Parallel version of validate_with_cross_encoder using multiprocessing.

        Args:
            similarity_df: DataFrame with app_text, onet_task, similarity
            cross_encoder_model: Model name
            threshold: Score threshold
            batch_size: Batch size per worker
            num_workers: Number of parallel workers
            use_cache: Whether to use cached scores

        Returns:
            Tuple of (validated_df, validation_report)
        """
        import multiprocessing as mp
        from functools import partial

        logger.info(f"Parallel cross-encoder validation with {num_workers} workers")

        # Extract pairs and IDs
        app_texts = similarity_df['app_text'].tolist()
        onet_task_ids = similarity_df['onet_task_id'].tolist() if 'onet_task_id' in similarity_df.columns else [f"task_{i}" for i in range(len(similarity_df))]
        onet_tasks = similarity_df['onet_task'].tolist()

        # Check cache
        cache_path = self._get_ce_cache_path(app_texts, onet_task_ids, cross_encoder_model)
        cached_scores = None

        if use_cache:
            cached_scores = self._load_ce_cache(cache_path)

        # Determine which pairs need processing
        if cached_scores is not None:
            logger.info("Using cached cross-encoder scores")
            # Build scores list from cache
            cross_encoder_scores = []
            for app, task_id in zip(app_texts, onet_task_ids):
                pair_key = (app, task_id)
                if pair_key in cached_scores:
                    cross_encoder_scores.append(cached_scores[pair_key])
                else:
                    cross_encoder_scores.append(None)  # Mark as missing

            # Check if any pairs are missing from cache
            missing_indices = [i for i, score in enumerate(cross_encoder_scores) if score is None]

            if missing_indices:
                logger.warning(f"Cache incomplete: {len(missing_indices)} pairs missing")
                logger.info("Recomputing all scores...")
                cached_scores = None  # Force full recomputation
                cross_encoder_scores = []
            else:
                logger.info(f"All {len(cross_encoder_scores)} scores retrieved from cache")

        # Compute scores if not cached or cache incomplete
        if cached_scores is None:
            logger.info("Computing cross-encoder scores in parallel...")

            # Split work into chunks
            total_pairs = len(similarity_df)
            chunk_size = (total_pairs + num_workers - 1) // num_workers

            chunks = []
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_pairs)
                if start_idx < total_pairs:
                    chunk_df = similarity_df.iloc[start_idx:end_idx].copy()
                    chunks.append((i, chunk_df, cross_encoder_model, batch_size, self.device))

            logger.info(f"Split {total_pairs} pairs into {len(chunks)} chunks of {chunk_size} each")

            # Process chunks in parallel
            worker_func = partial(_cross_encoder_worker_function)

            try:
                with mp.Pool(processes=num_workers) as pool:
                    results = pool.starmap(worker_func, chunks)

                # Merge results from all workers
                cross_encoder_scores = [None] * total_pairs
                for worker_id, chunk_indices, chunk_scores in results:
                    for idx, score in zip(chunk_indices, chunk_scores):
                        cross_encoder_scores[idx] = score
                    logger.info(f"Worker {worker_id} completed: {len(chunk_scores)} scores")

            except Exception as e:
                logger.error(f"Parallel processing failed: {e}")
                logger.info("Falling back to serial processing...")
                # Fall back to serial version
                return self.validate_with_cross_encoder(
                    similarity_df=similarity_df,
                    cross_encoder_model=cross_encoder_model,
                    threshold=threshold,
                    batch_size=batch_size,
                    use_cache=False
                )

            # Save to cache
            if use_cache:
                scores_dict = {
                    (app, task_id): score
                    for app, task_id, score in zip(app_texts, onet_task_ids, cross_encoder_scores)
                    if score is not None
                }
                self._save_ce_cache(cache_path, scores_dict, cross_encoder_model)

        # Add cross-encoder scores to dataframe
        similarity_df_with_ce = similarity_df.copy()
        similarity_df_with_ce['cross_encoder_score'] = cross_encoder_scores

        # Apply threshold filtering
        validated_df = similarity_df_with_ce[
            similarity_df_with_ce['cross_encoder_score'] >= threshold
        ].copy()

        # Sort by cross-encoder score in descending order
        validated_df = validated_df.sort_values('cross_encoder_score', ascending=False)

        # Create validation report
        total_matches = len(similarity_df_with_ce)
        validated_matches = len(validated_df)
        filtered_matches = total_matches - validated_matches

        validation_report = {
            'cross_encoder_model': cross_encoder_model,
            'threshold': threshold,
            'batch_size': batch_size,
            'num_workers': num_workers,
            'total_matches': total_matches,
            'validated_matches': validated_matches,
            'filtered_matches': filtered_matches,
            'validation_rate': validated_matches / total_matches if total_matches > 0 else 0.0,
            'cross_encoder_stats': {
                'min': float(np.min(cross_encoder_scores)),
                'max': float(np.max(cross_encoder_scores)),
                'mean': float(np.mean(cross_encoder_scores)),
                'std': float(np.std(cross_encoder_scores)),
            },
            'bge_vs_ce_correlation': float(np.corrcoef(
                similarity_df_with_ce['similarity'],
                similarity_df_with_ce['cross_encoder_score']
            )[0, 1]),
            'cache_used': cached_scores is not None,
            'parallel_mode': True
        }

        logger.info(f"Parallel cross-encoder validation completed:")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Total matches: {total_matches}")
        logger.info(f"  Validated matches: {validated_matches}")
        logger.info(f"  Filtered out: {filtered_matches}")
        logger.info(f"  Validation rate: {validation_report['validation_rate']:.3f}")
        logger.info(f"  Cross-encoder score range: {validation_report['cross_encoder_stats']['min']:.3f} - {validation_report['cross_encoder_stats']['max']:.3f}")
        logger.info(f"  BGE vs CE correlation: {validation_report['bge_vs_ce_correlation']:.3f}")
        logger.info(f"  Cache used: {validation_report['cache_used']}")

        return validated_df, validation_report

    def run_full_pipeline(self,
                         step3_file_path: str,
                         onet_file_path: str,
                         output_dir: str = "Data",
                         enable_fuzzy_dedup: bool = False,
                         use_onet_cache: bool = True,
                         use_apps_cache: bool = True,
                         use_cross_encoder_cache: bool = True,
                         use_similarities_cache: bool = True,
                         save_all_similarities: bool = False,
                         exclude_soc_15: bool = False,
                         skip_cross_encoder: bool = False,
                         cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                         cross_encoder_batch_size: int = 256,
                         bge_percentiles: Optional[List[int]] = None,
                         ce_thresholds: Optional[List[float]] = None) -> Tuple[pd.DataFrame, dict]:
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
        # Set defaults for threshold parameters
        if bge_percentiles is None:
            bge_percentiles = [20, 15, 10, 5, 1]
        if ce_thresholds is None:
            ce_thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]

        logger.info("Starting Step 4: O*NET Similarity Matching Pipeline")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Minimum similarity threshold: {self.minimum_similarity}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"O*NET cache: {'enabled' if use_onet_cache else 'disabled'}")
        logger.info(f"Apps cache: {'enabled' if use_apps_cache else 'disabled'}")
        logger.info(f"BGE percentiles: {bge_percentiles}")
        logger.info(f"CE thresholds: {ce_thresholds}")
        
        # Load Step 3 output
        logger.info(f"Loading Step 3 output from: {step3_file_path}")
        step3_df = pd.read_csv(step3_file_path)
        logger.info(f"Loaded {len(step3_df)} records from Step 3")
        
        # Phase A: Deduplicate applications
        dedup_df = self.deduplicate_applications(step3_df, enable_fuzzy=enable_fuzzy_dedup)
        
        # Save deduplicated applications
        dedup_file = os.path.join(output_dir, f"dedup_apps.parquet")
        dedup_df.to_parquet(dedup_file, index=False)
        logger.info(f"Saved deduplicated applications to: {dedup_file}")
        
        # Load O*NET tasks
        onet_df = self.load_onet_tasks(onet_file_path, exclude_soc_15=exclude_soc_15)
        
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
            dedup_df, apps_embeddings, onet_df, onet_embeddings, save_all_similarities=save_all_similarities,
            use_cache=use_similarities_cache, bge_percentiles=bge_percentiles
        )
        
        # Phase D: Validate results
        validation_metrics = self.validate_results(results_df)
        
        # Create deduplicated similarity scores and job mapping
        logger.info("Creating deduplicated similarity scores and job mapping")

        # Filter results_df to only include pairs above the global percentile threshold
        # This reduces the dataset from 202M rows to ~39M rows before the expensive groupby
        max_bge_percentile = max(bge_percentiles)  # e.g., 20

        # Safety check: ensure global_percentile_thresholds is available
        if not hasattr(self, 'global_percentile_thresholds') or self.global_percentile_thresholds is None:
            logger.warning("global_percentile_thresholds not available, computing from results_df")
            all_similarities_array = results_df['similarity'].values
            self.global_percentile_thresholds = {}
            for percentile in sorted(bge_percentiles):
                threshold = np.percentile(all_similarities_array, 100 - percentile)
                self.global_percentile_thresholds[percentile] = threshold
                logger.info(f"  Global {percentile}% percentile (top {percentile}%): {threshold:.4f}")

        ce_percentile_threshold = self.global_percentile_thresholds[max_bge_percentile]
        logger.info(f"Filtering to pairs above global {max_bge_percentile}% percentile ({ce_percentile_threshold:.4f})")

        # Pre-filter before groupby to improve performance
        filtered_results_df = results_df[results_df['similarity'] >= ce_percentile_threshold].copy()
        logger.info(f"Filtered from {len(results_df):,} to {len(filtered_results_df):,} pairs ({len(filtered_results_df)/len(results_df)*100:.1f}%)")

        # Create deduplicated similarity scores: group by (app_text, onet_task_id) and aggregate job_uids
        deduplicated_similarities = filtered_results_df.groupby(
            ['app_text', 'onet_task_id', 'onet_task', 'similarity'],
            as_index=False,
            sort=False  # Skip internal sorting; we sort afterward anyway
        ).agg({
            'job_uid': list,  # Faster than lambda x: list(set(x))
            'first_occurrence_tst_created': 'first'  # Earliest timestamp
        }).reset_index(drop=True)

        # Deduplicate job_uids after groupby (faster than doing it in lambda)
        deduplicated_similarities['job_uid'] = deduplicated_similarities['job_uid'].apply(
            lambda x: list(dict.fromkeys(x))  # Preserves order, removes duplicates
        )

        # Add num_jobs column
        deduplicated_similarities['num_jobs'] = deduplicated_similarities['job_uid'].map(len)

        logger.info(f"Deduplicated to {len(deduplicated_similarities)} unique (app_text, onet_task_id) pairs")

        # Sort both DataFrames by similarity in descending order
        results_df = results_df.sort_values('similarity', ascending=False)
        deduplicated_similarities = deduplicated_similarities.sort_values('similarity', ascending=False)
        
        # Create job mapping (app_text -> list of job_uids) with temporal information
        # CRITICAL FIX: Use filtered_results_df instead of results_df to avoid unnecessary computation on 2B rows
        job_mapping_agg = filtered_results_df.groupby('app_text', sort=False).agg({
            'job_uid': list,  # Faster than lambda x: list(set(x))
            'first_occurrence_tst_created': 'first'  # Earliest timestamp for this app_text
        }).reset_index()

        # Deduplicate job_uids
        job_mapping_agg['job_uid'] = job_mapping_agg['job_uid'].apply(
            lambda x: list(dict.fromkeys(x))  # Preserves order, removes duplicates
        )

        job_mapping = job_mapping_agg.copy()
        job_mapping['job_uids'] = job_mapping['job_uid'].apply(lambda x: '|'.join(sorted(x)))
        job_mapping['num_jobs'] = job_mapping['job_uid'].map(len)  # map() is faster than apply() for len
        job_mapping = job_mapping[['app_text', 'job_uids', 'num_jobs', 'first_occurrence_tst_created']].copy()
        
        logger.info(f"Deduplicated to {len(deduplicated_similarities)} unique app-task pairs")
        logger.info(f"Created job mapping for {len(job_mapping)} unique applications (temporal ordering preserved)")

        # Phase 3 & 4: Determine highest BGE percentile for cross-encoder and filter
        logger.info("Phase 3: Determining highest BGE percentile for cross-encoder")
        max_bge_percentile = max(bge_percentiles)  # e.g., 20 from [20, 10, 5, 1]

        # Use GLOBAL percentile calculated on ALL pairs (not filtered subset)
        ce_percentile_threshold = self.global_percentile_thresholds[max_bge_percentile]
        logger.info(f"Using GLOBAL {max_bge_percentile}% percentile threshold: {ce_percentile_threshold:.4f}")

        # Phase E: Cross-encoder validation (optional)
        cross_encoder_report = None  # Initialize to None (will be set if cross-encoder is used)
        if skip_cross_encoder:
            logger.info("Phase E: Skipping cross-encoder validation (using BGE scores only)")
            unified_matches = deduplicated_similarities.copy()
            # Add placeholder column for compatibility
            unified_matches['cross_encoder_score'] = np.nan
        else:
            ce_input_df = deduplicated_similarities[
                deduplicated_similarities['similarity'] >= ce_percentile_threshold
            ].copy()
            logger.info(f"Running cross-encoder on top {max_bge_percentile}% ({len(ce_input_df)} unique pairs)")

            # Cross-encoder validation (parallel or serial)
            num_workers = getattr(self, 'num_workers', 1)  # Get from instance if set
            if num_workers > 1:
                logger.info(f"Phase E: Parallel cross-encoder validation ({num_workers} workers)")
                cross_encoder_validated_df, cross_encoder_report = self.validate_with_cross_encoder_parallel(
                    similarity_df=ce_input_df.copy(),
                    cross_encoder_model=cross_encoder_model,
                    threshold=0.0,  # Keep ALL results, we'll filter by multiple thresholds later
                    batch_size=cross_encoder_batch_size,
                    num_workers=num_workers,
                    use_cache=use_cross_encoder_cache
                )
            else:
                logger.info("Phase E: Serial cross-encoder validation")
                cross_encoder_validated_df, cross_encoder_report = self.validate_with_cross_encoder(
                    similarity_df=ce_input_df.copy(),
                    cross_encoder_model=cross_encoder_model,
                    threshold=0.0,  # Keep ALL results, we'll filter by multiple thresholds later
                    batch_size=cross_encoder_batch_size,
                    use_cache=use_cross_encoder_cache
                )

            # Merge cross-encoder scores with BGE similarities
            unified_matches = deduplicated_similarities.copy()
            ce_scores = cross_encoder_validated_df[['app_text', 'onet_task_id', 'cross_encoder_score']].copy()
            unified_matches = unified_matches.merge(
                ce_scores,
                on=['app_text', 'onet_task_id'],
                how='left'
            )

        # Create boolean columns for all BGE percentile thresholds
        logger.info("Computing BGE percentile thresholds")
        bge_thresholds = {}

        # Use GLOBAL percentiles if available
        logger.info("Using GLOBAL percentile thresholds (calculated on ALL similarity pairs)")
        for p in bge_percentiles:
            threshold = self.global_percentile_thresholds[p]
            # Format percentile name: handle both integers (20) and floats (0.1)
            if isinstance(p, int) or p == int(p):
                pct_name = f'pct_{int(p):02d}'
            else:
                # For floats like 0.1, format as pct_0p1
                pct_str = f'{p:.1f}'.replace('.', 'p')
                pct_name = f'pct_{pct_str}'
            bge_thresholds[pct_name] = threshold
            logger.info(f"  {p}% percentile (top {p}%): {threshold:.4f}")

        # Add BGE boolean columns
        for pct_name, threshold in bge_thresholds.items():
            unified_matches[pct_name] = unified_matches['similarity'] >= threshold
            match_count = unified_matches[pct_name].sum()
            logger.info(f"  {pct_name}: {match_count:,} matches")

        # Create boolean columns for all cross-encoder thresholds (with NaN handling)
        logger.info("Computing cross-encoder threshold columns")
        for ce_thresh in ce_thresholds:
            col_name = f'ce_{ce_thresh:.1f}'
            if ce_thresh == 0.0:
                # ce_0.0 means no CE filtering - TRUE for all with CE scores, FALSE for NaN
                unified_matches[col_name] = unified_matches['cross_encoder_score'].notna()
                match_count = unified_matches[col_name].sum()
                logger.info(f"  {col_name}: {match_count:,} matches (has CE scores)")
            else:
                # Apply threshold only where CE score exists (NaN will become False)
                unified_matches[col_name] = (
                    (unified_matches['cross_encoder_score'] >= ce_thresh) &
                    unified_matches['cross_encoder_score'].notna()
                )
                match_count = unified_matches[col_name].sum()
                logger.info(f"  {col_name}: {match_count:,} matches")

        # Validation: Enforce CE <= BGE for each specification
        logger.info("Validating CE <= BGE monotonicity...")
        validation_errors = []
        bge_cols = []
        for p in bge_percentiles:
            if isinstance(p, int) or p == int(p):
                bge_cols.append(f'pct_{int(p):02d}')
            else:
                pct_str = f'{p:.1f}'.replace('.', 'p')
                bge_cols.append(f'pct_{pct_str}')
        ce_cols = [f'ce_{c:.1f}' for c in ce_thresholds if c > 0.0]  # Exclude ce_0.0 from validation
        for bge_col in bge_cols:
            for ce_col in ce_cols:
                combined_matches = unified_matches[bge_col] & unified_matches[ce_col]
                ce_only_matches = unified_matches[ce_col] & ~unified_matches[bge_col]
                if ce_only_matches.sum() > 0:
                    error_msg = f"VALIDATION ERROR: {ce_col} has {ce_only_matches.sum()} matches not in {bge_col}"
                    logger.error(error_msg)
                    validation_errors.append(error_msg)

        if validation_errors:
            raise ValueError(f"Validation failed with {len(validation_errors)} errors: {validation_errors[0]}")
        logger.info("✓ Monotonicity validation passed")

        # Save unified match file
        unified_output = os.path.join(output_dir, "task_exposure_matches_all_thresholds.parquet")
        unified_matches.to_parquet(unified_output, index=False, compression='snappy')
        logger.info(f"Saved unified matches file: {unified_output}")
        logger.info(f"  Total unique app-task pairs: {len(unified_matches):,}")

        # Skip backward compatibility files (redundant - all data is in unified_matches with boolean columns)
        # Individual threshold combinations can be filtered from the main file as needed:
        # df = pd.read_parquet('task_exposure_matches_all_thresholds.parquet')
        # top_20_ce06 = df[df['pct_20'] & df['ce_0.6']]
        logger.info("Skipping individual specification files (use main parquet file with boolean columns instead)")

        # Free memory - results_df is no longer needed
        # (unified_matches is the primary output with all threshold combinations and will be returned)
        del results_df
        gc.collect()

        # 2. Job mapping file (to link back to original job ads)
        # This preserves temporal ordering: each app_text shows first occurrence timestamp
        # and all job_uids that contain this text (for use in step 5)
        job_mapping_file = os.path.join(output_dir, "job_app_mapping.parquet")
        job_mapping.to_parquet(job_mapping_file, index=False, compression='snappy')
        logger.info(f"Saved job-application mapping to: {job_mapping_file}")
        
        # 3. CSV format for ALL similarity scores (comprehensive analysis)
        if all_similarities_df is not None:
            # Create deduplicated version of all similarities too
            all_deduplicated = all_similarities_df.drop_duplicates(
                subset=['app_text', 'onet_task_id']
            )[['app_text', 'onet_task_id', 'onet_task', 'similarity']].copy()
            
            # Sort all similarities by similarity in descending order
            all_deduplicated = all_deduplicated.sort_values('similarity', ascending=False)

            all_similarities_file = os.path.join(output_dir, f"all_similarity_scores.parquet")
            all_deduplicated.to_parquet(all_similarities_file, index=False, compression='snappy')
            logger.info(f"Saved deduplicated all similarity scores (Parquet) to: {all_similarities_file}")

            # Free memory
            del all_similarities_df, all_deduplicated
            gc.collect()

        # 4. Save validation metrics (including cross-encoder report if available)
        if cross_encoder_report is not None:
            validation_metrics['cross_encoder_validation'] = cross_encoder_report
        
        metrics_file = os.path.join(output_dir, f"validation_metrics.json")
        import json
        with open(metrics_file, 'w') as f:
            json.dump(validation_metrics, f, indent=2)
        logger.info(f"Saved validation metrics to: {metrics_file}")
        
        # 5. Save separate cross-encoder validation report if available
        if cross_encoder_report is not None:
            ce_report_file = os.path.join(output_dir, f"cross_encoder_validation_report.json")
            with open(ce_report_file, 'w') as f:
                json.dump(cross_encoder_report, f, indent=2)
            logger.info(f"Saved cross-encoder validation report to: {ce_report_file}")
        
        logger.info("Step 4 pipeline completed successfully!")

        return unified_matches, validation_metrics


def _cross_encoder_worker_function(worker_id: int,
                                   chunk_df: pd.DataFrame,
                                   model_name: str,
                                   batch_size: int,
                                   device: str) -> Tuple[int, List[int], List[float]]:
    """
    Worker function for parallel cross-encoder processing.
    Must be defined at module level for multiprocessing.

    Args:
        worker_id: Worker identifier
        chunk_df: Chunk of similarity pairs to process
        model_name: Cross-encoder model name
        batch_size: Batch size for inference
        device: Device to use (mps, cuda, cpu)

    Returns:
        Tuple of (worker_id, chunk_indices, list of scores)
    """
    import torch
    from sentence_transformers import CrossEncoder
    from tqdm import tqdm

    logger.info(f"Worker {worker_id} started: {len(chunk_df)} pairs on device {device}")

    # Initialize cross-encoder on worker
    try:
        cross_encoder = CrossEncoder(model_name, device=device)
    except Exception as e:
        logger.error(f"Worker {worker_id} failed to load model: {e}")
        raise

    # Prepare pairs
    app_texts = chunk_df['app_text'].tolist()
    onet_tasks = chunk_df['onet_task'].tolist()
    pairs = [[app, task] for app, task in zip(app_texts, onet_tasks)]

    # Store original indices to maintain mapping
    chunk_indices = list(range(len(chunk_df)))

    # Run inference
    scores = []
    for i in tqdm(range(0, len(pairs), batch_size),
                  desc=f"Worker {worker_id}",
                  leave=False):
        batch_pairs = pairs[i:i + batch_size]
        batch_scores = cross_encoder.predict(batch_pairs)

        if isinstance(batch_scores, (int, float)):
            batch_scores = [batch_scores]
        elif hasattr(batch_scores, 'tolist'):
            batch_scores = batch_scores.tolist()

        scores.extend(batch_scores)

    logger.info(f"Worker {worker_id} completed: {len(scores)} scores computed")
    return (worker_id, chunk_indices, scores)


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Step 4: AI Applications to O*NET Task Similarity Matching")

    parser.add_argument("--minimum-similarity", type=float, default=0.3,
                       help="Minimum BGE similarity threshold for keeping pairs (default: 0.3)")
    
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
    
    parser.add_argument("--exclude-soc-15", action="store_true",
                       help="Exclude SOC group 15 (Computer and Mathematical Occupations) from analysis")
    
    parser.add_argument("--cross-encoder-model", type=str, default="BAAI/bge-reranker-v2-m3",
                       help="Cross-encoder model name (always runs) (default: BAAI/bge-reranker-v2-m3)")

    parser.add_argument("--cross-encoder-batch-size", type=int, default=256,
                       help="Batch size for cross-encoder inference (default: 256)")

    parser.add_argument("--no-ce-cache", action="store_true",
                       help="Disable cross-encoder score caching (forces recomputation)")

    parser.add_argument("--no-similarities-cache", action="store_true",
                       help="Disable similarity computation caching (forces recomputation)")

    parser.add_argument("--bge-percentiles", type=float, nargs='+', default=[20, 15, 10, 5, 1],
                       help="BGE similarity percentile cutoffs (default: 20 15 10 5 1)")

    parser.add_argument("--ce-thresholds", type=float, nargs='+', default=[0.8, 0.6, 0.4, 0.2, 0.0],
                       help="Cross-encoder score thresholds (default: 0.8 0.6 0.4 0.2 0.0)")

    parser.add_argument("--num-workers", type=int, default=1,
                       help="Number of parallel workers for cross-encoder (default: 1)")

    parser.add_argument("--force-device", type=str,
                       choices=["auto", "mps", "cuda", "cpu"],
                       default="auto",
                       help="Force specific device (default: auto-detect)")

    parser.add_argument("--skip-cross-encoder", action="store_true",
                       help="Skip cross-encoder validation (use BGE scores only, saves memory)")

    return parser.parse_args()


def main():
    """
    Main execution function for Step 4.
    """
    args = parse_arguments()

    # Configuration from arguments
    MINIMUM_SIMILARITY = args.minimum_similarity
    ENABLE_FUZZY_DEDUP = args.enable_fuzzy_dedup
    USE_ONET_CACHE = not args.no_onet_cache
    USE_APPS_CACHE = not args.no_apps_cache
    BGE_PERCENTILES = args.bge_percentiles
    CE_THRESHOLDS = args.ce_thresholds
    
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
            minimum_similarity=MINIMUM_SIMILARITY,
            batch_size=batch_size,
            embeddings_dir=args.embeddings_dir
        )
    except Exception as e:
        logger.error(f"Failed to initialize matcher: {e}")
        return

    # Override device if requested
    if args.force_device != "auto":
        logger.info(f"Overriding device detection: {args.force_device}")
        matcher.device = args.force_device
        matcher.model = matcher.model.to(matcher.device)

    # Set num_workers on matcher instance for pipeline access
    matcher.num_workers = args.num_workers

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
            use_cross_encoder_cache=not args.no_ce_cache,
            use_similarities_cache=not args.no_similarities_cache,
            save_all_similarities=args.save_all_similarities,
            exclude_soc_15=args.exclude_soc_15,
            skip_cross_encoder=args.skip_cross_encoder,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            bge_percentiles=BGE_PERCENTILES,
            ce_thresholds=CE_THRESHOLDS
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