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
                 embeddings_dir: str = "Data/embeddings",
                 use_openai_embeddings: bool = False,
                 use_faiss: bool = False,
                 faiss_k: int = 200,
                 faiss_index_type: str = "Flat",
                 sample_percentiles: int = 10000000):
        """
        Initialize the matcher.

        Args:
            model_name: Hugging Face model name for embeddings
            minimum_similarity: Minimum BGE similarity threshold for keeping pairs (default: 0.3)
            similarity_threshold: Threshold for fuzzy deduplication clustering
            batch_size: Batch size for embedding computation
            embeddings_dir: Directory to store cached embeddings
            use_openai_embeddings: If True, load pre-generated OpenAI embeddings instead of BGE
            use_faiss: If True, use FAISS ANN search for top-k apps per task (default: False)
            faiss_k: Number of top-k apps to retrieve per task in FAISS mode (default: 200)
            faiss_index_type: FAISS index type (default: "Flat")
            sample_percentiles: Sample size for percentile calculation in exhaustive mode (default: 10M)
        """
        self.model_name = model_name
        self.minimum_similarity = minimum_similarity
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.embeddings_dir = embeddings_dir
        self.use_openai_embeddings = use_openai_embeddings
        self.use_faiss = use_faiss
        self.faiss_k = faiss_k
        self.faiss_index_type = faiss_index_type
        self.sample_percentiles = sample_percentiles

        # Create embeddings directory if it doesn't exist
        os.makedirs(embeddings_dir, exist_ok=True)

        # Initialize model only if not using OpenAI embeddings
        if not use_openai_embeddings:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
        else:
            logger.info("Using pre-generated OpenAI embeddings (cache-only mode)")
            self.model = None
        
        # Check if GPU is available (supports CUDA, MPS, and CPU) - only if using BGE
        if not use_openai_embeddings:
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
        else:
            self.device = None
    
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

    def _load_openai_embeddings(self, texts: List[str], text_type: str = "apps") -> Optional[np.ndarray]:
        """
        Load pre-generated OpenAI embeddings from cache (generated by generate_openai_embeddings.py).

        Args:
            texts: List of texts to find embeddings for
            text_type: Either "apps" or "tasks"

        Returns:
            Embeddings array if found and complete, None otherwise

        Raises:
            FileNotFoundError: If OpenAI cache file not found
            ValueError: If embeddings incomplete or mismatch
        """
        # Compute hash to find cache file (matches generate_openai_embeddings.py)
        sorted_texts = ''.join(sorted(texts))
        texts_hash = hashlib.md5(sorted_texts.encode()).hexdigest()[:8]
        cache_filename = f"openai_text-embedding-3-large_{text_type}_{texts_hash}.pkl"
        cache_path = Path(self.embeddings_dir) / cache_filename

        if not cache_path.exists():
            raise FileNotFoundError(
                f"OpenAI embeddings cache not found: {cache_path}\n"
                f"Generate with: python3 generate_openai_embeddings.py --task-type core"
            )

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            embeddings_dict = cache_data.get('embeddings', {})
            metadata = cache_data.get('metadata', {})

            # Validate that we have embeddings for all texts
            missing_texts = [t for t in texts if t not in embeddings_dict]
            if missing_texts:
                logger.error(f"Missing {len(missing_texts)} embeddings from OpenAI cache")
                logger.error(f"First 5 missing: {missing_texts[:5]}")
                raise ValueError(f"OpenAI cache incomplete: missing {len(missing_texts)}/{len(texts)} embeddings")

            # Build embeddings array in correct order
            embeddings = np.array([embeddings_dict[t] for t in texts], dtype=np.float32)

            logger.info(f"Loaded OpenAI embeddings from cache: {cache_path.name}")
            logger.info(f"  Model: {metadata.get('model')}, Dimensions: {metadata.get('dimensions')}")
            logger.info(f"  Texts: {len(texts):,}, Total tokens: {metadata.get('total_tokens', 'unknown'):,}")

            return embeddings

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to load OpenAI embeddings from {cache_path}: {e}")
            raise ValueError(f"Failed to load OpenAI cache: {e}")

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

    def _generate_run_hash(self, app_texts: List[str], onet_task_ids: List[str],
                           model_name: str) -> str:
        """
        Generate deterministic hash for this specific cross-encoder run.
        Used to validate checkpoint matches current run parameters.

        Args:
            app_texts: List of AI application texts
            onet_task_ids: List of O*NET task IDs
            model_name: Cross-encoder model name

        Returns:
            8-character hash string
        """
        content = f"{model_name}|{len(app_texts)}|{len(onet_task_ids)}"
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _get_ce_checkpoint_path(self, model_name: str,
                                total_pairs: int, run_hash: str,
                                start_idx: Optional[int] = None,
                                end_idx: Optional[int] = None) -> Path:
        """
        Generate checkpoint file path for cross-encoder worker progress.

        Uses content-based naming (chunk indices) for immutable checkpoints that work
        with any number of workers. Falls back to single-file naming if indices not provided.

        Args:
            model_name: Cross-encoder model name
            total_pairs: Total number of pairs being processed
            run_hash: Hash identifying this specific run
            start_idx: Starting index in full dataset (optional, for content-based naming)
            end_idx: Ending index in full dataset (optional, for content-based naming)

        Returns:
            Path to checkpoint file
        """
        checkpoint_dir = Path(self.embeddings_dir) / "cross_encoder_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        safe_model_name = model_name.replace("/", "_").replace("-", "_")

        # Use content-based naming if indices provided (preferred for resumable checkpoints)
        if start_idx is not None and end_idx is not None:
            filename = f"ce_chunk_{start_idx}_{end_idx}_{safe_model_name}_total{total_pairs}_{run_hash}.parquet"
        else:
            # Fallback to single merged file for backward compatibility
            filename = f"ce_merged_{safe_model_name}_total{total_pairs}_{run_hash}.parquet"

        return checkpoint_dir / filename

    def _load_ce_checkpoint(self, checkpoint_path: Path,
                            expected_metadata: dict) -> Optional[pd.DataFrame]:
        """
        Load and validate worker checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            expected_metadata: Expected metadata for validation

        Returns:
            DataFrame with global_index and cross_encoder_score, or None if invalid
        """
        if not checkpoint_path.exists():
            return None

        try:
            checkpoint_df = pd.read_parquet(checkpoint_path)

            # Validate structure
            required_cols = ['global_index', 'cross_encoder_score']
            if not all(col in checkpoint_df.columns for col in required_cols):
                logger.warning(f"Invalid checkpoint structure: {checkpoint_path.name}")
                return None

            # Validate metadata matches current run
            attrs = checkpoint_df.attrs
            if (attrs.get('model_name') != expected_metadata['model_name'] or
                attrs.get('total_pairs') != expected_metadata['total_pairs'] or
                attrs.get('run_hash') != expected_metadata['run_hash']):
                logger.warning(f"Checkpoint metadata mismatch: {checkpoint_path.name}")
                logger.warning(f"  Expected: {expected_metadata}")
                logger.warning(f"  Found: model={attrs.get('model_name')}, "
                             f"total_pairs={attrs.get('total_pairs')}, "
                             f"run_hash={attrs.get('run_hash')}")
                return None

            logger.info(f"Loaded valid checkpoint: {checkpoint_path.name} "
                       f"({len(checkpoint_df)} scores from {attrs.get('created_at', 'unknown')})")
            return checkpoint_df

        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path.name}: {e}")
            return None

    def _cleanup_ce_checkpoints(self, checkpoint_dir: Path, run_hash: str) -> None:
        """
        Delete all checkpoint files for this run after successful completion.

        Args:
            checkpoint_dir: Directory containing checkpoints
            run_hash: Hash identifying this run
        """
        try:
            pattern = f"*_{run_hash}.parquet"
            deleted_count = 0
            for checkpoint_file in checkpoint_dir.glob(pattern):
                checkpoint_file.unlink()
                deleted_count += 1
                logger.debug(f"Deleted checkpoint: {checkpoint_file.name}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} checkpoint files")

        except Exception as e:
            logger.warning(f"Failed to cleanup checkpoints: {e}")

    def _get_similarity_cache_path(self, app_texts: List[str], onet_tasks: List[str],
                                   minimum_similarity: float, mode: str = "exhaustive") -> Path:
        """
        Generate cache file path for similarity computation results.

        Args:
            app_texts: List of AI application texts
            onet_tasks: List of O*NET task texts
            minimum_similarity: Minimum similarity threshold used
            mode: "faiss" or "exhaustive" (default: "exhaustive")

        Returns:
            Path to cache file
        """
        # Get model identifier to include in cache filename (same as embeddings do)
        if self.use_openai_embeddings:
            model_identifier = "openai_text-embedding-3-large"
        else:
            # Extract model abbreviation from model name (e.g., "BAAI/bge-large-en-v1.5" -> "bge")
            model_abbr = self.model_name.split("/")[-1].split("-")[0]
            model_identifier = model_abbr

        # Include FAISS parameters in hash if FAISS mode
        if mode == "faiss":
            mode_suffix = f"_faiss_k{self.faiss_k}"
            cache_hash_input = f"{len(app_texts)}|{len(onet_tasks)}|{minimum_similarity}|{model_identifier}|faiss|{self.faiss_k}"
        else:
            mode_suffix = "_exhaustive"
            cache_hash_input = f"{len(app_texts)}|{len(onet_tasks)}|{minimum_similarity}|{model_identifier}|exhaustive"

        # Create stable deterministic hash
        content_hash = hashlib.md5(cache_hash_input.encode()).hexdigest()[:8]

        cache_filename = f"similarities_apps{len(app_texts)}_tasks{len(onet_tasks)}_min{minimum_similarity:.1f}{mode_suffix}_{model_identifier}_{content_hash}.pkl"
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
                               all_results_df: Optional[pd.DataFrame], top_n_percent: float,
                               mode: str = "exhaustive") -> None:
        """
        Save similarity computation results to cache.

        Args:
            cache_path: Path to save cache file
            filtered_results_df: Filtered similarity results
            all_results_df: Optional all similarity results (None in FAISS mode)
            top_n_percent: Top N percent threshold used
            mode: "faiss" or "exhaustive" (default: "exhaustive")
        """
        try:
            cache_data = {
                'filtered_results': filtered_results_df,
                'all_results': all_results_df,
                'top_n_percent': top_n_percent,
                'mode': mode,
                'created_at': datetime.now().isoformat()
            }

            # Save global percentiles if available (exhaustive mode only)
            if hasattr(self, 'global_percentile_thresholds') and self.global_percentile_thresholds is not None:
                cache_data['global_percentile_thresholds'] = self.global_percentile_thresholds

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"Saved {mode} mode similarity computation results to cache")
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
    
    def load_onet_tasks(self, onet_file_path: str, include_soc_15: bool = False,
                       filter_supplements: bool = True) -> pd.DataFrame:
        """
        Load O*NET task statements from Excel file.

        Args:
            onet_file_path: Path to O*NET Task Statements.xlsx file
            include_soc_15: If True, include SOC group 15 (Computer and Mathematical Occupations). Default is False (excluded).
            filter_supplements: If True, exclude Supplemental tasks (keep Core only)

        Returns:
            DataFrame with task_id, Task, and Task Type columns
        """
        logger.info(f"Loading O*NET tasks from: {onet_file_path}")

        try:
            # Try to read the Excel file
            onet_df = pd.read_excel(onet_file_path)

            # Check for expected columns
            if 'Task' not in onet_df.columns:
                raise ValueError("Expected 'Task' column not found in O*NET file")
            if 'Task ID' not in onet_df.columns:
                raise ValueError("Expected 'Task ID' column not found in O*NET file")

            # Use actual Task ID from the file (not row index)
            # Task IDs are unique identifiers in the O*NET file
            onet_df = onet_df.reset_index(drop=True)
            onet_df['task_id'] = onet_df['Task ID'].astype(int)

            # Clean and validate tasks
            onet_df = onet_df[onet_df['Task'].notna()].copy()
            onet_df['Task'] = onet_df['Task'].astype(str).str.strip()
            onet_df = onet_df[onet_df['Task'] != ''].copy()

            # Filter to exclude Supplemental tasks only if requested (preserve Task Type column)
            if filter_supplements and 'Task Type' in onet_df.columns:
                initial_count = len(onet_df)
                onet_df = onet_df[onet_df['Task Type'] != 'Supplemental'].copy()
                filtered_count = initial_count - len(onet_df)
                logger.info(f"Filtered out {filtered_count} Supplemental tasks, keeping {len(onet_df)} Core tasks")

            # Filter out SOC group 15 by default (unless include_soc_15 is True)
            if not include_soc_15:
                initial_count = len(onet_df)
                onet_df = onet_df[~onet_df['O*NET-SOC Code'].str.startswith('15-', na=False)].copy()
                excluded_count = initial_count - len(onet_df)
                logger.info(f"Excluded {excluded_count} tasks from SOC group 15 (Computer and Mathematical Occupations)")
            else:
                logger.info(f"Including SOC group 15 (Computer and Mathematical Occupations) tasks")

            logger.info(f"Loaded {len(onet_df)} O*NET task statements")
            return onet_df[['task_id', 'Task', 'Task Type', 'O*NET-SOC Code']]

        except Exception as e:
            logger.error(f"Error loading O*NET file: {e}")
            raise
    
    def embed_texts(self, texts: List[str], description: str = "texts",
                   cache_prefix: str = "texts", use_cache: bool = True,
                   text_type: str = "apps") -> np.ndarray:
        """
        Phase B: Embed texts using BGE-large model or load pre-generated OpenAI embeddings.

        Args:
            texts: List of text strings to embed
            description: Description for logging
            cache_prefix: Prefix for cache files (BGE only)
            use_cache: Whether to use cached embeddings
            text_type: Either "apps" or "tasks" (for OpenAI embeddings)

        Returns:
            Normalized embeddings array
        """
        logger.info(f"Phase B: Embedding {len(texts)} {description}")

        # If using OpenAI embeddings, load from pre-generated cache
        if self.use_openai_embeddings:
            logger.info("Loading pre-generated OpenAI embeddings from cache")
            try:
                embeddings = self._load_openai_embeddings(texts, text_type=text_type)
                logger.info(f"Embeddings shape: {embeddings.shape}")
                return embeddings
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Failed to load OpenAI embeddings: {e}")
                raise

        # Otherwise use BGE embeddings with caching
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

    def _build_faiss_index(self, embeddings: np.ndarray, index_type: str = "Flat", use_gpu: bool = True):
        """
        Build FAISS index on embeddings for efficient similarity search.

        Args:
            embeddings: Numpy array of embeddings (n_vectors, embedding_dim)
            index_type: FAISS index type ("Flat" for exact search)
            use_gpu: If True, attempt to use GPU for FAISS

        Returns:
            FAISS index object
        """
        try:
            import faiss
            faiss_available = True
            faiss_gpu = hasattr(faiss, 'StandardGpuResources')
            if faiss_gpu:
                logger.info("Detected faiss-gpu")
            else:
                logger.info("Using faiss-cpu")
        except ImportError:
            raise ImportError(
                "FAISS not installed. Please install FAISS to use --use-faiss mode:\n"
                "  For GPU (CUDA required): pip install faiss-gpu\n"
                "  For CPU only: pip install faiss-cpu\n"
                "Note: faiss-gpu and faiss-cpu conflict, only install one."
            )

        # Ensure embeddings are normalized (for cosine similarity with inner product)
        # Embeddings should already be normalized from encode(), but verify
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-5):
            logger.warning("Embeddings not normalized, normalizing now...")
            embeddings = embeddings / norms

        embedding_dim = embeddings.shape[1]
        n_vectors = embeddings.shape[0]

        logger.info(f"Building FAISS index (type: {index_type}, vectors: {n_vectors}, dim: {embedding_dim})")

        # Build index based on type
        if index_type == "Flat":
            # Flat index with inner product (cosine similarity on normalized vectors)
            index = faiss.IndexFlatIP(embedding_dim)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

        # Attempt GPU if requested and available
        if use_gpu and faiss_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move FAISS index to GPU: {e}. Using CPU.")

        # Add vectors to index
        index.add(embeddings.astype(np.float32))
        logger.info(f"FAISS index built with {index.ntotal} vectors")

        return index

    def compute_similarities_and_filter(self,
                                      apps_df: pd.DataFrame,
                                      apps_embeddings: np.ndarray,
                                      onet_df: pd.DataFrame,
                                      onet_embeddings: np.ndarray,
                                      use_cache: bool = True,
                                      per_task_mode: bool = False,
                                      bge_percentiles: Optional[List[int]] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Phase C: Compute similarities and apply filtering.

        Branches between two modes:
        - FAISS mode: Retrieve top-k apps per task, filter by threshold (no percentiles)
        - Exhaustive mode: Compute all pairs, calculate percentiles, filter by percentiles

        Args:
            apps_df: DataFrame with application strings
            apps_embeddings: Embeddings for application strings
            onet_df: DataFrame with O*NET tasks
            onet_embeddings: Embeddings for O*NET tasks
            use_cache: Whether to use cached similarity results (default: True)
            per_task_mode: Use per-task percentile filtering (exhaustive mode only)
            bge_percentiles: List of percentile thresholds (exhaustive mode only, default: [20, 15, 10, 5, 1])

        Returns:
            Tuple of (filtered_results_df, all_similarities_df)
            - FAISS mode: all_similarities_df is None (no percentile calculation)
            - Exhaustive mode: all_similarities_df contains sampled pairs for percentiles
        """
        logger.info("Phase C: Computing similarities and applying filtering")
        logger.info(f"Mode: {'FAISS (top-k apps per task)' if self.use_faiss else 'Exhaustive (global percentiles)'}")
        logger.info(f"Minimum similarity threshold: {self.minimum_similarity}")

        # Branch between FAISS and exhaustive modes
        if self.use_faiss:
            return self._compute_similarities_faiss(
                apps_df, apps_embeddings, onet_df, onet_embeddings, use_cache
            )
        else:
            return self._compute_similarities_exhaustive(
                apps_df, apps_embeddings, onet_df, onet_embeddings,
                use_cache, per_task_mode, bge_percentiles
            )

    def _compute_similarities_faiss(self,
                                    apps_df: pd.DataFrame,
                                    apps_embeddings: np.ndarray,
                                    onet_df: pd.DataFrame,
                                    onet_embeddings: np.ndarray,
                                    use_cache: bool = True) -> Tuple[pd.DataFrame, None]:
        """
        FAISS mode: For each O*NET task, retrieve top-k most similar apps.
        All pairs above threshold are considered matches (no percentile filtering).

        Args:
            apps_df: DataFrame with application strings
            apps_embeddings: Embeddings for application strings (will be indexed)
            onet_df: DataFrame with O*NET tasks
            onet_embeddings: Embeddings for O*NET tasks (will be queries)
            use_cache: Whether to use cached results

        Returns:
            Tuple of (results_df, None) - no all_similarities_df in FAISS mode
        """
        logger.info(f"FAISS mode: Retrieving top-{self.faiss_k} apps per task")

        # Check cache first
        app_texts = apps_df['app_text'].unique().tolist()
        onet_tasks = onet_df['Task'].tolist()
        cache_path = self._get_similarity_cache_path(app_texts, onet_tasks, self.minimum_similarity, mode="faiss")

        if use_cache:
            cache_result = self._load_similarity_cache(cache_path)
            if cache_result is not None:
                logger.info("Using cached FAISS similarity computation results")
                return cache_result[0], None  # Return only results_df, no all_similarities

        # Step 1: Build job_uid lookup
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

        # Step 2: Build FAISS index on app embeddings
        use_gpu = self.device and self.device in ['cuda', 'mps']
        faiss_index = self._build_faiss_index(
            apps_embeddings,
            index_type=self.faiss_index_type,
            use_gpu=use_gpu
        )

        # Step 3: Create app_text lookup by index
        unique_apps = apps_df['app_text'].unique()
        app_idx_to_text = {i: app_text for i, app_text in enumerate(unique_apps)}

        # Step 4: Loop over O*NET tasks and query FAISS
        task_id_list = onet_df['task_id'].tolist()
        task_text_list = onet_df['Task'].tolist()

        results = []
        n_tasks_saturated = 0  # Track how many tasks hit all k results above threshold

        for task_idx, task_id in enumerate(task_id_list):
            # Query FAISS: get top-k apps for this task
            task_embedding = onet_embeddings[task_idx:task_idx+1]  # Shape (1, dim)
            k_actual = min(self.faiss_k, len(apps_embeddings))

            # Search returns (similarities, app_indices)
            similarities, app_indices = faiss_index.search(task_embedding, k_actual)
            similarities = similarities[0]  # Flatten from (1, k) to (k,)
            app_indices = app_indices[0]

            # Filter by minimum threshold
            above_threshold = similarities >= self.minimum_similarity
            selected_app_indices = app_indices[above_threshold]
            selected_similarities = similarities[above_threshold]

            # Track saturation (indicator that k might be too small)
            if len(selected_app_indices) == k_actual:
                n_tasks_saturated += 1

            # For each selected app, add all job_uids to results
            for app_idx, similarity in zip(selected_app_indices, selected_similarities):
                app_text = app_idx_to_text[app_idx]
                job_uid_to_timestamp = job_uid_lookup.get(app_text, {})

                for job_uid, timestamp in job_uid_to_timestamp.items():
                    results.append({
                        'job_uid': job_uid,
                        'app_text': app_text,
                        'onet_task_id': task_id,
                        'onet_task': task_text_list[task_idx],
                        'similarity': float(similarity),
                        'first_occurrence_tst_created': timestamp
                    })

            if (task_idx + 1) % 100 == 0:
                logger.info(f"Processed task {task_idx + 1}/{len(task_id_list)}")

        # Step 5: Convert to DataFrame
        results_df = pd.DataFrame(results)
        logger.info(f"FAISS mode generated {len(results_df)} matches (all above threshold {self.minimum_similarity})")

        # Log saturation statistics
        saturation_pct = 100 * n_tasks_saturated / len(task_id_list) if len(task_id_list) > 0 else 0
        logger.info(f"Saturation: {n_tasks_saturated}/{len(task_id_list)} tasks ({saturation_pct:.1f}%) "
                   f"had all {self.faiss_k} retrieved apps above threshold")
        if saturation_pct > 50:
            logger.warning(f"High saturation ({saturation_pct:.1f}%)! Consider increasing --faiss-k to avoid missing matches.")

        # Save to cache
        if use_cache:
            self._save_similarity_cache(cache_path, results_df, None, self.minimum_similarity, mode="faiss")

        return results_df, None

    def _compute_similarities_exhaustive(self,
                                        apps_df: pd.DataFrame,
                                        apps_embeddings: np.ndarray,
                                        onet_df: pd.DataFrame,
                                        onet_embeddings: np.ndarray,
                                        use_cache: bool = True,
                                        per_task_mode: bool = False,
                                        bge_percentiles: Optional[List[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Exhaustive mode: Compute all app-task similarities, calculate global percentiles, filter by percentiles.
        Samples pairs for percentile calculation to reduce memory usage.

        Args:
            apps_df: DataFrame with application strings
            apps_embeddings: Embeddings for application strings
            onet_df: DataFrame with O*NET tasks
            onet_embeddings: Embeddings for O*NET tasks
            use_cache: Whether to use cached results
            per_task_mode: Use per-task percentile filtering
            bge_percentiles: List of percentile thresholds (default: [20, 15, 10, 5, 1])

        Returns:
            Tuple of (filtered_results_df, all_similarities_df)
        """
        logger.info("Exhaustive mode: Computing all app-task similarities")

        # Set default percentiles if not provided
        if bge_percentiles is None:
            bge_percentiles = [20, 15, 10, 5, 1]

        # Check cache first
        app_texts = apps_df['app_text'].unique().tolist()
        onet_tasks = onet_df['Task'].tolist()
        cache_path = self._get_similarity_cache_path(app_texts, onet_tasks, self.minimum_similarity, mode="exhaustive")

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
        all_results_df = pd.DataFrame()  # Always save all similarities for accurate percentile calculation

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

                if chunk_checkpoint_all.exists():
                    chunk_all_df = pd.read_parquet(chunk_checkpoint_all)
                    all_results_df = pd.concat([all_results_df, chunk_all_df], ignore_index=True)

                logger.info(f"Skipping processing for chunk {chunk_start}-{chunk_end} (already checkpointed)")
                continue

            logger.info(f"Processing chunk {chunk_start}-{chunk_end} of {len(unique_apps)} apps")

            # Pre-build lists for O(1) access during similarity computation
            # These maintain positional alignment with onet_embeddings array
            task_id_list = onet_df['task_id'].tolist()
            task_text_list = onet_df['Task'].tolist()

            # Accumulate results for this chunk only
            chunk_filtered_results = []
            chunk_all_results = []  # Sampled similarities for percentile calculation

            # Calculate sampling probability for percentile calculation
            total_possible_pairs = len(unique_apps) * len(onet_tasks) * max(1, len(job_uid_lookup) / len(unique_apps))
            sample_prob = min(1.0, self.sample_percentiles / total_possible_pairs)
            logger.info(f"Sampling probability for percentiles: {sample_prob:.6f} (target: {self.sample_percentiles:,} pairs)")

            for local_idx, app_text in enumerate(chunk_apps):
                global_idx = chunk_start + local_idx

                # Compute cosine similarities to all O*NET tasks
                app_embedding = apps_embeddings[global_idx:global_idx+1]  # Shape: (1, embedding_dim)
                similarities = np.dot(app_embedding, onet_embeddings.T)[0]  # Shape: (n_onet_tasks,)

                # Get job_uids and timestamps from pre-built lookup (O(1) dict access, not DataFrame scan)
                job_uid_to_timestamp = job_uid_lookup.get(app_text, {})
                matching_jobs = list(job_uid_to_timestamp.keys())

                # Randomly sample similarities for percentile calculation (memory optimization)
                for onet_idx, similarity in enumerate(similarities):
                    for job_uid in matching_jobs:
                        # Sample for percentile calculation
                        if np.random.random() < sample_prob:
                            chunk_all_results.append({
                                'job_uid': job_uid,
                                'app_text': app_text,
                                'onet_task_id': task_id_list[onet_idx],
                                'onet_task': task_text_list[onet_idx],
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
                            'onet_task_id': task_id_list[onet_idx],
                            'onet_task': task_text_list[onet_idx],
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

            if chunk_all_results:
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
        logger.info(f"Generated {len(all_results_df)} total similarity matches")

        # Calculate global percentile threshold on ALL similarity scores
        # This must be done BEFORE the 0.3 filter to get true top 20% of all pairs
        logger.info("Computing global percentile on ALL similarity scores")
        all_similarities_array = all_results_df['similarity'].values

        # Store percentile thresholds for later use by Phase E
        self.global_percentile_thresholds = {}
        for percentile in sorted(bge_percentiles):  # Use the passed percentiles, not hardcoded ones
            threshold = np.percentile(all_similarities_array, 100 - percentile)
            self.global_percentile_thresholds[percentile] = threshold
            logger.info(f"  Global {percentile}% percentile (top {percentile}%): {threshold:.4f}")

        logger.info(f"Global percentiles calculated from {len(all_similarities_array):,} total pairs")

        # Clean up chunk checkpoints after successful completion
        logger.info("Cleaning up chunk checkpoints...")
        for checkpoint_file in checkpoint_dir.glob(f"{checkpoint_prefix.name}_chunk_*.parquet"):
            checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_file.name}")

        # If in per-task mode, also compute per-task rankings
        if per_task_mode:
            filtered_results_df = self._compute_per_task_rankings(filtered_results_df)

        # Save to cache
        if use_cache:
            self._save_similarity_cache(cache_path, filtered_results_df, all_results_df, self.minimum_similarity)

        return filtered_results_df, all_results_df

    def _compute_per_task_rankings(self, results_df):
        """
        Compute per-task percentile rankings for each O*NET task.

        For each task, ranks all matching AI applications by similarity score,
        then converts ranks to percentiles (0-1 scale where 0 = best match).

        Args:
            results_df: DataFrame with columns ['app_text', 'onet_task_id', 'similarity']

        Returns:
            DataFrame with added 'per_task_rank_pct' column
        """
        logger.info("Computing per-task percentile rankings...")

        # Group by task and rank within each group
        results_df['per_task_rank'] = results_df.groupby('onet_task_id')['similarity'].rank(
            ascending=False, method='min'
        )

        # Convert ranks to percentiles (0-1 scale)
        task_counts = results_df.groupby('onet_task_id').size()
        results_df['per_task_rank_pct'] = results_df.apply(
            lambda row: (row['per_task_rank'] - 1) / (task_counts[row['onet_task_id']] - 1)
            if task_counts[row['onet_task_id']] > 1 else 0.0,
            axis=1
        )

        logger.info(f"  Added per-task rankings for {results_df['onet_task_id'].nunique()} tasks")

        return results_df

    def _filter_to_top_per_task_percentile(self, df, percentile):
        """
        Filter to top N% of matches per task (e.g., percentile=20 keeps top 20% per task).

        Args:
            df: DataFrame with 'per_task_rank_pct' column
            percentile: Percentile threshold (0-100)

        Returns:
            Filtered DataFrame
        """
        threshold = percentile / 100.0
        filtered = df[df['per_task_rank_pct'] <= threshold].copy()

        logger.info(f"  Filtered to top {percentile}% per task: {len(filtered):,} pairs")

        return filtered

    def _validate_per_task_monotonicity(self, unified_df, percentile_cols):
        """
        Validate that per-task percentile columns follow subset relationship.

        Each stricter percentile should be a subset of the looser one:
        pct_20 ⊇ pct_15 ⊇ pct_10 ⊇ pct_05 ⊇ pct_01

        Args:
            unified_df: DataFrame with boolean percentile columns
            percentile_cols: List of column names in order from loosest to strictest

        Raises:
            ValueError: If monotonicity is violated
        """
        logger.info("\nValidating per-task percentile monotonicity...")

        for i in range(len(percentile_cols) - 1):
            looser_col = percentile_cols[i]
            stricter_col = percentile_cols[i + 1]

            # Count violations where stricter=True but looser=False
            violations = unified_df[unified_df[stricter_col] & ~unified_df[looser_col]]

            if len(violations) > 0:
                raise ValueError(
                    f"Per-task monotonicity violation: {stricter_col} has {len(violations)} "
                    f"matches not in {looser_col}"
                )

            logger.info(f"  ✓ {stricter_col} ⊆ {looser_col}")

        logger.info("✓ Per-task monotonicity validation passed")

    def _calculate_percentiles_and_create_columns(self,
                                                  unified_matches: pd.DataFrame,
                                                  all_similarities_df: pd.DataFrame,
                                                  task_type_filter: Optional[str] = None,
                                                  bge_percentiles: Optional[List[float]] = None,
                                                  ce_thresholds: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Calculate percentiles and create boolean columns for a task subset.

        Args:
            unified_matches: Deduplicated matches with task metadata
            all_similarities_df: All similarity scores for percentile calculation
            task_type_filter: 'Core' to filter to core tasks, None for all tasks
            bge_percentiles: BGE percentile thresholds
            ce_thresholds: Cross-encoder thresholds

        Returns:
            DataFrame with boolean threshold columns
        """
        # Filter if specified
        if task_type_filter == 'Core':
            filtered_matches = unified_matches[unified_matches['task_type'] == 'Core'].copy()
            filtered_sims = all_similarities_df[
                all_similarities_df['onet_task_id'].isin(filtered_matches['onet_task_id'].unique())
            ].copy()
            logger.info(f"Filtered to {len(filtered_matches):,} Core task matches")
        else:
            filtered_matches = unified_matches.copy()
            filtered_sims = all_similarities_df.copy()
            logger.info(f"Using all {len(filtered_matches):,} task matches")

        # Calculate percentiles on filtered subset
        all_similarities_array = filtered_sims['similarity'].values
        local_percentile_thresholds = {}
        for percentile in sorted(bge_percentiles):
            threshold = np.percentile(all_similarities_array, 100 - percentile)
            local_percentile_thresholds[percentile] = threshold
            logger.info(f"  {percentile}% percentile: {threshold:.4f}")

        # Create BGE boolean columns
        for p in bge_percentiles:
            threshold = local_percentile_thresholds[p]
            # Format percentile name: handle both integers and floats
            if isinstance(p, int) or p == int(p):
                pct_name = f'pct_{int(p):02d}'
            else:
                pct_str = f'{p:.1f}'.replace('.', 'p')
                pct_name = f'pct_{pct_str}'
            filtered_matches[pct_name] = filtered_matches['similarity'] >= threshold
            logger.info(f"  {pct_name}: {filtered_matches[pct_name].sum():,} matches")

        # Create CE boolean columns
        for ce_thresh in ce_thresholds:
            col_name = f'ce_{ce_thresh:.1f}'
            if ce_thresh == 0.0:
                # ce_0.0 means "no CE filtering" - always True to act as a pass-through
                # This allows the pipeline to work with or without cross-encoder scores
                filtered_matches[col_name] = True
            else:
                filtered_matches[col_name] = (
                    (filtered_matches['cross_encoder_score'] >= ce_thresh) &
                    filtered_matches['cross_encoder_score'].notna()
                )
            logger.info(f"  {col_name}: {filtered_matches[col_name].sum():,} matches")

        return filtered_matches

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

    def _deduplicate_in_chunks(self,
                               filtered_df: pd.DataFrame,
                               chunk_size: int = 5_000_000,
                               output_dir: str = None) -> pd.DataFrame:
        """
        Memory-efficient deduplication using chunked processing and temporary Parquet files.

        Process large DataFrames in chunks to avoid memory exhaustion during groupby operations.
        Each chunk is deduplicated separately, then iteratively merged with re-deduplication
        across chunk boundaries.

        Args:
            filtered_df: DataFrame to deduplicate (already filtered by similarity threshold)
            chunk_size: Number of rows per chunk (default: 5M)
            output_dir: Directory for temporary files (default: Data/temp_chunks/)

        Returns:
            Deduplicated DataFrame with aggregated job_uid lists
        """
        import tempfile
        import shutil
        from datetime import datetime

        # Setup temp directory
        if output_dir is None:
            output_dir = "Data"
        temp_dir = os.path.join(output_dir, f"temp_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Starting chunked deduplication (chunk size: {chunk_size:,} rows)")
        logger.info(f"Temporary directory: {temp_dir}")

        try:
            # Step 1: Process chunks and save to temporary files
            n_rows = len(filtered_df)
            n_chunks = (n_rows + chunk_size - 1) // chunk_size  # Ceiling division
            chunk_files = []

            logger.info(f"Processing {n_rows:,} rows in {n_chunks} chunks")

            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_rows)

                logger.info(f"Chunk {i+1}/{n_chunks}: rows {start_idx:,} to {end_idx:,}")

                # Extract chunk
                chunk = filtered_df.iloc[start_idx:end_idx].copy()

                # Deduplicate within chunk
                chunk_dedup = chunk.groupby(
                    ['app_text', 'onet_task_id', 'onet_task', 'similarity'],
                    as_index=False,
                    sort=False
                ).agg({
                    'job_uid': list,
                    'first_occurrence_tst_created': 'first'
                }).reset_index(drop=True)

                # Deduplicate job_uids
                chunk_dedup['job_uid'] = chunk_dedup['job_uid'].apply(
                    lambda x: list(dict.fromkeys(x))
                )

                # Save to temp file
                chunk_file = os.path.join(temp_dir, f"chunk_{i:04d}.parquet")
                chunk_dedup.to_parquet(chunk_file, index=False, compression='snappy')
                chunk_files.append(chunk_file)

                logger.info(f"  Chunk {i+1} deduplicated: {len(chunk):,} → {len(chunk_dedup):,} unique pairs")

                # Free memory
                del chunk, chunk_dedup
                gc.collect()

            # Step 2: Iteratively merge chunks with re-deduplication
            logger.info(f"Merging {len(chunk_files)} chunks")

            # Start with first chunk
            merged = pd.read_parquet(chunk_files[0])
            logger.info(f"Loaded chunk 1: {len(merged):,} pairs")

            # Merge remaining chunks one at a time
            for i, chunk_file in enumerate(chunk_files[1:], start=2):
                next_chunk = pd.read_parquet(chunk_file)
                logger.info(f"Loaded chunk {i}: {len(next_chunk):,} pairs")

                # Concatenate
                before_merge = len(merged) + len(next_chunk)
                merged = pd.concat([merged, next_chunk], ignore_index=True)

                # Re-deduplicate across chunk boundary
                # Need to combine job_uid lists for same (app_text, onet_task_id) pairs
                merged = merged.groupby(
                    ['app_text', 'onet_task_id', 'onet_task', 'similarity'],
                    as_index=False,
                    sort=False
                ).agg({
                    'job_uid': lambda x: list(dict.fromkeys([uid for sublist in x for uid in sublist])),
                    'first_occurrence_tst_created': 'first'
                }).reset_index(drop=True)

                after_merge = len(merged)
                logger.info(f"  After merge and dedup: {before_merge:,} → {after_merge:,} pairs")

                # Free memory
                del next_chunk
                gc.collect()

            logger.info(f"Final deduplicated result: {len(merged):,} unique (app_text, onet_task_id) pairs")

        finally:
            # Step 3: Cleanup temporary files
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")

        return merged

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

        # Checkpoint detection and resume logic
        checkpoint_dir = Path(self.embeddings_dir) / "cross_encoder_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        run_hash = self._generate_run_hash(app_texts, onet_task_ids, cross_encoder_model)
        run_metadata = {
            'model_name': cross_encoder_model,
            'total_pairs': len(similarity_df),
            'run_hash': run_hash
        }

        # Detect and load existing checkpoints
        existing_checkpoints = {}

        # First, try to load salvaged checkpoints (converted from positional to content-based keys)
        salvaged_dir = Path(self.embeddings_dir) / "cross_encoder_checkpoints" / "salvaged"
        if salvaged_dir.exists():
            # Salvaged files have pattern: ce_worker{id}_{model_name}_{total_pairs}_{hash}.parquet
            # Only load salvaged files matching this run's hash to avoid mixing different datasets
            safe_model_name = cross_encoder_model.replace("/", "_").replace("-", "_")
            salvaged_pattern = f"ce_worker*_{safe_model_name}_total{len(similarity_df)}_{run_hash}.parquet"
            salvaged_files = sorted(salvaged_dir.glob(salvaged_pattern))
            if salvaged_files:
                logger.info(f"Found {len(salvaged_files)} salvaged checkpoints in {salvaged_dir}")

                # Accumulate DataFrames in a list (don't overwrite!)
                salvaged_dfs = []
                total_rows_loaded = 0

                for salvaged_file in salvaged_files:
                    try:
                        df = pd.read_parquet(salvaged_file)

                        # Validate schema
                        required_cols = ['app_text', 'onet_task_id', 'cross_encoder_score']
                        if not all(col in df.columns for col in required_cols):
                            logger.warning(f"  Skipping {salvaged_file.name}: missing required columns")
                            continue

                        # Validate no NaN/corruption
                        if df['app_text'].isna().any() or df['onet_task_id'].isna().any() or df['cross_encoder_score'].isna().any():
                            logger.warning(f"  Skipping {salvaged_file.name}: contains NaN values")
                            continue

                        # Validate no empty strings
                        if (df['app_text'] == '').any() or (df['onet_task_id'] == '').any():
                            logger.warning(f"  Skipping {salvaged_file.name}: contains empty strings")
                            continue

                        salvaged_dfs.append(df)
                        total_rows_loaded += len(df)
                        logger.info(f"  ✓ Loaded {salvaged_file.name}: {len(df):,} rows")

                    except Exception as e:
                        logger.warning(f"Failed to load salvaged checkpoint {salvaged_file.name}: {e}")

                # Concatenate all salvaged DataFrames
                if salvaged_dfs:
                    logger.info(f"\nMerging {len(salvaged_dfs)} salvaged checkpoint files...")
                    logger.info(f"  Total rows loaded: {total_rows_loaded:,}")

                    merged_df = pd.concat(salvaged_dfs, ignore_index=True)
                    logger.info(f"  After concatenation: {len(merged_df):,} rows")

                    # Deduplicate by content keys
                    before_dedup = len(merged_df)
                    merged_df = merged_df.drop_duplicates(
                        subset=['app_text', 'onet_task_id'],
                        keep='last'
                    )
                    after_dedup = len(merged_df)
                    duplicates_removed = before_dedup - after_dedup

                    if duplicates_removed > 0:
                        logger.info(f"  Removed {duplicates_removed:,} duplicate pairs")
                    logger.info(f"  Final unique pairs: {after_dedup:,}")

                    # Store the merged, deduplicated DataFrame
                    existing_checkpoints['salvaged'] = merged_df
                    logger.info(f"✓ Successfully loaded salvaged checkpoints: {after_dedup:,} unique pairs")

        # Then, try to load chunk-based checkpoints (only if no salvaged checkpoints found)
        if 'salvaged' not in existing_checkpoints:
            checkpoint_dir = Path(self.embeddings_dir) / "cross_encoder_checkpoints"
            if checkpoint_dir.exists():
                # Look for content-based chunk files matching pattern: ce_chunk_*_{model}_{hash}.parquet
                safe_model_name = cross_encoder_model.replace("/", "_").replace("-", "_")
                chunk_pattern = f"ce_chunk_*_{safe_model_name}_total{len(similarity_df)}_{run_hash}.parquet"
                chunk_files = sorted(checkpoint_dir.glob(chunk_pattern))

                for chunk_file in chunk_files:
                    try:
                        chunk_df = pd.read_parquet(chunk_file)
                        # Validate schema
                        required_cols = ['app_text', 'onet_task_id', 'cross_encoder_score']
                        if all(col in chunk_df.columns for col in required_cols):
                            existing_checkpoints[chunk_file.name] = chunk_df
                            logger.debug(f"Loaded chunk checkpoint {chunk_file.name}: {len(chunk_df)} pairs")
                    except Exception as e:
                        logger.warning(f"Could not load chunk checkpoint {chunk_file.name}: {e}")

        checkpoint_scores = None
        if existing_checkpoints:
            logger.info(f"Found {len(existing_checkpoints)} valid checkpoints for this run")
            all_checkpoint_dfs = list(existing_checkpoints.values())
            merged_checkpoints = pd.concat(all_checkpoint_dfs, ignore_index=True)

            # Check if we have content-based keys (new format) or positional indices (old format)
            if 'app_text' in merged_checkpoints.columns and 'onet_task_id' in merged_checkpoints.columns:
                # New format: content-based keys
                logger.info("Loading content-based checkpoint format")
                merged_checkpoints = merged_checkpoints.drop_duplicates(subset=['app_text', 'onet_task_id'], keep='last')
                checkpoint_scores = {
                    (row['app_text'], row['onet_task_id']): row['cross_encoder_score']
                    for _, row in merged_checkpoints.iterrows()
                }
            else:
                # Old format: positional indices (still supported for backward compatibility)
                logger.info("Loading positional index checkpoint format (legacy)")
                merged_checkpoints = merged_checkpoints.drop_duplicates(subset=['global_index'], keep='last')
                merged_checkpoints = merged_checkpoints.sort_values('global_index')
                checkpoint_scores = dict(zip(merged_checkpoints['global_index'],
                                            merged_checkpoints['cross_encoder_score']))

            logger.info(f"Loaded {len(checkpoint_scores)} pre-computed scores from checkpoints")
            coverage_pct = (len(checkpoint_scores) / len(similarity_df)) * 100
            logger.info(f"Checkpoint coverage: {len(checkpoint_scores):,}/{len(similarity_df):,} ({coverage_pct:.1f}%)")

            # If checkpoints cover everything, skip computation
            if len(checkpoint_scores) == len(similarity_df):
                logger.info("All scores found in checkpoints - skipping computation")

                # Check if content-based or positional keys
                is_content_based = isinstance(next(iter(checkpoint_scores.keys())), tuple)

                if is_content_based:
                    # Content-based: look up by (app_text, onet_task_id)
                    cross_encoder_scores = [
                        checkpoint_scores.get((row['app_text'], int(row['onet_task_id'])))
                        for _, row in similarity_df.iterrows()
                    ]
                else:
                    # Positional: look up by index
                    cross_encoder_scores = [checkpoint_scores.get(i) for i in range(len(similarity_df))]

                # Verify no missing
                if any(score is None for score in cross_encoder_scores):
                    logger.warning("Checkpoint coverage incomplete despite count match - recomputing")
                    checkpoint_scores = None
                else:
                    # Skip parallel processing, use checkpoint scores directly
                    cached_scores = True  # Use as flag to skip processing

        # Compute scores if not cached or cache incomplete
        if cached_scores is None:
            logger.info("Computing cross-encoder scores in parallel...")

            # Identify remaining work (exclude checkpointed pairs for flexible worker reassignment)
            total_pairs = len(similarity_df)
            if checkpoint_scores is not None:
                # Filter to only pairs NOT in checkpoint (works with both content-based and positional keys)
                is_content_based = isinstance(next(iter(checkpoint_scores.keys())), tuple)

                if is_content_based:
                    # Content-based keys: (app_text, onet_task_id)
                    # Ensure onet_task_id is consistent type (convert to int if needed)
                    completed_pairs = set()
                    for (app_text, task_id), score in checkpoint_scores.items():
                        # Normalize task_id to int for consistent matching
                        task_id_normalized = int(task_id) if not isinstance(task_id, int) else task_id
                        completed_pairs.add((app_text, task_id_normalized))

                    remaining_mask = [
                        (row['app_text'], int(row['onet_task_id'])) not in completed_pairs
                        for _, row in similarity_df.iterrows()
                    ]
                else:
                    # Positional indices (legacy format)
                    completed_indices = set(checkpoint_scores.keys())
                    remaining_mask = [i not in completed_indices for i in range(total_pairs)]

                # Build remaining_df with original indices preserved
                remaining_indices = np.where(remaining_mask)[0]
                remaining_df = similarity_df.iloc[remaining_indices].copy()
                remaining_df['_original_index'] = remaining_indices
                remaining_count = len(remaining_df)

                # Log comprehensive resume summary
                cached_count = len(checkpoint_scores)
                cached_pct = cached_count / total_pairs * 100
                remaining_pct = remaining_count / total_pairs * 100
                est_per_worker = remaining_count / num_workers
                logger.info("=" * 70)
                logger.info("RESUME FROM CHECKPOINT")
                logger.info("=" * 70)
                logger.info(f"Total pairs in dataset: {total_pairs:,}")
                logger.info(f"Cached from previous runs: {cached_count:,} ({cached_pct:.1f}%)")
                logger.info(f"Remaining to process: {remaining_count:,} ({remaining_pct:.1f}%)")
                logger.info(f"Workers for this run: {num_workers}")
                logger.info(f"Estimated pairs per worker: ~{est_per_worker:,.0f}")
                logger.info("=" * 70)
            else:
                remaining_df = similarity_df.copy()
                remaining_df['_original_index'] = np.arange(len(similarity_df))
                remaining_count = total_pairs
                logger.info(f"No checkpoints found - processing all {total_pairs:,} pairs with {num_workers} workers")

            # Split remaining work into chunks (allows different worker count on resume)
            chunk_size = (remaining_count + num_workers - 1) // num_workers

            chunks = []
            for i in range(num_workers):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, remaining_count)
                if start_idx < remaining_count:
                    chunk_df = remaining_df.iloc[start_idx:end_idx].copy()

                    # Get original indices for content-based checkpoint naming
                    original_indices = chunk_df['_original_index'].values
                    chunk_original_start = int(original_indices.min())
                    chunk_original_end = int(original_indices.max())

                    # Create checkpoint config with content-based naming
                    checkpoint_config = {
                        'enabled': getattr(self, 'checkpoint_enabled', True),
                        'interval': getattr(self, 'checkpoint_interval', 100000),
                        'checkpoint_path': self._get_ce_checkpoint_path(
                            cross_encoder_model, total_pairs, run_hash,
                            start_idx=chunk_original_start, end_idx=chunk_original_end
                        ),
                        'run_metadata': run_metadata
                    }

                    # Add worker row range for logging
                    worker_row_range = (start_idx, end_idx - 1)
                    chunks.append((i, chunk_df, cross_encoder_model, batch_size, self.device, checkpoint_config, worker_row_range))

                    # Log worker assignment within remaining work scope
                    logger.info(f"Worker {i}: Assigned rows {start_idx:,}-{end_idx-1:,} of {remaining_count:,} remaining pairs (original indices {chunk_original_start:,}-{chunk_original_end:,})")

            logger.info(f"Split {remaining_count:,} remaining pairs into {len(chunks)} chunks")

            # Process chunks in parallel with progress tracking
            worker_func = partial(_cross_encoder_worker_function)

            try:
                # Create a queue for progress updates
                progress_queue = mp.Manager().Queue()

                # Add progress_queue to each chunk
                chunks_with_queue = [(i, df, model, batch, dev, ckpt_cfg, row_range, progress_queue)
                                    for i, df, model, batch, dev, ckpt_cfg, row_range in chunks]

                with mp.Pool(processes=num_workers) as pool:
                    # Start async processing
                    async_result = pool.starmap_async(worker_func, chunks_with_queue)

                    # Track progress from all workers
                    from tqdm import tqdm
                    import queue
                    worker_progress = {}

                    # Initialize progress bar showing remaining work progress
                    checkpoint_coverage = len(checkpoint_scores) if checkpoint_scores is not None else 0
                    checkpoint_pct = checkpoint_coverage / total_pairs * 100 if total_pairs > 0 else 0

                    # Use remaining_count for progress tracking within this run
                    desc = f"Cross-encoder ({checkpoint_pct:.1f}% cached, processing {remaining_count:,} remaining)"
                    pbar = tqdm(total=remaining_count, desc=desc, unit=" pairs", disable=False)

                    last_total = 0  # Track progress within this run (0 to remaining_count)
                    while not async_result.ready():
                        try:
                            # Non-blocking get to avoid hanging
                            update = progress_queue.get(timeout=0.5)
                            worker_id = update['worker_id']

                            # Track per-worker progress
                            if worker_id not in worker_progress:
                                worker_progress[worker_id] = 0

                            curr_progress = update['pairs_processed']
                            worker_progress[worker_id] = curr_progress

                            # Update main progress bar
                            total_processed = sum(worker_progress.values())
                            increment = total_processed - last_total
                            if increment > 0:
                                pbar.update(increment)
                                last_total = total_processed

                            # Show active workers in postfix
                            active_workers = [f"W{wid}: {wp}" for wid, wp in sorted(worker_progress.items())]
                            pbar.set_postfix_str(" | ".join(active_workers[-2:]))  # Show last 2 workers
                        except queue.Empty:
                            pass
                        except Exception as e:
                            logger.debug(f"Progress update error: {e}")

                    # Final update for any remaining progress
                    pbar.close()

                    results = async_result.get()

                # Merge results from workers with checkpoint data
                cross_encoder_scores = [None] * total_pairs

                # First, populate from checkpoints (handle both content-based and positional keys)
                if checkpoint_scores is not None:
                    is_content_based = isinstance(next(iter(checkpoint_scores.keys())), tuple)

                    if is_content_based:
                        # Content-based keys: iterate through similarity_df and look up
                        for i, (_, row) in enumerate(similarity_df.iterrows()):
                            key = (row['app_text'], row['onet_task_id'])
                            if key in checkpoint_scores:
                                cross_encoder_scores[i] = checkpoint_scores[key]
                    else:
                        # Positional indices (legacy format)
                        for global_idx, score in checkpoint_scores.items():
                            if 0 <= global_idx < total_pairs:
                                cross_encoder_scores[global_idx] = score

                # Then, overlay fresh worker results (overwrites checkpoint data if recomputed)
                for worker_id, global_indices, chunk_scores in results:
                    for idx, score in zip(global_indices, chunk_scores):
                        cross_encoder_scores[idx] = score
                    logger.info(f"Worker {worker_id} completed: {len(chunk_scores)} scores")

                # Verify completeness
                missing_count = sum(1 for score in cross_encoder_scores if score is None)
                if missing_count > 0:
                    logger.error(f"Missing {missing_count} scores after parallel processing!")
                    logger.info("Falling back to serial processing...")
                    return self.validate_with_cross_encoder(
                        similarity_df=similarity_df,
                        cross_encoder_model=cross_encoder_model,
                        threshold=threshold,
                        batch_size=batch_size,
                        use_cache=False
                    )

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

            # Clean up checkpoints after successful completion
            if checkpoint_scores is not None or existing_checkpoints:
                logger.info("Cleaning up cross-encoder checkpoints...")
                self._cleanup_ce_checkpoints(checkpoint_dir, run_hash)

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
                         include_soc_15: bool = False,
                         skip_cross_encoder: bool = False,
                         cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                         cross_encoder_batch_size: int = 256,
                         bge_percentiles: Optional[List[float]] = None,
                         ce_thresholds: Optional[List[float]] = None,
                         task_type: str = 'both',
                         per_task_mode: bool = False,
                         onet_version: Optional[str] = None,
                         dedup_chunk_size: int = 10_000_000) -> Tuple[pd.DataFrame, dict]:
        """
        Run the complete pipeline from Step 3 output to O*NET similarity matches.

        Args:
            step3_file_path: Path to Step 3 filtered tasks CSV
            onet_file_path: Path to O*NET Task Statements Excel file
            output_dir: Directory to save output files
            enable_fuzzy_dedup: Whether to enable fuzzy deduplication
            use_onet_cache: Whether to use cached O*NET embeddings
            use_apps_cache: Whether to use cached application embeddings
            onet_version: O*NET version string (e.g., "20" or "30") for filename suffix

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
        
        # Load O*NET tasks - always load FULL set first (for embedding cache compatibility)
        # Then filter after embedding to reuse cached embeddings
        onet_df_full = self.load_onet_tasks(onet_file_path, include_soc_15=True, filter_supplements=False)

        # Phase B: Embed texts
        unique_app_texts = dedup_df['app_text'].unique().tolist()
        apps_embeddings = self.embed_texts(
            unique_app_texts,
            "application strings",
            cache_prefix="apps",
            use_cache=use_apps_cache,
            text_type="apps"
        )

        # Embed full O*NET task set (to reuse cache)
        onet_tasks_full = onet_df_full['Task'].tolist()
        onet_embeddings_full = self.embed_texts(
            onet_tasks_full,
            "O*NET tasks",
            cache_prefix="onet",
            use_cache=use_onet_cache,
            text_type="tasks"
        )

        # Now filter O*NET tasks and embeddings if SOC 15 should be excluded
        if not include_soc_15:
            # Filter out SOC 15 tasks
            soc15_mask = ~onet_df_full['O*NET-SOC Code'].str.startswith('15-', na=False)
            onet_df = onet_df_full[soc15_mask].copy().reset_index(drop=True)
            onet_embeddings = onet_embeddings_full[soc15_mask]

            excluded_count = len(onet_df_full) - len(onet_df)
            logger.info(f"Filtered embeddings: excluded {excluded_count} SOC 15 tasks, keeping {len(onet_df)} tasks")
        else:
            onet_df = onet_df_full
            onet_embeddings = onet_embeddings_full
            logger.info(f"Using all {len(onet_df)} O*NET tasks (including SOC 15)")
        
        # Phase C: Compute similarities and filter
        results_df, all_similarities_df = self.compute_similarities_and_filter(
            dedup_df, apps_embeddings, onet_df, onet_embeddings,
            use_cache=use_similarities_cache, per_task_mode=per_task_mode, bge_percentiles=bge_percentiles
        )
        
        # Phase D: Validate results
        validation_metrics = self.validate_results(results_df)
        
        # Create deduplicated similarity scores and job mapping
        logger.info("Creating deduplicated similarity scores and job mapping")

        # Get max percentile for filtering
        max_bge_percentile = max(bge_percentiles)  # e.g., 20

        # Phase D: Pre-filter and deduplicate (conditional based on mode)
        if per_task_mode:
            # Per-task mode: Skip deduplication for memory efficiency
            logger.info("Per-task mode: Skipping deduplication step for memory efficiency")
            deduplicated_similarities = results_df.copy()
        else:
            # Global mode: Apply global threshold and deduplicate
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

            # Decide whether to use chunked deduplication based on dataset size
            if len(filtered_results_df) > dedup_chunk_size:
                # Large dataset: use memory-efficient chunked deduplication
                logger.info(f"Dataset size ({len(filtered_results_df):,}) exceeds chunk size ({dedup_chunk_size:,})")
                logger.info("Using memory-efficient chunked deduplication")
                deduplicated_similarities = self._deduplicate_in_chunks(
                    filtered_df=filtered_results_df,
                    chunk_size=dedup_chunk_size,
                    output_dir=output_dir
                )
            else:
                # Small dataset: use standard in-memory deduplication
                logger.info(f"Dataset size ({len(filtered_results_df):,}) within limits, using standard deduplication")
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

        # Add ai_app_id: create a stable hash-based ID for each unique AI application text
        # Keep this lightweight and deterministic so downstream stages can join on it
        import hashlib
        deduplicated_similarities['ai_app_id'] = deduplicated_similarities['app_text'].apply(
            lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[:8].upper()  # 8-char uppercase hex
        )

        logger.info(f"Deduplicated to {len(deduplicated_similarities)} unique (app_text, onet_task_id) pairs")

        # Sort both DataFrames by similarity in descending order
        results_df = results_df.sort_values('similarity', ascending=False)
        deduplicated_similarities = deduplicated_similarities.sort_values('similarity', ascending=False)

        # Clean up massive results_df immediately - we only needed it for sorting
        logger.info("Freeing memory: deleting large results_df")
        del results_df
        gc.collect()

        # Create job mapping (app_text -> list of job_uids) with temporal information
        # TEMPORARY: Per-task mode skips this to avoid memory issues on current system
        # TODO: Optimize job_mapping generation for per-task mode so it doesn't exhaust memory
        if not per_task_mode:
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
            # Attach ai_app_id so downstream stages (5–7) can key on a stable identifier
            job_mapping['ai_app_id'] = job_mapping['app_text'].apply(
                lambda x: hashlib.md5(x.encode('utf-8')).hexdigest()[:8].upper()
            )
            job_mapping = job_mapping[['app_text', 'ai_app_id', 'job_uids', 'num_jobs', 'first_occurrence_tst_created']].copy()

            logger.info(f"Created job mapping for {len(job_mapping)} unique applications (temporal ordering preserved)")

            # Clean up filtered_results_df - we only needed it for job mapping
            logger.info("Freeing memory: deleting filtered_results_df")
            del filtered_results_df, job_mapping_agg
            gc.collect()
        else:
            logger.info("Per-task mode: Skipping job_mapping generation (memory optimization)")
            job_mapping = None

        logger.info(f"Deduplicated to {len(deduplicated_similarities)} unique app-task pairs")

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
            # Filter to pairs for cross-encoder validation (conditional based on mode)
            if per_task_mode:
                # Per-task mode: Use per-task filtering method
                ce_input_df = self._filter_to_top_per_task_percentile(
                    deduplicated_similarities.copy(),
                    percentile=max_bge_percentile
                )
            else:
                # Global mode: Use global threshold
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

        # Merge Task Type column for core vs all tasks split (do this BEFORE percentile calculations)
        unified_matches = unified_matches.merge(
            onet_df[['task_id', 'Task Type']].rename(columns={'Task Type': 'task_type'}),
            left_on='onet_task_id', right_on='task_id', how='left'
        ).drop(columns=['task_id'])
        logger.info(f"Added Task Type column for core/all task differentiation")

        # Process task types based on CLI argument
        # Split logic: Process core tasks and/or all tasks separately with their own percentile calculations
        job_mapping_file = None
        final_return_df = None

        # For 'core' mode: we'll delete unified_matches after processing core tasks
        # For 'all' mode: we'll delete unified_matches after processing all tasks
        # For 'both' mode: we keep unified_matches for the all tasks processing

        if task_type in ['core', 'both']:
            logger.info("="*80)
            logger.info("PROCESSING CORE TASKS")
            logger.info("="*80)

            # ADD VALIDATION: ensure global percentile thresholds were computed
            if (not hasattr(self, 'global_percentile_thresholds')) or (self.global_percentile_thresholds is None):
                raise ValueError(
                    "Global percentile thresholds not available. "
                    "This should not happen as all similarities are always saved. Contact developers."
                )

            # Filter to core tasks BEFORE calculating percentiles
            unified_matches_core = unified_matches[unified_matches['task_type'] == 'Core'].copy()
            logger.info(f"Filtered to {len(unified_matches_core):,} Core task matches (from {len(unified_matches):,} total)")

            # Use GLOBAL thresholds calculated in Phase C
            # Create boolean columns for BGE percentiles (conditional based on mode)
            if per_task_mode:
                # Per-task mode: Use per-task rank percentiles
                logger.info("Using PER-TASK percentile rankings for CORE TASKS")
                for p in bge_percentiles:
                    # Format percentile name: handle both integers (20) and floats (0.1)
                    if isinstance(p, int) or p == int(p):
                        pct_name = f'pct_{int(p):02d}'
                    else:
                        # For floats like 0.1, format as pct_0p1
                        pct_str = f'{p:.1f}'.replace('.', 'p')
                        pct_name = f'pct_{pct_str}'
                    pct_threshold = p / 100.0
                    unified_matches_core[pct_name] = unified_matches_core['per_task_rank_pct'] <= pct_threshold
                    match_count = unified_matches_core[pct_name].sum()
                    logger.info(f"  {pct_name}: {match_count:,} matches (per-task top {p}%)")
            else:
                # Global mode: Use global thresholds
                logger.info("Using GLOBAL percentile thresholds for CORE TASKS (calculated on all pairs)")
                bge_thresholds_core = {}
                for p in bge_percentiles:
                    # Format percentile name: handle both integers (20) and floats (0.1)
                    if isinstance(p, int) or p == int(p):
                        pct_name = f'pct_{int(p):02d}'
                    else:
                        # For floats like 0.1, format as pct_0p1
                        pct_str = f'{p:.1f}'.replace('.', 'p')
                        pct_name = f'pct_{pct_str}'
                    bge_thresholds_core[pct_name] = self.global_percentile_thresholds[p]
                    logger.info(f"  {p}% percentile (GLOBAL): {self.global_percentile_thresholds[p]:.4f}")

                # Add BGE boolean columns for core tasks
                for pct_name, threshold in bge_thresholds_core.items():
                    unified_matches_core[pct_name] = unified_matches_core['similarity'] >= threshold
                    match_count = unified_matches_core[pct_name].sum()
                    logger.info(f"  {pct_name}: {match_count:,} matches")

            # Create boolean columns for all cross-encoder thresholds (with NaN handling)
            logger.info("Computing cross-encoder threshold columns for CORE TASKS")
            for ce_thresh in ce_thresholds:
                col_name = f'ce_{ce_thresh:.1f}'
                if ce_thresh == 0.0:
                    # ce_0.0 means "no CE filtering" - always True to act as a pass-through
                    unified_matches_core[col_name] = True
                    match_count = len(unified_matches_core)
                    logger.info(f"  {col_name}: {match_count:,} matches (no CE filtering)")
                else:
                    # Apply threshold only where CE score exists (NaN will become False)
                    unified_matches_core[col_name] = (
                        (unified_matches_core['cross_encoder_score'] >= ce_thresh) &
                        unified_matches_core['cross_encoder_score'].notna()
                    )
                    match_count = unified_matches_core[col_name].sum()
                    logger.info(f"  {col_name}: {match_count:,} matches")

            # Save core tasks output with comprehensive naming including model, percentiles, and O*NET version
            if self.use_openai_embeddings:
                model_suffix = "_openai"
            else:
                # Extract model abbreviation from model name (e.g., "BAAI/bge-large-en-v1.5" -> "bge")
                model_abbr = self.model_name.split("/")[-1].split("-")[0]
                model_suffix = f"_{model_abbr}"

            # Build comprehensive filename with percentiles and O*NET version
            percentile_str = "_".join(str(int(p)) if isinstance(p, (int, float)) and p == int(p) else str(p)
                                      for p in sorted(bge_percentiles, reverse=True))
            ce_percentile_str = "_".join(f"{c:.1f}".replace(".", "p") for c in sorted(ce_thresholds, reverse=True) if c > 0.0)
            onet_suffix = f"_onet{onet_version}" if onet_version else ""

            core_output = os.path.join(output_dir,
                f"task_exposure_matches_all_thresholds{model_suffix}_bge{percentile_str}_ce{ce_percentile_str}{onet_suffix}_core.parquet")
            unified_matches_core.to_parquet(core_output, index=False, compression='snappy')
            logger.info(f"Saved CORE TASKS: {core_output} ({len(unified_matches_core):,} rows)")

            # Task suffix
            task_suffix = f"_{task_type}"
            # Save job mapping (shared for both)
            if task_type == 'core':
                job_mapping_file = os.path.join(output_dir,
                    f"job_app_mapping{model_suffix}_bge{percentile_str}_ce{ce_percentile_str}{onet_suffix}{task_suffix}.parquet")
                logger.info("Saving job mapping...")
                gc.collect()  # Force GC before large job_mapping write
                job_mapping.to_parquet(job_mapping_file, index=False, compression='snappy')
                logger.info(f"Saved job mapping: {job_mapping_file}")
                final_return_df = unified_matches_core
                # COMPREHENSIVE cleanup: delete ALL intermediate dataframes not needed for return
                # (but NOT unified_matches_core - we return it as final_return_df)
                logger.info("Freeing memory: comprehensive cleanup for core-only mode")
                del unified_matches, job_mapping, deduplicated_similarities
                # Delete cross-encoder intermediates if they exist
                if 'ce_input_df' in locals():
                    del ce_input_df
                if 'cross_encoder_validated_df' in locals():
                    del cross_encoder_validated_df
                if 'ce_scores' in locals():
                    del ce_scores
                # Set all_similarities_df to None instead of deleting (so it doesn't cause UnboundLocalError later)
                # This frees the massive dataframe (200M+ rows) from memory
                if all_similarities_df is not None:
                    all_similarities_df = None
                gc.collect()
                logger.info("Memory cleanup completed for core-only mode - freed all intermediate dataframes")

            if task_type == 'both':
                logger.info("Freeing memory: deleting unified_matches_core (keeping unified_matches for all tasks processing)")
                del unified_matches_core
                gc.collect()

        if task_type in ['all', 'both']:
            logger.info("="*80)
            logger.info("PROCESSING ALL TASKS")
            logger.info("="*80)

            logger.info(f"Using all {len(unified_matches):,} task matches")

            # Use GLOBAL thresholds calculated in Phase C
            logger.info("Using GLOBAL percentile thresholds for ALL TASKS (calculated on all pairs)")
            bge_thresholds_all = {}
            for p in bge_percentiles:
                # Format percentile name: handle both integers (20) and floats (0.1)
                if isinstance(p, int) or p == int(p):
                    pct_name = f'pct_{int(p):02d}'
                else:
                    # For floats like 0.1, format as pct_0p1
                    pct_str = f'{p:.1f}'.replace('.', 'p')
                    pct_name = f'pct_{pct_str}'
                bge_thresholds_all[pct_name] = self.global_percentile_thresholds[p]
                logger.info(f"  {p}% percentile (GLOBAL): {self.global_percentile_thresholds[p]:.4f}")

            # For 'both' mode, we add columns only to a copy; for 'all' mode, we add to unified_matches directly
            if task_type == 'both':
                unified_matches_all = unified_matches.copy()
            else:
                unified_matches_all = unified_matches

            # Add BGE boolean columns for all tasks
            for pct_name, threshold in bge_thresholds_all.items():
                unified_matches_all[pct_name] = unified_matches_all['similarity'] >= threshold
                match_count = unified_matches_all[pct_name].sum()
                logger.info(f"  {pct_name}: {match_count:,} matches")

            # Create boolean columns for all cross-encoder thresholds (with NaN handling)
            logger.info("Computing cross-encoder threshold columns for ALL TASKS")
            for ce_thresh in ce_thresholds:
                col_name = f'ce_{ce_thresh:.1f}'
                if ce_thresh == 0.0:
                    # ce_0.0 means "no CE filtering" - always True to act as a pass-through
                    unified_matches_all[col_name] = True
                    match_count = len(unified_matches_all)
                    logger.info(f"  {col_name}: {match_count:,} matches (no CE filtering)")
                else:
                    # Apply threshold only where CE score exists (NaN will become False)
                    unified_matches_all[col_name] = (
                        (unified_matches_all['cross_encoder_score'] >= ce_thresh) &
                        unified_matches_all['cross_encoder_score'].notna()
                    )
                    match_count = unified_matches_all[col_name].sum()
                    logger.info(f"  {col_name}: {match_count:,} matches")

            # Save all tasks output with comprehensive naming including model, percentiles, and O*NET version
            if self.use_openai_embeddings:
                model_suffix = "_openai"
            else:
                # Extract model abbreviation from model name (e.g., "BAAI/bge-large-en-v1.5" -> "bge")
                model_abbr = self.model_name.split("/")[-1].split("-")[0]
                model_suffix = f"_{model_abbr}"

            # Build comprehensive filename with percentiles and O*NET version
            percentile_str = "_".join(str(int(p)) if isinstance(p, (int, float)) and p == int(p) else str(p)
                                      for p in sorted(bge_percentiles, reverse=True))
            ce_percentile_str = "_".join(f"{c:.1f}".replace(".", "p") for c in sorted(ce_thresholds, reverse=True) if c > 0.0)
            onet_suffix = f"_onet{onet_version}" if onet_version else ""

            all_output = os.path.join(output_dir,
                f"task_exposure_matches_all_thresholds{model_suffix}_bge{percentile_str}_ce{ce_percentile_str}{onet_suffix}_all_tasks.parquet")

            # Force garbage collection before large parquet write
            logger.info("Saving ALL TASKS parquet file...")
            gc.collect()
            unified_matches_all.to_parquet(all_output, index=False, compression='snappy')
            logger.info(f"Saved ALL TASKS: {all_output} ({len(unified_matches_all):,} rows)")

            # Save job mapping if this is the only run
            if task_type == 'all':
                job_mapping_file = os.path.join(output_dir,
                    f"job_app_mapping{model_suffix}_bge{percentile_str}_ce{ce_percentile_str}{onet_suffix}.parquet")
                job_mapping.to_parquet(job_mapping_file, index=False, compression='snappy')
                logger.info(f"Saved job mapping: {job_mapping_file}")
                final_return_df = unified_matches_all
            elif task_type == 'both':
                # Save job mapping on second run (only save once)
                if job_mapping_file is None:
                    job_mapping_file = os.path.join(output_dir,
                        f"job_app_mapping{model_suffix}_bge{percentile_str}_ce{ce_percentile_str}{onet_suffix}.parquet")
                    job_mapping.to_parquet(job_mapping_file, index=False, compression='snappy')
                    logger.info(f"Saved job mapping: {job_mapping_file}")

            # Clean up all_tasks data after saving
            if task_type == 'both':
                del unified_matches_all
                gc.collect()

        # 3. Parquet format for ALL similarity scores (comprehensive analysis)
        # Skip for core-only mode - all_similarities_df already deleted in core cleanup above
        # For all/both modes, process the comprehensive similarity scores
        if all_similarities_df is not None and task_type != 'core':
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

        return final_return_df, validation_metrics


def _cross_encoder_worker_function(worker_id: int,
                                   chunk_df: pd.DataFrame,
                                   model_name: str,
                                   batch_size: int,
                                   device: str,
                                   checkpoint_config: Optional[dict] = None,
                                   worker_row_range: tuple = None,
                                   progress_queue=None) -> Tuple[int, List[int], List[float]]:
    """
    Worker function for parallel cross-encoder processing with checkpointing support.
    Must be defined at module level for multiprocessing.

    Args:
        worker_id: Worker identifier
        chunk_df: Chunk of similarity pairs to process (with original DataFrame indices preserved)
        model_name: Cross-encoder model name
        batch_size: Batch size for inference
        device: Device to use (mps, cuda, cpu)
        checkpoint_config: Optional checkpoint configuration dict
        worker_row_range: Tuple of (start_row, end_row) within remaining work for logging
        progress_queue: Optional queue for progress tracking

    Returns:
        Tuple of (worker_id, global_indices, list of scores)
    """
    import torch
    from sentence_transformers import CrossEncoder

    logger.info(f"Worker {worker_id} started: {len(chunk_df)} pairs on device {device}")

    # Use actual DataFrame indices from chunk (these map back to original similarity_df)
    global_indices = chunk_df.index.tolist()

    # Log row assignment within remaining work scope
    if worker_row_range is not None:
        start_row, end_row = worker_row_range
        logger.info(f"Worker {worker_id}: Processing rows {start_row:,}-{end_row:,} of remaining work")
    else:
        logger.info(f"Worker {worker_id}: Processing {len(global_indices):,} pairs")

    # Extract checkpoint settings
    enable_checkpointing = checkpoint_config is not None and checkpoint_config.get('enabled', False)
    checkpoint_interval = checkpoint_config.get('interval', 100000) if enable_checkpointing else None
    checkpoint_path = checkpoint_config.get('checkpoint_path') if enable_checkpointing else None
    run_metadata = checkpoint_config.get('run_metadata', {}) if enable_checkpointing else {}

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

    # Use actual DataFrame indices from chunk (these map back to original similarity_df)
    global_indices = chunk_df.index.tolist()

    # Run inference with checkpointing
    scores = []
    num_batches = (len(pairs) + batch_size - 1) // batch_size
    pairs_processed = 0
    last_checkpoint_at = 0

    for batch_idx, i in enumerate(range(0, len(pairs), batch_size)):
        batch_pairs = pairs[i:i + batch_size]
        batch_scores = cross_encoder.predict(batch_pairs, show_progress_bar=False)

        if isinstance(batch_scores, (int, float)):
            batch_scores = [batch_scores]
        elif hasattr(batch_scores, 'tolist'):
            batch_scores = batch_scores.tolist()

        scores.extend(batch_scores)
        pairs_processed = len(scores)

        # Save checkpoint if interval reached
        if enable_checkpointing and checkpoint_path is not None:
            if pairs_processed - last_checkpoint_at >= checkpoint_interval:
                try:
                    # Load existing checkpoint if it exists to preserve prior work
                    existing_df = None
                    if checkpoint_path.exists():
                        try:
                            existing_df = pd.read_parquet(checkpoint_path)
                        except Exception as e:
                            logger.warning(f"Worker {worker_id}: Could not load existing checkpoint: {e}")

                    # Create new data for this checkpoint interval (ALL scores so far to avoid index overlap)
                    # Use content-based keys for robust checkpoint resumption
                    new_checkpoint_df = pd.DataFrame({
                        'app_text': chunk_df['app_text'].iloc[:len(scores)].values,
                        'onet_task_id': chunk_df['onet_task_id'].iloc[:len(scores)].values,
                        'cross_encoder_score': scores
                    })
                    # Note: Exclude _original_index column from checkpoint - it's only for worker assignment

                    # Append to existing or use new
                    if existing_df is not None and len(existing_df) > 0:
                        checkpoint_df = pd.concat([existing_df, new_checkpoint_df], ignore_index=True)
                        # Deduplicate on content keys to prevent data loss
                        checkpoint_df = checkpoint_df.drop_duplicates(subset=['app_text', 'onet_task_id'], keep='last')
                    else:
                        checkpoint_df = new_checkpoint_df

                    checkpoint_df.attrs.update({
                        'worker_id': worker_id,
                        'pairs_completed': pairs_processed,
                        'total_pairs_in_chunk': len(pairs),
                        'created_at': datetime.now().isoformat(),
                        **run_metadata
                    })

                    checkpoint_df.to_parquet(checkpoint_path, index=False, compression='snappy')
                    last_checkpoint_at = pairs_processed
                    logger.debug(f"Worker {worker_id}: Checkpoint saved at {pairs_processed}/{len(pairs)} pairs (total in file: {len(checkpoint_df)})")

                except Exception as e:
                    logger.warning(f"Worker {worker_id}: Checkpoint save failed: {e}")

        # Send progress update to queue
        if progress_queue is not None:
            progress_queue.put({
                'worker_id': worker_id,
                'batches_completed': batch_idx + 1,
                'total_batches': num_batches,
                'pairs_processed': pairs_processed,
                'total_pairs': len(pairs)
            })

    # Save final checkpoint if enabled
    if enable_checkpointing and checkpoint_path is not None:
        try:
            # Load existing checkpoint if it exists to preserve prior work
            existing_df = None
            if checkpoint_path.exists():
                try:
                    existing_df = pd.read_parquet(checkpoint_path)
                except Exception as e:
                    logger.warning(f"Worker {worker_id}: Could not load existing checkpoint: {e}")

            # Create final checkpoint data using content-based keys
            final_checkpoint_df = pd.DataFrame({
                'app_text': chunk_df['app_text'].values,
                'onet_task_id': chunk_df['onet_task_id'].values,
                'cross_encoder_score': scores
            })
            # Note: Exclude _original_index column from checkpoint - it's only for worker assignment

            # Append to existing or use new
            if existing_df is not None and len(existing_df) > 0:
                checkpoint_df = pd.concat([existing_df, final_checkpoint_df], ignore_index=True)
                checkpoint_df = checkpoint_df.drop_duplicates(subset=['app_text', 'onet_task_id'], keep='last')
            else:
                checkpoint_df = final_checkpoint_df

            checkpoint_df.attrs.update({
                'worker_id': worker_id,
                'pairs_completed': len(checkpoint_df),
                'total_pairs_in_chunk': len(pairs),
                'completed': True,
                'created_at': datetime.now().isoformat(),
                **run_metadata
            })
            checkpoint_df.to_parquet(checkpoint_path, index=False, compression='snappy')
            logger.info(f"Worker {worker_id}: Final checkpoint saved ({len(checkpoint_df)} total scores in file)")
        except Exception as e:
            logger.warning(f"Worker {worker_id}: Final checkpoint save failed: {e}")

    logger.info(f"Worker {worker_id} completed: {len(scores)} scores computed")
    return (worker_id, global_indices, scores)


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

    parser.add_argument("--use-openai-embeddings", action="store_true",
                       help="Load pre-generated OpenAI text-embedding-3-large embeddings from cache (generated by generate_openai_embeddings.py)")

    parser.add_argument("--step3-file", type=str, default=None,
                       help="Path to Step 3 output file (auto-detect latest if not specified)")
    
    parser.add_argument("--onet-file", type=str, default=None,
                       help="Path to O*NET Task Statements file (auto-detect from version if not provided)")

    parser.add_argument("--onet-version", type=int, default=20,
                       help="O*NET version number (e.g., 20, 25, 30). Default: 20")

    parser.add_argument("--output-dir", type=str, default="Data",
                       help="Output directory for results")
    
    parser.add_argument("--embeddings-dir", type=str, default="Data/embeddings",
                       help="Directory for cached embeddings")
    
    parser.add_argument("--clear-cache", type=str, choices=["all", "onet", "apps"], default=None,
                       help="Clear cached embeddings before running")
    
    parser.add_argument("--list-cache", action="store_true",
                       help="List cached embeddings and exit")
    
    parser.add_argument("--include-soc-15", action="store_true",
                       help="Include SOC group 15 (Computer and Mathematical Occupations). Default is to exclude.")
    
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

    parser.add_argument("--checkpoint-interval", type=int, default=10000,
                    help="Save cross-encoder checkpoint every N pairs (default: 10000)")

    parser.add_argument("--no-checkpoint", action="store_true",
                       help="Disable cross-encoder checkpointing (not recommended for large runs)")

    parser.add_argument("--force-device", type=str,
                       choices=["auto", "mps", "cuda", "cpu"],
                       default="auto",
                       help="Force specific device (default: auto-detect)")

    parser.add_argument("--skip-cross-encoder", action="store_true",
                       help="Skip cross-encoder validation (use BGE scores only, saves memory)")

    parser.add_argument("--task-type", type=str,
                       choices=['core', 'all', 'both'],
                       default='both',
                       help="Task type: 'core' (Core only), 'all' (Core+Supp), 'both' (default)")

    parser.add_argument("--per-task", action="store_true",
                       help="Use per-task percentile filtering instead of global filtering. "
                            "In per-task mode, top N%% is computed separately for each O*NET task, "
                            "resulting in variable-sized match sets that better capture task-specific AI exposure.")

    parser.add_argument("--use-faiss", action="store_true",
                       help="Enable FAISS ANN search for top-k apps per task (alternative to exhaustive search). "
                            "FAISS mode retrieves top-k most similar apps for each O*NET task, "
                            "filters by minimum similarity threshold, and treats all remaining pairs as matches "
                            "(no percentile filtering). Output schema differs: no pct_XX columns in FAISS mode.")

    parser.add_argument("--faiss-k", type=int, default=200,
                       help="Number of top-k apps to retrieve per O*NET task in FAISS mode (default: 200). "
                            "Only used when --use-faiss is enabled.")

    parser.add_argument("--faiss-index-type", type=str, default="Flat",
                       choices=["Flat"],
                       help="FAISS index type (default: Flat for exact top-k search). "
                            "Only used when --use-faiss is enabled.")

    parser.add_argument("--sample-percentiles", type=int, default=10000000,
                       help="Sample size for percentile calculation in exhaustive mode (default: 10,000,000). "
                            "Randomly samples this many pairs for percentile computation instead of using all pairs. "
                            "Ignored in FAISS mode (no percentile calculation).")

    parser.add_argument("--dedup-chunk-size", type=int, default=10000000,
                       help="Chunk size for memory-efficient deduplication (default: 10,000,000). "
                            "If filtered results exceed this size, deduplication will process data in chunks "
                            "to avoid memory exhaustion. Lower values use less memory but may be slower. "
                            "Typical range: 5M-20M rows depending on available RAM.")

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

    # Determine O*NET file path
    if args.onet_file:
        onet_file = args.onet_file
    else:
        onet_file = os.path.join(args.output_dir, f"task_statements_{args.onet_version}.xlsx")

    logger.info(f"Using O*NET file: {onet_file}")

    # Validate input files exist
    if not os.path.exists(step3_file):
        logger.error(f"Step 3 file not found: {step3_file}")
        return

    if not os.path.exists(onet_file):
        logger.error(f"O*NET file not found: {onet_file}")
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
            embeddings_dir=args.embeddings_dir,
            use_openai_embeddings=args.use_openai_embeddings,
            use_faiss=args.use_faiss,
            faiss_k=args.faiss_k,
            faiss_index_type=args.faiss_index_type,
            sample_percentiles=args.sample_percentiles
        )
    except Exception as e:
        logger.error(f"Failed to initialize matcher: {e}")
        return

    # Override device if requested (only for BGE embeddings)
    if not args.use_openai_embeddings and args.force_device != "auto":
        logger.info(f"Overriding device detection: {args.force_device}")
        matcher.device = args.force_device
        matcher.model = matcher.model.to(matcher.device)

    # Set num_workers on matcher instance for pipeline access
    matcher.num_workers = args.num_workers

    # Set checkpoint settings on matcher instance
    matcher.checkpoint_enabled = not args.no_checkpoint
    matcher.checkpoint_interval = args.checkpoint_interval

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
            include_soc_15=args.include_soc_15,
            skip_cross_encoder=args.skip_cross_encoder,
            cross_encoder_model=args.cross_encoder_model,
            cross_encoder_batch_size=args.cross_encoder_batch_size,
            bge_percentiles=BGE_PERCENTILES,
            ce_thresholds=CE_THRESHOLDS,
            task_type=args.task_type,
            per_task_mode=args.per_task,
            onet_version=args.onet_version,
            dedup_chunk_size=args.dedup_chunk_size
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

        # Validate task_id to task_text alignment
        print("\n" + "="*60)
        print("VALIDATING TASK_ID TO TASK_TEXT ALIGNMENT")
        print("="*60)
        validate_task_id_alignment(args.output_dir, onet_version=args.onet_version)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def validate_task_id_alignment(output_dir, onet_version=None):
    """
    Validate that each onet_task_id in Stage 4 output has the correct onet_task text.

    For every row: task_text should match what Task ID actually contains in O*NET source.
    If any mismatch is found, raises AssertionError.
    """
    import glob

    logger.info("Validating task_id to task_text alignment...")

    # Find the output parquet file
    output_pattern = os.path.join(output_dir, "task_exposure_matches_all_thresholds*.parquet")
    output_files = glob.glob(output_pattern)

    if not output_files:
        logger.warning(f"No output parquet files found matching {output_pattern}")
        return

    output_file = output_files[0]
    logger.info(f"Loading output file: {output_file}")
    output = pd.read_parquet(output_file)

    # Load O*NET source for validation
    # Get project root by finding the Data directory
    current_dir = os.path.dirname(os.path.abspath(output_dir))
    while current_dir and current_dir != "/":
        if os.path.exists(os.path.join(current_dir, "Data", f"task_statements_{onet_version or '20'}.xlsx")):
            onet_file = os.path.join(current_dir, "Data", f"task_statements_{onet_version or '20'}.xlsx")
            break
        current_dir = os.path.dirname(current_dir)
    else:
        onet_file = os.path.join(os.getcwd(), "Data", f"task_statements_{onet_version or '20'}.xlsx")

    if not os.path.exists(onet_file):
        logger.warning(f"O*NET source file not found: {onet_file}, skipping validation")
        return

    logger.info(f"Loading O*NET source: {onet_file}")
    onet_source = pd.read_excel(onet_file)

    # Build lookup: Task ID -> Task text
    onet_lookup = dict(zip(onet_source['Task ID'].astype(int), onet_source['Task']))

    logger.info(f"Checking {len(output):,} rows for task_id/task_text alignment...")

    mismatches = []
    for idx, row in output.iterrows():
        task_id = int(row['onet_task_id'])
        actual_text = row['onet_task']

        if task_id not in onet_lookup:
            mismatches.append({
                'row': idx,
                'task_id': task_id,
                'error': 'Task ID not found in O*NET source'
            })
            continue

        expected_text = onet_lookup[task_id]
        if actual_text != expected_text:
            mismatches.append({
                'row': idx,
                'task_id': task_id,
                'expected': expected_text[:60],
                'actual': actual_text[:60]
            })

    if mismatches:
        logger.error(f"❌ VALIDATION FAILED: Found {len(mismatches)} mismatches")
        for i, mismatch in enumerate(mismatches[:5]):  # Show first 5
            logger.error(f"   Row {mismatch['row']}, Task ID {mismatch['task_id']}")
            if 'error' in mismatch:
                logger.error(f"      Error: {mismatch['error']}")
            else:
                logger.error(f"      Expected: {mismatch['expected']}...")
                logger.error(f"      Got: {mismatch['actual']}...")
        if len(mismatches) > 5:
            logger.error(f"   ... and {len(mismatches) - 5} more mismatches")
        raise AssertionError(f"Task ID alignment validation failed: {len(mismatches)} mismatches found")

    logger.info("✅ VALIDATION PASSED: All task_ids match their task_text")
    logger.info(f"   Validated {len(output):,} rows successfully")
    logger.info(f"   Task ID range: {output['onet_task_id'].min()} - {output['onet_task_id'].max()}")


if __name__ == "__main__":
    main()