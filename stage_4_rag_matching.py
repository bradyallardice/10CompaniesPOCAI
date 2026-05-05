#!/usr/bin/env python3
"""
Stage 4 (RAG Variant): Hybrid Retrieval + Reranking for AI Application → O*NET Task Matching

This is an alternative to stage_4_onet_similarity.py that uses a hybrid retrieval-augmented
approach instead of exhaustive all-pairs cosine similarity. The pipeline:

1. ENRICHMENT: O*NET tasks are enriched with occupation titles and Detailed Work Activities (DWAs)
   before embedding, giving the retriever more semantic surface area to match against.

2. HYBRID RETRIEVAL: For each extracted AI application, retrieve top-k O*NET tasks using:
   - Dense retrieval (BGE embeddings + cosine similarity)
   - Sparse retrieval (BM25 on tokenized text)
   - Reciprocal Rank Fusion (RRF) to combine both ranked lists

3. CROSS-ENCODER RERANKING: Top candidates from RRF are reranked using a cross-encoder model
   that sees both texts jointly (much higher quality than bi-encoder similarity alone).

4. THRESHOLDING: Apply configurable score thresholds and produce output compatible with Stage 5.

Output format is identical to stage_4_onet_similarity.py so Stage 5 can consume it unchanged.

Usage:
    python3 stage_4_rag_matching.py \\
        --step3-file Data/Testing/stage_3/1000_company_test/final_output_1000_sample.csv \\
        --output-dir Data/Testing/stage_4/rag_test/ \\
        --onet-file Data/task_statements_20.xlsx \\
        --tasks-to-dwas-file Data/Tasks_to_DWAs.xlsx \\
        --dwa-reference-file Data/DWA_Reference.xlsx \\
        --onet-version 20 \\
        --task-type core \\
        --dense-k 50 \\
        --sparse-k 50 \\
        --rrf-k 60 \\
        --rerank-top-n 30
"""

import pandas as pd
import numpy as np
import os
import hashlib
import re
import logging
import argparse
import pickle
import gc
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from collections import defaultdict

# Embedding and ML libraries
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import torch
    from tqdm import tqdm
except ImportError:
    print("ERROR: sentence-transformers and/or torch not installed.")
    print("Please install with: pip install sentence-transformers torch tqdm")
    exit(1)

# BM25 sparse retrieval
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    print("WARNING: rank_bm25 not installed. Sparse retrieval disabled.")
    print("Install with: pip install rank-bm25")

# NLTK for BM25 tokenization (lemmatization + stopword removal)
try:
    import ssl
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    from nltk import pos_tag, word_tokenize
    # Ensure required data is available (SSL workaround for macOS)
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    for resource in ['wordnet', 'stopwords', 'averaged_perceptron_tagger',
                      'averaged_perceptron_tagger_eng', 'punkt_tab']:
        nltk.download(resource, quiet=True)
    ssl._create_default_https_context = _orig_ctx
    _LEMMATIZER = WordNetLemmatizer()
    _STOPWORDS = set(stopwords.words('english'))
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    _LEMMATIZER = None
    _STOPWORDS = set()
    print("WARNING: nltk not installed. BM25 will use simple tokenization.")
    print("Install with: pip install nltk")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def canonicalize_text(text: str) -> str:
    """Normalize text for deduplication: lowercase, normalize quotes/whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[""''`]', '"', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def make_app_id(app_text: str) -> str:
    """Create a deterministic AI application ID (32-char uppercase hex MD5)."""
    return hashlib.md5(app_text.encode('utf-8')).hexdigest().upper()


def _penn_to_wordnet(tag: str) -> Optional[str]:
    """Map Penn Treebank POS tag to WordNet POS for lemmatization."""
    if tag.startswith('V'):
        return 'v'
    elif tag.startswith('N'):
        return 'n'
    elif tag.startswith('J'):
        return 'a'
    elif tag.startswith('R'):
        return 'r'
    return None


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Tokenize text for BM25: lowercase, remove punctuation, remove stopwords, lemmatize.

    Uses NLTK WordNet lemmatizer with POS-aware lemmatization so that verbs like
    "analyzing" become "analyze" and nouns like "documents" become "document".
    Falls back to simple whitespace tokenization if NLTK is unavailable.
    """
    if not HAS_NLTK:
        # Fallback: simple tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 1]

    text = text.lower()
    # Remove punctuation but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()

    # POS tag for accurate lemmatization
    tagged = pos_tag(tokens)

    result = []
    for word, tag in tagged:
        if len(word) <= 1:
            continue
        if word in _STOPWORDS:
            continue
        wn_pos = _penn_to_wordnet(tag)
        if wn_pos:
            lemma = _LEMMATIZER.lemmatize(word, wn_pos)
        else:
            lemma = _LEMMATIZER.lemmatize(word)
        result.append(lemma)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# O*NET data loading and enrichment
# ─────────────────────────────────────────────────────────────────────────────

def load_onet_tasks(onet_file: str, task_type: str = 'core',
                    include_soc_15: bool = False) -> pd.DataFrame:
    """
    Load O*NET task statements from Excel file.

    Args:
        onet_file: Path to Task Statements Excel file
        task_type: 'core', 'supplemental', or 'all'
        include_soc_15: Whether to include SOC group 15 (Computer & Math)

    Returns:
        DataFrame with columns: task_id, Task, Task Type, O*NET-SOC Code, Title
    """
    if not os.path.exists(onet_file):
        raise FileNotFoundError(f"O*NET file not found: {onet_file}")

    df = pd.read_excel(onet_file)

    required_cols = ['O*NET-SOC Code', 'Title', 'Task ID', 'Task', 'Task Type']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in O*NET file: {missing}. Available: {list(df.columns)}")

    # Filter by task type
    if task_type == 'core':
        df = df[df['Task Type'] == 'Core'].copy()
        logger.info(f"Filtered to Core tasks: {len(df)} rows")
    elif task_type == 'supplemental':
        df = df[df['Task Type'] == 'Supplemental'].copy()
        logger.info(f"Filtered to Supplemental tasks: {len(df)} rows")
    else:
        logger.info(f"Using all task types: {len(df)} rows")

    # Optionally exclude SOC group 15
    if not include_soc_15:
        before = len(df)
        df = df[~df['O*NET-SOC Code'].str.startswith('15-')].copy()
        logger.info(f"Excluded SOC 15: {before} -> {len(df)} rows")

    # Rename for internal consistency
    df = df.rename(columns={'Task ID': 'task_id'})

    logger.info(f"Loaded {len(df)} O*NET tasks from {onet_file}")
    logger.info(f"  Unique occupations: {df['O*NET-SOC Code'].nunique()}")
    logger.info(f"  Unique tasks: {df['task_id'].nunique()}")

    return df


def load_tasks_to_dwas(tasks_to_dwas_file: str) -> pd.DataFrame:
    """
    Load Tasks-to-DWAs mapping file.

    Returns DataFrame with columns: O*NET-SOC Code, Task ID, DWA ID, DWA Title
    """
    if not os.path.exists(tasks_to_dwas_file):
        raise FileNotFoundError(f"Tasks-to-DWAs file not found: {tasks_to_dwas_file}")

    df = pd.read_excel(tasks_to_dwas_file)

    required_cols = ['O*NET-SOC Code', 'Task ID', 'DWA ID', 'DWA Title']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in Tasks-to-DWAs file: {missing}. Available: {list(df.columns)}")

    logger.info(f"Loaded {len(df)} task-DWA mappings ({df['Task ID'].nunique()} unique tasks, "
                f"{df['DWA ID'].nunique()} unique DWAs)")
    return df 


def load_dwa_reference(dwa_reference_file: str) -> pd.DataFrame:
    """
    Load DWA Reference file with IWA (Intermediate Work Activity) context.

    Returns DataFrame with columns: DWA ID, DWA Title, IWA Title, Element Name
    """
    if not os.path.exists(dwa_reference_file):
        raise FileNotFoundError(f"DWA Reference file not found: {dwa_reference_file}")

    df = pd.read_excel(dwa_reference_file)

    required_cols = ['DWA ID', 'DWA Title', 'IWA Title', 'Element Name']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DWA Reference file: {missing}. Available: {list(df.columns)}")

    logger.info(f"Loaded {len(df)} DWA reference entries ({df['DWA ID'].nunique()} unique DWAs)")
    return df


def enrich_onet_tasks(onet_tasks: pd.DataFrame,
                      tasks_to_dwas: pd.DataFrame,
                      dwa_reference: Optional[pd.DataFrame] = None,
                      max_dwas_per_task: int = 5) -> pd.DataFrame:
    """
    Enrich O*NET task representations with occupation context and DWAs.

    For each task, creates an enriched text:
        "[Task statement] | Occupation: [occupation title] | DWAs: [dwa1]; [dwa2]; ..."

    This gives the retriever more semantic surface area for matching.

    Args:
        onet_tasks: DataFrame from load_onet_tasks()
        tasks_to_dwas: DataFrame from load_tasks_to_dwas()
        dwa_reference: Optional DataFrame from load_dwa_reference() (adds IWA context)
        max_dwas_per_task: Maximum number of DWAs to include per task

    Returns:
        DataFrame with added 'enriched_text' column
    """
    logger.info("Enriching O*NET tasks with occupation titles and DWAs...")

    # Build task_id -> list of DWA titles mapping
    # Group DWAs by (O*NET-SOC Code, Task ID) to handle task IDs that appear in multiple occupations
    task_dwas = tasks_to_dwas.groupby(['O*NET-SOC Code', 'Task ID'])['DWA Title'].apply(
        lambda x: list(x.unique())[:max_dwas_per_task]
    ).reset_index()
    task_dwas.columns = ['O*NET-SOC Code', 'task_id', 'dwa_titles']

    logger.info(f"  DWA coverage: {len(task_dwas)} (SOC, task_id) combinations have DWAs")

    # Merge DWAs into tasks
    enriched = onet_tasks.merge(
        task_dwas,
        on=['O*NET-SOC Code', 'task_id'],
        how='left'
    )

    # Fill missing DWAs with empty list
    enriched['dwa_titles'] = enriched['dwa_titles'].apply(
        lambda x: x if isinstance(x, list) else []
    )

    # Build enriched text
    def build_enriched_text(row):
        parts = [row['Task']]
        parts.append(f"Occupation: {row['Title']}")
        if row['dwa_titles']:
            dwa_str = "; ".join(row['dwa_titles'])
            parts.append(f"DWAs: {dwa_str}")
        return " | ".join(parts)

    enriched['enriched_text'] = enriched.apply(build_enriched_text, axis=1)

    # Stats
    has_dwas = enriched['dwa_titles'].apply(len).gt(0).sum()
    avg_dwas = enriched['dwa_titles'].apply(len).mean()
    logger.info(f"  Tasks with DWAs: {has_dwas}/{len(enriched)} ({100*has_dwas/len(enriched):.1f}%)")
    logger.info(f"  Average DWAs per task: {avg_dwas:.1f}")
    logger.info(f"  Sample enriched text: {enriched['enriched_text'].iloc[0][:200]}...")

    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication (matches existing Stage 4 logic)
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_applications(df: pd.DataFrame,
                             text_column: str = 'step3_output') -> pd.DataFrame:
    """
    Deduplicate AI application strings via exact hash matching.
    Preserves all job_uids for each unique application.

    Args:
        df: Input dataframe with AI applications from Stage 3
        text_column: Column containing the application text

    Returns:
        DataFrame with columns: app_text, job_uids (list), first_occurrence_tst_created, num_jobs
    """
    logger.info("Phase A: Deduplicating AI-application strings")

    df = df.copy()

    # Handle column name compatibility
    if 'original_job_uid' in df.columns and 'job_uid' not in df.columns:
        df['job_uid'] = df['original_job_uid']

    if 'job_uid' not in df.columns:
        raise ValueError(f"Column 'job_uid' not found. Available: {list(df.columns)}")
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")

    # Canonicalize text
    df['app_text'] = df[text_column].apply(canonicalize_text)
    df = df[df['app_text'] != ''].copy()
    initial_count = len(df)
    logger.info(f"  Initial rows: {initial_count}")

    # Sort by timestamp for temporal ordering
    if 'tst_created' in df.columns:
        df = df.sort_values('tst_created', ascending=True)

    # Hash-based exact deduplication
    df['text_hash'] = df['app_text'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

    dedup = df.groupby('text_hash').agg({
        'job_uid': lambda x: list(dict.fromkeys(x)),  # Unique, order-preserved
        'tst_created': 'first' if 'tst_created' in df.columns else lambda x: None,
        'app_text': 'first'
    }).reset_index()

    dedup['num_jobs'] = dedup['job_uid'].map(len)
    dedup = dedup.rename(columns={
        'job_uid': 'job_uids',
        'tst_created': 'first_occurrence_tst_created'
    })

    result = dedup[['app_text', 'job_uids', 'first_occurrence_tst_created', 'num_jobs']].copy()
    logger.info(f"  After deduplication: {len(result)} unique applications (from {initial_count} rows)")
    logger.info(f"  Total job_uids preserved: {result['num_jobs'].sum()}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Embedding generation
# ─────────────────────────────────────────────────────────────────────────────

def get_or_compute_embeddings(texts: List[str],
                              cache_path: str,
                              model: SentenceTransformer,
                              batch_size: int = 128,
                              prefix: str = "") -> np.ndarray:
    """
    Load embeddings from cache or compute and save them.

    Args:
        texts: List of strings to embed
        cache_path: Path to pickle cache file
        model: SentenceTransformer model
        batch_size: Batch size for encoding
        prefix: BGE query prefix (e.g., "Represent this sentence: ")

    Returns:
        numpy array of shape (len(texts), embedding_dim), L2-normalized
    """
    if os.path.exists(cache_path):
        logger.info(f"  Loading cached embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        if len(embeddings) == len(texts):
            logger.info(f"  Cache hit: {len(embeddings)} embeddings loaded")
            return embeddings
        else:
            logger.warning(f"  Cache size mismatch ({len(embeddings)} vs {len(texts)}), recomputing")

    logger.info(f"  Computing embeddings for {len(texts)} texts (batch_size={batch_size})")

    # Add prefix if provided (BGE models benefit from query prefixes)
    if prefix:
        texts_to_encode = [prefix + t for t in texts]
    else:
        texts_to_encode = texts

    embeddings = model.encode(
        texts_to_encode,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"  Saved embeddings to {cache_path}")

    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid retrieval: Dense + Sparse + RRF
# ─────────────────────────────────────────────────────────────────────────────

def dense_retrieve(query_embedding: np.ndarray,
                   corpus_embeddings: np.ndarray,
                   top_k: int = 50) -> List[Tuple[int, float]]:
    """
    Dense retrieval: cosine similarity between query and all corpus embeddings.

    Args:
        query_embedding: (embedding_dim,) normalized query vector
        corpus_embeddings: (n_corpus, embedding_dim) normalized corpus matrix
        top_k: Number of top results to return

    Returns:
        List of (corpus_index, similarity_score) sorted by score descending
    """
    similarities = corpus_embeddings @ query_embedding
    top_indices = np.argpartition(-similarities, min(top_k, len(similarities) - 1))[:top_k]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def sparse_retrieve(query_tokens: List[str],
                    bm25_index: 'BM25Okapi',
                    top_k: int = 50) -> List[Tuple[int, float]]:
    """
    Sparse retrieval: BM25 scoring over tokenized corpus.

    Args:
        query_tokens: Tokenized query
        bm25_index: Pre-built BM25 index
        top_k: Number of top results to return

    Returns:
        List of (corpus_index, bm25_score) sorted by score descending
    """
    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[int, float]]],
                           k: int = 60) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion to combine multiple ranked lists.

    RRF score for document d = sum over lists: 1 / (k + rank_in_list)
    where k is a smoothing constant (default 60 per original RRF paper).

    Args:
        ranked_lists: List of ranked lists, each containing (doc_id, score) tuples
        k: RRF smoothing constant

    Returns:
        Fused ranked list of (doc_id, rrf_score) sorted by score descending
    """
    rrf_scores = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, (doc_id, _score) in enumerate(ranked_list):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)  # rank is 0-indexed

    # Sort by RRF score descending
    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])
    return fused


def hybrid_retrieve_batch(app_embeddings: np.ndarray,
                          app_texts: List[str],
                          task_embeddings: np.ndarray,
                          task_texts_tokenized: Optional[List[List[str]]],
                          bm25_index: Optional['BM25Okapi'],
                          dense_k: int = 50,
                          sparse_k: int = 50,
                          rrf_k: int = 60,
                          final_k: int = 30,
                          use_sparse: bool = True) -> Dict[int, List[Tuple[int, float]]]:
    """
    Run hybrid retrieval for a batch of AI applications against the O*NET task corpus.

    Args:
        app_embeddings: (n_apps, dim) normalized embeddings for AI applications
        app_texts: List of application text strings
        task_embeddings: (n_tasks, dim) normalized embeddings for O*NET tasks
        task_texts_tokenized: Tokenized O*NET task texts (for BM25)
        bm25_index: Pre-built BM25 index on O*NET tasks
        dense_k: Top-k for dense retrieval per application
        sparse_k: Top-k for sparse retrieval per application
        rrf_k: RRF smoothing constant
        final_k: Number of final candidates per application after RRF
        use_sparse: Whether to include BM25 sparse retrieval

    Returns:
        Dict mapping app_index -> list of (task_index, rrf_score)
    """
    results = {}

    for i in tqdm(range(len(app_embeddings)), desc="Hybrid retrieval"):
        ranked_lists = []

        # Dense retrieval
        dense_results = dense_retrieve(app_embeddings[i], task_embeddings, top_k=dense_k)
        ranked_lists.append(dense_results)

        # Sparse retrieval (if available)
        if use_sparse and bm25_index is not None:
            query_tokens = tokenize_for_bm25(app_texts[i])
            if query_tokens:  # Only if we have tokens
                sparse_results = sparse_retrieve(query_tokens, bm25_index, top_k=sparse_k)
                ranked_lists.append(sparse_results)

        # Fuse with RRF
        if len(ranked_lists) > 1:
            fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k)
        else:
            # Only dense available
            fused = [(idx, score) for idx, score in ranked_lists[0]]

        results[i] = fused[:final_k]

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder reranking
# ─────────────────────────────────────────────────────────────────────────────

def rerank_with_cross_encoder(app_texts: List[str],
                              task_texts: List[str],
                              candidates: Dict[int, List[Tuple[int, float]]],
                              cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                              batch_size: int = 256,
                              rerank_top_n: Optional[int] = None) -> Dict[int, List[Tuple[int, float, float]]]:
    """
    Rerank candidate pairs using a cross-encoder model.

    The cross-encoder sees both texts jointly, producing much higher quality
    relevance scores than bi-encoder cosine similarity.

    Args:
        app_texts: List of AI application texts
        task_texts: List of O*NET task texts (plain task statement, not enriched)
        candidates: Dict mapping app_idx -> [(task_idx, rrf_score), ...]
        cross_encoder_model: Model name for reranking
        batch_size: Batch size for cross-encoder inference
        rerank_top_n: If set, only rerank the top N candidates per app (rest get NaN CE score)

    Returns:
        Dict mapping app_idx -> [(task_idx, rrf_score, cross_encoder_score), ...]
        sorted by cross_encoder_score descending
    """
    logger.info(f"Loading cross-encoder model: {cross_encoder_model}")
    ce_model = CrossEncoder(cross_encoder_model)

    # Build all (app_text, task_text) pairs for batch scoring
    pairs = []
    pair_keys = []  # (app_idx, task_idx, rrf_score)

    for app_idx, task_candidates in candidates.items():
        n_to_rerank = len(task_candidates) if rerank_top_n is None else min(rerank_top_n, len(task_candidates))
        for task_idx, rrf_score in task_candidates[:n_to_rerank]:
            pairs.append((app_texts[app_idx], task_texts[task_idx]))
            pair_keys.append((app_idx, task_idx, rrf_score))

    total_pairs = len(pairs)
    logger.info(f"Reranking {total_pairs:,} candidate pairs with cross-encoder")

    # Score in batches — use raw logit scores (same as existing Stage 4)
    # BGE reranker outputs raw relevance scores where higher = more relevant
    all_scores = []
    for start in tqdm(range(0, total_pairs, batch_size), desc="Cross-encoder reranking"):
        batch = pairs[start:start + batch_size]
        scores = ce_model.predict(batch, show_progress_bar=False)
        if isinstance(scores, (int, float)):
            scores = [scores]
        elif hasattr(scores, 'tolist'):
            scores = scores.tolist()
        all_scores.extend(scores)

    # Reconstruct results grouped by app_idx
    reranked = defaultdict(list)
    for (app_idx, task_idx, rrf_score), ce_score in zip(pair_keys, all_scores):
        reranked[app_idx].append((task_idx, rrf_score, float(ce_score)))

    # Add non-reranked candidates (if rerank_top_n < total candidates)
    if rerank_top_n is not None:
        for app_idx, task_candidates in candidates.items():
            reranked_task_ids = {t[0] for t in reranked[app_idx]}
            for task_idx, rrf_score in task_candidates[rerank_top_n:]:
                if task_idx not in reranked_task_ids:
                    reranked[app_idx].append((task_idx, rrf_score, float('nan')))

    # Sort each app's candidates by CE score descending (NaN goes to end)
    for app_idx in reranked:
        reranked[app_idx].sort(key=lambda x: -x[2] if not math.isnan(x[2]) else float('-inf'))

    logger.info(f"Reranking complete. Sample top CE scores: "
                f"{[round(reranked[0][j][2], 3) for j in range(min(5, len(reranked[0])))]}")

    return dict(reranked)


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting (Stage 5 compatible)
# ─────────────────────────────────────────────────────────────────────────────

def build_output_dataframe(app_dedup: pd.DataFrame,
                           task_groups: pd.DataFrame,
                           reranked_results: Dict[int, List[Tuple[int, float, float]]],
                           app_texts: List[str],
                           task_type: str,
                           ce_thresholds: List[float],
                           dense_similarities: Optional[Dict[int, Dict[int, float]]] = None
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Stage 5-compatible output DataFrames from reranked results.

    Uses cross-encoder scores as the primary quality signal (no BGE percentile columns).
    Stage 5 filters on ce_* boolean columns.

    Produces:
    1. Main matches file: one row per (app_text, onet_task_id) pair with scores
    2. Job mapping file: one row per unique app_text with job_uids

    Args:
        app_dedup: Deduplicated applications DataFrame
        task_groups: Unique O*NET tasks DataFrame (same row ordering as embeddings/retrieval indices)
        reranked_results: Dict from rerank step: app_idx -> [(task_idx, rrf_score, ce_score)]
        app_texts: Ordered list of app texts (index-aligned with reranked_results keys)
        task_type: 'core' or 'all'
        ce_thresholds: List of CE thresholds (e.g., [0.8, 0.6, 0.4, 0.2, 0.0])
        dense_similarities: Optional dict of app_idx -> {task_idx: cosine_sim} for BGE scores

    Returns:
        (matches_df, job_mapping_df)
    """
    logger.info("Building Stage 5-compatible output...")

    # Build positional index -> (task_id, task_text) mapping
    # CRITICAL: task_groups must have the same row ordering as the embeddings/retrieval indices
    task_idx_to_id = dict(enumerate(task_groups['task_id'].values))
    task_idx_to_text = dict(enumerate(task_groups['Task'].values))

    # Build app_text -> dedup info lookup
    app_text_to_info = {}
    for idx, row in app_dedup.iterrows():
        app_text_to_info[row['app_text']] = {
            'job_uids': row['job_uids'],
            'first_occurrence_tst_created': row['first_occurrence_tst_created'],
            'num_jobs': row['num_jobs']
        }

    # Collect all rows
    rows = []
    for app_idx, task_candidates in reranked_results.items():
        app_text = app_texts[app_idx]
        info = app_text_to_info.get(app_text, {})
        ai_app_id = make_app_id(app_text)

        for task_idx, rrf_score, ce_score in task_candidates:
            task_id = task_idx_to_id.get(task_idx)
            task_text = task_idx_to_text.get(task_idx)

            if task_id is None:
                continue

            # Get dense (BGE) similarity if available, else use RRF score as similarity proxy
            if dense_similarities and app_idx in dense_similarities:
                similarity = dense_similarities[app_idx].get(task_idx, 0.0)
            else:
                similarity = rrf_score  # Fallback: use RRF score

            rows.append({
                'app_text': app_text,
                'onet_task_id': int(task_id),
                'onet_task': task_text,
                'similarity': float(similarity),
                'cross_encoder_score': float(ce_score) if not math.isnan(ce_score) else np.nan,
                'job_uid': info.get('job_uids', []),
                'first_occurrence_tst_created': info.get('first_occurrence_tst_created'),
                'num_jobs': info.get('num_jobs', 0),
                'ai_app_id': ai_app_id,
                'task_type': task_type.capitalize() if task_type == 'core' else task_type,
                'rrf_score': float(rrf_score),
            })

    matches_df = pd.DataFrame(rows)
    logger.info(f"  Total match rows: {len(matches_df):,}")

    if len(matches_df) == 0:
        raise ValueError("No matches found. Check input data and retrieval parameters.")

    # Deduplicate on (app_text, onet_task_id) - keep highest CE score
    before_dedup = len(matches_df)
    matches_df = matches_df.sort_values('cross_encoder_score', ascending=False, na_position='last')
    matches_df = matches_df.drop_duplicates(subset=['app_text', 'onet_task_id'], keep='first')
    logger.info(f"  After (app_text, onet_task_id) dedup: {len(matches_df):,} (removed {before_dedup - len(matches_df):,})")

    # Sort by CE score descending (primary quality signal)
    matches_df = matches_df.sort_values('cross_encoder_score', ascending=False, na_position='last')

    # Log similarity stats for reference
    all_sims = matches_df['similarity'].values
    logger.info(f"  BGE similarity stats: min={all_sims.min():.4f}, max={all_sims.max():.4f}, "
                f"mean={all_sims.mean():.4f}, median={np.median(all_sims):.4f}")

    # Log CE score stats
    ce_scores = matches_df['cross_encoder_score'].dropna()
    if len(ce_scores) > 0:
        logger.info(f"  CE score stats: min={ce_scores.min():.4f}, max={ce_scores.max():.4f}, "
                    f"mean={ce_scores.mean():.4f}, median={ce_scores.median():.4f}")

    # Add CE threshold boolean columns (primary filtering mechanism)
    for ce_thresh in ce_thresholds:
        col_name = f'ce_{ce_thresh:.1f}'
        if ce_thresh == 0.0:
            matches_df[col_name] = True
        else:
            matches_df[col_name] = (
                (matches_df['cross_encoder_score'] >= ce_thresh) &
                matches_df['cross_encoder_score'].notna()
            )
        n_matches = matches_df[col_name].sum()
        logger.info(f"  {col_name}: {n_matches:,} matches")

    # Build job mapping DataFrame
    job_mapping_rows = []
    for _, row in app_dedup.iterrows():
        job_mapping_rows.append({
            'app_text': row['app_text'],
            'ai_app_id': make_app_id(row['app_text']),
            'job_uids': '|'.join(sorted(row['job_uids'])) if isinstance(row['job_uids'], list) else str(row['job_uids']),
            'num_jobs': row['num_jobs'],
            'first_occurrence_tst_created': row['first_occurrence_tst_created']
        })
    job_mapping_df = pd.DataFrame(job_mapping_rows)

    logger.info(f"  Job mapping: {len(job_mapping_df)} unique applications")

    return matches_df, job_mapping_df


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class HybridRAGMatcher:
    """
    Hybrid retrieval-augmented matching pipeline for AI applications → O*NET tasks.
    """

    def __init__(self,
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                 embeddings_dir: str = "Data/embeddings",
                 batch_size: int = 128,
                 dense_k: int = 50,
                 sparse_k: int = 50,
                 rrf_k: int = 60,
                 skip_cross_encoder: bool = False,
                 skip_sparse: bool = False):
        """
        Initialize the hybrid RAG matcher.

        Args:
            model_name: SentenceTransformer model for dense embeddings
            cross_encoder_model: Cross-encoder model for reranking
            embeddings_dir: Directory for cached embeddings
            batch_size: Batch size for embedding/reranking
            dense_k: Top-k for dense retrieval per query
            sparse_k: Top-k for sparse (BM25) retrieval per query
            rrf_k: RRF smoothing constant
            skip_cross_encoder: If True, skip cross-encoder reranking
            skip_sparse: If True, skip BM25 sparse retrieval (dense only)
        """
        self.model_name = model_name
        self.cross_encoder_model = cross_encoder_model
        self.embeddings_dir = embeddings_dir
        self.batch_size = batch_size
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.rrf_k = rrf_k
        self.skip_cross_encoder = skip_cross_encoder
        self.skip_sparse = skip_sparse or (not HAS_BM25)

        os.makedirs(embeddings_dir, exist_ok=True)

        # Load bi-encoder model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Device setup
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

        if self.device == "cpu":
            self.batch_size = min(self.batch_size, 64)
        elif self.device == "mps":
            self.batch_size = min(self.batch_size, 128)

        self.model = self.model.to(self.device)

    def run(self,
            step3_file: str,
            onet_file: str,
            tasks_to_dwas_file: str,
            dwa_reference_file: Optional[str] = None,
            output_dir: str = "Data/Testing/stage_4/rag/",
            text_column: str = 'step3_output',
            task_type: str = 'core',
            include_soc_15: bool = False,
            onet_version: int = 20,
            ce_thresholds: Optional[List[float]] = None,
            max_dwas_per_task: int = 5) -> pd.DataFrame:
        """
        Run the full hybrid RAG matching pipeline.

        Args:
            step3_file: Path to Stage 3 output CSV
            onet_file: Path to O*NET Task Statements Excel file
            tasks_to_dwas_file: Path to Tasks-to-DWAs Excel file
            dwa_reference_file: Optional path to DWA Reference Excel file
            output_dir: Output directory for results
            text_column: Column name in Stage 3 file containing AI application text
            task_type: 'core' or 'all'
            include_soc_15: Whether to include SOC group 15
            onet_version: O*NET version number
            ce_thresholds: CE thresholds (default: [0.8, 0.6, 0.4, 0.2, 0.0])
            max_dwas_per_task: Max DWAs per task for enrichment

        Returns:
            Final matches DataFrame
        """
        if ce_thresholds is None:
            ce_thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]

        os.makedirs(output_dir, exist_ok=True)

        # ── Step 1: Load and validate inputs ──
        logger.info("=" * 80)
        logger.info("STEP 1: Loading and validating inputs")
        logger.info("=" * 80)

        if not os.path.exists(step3_file):
            raise FileNotFoundError(f"Stage 3 file not found: {step3_file}")

        stage3_df = pd.read_csv(step3_file)
        logger.info(f"Loaded Stage 3 data: {len(stage3_df)} rows, columns: {list(stage3_df.columns)}")

        if text_column not in stage3_df.columns:
            raise ValueError(f"Column '{text_column}' not found in {step3_file}. "
                             f"Available: {list(stage3_df.columns)}")

        # Load O*NET tasks
        onet_tasks = load_onet_tasks(onet_file, task_type=task_type, include_soc_15=include_soc_15)

        # Load DWA data
        tasks_to_dwas = load_tasks_to_dwas(tasks_to_dwas_file)
        dwa_ref = load_dwa_reference(dwa_reference_file) if dwa_reference_file else None

        # ── Step 2: Enrich O*NET tasks ──
        logger.info("=" * 80)
        logger.info("STEP 2: Enriching O*NET tasks with DWAs")
        logger.info("=" * 80)

        enriched_tasks = enrich_onet_tasks(
            onet_tasks, tasks_to_dwas, dwa_ref, max_dwas_per_task=max_dwas_per_task
        )

        # For retrieval, we need unique tasks (task_id is unique, even if appearing in multiple occupations)
        # Use enriched text for embedding, but keep one row per unique task_id
        # When a task appears under multiple occupations, concatenate the occupation context
        task_groups = enriched_tasks.groupby('task_id').agg({
            'Task': 'first',
            'Task Type': 'first',
            'enriched_text': lambda x: ' | '.join(x.unique()),  # Combine enrichments from different occupations
        }).reset_index()

        unique_task_texts_enriched = task_groups['enriched_text'].tolist()
        unique_task_texts_plain = task_groups['Task'].tolist()
        unique_task_ids = task_groups['task_id'].tolist()

        logger.info(f"Unique tasks for retrieval: {len(unique_task_texts_enriched)}")

        # ── Step 3: Deduplicate AI applications ──
        logger.info("=" * 80)
        logger.info("STEP 3: Deduplicating AI applications")
        logger.info("=" * 80)

        app_dedup = deduplicate_applications(stage3_df, text_column=text_column)
        app_texts = app_dedup['app_text'].tolist()

        # ── Step 4: Compute embeddings ──
        logger.info("=" * 80)
        logger.info("STEP 4: Computing embeddings")
        logger.info("=" * 80)

        # Hash for cache key
        app_hash = hashlib.md5("".join(sorted(app_texts)).encode()).hexdigest()[:12]
        task_hash = hashlib.md5("".join(sorted(unique_task_texts_enriched)).encode()).hexdigest()[:12]

        app_cache = os.path.join(self.embeddings_dir, f"rag_apps_{self.model_name.split('/')[-1]}_{app_hash}.pkl")
        task_cache = os.path.join(self.embeddings_dir, f"rag_tasks_enriched_{self.model_name.split('/')[-1]}_{task_hash}.pkl")

        logger.info("Computing AI application embeddings...")
        app_embeddings = get_or_compute_embeddings(
            app_texts, app_cache, self.model, batch_size=self.batch_size
        )

        logger.info("Computing enriched O*NET task embeddings...")
        task_embeddings = get_or_compute_embeddings(
            unique_task_texts_enriched, task_cache, self.model, batch_size=self.batch_size
        )

        # ── Step 5: Build BM25 index (sparse retrieval) ──
        bm25_index = None
        if not self.skip_sparse:
            logger.info("=" * 80)
            logger.info("STEP 5: Building BM25 index for sparse retrieval")
            logger.info("=" * 80)

            # Tokenize enriched task texts for BM25
            task_tokens = [tokenize_for_bm25(t) for t in unique_task_texts_enriched]
            bm25_index = BM25Okapi(task_tokens)
            logger.info(f"BM25 index built: {len(task_tokens)} documents, "
                        f"avg tokens/doc: {np.mean([len(t) for t in task_tokens]):.1f}")
        else:
            logger.info("STEP 5: Skipping BM25 (sparse retrieval disabled)")

        # ── Step 6: Hybrid retrieval ──
        logger.info("=" * 80)
        logger.info("STEP 6: Hybrid retrieval (dense + sparse + RRF)")
        logger.info("=" * 80)

        # Determine how many candidates to keep after RRF
        # When sparse is enabled, keep dense_k + sparse_k so BM25-unique finds
        # survive into the CE reranking pool
        if not self.skip_sparse:
            final_k = self.dense_k + self.sparse_k
        else:
            final_k = self.dense_k

        candidates = hybrid_retrieve_batch(
            app_embeddings=app_embeddings,
            app_texts=app_texts,
            task_embeddings=task_embeddings,
            task_texts_tokenized=None,
            bm25_index=bm25_index,
            dense_k=self.dense_k,
            sparse_k=self.sparse_k,
            rrf_k=self.rrf_k,
            final_k=final_k,
            use_sparse=not self.skip_sparse
        )

        total_candidates = sum(len(v) for v in candidates.values())
        logger.info(f"Retrieved {total_candidates:,} total candidates for {len(candidates)} applications")
        logger.info(f"Average candidates per app: {total_candidates / max(len(candidates), 1):.1f}")

        # Collect dense similarities for all retrieved candidates (for output)
        logger.info("Computing dense similarities for all retrieved candidates...")
        dense_similarities = {}
        for app_idx, task_candidates in candidates.items():
            dense_similarities[app_idx] = {}
            for task_idx, _ in task_candidates:
                sim = float(app_embeddings[app_idx] @ task_embeddings[task_idx])
                dense_similarities[app_idx][task_idx] = sim

        # ── Step 7: Cross-encoder reranking ──
        logger.info("=" * 80)
        logger.info("STEP 7: Cross-encoder reranking")
        logger.info("=" * 80)

        if self.skip_cross_encoder:
            logger.info("Skipping cross-encoder (--skip-cross-encoder flag)")
            # Convert candidates to reranked format with NaN CE scores
            reranked = {}
            for app_idx, task_candidates in candidates.items():
                reranked[app_idx] = [(t_idx, rrf_score, float('nan'))
                                     for t_idx, rrf_score in task_candidates]
        else:
            reranked = rerank_with_cross_encoder(
                app_texts=app_texts,
                task_texts=unique_task_texts_plain,  # Use plain task text for CE (not enriched)
                candidates=candidates,
                cross_encoder_model=self.cross_encoder_model,
                batch_size=self.batch_size,
                rerank_top_n=None  # Rerank ALL candidates
            )

        # ── Step 8: Build output ──
        logger.info("=" * 80)
        logger.info("STEP 8: Building Stage 5-compatible output")
        logger.info("=" * 80)

        matches_df, job_mapping_df = build_output_dataframe(
            app_dedup=app_dedup,
            task_groups=task_groups,
            reranked_results=reranked,
            app_texts=app_texts,
            task_type=task_type,
            ce_thresholds=ce_thresholds,
            dense_similarities=dense_similarities
        )

        # ── Step 9: Save outputs ──
        logger.info("=" * 80)
        logger.info("STEP 9: Saving outputs")
        logger.info("=" * 80)

        # Build filename components
        model_abbr = self.model_name.split("/")[-1].split("-")[0]
        model_suffix = f"_{model_abbr}"
        ce_str = "_".join(f"{c:.1f}".replace(".", "p") for c in sorted(ce_thresholds, reverse=True) if c > 0.0)
        onet_suffix = f"_onet{onet_version}" if onet_version else ""
        task_suffix = f"_{task_type}"
        retrieval_suffix = f"_dk{self.dense_k}_sk{self.sparse_k if not self.skip_sparse else 0}"

        # Main matches file
        matches_file = os.path.join(
            output_dir,
            f"task_exposure_matches_rag{model_suffix}_ce{ce_str}{retrieval_suffix}{onet_suffix}{task_suffix}.parquet"
        )

        # Drop the rrf_score column before saving (not needed by Stage 5)
        output_cols = [c for c in matches_df.columns if c != 'rrf_score']
        matches_df[output_cols].to_parquet(matches_file, index=False, compression='snappy')
        logger.info(f"Saved matches: {matches_file} ({len(matches_df):,} rows)")

        # Also save with rrf_score for diagnostics
        diagnostics_file = os.path.join(output_dir, f"rag_diagnostics{model_suffix}{onet_suffix}{task_suffix}.parquet")
        matches_df.to_parquet(diagnostics_file, index=False, compression='snappy')
        logger.info(f"Saved diagnostics (with RRF scores): {diagnostics_file}")

        # Job mapping file
        job_mapping_file = os.path.join(
            output_dir,
            f"job_app_mapping_rag{model_suffix}_ce{ce_str}{retrieval_suffix}{onet_suffix}{task_suffix}.parquet"
        )
        job_mapping_df.to_parquet(job_mapping_file, index=False, compression='snappy')
        logger.info(f"Saved job mapping: {job_mapping_file} ({len(job_mapping_df)} applications)")

        # ── Summary ──
        logger.info("=" * 80)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  Stage 3 input: {len(stage3_df):,} rows")
        logger.info(f"  Unique AI applications: {len(app_texts):,}")
        logger.info(f"  O*NET tasks ({task_type}): {len(unique_task_texts_enriched):,}")
        logger.info(f"  Tasks with DWA enrichment: {enriched_tasks['dwa_titles'].apply(len).gt(0).sum():,}")
        logger.info(f"  Retrieval mode: {'hybrid (dense+BM25+RRF)' if not self.skip_sparse else 'dense only'}")
        logger.info(f"  Dense top-k: {self.dense_k}, Sparse top-k: {self.sparse_k}")
        logger.info(f"  RRF k: {self.rrf_k}, Final candidates: {final_k}")
        logger.info(f"  Cross-encoder: {'enabled (all candidates)' if not self.skip_cross_encoder else 'disabled'}")
        logger.info(f"  Output matches: {len(matches_df):,} (app, task) pairs")
        logger.info(f"  Output job mapping: {len(job_mapping_df):,} unique applications")
        logger.info(f"  Files saved to: {output_dir}")

        # Validation checks
        logger.info("Running output validation...")
        required_cols = ['app_text', 'onet_task_id', 'onet_task', 'similarity',
                         'cross_encoder_score', 'ai_app_id', 'job_uid']
        missing_cols = [c for c in required_cols if c not in matches_df.columns]
        if missing_cols:
            raise ValueError(f"Output missing required columns for Stage 5: {missing_cols}")

        # Validate ai_app_id alignment
        sample = matches_df.head(100)
        expected_ids = sample['app_text'].apply(make_app_id)
        if not (sample['ai_app_id'] == expected_ids).all():
            raise ValueError("ai_app_id does not match MD5 hash of app_text")

        # Check similarity range
        sim_min, sim_max = matches_df['similarity'].min(), matches_df['similarity'].max()
        if sim_min < -0.01 or sim_max > 1.01:
            logger.warning(f"Similarity values outside [0,1]: min={sim_min:.4f}, max={sim_max:.4f}")

        # CE threshold monotonicity
        sorted_thresholds = sorted(ce_thresholds, reverse=True)
        for i in range(len(sorted_thresholds) - 1):
            stricter = f'ce_{sorted_thresholds[i]:.1f}'
            looser = f'ce_{sorted_thresholds[i+1]:.1f}'
            if stricter in matches_df.columns and looser in matches_df.columns:
                if matches_df[stricter].sum() > matches_df[looser].sum():
                    logger.warning(f"CE threshold monotonicity violated: {stricter} > {looser}")

        logger.info("Validation passed.")

        return matches_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 4 (RAG): Hybrid retrieval + reranking for AI app → O*NET task matching",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input files
    parser.add_argument('--step3-file', required=True,
                        help='Path to Stage 3 output CSV')
    parser.add_argument('--onet-file', required=True,
                        help='Path to O*NET Task Statements Excel file')
    parser.add_argument('--tasks-to-dwas-file', required=True,
                        help='Path to Tasks-to-DWAs Excel file (O*NET)')
    parser.add_argument('--dwa-reference-file', default=None,
                        help='Path to DWA Reference Excel file (optional, adds IWA context)')
    parser.add_argument('--content-column', default='step3_output',
                        help='Column name in Stage 3 file containing AI application text')

    # O*NET options
    parser.add_argument('--onet-version', type=int, default=20,
                        help='O*NET version (default: 20)')
    parser.add_argument('--task-type', choices=['core', 'supplemental', 'all'], default='core',
                        help='O*NET task type filter (default: core)')
    parser.add_argument('--include-soc-15', action='store_true',
                        help='Include SOC group 15 (Computer & Math occupations)')
    parser.add_argument('--max-dwas-per-task', type=int, default=5,
                        help='Maximum DWAs per task for enrichment (default: 5)')

    # Model options
    parser.add_argument('--model-name', default='BAAI/bge-large-en-v1.5',
                        help='SentenceTransformer model for dense embeddings')
    parser.add_argument('--cross-encoder-model', default='BAAI/bge-reranker-v2-m3',
                        help='Cross-encoder model for reranking')

    # Retrieval parameters
    parser.add_argument('--dense-k', type=int, default=50,
                        help='Top-k for dense retrieval per query (default: 50)')
    parser.add_argument('--sparse-k', type=int, default=50,
                        help='Top-k for BM25 sparse retrieval per query (default: 50)')
    parser.add_argument('--rrf-k', type=int, default=60,
                        help='RRF smoothing constant (default: 60)')

    # Flags
    parser.add_argument('--skip-cross-encoder', action='store_true',
                        help='Skip cross-encoder reranking (use RRF scores only)')
    parser.add_argument('--skip-sparse', action='store_true',
                        help='Skip BM25 sparse retrieval (dense only)')

    # Threshold options
    parser.add_argument('--ce-thresholds', nargs='+', type=float, default=[0.8, 0.6, 0.4, 0.2, 0.0],
                        help='Cross-encoder score thresholds (default: 0.8 0.6 0.4 0.2 0.0)')

    # Output
    parser.add_argument('--output-dir', default='Data/Testing/stage_4/rag/',
                        help='Output directory (default: Data/Testing/stage_4/rag/)')
    parser.add_argument('--embeddings-dir', default='Data/embeddings',
                        help='Directory for cached embeddings (default: Data/embeddings)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for embedding/reranking (default: 128)')

    # Grid search mode
    parser.add_argument('--grid-search', action='store_true',
                        help='Run grid search over retrieval parameters')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def run_grid_search(step3_file: str,
                    onet_file: str,
                    tasks_to_dwas_file: str,
                    dwa_reference_file: Optional[str],
                    output_dir: str,
                    embeddings_dir: str,
                    text_column: str = 'step3_output',
                    task_type: str = 'core',
                    include_soc_15: bool = False,
                    onet_version: int = 20,
                    batch_size: int = 128,
                    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
                    model_name: str = "BAAI/bge-large-en-v1.5"):
    """
    Grid search over retrieval parameters to find the configuration that maximises
    the number of high-confidence (high-CE) matches.

    Evaluation logic:
    Since we have no gold-standard task-matching labels, we use the cross-encoder as
    an oracle: a match is "real" when its CE score is high. A better retrieval config
    surfaces more real matches into the candidate pool for the CE to confirm.

    We measure:
    - n_matches_ce_X: count of pairs above each CE threshold (0.3, 0.5, 0.7, 0.9)
    - apps_with_match_X: how many apps have at least one match above threshold
    - mean_top1_ce: average CE score of the best match per app
    - mean_top3_ce: average CE score of the top-3 matches per app

    Grid dimensions:
    - dense_k: [30, 50, 100, 200]
    - sparse_k: [0, 30, 50, 100]  (0 = dense-only, skip BM25)
    - max_dwas_per_task: [0, 3, 5, 10]  (0 = no DWA enrichment)
    """
    import json
    from itertools import product

    BGE_QUERY_PREFIX = "Represent this sentence: "

    # ── Grid definition ──
    dense_k_values = [30, 50, 100, 200]
    sparse_k_values = [0, 30, 50, 100]
    max_dwas_values = [0]
    use_prefix_values = [True]

    configs = list(product(dense_k_values, sparse_k_values, max_dwas_values, use_prefix_values))
    logger.info(f"Grid search: {len(configs)} configurations to evaluate")

    # ── Pre-load data once (shared across all configs) ──
    logger.info("Pre-loading shared data...")
    if not os.path.exists(step3_file):
        raise FileNotFoundError(f"Stage 3 file not found: {step3_file}")
    stage3_df = pd.read_csv(step3_file)
    logger.info(f"Stage 3: {len(stage3_df)} rows")

    onet_tasks = load_onet_tasks(onet_file, task_type=task_type, include_soc_15=include_soc_15)
    tasks_to_dwas = load_tasks_to_dwas(tasks_to_dwas_file)
    dwa_ref = load_dwa_reference(dwa_reference_file) if dwa_reference_file else None

    # Deduplicate apps once
    app_dedup = deduplicate_applications(stage3_df, text_column=text_column)
    app_texts = app_dedup['app_text'].tolist()
    n_apps = len(app_texts)

    # Load bi-encoder model once
    logger.info(f"Loading embedding model: {model_name}")
    bi_encoder = SentenceTransformer(model_name)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    bi_encoder = bi_encoder.to(device)

    # Compute app embeddings for both prefix variants
    app_hash = hashlib.md5("".join(sorted(app_texts)).encode()).hexdigest()[:12]

    # Without prefix
    app_cache_no = os.path.join(embeddings_dir, f"rag_apps_{model_name.split('/')[-1]}_{app_hash}.pkl")
    logger.info("Computing app embeddings (no prefix)...")
    app_emb_no_prefix = get_or_compute_embeddings(
        app_texts, app_cache_no, bi_encoder, batch_size=batch_size
    )

    # With BGE query prefix
    app_cache_pfx = os.path.join(embeddings_dir, f"rag_apps_{model_name.split('/')[-1]}_prefix_{app_hash}.pkl")
    logger.info("Computing app embeddings (with prefix)...")
    app_emb_with_prefix = get_or_compute_embeddings(
        app_texts, app_cache_pfx, bi_encoder, batch_size=batch_size,
        prefix=BGE_QUERY_PREFIX
    )

    app_embeddings_cache = {
        False: app_emb_no_prefix,
        True: app_emb_with_prefix,
    }

    # Load cross-encoder once
    logger.info(f"Loading cross-encoder: {cross_encoder_model}")
    ce_model = CrossEncoder(cross_encoder_model)

    # ── Pre-compute task embeddings for each DWA setting ──
    # (task embeddings change with max_dwas but NOT with dense_k/sparse_k)
    task_data_cache = {}  # max_dwas -> (task_groups, task_embeddings, bm25_index, task_texts_plain)

    for max_dwas in set(max_dwas_values):
        logger.info(f"Pre-computing task data for max_dwas={max_dwas}...")

        if max_dwas == 0:
            # No DWA enrichment — use plain task + occupation only
            enriched = onet_tasks.copy()
            enriched['dwa_titles'] = [[] for _ in range(len(enriched))]
            enriched['enriched_text'] = enriched.apply(
                lambda r: f"{r['Task']} | Occupation: {r['Title']}", axis=1
            )
        else:
            enriched = enrich_onet_tasks(onet_tasks, tasks_to_dwas, dwa_ref,
                                         max_dwas_per_task=max_dwas)

        task_groups = enriched.groupby('task_id').agg({
            'Task': 'first',
            'Task Type': 'first',
            'enriched_text': lambda x: ' | '.join(x.unique()),
        }).reset_index()

        enriched_texts = task_groups['enriched_text'].tolist()
        plain_texts = task_groups['Task'].tolist()

        # Embeddings
        task_hash = hashlib.md5("".join(sorted(enriched_texts)).encode()).hexdigest()[:12]
        task_cache_path = os.path.join(embeddings_dir,
                                        f"rag_tasks_enriched_{model_name.split('/')[-1]}_dwa{max_dwas}_{task_hash}.pkl")
        task_emb = get_or_compute_embeddings(
            enriched_texts, task_cache_path, bi_encoder, batch_size=batch_size
        )

        # BM25 index
        task_tokens = [tokenize_for_bm25(t) for t in enriched_texts]
        bm25 = BM25Okapi(task_tokens) if HAS_BM25 else None

        task_data_cache[max_dwas] = (task_groups, task_emb, bm25, plain_texts)

    # ── Run grid ──
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for config_idx, (dense_k, sparse_k, max_dwas, use_prefix) in enumerate(configs):
        skip_sparse = (sparse_k == 0)
        pfx_tag = "pfx" if use_prefix else "nopfx"
        config_name = f"dk{dense_k}_sk{sparse_k}_dwa{max_dwas}_{pfx_tag}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Config {config_idx+1}/{len(configs)}: {config_name}")
        logger.info(f"{'='*60}")

        task_groups, task_emb, bm25, plain_texts = task_data_cache[max_dwas]
        app_embeddings = app_embeddings_cache[use_prefix]

        # Retrieval
        candidates = hybrid_retrieve_batch(
            app_embeddings=app_embeddings,
            app_texts=app_texts,
            task_embeddings=task_emb,
            task_texts_tokenized=None,
            bm25_index=bm25 if not skip_sparse else None,
            dense_k=dense_k,
            sparse_k=sparse_k,
            rrf_k=60,
            final_k=dense_k + sparse_k,  # Keep enough candidates so BM25-unique finds survive into CE
            use_sparse=not skip_sparse
        )

        total_candidates = sum(len(v) for v in candidates.values())

        # Cross-encoder reranking (all candidates)
        pairs = []
        pair_keys = []
        for app_idx, task_candidates in candidates.items():
            for task_idx, rrf_score in task_candidates:
                pairs.append((app_texts[app_idx], plain_texts[task_idx]))
                pair_keys.append((app_idx, task_idx, rrf_score))

        logger.info(f"  Reranking {len(pairs):,} pairs...")
        all_scores = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            scores = ce_model.predict(batch, show_progress_bar=False)
            if isinstance(scores, (int, float)):
                scores = [scores]
            elif hasattr(scores, 'tolist'):
                scores = scores.tolist()
            all_scores.extend(scores)

        # Build per-app results
        app_ce_scores = defaultdict(list)
        for (app_idx, task_idx, rrf_score), ce_score in zip(pair_keys, all_scores):
            app_ce_scores[app_idx].append(ce_score)

        # Flatten all CE scores
        all_ce = np.array(all_scores)

        # Compute metrics
        ce_thresholds_eval = [0.3, 0.5, 0.7, 0.9]
        row = {
            'config': config_name,
            'dense_k': dense_k,
            'sparse_k': sparse_k,
            'max_dwas': max_dwas,
            'use_prefix': use_prefix,
            'total_candidates': total_candidates,
            'total_pairs_reranked': len(pairs),
        }

        for t in ce_thresholds_eval:
            n_above = int((all_ce >= t).sum())
            n_apps_with = sum(1 for scores in app_ce_scores.values()
                              if any(s >= t for s in scores))
            row[f'n_matches_ce{t}'] = n_above
            row[f'n_apps_ce{t}'] = n_apps_with

        # Top-1 and top-3 CE per app
        top1_ces = []
        top3_ces = []
        for app_idx in range(n_apps):
            scores = sorted(app_ce_scores.get(app_idx, [0.0]), reverse=True)
            top1_ces.append(scores[0])
            top3_ces.append(np.mean(scores[:3]))

        row['mean_top1_ce'] = float(np.mean(top1_ces))
        row['mean_top3_ce'] = float(np.mean(top3_ces))
        row['median_top1_ce'] = float(np.median(top1_ces))

        results.append(row)

        # Log summary
        logger.info(f"  Candidates: {total_candidates:,} | Reranked: {len(pairs):,}")
        for t in ce_thresholds_eval:
            logger.info(f"  CE>={t}: {row[f'n_matches_ce{t}']:>5} matches, "
                        f"{row[f'n_apps_ce{t}']:>4}/{n_apps} apps")
        logger.info(f"  Mean top-1 CE: {row['mean_top1_ce']:.4f} | "
                    f"Mean top-3 CE: {row['mean_top3_ce']:.4f}")

    # ── Save and display results ──
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_dir, "grid_search_results.csv")
    results_df.to_csv(results_file, index=False)
    logger.info(f"\nGrid search results saved to {results_file}")

    # Sort by primary metric: n_matches_ce0.5 (number of confident matches)
    results_df = results_df.sort_values('n_matches_ce0.5', ascending=False)

    logger.info("\n" + "=" * 100)
    logger.info("GRID SEARCH RESULTS (sorted by n_matches at CE >= 0.5)")
    logger.info("=" * 100)
    display_cols = ['config', 'total_candidates',
                    'n_matches_ce0.3', 'n_matches_ce0.5', 'n_matches_ce0.7', 'n_matches_ce0.9',
                    'n_apps_ce0.5', 'mean_top1_ce', 'mean_top3_ce']
    print(results_df[display_cols].to_string(index=False))

    # Best config
    best = results_df.iloc[0]
    logger.info(f"\nBest config: {best['config']}")
    logger.info(f"  dense_k={int(best['dense_k'])}, sparse_k={int(best['sparse_k'])}, "
                f"max_dwas={int(best['max_dwas'])}")
    logger.info(f"  {int(best['n_matches_ce0.5'])} matches at CE>=0.5, "
                f"{int(best['n_apps_ce0.5'])}/{n_apps} apps with a match")

    return results_df


def main():
    args = parse_args()

    if args.grid_search:
        logger.info("=" * 80)
        logger.info("Stage 4 (RAG): GRID SEARCH MODE")
        logger.info("=" * 80)

        run_grid_search(
            step3_file=args.step3_file,
            onet_file=args.onet_file,
            tasks_to_dwas_file=args.tasks_to_dwas_file,
            dwa_reference_file=args.dwa_reference_file,
            output_dir=args.output_dir,
            embeddings_dir=args.embeddings_dir,
            text_column=args.content_column,
            task_type=args.task_type,
            include_soc_15=args.include_soc_15,
            onet_version=args.onet_version,
            batch_size=args.batch_size,
            cross_encoder_model=args.cross_encoder_model,
            model_name=args.model_name
        )
        return

    logger.info("=" * 80)
    logger.info("Stage 4 (RAG): Hybrid Retrieval + Reranking Pipeline")
    logger.info("=" * 80)
    logger.info(f"  Step 3 file: {args.step3_file}")
    logger.info(f"  O*NET file: {args.onet_file}")
    logger.info(f"  Tasks-to-DWAs file: {args.tasks_to_dwas_file}")
    logger.info(f"  DWA Reference file: {args.dwa_reference_file}")
    logger.info(f"  Task type: {args.task_type}")
    logger.info(f"  Dense model: {args.model_name}")
    logger.info(f"  Cross-encoder: {args.cross_encoder_model}")
    logger.info(f"  Dense k: {args.dense_k}, Sparse k: {args.sparse_k}")
    logger.info(f"  RRF k: {args.rrf_k}")
    logger.info(f"  Skip CE: {args.skip_cross_encoder}, Skip sparse: {args.skip_sparse}")
    logger.info(f"  Output dir: {args.output_dir}")

    matcher = HybridRAGMatcher(
        model_name=args.model_name,
        cross_encoder_model=args.cross_encoder_model,
        embeddings_dir=args.embeddings_dir,
        batch_size=args.batch_size,
        dense_k=args.dense_k,
        sparse_k=args.sparse_k,
        rrf_k=args.rrf_k,
        skip_cross_encoder=args.skip_cross_encoder,
        skip_sparse=args.skip_sparse
    )

    matches_df = matcher.run(
        step3_file=args.step3_file,
        onet_file=args.onet_file,
        tasks_to_dwas_file=args.tasks_to_dwas_file,
        dwa_reference_file=args.dwa_reference_file,
        output_dir=args.output_dir,
        text_column=args.content_column,
        task_type=args.task_type,
        include_soc_15=args.include_soc_15,
        onet_version=args.onet_version,
        ce_thresholds=args.ce_thresholds,
        max_dwas_per_task=args.max_dwas_per_task
    )

    logger.info("Pipeline complete.")


if __name__ == '__main__':
    main()
