#!/usr/bin/env python3
"""
OpenAI Text Embedding Generation Script

Generates and caches OpenAI text-embedding-3-large embeddings for AI applications
and O*NET tasks. These embeddings can later be used by Stage 4 as an alternative to BGE.

**Integration with Stage 4**: Stage 4 now automatically calls this module when
`--use-openai-embeddings` is used. It checks cache first, and if missing, generates
embeddings with user approval. This script remains available for manual pre-generation
or regeneration of corrupted caches.

Features:
- Cache-first: Always loads from cache by default, never calls API unless needed
- Batched API calls: Processes texts in configurable batches with exponential backoff
- Cost tracking: Estimates and logs API costs
- Multiple task types: Supports core_tasks, all_tasks, and both
- Check-only mode: Verify cache status without generating embeddings
- Force regeneration: Rebuild cache even if it already exists

Usage (Standalone):
    # Generate embeddings for AI apps and core tasks
    export OPENAI_API_KEY="sk-..."
    python3 generate_openai_embeddings.py \\
        --step3-file Data/final_output_step3.csv \\
        --onet-version 20 \\
        --task-type core

    # Load from cache (default, no API calls)
    python3 generate_openai_embeddings.py \\
        --step3-file Data/final_output_step3.csv \\
        --onet-version 20 \\
        --task-type core

    # Check cache status without generating
    python3 generate_openai_embeddings.py \\
        --step3-file Data/final_output_step3.csv \\
        --onet-version 20 \\
        --task-type core \\
        --check-only

    # Force regenerate embeddings (overwrites cache)
    python3 generate_openai_embeddings.py \\
        --step3-file Data/final_output_step3.csv \\
        --onet-version 20 \\
        --task-type core \\
        --force-regenerate

Usage (Via Stage 4):
    # Stage 4 automatically generates embeddings if cache missing
    python3 stage_4_onet_similarity.py \\
        --use-openai-embeddings \\
        --step3-file Data/test.csv \\
        --onet-version 20

When to use standalone script:
- Pre-generating embeddings before running Stage 4
- Regenerating corrupted cache files
- Testing with different O*NET versions or task types
- Manual cost inspection and approval workflow
"""

import pandas as pd
import numpy as np
import os
import logging
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import argparse
import pickle
import hashlib
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from config.env
load_dotenv('config.env', override=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OpenAI API model and pricing
OPENAI_MODEL = "text-embedding-3-large"
OPENAI_DIMS = 3072
OPENAI_PRICING_PER_MTK = 0.13  # $0.13 per 1M tokens

# Export list for module import
__all__ = [
    # Core embedding functions
    'embed_with_openai_api',
    'save_embedding_cache',
    'load_embedding_cache',
    'generate_cache_key',
    'get_openai_cache_path',
    # Data loading functions
    'load_ai_applications',
    'load_onet_tasks',
    'canonicalize_text',
    # Utility functions
    'check_cache_status',
    # Constants
    'OPENAI_MODEL',
    'OPENAI_DIMS',
    'OPENAI_PRICING_PER_MTK',
]


def generate_cache_key(texts: List[str]) -> str:
    """
    Generate a deterministic hash key for a list of texts.

    Args:
        texts: List of text strings

    Returns:
        8-character MD5 hash of sorted text list
    """
    sorted_texts = ''.join(sorted(texts))
    return hashlib.md5(sorted_texts.encode()).hexdigest()[:8]


def get_openai_cache_path(embeddings_dir: Path, text_type: str, hash_key: str) -> Path:
    """
    Get cache file path for OpenAI embeddings.

    Args:
        embeddings_dir: Directory to store embeddings
        text_type: Either "apps" or "tasks"
        hash_key: Hash of text content

    Returns:
        Path object for cache file
    """
    cache_name = f"openai_{OPENAI_MODEL}_{text_type}_{hash_key}.pkl"
    return embeddings_dir / cache_name


def load_embedding_cache(cache_path: Path) -> Optional[Dict]:
    """
    Load embeddings from cache file.

    Args:
        cache_path: Path to pickle file

    Returns:
        Dictionary with 'embeddings' and 'metadata' keys, or None if cache doesn't exist
    """
    if not cache_path.exists():
        logger.info(f"Cache not found: {cache_path.name}")
        return None

    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        metadata = cache_data.get('metadata', {})
        embeddings = cache_data.get('embeddings', {})

        logger.info(f"Loaded cache: {cache_path.name}")
        logger.info(f"  Model: {metadata.get('model')}, Dimensions: {metadata.get('dimensions')}")
        logger.info(f"  Texts: {metadata.get('num_texts'):,}, Tokens: {metadata.get('total_tokens'):,}")
        logger.info(f"  Cost: ${metadata.get('cost_usd', 0):.4f}")

        return cache_data
    except Exception as e:
        logger.warning(f"Failed to load cache {cache_path.name}: {e}")
        return None


def save_embedding_cache(embeddings: Dict[str, np.ndarray],
                        cache_path: Path,
                        total_tokens: int,
                        text_type: str) -> None:
    """
    Save embeddings to cache file.

    Args:
        embeddings: Dictionary mapping text -> embedding array
        cache_path: Path to save pickle file
        total_tokens: Total tokens used for cost tracking
        text_type: Type of text ("apps" or "tasks")
    """
    cost_usd = (total_tokens / 1_000_000) * OPENAI_PRICING_PER_MTK

    cache_data = {
        'embeddings': embeddings,
        'metadata': {
            'model': OPENAI_MODEL,
            'dimensions': OPENAI_DIMS,
            'timestamp': datetime.now().isoformat(),
            'num_texts': len(embeddings),
            'total_tokens': total_tokens,
            'cost_usd': cost_usd,
            'text_type': text_type,
            'api_version': 'v1'
        }
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)

    logger.info(f"Saved cache: {cache_path.name}")
    logger.info(f"  Texts: {len(embeddings):,}, Tokens: {total_tokens:,}")
    logger.info(f"  Cost: ${cost_usd:.4f}")


def embed_with_openai_api(texts: List[str],
                         batch_size: int = 100,
                         max_retries: int = 5) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Generate embeddings using OpenAI API with batching and retry logic.

    Args:
        texts: List of texts to embed
        batch_size: Number of texts per API call
        max_retries: Maximum retries for rate limit errors

    Returns:
        Tuple of (embeddings_dict, total_tokens)

    Raises:
        ImportError: If OpenAI package not installed
        ValueError: If OPENAI_API_KEY not set
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")

    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set. "
                        "Set it with: export OPENAI_API_KEY='sk-...'")

    client = OpenAI(api_key=api_key)
    embeddings = {}
    total_tokens = 0

    logger.info(f"Embedding {len(texts):,} texts with {OPENAI_MODEL}")
    logger.info(f"  Batch size: {batch_size}, Max retries: {max_retries}")

    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_num = i // batch_size + 1

        logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)")

        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = client.embeddings.create(
                    input=batch,
                    model=OPENAI_MODEL
                )

                # Extract embeddings
                for j, embedding_obj in enumerate(response.data):
                    text = batch[j]
                    embeddings[text] = np.array(embedding_obj.embedding, dtype=np.float32)

                # Track token usage
                batch_tokens = response.usage.total_tokens
                total_tokens += batch_tokens

                logger.info(f"    Batch {batch_num} complete: {batch_tokens} tokens")
                break  # Success, move to next batch

            except Exception as e:
                error_str = str(e).lower()
                if 'rate_limit' in error_str and retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2 ** retry_count  # Exponential backoff
                    logger.warning(f"    Rate limit, retrying in {wait_time}s "
                                 f"(attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"    Batch {batch_num} failed: {e}")
                    raise

    logger.info(f"Embedding complete: {len(embeddings):,} texts, {total_tokens:,} tokens")
    cost_usd = (total_tokens / 1_000_000) * OPENAI_PRICING_PER_MTK
    logger.info(f"  Estimated cost: ${cost_usd:.4f}")

    return embeddings, total_tokens


def canonicalize_text(text: str) -> str:
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


def load_ai_applications(step3_file: Path) -> List[str]:
    """
    Load and deduplicate AI applications from Stage 3 output using Stage 4 logic.

    Replicates the exact deduplication approach from stage_4_onet_similarity_pertask.py:
    1. Load step3_output column
    2. Canonicalize text (lowercase, normalize quotes/whitespace)
    3. Exact hash deduplication (keep first occurrence)
    4. Return unique deduplicated strings

    Args:
        step3_file: Path to Stage 3 CSV output

    Returns:
        List of deduplicated AI application texts
    """
    logger.info(f"Loading AI applications from {step3_file}")

    if not step3_file.exists():
        raise FileNotFoundError(f"Stage 3 file not found: {step3_file}")

    df = pd.read_csv(step3_file)

    if 'step3_output' not in df.columns:
        raise ValueError(f"Expected 'step3_output' column in {step3_file}")

    logger.info(f"  Loaded {len(df):,} records from Stage 3 output")

    # Canonicalize text
    df = df.copy()
    df['canonicalized'] = df['step3_output'].apply(canonicalize_text)

    # Remove empty strings
    df = df[df['canonicalized'] != ''].copy()
    initial_count = len(df)
    logger.info(f"  After removing empty strings: {initial_count:,} records")

    # Exact hash deduplication (same as Stage 4)
    df['text_hash'] = df['canonicalized'].apply(lambda x: hashlib.md5(x.encode()).hexdigest())

    # Keep first occurrence of each hash
    dedup_exact = df.groupby('text_hash').agg({
        'canonicalized': 'first'
    }).reset_index()

    unique_apps = dedup_exact['canonicalized'].tolist()
    logger.info(f"  After exact deduplication: {len(unique_apps):,} unique AI applications")

    return unique_apps


def load_onet_tasks(onet_version: int, task_type: str = 'core',
                   output_dir: str = 'Data', include_soc_15: bool = False) -> List[str]:
    """
    Load O*NET task statements from Excel file for embedding.

    IMPORTANT: This function ALWAYS loads the FULL set of tasks (including Supplemental
    and SOC 15 by default) to match Stage 4's behavior. The task_type parameter is
    accepted for compatibility but is currently ignored for OpenAI embeddings.

    Stage 4 always loads all 19,530 tasks for embedding, then filters them AFTER
    computing similarities. To avoid hash mismatches, OpenAI embeddings must be
    generated for the same full task set.

    Args:
        onet_version: O*NET version number (e.g., 20)
        task_type: Ignored - kept for CLI compatibility (accepted but not used)
        output_dir: Directory where task_statements file is located (default: 'Data')
        include_soc_15: If True, include SOC group 15 tasks (default: True, to match Stage 4)

    Returns:
        List of O*NET task texts for embedding (full set: ~19,530 tasks)

    Raises:
        FileNotFoundError: If task_statements Excel file not found
        ValueError: If expected columns not found
    """
    onet_file_path = Path(output_dir) / f"task_statements_{onet_version}.xlsx"

    if not onet_file_path.exists():
        raise FileNotFoundError(f"O*NET task statements file not found: {onet_file_path}")

    logger.info(f"Loading O*NET tasks from: {onet_file_path}")
    logger.info(f"  NOTE: Always loading FULL task set (ignoring --task-type for compatibility with Stage 4)")

    try:
        # Read the Excel file (matches Stage 4 exactly)
        onet_df = pd.read_excel(onet_file_path)

        # Check for expected columns
        if 'Task' not in onet_df.columns:
            raise ValueError(f"Expected 'Task' column not found in {onet_file_path}")

        # Create task_id as row index (for consistency with Stage 4)
        onet_df = onet_df.reset_index()
        onet_df['task_id'] = onet_df.index

        # Clean and validate tasks
        onet_df = onet_df[onet_df['Task'].notna()].copy()
        onet_df['Task'] = onet_df['Task'].astype(str).str.strip()
        onet_df = onet_df[onet_df['Task'] != ''].copy()

        # ALWAYS include both Core and Supplemental tasks (to match Stage 4 behavior)
        logger.info(f"  Including both Core and Supplemental tasks")

        # Filter out SOC group 15 ONLY if explicitly requested (Stage 4 includes it by default)
        if not include_soc_15:
            initial_count = len(onet_df)
            onet_df = onet_df[~onet_df['O*NET-SOC Code'].str.startswith('15-', na=False)].copy()
            excluded_count = initial_count - len(onet_df)
            logger.info(f"  Excluded {excluded_count} tasks from SOC group 15 (Computer and Mathematical Occupations)")
        else:
            logger.info(f"  Including SOC group 15 (Computer and Mathematical Occupations) tasks")

        # Extract Task column for embedding
        task_texts = onet_df['Task'].tolist()
        logger.info(f"  Loaded {len(task_texts):,} O*NET task statements for embedding")

        return task_texts

    except Exception as e:
        logger.error(f"Error loading O*NET tasks from {onet_file_path}: {e}")
        raise


def check_cache_status(embeddings_dir: Path,
                      apps_hash: str,
                      tasks_hashes: Dict[str, str]) -> Dict[str, bool]:
    """
    Check which cache files exist.

    Args:
        embeddings_dir: Directory with cached embeddings
        apps_hash: Hash of AI applications
        tasks_hashes: Dictionary mapping task_type -> hash

    Returns:
        Dictionary mapping cache name -> exists (True/False)
    """
    status = {}

    # Check apps cache
    apps_cache = get_openai_cache_path(embeddings_dir, "apps", apps_hash)
    status['apps'] = apps_cache.exists()
    logger.info(f"Apps cache: {'EXISTS' if status['apps'] else 'MISSING'} ({apps_cache.name})")

    # Check task caches
    for task_type, task_hash in tasks_hashes.items():
        tasks_cache = get_openai_cache_path(embeddings_dir, "tasks", task_hash)
        status[f'tasks_{task_type}'] = tasks_cache.exists()
        logger.info(f"Tasks ({task_type}) cache: {'EXISTS' if status[f'tasks_{task_type}'] else 'MISSING'} ({tasks_cache.name})")

    return status


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Generate and cache OpenAI text-embedding-3-large embeddings for AI apps and O*NET tasks"
    )

    parser.add_argument('--step3-file',
                       type=Path,
                       default=Path("Data/Testing/stage_3/1000_company_test/final_output_1000_sample.csv"),
                       help='Path to Stage 3 output CSV (default: Data/Testing/stage_3/1000_company_test/final_output_1000_sample.csv)')

    parser.add_argument('--onet-version',
                       type=int,
                       default=20,
                       help='O*NET version number (default: 20)')

    parser.add_argument('--task-type',
                       choices=['core', 'all', 'both'],
                       default='core',
                       help='Which O*NET tasks to load (default: core)')

    parser.add_argument('--embeddings-dir',
                       type=Path,
                       default=Path("Data/embeddings"),
                       help='Directory to store embeddings (default: Data/embeddings)')

    parser.add_argument('--batch-size',
                       type=int,
                       default=100,
                       help='Batch size for API calls (default: 100)')

    parser.add_argument('--force-regenerate',
                       action='store_true',
                       help='Force regenerate embeddings even if cache exists')

    parser.add_argument('--check-only',
                       action='store_true',
                       help='Check cache status without generating embeddings')

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("OpenAI Embedding Generation Script")
    logger.info("="*60)

    # Load inputs
    logger.info("\nPhase 1: Loading inputs...")

    if not args.step3_file.exists():
        logger.error(f"Stage 3 file not found: {args.step3_file}")
        exit(1)

    try:
        apps = load_ai_applications(args.step3_file)
    except Exception as e:
        logger.error(f"Failed to load AI applications: {e}")
        exit(1)

    try:
        # Always load FULL task set (including SOC 15) to match Stage 4's behavior
        # This ensures hash compatibility with Stage 4's embedding caching
        tasks = load_onet_tasks(args.onet_version, args.task_type, include_soc_15=True)
    except Exception as e:
        logger.error(f"Failed to load O*NET tasks: {e}")
        exit(1)

    # Generate cache keys
    logger.info("\nPhase 2: Computing cache keys...")
    apps_hash = generate_cache_key(apps)
    logger.info(f"Apps hash: {apps_hash}")

    tasks_hashes = {}
    if args.task_type in ['core', 'both']:
        core_tasks = [t for t in tasks if not t.startswith("(SAL)")]  # Simple filter
        tasks_hashes['core'] = generate_cache_key(core_tasks)
        logger.info(f"Core tasks hash: {tasks_hashes['core']}")

    if args.task_type in ['all', 'both']:
        all_tasks = tasks
        tasks_hashes['all'] = generate_cache_key(all_tasks)
        logger.info(f"All tasks hash: {tasks_hashes['all']}")

    # Check cache status
    logger.info("\nPhase 3: Checking cache status...")
    cache_status = check_cache_status(args.embeddings_dir, apps_hash, tasks_hashes)

    all_cached = all(cache_status.values())

    if args.check_only:
        logger.info("\n--check-only flag set, exiting without generating")
        exit(0)

    # Load or generate embeddings
    logger.info("\nPhase 4: Loading/generating embeddings...")

    apps_embeddings = None
    apps_tokens = 0

    apps_cache_path = get_openai_cache_path(args.embeddings_dir, "apps", apps_hash)

    if cache_status['apps'] and not args.force_regenerate:
        logger.info("Loading apps embeddings from cache...")
        cache_data = load_embedding_cache(apps_cache_path)
        if cache_data:
            apps_embeddings = cache_data['embeddings']
            apps_tokens = cache_data['metadata'].get('total_tokens', 0)

    if apps_embeddings is None:
        logger.info("Generating apps embeddings via OpenAI API...")

        # Show cost warning
        estimated_tokens = len(apps) * 20  # Rough estimate
        estimated_cost = (estimated_tokens / 1_000_000) * OPENAI_PRICING_PER_MTK

        logger.warning("="*60)
        logger.warning("API CALL WARNING")
        logger.warning(f"  Texts to embed: {len(apps):,} AI applications")
        logger.warning(f"  Estimated tokens: ~{estimated_tokens:,}")
        logger.warning(f"  Estimated cost: ${estimated_cost:.4f}")
        logger.warning("  Press Ctrl+C to cancel, or wait 5 seconds to continue...")
        logger.warning("="*60)
        time.sleep(5)

        try:
            apps_embeddings, apps_tokens = embed_with_openai_api(apps, args.batch_size)
            save_embedding_cache(apps_embeddings, apps_cache_path, apps_tokens, "apps")
        except Exception as e:
            logger.error(f"Failed to generate apps embeddings: {e}")
            exit(1)

    # Load or generate task embeddings
    for task_type, task_hash in tasks_hashes.items():
        tasks_cache_path = get_openai_cache_path(args.embeddings_dir, "tasks", task_hash)

        tasks_embeddings = None
        tasks_tokens = 0

        if cache_status.get(f'tasks_{task_type}') and not args.force_regenerate:
            logger.info(f"Loading {task_type} tasks embeddings from cache...")
            cache_data = load_embedding_cache(tasks_cache_path)
            if cache_data:
                tasks_embeddings = cache_data['embeddings']
                tasks_tokens = cache_data['metadata'].get('total_tokens', 0)

        if tasks_embeddings is None:
            logger.info(f"Generating {task_type} tasks embeddings via OpenAI API...")

            # Get task list for this type
            if task_type == 'core':
                task_texts = [t for t in tasks if not t.startswith("(SAL)")]
            else:
                task_texts = tasks

            # Show cost warning
            estimated_tokens = len(task_texts) * 18  # Rough estimate
            estimated_cost = (estimated_tokens / 1_000_000) * OPENAI_PRICING_PER_MTK

            logger.warning("="*60)
            logger.warning(f"API CALL WARNING ({task_type} tasks)")
            logger.warning(f"  Texts to embed: {len(task_texts):,} O*NET tasks")
            logger.warning(f"  Estimated tokens: ~{estimated_tokens:,}")
            logger.warning(f"  Estimated cost: ${estimated_cost:.4f}")
            logger.warning("  Press Ctrl+C to cancel, or wait 5 seconds to continue...")
            logger.warning("="*60)
            time.sleep(5)

            try:
                tasks_embeddings, tasks_tokens = embed_with_openai_api(task_texts, args.batch_size)
                save_embedding_cache(tasks_embeddings, tasks_cache_path, tasks_tokens, f"tasks_{task_type}")
            except Exception as e:
                logger.error(f"Failed to generate {task_type} tasks embeddings: {e}")
                exit(1)

    logger.info("\n" + "="*60)
    logger.info("SUCCESS: All embeddings generated and cached")
    logger.info("="*60)
    logger.info(f"\nCache files saved in: {args.embeddings_dir}")
    logger.info("These can be loaded by Stage 4 with --use-openai-embeddings flag")


if __name__ == "__main__":
    main()
