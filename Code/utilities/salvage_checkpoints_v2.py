#!/usr/bin/env python3
"""
Salvage existing checkpoint data by converting positional indices to content-based keys.

This script uses the actual stage_4 code to ensure we get the exact same ordering.
"""

import pandas as pd
import sys
import logging
from pathlib import Path

# Import from stage_4 to reuse exact same logic
sys.path.insert(0, str(Path(__file__).parent))
from stage_4_onet_similarity import ONETSimilarityMatcher

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def salvage_checkpoints(step3_file, onet_file, task_type='core', use_openai=False):
    """
    Salvage checkpoints by:
    1. Using stage_4 code to load apps and tasks (ensures exact same ordering)
    2. Creating index-to-content mapping
    3. Converting checkpoint files from positional to content-based keys
    """

    logger.info("=" * 70)
    logger.info("CHECKPOINT SALVAGE - Using stage_4 code for exact ordering")
    logger.info("=" * 70)

    # Initialize matcher (uses same code as actual run)
    matcher = ONETSimilarityMatcher(
        step3_output_file=step3_file,
        onet_task_file=onet_file,
        use_openai_embeddings=use_openai,
        task_type=task_type
    )

    # Load embeddings (this gives us the exact ordering used in the run)
    logger.info("\nStep 1: Loading embeddings using stage_4 code...")
    app_embeddings = matcher._load_or_generate_embeddings(matcher.app_texts, 'apps')
    task_embeddings = matcher._load_or_generate_embeddings(matcher.onet_tasks, 'tasks')

    logger.info(f"  Loaded {len(matcher.app_texts)} app embeddings")
    logger.info(f"  Loaded {len(matcher.onet_tasks)} task embeddings")

    # Get lists in exact same order as stage_4 creates them
    app_texts = matcher.app_texts
    onet_task_ids = matcher.onet_task_ids
    onet_tasks = matcher.onet_tasks

    total_pairs = len(app_texts) * len(onet_tasks)
    logger.info(f"  Total pairs: {total_pairs:,}")

    # Generate hash (same method as stage_4)
    import hashlib
    hash_content = f"BAAI/bge-reranker-v2-m3|{len(app_texts)}|{len(onet_tasks)}"
    expected_hash = hashlib.md5(hash_content.encode()).hexdigest()[:8]
    logger.info(f"  Expected hash: {expected_hash}")

    # Step 2: Create index lookup function
    logger.info("\nStep 2: Creating index-to-content mapping...")

    def global_index_to_content(idx):
        """Convert global index to (app_text, onet_task_id) pair."""
        n_tasks = len(onet_tasks)
        app_idx = idx // n_tasks
        task_idx = idx % n_tasks

        if app_idx >= len(app_texts) or task_idx >= len(onet_tasks):
            return None

        return (app_texts[app_idx], onet_task_ids[task_idx])

    # Step 3: Load and convert checkpoints
    logger.info("\nStep 3: Converting checkpoint files...")

    checkpoint_dir = Path("Data/embeddings/cross_encoder_checkpoints")
    salvaged_dir = checkpoint_dir / "salvaged"
    salvaged_dir.mkdir(exist_ok=True)

    total_salvaged = 0
    total_invalid = 0

    for checkpoint_file in sorted(checkpoint_dir.glob("ce_worker*.parquet")):
        if "salvaged" in str(checkpoint_file):
            continue

        logger.info(f"\n  Processing {checkpoint_file.name}...")

        # Load checkpoint
        df_checkpoint = pd.read_parquet(checkpoint_file)
        logger.info(f"    Loaded {len(df_checkpoint):,} rows")

        # Convert indices to content keys
        content_data = []
        invalid_count = 0

        for _, row in df_checkpoint.iterrows():
            global_idx = int(row['global_index'])
            content_key = global_index_to_content(global_idx)

            if content_key is None:
                invalid_count += 1
                continue

            app_text, task_id = content_key
            content_data.append({
                'app_text': app_text,
                'onet_task_id': task_id,
                'cross_encoder_score': row['cross_encoder_score']
            })

        if invalid_count > 0:
            logger.warning(f"    Skipped {invalid_count:,} invalid indices")
            total_invalid += invalid_count

        # Save converted checkpoint
        df_salvaged = pd.DataFrame(content_data)
        salvaged_file = salvaged_dir / checkpoint_file.name
        df_salvaged.to_parquet(salvaged_file)

        logger.info(f"    Saved {len(df_salvaged):,} content-keyed rows")
        total_salvaged += len(df_salvaged)

    logger.info("\n" + "=" * 70)
    logger.info(f"SALVAGE COMPLETE")
    logger.info(f"  Total rows salvaged: {total_salvaged:,}")
    logger.info(f"  Invalid indices skipped: {total_invalid:,}")
    logger.info(f"  Salvaged files: {salvaged_dir}")
    logger.info("=" * 70)

    return total_salvaged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Salvage checkpoint data")
    parser.add_argument("--step3-file", required=True, help="Path to step3 output CSV")
    parser.add_argument("--onet-file", required=True, help="Path to O*NET task file")
    parser.add_argument("--task-type", default="core", help="Task type (core/all)")
    parser.add_argument("--use-openai-embeddings", action="store_true", help="Use OpenAI embeddings")

    args = parser.parse_args()

    salvage_checkpoints(
        step3_file=args.step3_file,
        onet_file=args.onet_file,
        task_type=args.task_type,
        use_openai=args.use_openai_embeddings
    )
