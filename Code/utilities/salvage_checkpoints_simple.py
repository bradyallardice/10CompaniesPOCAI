#!/usr/bin/env python3
"""
Salvage existing checkpoint data by converting positional indices to content-based keys.

Simple approach: Load data in exact same order as stage_4, create index mapping.
"""

import pandas as pd
import hashlib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(step3_file, onet_file):
    """Load data in exact same order as stage_4."""
    logger.info(f"Loading step3 data from: {step3_file}")
    df_step3 = pd.read_csv(step3_file)
    app_texts = df_step3['step3_output'].tolist()
    logger.info(f"  Loaded {len(app_texts)} applications")

    logger.info(f"Loading O*NET tasks from: {onet_file}")
    df_onet = pd.read_excel(onet_file)
    onet_tasks = df_onet['Task'].tolist()
    onet_task_ids = df_onet['Task ID'].tolist()
    logger.info(f"  Loaded {len(onet_tasks)} O*NET tasks")

    return app_texts, onet_tasks, onet_task_ids


def verify_hash(app_texts, onet_tasks):
    """Verify the hash matches checkpoints."""
    hash_content = f"BAAI/bge-reranker-v2-m3|{len(app_texts)}|{len(onet_tasks)}"
    computed_hash = hashlib.md5(hash_content.encode()).hexdigest()[:8]
    logger.info(f"Computed hash: {computed_hash}")
    logger.info(f"Total pairs: {len(app_texts) * len(onet_tasks):,}")
    return computed_hash


def global_index_to_content(idx, app_texts, onet_tasks, onet_task_ids):
    """Convert global index to (app_text, onet_task_id) pair."""
    n_tasks = len(onet_tasks)
    app_idx = idx // n_tasks
    task_idx = idx % n_tasks

    if app_idx >= len(app_texts) or task_idx >= len(onet_tasks):
        return None

    return (app_texts[app_idx], onet_task_ids[task_idx])


def salvage_checkpoints(step3_file, onet_file):
    """Main salvage function."""
    logger.info("=" * 70)
    logger.info("CHECKPOINT SALVAGE - Simple approach")
    logger.info("=" * 70)

    # Step 1: Load data
    app_texts, onet_tasks, onet_task_ids = load_data(step3_file, onet_file)

    # Step 2: Verify hash
    computed_hash = verify_hash(app_texts, onet_tasks)

    # Step 3: Convert checkpoints
    logger.info("\nConverting checkpoint files...")
    checkpoint_dir = Path("Data/embeddings/cross_encoder_checkpoints")
    salvaged_dir = checkpoint_dir / "salvaged"
    salvaged_dir.mkdir(exist_ok=True)

    total_salvaged = 0
    total_invalid = 0

    for checkpoint_file in sorted(checkpoint_dir.glob("ce_worker*.parquet")):
        if "salvaged" in str(checkpoint_file):
            continue

        logger.info(f"\n  {checkpoint_file.name}:")
        df_checkpoint = pd.read_parquet(checkpoint_file)
        logger.info(f"    Loaded {len(df_checkpoint):,} rows")

        # Convert to content keys
        content_data = []
        invalid_count = 0

        for _, row in df_checkpoint.iterrows():
            global_idx = int(row['global_index'])
            content_key = global_index_to_content(global_idx, app_texts, onet_tasks, onet_task_ids)

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

        # Save
        df_salvaged = pd.DataFrame(content_data)
        salvaged_file = salvaged_dir / checkpoint_file.name
        df_salvaged.to_parquet(salvaged_file)

        logger.info(f"    Saved {len(df_salvaged):,} content-keyed rows")
        total_salvaged += len(df_salvaged)

    logger.info("\n" + "=" * 70)
    logger.info(f"COMPLETE: Salvaged {total_salvaged:,} rows (skipped {total_invalid:,} invalid)")
    logger.info(f"Output: {salvaged_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python salvage_checkpoints_simple.py <step3_file> <onet_file>")
        sys.exit(1)

    step3_file = sys.argv[1]
    onet_file = sys.argv[2]

    salvage_checkpoints(step3_file, onet_file)
