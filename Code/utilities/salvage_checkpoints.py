#!/usr/bin/env python3
"""
Salvage existing checkpoint data by converting positional indices to content-based keys.

This script:
1. Regenerates the similarity_df with the same parameters as the original checkpoints
2. Loads existing checkpoint files
3. Maps each checkpoint index to (app_text, onet_task_id) content keys
4. Saves new content-based checkpoint files for the refactored system

The deterministic hash proves similarity_df will be identical in order and size.
"""

import pandas as pd
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def regenerate_similarity_df(step3_file: str = None):
    """Regenerate the similarity DataFrame using the same logic as stage_4."""
    logger.info("Regenerating similarity_df from embeddings...")

    # Load AI applications
    if step3_file:
        app_file = Path(step3_file)
    else:
        # Try to find the file used in the checkpoint run
        app_file = Path("Data/Testing/stage_3/1000_company_test/final_output_1000_sample.csv")
        if not app_file.exists():
            app_file = Path("Data/ai_development_deduplicated_custom.csv")

    logger.info(f"Loading AI applications from {app_file}")
    df_apps = pd.read_csv(app_file)
    app_texts = df_apps['step3_output'].tolist()

    # Load O*NET tasks (you used task_statements_20.xlsx)
    onet_file = Path("Data/task_statements_20.xlsx")
    logger.info(f"Loading O*NET tasks from {onet_file}")
    df_tasks = pd.read_excel(onet_file)
    onet_task_ids = df_tasks['Task ID'].tolist()

    # Verify counts match the checkpoint hash
    expected_hash_content = f"BAAI/bge-reranker-v2-m3|{len(app_texts)}|{len(onet_task_ids)}"
    expected_hash = hashlib.md5(expected_hash_content.encode()).hexdigest()[:8]
    logger.info(f"Expected hash: {expected_hash}")
    logger.info(f"Counts: {len(app_texts)} apps × {len(onet_task_ids)} tasks = {len(app_texts) * len(onet_task_ids):,} pairs")

    # Create similarity DataFrame (same structure as stage_4)
    similarity_pairs = []
    for i, app_text in enumerate(app_texts):
        for j, onet_task_id in enumerate(onet_task_ids):
            similarity_pairs.append({
                'app_text': app_text,
                'onet_task': onet_task_id,
                'index_pair': (i, j)  # Keep track of app/task indices
            })

    similarity_df = pd.DataFrame(similarity_pairs)
    logger.info(f"Created similarity_df with {len(similarity_df):,} rows")

    return similarity_df, app_texts, onet_task_ids


def salvage_checkpoints(similarity_df: pd.DataFrame, app_texts, onet_task_ids):
    """Convert existing checkpoints from positional indices to content-based keys."""

    checkpoint_dir = Path("Data/embeddings/cross_encoder_checkpoints")
    salvaged_dir = checkpoint_dir / "salvaged"
    salvaged_dir.mkdir(exist_ok=True)

    total_rows = 0

    for checkpoint_file in sorted(checkpoint_dir.glob("ce_worker*.parquet")):
        if checkpoint_file.parent.name == "salvaged":
            continue  # Skip already salvaged files

        logger.info(f"\nProcessing {checkpoint_file.name}...")

        # Load checkpoint with positional indices
        df_checkpoint = pd.read_parquet(checkpoint_file)
        metadata = dict(df_checkpoint.attrs) if hasattr(df_checkpoint, 'attrs') else {}

        logger.info(f"  Rows: {len(df_checkpoint):,}")
        logger.info(f"  Global indices range: {df_checkpoint['global_index'].min():,} - {df_checkpoint['global_index'].max():,}")

        # Map positional indices to content keys
        content_keys = []
        scores = []
        invalid_count = 0

        for _, row in df_checkpoint.iterrows():
            global_idx = int(row['global_index'])

            # Validate index is in range
            if global_idx < 0 or global_idx >= len(similarity_df):
                invalid_count += 1
                continue

            # Get content from similarity_df at this index
            sim_row = similarity_df.iloc[global_idx]
            app_text = sim_row['app_text']
            onet_task = sim_row['onet_task']

            content_keys.append((app_text, onet_task))
            scores.append(row['cross_encoder_score'])

        if invalid_count > 0:
            logger.warning(f"  Skipped {invalid_count:,} invalid indices")

        # Create new DataFrame with content-based keys
        df_salvaged = pd.DataFrame({
            'app_text': [k[0] for k in content_keys],
            'onet_task_id': [k[1] for k in content_keys],
            'cross_encoder_score': scores
        })

        # Save with metadata
        salvaged_file = salvaged_dir / checkpoint_file.name
        df_salvaged.to_parquet(salvaged_file)

        # Preserve metadata
        if hasattr(df_salvaged, 'attrs'):
            df_salvaged.attrs = metadata

        logger.info(f"  Saved {len(df_salvaged):,} content-keyed rows to {salvaged_file.name}")
        total_rows += len(df_salvaged)

    logger.info(f"\n✓ Total rows salvaged: {total_rows:,}")
    logger.info(f"  Salvaged checkpoints saved to: {salvaged_dir}")
    logger.info(f"  Next: Update stage_4_onet_similarity.py to load from salvaged/ dir")


if __name__ == "__main__":
    import sys

    logger.info("=" * 60)
    logger.info("CHECKPOINT SALVAGE SCRIPT")
    logger.info("=" * 60)

    # Optional: specify step3 file as command-line argument
    step3_file = sys.argv[1] if len(sys.argv) > 1 else None

    # Step 1: Regenerate similarity_df
    similarity_df, app_texts, onet_task_ids = regenerate_similarity_df(step3_file)

    # Step 2: Salvage checkpoints
    salvage_checkpoints(similarity_df, app_texts, onet_task_ids)

    logger.info("\n" + "=" * 60)
    logger.info("SALVAGE COMPLETE")
    logger.info("=" * 60)
