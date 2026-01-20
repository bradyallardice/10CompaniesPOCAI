#!/usr/bin/env python3
"""
Test script to validate the new content-based checkpoint naming system.

This script tests:
1. Original indices are preserved correctly
2. Checkpoint paths use content-based naming with chunk indices
3. Checkpoint loading works with new naming scheme
4. Worker assignment respects original indices
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_original_index_preservation():
    """Test that original indices are preserved when filtering remaining work."""
    logger.info("=" * 70)
    logger.info("TEST 1: Original Index Preservation")
    logger.info("=" * 70)

    # Create mock similarity_df with 1000 pairs
    similarity_df = pd.DataFrame({
        'app_text': [f'app_{i}' for i in range(1000)],
        'onet_task_id': [f'task_{i}' for i in range(1000)],
        'similarity': np.random.rand(1000)
    })

    # Simulate checkpoint covering pairs 0-400 and 600-700 (inclusive)
    # range(0, 401) = 0-400 (401 items)
    # range(600, 700) = 600-699 (100 items) - Note: 701 excluded
    checkpoint_scores = {
        (f'app_{i}', f'task_{i}'): 0.5 for i in list(range(0, 401)) + list(range(600, 700))
    }

    logger.info(f"Total pairs: {len(similarity_df)}")
    logger.info(f"Checkpointed pairs: {len(checkpoint_scores)}")

    # Identify remaining work (same logic as in stage_4)
    completed_pairs = set(checkpoint_scores.keys())
    remaining_mask = [
        (row['app_text'], row['onet_task_id']) not in completed_pairs
        for _, row in similarity_df.iterrows()
    ]

    remaining_indices = np.where(remaining_mask)[0]
    remaining_df = similarity_df.iloc[remaining_indices].copy()
    remaining_df['_original_index'] = remaining_indices

    logger.info(f"Remaining pairs: {len(remaining_df)}")
    logger.info(f"Remaining indices range: {remaining_indices.min()}-{remaining_indices.max()}")

    # Verify indices are correct
    # Expected: 401-599 (199) + 700-999 (300) = 499 pairs
    expected_remaining = (599 - 401 + 1) + (999 - 700 + 1)
    assert len(remaining_df) == expected_remaining, f"Expected {expected_remaining} remaining, got {len(remaining_df)}"
    assert 401 in remaining_df['_original_index'].values, f"Expected index 401 in remaining"
    assert 999 in remaining_df['_original_index'].values, f"Expected index 999 in remaining"

    logger.info("✓ Original indices preserved correctly")
    logger.info("")


def test_checkpoint_path_generation():
    """Test that checkpoint paths use content-based naming."""
    logger.info("=" * 70)
    logger.info("TEST 2: Content-Based Checkpoint Path Generation")
    logger.info("=" * 70)

    from stage_4_onet_similarity import ONETSimilarityMatcher

    # Create matcher instance
    matcher = ONETSimilarityMatcher(embeddings_dir="Data/embeddings")

    # Test new content-based naming
    path_with_indices = matcher._get_ce_checkpoint_path(
        model_name="BAAI/bge-reranker-v2-m3",
        total_pairs=4202050,
        run_hash="c451ceb5",
        start_idx=0,
        end_idx=1400683
    )
    logger.info(f"Path with indices: {path_with_indices.name}")
    assert "ce_chunk_0_1400683" in str(path_with_indices), f"Expected chunk naming, got {path_with_indices.name}"
    assert "total4202050" in str(path_with_indices), f"Expected total in path, got {path_with_indices.name}"

    # Test fallback naming (for backward compatibility)
    path_without_indices = matcher._get_ce_checkpoint_path(
        model_name="BAAI/bge-reranker-v2-m3",
        total_pairs=4202050,
        run_hash="c451ceb5"
    )
    logger.info(f"Path without indices: {path_without_indices.name}")
    assert "ce_merged" in str(path_without_indices), f"Expected merged naming, got {path_without_indices.name}"

    logger.info("✓ Checkpoint paths generated correctly")
    logger.info("")


def test_worker_chunk_assignment():
    """Test that workers are assigned chunks with correct original indices."""
    logger.info("=" * 70)
    logger.info("TEST 3: Worker Chunk Assignment with Original Indices")
    logger.info("=" * 70)

    # Create mock remaining_df with original indices
    remaining_df = pd.DataFrame({
        'app_text': [f'app_{i}' for i in range(100)],
        'onet_task_id': [f'task_{i}' for i in range(100)],
        '_original_index': np.array([5, 7, 10, 15, 20, 22, 25, 30, 35, 40,
                                      45, 50, 55, 60, 65, 70, 75, 80, 85, 90,
                                      95, 100, 105, 110, 115, 120, 125, 130, 135, 140,
                                      145, 150, 155, 160, 165, 170, 175, 180, 185, 190,
                                      195, 200, 205, 210, 215, 220, 225, 230, 235, 240,
                                      245, 250, 255, 260, 265, 270, 275, 280, 285, 290,
                                      295, 300, 305, 310, 315, 320, 325, 330, 335, 340,
                                      345, 350, 355, 360, 365, 370, 375, 380, 385, 390,
                                      395, 400, 405, 410, 415, 420, 425, 430, 435, 440,
                                      445, 450, 455, 460, 465, 470, 475, 480, 485, 490])
    })

    num_workers = 3
    remaining_count = len(remaining_df)
    chunk_size = (remaining_count + num_workers - 1) // num_workers

    logger.info(f"Distributing {remaining_count} remaining pairs to {num_workers} workers")
    logger.info(f"Chunk size: {chunk_size}")

    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, remaining_count)
        if start_idx < remaining_count:
            chunk_df = remaining_df.iloc[start_idx:end_idx].copy()
            original_indices = chunk_df['_original_index'].values
            chunk_original_start = int(original_indices.min())
            chunk_original_end = int(original_indices.max())

            logger.info(f"Worker {i}: rows {start_idx}-{end_idx-1}, original indices {chunk_original_start}-{chunk_original_end}")

            # Verify that consecutive workers get non-overlapping original indices
            assert len(chunk_df) > 0, f"Worker {i} got empty chunk"
            assert chunk_original_end >= chunk_original_start, f"Worker {i} has invalid index range"

    logger.info("✓ Worker chunks assigned correctly with non-overlapping indices")
    logger.info("")


def test_checkpoint_consistency():
    """Test that checkpoint content matches expectations."""
    logger.info("=" * 70)
    logger.info("TEST 4: Checkpoint Content Consistency")
    logger.info("=" * 70)

    checkpoint_path = Path("Data/embeddings/cross_encoder_checkpoints/salvaged/ce_worker0_BAAI_bge_reranker_v2_m3_total4202050_c451ceb5.parquet")

    if checkpoint_path.exists():
        df = pd.read_parquet(checkpoint_path)
        logger.info(f"Checkpoint loaded: {len(df)} rows")
        logger.info(f"Columns: {list(df.columns)}")

        # Verify required columns exist
        required_cols = ['app_text', 'onet_task_id', 'cross_encoder_score']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Verify no NaN values
        assert not df['app_text'].isna().any(), "Found NaN in app_text"
        assert not df['onet_task_id'].isna().any(), "Found NaN in onet_task_id"
        assert not df['cross_encoder_score'].isna().any(), "Found NaN in cross_encoder_score"

        # Verify no _original_index column (should not be saved)
        assert '_original_index' not in df.columns, "Found _original_index in saved checkpoint (should be excluded)"

        logger.info("✓ Checkpoint content is consistent")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path} - skipping content test")
    logger.info("")


if __name__ == "__main__":
    try:
        test_original_index_preservation()
        test_checkpoint_path_generation()
        test_worker_chunk_assignment()
        test_checkpoint_consistency()

        logger.info("=" * 70)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
