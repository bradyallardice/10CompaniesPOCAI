#!/usr/bin/env python3
"""
Fix corrupted checkpoint files using position-based remapping.

All checkpoint files were created in December 2025 with buggy code that
stored row positions instead of O*NET Task IDs. This script:
1. Reads corrupted files from original directories
2. Applies position-based remapping (treat stored value as row index)
3. Writes fixed files to new _fixed directories
4. Validates all Task IDs are in valid O*NET range
"""

import pandas as pd
from pathlib import Path
import sys


def build_onet_df(filter_type):
    """Build O*NET DataFrame with appropriate filtering.

    filter_type: 'core' for tasks14498 (no supplemental, no SOC 15)
                 'all' for tasks18840 (all tasks including supplemental and SOC 15)
    """
    onet_df = pd.read_excel('Data/task_statements_20.xlsx')
    onet_df['task_id'] = onet_df['Task ID'].astype(int)

    # Remove NaN/empty tasks
    onet_df = onet_df[onet_df['Task'].notna()].copy()
    onet_df['Task'] = onet_df['Task'].astype(str).str.strip()
    onet_df = onet_df[onet_df['Task'] != ''].copy()

    if filter_type == 'core':
        # Core tasks only: remove supplemental and SOC 15
        onet_df = onet_df[onet_df['Task Type'] != 'Supplemental'].copy()
        onet_df = onet_df[~onet_df['O*NET-SOC Code'].str.startswith('15-', na=False)].copy()
    elif filter_type == 'all':
        # All tasks: don't filter by Task Type or SOC
        pass
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    onet_df = onet_df.reset_index(drop=True)
    return onet_df[['task_id', 'Task']]


def detect_checkpoint_type(checkpoint_filename, df=None):
    """Detect which filtering was used based on checkpoint filename or data.

    If df is provided, inspects the Task IDs in the data to detect filtering.
    Otherwise, uses filename patterns.

    Returns: 'core' if CORE filtering (no supplemental, no SOC 15)
             'all' if ALL filtering (includes SOC 15)
    """
    # First try filename patterns
    if 'tasks14498' in checkpoint_filename:
        return 'core'
    elif 'tasks18840' in checkpoint_filename:
        return 'all'

    # If no filename pattern, analyze the data
    if df is not None and 'onet_task_id' in df.columns:
        # Check if any Task ID is > 13536 (max CORE) or looks like a position
        max_id = df['onet_task_id'].max()

        # If max ID > 13536, must be ALL filtering or row positions
        # If it looks like row positions (0-19529), we'll handle it with ALL
        if max_id > 13536:
            return 'all'
        else:
            return 'core'

    # Default to all if unknown (CE files use ALL)
    return 'all'


def fix_checkpoint_directory(input_dir, onet_df_all, onet_df_core):
    """Fix all checkpoints using position-based remapping.

    All files created Dec 2025 contain row positions instead of Task IDs.
    This function:
    1. Reads from input_dir
    2. Applies position-based remapping
    3. Writes to input_dir_fixed
    4. Validates results
    """

    if not input_dir.exists():
        print(f"\n  Input directory not found: {input_dir}")
        return 0, 0

    # Create output directory
    output_dir = input_dir.parent / f"{input_dir.name}_fixed"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_files = sorted(input_dir.glob('*.parquet'))
    checkpoint_files = [f for f in checkpoint_files if '.backup' not in str(f)]

    if len(checkpoint_files) == 0:
        print(f"\n  No checkpoint files found in: {input_dir}")
        return 0, 0

    print(f"\n  Found {len(checkpoint_files)} checkpoint files")
    print(f"  Reading from: {input_dir.name}")
    print(f"  Writing to:   {output_dir.name}")

    fixed_count = 0
    error_count = 0

    for checkpoint_file in checkpoint_files:
        try:
            print(f"\n    Processing: {checkpoint_file.name}")

            # Load checkpoint
            df = pd.read_parquet(checkpoint_file)

            if 'onet_task_id' not in df.columns:
                print(f"      (skipping - no onet_task_id column)")
                continue

            # Detect filtering type by examining the data
            checkpoint_type = detect_checkpoint_type(checkpoint_file.name, df)
            onet_df = onet_df_core if checkpoint_type == 'core' else onet_df_all
            print(f"      Using {'CORE' if checkpoint_type == 'core' else 'ALL'} task filtering")

            has_task_text = 'onet_task' in df.columns

            # Build position-based lookup lists
            task_id_list = onet_df['task_id'].tolist()
            task_text_list = onet_df['Task'].tolist()
            onet_task_ids_set = set(task_id_list)
            id_to_text = dict(zip(task_id_list, task_text_list))

            print(f"      Applying position-based remapping...")

            # Fix using position-based lookup
            # Treat each stored onet_task_id value as a row position
            def fix_using_position(row_pos):
                pos = int(row_pos)
                if 0 <= pos < len(task_id_list):
                    return task_id_list[pos]  # Get actual Task ID at this position
                return None

            df['onet_task_id'] = df['onet_task_id'].apply(fix_using_position)

            # Rebuild onet_task if present
            if has_task_text:
                df['onet_task'] = df['onet_task_id'].map(id_to_text)

            # Validate: check for null values
            null_task_id = df['onet_task_id'].isna().sum()
            if null_task_id > 0:
                print(f"      ✗ ERROR: {null_task_id} null task IDs after fix")
                print(f"        (some row positions were out of range)")
                error_count += 1
                continue

            if has_task_text:
                null_task_text = df['onet_task'].isna().sum()
                if null_task_text > 0:
                    print(f"      ✗ ERROR: {null_task_text} null task texts after fix")
                    error_count += 1
                    continue

            # Validate: sample check that all Task IDs are valid
            sample_size = min(100, len(df))
            validation_errors = 0

            for idx in df.sample(sample_size).index:
                task_id = int(df.loc[idx, 'onet_task_id'])

                # Check Task ID exists in O*NET source
                if task_id not in onet_task_ids_set:
                    validation_errors += 1

                # If has text, check alignment
                if has_task_text:
                    task_text = df.loc[idx, 'onet_task']
                    expected_text = id_to_text.get(task_id)
                    if expected_text is None or task_text != expected_text:
                        validation_errors += 1

            if validation_errors > 0:
                print(f"      ✗ VALIDATION FAILED: {validation_errors}/{sample_size} sample errors")
                error_count += 1
                continue

            # Save to output directory
            output_file = output_dir / checkpoint_file.name
            df.to_parquet(output_file)
            print(f"      ✓ Fixed and validated ({len(df):,} rows)")
            fixed_count += 1

        except Exception as e:
            print(f"      ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1

    return fixed_count, error_count


def main():
    print("=" * 80)
    print("FIXING CORRUPTED CHECKPOINT FILES - Position-Based Remapping")
    print("=" * 80)

    # Load O*NET source
    onet_file = 'Data/task_statements_20.xlsx'
    if not Path(onet_file).exists():
        print(f"ERROR: O*NET source file not found: {onet_file}")
        sys.exit(1)

    print(f"\nLoading O*NET sources with different filtering...")

    # Build both versions
    onet_df_core = build_onet_df('core')
    onet_df_all = build_onet_df('all')

    print(f"  CORE tasks (no supplemental, no SOC 15): {len(onet_df_core):,} tasks")
    print(f"  ALL tasks (with supplemental, with SOC 15): {len(onet_df_all):,} tasks")

    # Fix BGE bi-encoder checkpoints
    print("\n" + "=" * 80)
    print("FIXING BGE BI-ENCODER CHECKPOINTS")
    print("=" * 80)
    similarities_dir = Path('Data/embeddings/checkpoints')
    sim_fixed, sim_errors = fix_checkpoint_directory(similarities_dir, onet_df_all, onet_df_core)

    # Fix cross-encoder checkpoints
    print("\n" + "=" * 80)
    print("FIXING CROSS-ENCODER CHECKPOINTS")
    print("=" * 80)
    ce_dir = Path('Data/embeddings/cross_encoder_checkpoints')
    ce_fixed, ce_errors = fix_checkpoint_directory(ce_dir, onet_df_all, onet_df_core)

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"BGE bi-encoder checkpoints:")
    print(f"  Fixed:  {sim_fixed}")
    print(f"  Errors: {sim_errors}")
    print(f"  Output: Data/embeddings/checkpoints_fixed/")

    print(f"\nCross-encoder checkpoints:")
    print(f"  Fixed:  {ce_fixed}")
    print(f"  Errors: {ce_errors}")
    print(f"  Output: Data/embeddings/cross_encoder_checkpoints_fixed/")

    total_fixed = sim_fixed + ce_fixed
    total_errors = sim_errors + ce_errors

    print(f"\nTOTAL:")
    print(f"  Fixed:  {total_fixed} checkpoints")
    print(f"  Errors: {total_errors} checkpoints")

    if total_errors > 0:
        print("\n⚠️  Some checkpoints had errors. Check output above.")
        sys.exit(1)

    if total_fixed > 0:
        print("\n✅ All checkpoint fixes validated successfully!")
        print("\nFixed files are in:")
        print(f"  - Data/embeddings/checkpoints_fixed/")
        print(f"  - Data/embeddings/cross_encoder_checkpoints_fixed/")
    else:
        print("\n⚠️  No checkpoints were processed.")

    print("=" * 80)


if __name__ == "__main__":
    main()
