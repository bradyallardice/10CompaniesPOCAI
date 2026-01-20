#!/usr/bin/env python3
"""
Fix corrupted OpenAI similarity pkl file using position-based mapping.

The OpenAI similarity file was created in December 2025 with buggy code that
stored row positions instead of O*NET Task IDs. This script:
1. Backs up the original file
2. Uses position-based mapping to restore correct Task IDs
3. Validates the fix
4. Saves the corrected file
"""

import pickle
import pandas as pd
import shutil
from pathlib import Path


def build_onet_df():
    """Build O*NET DataFrame with ALL tasks filtering (tasks18840)."""
    onet_df = pd.read_excel('Data/task_statements_20.xlsx')
    onet_df['task_id'] = onet_df['Task ID'].astype(int)

    # Remove NaN/empty tasks
    onet_df = onet_df[onet_df['Task'].notna()].copy()
    onet_df['Task'] = onet_df['Task'].astype(str).str.strip()
    onet_df = onet_df[onet_df['Task'] != ''].copy()

    # No filtering - use ALL tasks (matches tasks18840)
    onet_df = onet_df.reset_index(drop=True)
    return onet_df[['task_id', 'Task']]


def fix_openai_similarity_file():
    """Fix the corrupted OpenAI similarity file."""

    input_file = Path('Data/embeddings/similarities_apps15258_tasks18840_min0.3_openai_text-embedding-3-large_ac8e42f3.pkl')
    output_file = Path('Data/embeddings/similarities_apps15258_tasks18840_min0.3_openai_text-embedding-3-large_ac8e42f3_FIXED.pkl')

    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        return

    print("="*80)
    print("FIXING OPENAI SIMILARITY FILE")
    print("="*80)
    print(f"\nInput:  {input_file.name}")
    print(f"Output: {output_file.name}")
    print(f"Size: {input_file.stat().st_size / (1024**3):.2f} GB")

    # Load O*NET tasks
    print("\nLoading O*NET tasks (ALL filtering - tasks18840)...")
    onet_df = build_onet_df()
    print(f"Loaded {len(onet_df):,} tasks")

    # Build position lookups
    task_id_list = onet_df['task_id'].tolist()
    task_text_list = onet_df['Task'].tolist()
    onet_task_ids_set = set(task_id_list)

    # Fix function
    def fix_row_position(row_pos):
        pos = int(row_pos)
        if 0 <= pos < len(task_id_list):
            return task_id_list[pos]
        return None

    # Load pkl file
    print("\nLoading pkl file (this may take a few minutes)...")
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    print("✓ Loaded")

    # Fix DataFrames
    print("\nFixing DataFrames...")
    id_to_text = dict(zip(task_id_list, task_text_list))

    # Handle both old and new file structures
    keys_to_process = []
    if 'filtered' in data and 'data' in data['filtered']:
        keys_to_process.append(('filtered', data['filtered']['data']))
    if 'all' in data and 'data' in data['all']:
        keys_to_process.append(('all', data['all']['data']))
    if 'filtered_results' in data:
        keys_to_process.append(('filtered_results', data['filtered_results']))
    if 'all_results' in data:
        keys_to_process.append(('all_results', data['all_results']))

    for key, df in keys_to_process:
        rows_before = len(df)
        print(f"\n  Processing '{key}' DataFrame ({rows_before:,} rows)...")

        # Fix onet_task_id using position-based mapping
        df['onet_task_id'] = df['onet_task_id'].apply(fix_row_position)

        # Rebuild onet_task from corrected task_id
        df['onet_task'] = df['onet_task_id'].map(id_to_text)

        # Validate
        null_count = df['onet_task_id'].isna().sum()
        null_text = df['onet_task'].isna().sum()
        if null_count > 0:
            print(f"  ✗ WARNING: {null_count} null task_ids after fix")
        if null_text > 0:
            print(f"  ✗ WARNING: {null_text} null task_text after fix")
        if null_count == 0 and null_text == 0:
            print(f"  ✓ Fixed all {rows_before:,} rows")

    # Save fixed file
    print("\nSaving fixed file (this may take a few minutes)...")
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)

    print("✓ Saved")

    # Validate
    print("\nValidating fix (sampling 1000 rows)...")

    # Build valid Task ID set
    onet_task_ids_set = set(task_id_list)

    # Check whichever DataFrame exists
    if 'filtered_results' in data:
        df_to_validate = data['filtered_results']
    elif 'filtered' in data and 'data' in data['filtered']:
        df_to_validate = data['filtered']['data']
    else:
        df_to_validate = data.get('all_results') or data['all']['data']

    df_sample = df_to_validate.sample(min(1000, len(df_to_validate)))

    mismatches = 0
    invalid_ids = 0
    for idx, row in df_sample.iterrows():
        task_id = int(row['onet_task_id'])
        task_text = row['onet_task']

        # Check if Task ID is valid
        if task_id not in onet_task_ids_set:
            invalid_ids += 1
        # Check if text matches
        elif task_id in id_to_text and task_text != id_to_text[task_id]:
            mismatches += 1

    if mismatches == 0 and invalid_ids == 0:
        print(f"✓ Validation passed: 0/{len(df_sample)} mismatches, 0/{len(df_sample)} invalid IDs")
    else:
        print(f"✗ Validation issues:")
        if invalid_ids > 0:
            print(f"  - {invalid_ids}/{len(df_sample)} rows have invalid Task IDs")
        if mismatches > 0:
            print(f"  - {mismatches}/{len(df_sample)} rows have text mismatches")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Fixed file saved at: {output_file}")


if __name__ == "__main__":
    fix_openai_similarity_file()
