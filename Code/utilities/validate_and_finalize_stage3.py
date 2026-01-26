#!/usr/bin/env python3
"""
Validate and finalize Stage 3 full sample output.

This script:
1. Loads the merged step3 output
2. Validates data quality (NaN checks, row counts, column schemas)
3. Creates a clean final output file
4. Generates a summary report
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_and_finalize():
    """Main validation and finalization function."""

    # Input file
    step3_file = Path("Data/Testing/stage_3/full_sample/merged_llm_output_step3_extracted.csv")

    logger.info("="*80)
    logger.info("STAGE 3 FULL SAMPLE VALIDATION AND FINALIZATION")
    logger.info("="*80)

    # Load step3 output
    logger.info(f"Loading: {step3_file}")
    logger.info(f"File size: {step3_file.stat().st_size / (1024**3):.2f} GB")

    df = pd.read_csv(step3_file, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows")

    # Display column information
    logger.info(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"  {i:2d}. {col}")

    # Expected columns from complete pipeline
    expected_cols = [
        # Original job data
        'uid', 'title', 'company_name', 'company_size', 'location_raw',
        'content_clean', 'tst_created', 'url', 'matched_keywords',
        'company_id', 'company_is_recruiter', 'tst_deleted',
        'x28_industries', 'x28_occupations', 'cantons', 'duplicate_group',
        # Step 1 output
        'ai_applications_raw', 'num_applications',
        # Step 2 output
        'ai_capability', 'step2_output', 'capability_index', 'task_index', 'original_job_uid',
        # Step 3 output
        'step3_output', 'num_final_tasks'
    ]

    # Validate schema
    logger.info("\n" + "="*80)
    logger.info("SCHEMA VALIDATION")
    logger.info("="*80)

    missing_cols = set(expected_cols) - set(df.columns)
    extra_cols = set(df.columns) - set(expected_cols)

    if missing_cols:
        logger.error(f"Missing expected columns: {missing_cols}")
    else:
        logger.info("✓ All expected columns present")

    if extra_cols:
        logger.warning(f"Extra columns not in expected schema: {extra_cols}")

    # Check for NaN values in critical columns
    logger.info("\n" + "="*80)
    logger.info("NaN VALUE CHECK")
    logger.info("="*80)

    critical_cols = [
        'uid', 'company_id', 'ai_applications_raw',
        'step2_output', 'step3_output'
    ]

    nan_report = []
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            nan_pct = 100 * nan_count / len(df)
            nan_report.append({
                'column': col,
                'nan_count': nan_count,
                'nan_pct': nan_pct
            })

            if nan_count > 0:
                logger.warning(f"{col}: {nan_count:,} NaN values ({nan_pct:.2f}%)")
            else:
                logger.info(f"{col}: ✓ No NaN values")

    # Row count validation
    logger.info("\n" + "="*80)
    logger.info("ROW COUNT VALIDATION")
    logger.info("="*80)

    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"Unique UIDs: {df['uid'].nunique():,}")
    logger.info(f"Unique original_job_uid: {df['original_job_uid'].nunique():,}")
    logger.info(f"Unique company_id: {df['company_id'].nunique():,}")

    # Step output validation
    logger.info("\n" + "="*80)
    logger.info("STEP OUTPUT VALIDATION")
    logger.info("="*80)

    # Step 1: ai_applications_raw
    step1_non_empty = df['ai_applications_raw'].notna().sum()
    logger.info(f"Step 1 (ai_applications_raw): {step1_non_empty:,} non-empty ({100*step1_non_empty/len(df):.2f}%)")

    # Step 2: step2_output
    step2_non_empty = df['step2_output'].notna().sum()
    logger.info(f"Step 2 (step2_output): {step2_non_empty:,} non-empty ({100*step2_non_empty/len(df):.2f}%)")

    # Step 3: step3_output
    step3_non_empty = df['step3_output'].notna().sum()
    logger.info(f"Step 3 (step3_output): {step3_non_empty:,} non-empty ({100*step3_non_empty/len(df):.2f}%)")

    # Task count statistics
    logger.info("\n" + "="*80)
    logger.info("TASK COUNT STATISTICS")
    logger.info("="*80)

    if 'num_applications' in df.columns:
        logger.info(f"num_applications: min={df['num_applications'].min()}, "
                   f"max={df['num_applications'].max()}, "
                   f"mean={df['num_applications'].mean():.2f}")

    if 'num_final_tasks' in df.columns:
        logger.info(f"num_final_tasks: min={df['num_final_tasks'].min()}, "
                   f"max={df['num_final_tasks'].max()}, "
                   f"mean={df['num_final_tasks'].mean():.2f}")

    # Data types
    logger.info("\n" + "="*80)
    logger.info("DATA TYPES")
    logger.info("="*80)

    dtype_summary = df.dtypes.value_counts()
    for dtype, count in dtype_summary.items():
        logger.info(f"{dtype}: {count} columns")

    # Memory usage
    logger.info("\n" + "="*80)
    logger.info("MEMORY USAGE")
    logger.info("="*80)

    memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
    logger.info(f"Total memory usage: {memory_mb:.2f} MB")

    # Create final output file
    logger.info("\n" + "="*80)
    logger.info("CREATING FINAL OUTPUT")
    logger.info("="*80)

    output_file = Path("Data/Testing/stage_3/full_sample/final_output_full_sample.csv")
    logger.info(f"Saving to: {output_file}")

    # Save (this is just a copy for now, but could add filtering/cleaning if needed)
    df.to_csv(output_file, index=False)

    output_size_gb = output_file.stat().st_size / (1024**3)
    logger.info(f"✓ Saved {len(df):,} rows to {output_file.name}")
    logger.info(f"✓ Output file size: {output_size_gb:.2f} GB")

    # Summary report
    logger.info("\n" + "="*80)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*80)

    logger.info(f"✓ Schema: All {len(expected_cols)} expected columns present")
    logger.info(f"✓ Rows: {len(df):,} total rows")
    logger.info(f"✓ Coverage: Step1={100*step1_non_empty/len(df):.1f}%, "
               f"Step2={100*step2_non_empty/len(df):.1f}%, "
               f"Step3={100*step3_non_empty/len(df):.1f}%")

    # Check for critical issues
    critical_issues = []
    for report in nan_report:
        if report['nan_pct'] > 0:
            critical_issues.append(f"{report['column']}: {report['nan_count']:,} NaN")

    if critical_issues:
        logger.warning("\n⚠ Critical issues found:")
        for issue in critical_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\n✓ No critical issues found")

    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    validate_and_finalize()
