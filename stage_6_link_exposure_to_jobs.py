#!/usr/bin/env python3
"""
Stage 6: Link Firm×Year Exposure to Actually Existing Jobs

Goal: Intersect Stage 5 occupation exposures with jobs that actually exist at a firm 
in a year, producing a clean, analysis-safe table keyed by (company_id, year, title, occ_code).

Author: Adapted from Allardice/Kurer methodology
Date: September 2025
"""

import argparse
import pandas as pd
import psycopg2
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import os
from datetime import datetime
import re
# These imports are used in the firm report functions
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm

def log(msg):
    """Print timestamped log message"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def log_transformation(operation_name, before_count, after_count, details=""):
    """Log data transformation with before/after counts and loss percentage."""
    lost = before_count - after_count
    pct_lost = (lost / before_count * 100) if before_count > 0 else 0
    loss_msg = f"[LOST {lost:,} ({pct_lost:.1f}%)]" if lost > 0 else "[No loss]"
    detail_msg = f" - {details}" if details else ""
    log(f"{operation_name}: {before_count:,} → {after_count:,} {loss_msg}{detail_msg}")

def log_sample_records(df, prefix, n=5, cols=None):
    """Log sample records for debugging."""
    if len(df) == 0:
        log(f"{prefix}: [empty]")
        return
    sample_df = df.head(n)
    if cols:
        # Only select columns that exist
        cols = [c for c in cols if c in sample_df.columns]
        if cols:
            sample_df = sample_df[cols]
    log(f"{prefix} (showing {min(n, len(df))} of {len(df):,}):")
    for idx, row in sample_df.iterrows():
        log(f"  {dict(row)}")

def compute_file_hash(filepath):
    """Compute MD5 hash of file for drift detection"""
    if not Path(filepath).exists():
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def deduplicate_jobs_within_companies(df_jobs, similarity_threshold=0.99):
    """
    Deduplicate jobs within each company using Stage 2's approach.

    Process:
    1. Exact deduplication on content_clean + title + company_id
    2. TF-IDF similarity within company groups
    3. Keep earliest job (by tst_created) when deduplicating
    4. Extend each unique job from first appearance through 2025
    """
    log(f"🔧 Starting job deduplication within companies (threshold: {similarity_threshold})...")
    initial_count = len(df_jobs)

    # Step 1: Remove exact duplicates first (company_id + title + content_clean equivalent)
    log("Step 1: Removing exact duplicates (company_id + title + x28_occupations)...")
    # Sort by year to keep earliest posting when deduplicating
    df_sorted = df_jobs.sort_values(['company_id', 'year'], ascending=True)

    # Create deduplication key - using available fields from jobs cache
    df_sorted['dedup_key'] = (
        df_sorted['company_id'].astype(str) + '|||' +
        df_sorted['title'].fillna('').str.lower().str.strip() + '|||' +
        df_sorted['x28_occupations'].astype(str)
    )

    before_exact_dedup = len(df_sorted)
    df_step1 = df_sorted.drop_duplicates(subset=['dedup_key'], keep='first')
    log_transformation(
        "Exact deduplication by (company_id, title, x28_occupations)",
        before_exact_dedup,
        len(df_step1),
        f"dedup_key='{df_sorted['dedup_key'].name}'"
    )

    if len(df_step1) == 0:
        return df_step1

    # For jobs cache, we don't have content_clean, so we'll just use exact deduplication
    # and then extend jobs through time

    # Get the earliest year for each unique job
    df_step1 = df_step1.copy()
    df_step1['start_year'] = df_step1.groupby('dedup_key')['year'].transform('min')

    min_year = int(df_step1['start_year'].min())
    max_year = 2025
    log(f"Step 2: Extending unique jobs through time ({min_year}-{max_year})...")

    # Create a list to store all extended jobs
    extended_jobs = []

    # Group by the unique job key and extend each through time
    base_jobs_count = len(df_step1)
    for _, group in df_step1.groupby('dedup_key'):
        base_job = group.iloc[0].copy()  # Take the earliest occurrence
        start_year = int(base_job['start_year'])

        # Extend from start_year through 2025
        for year in range(start_year, 2026):
            extended_job = base_job.copy()
            extended_job['year'] = year
            extended_jobs.append(extended_job)

    df_extended = pd.DataFrame(extended_jobs)
    df_extended = df_extended.drop(columns=['dedup_key', 'start_year'])

    before_extension = len(df_step1)
    after_extension = len(df_extended)
    new_records = after_extension - before_extension
    log(f"Time extension: Created {new_records:,} new job-year records ({before_extension:,} base jobs → {after_extension:,} total)")

    # Remove duplicates that might have been created by the extension process
    before_final_dedup = len(df_extended)
    df_final = df_extended.drop_duplicates(subset=['company_id', 'year', 'title'], keep='first')
    log_transformation(
        "Final deduplication by (company_id, year, title)",
        before_final_dedup,
        len(df_final)
    )

    log(f"✅ Job deduplication complete:")
    log(f"   Original jobs: {initial_count:,}")
    log(f"   After exact dedup: {before_extension:,}")
    log(f"   After time extension: {after_extension:,}")
    log(f"   After final dedup: {len(df_final):,}")

    return df_final

def load_ai_applications_data(filepath):
    """Load AI applications data and aggregate by firm-year"""
    log(f"📊 Loading AI applications data: {Path(filepath).name}")

    df = pd.read_csv(filepath)
    log(f"✅ Loaded {len(df):,} AI application records")

    # Validate required columns
    required_cols = ['step3_output', 'company_id', 'tst_created']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in AI applications file: {missing_cols}")

    # Filter out empty AI applications
    df_filtered = df[df['step3_output'].notna() & (df['step3_output'].str.strip() != '')].copy()
    log(f"✅ Filtered to {len(df_filtered):,} non-empty AI applications")

    # Extract year from tst_created using project standard approach
    def extract_year_from_string(date_str):
        try:
            if pd.isna(date_str) or date_str == '':
                return None
            # Extract first 4 characters as year (YYYY-MM-DD format)
            year = int(str(date_str)[:4])
            # Basic validation - reasonable year range
            if 2000 <= year <= 2030:
                return year
            else:
                return None
        except:
            return None

    df_filtered['year'] = df_filtered['tst_created'].apply(extract_year_from_string)
    df_filtered = df_filtered[df_filtered['year'].notna()].copy()  # Filter out invalid years
    df_filtered['year'] = df_filtered['year'].astype(int)

    # Aggregate by company_id and year
    df_agg = df_filtered.groupby(['company_id', 'year']).agg({
        'step3_output': 'count'  # Count AI applications per firm-year
    }).reset_index()

    df_agg.columns = ['company_id', 'year', 'total_ai_apps_all']

    log(f"✅ Aggregated to {len(df_agg):,} firm-year combinations")
    return df_agg

def load_ai_jobs_data(filepath):
    """Load AI jobs data (ai_development_deduplicated_custom.csv) and aggregate by firm-year"""
    log(f"📊 Loading AI jobs data: {Path(filepath).name}")

    if not Path(filepath).exists():
        log(f"⚠️  AI jobs file not found: {filepath}")
        return pd.DataFrame(columns=['company_id', 'year', 'total_unique_ai_jobs'])

    df = pd.read_csv(filepath)
    log(f"✅ Loaded {len(df):,} AI job records")

    # Validate required columns
    required_cols = ['company_id', 'tst_created']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in AI jobs file: {missing_cols}")

    # Extract year from tst_created using project standard approach
    def extract_year_from_string(date_str):
        try:
            if pd.isna(date_str) or date_str == '':
                return None
            # Extract first 4 characters as year (YYYY-MM-DD format)
            year = int(str(date_str)[:4])
            # Basic validation - reasonable year range
            if 2000 <= year <= 2030:
                return year
            else:
                return None
        except:
            return None

    df['year'] = df['tst_created'].apply(extract_year_from_string)
    df = df[df['year'].notna()].copy()  # Filter out invalid years
    df['year'] = df['year'].astype(int)

    # Count unique AI jobs per firm-year
    df_agg = df.groupby(['company_id', 'year']).size().reset_index(name='total_unique_ai_jobs')

    log(f"✅ Aggregated to {len(df_agg):,} firm-year combinations with AI jobs")
    return df_agg

def generate_firm_summary_report(df_linked, df_jobs_cache, args, firm_report_output):
    """Generate comprehensive firm-level AI summary report"""
    log("🚀 Starting firm summary report generation...")

    try:
        # Step 1: Deduplicate jobs within companies and extend through time
        log("📊 Processing job data...")
        df_jobs_dedup = deduplicate_jobs_within_companies(df_jobs_cache)

        # Count unique jobs per firm-year
        df_job_counts = df_jobs_dedup.groupby(['company_id', 'company_name', 'year']).size().reset_index(name='total_unique_job_ads')

        # Step 2: Load AI applications data (all apps, not just linked)
        ai_apps_file = args.ai_apps_file
        log(f"Using AI applications file: {ai_apps_file}")
        df_ai_apps_all = load_ai_applications_data(ai_apps_file)

        # Step 3: Load AI jobs data (unique AI jobs)
        ai_jobs_file = args.ai_jobs_file
        log(f"Using AI jobs file: {ai_jobs_file}")
        df_ai_jobs = load_ai_jobs_data(ai_jobs_file)

        # Step 4: Extract Stage 6 linked data metrics for each specification
        log("📊 Processing Stage 6 exposure data (multi-specification)...")

        # Determine which occupation column exists (ISCO or ONET)
        if 'isco08_4d' in df_linked.columns:
            occ_col = 'isco08_4d'
            log("Using ISCO occupation codes for aggregation")
        elif 'onet_soc' in df_linked.columns:
            occ_col = 'onet_soc'
            log("Using ONET occupation codes for aggregation")
        else:
            raise ValueError("Neither 'isco08_4d' nor 'onet_soc' columns found in Stage 6 data")

        # Identify all exposure columns (they follow pattern: *_exposure_avg_p{XX}_ce{YY})
        exposure_cols = [c for c in df_linked.columns if '_exposure_avg_' in c and '_ce' in c]
        log(f"Found {len(exposure_cols)} exposure specifications in linked data")

        # Build specifications list from column names using regex
        # Handle two naming formats:
        # 1. ISCO format: 'pct_20_ce_0.0' from 'hampole_ai_exposure_avg_pct_20_ce_0.0'
        # 2. O*NET format: 'p20_ce08' from 'hampole_ai_exposure_avg_p20_ce08'
        specifications = set()
        for col in exposure_cols:
            if 'hampole' in col:
                # Try ISCO format first (pct_XX_ce_0.X)
                match = re.search(r'(pct_\w+_ce_[\d.]+)$', col)
                if match:
                    spec_str = match.group(1)  # e.g., 'pct_20_ce_0.0'
                    specifications.add(spec_str)
                else:
                    # Try O*NET format (pXX_ceXX)
                    match = re.search(r'(p\d+_ce\d+)$', col)
                    if match:
                        spec_str = match.group(1)  # e.g., 'p20_ce08'
                        specifications.add(spec_str)

        specifications = sorted(list(specifications))
        log(f"Parsed {len(specifications)} specifications from column names")

        # Aggregate Stage 6 linked data by firm-year-specification
        # This creates one row per firm-year-specification combination
        stage6_rows = []

        for firm_year_group, group_data in df_linked.groupby(['company_id', 'company_name', 'year']):
            company_id, company_name, year = firm_year_group

            for spec_str in specifications:
                # Build column names for this specification using full spec string
                hampole_col = f'hampole_ai_exposure_avg_{spec_str}'
                binary_col = f'binary_ai_exposure_avg_{spec_str}'
                apps_col = f'n_ai_apps_firm_year_{spec_str}'
                tasks_col = f'total_tasks_occupation_{spec_str}'

                # Check if columns exist
                if hampole_col not in group_data.columns:
                    # Skip this specification if expected column doesn't exist
                    continue

                # Extract values from existing columns
                total_apps = group_data[apps_col].iloc[0] if apps_col in group_data.columns else 0
                # Count how many occupations have >0 exposure
                ai_exposed_tasks = (group_data[hampole_col] > 0).sum()

                # Get total onet tasks using spec-specific column
                total_tasks = group_data[tasks_col].sum() if tasks_col in group_data.columns else 0

                row = {
                    'company_id': company_id,
                    'company_name': company_name,
                    'year': year,
                    'specification': spec_str,
                    'total_unique_job_ads': group_data['title'].nunique() if 'title' in group_data.columns else 0,
                    'total_onet_tasks': int(total_tasks),
                    'total_ai_apps_linked': int(total_apps),
                    'ai_exposed_tasks': int(ai_exposed_tasks),
                }

                # Calculate exposure ratio
                if row['total_onet_tasks'] > 0:
                    row['firm_exposure'] = round(row['ai_exposed_tasks'] / row['total_onet_tasks'], 4)
                else:
                    row['firm_exposure'] = 0.0

                stage6_rows.append(row)

        stage6_agg = pd.DataFrame(stage6_rows)

        # Step 5: Merge all data sources
        log("🔗 Merging data sources (long format)...")

        # Standardize company_id types across all dataframes before merging
        df_report = stage6_agg.copy()
        df_report['company_id'] = df_report['company_id'].astype(int)

        df_job_counts['company_id'] = df_job_counts['company_id'].astype(int)
        df_ai_apps_all['company_id'] = df_ai_apps_all['company_id'].astype(int)
        df_ai_jobs['company_id'] = df_ai_jobs['company_id'].astype(int)

        # Merge job counts (1 row per firm-year) - will expand to 25 rows per spec
        # First check if merge keys exist
        if 'total_unique_job_ads' not in df_report.columns:
            # Need to merge with df_job_counts
            before_merge = len(df_report)
            df_report = df_report.merge(
                df_job_counts[['company_id', 'company_name', 'year', 'total_unique_job_ads']],
                on=['company_id', 'company_name', 'year'],
                how='left'
            )
            after_merge = len(df_report)
            merge_success = after_merge / before_merge * 100 if before_merge > 0 else 0
            log(f"Merged job counts: {before_merge:,} → {after_merge:,} ({merge_success:.1f}% retained)")
            if 'total_unique_job_ads' not in df_report.columns:
                raise ValueError("Failed to merge total_unique_job_ads from df_job_counts")

        # Merge AI applications (all) (1 row per firm-year) - will expand to 25 rows per spec
        before_merge = len(df_report)
        df_report = df_report.merge(
            df_ai_apps_all,
            on=['company_id', 'year'],
            how='left'
        )
        after_merge = len(df_report)
        merge_success = after_merge / before_merge * 100 if before_merge > 0 else 0
        log(f"Merged AI applications: {before_merge:,} → {after_merge:,} ({merge_success:.1f}% retained)")

        # Merge AI jobs (1 row per firm-year) - will expand to 25 rows per spec
        before_merge = len(df_report)
        df_report = df_report.merge(
            df_ai_jobs,
            on=['company_id', 'year'],
            how='left'
        )
        after_merge = len(df_report)
        merge_success = after_merge / before_merge * 100 if before_merge > 0 else 0
        log(f"Merged unique AI jobs: {before_merge:,} → {after_merge:,} ({merge_success:.1f}% retained)")

        # Fill missing values with 0 and ensure required columns exist
        required_numeric_cols = ['total_ai_apps_all', 'total_unique_ai_jobs', 'total_ai_apps_linked',
                                'total_onet_tasks', 'ai_exposed_tasks', 'total_unique_job_ads']

        for col in required_numeric_cols:
            if col not in df_report.columns:
                raise ValueError(f"Required column '{col}' missing from report after merges")
            df_report[col] = df_report[col].fillna(0).astype(int)

        # Step 6: Calculate percentage and exposure metrics
        log("📊 Calculating percentage and exposure metrics...")

        # Yearly percentage: AI jobs / total jobs for each year
        df_report['pct_ai_ads_yearly'] = (df_report['total_unique_ai_jobs'] / df_report['total_unique_job_ads'] * 100).round(2)

        # Cumulative percentage: group by company and calculate cumulative sums
        df_report['cumulative_ai_jobs'] = df_report.groupby('company_id')['total_unique_ai_jobs'].cumsum()
        df_report['cumulative_total_jobs'] = df_report.groupby('company_id')['total_unique_job_ads'].cumsum()
        df_report['pct_ai_ads_cumulative'] = (df_report['cumulative_ai_jobs'] / df_report['cumulative_total_jobs'] * 100).round(2)

        # Firm AI exposure: exposed tasks / total tasks
        df_report['firm_ai_exposure'] = (df_report['ai_exposed_tasks'] / df_report['total_onet_tasks']).round(2)

        # Replace inf and NaN values with 0 (occurs when dividing by 0)
        df_report['pct_ai_ads_yearly'] = df_report['pct_ai_ads_yearly'].replace([float('inf'), -float('inf')], 0).fillna(0)
        df_report['pct_ai_ads_cumulative'] = df_report['pct_ai_ads_cumulative'].replace([float('inf'), -float('inf')], 0).fillna(0)
        df_report['firm_ai_exposure'] = df_report['firm_ai_exposure'].replace([float('inf'), -float('inf')], 0).fillna(0)

        # Drop temporary cumulative columns
        df_report = df_report.drop(columns=['cumulative_ai_jobs', 'cumulative_total_jobs'])

        # Step 7: Final report formatting and validation
        log("✅ Finalizing report (long format with 25 specifications)...")

        # Ensure we have all required columns in the right order
        final_columns = [
            'company_id', 'company_name', 'year', 'specification',
            'total_unique_job_ads', 'total_unique_ai_jobs',
            'pct_ai_ads_yearly', 'pct_ai_ads_cumulative',
            'total_ai_apps_all', 'total_ai_apps_linked',
            'total_onet_tasks', 'ai_exposed_tasks', 'firm_exposure'
        ]

        # Only keep columns that exist in the dataframe
        final_columns = [c for c in final_columns if c in df_report.columns]
        df_report = df_report[final_columns]

        # Add specification distribution
        spec_dist = df_report['specification'].value_counts()
        log("Specification distribution in firm report:")
        for spec, count in spec_dist.items():
            pct = count / len(df_report) * 100
            log(f"  {spec}: {count:,} ({pct:.1f}%)")

        # Sort by company_id, year, and specification
        df_report = df_report.sort_values(['company_id', 'year', 'specification']).reset_index(drop=True)

        # Save the report
        df_report.to_csv(firm_report_output, index=False)

        log(f"✅ Firm summary report generated successfully (long format)!")
        log(f"   Report saved: {firm_report_output}")
        log(f"   Firms covered: {df_report['company_id'].nunique():,}")
        log(f"   Total firm-year-specification records: {len(df_report):,}")
        log(f"   Specifications per firm-year: 25 (5 BGE percentiles × 5 CE thresholds)")
        log(f"   Year range: {df_report['year'].min()}-{df_report['year'].max()}")

        # Show summary statistics
        log("📊 Summary statistics (aggregated across all specifications):")
        log(f"   Firms with AI jobs: {df_report['total_unique_ai_jobs'].nunique():,}")
        log(f"   Firms with AI applications (all): {(df_report['total_ai_apps_all'] > 0).sum():,}")
        log(f"   Total spec-rows with linked AI apps: {(df_report['total_ai_apps_linked'] > 0).sum():,}")
        log(f"   Total spec-rows with AI-exposed O*NET tasks: {(df_report['ai_exposed_tasks'] > 0).sum():,}")

        return df_report

    except Exception as e:
        log(f"❌ Error generating firm summary report: {e}")
        raise



def parse_x28_occupations(x28_str):
    """Parse X28 occupations field into list of clean strings"""
    # Handle PostgreSQL arrays (which come as numpy arrays or lists)
    if hasattr(x28_str, '__iter__') and not isinstance(x28_str, str):
        # It's already an array/list from PostgreSQL
        return [str(x).strip() for x in x28_str if x and str(x).strip()]

    # Handle null/empty values (but only for strings)
    try:
        if not x28_str or pd.isna(x28_str) or str(x28_str).strip() in ('', 'null', 'None'):
            return []
    except ValueError:
        # If we get here, x28_str is likely an array that wasn't caught above
        if hasattr(x28_str, '__len__') and len(x28_str) == 0:
            return []
        # For non-empty arrays, convert to string list
        return [str(x).strip() for x in x28_str if x and str(x).strip()]

    x28_str = str(x28_str).strip()

    # Handle JSON-style arrays: ['123', '456'] or ["123", "456"]
    if x28_str.startswith('[') and x28_str.endswith(']'):
        try:
            # Try JSON parsing first
            parsed = json.loads(x28_str)
            if isinstance(parsed, list):
                return [str(x).strip().strip('\'"') for x in parsed if x]
        except:
            # Fallback: manual parsing for space-separated or comma-separated values
            x28_str = x28_str[1:-1]  # Remove brackets

            # Split by comma first, then by space for each part
            codes = []
            for part in x28_str.split(','):
                part = part.strip().strip('\'"')
                if part:
                    # Handle space-separated codes within this part
                    for code in part.split():
                        code = code.strip().strip('\'"')
                        if code:
                            codes.append(code)

            # If no comma-separated values found, try space-separated
            if not codes:
                for code in x28_str.split():
                    code = code.strip().strip('\'"')
                    if code:
                        codes.append(code)

            return codes

    # Handle comma-separated values
    codes = []
    for code in x28_str.split(','):
        code = code.strip().strip('\'"')
        if code:
            # Also handle space-separated codes within each comma-separated part
            for subcode in code.split():
                subcode = subcode.strip().strip('\'"')
                if subcode:
                    codes.append(subcode)

    return codes

def validate_jobs_data(df_jobs_exploded, fail_on_warn=True):
    """Validate jobs data with fail-fast checks"""
    total_rows = len(df_jobs_exploded)
    
    if total_rows == 0:
        raise ValueError("❌ FATAL: 0 rows after job expansion")
    
    # Check for empty X28 occupations
    empty_x28 = df_jobs_exploded[df_jobs_exploded['x28_codes'].isna()]
    empty_pct = (len(empty_x28) / total_rows) * 100
    
    if empty_pct > 2.0:
        msg = f"❌ FATAL: {empty_pct:.1f}% rows have empty x28_occupations (threshold: 2%)"
        if fail_on_warn:
            raise ValueError(msg)
        else:
            log(f"⚠️  WARNING: {msg}")
    
    # Check for duplicates

    key_cols = ['company_id', 'year', 'x28_codes']
    duplicates = df_jobs_exploded.duplicated(subset=key_cols).sum()
    if duplicates > 0:
        msg = f"❌ FATAL: {duplicates} duplicate (company_id, year, x28_occupations) combinations"
        if fail_on_warn:
            raise ValueError(msg)
        else:
            log(f"⚠️  WARNING: {msg}")
    
    log(f"✅ Jobs validation passed: {total_rows:,} rows, {empty_pct:.1f}% empty X28")

def build_jobs_from_db(conn, year_max, fail_on_warn=True):
    """Build firm×year×title job list from job_postings_unified"""

    # Check for cached job list file
    cache_file = f"Data/stage6_job_cache_max{year_max}.parquet"
    cache_path = Path(cache_file)

    if cache_path.exists():
        log(f"📁 Found cached job list: {cache_file}")
        log("💾 Loading from cache instead of database...")
        try:
            df_jobs = pd.read_parquet(cache_file)
            log(f"✅ Loaded {len(df_jobs):,} jobs from cache")

            # Parse X28 occupations from cached data
            log("🔧 Parsing X28 occupations from cache...")
            df_jobs['x28_codes'] = df_jobs['x28_occupations'].apply(parse_x28_occupations)

            before_explosion = len(df_jobs)
            df_jobs_exploded = df_jobs.explode('x28_codes').reset_index(drop=True)
            after_explosion = len(df_jobs_exploded)
            avg_codes = after_explosion / before_explosion if before_explosion > 0 else 0
            log(f"X28 explosion: {before_explosion:,} → {after_explosion:,} (avg {avg_codes:.1f} codes/job)")

            before_dedup = len(df_jobs_exploded)
            df_jobs_exploded = df_jobs_exploded.drop_duplicates(subset=['company_id', 'year', 'x28_codes'])
            log_transformation(
                "Deduplication by (company_id, year, x28_codes)",
                before_dedup,
                len(df_jobs_exploded)
            )

            # Validate cached data
            validate_jobs_data(df_jobs_exploded, fail_on_warn)

            return df_jobs_exploded
        except Exception as e:
            log(f"⚠️  Cache loading failed: {e}")
            log("🔄 Falling back to database query...")

    log("🔍 Building job list from database...")
    
    sql = f"""
    WITH base AS (
      SELECT
        company_id,
        company_name,
        title,
        x28_occupations,
        DATE_PART('year', tst_created) AS job_year,
        MIN(DATE_PART('year', tst_created)) OVER (PARTITION BY company_id, title) AS start_year
      FROM job_postings_unified
      WHERE company_id IS NOT NULL AND title IS NOT NULL AND tst_created IS NOT NULL
    ),
    years AS (
      SELECT DISTINCT 
        company_id, 
        company_name, 
        title, 
        x28_occupations, 
        start_year::int,
        generate_series(start_year::int, {year_max}, 1) AS year
      FROM base
    )
    SELECT company_id, company_name, year, title, x28_occupations 
    FROM years 
    ORDER BY company_id, year, title;
    """
    
    log(f"📊 Executing job expansion query (year_max={year_max})...")
    df_jobs = pd.read_sql(sql, conn)
    log(f"✅ Retrieved {len(df_jobs):,} job-year records")

    # Parse X28 occupations
    log("🔧 Parsing X28 occupations...")
    df_jobs['x28_codes'] = df_jobs['x28_occupations'].apply(parse_x28_occupations)

    before_explosion = len(df_jobs)
    df_jobs_exploded = df_jobs.explode('x28_codes').reset_index(drop=True)
    after_explosion = len(df_jobs_exploded)
    avg_codes = after_explosion / before_explosion if before_explosion > 0 else 0
    log(f"X28 explosion: {before_explosion:,} → {after_explosion:,} (avg {avg_codes:.1f} codes/job)")

    before_dedup1 = len(df_jobs_exploded)
    df_jobs_exploded = df_jobs_exploded.drop_duplicates(subset=['company_id', 'year', 'x28_codes'])
    log_transformation(
        "Deduplication by (company_id, year, x28_codes)",
        before_dedup1,
        len(df_jobs_exploded)
    )
    # Validate
    validate_jobs_data(df_jobs_exploded, fail_on_warn)

    # Save to cache for future runs (before parsing x28_codes to preserve raw data)
    try:
        # Create a copy for caching without the parsed x28_codes column
        df_cache = df_jobs_exploded.drop(columns=['x28_codes']).copy()
        df_cache.to_parquet(cache_file, index=False)
        log(f"💾 Saved job list to cache: {cache_file}")
    except Exception as e:
        log(f"⚠️  Cache saving failed (continuing anyway): {e}")

    before_dedup2 = len(df_jobs_exploded)
    log(f"Final jobs from database: {before_dedup2:,} records")

    return df_jobs_exploded

def load_crosswalk_x28_to_isco(filepath):
    """Load X28 to ISCO crosswalk with auto-detection"""
    log(f"📋 Loading X28→ISCO crosswalk: {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Crosswalk file not found: {filepath}")

    df = pd.read_csv(filepath, sep=';')
    
    # Auto-detect column names (look for X28 and ISCO patterns)
    x28_col = 'occupation_id'
    isco_col = 'ch_isco_4d'
    
    if not x28_col or not isco_col:
        raise ValueError(f"Could not auto-detect X28 and ISCO columns in {filepath}. Columns: {list(df.columns)}")
    
    log(f"✅ Detected columns: X28='{x28_col}', ISCO='{isco_col}'")
    
    # Clean and validate
    crosswalk = df[[x28_col, isco_col]].dropna().drop_duplicates()
    crosswalk.columns = ['x28_code', 'isco_code']
    
    # Convert to string and clean
    crosswalk['x28_code'] = crosswalk['x28_code'].astype(str).str.strip()
    crosswalk['isco_code'] = crosswalk['isco_code'].astype(str).str.strip()
    
    log(f"✅ Loaded {len(crosswalk):,} X28→ISCO mappings")
    return crosswalk

def load_crosswalk_isco_to_onet(filepath):
    """Load ISCO to ONET crosswalk from Excel with auto-detection"""
    log(f"📋 Loading ISCO→ONET crosswalk: {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Crosswalk file not found: {filepath}")
    
    # Read Excel file, skipping header rows (data starts at row 3, index 2)
    df = pd.read_excel(filepath, skiprows=2)

    # Use first row as column names and drop it
    if len(df) > 0:
        df.columns = df.iloc[0]  # Use first row as column names
        df = df.drop(df.index[0])  # Drop the header row
        df = df.reset_index(drop=True)
    
    # Use hardcoded column names from ESCO_to_ONET-SOC.xlsx
    isco_col = 'ESCO/ISCO Code'
    onet_col = 'O*NET-SOC 2019 Code'

    if isco_col not in df.columns or onet_col not in df.columns:
        raise ValueError(f"Expected columns '{isco_col}' and '{onet_col}' not found in {filepath}. Columns: {list(df.columns)}")

    log(f"✅ Using columns: ISCO='{isco_col}', ONET='{onet_col}'")
    
    # Clean and validate
    crosswalk = df[[isco_col, onet_col]].dropna().drop_duplicates()
    crosswalk.columns = ['isco_code', 'onet_soc']
    
    # Convert to string and clean
    crosswalk['isco_code'] = crosswalk['isco_code'].astype(str).str.strip()
    crosswalk['onet_soc'] = crosswalk['onet_soc'].astype(str).str.strip()
    
    log(f"✅ Loaded {len(crosswalk):,} ISCO→ONET mappings")
    return crosswalk

def find_stage5_file(stage5_dir, occ_code, task_type='core'):
    """Find Stage 5 exposure file in specified directory

    Searches for pattern: {occ_code}_firm_year_exposure_{task_type}_tasks_all_specs.csv
    """
    task_suffix = f"_{task_type}_tasks"
    search_dir = Path(stage5_dir)

    if not search_dir.exists():
        raise FileNotFoundError(f"Stage 5 directory not found: {stage5_dir}")

    pattern = f"{occ_code}_firm_year_exposure{task_suffix}_all_specs.csv"
    target_file = search_dir / pattern

    if target_file.exists():
        log(f"✅ Found Stage 5 file: {target_file.name}")
        return str(target_file)

    # If not found, show available files for debugging
    available_files = list(search_dir.glob(f"{occ_code}_*exposure*.csv"))
    available_names = [f.name for f in available_files]

    raise FileNotFoundError(
        f"Stage 5 file not found: {pattern}\n"
        f"Searched in: {stage5_dir}\n"
        f"Available files: {available_names if available_names else 'None found'}"
    )

def load_stage5_exposures(filepath, occ_code):
    """Load Stage 5 exposure file"""
    log(f"📊 Loading Stage 5 exposures: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = ['company_id', 'year']
    occ_col = 'isco08_4d' if occ_code == 'isco' else 'onet_soc'
    required_cols.append(occ_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Stage 5 file: {missing_cols}")
    
    log(f"✅ Loaded {len(df):,} exposure records")
    return df

def apply_crosswalks(df_jobs_exploded, x28_isco_xwalk, isco_onet_xwalk=None, target_occ='isco'):
    """Apply crosswalks to map job X28 codes to target occupation codes

    Returns exploded format with one row per occupation code for joining.
    """
    log(f"🔧 Applying crosswalks to map to {target_occ.upper()}...")

    # Explode X28 codes (one row per code)
    before_filter = len(df_jobs_exploded)
    df_jobs_exploded = df_jobs_exploded[df_jobs_exploded['x28_codes'].notna()].copy()
    log_transformation(
        "Filter: Remove jobs with null X28 codes",
        before_filter,
        len(df_jobs_exploded)
    )

    # Ensure x28_codes are scalar values (fix for unhashable list error)
    def flatten_x28_code(x):
        if isinstance(x, list):
            return x[0] if len(x) > 0 else None
        return x

    before_flatten_check = len(df_jobs_exploded)
    df_jobs_exploded['x28_codes'] = df_jobs_exploded['x28_codes'].apply(flatten_x28_code)
    df_jobs_exploded = df_jobs_exploded[df_jobs_exploded['x28_codes'].notna()].copy()
    log_transformation(
        "Filter: Remove jobs where X28 flattening failed",
        before_flatten_check,
        len(df_jobs_exploded)
    )

    if len(df_jobs_exploded) == 0:
        raise ValueError("❌ FATAL: No valid X28 codes after expansion")

    # Map X28 to ISCO
    df_mapped = df_jobs_exploded.merge(
        x28_isco_xwalk,
        left_on='x28_codes',
        right_on='x28_code',
        how='left'
    )

    # Check mapping success
    unmapped_x28 = df_mapped['isco_code'].isna().sum()
    mapping_rate = ((len(df_mapped) - unmapped_x28) / len(df_mapped)) * 100

    log(f"X28→ISCO merge success: {len(df_mapped) - unmapped_x28:,} of {len(df_mapped):,} ({mapping_rate:.1f}%)")

    # DEBUG: Analyze X28→ISCO mapping gaps
    if unmapped_x28 > 0:
        # Find which X28 codes are missing from crosswalk
        jobs_x28_codes = set(df_jobs_exploded['x28_codes'].unique())
        xwalk_x28_codes = set(x28_isco_xwalk['x28_code'].unique())
        missing_x28_codes = jobs_x28_codes - xwalk_x28_codes

        log(f"🔍 DEBUG: X28 codes in jobs: {len(jobs_x28_codes):,}")
        log(f"🔍 DEBUG: X28 codes in crosswalk: {len(xwalk_x28_codes):,}")
        log(f"🔍 DEBUG: Missing X28 codes from crosswalk: {len(missing_x28_codes):,}")

        if len(missing_x28_codes) > 0:
            log(f"🔍 DEBUG: Sample missing X28 codes: {sorted(list(missing_x28_codes))[:20]}")

            # Count jobs affected by each missing X28 code
            unmapped_jobs = df_mapped[df_mapped['isco_code'].isna()]
            if len(unmapped_jobs) > 0:
                missing_counts = unmapped_jobs['x28_codes'].value_counts()
                log(f"🔍 DEBUG: Top missing X28 codes by job count:")
                for code, count in missing_counts.head(15).items():
                    log(f"  {code}: {count:,} jobs")

        # Also check for X28 codes in crosswalk but not used in jobs
        unused_x28_codes = xwalk_x28_codes - jobs_x28_codes
        log(f"🔍 DEBUG: X28 codes in crosswalk but not in jobs: {len(unused_x28_codes):,}")
        if len(unused_x28_codes) > 0:
            log(f"🔍 DEBUG: Sample unused X28 codes: {sorted(list(unused_x28_codes))[:10]}")

        # Summary note about expected vs problematic gaps
        log(f"📝 NOTE: X28→ISCO gaps may indicate:")
        log(f"  - New X28 codes not yet in crosswalk (expected)")
        log(f"  - Obsolete X28 codes removed from crosswalk (expected)")
        log(f"  - Crosswalk completeness issues (investigate if >5% unmapped)")
    
    if target_occ == 'isco':
        # ISCO pathway: Direct matching without reverse ISCO→ONET conversion
        # Jobs and exposures both use ISCO codes - direct domain matching
        log(f"✓ Using ISCO domain for jobs-to-exposure matching (recommended pathway)")

        # Rename isco_code to isco08_4d to match Stage 5 exposure file column names
        df_mapped = df_mapped.rename(columns={'isco_code': 'isco08_4d'})

    elif target_occ == 'onet' and isco_onet_xwalk is not None:
        # ONET pathway: Map ISCO to ONET (note: this has known information loss issues)
        # Further map ISCO to ONET with format conversion
        # X28→ISCO produces codes like '110', '210', '1111'
        # ISCO→ONET expects codes like '0110.10', '0210.2', '1111.1'

        # Prepare ISCO codes for matching
        df_mapped = df_mapped[df_mapped['isco_code'].notna()].copy()
        log(f"🔍 DEBUG: Starting ISCO→ONET mapping with {len(df_mapped):,} records")

        # Convert ISCO codes to 4-digit zero-padded format
        df_mapped['isco_code_padded'] = df_mapped['isco_code'].astype(str).str.zfill(4)

        # DEBUG: Show sample ISCO codes from jobs
        unique_isco_jobs = df_mapped['isco_code'].unique()[:10]
        unique_isco_padded = df_mapped['isco_code_padded'].unique()[:10]
        log(f"🔍 DEBUG: Sample ISCO codes from jobs (original): {unique_isco_jobs}")
        log(f"🔍 DEBUG: Sample ISCO codes from jobs (padded): {unique_isco_padded}")

        # Create a crosswalk that matches base ISCO codes (ignoring decimal suffixes)
        isco_onet_base = isco_onet_xwalk.copy()
        isco_onet_base['isco_base'] = isco_onet_base['isco_code'].astype(str).str.split('.').str[0]

        # DEBUG: Show sample ISCO codes from crosswalk
        unique_isco_xwalk = isco_onet_base['isco_code'].unique()[:10]
        unique_isco_base = isco_onet_base['isco_base'].unique()[:10]
        log(f"🔍 DEBUG: Sample ISCO codes from crosswalk (original): {unique_isco_xwalk}")
        log(f"🔍 DEBUG: Sample ISCO codes from crosswalk (base): {unique_isco_base}")

        # Remove duplicates by keeping first occurrence of each base ISCO code
        isco_onet_base = isco_onet_base.drop_duplicates(subset=['isco_base']).reset_index(drop=True)
        log(f"🔍 DEBUG: After dedup, crosswalk has {len(isco_onet_base):,} unique base codes")

        # DEBUG: Check overlap between job codes and crosswalk codes
        jobs_set = set(df_mapped['isco_code_padded'].unique())
        xwalk_set = set(isco_onet_base['isco_base'].unique())
        overlap = jobs_set.intersection(xwalk_set)
        missing_from_xwalk = jobs_set - xwalk_set

        log(f"🔍 DEBUG: Overlap between job codes and crosswalk codes: {len(overlap)} codes")
        if len(overlap) > 0:
            log(f"🔍 DEBUG: Sample overlapping codes: {list(overlap)[:10]}")

        log(f"🔍 DEBUG: Missing from crosswalk: {len(missing_from_xwalk)} codes")
        if len(missing_from_xwalk) > 0:
            log(f"🔍 DEBUG: Sample missing codes: {sorted(list(missing_from_xwalk))[:20]}")

            # CRITICAL VALIDATION: Check if any specific 4-digit unit group codes are missing
            # Broad codes end in multiple zeros (e.g. 2000, 2100, 3000) - these are expected to be missing
            # Specific codes have varied endings (e.g. 2111, 2112, 2523) - these should not be missing

            def is_broad_code(code):
                """Check if a 4-digit code represents a broad category (ends in 00 or 0)"""
                if len(code) != 4:
                    return len(code) < 4  # Codes shorter than 4 digits are broad
                # 4-digit codes ending in 00 or 0X are broad categories
                return code.endswith('00') or (code.endswith('0') and not code.endswith('00'))

            specific_missing = [code for code in missing_from_xwalk if not is_broad_code(code)]
            broad_missing = [code for code in missing_from_xwalk if is_broad_code(code)]

            if specific_missing:
                log(f"🚨 ERROR: {len(specific_missing)} specific 4-digit ISCO unit group codes missing from crosswalk!")
                log(f"Missing specific codes: {sorted(specific_missing)}")
                raise ValueError(f"ISCO→ONET crosswalk is missing specific unit group codes: {specific_missing[:10]}. "
                               f"This indicates incomplete crosswalk coverage for specific occupations.")

            log(f"📝 NOTE: {len(broad_missing)} broad ISCO codes are missing from crosswalk")
            log(f"This is expected - ESCO crosswalk only covers specific 4-digit unit groups, not broad categories")

            # Count how many jobs each missing code represents
            missing_counts = df_mapped[df_mapped['isco_code_padded'].isin(missing_from_xwalk)]['isco_code_padded'].value_counts()
            log(f"🔍 DEBUG: Top missing codes by job count:")
            for code, count in missing_counts.head(10).items():
                if is_broad_code(code):
                    code_type = "broad category"
                else:
                    code_type = "specific unit group"
                log(f"  {code} ({code_type}): {count:,} jobs")

        # Also check for ISCO codes in crosswalk but not in jobs
        extra_in_xwalk = xwalk_set - jobs_set
        log(f"🔍 DEBUG: ISCO codes in crosswalk but not in jobs: {len(extra_in_xwalk)} codes")
        if len(extra_in_xwalk) > 0:
            log(f"🔍 DEBUG: Sample unused crosswalk codes: {sorted(list(extra_in_xwalk))[:10]}")

        # Merge on the base ISCO codes
        df_mapped = df_mapped.merge(
            isco_onet_base[['isco_base', 'onet_soc']],
            left_on='isco_code_padded',
            right_on='isco_base',
            how='left'
        )

        # DEBUG: Check merge results
        matched_records = df_mapped['onet_soc'].notna().sum()
        log(f"🔍 DEBUG: After merge, {matched_records:,} records have ONET codes")

        # Clean up temporary columns
        df_mapped = df_mapped.drop(columns=['isco_code_padded', 'isco_base'])

        unmapped_isco = df_mapped['onet_soc'].isna().sum()
        final_mapping_rate = ((len(df_mapped) - unmapped_isco) / len(df_mapped)) * 100

        log(f"ISCO→ONET merge success: {len(df_mapped) - unmapped_isco:,} of {len(df_mapped):,} ({final_mapping_rate:.1f}%)")
    
    # Remove unmapped records and return exploded format
    occ_col = 'isco08_4d' if target_occ == 'isco' else 'onet_soc'

    unmapped_count = df_mapped[occ_col].isna().sum()
    log(f"Unmapped jobs to be filtered: {unmapped_count:,}")

    before_filter = len(df_mapped)
    df_final = df_mapped[df_mapped[occ_col].notna()].copy()
    log_transformation(
        f"Filter: Remove unmapped {occ_col}",
        before_filter,
        len(df_final),
        f"{unmapped_count:,} unmapped removed"
    )

    # Clean up columns - remove x28_codes and x28_code columns
    cols_to_keep = ['company_id', 'company_name', 'year', 'title', occ_col]
    before_dedup = len(df_final)
    df_final = df_final[cols_to_keep].drop_duplicates().reset_index(drop=True)
    log_transformation(
        "Final deduplication (all columns)",
        before_dedup,
        len(df_final)
    )

    log(f"✅ Final mapped jobs: {len(df_final):,} records")
    return df_final

def link_jobs_to_exposures(df_jobs_mapped, df_exposures):
    """Core linking logic: join jobs to exposures and generate diagnostics"""
    log("🔗 Performing inner join between jobs and exposures...")

    jobs_count = len(df_jobs_mapped)
    exposures_count = len(df_exposures)
    log(f"Preparing to link: {jobs_count:,} jobs × {exposures_count:,} exposures")

    # Determine occupation column name
    occ_col = 'isco08_4d' if 'isco08_4d' in df_jobs_mapped.columns else 'onet_soc'
    exp_occ_col = 'isco08_4d' if 'isco08_4d' in df_exposures.columns else 'onet_soc'

    # Ensure consistent data types for join keys
    df_jobs_mapped = df_jobs_mapped.copy()
    df_exposures = df_exposures.copy()

    # Convert company_id to int for consistent joining
    df_jobs_mapped['company_id'] = df_jobs_mapped['company_id'].astype(int)
    df_exposures['company_id'] = df_exposures['company_id'].astype(int)

    # Convert year to int for consistent joining
    df_jobs_mapped['year'] = df_jobs_mapped['year'].astype(int)
    df_exposures['year'] = df_exposures['year'].astype(int)

    # Convert occupation codes to string for consistent joining
    df_jobs_mapped[occ_col] = df_jobs_mapped[occ_col].astype(str).str.strip()
    df_exposures[exp_occ_col] = df_exposures[exp_occ_col].astype(str).str.strip()

    # Handle different column names between jobs and exposures
    if occ_col != exp_occ_col:
        # If column names differ, rename one to match the other for joining
        if exp_occ_col == 'isco08_4d':
            # Rename jobs column to match exposure column
            df_jobs_mapped = df_jobs_mapped.rename(columns={occ_col: exp_occ_col})
            join_occ_col = exp_occ_col
        else:
            # Rename exposure column to match jobs column
            df_exposures = df_exposures.rename(columns={exp_occ_col: occ_col})
            join_occ_col = occ_col
    else:
        join_occ_col = occ_col

    # Inner join on (company_id, year, occupation_code)
    join_keys = ['company_id', 'year', join_occ_col]
    df_linked = df_jobs_mapped.merge(
        df_exposures,
        on=join_keys,
        how='inner'
    )

    after_merge = len(df_linked)
    join_rate = (after_merge / jobs_count * 100) if jobs_count > 0 else 0
    log(f"Inner join result: {after_merge:,} records (join rate: {join_rate:.1f}%)")

    # Clean up duplicate company_name columns (merge creates _x and _y suffixes)
    if 'company_name_x' in df_linked.columns and 'company_name_y' in df_linked.columns:
        # Use company_name from jobs data (usually more complete/reliable)
        df_linked['company_name'] = df_linked['company_name_x'].fillna(df_linked['company_name_y'])
        df_linked = df_linked.drop(columns=['company_name_x', 'company_name_y'])
        log("🔧 Cleaned up duplicate company_name columns")

    # Deduplicate to ensure only one row per (year, company, onet-soc) combination
    # Keep first occurrence to preserve data
    before_dedup = len(df_linked)
    dedup_keys = ['company_id', 'company_name', 'year', join_occ_col]
    df_linked = df_linked.drop_duplicates(subset=dedup_keys, keep='first')

    if before_dedup > len(df_linked):
        log_transformation(
            f"Deduplication by (company_id, year, {join_occ_col})",
            before_dedup,
            len(df_linked),
            "kept='first'"
        )
        # Show sample duplicates
        duplicates = df_linked[df_linked.duplicated(subset=dedup_keys, keep=False)].head(10)
        if len(duplicates) > 0:
            log_sample_records(
                duplicates.sort_values(dedup_keys),
                "Sample duplicate records",
                n=10,
                cols=['company_id', 'year', join_occ_col, 'company_name']
            )

    log(f"✅ Successfully linked {len(df_linked):,} unique job-exposure records")

    # Log exposure columns for transparency
    exposure_cols = [c for c in df_linked.columns if '_exposure_avg_p' in c and '_ce' in c]
    if exposure_cols:
        log(f"\n📊 Exposure specifications in linked data:")
        # Extract unique specifications from column names
        specs = set()
        for col in exposure_cols:
            if 'hampole' in col:
                parts = col.split('_')
                pct_part = [p for p in parts if p.startswith('p')][-1]
                ce_part = [p for p in parts if p.startswith('ce')][-1]
                specs.add((pct_part, ce_part))

        for pct, ce in sorted(specs):
            hampole_col = f'hampole_ai_exposure_avg_{pct}_{ce}'
            if hampole_col in df_linked.columns:
                non_zero = (df_linked[hampole_col] > 0).sum()
                log(f"  {pct}_{ce}: {non_zero:,} non-zero Hampole exposures")

    # Generate unmatched jobs diagnostic
    df_unmatched_jobs = df_jobs_mapped[
        ~df_jobs_mapped.set_index(join_keys).index.isin(
            df_linked.set_index(join_keys).index
        )
    ].copy()

    log(f"📊 Unmatched jobs: {len(df_unmatched_jobs):,} records")
    if len(df_unmatched_jobs) > 0:
        log_sample_records(
            df_unmatched_jobs,
            "Sample unmatched jobs",
            n=10,
            cols=['company_id', 'year', join_occ_col, 'company_name']
        )

    # Generate unmatched exposures diagnostic
    df_unmatched_exposures = df_exposures[
        ~df_exposures.set_index(join_keys).index.isin(
            df_linked.set_index(join_keys).index
        )
    ].copy()

    log(f"📊 Unmatched exposures: {len(df_unmatched_exposures):,} records")
    if len(df_unmatched_exposures) > 0:
        log_sample_records(
            df_unmatched_exposures,
            "Sample unmatched exposures",
            n=10,
            cols=['company_id', 'year', join_occ_col]
        )
    
    return df_linked, df_unmatched_jobs, df_unmatched_exposures

def main():
    parser = argparse.ArgumentParser(description="Stage 6: Link Firm×Year Exposure to Jobs")
    parser.add_argument('--stage5_dir', type=str, required=True,
                       help='Path to Stage 5 input directory (e.g., Data/Testing/stage_5_core)')
    parser.add_argument('--crosswalk_dir', type=str, default='Data',
                       help='Path to crosswalk files directory (default: Data/)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory for all Stage 6 files')
    parser.add_argument('--job_list', choices=['job_ads', 'shp'], default='job_ads',
                       help='Source for job list')
    parser.add_argument('--occ_code', choices=['isco', 'onet'], default='isco',
                       help='Target occupation coding system')
    parser.add_argument('--time_var', type=bool, default=True,
                       help='Use time variant exposure data')
    parser.add_argument('--year_max', type=int, default=2025,
                       help='Maximum year for job expansion')
    parser.add_argument('--fail_on_warn', action='store_false', default=True,
                       help='Treat warnings as errors (use --fail_on_warn to disable)')
    parser.add_argument('--generate_firm_report', action='store_true',
                       help='Generate firm-level AI summary report')
    parser.add_argument('--task_type', type=str,
                       choices=['core', 'all'],
                       default='core',
                       help="Task type to process (default: core)")
    parser.add_argument('--job_app_mapping_file', type=str,
                       default=None,
                       help="Optional path to Stage 4 job_app_mapping parquet to attach ai_app_id per job_uid")
    parser.add_argument('--ai_apps_file', type=str,
                       default=None,
                       help="Path to Stage 3 output file containing AI applications (required if --generate_firm_report is used)")
    parser.add_argument('--ai_jobs_file', type=str,
                       default='Data/ai_development_deuplicated_custom.csv',
                       help="Path to AI jobs file for firm reports ai_development_deuplicated_custom.csv (required if --generate_firm_report is used)")

    args = parser.parse_args()

    # Auto-generate all output paths based on output_dir, task_type, and occ_code
    task_suffix = f"_{args.task_type}"
    occ_suffix = f"_{args.occ_code}"

    main_output = Path(args.output_dir) / f"stage6_jobs_linked{task_suffix}{occ_suffix}.csv"
    unmatched_jobs = Path(args.output_dir) / f"stage6_unmatched_jobs{task_suffix}{occ_suffix}.csv"
    unmatched_exposures = Path(args.output_dir) / f"stage6_unmatched_exposures{task_suffix}{occ_suffix}.csv"
    firm_report_output = Path(args.output_dir) / f"firm_ai_summary_report{task_suffix}{occ_suffix}.csv"

    log("🚀 Starting Stage 6: Link Firm×Year Exposure to Jobs")
    log("=" * 60)

    # Validate required directories exist and create output directory
    if not Path(args.stage5_dir).exists():
        raise FileNotFoundError(f"Stage 5 input directory not found: {args.stage5_dir}")
    if not Path(args.crosswalk_dir).exists():
        raise FileNotFoundError(f"Crosswalk directory not found: {args.crosswalk_dir}")

    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log(f"✅ Output directory ready: {args.output_dir}")

    # Load environment and connect to database
    load_dotenv('config.env')
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD') 
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    
    conn = psycopg2.connect(
        dbname=db_name, user=db_user, password=db_password,
        host=db_host, port=db_port
    )
    
    try:
        # Build job list
        if args.job_list == 'job_ads':
            df_jobs_exploded = build_jobs_from_db(conn, args.year_max, args.fail_on_warn)
        else:
            raise NotImplementedError("SHP job list not yet implemented")
        
        # Load crosswalks
        x28_isco_path = Path(args.crosswalk_dir) / "240711_occupation_to_ch_isco_19.csv"
        x28_isco_xwalk = load_crosswalk_x28_to_isco(x28_isco_path)

        isco_onet_xwalk = None
        if args.occ_code == 'onet':
            isco_onet_path = Path(args.crosswalk_dir) / "ESCO_to_ONET-SOC.xlsx"
            isco_onet_xwalk = load_crosswalk_isco_to_onet(isco_onet_path)
        
        # Apply crosswalks
        df_jobs_mapped = apply_crosswalks(
            df_jobs_exploded, x28_isco_xwalk, isco_onet_xwalk, args.occ_code
        )
        
        # Load Stage 5 exposures
        stage5_path = find_stage5_file(args.stage5_dir, args.occ_code, args.task_type)
        df_exposures = load_stage5_exposures(stage5_path, args.occ_code)
        
        log(f"🔗 Linking jobs to exposures...")
        
        # Perform core linking logic
        df_linked, df_unmatched_jobs, df_unmatched_exposures = link_jobs_to_exposures(
            df_jobs_mapped, df_exposures
        )

        # Optional: attach ai_app_id per job from Stage 4 mapping so downstream analyses can trace back to apps
        if args.job_app_mapping_file:
            if not Path(args.job_app_mapping_file).exists():
                log(f"⚠️  job_app_mapping_file not found: {args.job_app_mapping_file}")
            else:
                # We can only attach ai_app_ids if df_linked still has job_uid in it
                if 'job_uid' not in df_linked.columns:
                    log("⚠️  Stage 6 linked data has no 'job_uid' column; skipping ai_app_id attachment")
                else:
                    try:
                        jam = pd.read_parquet(args.job_app_mapping_file)
                        if {'job_uids', 'ai_app_id'}.issubset(jam.columns):
                            jam_exp = jam[['job_uids', 'ai_app_id']].copy()
                            jam_exp['job_uid'] = jam_exp['job_uids'].astype(str).str.split('|')
                            jam_exp = jam_exp.explode('job_uid')
                            jam_exp = jam_exp.dropna(subset=['job_uid'])
                            jam_exp['job_uid'] = jam_exp['job_uid'].astype(str)
                            # Aggregate ai_app_ids per job to avoid row explosion
                            jam_grouped = jam_exp.groupby('job_uid')['ai_app_id'].agg(lambda s: sorted(set(s))).reset_index()
                            jam_grouped.rename(columns={'ai_app_id': 'ai_app_ids'}, inplace=True)
                            df_linked = df_linked.merge(jam_grouped, on='job_uid', how='left')
                            log(f"✅ Attached ai_app_ids from mapping to linked jobs (coverage: {(~df_linked['ai_app_ids'].isna()).sum():,} rows)")
                        else:
                            log("⚠️  job_app_mapping_file missing required columns {job_uids, ai_app_id}; skipping ai_app_id attachment")
                    except Exception as e:
                        log(f"⚠️  Failed to attach ai_app_ids from mapping: {e}")
        
        # Save outputs
        log(f"💾 Saving results...")
        df_linked.to_csv(main_output, index=False)
        log(f"✅ Main results saved: {main_output}")

        if len(df_unmatched_jobs) > 0:
            df_unmatched_jobs.to_csv(unmatched_jobs, index=False)
            log(f"📋 Unmatched jobs saved: {unmatched_jobs}")

        if len(df_unmatched_exposures) > 0:
            df_unmatched_exposures.to_csv(unmatched_exposures, index=False)
            log(f"📋 Unmatched exposures saved: {unmatched_exposures}")
        
        # Generate firm summary report if requested
        if args.generate_firm_report:
            # Validate required files for firm report
            if not args.ai_apps_file:
                raise ValueError("--ai_apps_file is required when --generate_firm_report is used")
            if not args.ai_jobs_file:
                raise ValueError("--ai_jobs_file is required when --generate_firm_report is used")

            log("🚀 Generating firm summary report...")
            try:
                generate_firm_summary_report(df_linked, df_jobs_exploded, args, firm_report_output)
            except Exception as e:
                log(f"❌ Firm report generation failed: {e}")
                log("⚠️  Continuing with main Stage 6 processing...")

        # Pipeline summary
        log("\n" + "=" * 80)
        log("STAGE 6 PIPELINE SUMMARY")
        log("=" * 80)
        log(f"Input jobs from database: {len(df_jobs_exploded):,}")
        log(f"Jobs after crosswalks: {len(df_jobs_mapped):,}")
        log(f"Exposure records loaded: {len(df_exposures):,}")
        log(f"Successfully linked: {len(df_linked):,}")
        pipeline_efficiency = (len(df_linked) / len(df_jobs_exploded) * 100) if len(df_jobs_exploded) > 0 else 0
        log(f"Overall pipeline efficiency: {pipeline_efficiency:.1f}%")
        log(f"Unmatched jobs: {len(df_unmatched_jobs):,}")
        log(f"Unmatched exposures: {len(df_unmatched_exposures):,}")
        log("=" * 80)

        log(f"✅ Stage 6 complete! Linked {len(df_linked):,} records")

    finally:
        conn.close()

if __name__ == "__main__":
    main()