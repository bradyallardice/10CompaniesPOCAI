#!/usr/bin/env python3
"""
Standalone script to generate firm report from existing Stage 6 outputs
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def log(msg):
    """Print timestamped log message"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def deduplicate_jobs_within_companies(df_jobs, similarity_threshold=1):
    """
    Deduplicate jobs within each company using Stage 2's approach.
    """
    log(f"🔧 Starting job deduplication within companies (threshold: {similarity_threshold})...")
    initial_count = len(df_jobs)

    # Step 1: Remove exact duplicates first (company_id + title + x28_occupations)
    log("Step 1: Removing exact duplicates (company_id + title + x28_occupations)...")
    # Sort by year to keep earliest posting when deduplicating
    df_sorted = df_jobs.sort_values(['company_id', 'year'], ascending=True)

    # Create deduplication key - using available fields from jobs cache
    df_sorted['dedup_key'] = (
        df_sorted['company_id'].astype(str) + '|||' +
        df_sorted['title'].fillna('').str.lower().str.strip() + '|||' +
        df_sorted['x28_occupations'].astype(str)
    )

    df_step1 = df_sorted.drop_duplicates(subset=['dedup_key'], keep='first')
    exact_removed = initial_count - len(df_step1)
    log(f"Removed {exact_removed:,} exact duplicates")
    log(f"Remaining: {len(df_step1):,} jobs")

    if len(df_step1) == 0:
        return df_step1

    # For jobs cache, we don't have content_clean, so we'll just use exact deduplication
    # and then extend jobs through time
    log("Step 2: Extending unique jobs through time (2019-2025)...")

    # Get the earliest year for each unique job
    df_step1 = df_step1.copy()
    df_step1['start_year'] = df_step1.groupby('dedup_key')['year'].transform('min')

    # Create a list to store all extended jobs
    extended_jobs = []

    # Group by the unique job key and extend each through time
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

    # Remove duplicates that might have been created by the extension process
    df_final = df_extended.drop_duplicates(subset=['company_id', 'year', 'title'], keep='first')

    log(f"✅ Job deduplication complete:")
    log(f"   Original jobs: {initial_count:,}")
    log(f"   After exact dedup: {len(df_step1):,}")
    log(f"   After time extension: {len(df_final):,}")

    return df_final

def find_latest_ai_applications_file(data_dir):
    """Find the latest AI applications file matching pattern ai_development*step1_extracted_step2_step3.csv"""
    log("🔍 Looking for AI applications file...")

    llm_output_dir = Path(data_dir) / "llm_output"
    if not llm_output_dir.exists():
        raise FileNotFoundError(f"LLM output directory not found: {llm_output_dir}")

    # Find files matching pattern
    pattern = "ai_development*step1_extracted_step2_step3.csv"
    matching_files = list(llm_output_dir.glob(pattern))

    if not matching_files:
        raise FileNotFoundError(f"No AI applications files found matching pattern: {pattern}")

    # Get the latest file by modification time
    latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
    log(f"✅ Found latest AI applications file: {latest_file.name}")

    return str(latest_file)

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

def generate_firm_summary_report(df_linked, df_jobs_cache, data_dir, output_file):
    """Generate comprehensive firm-level AI summary report"""
    log("🚀 Starting firm summary report generation...")

    try:
        # Step 1: Deduplicate jobs within companies and extend through time
        log("📊 Processing job data...")
        df_jobs_dedup = deduplicate_jobs_within_companies(df_jobs_cache)

        # Ensure company_id and year are int for consistent merging
        df_jobs_dedup['company_id'] = df_jobs_dedup['company_id'].astype(int)
        df_jobs_dedup['year'] = df_jobs_dedup['year'].astype(int)

        # Count unique jobs per firm-year
        df_job_counts = df_jobs_dedup.groupby(['company_id', 'company_name', 'year']).size().reset_index(name='total_unique_job_ads')

        # Step 2: Load AI applications data (all apps, not just linked)
        try:
            ai_apps_file = find_latest_ai_applications_file(data_dir)
            df_ai_apps_all = load_ai_applications_data(ai_apps_file)
            # Ensure consistent data types
            df_ai_apps_all['company_id'] = df_ai_apps_all['company_id'].astype(int)
            df_ai_apps_all['year'] = df_ai_apps_all['year'].astype(int)
        except FileNotFoundError as e:
            log(f"⚠️  Warning: {e}")
            df_ai_apps_all = pd.DataFrame(columns=['company_id', 'year', 'total_ai_apps_all'])

        # Step 3: Load AI jobs data (unique AI jobs)
        ai_jobs_file = Path(data_dir) / "ai_development_deduplicated_custom.csv"
        df_ai_jobs = load_ai_jobs_data(ai_jobs_file)
        # Ensure consistent data types
        df_ai_jobs['company_id'] = df_ai_jobs['company_id'].astype(int)
        df_ai_jobs['year'] = df_ai_jobs['year'].astype(int)

        # Step 4: Extract Stage 6 linked data metrics
        log("📊 Processing Stage 6 exposure data...")

        # Ensure consistent data types in df_linked
        df_linked['company_id'] = df_linked['company_id'].astype(int)
        df_linked['year'] = df_linked['year'].astype(int)

        # Determine which occupation column exists (ISCO or ONET)
        if 'isco08_4d' in df_linked.columns:
            occ_col = 'isco08_4d'
            log("Using ISCO occupation codes for aggregation")
        elif 'onet_soc' in df_linked.columns:
            occ_col = 'onet_soc'
            log("Using ONET occupation codes for aggregation")
        else:
            raise ValueError("Neither 'isco08_4d' nor 'onet_soc' columns found in Stage 6 data")

        # Aggregate Stage 6 linked data by firm-year
        stage6_agg = df_linked.groupby(['company_id', 'company_name', 'year']).agg({
            'n_ai_apps_firm_year': 'first',  # This should be consistent within firm-year
            'total_tasks_occupation': 'sum',  # Sum all tasks across occupations at this firm
            'hampole_ai_exposure_avg': lambda x: (x > 0).sum()  # Count AI-exposed tasks
        }).reset_index()

        stage6_agg.columns = ['company_id', 'company_name', 'year', 'total_ai_apps_linked', 'total_onet_tasks', 'ai_exposed_tasks']

        # Ensure consistent data types in aggregated data
        stage6_agg['company_id'] = stage6_agg['company_id'].astype(int)
        stage6_agg['year'] = stage6_agg['year'].astype(int)

        # Step 5: Merge all data sources
        log("🔗 Merging data sources...")

        # Start with job counts (this defines our universe of firms)
        df_report = df_job_counts.copy()

        # Merge AI applications (all)
        df_report = df_report.merge(
            df_ai_apps_all,
            on=['company_id', 'year'],
            how='left'
        )

        # Merge AI jobs
        df_report = df_report.merge(
            df_ai_jobs,
            on=['company_id', 'year'],
            how='left'
        )

        # Merge Stage 6 data
        df_report = df_report.merge(
            stage6_agg,
            on=['company_id', 'company_name', 'year'],
            how='left'
        )

        # Fill missing values with 0
        numeric_cols = ['total_ai_apps_all', 'total_unique_ai_jobs', 'total_ai_apps_linked',
                       'total_onet_tasks', 'ai_exposed_tasks']
        df_report[numeric_cols] = df_report[numeric_cols].fillna(0)

        # Convert to integers where appropriate
        for col in numeric_cols:
            df_report[col] = df_report[col].astype(int)

        # Step 6: Calculate percentage and exposure metrics
        log("📊 Calculating percentage and exposure metrics...")

        # Yearly percentage: AI jobs / total jobs for each year
        df_report['pct_ai_ads_yearly'] = (df_report['total_unique_ai_jobs'] / df_report['total_unique_job_ads'] * 100).round(2)

        # Cumulative percentage: group by company and calculate cumulative sums
        df_report['cumulative_ai_jobs'] = df_report.groupby('company_id')['total_unique_ai_jobs'].cumsum()
        df_report['cumulative_total_jobs'] = df_report.groupby('company_id')['total_unique_job_ads'].cumsum()
        df_report['pct_ai_ads_cumulative'] = (df_report['cumulative_ai_jobs'] / df_report['cumulative_total_jobs'] * 100).round(2)

        # Firm AI exposure: exposed tasks / total tasks
        df_report['firm_ai_exposure'] = (df_report['ai_exposed_tasks'] / df_report['total_onet_tasks'] * 100).round(2)

        # Replace inf and NaN values with 0 (occurs when dividing by 0)
        df_report['pct_ai_ads_yearly'] = df_report['pct_ai_ads_yearly'].replace([float('inf'), -float('inf')], pd.NA)
        df_report['pct_ai_ads_cumulative'] = df_report['pct_ai_ads_cumulative'].replace([float('inf'), -float('inf')], pd.NA)
        df_report['firm_ai_exposure'] = df_report['firm_ai_exposure'].replace([float('inf'), -float('inf')], pd.NA).fillna(0)

        # Drop temporary cumulative columns
        df_report = df_report.drop(columns=['cumulative_ai_jobs', 'cumulative_total_jobs'])

        # Step 7: Final report formatting and validation
        log("✅ Finalizing report...")

        # Ensure we have all required columns in the right order
        final_columns = [
            'company_id', 'company_name', 'year',
            'total_unique_job_ads', 'total_unique_ai_jobs',
            'pct_ai_ads_yearly', 'pct_ai_ads_cumulative',
            'total_ai_apps_all', 'total_ai_apps_linked',
            'total_onet_tasks', 'ai_exposed_tasks', 'firm_ai_exposure'
        ]

        df_report = df_report[final_columns]

        # Sort by company_id and year
        df_report = df_report.sort_values(['company_id', 'year']).reset_index(drop=True)

        # Save the report
        df_report.to_csv(output_file, index=False)

        log(f"✅ Firm summary report generated successfully!")
        log(f"   Report saved: {output_file}")
        log(f"   Firms covered: {df_report['company_id'].nunique():,}")
        log(f"   Total firm-year records: {len(df_report):,}")
        log(f"   Year range: {df_report['year'].min()}-{df_report['year'].max()}")

        # Show summary statistics
        log("📊 Summary statistics:")
        log(f"   Firms with AI jobs: {(df_report['total_unique_ai_jobs'] > 0).sum():,}")
        log(f"   Firms with AI applications (all): {(df_report['total_ai_apps_all'] > 0).sum():,}")
        log(f"   Firms with linked AI applications: {(df_report['total_ai_apps_linked'] > 0).sum():,}")
        log(f"   Firms with AI-exposed O*NET tasks: {(df_report['ai_exposed_tasks'] > 0).sum():,}")

        return df_report

    except Exception as e:
        log(f"❌ Error generating firm summary report: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate firm report from existing Stage 6 outputs")
    parser.add_argument('--stage6_file', default='Data/stage6_jobs_linked.csv',
                       help='Stage 6 linked jobs file')
    parser.add_argument('--job_cache', default='Data/stage6_job_cache_max2025.parquet',
                       help='Job cache file')
    parser.add_argument('--data_dir', default='Data',
                       help='Data directory')
    parser.add_argument('--output', default='Data/firm_ai_summary_report.csv',
                       help='Output file path')

    args = parser.parse_args()

    log("🚀 Starting standalone firm report generation")
    log("=" * 60)

    # Load existing Stage 6 output
    log(f"📂 Loading Stage 6 output: {args.stage6_file}")
    df_linked = pd.read_csv(args.stage6_file)
    log(f"✅ Loaded {len(df_linked):,} linked job-exposure records")

    # Load job cache
    log(f"📂 Loading job cache: {args.job_cache}")
    df_jobs_cache = pd.read_parquet(args.job_cache)
    log(f"✅ Loaded {len(df_jobs_cache):,} cached jobs")

    # Generate report
    generate_firm_summary_report(df_linked, df_jobs_cache, args.data_dir, args.output)

    log("✅ Done!")

if __name__ == "__main__":
    main()
