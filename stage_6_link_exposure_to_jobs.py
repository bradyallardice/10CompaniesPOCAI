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
import re
import yaml
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from datetime import datetime

def log(msg):
    """Print timestamped log message"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def compute_file_hash(filepath):
    """Compute MD5 hash of file for drift detection"""
    if not Path(filepath).exists():
        return None
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def parse_x28_occupations(x28_str):
    """Parse X28 occupations field into list of clean strings"""
    if not x28_str or pd.isna(x28_str) or str(x28_str).strip() in ('', 'null', 'None'):
        return []
    
    x28_str = str(x28_str).strip()
    
    # Handle JSON-style arrays: ['123', '456'] or ["123", "456"]
    if x28_str.startswith('[') and x28_str.endswith(']'):
        try:
            # Try JSON parsing first
            parsed = json.loads(x28_str)
            if isinstance(parsed, list):
                return [str(x).strip().strip('\'"') for x in parsed if x]
        except:
            # Fallback: manual parsing
            x28_str = x28_str[1:-1]  # Remove brackets
    
    # Handle comma-separated values
    codes = []
    for code in x28_str.split(','):
        code = code.strip().strip('\'"')
        if code:
            codes.append(code)
    
    return codes

def validate_jobs_data(df_jobs, fail_on_warn=True):
    """Validate jobs data with fail-fast checks"""
    total_rows = len(df_jobs)
    
    if total_rows == 0:
        raise ValueError("❌ FATAL: 0 rows after job expansion")
    
    # Check for empty X28 occupations
    empty_x28 = df_jobs['x28_codes'].apply(lambda x: len(x) == 0).sum()
    empty_pct = (empty_x28 / total_rows) * 100
    
    if empty_pct > 2.0:
        msg = f"❌ FATAL: {empty_pct:.1f}% rows have empty x28_occupations (threshold: 2%)"
        if fail_on_warn:
            raise ValueError(msg)
        else:
            log(f"⚠️  WARNING: {msg}")
    
    # Check for duplicates
    key_cols = ['company_id', 'year', 'title']
    duplicates = df_jobs.duplicated(subset=key_cols).sum()
    if duplicates > 0:
        raise ValueError(f"❌ FATAL: {duplicates} duplicate (company_id, year, title) combinations")
    
    log(f"✅ Jobs validation passed: {total_rows:,} rows, {empty_pct:.1f}% empty X28")

def build_jobs_from_db(conn, year_max, fail_on_warn=True):
    """Build firm×year×title job list from job_postings_unified"""
    log("🔍 Building job list from database...")
    
    sql = f"""
    WITH base AS (
      SELECT 
        company_id, 
        company_name, 
        title, 
        x28_occupations,
        DATE_PART('year', MIN(tst_created)) OVER (PARTITION BY company_id, title) AS start_year
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
    
    # Validate
    validate_jobs_data(df_jobs, fail_on_warn)
    
    return df_jobs

def load_crosswalk_x28_to_isco(filepath):
    """Load X28 to ISCO crosswalk with auto-detection"""
    log(f"📋 Loading X28→ISCO crosswalk: {filepath}")
    
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Crosswalk file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Auto-detect column names (look for X28 and ISCO patterns)
    x28_col = None
    isco_col = None
    
    for col in df.columns:
        if 'x28' in col.lower() or '28' in col.lower():
            x28_col = col
        if 'isco' in col.lower():
            isco_col = col
    
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
    
    # Try to read Excel file (auto-detect sheet)
    xl_file = pd.ExcelFile(filepath)
    
    # Look for the most likely sheet
    sheet_candidates = [s for s in xl_file.sheet_names if 'isco' in s.lower() or 'onet' in s.lower()]
    if not sheet_candidates:
        sheet_candidates = xl_file.sheet_names[:1]  # Use first sheet as fallback
    
    df = pd.read_excel(filepath, sheet_name=sheet_candidates[0])
    
    # Auto-detect column names
    isco_col = None
    onet_col = None
    
    for col in df.columns:
        if 'isco' in col.lower():
            isco_col = col
        if 'onet' in col.lower() or 'soc' in col.lower():
            onet_col = col
    
    if not isco_col or not onet_col:
        raise ValueError(f"Could not auto-detect ISCO and ONET columns in {filepath}. Columns: {list(df.columns)}")
    
    log(f"✅ Detected columns: ISCO='{isco_col}', ONET='{onet_col}'")
    
    # Clean and validate
    crosswalk = df[[isco_col, onet_col]].dropna().drop_duplicates()
    crosswalk.columns = ['isco_code', 'onet_soc']
    
    # Convert to string and clean
    crosswalk['isco_code'] = crosswalk['isco_code'].astype(str).str.strip()
    crosswalk['onet_soc'] = crosswalk['onet_soc'].astype(str).str.strip()
    
    log(f"✅ Loaded {len(crosswalk):,} ISCO→ONET mappings")
    return crosswalk

def find_stage5_file(in_dir, occ_code, time_var):
    """Find Stage 5 exposure file with pattern matching"""
    suffix = "time_variant" if time_var else "time_invariant"
    pattern = f"{occ_code}_firm_year_ai_exposure_{suffix}.csv"
    filepath = Path(in_dir) / pattern
    
    if not filepath.exists():
        raise FileNotFoundError(f"Stage 5 file not found: {filepath}")
    
    return filepath

def load_stage5_exposures(filepath, occ_code):
    """Load Stage 5 exposure file"""
    log(f"📊 Loading Stage 5 exposures: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = ['company_id', 'year']
    occ_col = 'isco_code' if occ_code == 'isco' else 'onet_soc'
    required_cols.append(occ_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Stage 5 file: {missing_cols}")
    
    log(f"✅ Loaded {len(df):,} exposure records")
    return df

def apply_crosswalks(df_jobs, x28_isco_xwalk, isco_onet_xwalk=None, target_occ='isco'):
    """Apply crosswalks to map job X28 codes to target occupation codes
    
    Returns exploded format with one row per occupation code for joining.
    """
    log(f"🔧 Applying crosswalks to map to {target_occ.upper()}...")
    
    # Explode X28 codes (one row per code)
    df_expanded = df_jobs.explode('x28_codes').reset_index(drop=True)
    df_expanded = df_expanded[df_expanded['x28_codes'].notna()].copy()
    
    if len(df_expanded) == 0:
        raise ValueError("❌ FATAL: No valid X28 codes after expansion")
    
    # Map X28 to ISCO
    df_mapped = df_expanded.merge(
        x28_isco_xwalk, 
        left_on='x28_codes', 
        right_on='x28_code', 
        how='left'
    )
    
    # Check mapping success
    unmapped_x28 = df_mapped['isco_code'].isna().sum()
    mapping_rate = ((len(df_mapped) - unmapped_x28) / len(df_mapped)) * 100
    
    log(f"📊 X28→ISCO mapping: {mapping_rate:.1f}% success ({unmapped_x28:,} unmapped)")
    
    if target_occ == 'onet' and isco_onet_xwalk is not None:
        # Further map ISCO to ONET
        df_mapped = df_mapped[df_mapped['isco_code'].notna()].merge(
            isco_onet_xwalk,
            on='isco_code',
            how='left'
        )
        
        unmapped_isco = df_mapped['onet_soc'].isna().sum()
        final_mapping_rate = ((len(df_mapped) - unmapped_isco) / len(df_mapped)) * 100
        
        log(f"📊 ISCO→ONET mapping: {final_mapping_rate:.1f}% success ({unmapped_isco:,} unmapped)")
    
    # Remove unmapped records and return exploded format
    occ_col = 'isco_code' if target_occ == 'isco' else 'onet_soc'
    df_final = df_mapped[df_mapped[occ_col].notna()].copy()
    
    # Clean up columns - remove x28_codes and x28_code columns
    cols_to_keep = ['company_id', 'company_name', 'year', 'title', occ_col]
    df_final = df_final[cols_to_keep].drop_duplicates().reset_index(drop=True)
    
    log(f"✅ Final mapped jobs: {len(df_final):,} records")
    return df_final

def link_jobs_to_exposures(df_jobs_mapped, df_exposures):
    """Core linking logic: join jobs to exposures and generate diagnostics"""
    log("🔗 Performing inner join between jobs and exposures...")
    
    # Determine occupation column name
    occ_col = 'isco_code' if 'isco_code' in df_jobs_mapped.columns else 'onet_soc'
    
    # Ensure consistent data types for join keys
    df_jobs_mapped = df_jobs_mapped.copy()
    df_exposures = df_exposures.copy()
    
    # Convert company_id to int for consistent joining
    df_jobs_mapped['company_id'] = df_jobs_mapped['company_id'].astype(int)
    df_exposures['company_id'] = df_exposures['company_id'].astype(int)
    
    # Convert year to int for consistent joining
    df_jobs_mapped['year'] = df_jobs_mapped['year'].astype(int)
    df_exposures['year'] = df_exposures['year'].astype(int)
    
    # Convert occupation codes to int for consistent joining
    df_jobs_mapped[occ_col] = df_jobs_mapped[occ_col].astype(int)
    df_exposures[occ_col] = df_exposures[occ_col].astype(int)
    
    # Inner join on (company_id, year, occupation_code)
    join_keys = ['company_id', 'year', occ_col]
    df_linked = df_jobs_mapped.merge(
        df_exposures,
        on=join_keys,
        how='inner'
    )
    
    log(f"✅ Successfully linked {len(df_linked):,} job-exposure records")
    
    # Generate unmatched jobs diagnostic
    df_unmatched_jobs = df_jobs_mapped[
        ~df_jobs_mapped.set_index(join_keys).index.isin(
            df_linked.set_index(join_keys).index
        )
    ].copy()
    
    log(f"📊 Unmatched jobs: {len(df_unmatched_jobs):,} records")
    
    # Generate unmatched exposures diagnostic  
    df_unmatched_exposures = df_exposures[
        ~df_exposures.set_index(join_keys).index.isin(
            df_linked.set_index(join_keys).index
        )
    ].copy()
    
    log(f"📊 Unmatched exposures: {len(df_unmatched_exposures):,} records")
    
    return df_linked, df_unmatched_jobs, df_unmatched_exposures

def main():
    parser = argparse.ArgumentParser(description="Stage 6: Link Firm×Year Exposure to Jobs")
    parser.add_argument('--job_list', choices=['job_ads', 'shp'], default='job_ads',
                       help='Source for job list')
    parser.add_argument('--occ_code', choices=['isco', 'onet'], default='isco',
                       help='Target occupation coding system')  
    parser.add_argument('--time_var', type=bool, default=True,
                       help='Use time variant exposure data')
    parser.add_argument('--year_max', type=int, default=2025,
                       help='Maximum year for job expansion')
    parser.add_argument('--in_dir', default='Data',
                       help='Input directory')
    parser.add_argument('--out', default='Data/stage6_jobs_linked.csv',
                       help='Output file path')
    parser.add_argument('--unmatched_jobs', default='Data/stage6_unmatched_jobs.csv',
                       help='Unmatched jobs output')
    parser.add_argument('--unmatched_exposures', default='Data/stage6_unmatched_exposures.csv', 
                       help='Unmatched exposures output')
    parser.add_argument('--fail_on_warn', type=bool, default=True,
                       help='Treat warnings as errors')
    parser.add_argument('--allow_drift', type=bool, default=False,
                       help='Allow crosswalk file hash changes')
    
    args = parser.parse_args()
    
    log("🚀 Starting Stage 6: Link Firm×Year Exposure to Jobs")
    log("=" * 60)
    
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
            df_jobs = build_jobs_from_db(conn, args.year_max, args.fail_on_warn)
        else:
            raise NotImplementedError("SHP job list not yet implemented")
        
        # Load crosswalks
        x28_isco_path = Path(args.in_dir) / "240711_occupation_to_ch_isco_19.csv"
        x28_isco_xwalk = load_crosswalk_x28_to_isco(x28_isco_path)
        
        isco_onet_xwalk = None
        if args.occ_code == 'onet':
            isco_onet_path = Path(args.in_dir) / "ESCO_to_ONET-SOC.xlsx"
            isco_onet_xwalk = load_crosswalk_isco_to_onet(isco_onet_path)
        
        # Apply crosswalks
        df_jobs_mapped = apply_crosswalks(
            df_jobs, x28_isco_xwalk, isco_onet_xwalk, args.occ_code
        )
        
        # Load Stage 5 exposures
        stage5_path = find_stage5_file(args.in_dir, args.occ_code, args.time_var)
        df_exposures = load_stage5_exposures(stage5_path, args.occ_code)
        
        log(f"🔗 Linking jobs to exposures...")
        
        # Perform core linking logic
        df_linked, df_unmatched_jobs, df_unmatched_exposures = link_jobs_to_exposures(
            df_jobs_mapped, df_exposures
        )
        
        # Save outputs
        log(f"💾 Saving results...")
        df_linked.to_csv(args.out, index=False)
        log(f"✅ Main results saved: {args.out}")
        
        if len(df_unmatched_jobs) > 0:
            df_unmatched_jobs.to_csv(args.unmatched_jobs, index=False)
            log(f"📋 Unmatched jobs saved: {args.unmatched_jobs}")
        
        if len(df_unmatched_exposures) > 0:
            df_unmatched_exposures.to_csv(args.unmatched_exposures, index=False)
            log(f"📋 Unmatched exposures saved: {args.unmatched_exposures}")
        
        log(f"✅ Stage 6 complete! Linked {len(df_linked):,} records")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()