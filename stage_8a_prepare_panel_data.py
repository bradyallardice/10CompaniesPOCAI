#!/usr/bin/env python3
"""
Stage 8a: SHP Panel Data Preparation for Econometric Analysis

Purpose:
  Load SHP exposure data, create fixed effects indicators, prepare outcome variables,
  and produce analysis-ready panel dataset for Specs 1, 2, 3 (from specification-shp-panel-econometrics.md)

Input:
  Data/shp_exposure/shp_exposure_isco4d.csv (479MB, ~200k person-years)

Output:
  Data/shp_panel_prepared.csv (analysis-ready panel with firm_year, occ_year, outcomes)
  Data/shp_panel_preparation_log.txt (diagnostics)

Spec Reference:
  Specification: docs/specification-shp-panel-econometrics.md
  Primary spec: Firm-Year FE + Occupation-Year FE + Person FE

Author: Claude Code
Date: 2026-04-10
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup
project_root = Path(__file__).parent
data_dir = project_root / "Data"
log_file = data_dir / "shp_panel_preparation_log.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_paths():
    """Verify required input files exist."""
    exposure_file = data_dir / "shp_exposure" / "shp_exposure_isco4d.csv"
    if not exposure_file.exists():
        raise FileNotFoundError(
            f"Required input file not found: {exposure_file}\n"
            f"Please run stage_6_shp_exposure.py first to generate SHP exposure data."
        )
    return exposure_file

def load_shp_exposure(exposure_file):
    """Load SHP exposure data with selective columns."""
    logger.info(f"Loading SHP exposure data from {exposure_file.name} (~480MB)...")

    # Column selection: person, year, firm/occupation, outcomes, exposure
    core_cols = [
        'idpers', 'year', 'firm_id', 'isco08_4d', 'isco08_title', 'company_id', 'company_name',
        # Political outcomes
        'pp10',   # Left-right placement (1=left, 10=right)
        'pp15',   # Nativism (1=equal chances, 3=better for Swiss)
        'pp13',   # Social benefits preference (1=less, 3=more)
        'pp17',   # Tax high incomes (1=reduce, 3=increase)
        'pp22',   # Gender equality (0=gone too far, 10=not enough)
        'pp19',   # Vote choice (categorical)
        # Economic outcomes
        'pw86',   # Job insecurity (1=not worried, 4=very worried)
        'pw77',   # Hours worked per week, current main job (loaded as object: contains string codes)
        'iwyn',   # Yearly work income (CHF)
        'wstat',  # Employment status
        # Demographics
        'age', 'sex', 'educat', 'civsta', 'nat_1_',
        # Exposure measures (all aggregation levels available)
        'hampole_ai_exposure_avg_foy', 'binary_ai_exposure_avg_foy',  # Firm-Occ-Year
        'hampole_ai_exposure_avg_fo', 'binary_ai_exposure_avg_fo',    # Firm-Occ
        'hampole_ai_exposure_avg_oy', 'binary_ai_exposure_avg_oy',    # Occ-Year
        'hampole_ai_exposure_avg_o', 'binary_ai_exposure_avg_o',      # Occ
        'hampole_ai_exposure_avg_fy', 'binary_ai_exposure_avg_fy',    # Firm-Year
        'hampole_ai_exposure_avg_f', 'binary_ai_exposure_avg_f',      # Firm
        'hampole_occupation_exposure_foy', 'hampole_occupation_exposure_o',
        'total_tasks_occupation', 'log_ai_intensity', 'n_ai_apps_firm_year'
    ]

    df = pd.read_csv(
        exposure_file,
        dtype={
            'idpers': 'int64',
            'year': 'int32',
            'firm_id': 'float64',
            'age': 'float64',
            'pp10': 'float64', 'pp13': 'float64', 'pp15': 'float64',
            'pp17': 'float64', 'pp19': 'object', 'pp22': 'float64',
            'pw86': 'float64', 'iwyn': 'float64'
        },
        usecols=core_cols,
        low_memory=False
    )

    logger.info(f"Loaded {len(df):,} person-year observations, {df['idpers'].nunique():,} unique persons")
    return df

def create_fixed_effect_indicators(df):
    """Create firm-year and occupation-year identifiers for FE regressions."""
    logger.info("Creating fixed effects indicators...")

    # Firm-Year FE: firm_id_year (string to preserve integer firm_ids)
    df['firm_year'] = (df['firm_id'].astype('Int64', errors='ignore')
                       .astype(str) + '_' + df['year'].astype(str))

    # Occupation-Year FE: isco_year
    df['occ_year'] = df['isco08_4d'].astype(str) + '_' + df['year'].astype(str)

    # Verify creation
    n_persons = df['idpers'].nunique()
    n_firm_years = df['firm_year'].nunique()
    n_occ_years = df['occ_year'].nunique()
    n_with_firm = df['firm_id'].notna().sum()

    logger.info(f"Fixed effects created:")
    logger.info(f"  - Persons: {n_persons:,}")
    logger.info(f"  - Firm-Year FE: {n_firm_years:,} unique values")
    logger.info(f"  - Occupation-Year FE: {n_occ_years:,} unique values")
    logger.info(f"  - Rows with firm_id: {n_with_firm:,} ({100*n_with_firm/len(df):.1f}%)")

    return df

def prepare_outcomes(df):
    """Create standardized outcome variables for analysis."""
    logger.info("Preparing outcome variables...")

    outcomes_created = {}

    # 1. Left-right placement (continuous, 1-10 scale)
    if 'pp10' in df.columns:
        df['outcome_leftright'] = df['pp10'].astype('float64')
        outcomes_created['leftright'] = f"Non-missing: {df['outcome_leftright'].notna().sum():,}"

    # 2. Nativism (continuous, 1-3 scale, higher = more nativist)
    if 'pp15' in df.columns:
        df['outcome_nativism'] = df['pp15'].astype('float64')
        outcomes_created['nativism'] = f"Non-missing: {df['outcome_nativism'].notna().sum():,}"

    # 3. Social benefits preference (continuous, 1-3)
    if 'pp13' in df.columns:
        df['outcome_welfare'] = df['pp13'].astype('float64')
        outcomes_created['welfare'] = f"Non-missing: {df['outcome_welfare'].notna().sum():,}"

    # 4. Gender equality (continuous, 0-10, higher = pro-equality)
    if 'pp22' in df.columns:
        df['outcome_gender_equality'] = df['pp22'].astype('float64')
        outcomes_created['gender_equality'] = f"Non-missing: {df['outcome_gender_equality'].notna().sum():,}"

    # 5. Tax high incomes (continuous, 1-3, higher = more redistributive)
    if 'pp17' in df.columns:
        df['outcome_redistributive'] = df['pp17'].astype('float64')
        outcomes_created['redistributive'] = f"Non-missing: {df['outcome_redistributive'].notna().sum():,}"

    # 6. Job insecurity (continuous, 1-4, higher = less secure)
    if 'pw86' in df.columns:
        df['outcome_job_insecurity'] = df['pw86'].astype('float64')
        outcomes_created['job_insecurity'] = f"Non-missing: {df['outcome_job_insecurity'].notna().sum():,}"

    # 7. Work income (continuous, CHF, log-transformed)
    if 'iwyn' in df.columns:
        df['outcome_log_income'] = np.log(df['iwyn'] + 1)  # Add 1 to handle 0s
        outcomes_created['log_income'] = f"Non-missing: {(df['iwyn'] > 0).sum():,}"

    # 8. Hours worked per week (continuous, actual hours at main job)
    if 'pw77' in df.columns:
        pw77_numeric = pd.to_numeric(df['pw77'], errors='coerce')
        df['outcome_hours_worked'] = pw77_numeric.where(
            (pw77_numeric >= 1) & (pw77_numeric <= 99)
        )
        n_dropped = pw77_numeric.notna().sum() - df['outcome_hours_worked'].notna().sum()
        outcomes_created['hours_worked'] = (
            f"Non-missing: {df['outcome_hours_worked'].notna().sum():,} "
            f"(dropped {n_dropped:,} out-of-range)"
        )

    # 9. Vote choice (categorical, binary indicators for major parties)
    if 'pp19' in df.columns:
        # Create binary indicators for left (SP/GPS), center (CVP/FDP), right (SVP), other
        df['vote_sp_gps'] = ((df['pp19'].str.contains('SP|GPS', na=False)).astype(int)
                             if df['pp19'].dtype == 'object' else np.nan)
        df['vote_svp'] = ((df['pp19'].str.contains('SVP', na=False)).astype(int)
                          if df['pp19'].dtype == 'object' else np.nan)
        outcomes_created['vote_choice'] = f"Non-missing: {df['pp19'].notna().sum():,}"

    logger.info("Outcome variables prepared:")
    for outcome_name, summary in outcomes_created.items():
        logger.info(f"  - {outcome_name}: {summary}")

    return df

def construct_separation_outcome(df):
    """Forward-looking firm separation: separation_t1 = 1 if firm_id changes between t and t+1.
    Defined only when worker observed in t+1 with firm_id in both periods (right-censored otherwise)."""
    logger.info("Constructing forward-looking separation outcome...")

    df = df.sort_values(['idpers', 'year']).copy()
    df['firm_id_next'] = df.groupby('idpers')['firm_id'].shift(-1)
    df['year_next'] = df.groupby('idpers')['year'].shift(-1)

    consecutive = (df['year_next'] == df['year'] + 1)
    has_both_firms = df['firm_id'].notna() & df['firm_id_next'].notna()

    df['separation_t1'] = np.where(
        consecutive & has_both_firms,
        (df['firm_id'] != df['firm_id_next']).astype(float),
        np.nan
    )

    n_total = df['separation_t1'].notna().sum()
    n_separated = (df['separation_t1'] == 1).sum()
    sep_rate = 100 * n_separated / n_total if n_total > 0 else 0
    logger.info(f"Separation outcome:")
    logger.info(f"  - Person-years with defined separation: {n_total:,}")
    logger.info(f"  - Separations: {n_separated:,} ({sep_rate:.1f}%)")

    return df.drop(columns=['firm_id_next', 'year_next'])

def prepare_controls(df):
    """Create control variables for regression."""
    logger.info("Preparing control variables...")

    controls_created = {}

    # Age
    if 'age' in df.columns:
        df['age_centered'] = df['age'] - df['age'].mean()
        df['age_squared'] = df['age_centered'] ** 2
        controls_created['age'] = "Centered and squared"

    # Gender (binary)
    if 'sex' in df.columns:
        df['female'] = (df['sex'] == 2).astype(int)  # Assuming 2 = female
        controls_created['female'] = f"{df['female'].sum():,} women"

    # Education (ordinal, 1-8, higher = more educated)
    if 'educat' in df.columns:
        df['education'] = df['educat'].astype('float64')
        controls_created['education'] = f"Range: {df['education'].min()}-{df['education'].max()}"

    # Employment status (categorical)
    if 'wstat' in df.columns:
        df['employed'] = (df['wstat'] == 1).astype(int)  # 1 = employed
        controls_created['employed'] = f"{df['employed'].sum():,} employed"

    logger.info("Control variables prepared:")
    for ctrl_name, summary in controls_created.items():
        logger.info(f"  - {ctrl_name}: {summary}")

    return df

def select_analysis_sample(df):
    """Define and select the analysis sample for regression."""
    logger.info("Selecting analysis sample...")

    # Sample 1: All rows with firm_id (required for firm-year FE)
    df_with_firm = df[df['firm_id'].notna()].copy()
    logger.info(f"Rows with firm_id: {len(df_with_firm):,} ({100*len(df_with_firm)/len(df):.1f}%)")

    # Sample 2: All rows with firm_id AND occupation (required for all specs)
    df_with_both = df_with_firm[df_with_firm['isco08_4d'].notna()].copy()
    logger.info(f"Rows with firm_id AND occupation: {len(df_with_both):,} ({100*len(df_with_both)/len(df_with_firm):.1f}% of firm rows)")

    # Sample 3: Analysis sample = firm_id present (default for primary specs)
    # Outcome-specific samples will be defined at regression time
    df_analysis = df_with_firm.copy()

    return df_analysis

def validate_output(df):
    """Validate prepared data before saving."""
    logger.info("Validating prepared data...")

    # Check for required columns
    required_cols = ['idpers', 'year', 'firm_id', 'firm_year', 'occ_year']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns after preparation: {missing_cols}")

    # Check for data quality issues
    if df['idpers'].isna().any():
        raise ValueError("Person ID has missing values")
    if df['year'].isna().any():
        raise ValueError("Year has missing values")

    # Log sample composition
    logger.info("Data validation passed:")
    logger.info(f"  - Total rows: {len(df):,}")
    logger.info(f"  - Unique persons: {df['idpers'].nunique():,}")
    logger.info(f"  - Year range: {df['year'].min()}-{df['year'].max()}")
    logger.info(f"  - Firm-year cells: {df['firm_year'].nunique():,}")
    logger.info(f"  - Occupation-year cells: {df['occ_year'].nunique():,}")

    # Outcome coverage
    outcome_cols = [col for col in df.columns if col.startswith('outcome_')]
    logger.info(f"Outcome variables ({len(outcome_cols)} total):")
    for col in outcome_cols:
        n_nonmissing = df[col].notna().sum()
        pct = 100 * n_nonmissing / len(df)
        logger.info(f"  - {col}: {n_nonmissing:,} ({pct:.1f}%)")

def main():
    logger.info("=" * 80)
    logger.info("STAGE 8A: SHP PANEL DATA PREPARATION")
    logger.info("=" * 80)

    try:
        # Step 1: Validate input files
        logger.info("\nStep 1: Validating input files...")
        exposure_file = validate_paths()

        # Step 2: Load data
        logger.info("\nStep 2: Loading SHP exposure data...")
        df = load_shp_exposure(exposure_file)

        # Step 3: Create FE indicators
        logger.info("\nStep 3: Creating fixed effects indicators...")
        df = create_fixed_effect_indicators(df)

        # Step 4: Prepare outcomes
        logger.info("\nStep 4: Preparing outcome variables...")
        df = prepare_outcomes(df)

        # Step 5: Prepare controls
        logger.info("\nStep 5: Preparing control variables...")
        df = prepare_controls(df)

        # Step 6: Select analysis sample
        logger.info("\nStep 6: Selecting analysis sample...")
        df = select_analysis_sample(df)

        # Step 7: Validate
        logger.info("\nStep 7: Validating prepared data...")
        validate_output(df)

        # Step 8: Save
        logger.info("\nStep 8: Saving prepared dataset...")
        output_file = data_dir / "shp_panel_prepared.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"✓ Saved: {output_file}")
        logger.info(f"  - Size: {output_file.stat().st_size / (1024**2):.1f} MB")

        logger.info("\n" + "=" * 80)
        logger.info("✓ STAGE 8A COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nNext step: Run stage_8b_robustness_checks.py to fit Specs 1, 2, 3")

        return 0

    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
