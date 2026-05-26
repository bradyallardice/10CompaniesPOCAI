#!/usr/bin/env python3
"""
Stage 6 SHP: Link AI Exposure to Swiss Household Panel Respondents

This script links Stage 5 AI exposure measures (at firm×occupation×year level) to
individual respondents in the Swiss Household Panel (SHP), enabling analysis of how
individuals respond to firm-level AI adoption over time.

Produces 4 output files at different ISCO aggregation levels:
  1. shp_exposure_isco4d.csv — exact 4-digit ISCO match
  2. shp_exposure_isco3d.csv — 3-digit ISCO match (averaged from 4d exposures)
  3. shp_exposure_isco2d.csv — 2-digit ISCO match (averaged from 4d exposures)
  4. shp_exposure_isco4d_fallback.csv — hierarchical: try 4d → 3d → 2d

Usage:
    python3 stage_6_shp_exposure.py \\
        --shp-long-file ~/Dropbox/kurer_allardice_technology/data/original/shp/swissubase_932_11_0/data/Data_STATA/SHP-Data-Longfile-STATA/shplong_p_user.dta \\
        --shp-firmid-file ~/Dropbox/kurer_allardice_technology/data/created/shp_firmid_anon.csv \\
        --exposure-file Data/isco_firm_year_exposure_core_tasks_pct_05_ce_0.0_BGE.csv \\
        --output-dir Data/shp_exposure/ \\
        --generate-mapping-file
"""

import pandas as pd
import numpy as np
import os
import logging
import argparse
import gc
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Firm ID transformation: firm_id = (company_id + 13) * 13
FIRM_ID_OFFSET = 13
FIRM_ID_MULTIPLIER = 13

# SHP employment status codes
SHP_EMPLOYED = 1
SHP_NOT_EMPLOYED = 2
SHP_MISSING_CODES = {-3, -7, -8}

# SHP employer change codes (pw18): 2 = new employer, 3 = new employer (different coding)
SHP_EMPLOYER_CHANGED_CODES = {2, 3}

# Core exposure metrics — these get suffixed per exposure level
# Zero-fill is semantically correct for these (no AI exposure → 0 displacement)
EXPOSURE_CORE_COLUMNS = [
    'hampole_ai_exposure_avg',
    'binary_ai_exposure_avg',
    'hampole_occupation_exposure',
    'binary_occupation_exposure',
    # Expertise change: net change in importance-weighted avg expertise of
    # non-AI-exposed tasks vs all tasks. Populated only if Stage 5 was run
    # with --expertise-file. Zero = no AI exposure for this firm/occ/year.
    'expertise_change',
]

# Auxiliary exposure columns — only kept from Level 1 (foy), unsuffixed
# These are NOT zero-filled (zero is not a meaningful value for them)
EXPOSURE_AUX_COLUMNS = [
    'total_tasks_occupation',
    'total_importance_weight',
    'log_ai_intensity',
    'n_ai_apps_firm_year',
    'n_onet_codes_contributing',
    # Expertise levels: baseline & remaining (occupation-task properties).
    # Populated only if Stage 5 was run with --expertise-file.
    'baseline_expertise',
    'remaining_expertise',
]

# All exposure columns (backward compat)
EXPOSURE_VALUE_COLUMNS = EXPOSURE_CORE_COLUMNS + EXPOSURE_AUX_COLUMNS

# 6 exposure levels with their suffixes and merge key structure
EXPOSURE_LEVELS = {
    'foy': {  # Level 1: Firm × Occupation × Year (time-variant)
        'suffix': '_foy',
        'label': 'Firm×Occ×Year',
        'merge_keys_4d': ['firm_id', 'isco08_4d', 'year'],
        'merge_keys_3d': ['firm_id', 'isco_3d', 'year'],
        'merge_keys_2d': ['firm_id', 'isco_2d', 'year'],
        'has_firm': True,
        'has_isco': True,
        'has_year': True,
    },
    'fo': {  # Level 2: Firm × Occupation (time-invariant)
        'suffix': '_fo',
        'label': 'Firm×Occ (time-invariant)',
        'merge_keys_4d': ['firm_id', 'isco08_4d'],
        'merge_keys_3d': ['firm_id', 'isco_3d'],
        'merge_keys_2d': ['firm_id', 'isco_2d'],
        'has_firm': True,
        'has_isco': True,
        'has_year': False,
    },
    'oy': {  # Level 3: Occupation × Year
        'suffix': '_oy',
        'label': 'Occ×Year',
        'merge_keys_4d': ['isco08_4d', 'year'],
        'merge_keys_3d': ['isco_3d', 'year'],
        'merge_keys_2d': ['isco_2d', 'year'],
        'has_firm': False,
        'has_isco': True,
        'has_year': True,
    },
    'o': {  # Level 4: Occupation (time-invariant)
        'suffix': '_o',
        'label': 'Occ (time-invariant)',
        'merge_keys_4d': ['isco08_4d'],
        'merge_keys_3d': ['isco_3d'],
        'merge_keys_2d': ['isco_2d'],
        'has_firm': False,
        'has_isco': True,
        'has_year': False,
    },
    'fy': {  # Level 5: Firm × Year (derived from Level 1)
        'suffix': '_fy',
        'label': 'Firm×Year',
        'merge_keys_4d': ['firm_id', 'year'],
        'merge_keys_3d': ['firm_id', 'year'],
        'merge_keys_2d': ['firm_id', 'year'],
        'has_firm': True,
        'has_isco': False,
        'has_year': True,
    },
    'f': {  # Level 6: Firm (time-invariant, derived from Level 2)
        'suffix': '_f',
        'label': 'Firm (time-invariant)',
        'merge_keys_4d': ['firm_id'],
        'merge_keys_3d': ['firm_id'],
        'merge_keys_2d': ['firm_id'],
        'has_firm': True,
        'has_isco': False,
        'has_year': False,
    },
}

# SHP columns needed for processing (beyond the 964 survey variables).
# is4maj was present in SHP v10 longfile (swissubase_932_10_0) but dropped in v11
# (swissubase_932_11_0, released Feb 2026). 4-digit ISCO matching is now best-effort:
# if is4maj is absent we proceed with is3maj as the finest granularity.
SHP_KEY_COLUMNS = ['idpers', 'year', 'is3maj', 'is2maj', 'is1maj',
                   'pw01', 'pw02', 'pw03', 'pw18']


# =============================================================================
# Firm ID Mapping
# =============================================================================

def company_id_to_firm_id(company_id: int) -> int:
    """Convert company_id to anonymized firm_id."""
    return (company_id + FIRM_ID_OFFSET) * FIRM_ID_MULTIPLIER


def firm_id_to_company_id(firm_id: int) -> int:
    """Reverse the firm_id transformation to recover company_id."""
    return (firm_id // FIRM_ID_MULTIPLIER) - FIRM_ID_OFFSET


def generate_mapping_file(exposure_file: str, output_path: str) -> pd.DataFrame:
    """
    Generate company_id ↔ firm_id mapping from Stage 5 exposure data.

    Args:
        exposure_file: Path to Stage 5 ISCO exposure CSV
        output_path: Where to save the mapping CSV

    Returns:
        DataFrame with company_id and firm_id columns
    """
    logger.info(f"Generating company_id → firm_id mapping from {exposure_file}")

    df = pd.read_csv(exposure_file, usecols=['company_id'])
    unique_ids = df['company_id'].dropna().unique()

    mapping = pd.DataFrame({
        'company_id': unique_ids,
        'firm_id': [company_id_to_firm_id(cid) for cid in unique_ids]
    })

    # Validate bijectivity
    if mapping['firm_id'].nunique() != len(mapping):
        raise ValueError("firm_id transformation is not bijective — duplicate firm_ids detected")

    # Validate round-trip
    recovered = mapping['firm_id'].apply(firm_id_to_company_id)
    if not (recovered == mapping['company_id']).all():
        raise ValueError("firm_id round-trip validation failed")

    mapping.to_csv(output_path, index=False)
    logger.info(f"Saved mapping file: {len(mapping):,} unique companies → {output_path}")

    return mapping


# =============================================================================
# SHP Data Loading and Preparation
# =============================================================================

def load_shp_long(shp_long_file: str) -> pd.DataFrame:
    """
    Load the SHP person-level long file from STATA format.

    Loads all columns (964 variables) to preserve the full survey dataset.

    Args:
        shp_long_file: Path to shplong_p_user.dta

    Returns:
        DataFrame with all SHP variables
    """
    if not os.path.exists(shp_long_file):
        raise FileNotFoundError(f"SHP long file not found: {shp_long_file}")

    logger.info(f"Loading SHP long file: {shp_long_file}")
    logger.info("  This may take a moment (964 variables)...")

    # convert_categoricals=False avoids ValueError from non-unique STATA value labels
    # (e.g., column pp02 has duplicate category labels)
    shp = pd.read_stata(shp_long_file, convert_categoricals=False)

    logger.info(f"  Loaded: {len(shp):,} person-year observations, {len(shp.columns)} variables")
    logger.info(f"  Year range: {shp['year'].min()} - {shp['year'].max()}")
    logger.info(f"  Unique persons: {shp['idpers'].nunique():,}")

    # Validate key columns exist
    missing_cols = [c for c in SHP_KEY_COLUMNS if c not in shp.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in SHP file: {missing_cols}\n"
            f"Available columns (first 20): {list(shp.columns[:20])}"
        )

    return shp


def load_shp_firmid(shp_firmid_file: str) -> pd.DataFrame:
    """
    Load the anonymized firm ID file (idpers, year, firm_id).

    Args:
        shp_firmid_file: Path to shp_firmid_anon.csv

    Returns:
        DataFrame with idpers, year, firm_id
    """
    if not os.path.exists(shp_firmid_file):
        raise FileNotFoundError(f"SHP firm ID file not found: {shp_firmid_file}")

    firmid = pd.read_csv(shp_firmid_file)

    expected_cols = {'idpers', 'year', 'firm_id'}
    if not expected_cols.issubset(set(firmid.columns)):
        raise ValueError(
            f"Expected columns {expected_cols} in firm ID file.\n"
            f"Found: {list(firmid.columns)}"
        )

    n_with_firmid = firmid['firm_id'].notna().sum()
    n_total = len(firmid)
    logger.info(f"Loaded firm ID file: {n_total:,} rows, {n_with_firmid:,} with firm_id ({100*n_with_firmid/n_total:.1f}%)")

    return firmid[['idpers', 'year', 'firm_id']]


def fill_down_firm_id(shp: pd.DataFrame) -> pd.DataFrame:
    """
    Fill forward firm_id within person panels, respecting employment changes.

    Replicates the R logic from 3_shp_long.R:
    - firm_id is only available through wave 23 (year 2021)
    - Fill forward if respondent has not changed employer
    - Reset on: new firm_id observed, employer change (pw18=2 or 3), explicit non-employment
    - Allow filling through data gaps if employment resumes without employer change

    Args:
        shp: DataFrame with firm_id, pw01, pw02, pw03, pw18 columns

    Returns:
        DataFrame with new firm_id_filled column
    """
    logger.info("Filling down firm_id based on employment change logic...")

    n_before = shp['firm_id'].notna().sum()

    shp = shp.sort_values(['idpers', 'year']).copy()

    # Identify conditions (matching R logic exactly)
    shp['_firm_id_known'] = shp['firm_id'].notna()

    # is_employed: any of pw01, pw02, pw03 == 1
    shp['_is_employed'] = (
        (shp['pw01'] == SHP_EMPLOYED) |
        (shp['pw02'] == SHP_EMPLOYED) |
        (shp['pw03'] == SHP_EMPLOYED)
    )

    # employment_missing: all of pw01, pw02, pw03 in missing codes (-3, -7, -8)
    shp['_employment_missing'] = (
        shp['pw01'].isin(SHP_MISSING_CODES) &
        shp['pw02'].isin(SHP_MISSING_CODES) &
        shp['pw03'].isin(SHP_MISSING_CODES)
    )

    # explicitly_not_employed: all of pw01, pw02, pw03 == 2
    shp['_explicitly_not_employed'] = (
        (shp['pw01'] == SHP_NOT_EMPLOYED) &
        (shp['pw02'] == SHP_NOT_EMPLOYED) &
        (shp['pw03'] == SHP_NOT_EMPLOYED)
    )

    # employer_changed: pw18 in (2, 3)
    shp['_employer_changed'] = shp['pw18'].isin(SHP_EMPLOYER_CHANGED_CODES)

    # Create fill blocks: reset when we have new firm_id, employer change,
    # or explicit non-employment (but NOT if just missing data)
    shp['_block_reset'] = (
        shp['_firm_id_known'] |
        shp['_employer_changed'] |
        (shp['_explicitly_not_employed'] & ~shp['_employment_missing'])
    )

    # Cumulative sum within each person to create fill blocks
    shp['_fill_block'] = shp.groupby('idpers')['_block_reset'].cumsum()

    # Set firm_id_filled: use known firm_id, or NA if explicitly not employed
    shp['firm_id_filled'] = np.where(
        shp['_firm_id_known'],
        shp['firm_id'],
        np.where(
            shp['_explicitly_not_employed'],
            np.nan,
            np.nan  # will be filled forward
        )
    )

    # Fill forward within (idpers, fill_block) groups
    shp['firm_id_filled'] = shp.groupby(['idpers', '_fill_block'])['firm_id_filled'].ffill()

    # Clean up temporary columns
    temp_cols = [c for c in shp.columns if c.startswith('_')]
    shp = shp.drop(columns=temp_cols)

    n_after = shp['firm_id_filled'].notna().sum()
    logger.info(f"  firm_id before fill: {n_before:,}")
    logger.info(f"  firm_id after fill:  {n_after:,} (+{n_after - n_before:,})")

    return shp


def prepare_shp_data(shp: pd.DataFrame, year_min: int = 2012, year_max: int = 2023) -> pd.DataFrame:
    """
    Final preparation of SHP data for merging.

    Args:
        shp: DataFrame after fill-down
        year_min: Minimum year to include (default 2012, when firm_id starts)
        year_max: Maximum year to include

    Returns:
        Prepared DataFrame with isco08_4d, isco_3d, isco_2d columns
    """
    logger.info(f"Preparing SHP data for merge (years {year_min}-{year_max})...")

    # Filter years
    shp = shp[(shp['year'] >= year_min) & (shp['year'] <= year_max)].copy()
    logger.info(f"  After year filter: {len(shp):,} rows")

    # Rename: use filled firm_id as primary, keep original as firm_id_2021
    shp = shp.rename(columns={'firm_id': 'firm_id_2021', 'firm_id_filled': 'firm_id'})

    # Extract ISCO codes (negative values → NaN, matching R: shp$is4maj[shp$is4maj<0] <- NA)
    # is4maj absent in SHP v11 longfile — set isco08_4d to NaN so 4d-merged outputs stay
    # schema-compatible but contain no matches; analysis effectively falls back to 3d.
    if 'is4maj' in shp.columns:
        shp['isco08_4d'] = shp['is4maj'].where(shp['is4maj'] >= 0, other=np.nan)
    else:
        logger.warning("  is4maj not in SHP longfile (v11+) — 4-digit ISCO matching disabled, using 3-digit as max granularity")
        shp['isco08_4d'] = np.nan
    shp['isco_3d'] = shp['is3maj'].where(shp['is3maj'] >= 0, other=np.nan)
    shp['isco_2d'] = shp['is2maj'].where(shp['is2maj'] >= 0, other=np.nan)
    shp['isco_1d'] = shp['is1maj'].where(shp['is1maj'] >= 0, other=np.nan)

    # Convert ISCO codes to integers where non-null (they come as float from STATA)
    for col in ['isco08_4d', 'isco_3d', 'isco_2d', 'isco_1d']:
        shp[col] = shp[col].astype('Int64')  # nullable integer

    n_with_firm = shp['firm_id'].notna().sum()
    n_with_isco = shp['isco08_4d'].notna().sum()
    n_with_both = (shp['firm_id'].notna() & shp['isco08_4d'].notna()).sum()

    logger.info(f"  Rows with firm_id: {n_with_firm:,}")
    logger.info(f"  Rows with isco08_4d: {n_with_isco:,}")
    logger.info(f"  Rows with both: {n_with_both:,}")
    logger.info(f"  Unique persons: {shp['idpers'].nunique():,}")

    return shp


# =============================================================================
# Exposure Data Preparation
# =============================================================================

def load_exposure_data(exposure_file: str, mapping_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load Stage 5 ISCO exposure data and add firm_id column.

    Args:
        exposure_file: Path to ISCO exposure CSV
        mapping_file: Optional path to company_id → firm_id mapping CSV.
                      If None, computes firm_id on the fly.

    Returns:
        DataFrame with firm_id added alongside company_id
    """
    if not os.path.exists(exposure_file):
        raise FileNotFoundError(f"Exposure file not found: {exposure_file}")

    logger.info(f"Loading exposure data: {exposure_file}")
    exposure = pd.read_csv(exposure_file)
    logger.info(f"  Loaded: {len(exposure):,} rows, {exposure['company_id'].nunique():,} companies, "
                f"{exposure['isco08_4d'].nunique():,} occupations")

    # Detect which exposure columns are present
    present_exp_cols = [c for c in EXPOSURE_VALUE_COLUMNS if c in exposure.columns]
    missing_exp_cols = [c for c in EXPOSURE_VALUE_COLUMNS if c not in exposure.columns]
    if missing_exp_cols:
        logger.warning(f"  Missing exposure columns (will be skipped): {missing_exp_cols}")

    # Add firm_id
    if mapping_file and os.path.exists(mapping_file):
        logger.info(f"  Loading firm_id mapping from {mapping_file}")
        mapping = pd.read_csv(mapping_file)
        exposure = exposure.merge(mapping, on='company_id', how='left')
        unmapped = exposure['firm_id'].isna().sum()
        if unmapped > 0:
            logger.warning(f"  {unmapped:,} exposure rows have no firm_id mapping")
    else:
        logger.info("  Computing firm_id from company_id")
        exposure['firm_id'] = exposure['company_id'].apply(company_id_to_firm_id)

    # Ensure isco08_4d is Int64 (to match SHP)
    exposure['isco08_4d'] = exposure['isco08_4d'].astype('Int64')

    logger.info(f"  Unique firm_ids: {exposure['firm_id'].nunique():,}")
    logger.info(f"  Year range: {exposure['year'].min()} - {exposure['year'].max()}")

    return exposure


def load_exposure_generic(exposure_file: str, level_name: str,
                          mapping_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load any Stage 5 exposure file and add firm_id if company_id is present.

    Handles both time-variant (has 'year') and time-invariant files.
    For occupation-only files (still have company_id from Stage 5), collapses
    to unique occupation(-year) rows by averaging across firms.

    Args:
        exposure_file: Path to Stage 5 ISCO exposure CSV
        level_name: One of EXPOSURE_LEVELS keys (e.g., 'fo', 'oy', 'o')
        mapping_file: Optional path to company_id → firm_id mapping CSV

    Returns:
        DataFrame ready for merging (with firm_id if firm-based level,
        collapsed to occ-level if occ-only level)
    """
    if not os.path.exists(exposure_file):
        raise FileNotFoundError(f"Exposure file not found: {exposure_file}")

    level_config = EXPOSURE_LEVELS[level_name]
    logger.info(f"Loading {level_config['label']} exposure: {exposure_file}")

    exposure = pd.read_csv(exposure_file)
    logger.info(f"  Loaded: {len(exposure):,} rows")

    # Detect which core columns are present
    core_cols = [c for c in EXPOSURE_CORE_COLUMNS if c in exposure.columns]
    if not core_cols:
        raise ValueError(
            f"No core exposure columns found in {exposure_file}.\n"
            f"Expected any of: {EXPOSURE_CORE_COLUMNS}\n"
            f"Found: {list(exposure.columns)}"
        )

    # Ensure isco08_4d is Int64 if present
    if 'isco08_4d' in exposure.columns:
        exposure['isco08_4d'] = exposure['isco08_4d'].astype('Int64')

    if level_config['has_firm']:
        # Add firm_id from company_id
        if 'company_id' in exposure.columns:
            if mapping_file and os.path.exists(mapping_file):
                mapping = pd.read_csv(mapping_file)
                exposure = exposure.merge(mapping, on='company_id', how='left')
            else:
                exposure['firm_id'] = exposure['company_id'].apply(company_id_to_firm_id)
            logger.info(f"  Unique firms: {exposure['firm_id'].nunique():,}")
        else:
            raise ValueError(f"Firm-based level '{level_name}' requires company_id column")
    else:
        # Occupation-only level: collapse across firms
        # occupation_exposure is constant across firms; ai_exposure_avg varies by firm intensity
        group_cols = []
        if 'isco08_4d' in exposure.columns:
            group_cols.append('isco08_4d')
        if level_config['has_year'] and 'year' in exposure.columns:
            group_cols.append('year')

        if not group_cols:
            raise ValueError(f"Cannot determine group columns for level '{level_name}'")

        n_before = len(exposure)
        exposure = exposure.groupby(group_cols, as_index=False)[core_cols].mean()
        logger.info(f"  Collapsed {n_before:,} firm-rows → {len(exposure):,} occ-rows (averaged across firms)")

    if level_config['has_year'] and 'year' in exposure.columns:
        logger.info(f"  Year range: {exposure['year'].min()} - {exposure['year'].max()}")
    elif not level_config['has_year']:
        logger.info(f"  Time-invariant (no year dimension)")

    return exposure


def derive_firm_level_exposure(exposure_firm_occ: pd.DataFrame,
                                time_variant: bool = True) -> pd.DataFrame:
    """
    Derive firm-level exposure by averaging across occupations within each firm.

    Args:
        exposure_firm_occ: Firm × occupation (× year) exposure DataFrame
        time_variant: If True, group by (firm_id, year); if False, group by (firm_id)

    Returns:
        Firm-level exposure DataFrame
    """
    core_cols = [c for c in EXPOSURE_CORE_COLUMNS if c in exposure_firm_occ.columns]

    if time_variant:
        group_cols = ['firm_id', 'year']
        label = "firm×year"
    else:
        group_cols = ['firm_id']
        label = "firm (time-invariant)"

    # Only use group cols that exist
    group_cols = [c for c in group_cols if c in exposure_firm_occ.columns]

    n_before = len(exposure_firm_occ)
    result = exposure_firm_occ.groupby(group_cols, as_index=False)[core_cols].mean()
    logger.info(f"  Derived {label} exposure: {n_before:,} firm×occ rows → {len(result):,} firm rows")

    return result


def aggregate_exposure_generic(exposure_4d: pd.DataFrame,
                                target_digit: int,
                                group_cols_base: List[str],
                                employment_weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate 4-digit ISCO exposure to N-digit level with flexible group keys.

    Args:
        exposure_4d: 4-digit exposure DataFrame
        target_digit: Target ISCO digits (2 or 3)
        group_cols_base: Base group columns (without ISCO), e.g., ['firm_id', 'year'] or ['year']
        employment_weights: Optional employment weights

    Returns:
        DataFrame with exposure at target ISCO level
    """
    if target_digit == 3:
        isco_col = 'isco_3d'
        divisor = 10
    elif target_digit == 2:
        isco_col = 'isco_2d'
        divisor = 100
    else:
        raise ValueError(f"target_digit must be 2 or 3, got {target_digit}")

    df = exposure_4d.copy()
    df[isco_col] = (df['isco08_4d'] // divisor).astype('Int64')

    core_cols = [c for c in EXPOSURE_CORE_COLUMNS if c in df.columns]
    group_cols = group_cols_base + [isco_col]

    if employment_weights is not None:
        employment_weights = employment_weights.copy()
        employment_weights['isco08_4d'] = employment_weights['isco08_4d'].astype('Int64')
        df = df.merge(employment_weights[['isco08_4d', 'weight']], on='isco08_4d', how='left')
        df['weight'] = df['weight'].fillna(1.0)

        for col in core_cols:
            df[f'_w_{col}'] = df[col] * df['weight']

        agg_funcs = {f'_w_{col}': 'sum' for col in core_cols}
        agg_funcs['weight'] = 'sum'
        result = df.groupby(group_cols, as_index=False).agg(agg_funcs)

        for col in core_cols:
            result[col] = result[f'_w_{col}'] / result['weight']
            result = result.drop(columns=[f'_w_{col}'])
        result = result.drop(columns=['weight'])
    else:
        result = df.groupby(group_cols, as_index=False)[core_cols].mean()

    return result


def aggregate_exposure_to_3d(exposure_4d: pd.DataFrame,
                             employment_weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate 4-digit ISCO exposure to 3-digit level.

    Args:
        exposure_4d: 4-digit exposure DataFrame with firm_id, isco08_4d, year
        employment_weights: Optional DataFrame with isco08_4d and weight columns
                           for employment-weighted averaging. If None, uses simple mean.

    Returns:
        DataFrame with exposure at 3-digit ISCO level
    """
    logger.info("Aggregating exposure: 4-digit → 3-digit ISCO...")

    df = exposure_4d.copy()
    # Derive 3-digit code from 4-digit
    df['isco_3d'] = (df['isco08_4d'] // 10).astype('Int64')

    # Identify exposure value columns present in data
    exp_cols = [c for c in EXPOSURE_VALUE_COLUMNS if c in df.columns]

    group_cols = ['firm_id', 'year', 'isco_3d']

    if employment_weights is not None:
        logger.info("  Using employment-weighted averaging")
        # Merge weights
        employment_weights = employment_weights.copy()
        employment_weights['isco08_4d'] = employment_weights['isco08_4d'].astype('Int64')
        df = df.merge(employment_weights[['isco08_4d', 'weight']], on='isco08_4d', how='left')
        df['weight'] = df['weight'].fillna(1.0)  # default weight for unmatched

        # Weighted average
        agg_dict = {}
        for col in exp_cols:
            df[f'_weighted_{col}'] = df[col] * df['weight']

        agg_funcs = {f'_weighted_{col}': 'sum' for col in exp_cols}
        agg_funcs['weight'] = 'sum'

        result = df.groupby(group_cols, as_index=False).agg(agg_funcs)

        for col in exp_cols:
            result[col] = result[f'_weighted_{col}'] / result['weight']
            result = result.drop(columns=[f'_weighted_{col}'])
        result = result.drop(columns=['weight'])
    else:
        logger.info("  Using simple mean (no employment weights provided)")
        result = df.groupby(group_cols, as_index=False)[exp_cols].mean()

    logger.info(f"  Aggregated: {len(exposure_4d):,} rows (4d) → {len(result):,} rows (3d)")
    logger.info(f"  Unique 3-digit codes: {result['isco_3d'].nunique():,}")

    return result


def aggregate_exposure_to_2d(exposure_4d: pd.DataFrame,
                             employment_weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate 4-digit ISCO exposure to 2-digit level.

    Args:
        exposure_4d: 4-digit exposure DataFrame with firm_id, isco08_4d, year
        employment_weights: Optional DataFrame with isco08_4d and weight columns

    Returns:
        DataFrame with exposure at 2-digit ISCO level
    """
    logger.info("Aggregating exposure: 4-digit → 2-digit ISCO...")

    df = exposure_4d.copy()
    df['isco_2d'] = (df['isco08_4d'] // 100).astype('Int64')

    exp_cols = [c for c in EXPOSURE_VALUE_COLUMNS if c in df.columns]
    group_cols = ['firm_id', 'year', 'isco_2d']

    if employment_weights is not None:
        logger.info("  Using employment-weighted averaging")
        employment_weights = employment_weights.copy()
        employment_weights['isco08_4d'] = employment_weights['isco08_4d'].astype('Int64')
        df = df.merge(employment_weights[['isco08_4d', 'weight']], on='isco08_4d', how='left')
        df['weight'] = df['weight'].fillna(1.0)

        for col in exp_cols:
            df[f'_weighted_{col}'] = df[col] * df['weight']

        agg_funcs = {f'_weighted_{col}': 'sum' for col in exp_cols}
        agg_funcs['weight'] = 'sum'

        result = df.groupby(group_cols, as_index=False).agg(agg_funcs)

        for col in exp_cols:
            result[col] = result[f'_weighted_{col}'] / result['weight']
            result = result.drop(columns=[f'_weighted_{col}'])
        result = result.drop(columns=['weight'])
    else:
        logger.info("  Using simple mean (no employment weights provided)")
        result = df.groupby(group_cols, as_index=False)[exp_cols].mean()

    logger.info(f"  Aggregated: {len(exposure_4d):,} rows (4d) → {len(result):,} rows (2d)")
    logger.info(f"  Unique 2-digit codes: {result['isco_2d'].nunique():,}")

    return result


# =============================================================================
# Merge Logic
# =============================================================================

def merge_shp_exposure_4d(shp: pd.DataFrame, exposure_4d: pd.DataFrame) -> pd.DataFrame:
    """
    Merge SHP with exposure at exact 4-digit ISCO level.

    Join key: (firm_id, isco08_4d, year)

    Args:
        shp: Prepared SHP DataFrame
        exposure_4d: 4-digit exposure DataFrame with firm_id

    Returns:
        Merged DataFrame (all SHP rows preserved)
    """
    logger.info("Merging SHP × Exposure at 4-digit ISCO level...")

    exp_cols = [c for c in EXPOSURE_VALUE_COLUMNS if c in exposure_4d.columns]
    merge_cols = ['firm_id', 'isco08_4d', 'year']

    # Select only needed columns from exposure for merge
    exposure_subset = exposure_4d[merge_cols + exp_cols + ['company_id', 'isco08_title', 'company_name']].copy()
    # Deduplicate exposure on merge keys (should already be unique, but safety check)
    n_before_dedup = len(exposure_subset)
    exposure_subset = exposure_subset.drop_duplicates(subset=merge_cols)
    if len(exposure_subset) < n_before_dedup:
        logger.warning(f"  Deduplicated exposure: {n_before_dedup:,} → {len(exposure_subset):,}")

    result = shp.merge(exposure_subset, on=merge_cols, how='left')

    _log_merge_coverage(result, exp_cols, "4-digit ISCO")

    return result


def merge_shp_exposure_3d(shp: pd.DataFrame, exposure_3d: pd.DataFrame) -> pd.DataFrame:
    """Merge SHP with exposure at 3-digit ISCO level."""
    logger.info("Merging SHP × Exposure at 3-digit ISCO level...")

    exp_cols = [c for c in EXPOSURE_VALUE_COLUMNS if c in exposure_3d.columns]
    merge_cols = ['firm_id', 'isco_3d', 'year']

    exposure_subset = exposure_3d[merge_cols + exp_cols].drop_duplicates(subset=merge_cols)
    result = shp.merge(exposure_subset, on=merge_cols, how='left')

    _log_merge_coverage(result, exp_cols, "3-digit ISCO")

    return result


def merge_shp_exposure_2d(shp: pd.DataFrame, exposure_2d: pd.DataFrame) -> pd.DataFrame:
    """Merge SHP with exposure at 2-digit ISCO level."""
    logger.info("Merging SHP × Exposure at 2-digit ISCO level...")

    exp_cols = [c for c in EXPOSURE_VALUE_COLUMNS if c in exposure_2d.columns]
    merge_cols = ['firm_id', 'isco_2d', 'year']

    exposure_subset = exposure_2d[merge_cols + exp_cols].drop_duplicates(subset=merge_cols)
    result = shp.merge(exposure_subset, on=merge_cols, how='left')

    _log_merge_coverage(result, exp_cols, "2-digit ISCO")

    return result


def merge_shp_exposure_fallback(shp: pd.DataFrame,
                                 exposure_4d: pd.DataFrame,
                                 exposure_3d: pd.DataFrame,
                                 exposure_2d: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical fallback merge: try 4d → 3d → 2d.

    Adds a 'match_level' column indicating which level matched.
    """
    logger.info("Merging SHP × Exposure with hierarchical fallback (4d → 3d → 2d)...")

    # Use only columns that exist in ALL three DataFrames (core cols for 3d/2d)
    core_cols = [c for c in EXPOSURE_CORE_COLUMNS if c in exposure_4d.columns]
    # 4d may have additional aux/extra columns
    aux_cols_4d = [c for c in EXPOSURE_AUX_COLUMNS if c in exposure_4d.columns]
    extra_cols_4d = [c for c in ['company_id', 'isco08_title', 'company_name'] if c in exposure_4d.columns]
    all_cols_4d = core_cols + aux_cols_4d + extra_cols_4d

    # Strategy: do three separate left-joins (4d, 3d, 2d), each with suffixed columns,
    # then pick the best match level per row.

    result = shp.copy()
    result = result.reset_index(drop=True)

    # 4-digit lookup (includes aux + extra cols)
    merge_keys_4d = ['firm_id', 'isco08_4d', 'year']
    exp_4d_sub = exposure_4d[merge_keys_4d + all_cols_4d].drop_duplicates(subset=merge_keys_4d)
    exp_4d_renamed = exp_4d_sub.rename(columns={c: f'{c}_4d' for c in all_cols_4d})
    result = result.merge(exp_4d_renamed, on=merge_keys_4d, how='left')

    # 3-digit lookup (core cols only — 3d df doesn't have aux cols)
    merge_keys_3d = ['firm_id', 'isco_3d', 'year']
    cols_3d = [c for c in core_cols if c in exposure_3d.columns]
    exp_3d_sub = exposure_3d[merge_keys_3d + cols_3d].drop_duplicates(subset=merge_keys_3d)
    exp_3d_renamed = exp_3d_sub.rename(columns={c: f'{c}_3d' for c in cols_3d})
    result = result.merge(exp_3d_renamed, on=merge_keys_3d, how='left')

    # 2-digit lookup (core cols only)
    merge_keys_2d = ['firm_id', 'isco_2d', 'year']
    cols_2d = [c for c in core_cols if c in exposure_2d.columns]
    exp_2d_sub = exposure_2d[merge_keys_2d + cols_2d].drop_duplicates(subset=merge_keys_2d)
    exp_2d_renamed = exp_2d_sub.rename(columns={c: f'{c}_2d' for c in cols_2d})
    result = result.merge(exp_2d_renamed, on=merge_keys_2d, how='left')

    # Pick best match: prefer 4d > 3d > 2d
    primary_4d = f'{core_cols[0]}_4d'
    primary_3d = f'{core_cols[0]}_3d'
    primary_2d = f'{core_cols[0]}_2d'

    has_4d = result[primary_4d].notna()
    has_3d = result[primary_3d].notna() & ~has_4d
    has_2d = result[primary_2d].notna() & ~has_4d & ~has_3d

    result['match_level'] = 'none'
    result.loc[has_4d, 'match_level'] = '4d'
    result.loc[has_3d, 'match_level'] = '3d'
    result.loc[has_2d, 'match_level'] = '2d'

    # Coalesce core exposure values: take from best match level
    for col in core_cols:
        result[col] = np.where(has_4d, result[f'{col}_4d'],
                      np.where(has_3d, result[f'{col}_3d'],
                      np.where(has_2d, result[f'{col}_2d'], np.nan)))

    # Aux + extra columns from 4d only
    for col in aux_cols_4d + extra_cols_4d:
        result[col] = np.where(has_4d, result[f'{col}_4d'], np.nan)

    # Drop temporary suffixed columns
    drop_cols = ([f'{c}_4d' for c in all_cols_4d] +
                 [f'{c}_3d' for c in cols_3d] +
                 [f'{c}_2d' for c in cols_2d])
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])

    n_4d = has_4d.sum()
    n_3d = has_3d.sum()
    n_2d = has_2d.sum()
    logger.info(f"  4-digit matches: {n_4d:,}")
    logger.info(f"  3-digit matches (fallback): {n_3d:,}")
    logger.info(f"  2-digit matches (fallback): {n_2d:,}")

    _log_merge_coverage(result, core_cols, "hierarchical fallback")

    # Log match level distribution
    match_counts = result['match_level'].value_counts()
    logger.info("  Match level distribution:")
    for level, count in match_counts.items():
        logger.info(f"    {level}: {count:,} ({100*count/len(result):.1f}%)")

    return result


def merge_exposure_level(shp: pd.DataFrame, exposure: pd.DataFrame,
                          level_name: str, isco_mode: str = '4d') -> pd.DataFrame:
    """
    Merge one exposure level onto SHP with suffixed column names.

    Args:
        shp: SHP DataFrame (all rows preserved via left join)
        exposure: Prepared exposure DataFrame for this level
        level_name: Key into EXPOSURE_LEVELS (e.g., 'foy', 'oy', 'f')
        isco_mode: Which ISCO granularity ('4d', '3d', '2d')

    Returns:
        SHP DataFrame with new suffixed exposure columns added
    """
    level_config = EXPOSURE_LEVELS[level_name]
    suffix = level_config['suffix']
    merge_keys = level_config[f'merge_keys_{isco_mode}']

    core_cols = [c for c in EXPOSURE_CORE_COLUMNS if c in exposure.columns]
    if not core_cols:
        logger.warning(f"  No core exposure columns in {level_name} data, skipping merge")
        return shp

    # Build subset for merge: only merge keys + core columns
    available_keys = [k for k in merge_keys if k in exposure.columns]
    if set(available_keys) != set(merge_keys):
        missing = set(merge_keys) - set(available_keys)
        logger.warning(f"  Missing merge keys for {level_name}: {missing}, skipping")
        return shp

    # Deduplicate exposure on merge keys
    exp_subset = exposure[available_keys + core_cols].drop_duplicates(subset=available_keys)

    # Rename core columns with suffix BEFORE merge to avoid collisions
    rename_dict = {col: f'{col}{suffix}' for col in core_cols}
    exp_subset = exp_subset.rename(columns=rename_dict)

    result = shp.merge(exp_subset, on=available_keys, how='left')

    # Log coverage
    primary_suffixed = f'{core_cols[0]}{suffix}'
    n_with = result[primary_suffixed].notna().sum()
    n_total = len(result)
    logger.info(f"  {level_config['label']} ({isco_mode}): {n_with:,}/{n_total:,} matched ({100*n_with/n_total:.2f}%)")

    return result


def fill_zeros_for_level(result: pd.DataFrame, level_name: str,
                          observable_firm_ids: set) -> pd.DataFrame:
    """
    Zero-fill for one exposure level based on its type.

    - Firm-based levels: zero-fill if firm is in X28 database
    - Occ-only levels: zero-fill if person has an ISCO code (all occupations observable)
    - Firm-only levels: zero-fill if firm is in X28 database (regardless of ISCO)

    Args:
        result: Merged SHP DataFrame
        level_name: Key into EXPOSURE_LEVELS
        observable_firm_ids: Set of firm_ids from X28 database
    """
    level_config = EXPOSURE_LEVELS[level_name]
    suffix = level_config['suffix']
    core_cols_suffixed = [f'{c}{suffix}' for c in EXPOSURE_CORE_COLUMNS
                          if f'{c}{suffix}' in result.columns]

    if not core_cols_suffixed:
        return result

    primary_col = core_cols_suffixed[0]
    no_exposure = result[primary_col].isna()

    if level_config['has_firm'] and level_config['has_isco']:
        # Firm × Occ levels: need firm in X28 AND has ISCO
        fill_mask = (result['firm_in_exposure_data'] &
                     result['isco08_4d'].notna() &
                     no_exposure)
    elif level_config['has_firm'] and not level_config['has_isco']:
        # Firm-only levels: need firm in X28 (no ISCO requirement)
        fill_mask = result['firm_in_exposure_data'] & no_exposure
    elif not level_config['has_firm'] and level_config['has_isco']:
        # Occ-only levels: all occupations are observable, so anyone with ISCO gets 0
        fill_mask = result['isco08_4d'].notna() & no_exposure
    else:
        # Pure time-level (shouldn't happen)
        fill_mask = pd.Series(False, index=result.index)

    n_filled = fill_mask.sum()
    for col in core_cols_suffixed:
        result.loc[fill_mask, col] = 0.0

    logger.info(f"  Zero-fill {level_config['label']}: {n_filled:,} rows filled with 0")

    return result


def _log_merge_coverage(result: pd.DataFrame, exp_cols: List[str], label: str):
    """Log merge coverage statistics."""
    primary_col = 'hampole_ai_exposure_avg' if 'hampole_ai_exposure_avg' in exp_cols else exp_cols[0]

    n_total = len(result)
    n_with_exposure = result[primary_col].notna().sum()
    n_with_firm = result['firm_id'].notna().sum()

    logger.info(f"  {label} merge results:")
    logger.info(f"    Total person-year rows: {n_total:,}")
    logger.info(f"    With firm_id: {n_with_firm:,}")
    logger.info(f"    With exposure: {n_with_exposure:,} ({100*n_with_exposure/n_total:.2f}%)")

    if n_with_exposure > 0:
        logger.info(f"    Unique persons with exposure: {result.loc[result[primary_col].notna(), 'idpers'].nunique():,}")
        logger.info(f"    Unique firms with exposure: {result.loc[result[primary_col].notna(), 'firm_id'].nunique():,}")


# =============================================================================
# Zero-Fill Logic
# =============================================================================

def fill_zeros_for_matched_firms(result: pd.DataFrame, exp_cols: List[str],
                                 observable_firm_ids: set) -> pd.DataFrame:
    """
    Fill exposure with 0 for respondent-year observations where the firm IS in
    the X28 job ad database but has no AI exposure for this occupation/year.

    Three categories of missing exposure (all have firm_id + ISCO but no merge match):
      1. Firm IS in X28 database → fill with 0 (true zero: firm observable, no AI for this occ/year)
      2. Firm NOT in X28 database → leave NaN (unknown: firm not in our job ad database)
      3. No firm_id or no ISCO code → leave NaN (truly missing)

    A 'firm_in_exposure_data' boolean column is added so downstream analysis can
    distinguish observable firms (true zeros + treated) from unobservable firms.

    Args:
        result: Merged SHP × exposure DataFrame
        exp_cols: List of exposure value column names
        observable_firm_ids: Set of firm_ids from the full X28 job ad database
    """
    logger.info("Filling zeros for matched firms without AI exposure...")

    primary_col = exp_cols[0] if exp_cols else 'hampole_ai_exposure_avg'

    # --- Flag: is the respondent's firm in our exposure data? ---
    has_firm = result['firm_id'].notna()
    result['firm_in_exposure_data'] = has_firm & result['firm_id'].isin(observable_firm_ids)

    n_in_exposure = result['firm_in_exposure_data'].sum()
    n_with_firm = has_firm.sum()
    n_not_in_exposure = n_with_firm - n_in_exposure
    logger.info(f"  Firm_id present: {n_with_firm:,}")
    logger.info(f"    Firm IN X28 job ad database: {n_in_exposure:,}")
    logger.info(f"    Firm NOT in X28 (unknown): {n_not_in_exposure:,}")

    # --- Identify rows to fill ---
    has_isco = result['isco08_4d'].notna()
    no_exposure = result[primary_col].isna()

    # True zeros: firm is in exposure data, has ISCO, but no match for this occ/year
    true_zero_mask = result['firm_in_exposure_data'] & has_isco & no_exposure
    n_true_zero = true_zero_mask.sum()

    # Unknown: firm has firm_id but is NOT in exposure data
    unknown_mask = has_firm & ~result['firm_in_exposure_data'] & has_isco & no_exposure
    n_unknown = unknown_mask.sum()

    # Fill only true zeros with 0; leave unknowns as NaN
    for col in exp_cols:
        result.loc[true_zero_mask, col] = 0.0

    logger.info(f"  Filled {n_true_zero:,} rows with 0 (firm in exposure data, no AI for this occ/year)")
    logger.info(f"  Left {n_unknown:,} rows as NaN (firm NOT in exposure data — unknown)")

    n_still_missing = result[primary_col].isna().sum()
    logger.info(f"  Total NaN remaining: {n_still_missing:,} (unknown firms + no firm_id/ISCO)")

    return result


# =============================================================================
# Diagnostics
# =============================================================================

def compute_diagnostics(results: Dict[str, pd.DataFrame], output_dir: str):
    """
    Compute and save diagnostics across all merge levels and exposure levels.

    Args:
        results: Dict mapping output mode (isco4d, isco3d, etc.) to merged DataFrame
        output_dir: Directory for diagnostics output
    """
    logger.info("\n" + "=" * 70)
    logger.info("DIAGNOSTICS SUMMARY")
    logger.info("=" * 70)

    diagnostics_rows = []

    # Report on each output mode × exposure level
    for output_mode, df in results.items():
        n_total = len(df)

        # Report on each exposure level present in this output
        for level_name, level_config in EXPOSURE_LEVELS.items():
            suffix = level_config['suffix']
            primary_col = f'hampole_ai_exposure_avg{suffix}'

            if primary_col not in df.columns:
                continue

            n_nonzero = (df[primary_col] > 0).sum()
            n_zero = (df[primary_col] == 0).sum()
            n_missing = df[primary_col].isna().sum()
            n_persons = df.loc[df[primary_col] > 0, 'idpers'].nunique() if n_nonzero > 0 else 0
            mean_exp = df.loc[df[primary_col] > 0, primary_col].mean() if n_nonzero > 0 else 0

            diagnostics_rows.append({
                'output_mode': output_mode,
                'exposure_level': level_name,
                'label': level_config['label'],
                'total_rows': n_total,
                'non_zero_exposure': n_nonzero,
                'zero_exposure': n_zero,
                'missing_exposure': n_missing,
                'pct_non_zero': round(100 * n_nonzero / n_total, 2),
                'unique_persons': n_persons,
                'mean_exposure_when_nonzero': round(mean_exp, 6),
            })

            logger.info(f"  [{output_mode} / {level_config['label']}] "
                        f"nonzero={n_nonzero:,}, zero={n_zero:,}, "
                        f"missing={n_missing:,}, persons={n_persons:,}")

        # foy columns in fallback mode are now suffixed with _foy (same as other output modes)

    # Variance decomposition for 4d level using foy
    if 'isco4d' in results:
        df_4d = results['isco4d']
        foy_col = 'hampole_ai_exposure_avg_foy'
        if foy_col in df_4d.columns:
            _compute_variance_decomposition(df_4d, foy_col, 'isco08_4d', '4-digit ISCO (foy)')

    # Save diagnostics CSV
    diag_df = pd.DataFrame(diagnostics_rows)
    diag_path = os.path.join(output_dir, 'shp_exposure_diagnostics.csv')
    diag_df.to_csv(diag_path, index=False)
    logger.info(f"\nDiagnostics saved to: {diag_path}")


def _compute_variance_decomposition(df: pd.DataFrame, exposure_col: str,
                                      occ_col: str, label: str):
    """Compute within-occupation vs between-occupation variance decomposition."""
    valid = df[[occ_col, exposure_col]].dropna()
    valid = valid[valid[exposure_col] > 0]

    if len(valid) < 10:
        logger.info(f"  Variance decomposition ({label}): insufficient data ({len(valid)} rows)")
        return

    total_var = valid[exposure_col].var()
    if total_var == 0:
        logger.info(f"  Variance decomposition ({label}): zero total variance")
        return

    # Between-occupation variance: variance of group means
    group_means = valid.groupby(occ_col)[exposure_col].mean()
    # Map group means back to get between-component
    valid_with_mean = valid.copy()
    valid_with_mean['_group_mean'] = valid_with_mean[occ_col].map(group_means)
    between_var = valid_with_mean['_group_mean'].var()
    within_var = total_var - between_var

    within_pct = 100 * within_var / total_var
    between_pct = 100 * between_var / total_var

    logger.info(f"\n  Variance decomposition ({label}):")
    logger.info(f"    Total variance: {total_var:.8f}")
    logger.info(f"    Within-occupation: {within_pct:.1f}%")
    logger.info(f"    Between-occupation: {between_pct:.1f}%")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(args):
    """Execute the full SHP-exposure linking pipeline."""

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Generate or load mapping file
    # -------------------------------------------------------------------------
    mapping_path = args.mapping_file
    if args.generate_mapping_file or not os.path.exists(mapping_path):
        if not os.path.exists(mapping_path):
            logger.info(f"Mapping file not found at {mapping_path}, generating...")
        generate_mapping_file(args.exposure_file, mapping_path)
    else:
        logger.info(f"Using existing mapping file: {mapping_path}")

    # -------------------------------------------------------------------------
    # Step 2: Load and prepare SHP data
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Load and prepare SHP data")
    logger.info("=" * 70)

    shp = load_shp_long(args.shp_long_file)
    firmid = load_shp_firmid(args.shp_firmid_file)

    # Merge firm_id onto SHP
    logger.info("Merging firm_id onto SHP panel...")
    n_before = len(shp)
    shp = shp.merge(firmid, on=['idpers', 'year'], how='left')
    if len(shp) != n_before:
        raise ValueError(
            f"SHP row count changed after firm_id merge: {n_before:,} → {len(shp):,}. "
            f"This suggests duplicate (idpers, year) entries in shp_firmid_anon.csv"
        )

    n_matched = shp['firm_id'].notna().sum()
    logger.info(f"  firm_id matched: {n_matched:,} / {len(shp):,}")

    # Fill down firm_id
    shp = fill_down_firm_id(shp)

    # Prepare for merge
    shp = prepare_shp_data(shp, year_min=args.year_min, year_max=args.year_max)

    gc.collect()

    # -------------------------------------------------------------------------
    # Step 3: Load and prepare ALL exposure levels
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Load and prepare exposure data (up to 6 levels)")
    logger.info("=" * 70)

    # Load employment weights if provided
    employment_weights = None
    if args.employment_weights and os.path.exists(args.employment_weights):
        logger.info(f"Loading employment weights: {args.employment_weights}")
        employment_weights = pd.read_csv(args.employment_weights)
        if 'isco08_4d' not in employment_weights.columns or 'weight' not in employment_weights.columns:
            raise ValueError(
                f"Employment weights file must have 'isco08_4d' and 'weight' columns.\n"
                f"Found: {list(employment_weights.columns)}"
            )
    elif args.employment_weights:
        raise FileNotFoundError(f"Employment weights file not found: {args.employment_weights}")

    # Build set of all firm_ids observable in the X28 job ad database.
    x28_mapping_path = os.path.join(os.path.dirname(args.exposure_file), 'company_mapping.csv')
    if not os.path.exists(x28_mapping_path):
        x28_mapping_path = os.path.join('Data', 'company_mapping.csv')
    if not os.path.exists(x28_mapping_path):
        raise FileNotFoundError(
            f"X28 company mapping not found. Expected at Data/company_mapping.csv.\n"
            f"This file contains all company_ids from the X28 job ad database."
        )
    x28_companies = pd.read_csv(x28_mapping_path, usecols=['company_id'])
    observable_firm_ids = set(
        company_id_to_firm_id(int(cid)) for cid in x28_companies['company_id'].unique()
    )
    logger.info(f"X28 database contains {len(observable_firm_ids):,} unique firm_ids (all observable firms)")

    # --- Level 1 (foy): Firm × Occupation × Year (always loaded) ---
    logger.info("\n--- Level 1: Firm × Occupation × Year ---")
    exposure_foy_4d = load_exposure_data(args.exposure_file, mapping_file=mapping_path)
    logger.info(f"  Stage 5 AI-adopting firms: {exposure_foy_4d['firm_id'].nunique():,}")

    # Determine base group cols for foy aggregation
    foy_group_base = ['firm_id', 'year']
    exposure_foy_3d = aggregate_exposure_generic(exposure_foy_4d, 3, foy_group_base, employment_weights)
    exposure_foy_2d = aggregate_exposure_generic(exposure_foy_4d, 2, foy_group_base, employment_weights)

    # Collect all levels: {level_name: {4d: df, 3d: df, 2d: df}}
    all_levels = {
        'foy': {'4d': exposure_foy_4d, '3d': exposure_foy_3d, '2d': exposure_foy_2d},
    }

    # --- Level 2 (fo): Firm × Occupation (time-invariant) ---
    if args.exposure_file_firm_occ:
        logger.info("\n--- Level 2: Firm × Occupation (time-invariant) ---")
        exposure_fo_4d = load_exposure_generic(args.exposure_file_firm_occ, 'fo', mapping_path)
        fo_group_base = ['firm_id']
        exposure_fo_3d = aggregate_exposure_generic(exposure_fo_4d, 3, fo_group_base, employment_weights)
        exposure_fo_2d = aggregate_exposure_generic(exposure_fo_4d, 2, fo_group_base, employment_weights)
        all_levels['fo'] = {'4d': exposure_fo_4d, '3d': exposure_fo_3d, '2d': exposure_fo_2d}

    # --- Level 3 (oy): Occupation × Year ---
    if args.exposure_file_occ_year:
        logger.info("\n--- Level 3: Occupation × Year ---")
        exposure_oy_4d = load_exposure_generic(args.exposure_file_occ_year, 'oy', mapping_path)
        oy_group_base = ['year']
        exposure_oy_3d = aggregate_exposure_generic(exposure_oy_4d, 3, oy_group_base, employment_weights)
        exposure_oy_2d = aggregate_exposure_generic(exposure_oy_4d, 2, oy_group_base, employment_weights)
        all_levels['oy'] = {'4d': exposure_oy_4d, '3d': exposure_oy_3d, '2d': exposure_oy_2d}

    # --- Level 4 (o): Occupation (time-invariant) ---
    if args.exposure_file_occ:
        logger.info("\n--- Level 4: Occupation (time-invariant) ---")
        exposure_o_4d = load_exposure_generic(args.exposure_file_occ, 'o', mapping_path)
        o_group_base = []
        exposure_o_3d = aggregate_exposure_generic(exposure_o_4d, 3, o_group_base, employment_weights)
        exposure_o_2d = aggregate_exposure_generic(exposure_o_4d, 2, o_group_base, employment_weights)
        all_levels['o'] = {'4d': exposure_o_4d, '3d': exposure_o_3d, '2d': exposure_o_2d}

    # --- Level 5 (fy): Firm × Year (derived from Level 1) ---
    logger.info("\n--- Level 5: Firm × Year (derived from Level 1) ---")
    exposure_fy = derive_firm_level_exposure(exposure_foy_4d, time_variant=True)
    # No ISCO aggregation needed — same data for all ISCO modes
    all_levels['fy'] = {'4d': exposure_fy, '3d': exposure_fy, '2d': exposure_fy}

    # --- Level 6 (f): Firm (time-invariant, derived from Level 2 if available, else Level 1) ---
    if 'fo' in all_levels:
        logger.info("\n--- Level 6: Firm (time-invariant, derived from Level 2) ---")
        exposure_f = derive_firm_level_exposure(all_levels['fo']['4d'], time_variant=False)
    else:
        logger.info("\n--- Level 6: Firm (time-invariant, derived from Level 1) ---")
        exposure_f = derive_firm_level_exposure(exposure_foy_4d, time_variant=False)
    all_levels['f'] = {'4d': exposure_f, '3d': exposure_f, '2d': exposure_f}

    active_levels = list(all_levels.keys())
    logger.info(f"\nActive exposure levels: {active_levels}")

    # -------------------------------------------------------------------------
    # Step 4: Merge all exposure levels onto SHP — 4 output modes
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Merge SHP with exposure data (all levels)")
    logger.info("=" * 70)

    # Add firm_in_exposure_data flag once (used by all firm-based levels)
    has_firm = shp['firm_id'].notna()
    shp['firm_in_exposure_data'] = has_firm & shp['firm_id'].isin(observable_firm_ids)

    results = {}

    for isco_mode, isco_label in [('4d', 'isco4d'), ('3d', 'isco3d'), ('2d', 'isco2d')]:
        logger.info(f"\n--- Output mode: {isco_label} ---")
        result = shp.copy()

        # Merge each active level, adding suffixed columns
        for level_name in active_levels:
            exp_data = all_levels[level_name][isco_mode]
            result = merge_exposure_level(result, exp_data, level_name, isco_mode)
            result = fill_zeros_for_level(result, level_name, observable_firm_ids)

        # For Level 1 (foy), also carry through auxiliary columns unsuffixed
        # These come from the 4d foy merge and contain metadata (task counts etc.)
        if isco_mode == '4d':
            aux_exposure = exposure_foy_4d.copy()
            aux_cols = [c for c in EXPOSURE_AUX_COLUMNS if c in aux_exposure.columns]
            extra_cols = [c for c in ['company_id', 'isco08_title', 'company_name'] if c in aux_exposure.columns]
            merge_keys = ['firm_id', 'isco08_4d', 'year']
            aux_subset = aux_exposure[merge_keys + aux_cols + extra_cols].drop_duplicates(subset=merge_keys)
            result = result.merge(aux_subset, on=merge_keys, how='left')

        results[isco_label] = result

    # Mode D: Hierarchical fallback (uses existing foy merge for backward compat)
    logger.info("\n--- Output mode: fallback ---")
    result_fallback = merge_shp_exposure_fallback(shp, exposure_foy_4d, exposure_foy_3d, exposure_foy_2d)
    # For fallback, also merge all non-foy levels at 4d (since fallback handles its own ISCO cascading)
    for level_name in active_levels:
        if level_name == 'foy':
            continue  # Already handled by fallback merge (unsuffixed)
        exp_data = all_levels[level_name]['4d']
        result_fallback = merge_exposure_level(result_fallback, exp_data, level_name, '4d')
        result_fallback = fill_zeros_for_level(result_fallback, level_name, observable_firm_ids)
    # Zero-fill the unsuffixed foy columns from fallback merge
    exp_cols_foy = [c for c in EXPOSURE_VALUE_COLUMNS if c in result_fallback.columns]
    result_fallback = fill_zeros_for_matched_firms(result_fallback, exp_cols_foy, observable_firm_ids)
    # Rename unsuffixed foy core columns to _foy so stage_7 can find them consistently
    foy_suffix = EXPOSURE_LEVELS['foy']['suffix']  # '_foy'
    foy_rename = {c: f'{c}{foy_suffix}' for c in EXPOSURE_CORE_COLUMNS if c in result_fallback.columns}
    result_fallback = result_fallback.rename(columns=foy_rename)
    logger.info(f"  Renamed foy columns to suffix '{foy_suffix}': {list(foy_rename.keys())}")
    results['fallback'] = result_fallback

    gc.collect()

    # -------------------------------------------------------------------------
    # Step 5: Save outputs
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Save output files")
    logger.info("=" * 70)

    output_files = {
        'isco4d': 'shp_exposure_isco4d.csv',
        'isco3d': 'shp_exposure_isco3d.csv',
        'isco2d': 'shp_exposure_isco2d.csv',
        'fallback': 'shp_exposure_isco4d_fallback.csv',
    }

    for level, filename in output_files.items():
        path = os.path.join(args.output_dir, filename)
        results[level].to_csv(path, index=False)
        logger.info(f"  Saved {level}: {path} ({len(results[level]):,} rows)")

    # -------------------------------------------------------------------------
    # Step 6: Diagnostics
    # -------------------------------------------------------------------------
    compute_diagnostics(results, args.output_dir)

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 6 SHP: Link AI Exposure to Swiss Household Panel Respondents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (paths below use the v11 SHP release and Thomas's May 2026 firmid file):
  # Basic run
  python3 stage_6_shp_exposure.py \\
      --shp-long-file ~/Dropbox/kurer_allardice_technology/data/original/shp/swissubase_932_11_0/data/Data_STATA/SHP-Data-Longfile-STATA/shplong_p_user.dta \\
      --shp-firmid-file ~/Dropbox/kurer_allardice_technology/data/created/shp_firmid_anon.csv \\
      --exposure-file Data/isco_firm_year_exposure_core_tasks_pct_05_ce_0.0_BGE.csv \\
      --output-dir Data/shp_exposure/

  # With employment weights for ISCO aggregation
  python3 stage_6_shp_exposure.py \\
      --shp-long-file ~/Dropbox/kurer_allardice_technology/data/original/shp/swissubase_932_11_0/data/Data_STATA/SHP-Data-Longfile-STATA/shplong_p_user.dta \\
      --shp-firmid-file ~/Dropbox/kurer_allardice_technology/data/created/shp_firmid_anon.csv \\
      --exposure-file Data/isco_firm_year_exposure_core_tasks_pct_05_ce_0.0_BGE.csv \\
      --employment-weights Data/isco_employment_weights.csv

  # Generate fresh mapping file
  python3 stage_6_shp_exposure.py \\
      --shp-long-file ~/Dropbox/kurer_allardice_technology/data/original/shp/swissubase_932_11_0/data/Data_STATA/SHP-Data-Longfile-STATA/shplong_p_user.dta \\
      --shp-firmid-file ~/Dropbox/kurer_allardice_technology/data/created/shp_firmid_anon.csv \\
      --exposure-file Data/isco_firm_year_exposure_core_tasks_pct_05_ce_0.0_BGE.csv \\
      --generate-mapping-file
        """
    )

    # Required inputs
    parser.add_argument('--shp-long-file', required=True,
                        help='Path to SHP long file (shplong_p_user.dta)')
    parser.add_argument('--shp-firmid-file', required=True,
                        help='Path to anonymized firm ID file (shp_firmid_anon.csv)')
    parser.add_argument('--exposure-file', required=True,
                        help='Path to Stage 5 ISCO exposure CSV')

    # Output
    parser.add_argument('--output-dir', default='Data/shp_exposure/',
                        help='Output directory for merged files (default: Data/shp_exposure/)')

    # Mapping
    parser.add_argument('--mapping-file', default='Data/company_id_to_firm_id.csv',
                        help='Path to company_id ↔ firm_id mapping CSV (default: Data/company_id_to_firm_id.csv)')
    parser.add_argument('--generate-mapping-file', action='store_true',
                        help='Force regeneration of mapping file from exposure data')

    # Additional exposure files (optional — levels 2-4)
    parser.add_argument('--exposure-file-firm-occ', default=None,
                        help='Path to Stage 5 firm×occ time-invariant exposure CSV (Level 2)')
    parser.add_argument('--exposure-file-occ-year', default=None,
                        help='Path to Stage 5 occupation×year exposure CSV (Level 3)')
    parser.add_argument('--exposure-file-occ', default=None,
                        help='Path to Stage 5 occupation time-invariant exposure CSV (Level 4)')

    # Optional weights
    parser.add_argument('--employment-weights', default=None,
                        help='Path to employment weights CSV (columns: isco08_4d, weight). '
                             'If not provided, uses simple mean for ISCO aggregation.')

    # Year range
    parser.add_argument('--year-min', type=int, default=2012,
                        help='Minimum year to include (default: 2012)')
    parser.add_argument('--year-max', type=int, default=2023,
                        help='Maximum year to include (default: 2023)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
