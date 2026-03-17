#!/usr/bin/env python3
"""
Stage 7 SHP: Exploratory Data Analysis of AI Exposure and Individual Outcomes

Analyzes the relationship between firm-level AI exposure (from Stages 5/6) and
individual-level economic and political outcomes in the Swiss Household Panel (SHP).

This is the "alternative Stage 7" for the SHP arm of the pipeline. The existing
Stage 7 (stage_7_analyze_results.py) analyzes job-ad level exposure; this script
analyzes person-year panel data from the SHP with linked AI exposure.

Input: SHP exposure file with hierarchical fallback matching (from stage_6_shp_exposure.py)
Output: Tables (CSV) and figures (PNG) in structured output directory

Usage:
    python3 stage_7_shp_eda.py \\
        --input Data/shp_exposure/shp_exposure_isco4d_fallback.csv \\
        --output-dir Data/shp_eda/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI use
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from scipy import stats

# Attempt to import statsmodels; fail fast if not available
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols as sm_ols
except ImportError:
    raise ImportError(
        "statsmodels is required for regression analyses.\n"
        "Install with: pip install statsmodels"
    )

# Attempt to import linearmodels for panel FE; fall back to demeaned OLS if unavailable
try:
    from linearmodels.panel import PanelOLS
    HAS_LINEARMODELS = True
except ImportError:
    HAS_LINEARMODELS = False

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

FIGURE_DPI = 300
FIGURE_FORMAT = 'png'

# Main exposure variable (default; overridden per-level at runtime)
PRIMARY_EXPOSURE = 'hampole_ai_exposure_avg'

# Exposure level definitions: maps level code → column suffix from Stage 6 SHP
EXPOSURE_LEVEL_SUFFIXES = {
    'foy': '_foy',    # Firm × Occupation × Year
    'fo':  '_fo',     # Firm × Occupation (time-invariant)
    'oy':  '_oy',     # Occupation × Year
    'o':   '_o',      # Occupation (time-invariant)
    'fy':  '_fy',     # Firm × Year
    'f':   '_f',      # Firm (time-invariant)
}

EXPOSURE_LEVEL_LABELS = {
    'foy': 'Firm×Occ×Year',
    'fo':  'Firm×Occ (time-invariant)',
    'oy':  'Occ×Year',
    'o':   'Occ (time-invariant)',
    'fy':  'Firm×Year',
    'f':   'Firm (time-invariant)',
}

# All exposure variables from Stage 5/6 (unsuffixed, for backward compat)
EXPOSURE_COLUMNS = [
    'hampole_ai_exposure_avg',
    'binary_ai_exposure_avg',
    'hampole_occupation_exposure',
    'binary_occupation_exposure',
    'log_ai_intensity',
    'n_ai_apps_firm_year',
    'n_onet_codes_contributing',
    'total_tasks_occupation',
    'total_importance_weight',
]

# SHP missing values: all negative values are missing
# (-1 = inapplicable, -2 = no answer, -3 = does not know, -7 = filter error, -8 = other error)
# Recoding uses df[var] < 0 to catch any negative sentinel


def significance_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    elif p < 0.1:
        return '+'
    else:
        return ''


# Human-readable labels for all outcome variables
VARIABLE_LABELS = {
    'iwyn': 'Yearly work income (net)',
    'wstat': 'Working/employment status',
    'pw86': 'Job security (self-rated)',
    'pw86a': 'Job security (self-rated)',
    'pp10': 'Left-right self-placement (0-10)',
    'pp13': 'Social benefits/expenses preference',
    'pp17': 'Taxes on high incomes preference',
    'pp22': 'Gender equality measures preference',
    'pp16': 'Environment vs. economic growth',
    'pp15': 'Equal chances for foreigners',
    'pp19': 'Vote choice / party preference',
}


def get_var_label(var: str) -> str:
    """Return human-readable label for a variable, or the variable name itself."""
    return VARIABLE_LABELS.get(var, var)


# Economic outcome variables
ECONOMIC_OUTCOMES = {
    'iwyn': 'Yearly work income (net)',
    'wstat': 'Working/employment status',
}

# Job security: pw86a exists in some SHP versions with more waves; pw86 is the fallback
JOB_SECURITY_CANDIDATES = ['pw86a', 'pw86']

# Political outcome variables (ordinal scales, treatable as continuous for regressions)
POLITICAL_OUTCOMES_CONTINUOUS = {
    'pp10': 'Left-right self-placement (0-10)',
    'pp13': 'Social benefits/expenses preference',
    'pp17': 'Taxes on high incomes preference',
    'pp22': 'Gender equality measures preference',
    'pp16': 'Environment vs. economic growth',
    'pp15': 'Equal chances for foreigners',
}

# Categorical political variable (do NOT compute means)
POLITICAL_OUTCOMES_CATEGORICAL = {
    'pp19': 'Vote choice / party preference',
}


# =============================================================================
# Data Loading and Cleaning
# =============================================================================

def load_and_clean_data(input_file: str) -> pd.DataFrame:
    """
    Load the SHP exposure fallback file and recode missing values.

    Args:
        input_file: Path to shp_exposure_isco4d_fallback.csv

    Returns:
        Cleaned DataFrame with missing values recoded and derived indicators
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logger.info(f"Loading SHP exposure data: {input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    logger.info(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    logger.info(f"  Year range: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"  Unique persons: {df['idpers'].nunique():,}")

    # Validate required columns (match_level is optional — only in fallback file)
    # Check for idpers and year; exposure column validated per-level in run_single_level
    required_cols = ['idpers', 'year']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns (first 30): {list(df.columns[:30])}"
        )

    # Check that at least one exposure column exists (suffixed or unsuffixed)
    any_exposure = any(c for c in df.columns if 'hampole_ai_exposure_avg' in c)
    if not any_exposure:
        raise ValueError(
            f"No exposure columns found (expected 'hampole_ai_exposure_avg' or suffixed variants).\n"
            f"Available columns (first 30): {list(df.columns[:30])}"
        )

    # --- Identify which outcome variables are available ---
    all_outcome_vars = []

    # Economic outcomes
    for var in ECONOMIC_OUTCOMES:
        if var in df.columns:
            all_outcome_vars.append(var)
        else:
            logger.warning(f"  Economic outcome variable '{var}' not found in data")

    # Job security: pick the one with best coverage
    job_security_var = _select_job_security_variable(df)
    if job_security_var:
        all_outcome_vars.append(job_security_var)

    # Political outcomes (continuous)
    for var in POLITICAL_OUTCOMES_CONTINUOUS:
        if var in df.columns:
            all_outcome_vars.append(var)
        else:
            logger.warning(f"  Political outcome variable '{var}' not found in data")

    # Political outcome (categorical)
    for var in POLITICAL_OUTCOMES_CATEGORICAL:
        if var in df.columns:
            all_outcome_vars.append(var)
        else:
            logger.warning(f"  Political outcome variable '{var}' not found in data")

    # --- Recode negative values to NaN for all outcome variables ---
    logger.info("Recoding SHP missing value codes (negative values) to NaN...")
    for var in all_outcome_vars:
        if var not in df.columns:
            continue
        n_negative = (df[var] < 0).sum()
        if n_negative > 0:
            df.loc[df[var] < 0, var] = np.nan
            logger.info(f"  {var}: recoded {n_negative:,} negative values to NaN")

    # --- Exposure indicators are created per-level in run_single_level ---
    # Initialize placeholder columns (will be overwritten for each level)
    df['has_exposure'] = 0
    df['exposed_ever'] = 0
    logger.info("  Exposure indicators will be set per-level in run_single_level()")

    # --- Log summary ---
    if 'match_level' in df.columns:
        logger.info("\nMatch level distribution:")
        match_counts = df['match_level'].value_counts()
        for level, count in match_counts.items():
            logger.info(f"  {level}: {count:,} ({100*count/len(df):.1f}%)")
    else:
        logger.info("  (match_level column not present — single-level file)")

    return df


def _select_job_security_variable(df: pd.DataFrame, log: bool = True) -> Optional[str]:
    """
    Select the best job security variable (pw86a preferred, fallback to pw86).

    Args:
        df: DataFrame with SHP variables
        log: Whether to log selection details (set False to suppress duplicate logs)

    Returns:
        Variable name with best coverage, or None if neither exists
    """
    best_var = None
    best_count = 0

    for var in JOB_SECURITY_CANDIDATES:
        if var in df.columns:
            # Count valid (non-negative, non-NaN) observations
            valid = df[var].notna() & (df[var] >= 0)
            n_valid = valid.sum()
            if log:
                logger.info(f"  Job security variable '{var}': {n_valid:,} valid observations")
            if n_valid > best_count:
                best_count = n_valid
                best_var = var

    if best_var:
        if log:
            logger.info(f"  Selected job security variable: {best_var} ({best_count:,} valid obs)")
    else:
        if log:
            logger.warning("  No job security variable (pw86, pw86a) found in data")

    return best_var


# =============================================================================
# Section 1: Descriptive Statistics
# =============================================================================

def section_1_descriptive_stats(df: pd.DataFrame, output_dir: str,
                                job_security_var: Optional[str] = None):
    """
    Generate descriptive statistics for all outcome and exposure variables.

    Produces:
      - Summary stats table (full sample and by exposure status)
      - Exposure distribution (among exposed)
      - Coverage table
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 1: DESCRIPTIVE STATISTICS")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    # --- Build list of outcome variables to summarize ---
    outcome_vars = []
    outcome_labels = {}

    for var, label in ECONOMIC_OUTCOMES.items():
        if var in df.columns:
            outcome_vars.append(var)
            outcome_labels[var] = label

    if job_security_var and job_security_var in df.columns:
        outcome_vars.append(job_security_var)
        outcome_labels[job_security_var] = f'Job security ({job_security_var})'

    for var, label in POLITICAL_OUTCOMES_CONTINUOUS.items():
        if var in df.columns:
            outcome_vars.append(var)
            outcome_labels[var] = label

    for var, label in POLITICAL_OUTCOMES_CATEGORICAL.items():
        if var in df.columns:
            outcome_vars.append(var)
            outcome_labels[var] = label

    # --- Summary statistics: full sample ---
    logger.info("\n--- Summary Statistics (Full Sample) ---")
    summary_rows = []

    categorical_vars = set(POLITICAL_OUTCOMES_CATEGORICAL.keys()) | {'wstat'}
    for var in outcome_vars:
        if var not in df.columns:
            continue
        series = df[var].dropna()
        if var in categorical_vars:
            # Categorical variables: report N, unique categories, mode — not mean/sd
            row = {
                'variable': var,
                'label': outcome_labels.get(var, var),
                'N': len(series),
                'mean': np.nan,
                'sd': np.nan,
                'min': series.min(),
                'p25': np.nan,
                'median': np.nan,
                'p75': np.nan,
                'max': series.max(),
                'pct_missing': 100 * df[var].isna().sum() / len(df),
                'n_categories': series.nunique(),
            }
            summary_rows.append(row)
            logger.info(f"  {var} (categorical): N={row['N']:,}, {row['n_categories']} unique values")
        else:
            row = {
                'variable': var,
                'label': outcome_labels.get(var, var),
                'N': len(series),
                'mean': series.mean(),
                'sd': series.std(),
                'min': series.min(),
                'p25': series.quantile(0.25),
                'median': series.median(),
                'p75': series.quantile(0.75),
                'max': series.max(),
                'pct_missing': 100 * df[var].isna().sum() / len(df),
            }
            summary_rows.append(row)
            logger.info(f"  {var}: N={row['N']:,}, mean={row['mean']:.3f}, sd={row['sd']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(tables_dir, 'descriptive_stats_full_sample.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  Saved: {summary_path}")

    # --- Summary statistics: by exposure status ---
    logger.info("\n--- Summary Statistics by Exposure Status ---")
    by_exposure_rows = []

    for var in outcome_vars:
        if var not in df.columns:
            continue
        for exposed_val, group_label in [(0, 'Not exposed'), (1, 'Exposed')]:
            sub = df.loc[df['has_exposure'] == exposed_val, var].dropna()
            if len(sub) == 0:
                continue
            if var in categorical_vars:
                row = {
                    'variable': var,
                    'label': outcome_labels.get(var, var),
                    'group': group_label,
                    'N': len(sub),
                    'mean': np.nan,
                    'sd': np.nan,
                    'min': sub.min(),
                    'median': np.nan,
                    'max': sub.max(),
                    'n_categories': sub.nunique(),
                }
            else:
                row = {
                    'variable': var,
                    'label': outcome_labels.get(var, var),
                    'group': group_label,
                    'N': len(sub),
                    'mean': sub.mean(),
                    'sd': sub.std(),
                    'min': sub.min(),
                    'median': sub.median(),
                    'max': sub.max(),
                }
            by_exposure_rows.append(row)

    by_exposure_df = pd.DataFrame(by_exposure_rows)
    by_exp_path = os.path.join(tables_dir, 'descriptive_stats_by_exposure.csv')
    by_exposure_df.to_csv(by_exp_path, index=False)
    logger.info(f"  Saved: {by_exp_path}")

    # Print comparison for key variables
    for var in outcome_vars:
        if var in categorical_vars:
            continue  # skip categorical variables from mean comparison
        exposed = df.loc[df['has_exposure'] == 1, var].dropna()
        not_exposed = df.loc[df['has_exposure'] == 0, var].dropna()
        if len(exposed) > 0 and len(not_exposed) > 0:
            logger.info(f"  {var}: exposed mean={exposed.mean():.3f} (N={len(exposed):,}), "
                        f"not exposed mean={not_exposed.mean():.3f} (N={len(not_exposed):,})")

    # --- Exposure distribution (among those with non-zero exposure) ---
    logger.info("\n--- Exposure Distribution (Non-Zero) ---")
    exposed_vals = df.loc[df[PRIMARY_EXPOSURE] > 0, PRIMARY_EXPOSURE]

    if len(exposed_vals) > 0:
        exp_stats = {
            'N': len(exposed_vals),
            'mean': exposed_vals.mean(),
            'sd': exposed_vals.std(),
            'min': exposed_vals.min(),
            'p10': exposed_vals.quantile(0.10),
            'p25': exposed_vals.quantile(0.25),
            'median': exposed_vals.median(),
            'p75': exposed_vals.quantile(0.75),
            'p90': exposed_vals.quantile(0.90),
            'p99': exposed_vals.quantile(0.99),
            'max': exposed_vals.max(),
        }
        logger.info(f"  N={exp_stats['N']:,}, mean={exp_stats['mean']:.6f}, "
                    f"median={exp_stats['median']:.6f}, max={exp_stats['max']:.6f}")

        exp_stats_df = pd.DataFrame([exp_stats])
        exp_stats_path = os.path.join(tables_dir, 'exposure_distribution_nonzero.csv')
        exp_stats_df.to_csv(exp_stats_path, index=False)
        logger.info(f"  Saved: {exp_stats_path}")

        # Histogram of exposure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(exposed_vals, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel(f'{PRIMARY_EXPOSURE}')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of AI Exposure (N={len(exposed_vals):,} non-zero observations)')
        ax.axvline(exposed_vals.median(), color='red', linestyle='--',
                    label=f'Median: {exposed_vals.median():.6f}')
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, 'exposure_distribution_nonzero.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")

    # --- Coverage table ---
    logger.info("\n--- Coverage Table ---")
    coverage_rows = []
    for var in outcome_vars:
        if var not in df.columns:
            continue
        has_outcome = df[var].notna()
        has_exp = df[PRIMARY_EXPOSURE].notna() & (df[PRIMARY_EXPOSURE] > 0)
        has_both = has_outcome & has_exp

        coverage_rows.append({
            'variable': var,
            'label': outcome_labels.get(var, var),
            'n_valid_outcome': has_outcome.sum(),
            'n_with_exposure': has_exp.sum(),
            'n_both': has_both.sum(),
            'pct_outcome_of_total': 100 * has_outcome.sum() / len(df),
        })

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = os.path.join(tables_dir, 'outcome_exposure_coverage.csv')
    coverage_df.to_csv(coverage_path, index=False)
    logger.info(f"  Saved: {coverage_path}")


# =============================================================================
# Section 2: Economic Outcomes EDA
# =============================================================================

def section_2_economic_outcomes(df: pd.DataFrame, output_dir: str,
                                job_security_var: Optional[str] = None):
    """
    Analyze economic outcomes (income, employment, job security) by AI exposure.

    Produces:
      - Income by exposure status with t-test
      - Binscatter of income vs exposure
      - Income trends over time
      - Within-person income changes
      - Employment status distribution
      - Transition matrices
      - Job security analysis
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 2: ECONOMIC OUTCOMES EDA")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    # -------------------------------------------------------------------------
    # 2.1: Income Analysis (iwyn)
    # -------------------------------------------------------------------------
    if 'iwyn' in df.columns:
        _analyze_income(df, tables_dir, figures_dir)
    else:
        logger.warning("  Variable 'iwyn' not available, skipping income analysis")

    # -------------------------------------------------------------------------
    # 2.2: Employment Status (wstat)
    # -------------------------------------------------------------------------
    if 'wstat' in df.columns:
        _analyze_employment(df, tables_dir, figures_dir)
    else:
        logger.warning("  Variable 'wstat' not available, skipping employment analysis")

    # -------------------------------------------------------------------------
    # 2.3: Job Security
    # -------------------------------------------------------------------------
    if job_security_var and job_security_var in df.columns:
        _analyze_job_security(df, job_security_var, tables_dir, figures_dir)
    else:
        logger.warning("  No job security variable available, skipping")


def _analyze_income(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """Analyze income (iwyn) by exposure status."""
    logger.info("\n--- 2.1: Income Analysis (iwyn) ---")

    # Mean income by exposure status with t-test
    exposed = df.loc[df['has_exposure'] == 1, 'iwyn'].dropna()
    not_exposed = df.loc[df['has_exposure'] == 0, 'iwyn'].dropna()

    logger.info(f"  Exposed: N={len(exposed):,}, mean={exposed.mean():,.0f}, sd={exposed.std():,.0f}")
    logger.info(f"  Not exposed: N={len(not_exposed):,}, mean={not_exposed.mean():,.0f}, sd={not_exposed.std():,.0f}")

    if len(exposed) > 1 and len(not_exposed) > 1:
        t_stat, p_val = stats.ttest_ind(exposed, not_exposed, equal_var=False)
        logger.info(f"  Welch t-test: t={t_stat:.4f}, p={p_val:.6f}")

        ttest_df = pd.DataFrame([{
            'group_exposed_N': len(exposed),
            'group_exposed_mean': exposed.mean(),
            'group_exposed_sd': exposed.std(),
            'group_not_exposed_N': len(not_exposed),
            'group_not_exposed_mean': not_exposed.mean(),
            'group_not_exposed_sd': not_exposed.std(),
            'difference': exposed.mean() - not_exposed.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
        }])
        ttest_path = os.path.join(tables_dir, 'income_by_exposure_ttest.csv')
        ttest_df.to_csv(ttest_path, index=False)
        logger.info(f"  Saved: {ttest_path}")

    # Binscatter: income vs exposure (among exposed)
    df_exposed = df.loc[(df[PRIMARY_EXPOSURE] > 0) & df['iwyn'].notna()].copy()
    if len(df_exposed) > 10:
        n_bins = min(20, len(df_exposed) // 5)
        if n_bins >= 2:
            df_exposed['exposure_bin'] = pd.qcut(
                df_exposed[PRIMARY_EXPOSURE], q=n_bins, duplicates='drop'
            )
            binscatter = df_exposed.groupby('exposure_bin', observed=True)['iwyn'].agg(['mean', 'count', 'sem'])
            binscatter['bin_midpoint'] = [interval.mid for interval in binscatter.index]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(binscatter['bin_midpoint'], binscatter['mean'], s=binscatter['count'] * 2,
                       alpha=0.7, edgecolors='black')
            ax.set_xlabel(f'AI Exposure ({PRIMARY_EXPOSURE})')
            ax.set_ylabel('Mean Income (iwyn)')
            ax.set_title(f'Binscatter: Income vs. AI Exposure (N={len(df_exposed):,})')

            # Add trend line
            z = np.polyfit(binscatter['bin_midpoint'], binscatter['mean'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(binscatter['bin_midpoint'].min(), binscatter['bin_midpoint'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'Linear fit')
            ax.legend()
            fig.tight_layout()
            fig_path = os.path.join(figures_dir, 'income_vs_exposure_binscatter.png')
            fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
            plt.close(fig)
            logger.info(f"  Saved: {fig_path}")

    # Income trends over time by exposure status
    income_by_year = df.loc[df['iwyn'].notna()].groupby(['year', 'has_exposure'])['iwyn'].agg(
        ['mean', 'count', 'sem']
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for exposed_val, label, color in [(0, 'Not Exposed', 'steelblue'), (1, 'Exposed', 'coral')]:
        sub = income_by_year[income_by_year['has_exposure'] == exposed_val]
        ax.plot(sub['year'], sub['mean'], marker='o', label=f'{label}', color=color)
        ax.fill_between(sub['year'],
                        sub['mean'] - 1.96 * sub['sem'],
                        sub['mean'] + 1.96 * sub['sem'],
                        alpha=0.2, color=color)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Income (iwyn)')
    ax.set_title('Income Trends by AI Exposure Status')
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, 'income_trends_by_exposure.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    income_trends_path = os.path.join(tables_dir, 'income_trends_by_exposure.csv')
    income_by_year.to_csv(income_trends_path, index=False)
    logger.info(f"  Saved: {income_trends_path}")

    # Within-person income changes when exposure changes
    _within_person_income_changes(df, tables_dir, figures_dir)


def _within_person_income_changes(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """Analyze within-person income changes conditional on exposure changes."""
    logger.info("\n  Within-person income changes:")

    # Keep only person-years with valid income
    panel = df.loc[df['iwyn'].notna(), ['idpers', 'year', 'iwyn', 'has_exposure', PRIMARY_EXPOSURE]].copy()
    panel = panel.sort_values(['idpers', 'year'])

    # Compute year-over-year changes within person
    panel['iwyn_lag'] = panel.groupby('idpers')['iwyn'].shift(1)
    panel['exposure_lag'] = panel.groupby('idpers')['has_exposure'].shift(1)
    panel['iwyn_change'] = panel['iwyn'] - panel['iwyn_lag']
    panel['exposure_change'] = panel['has_exposure'] - panel['exposure_lag']

    panel_valid = panel.dropna(subset=['iwyn_change', 'exposure_change'])

    # Categorize exposure transitions
    transitions = {
        'Stay unexposed (0->0)': panel_valid[panel_valid['exposure_change'] == 0]
                                 .loc[panel_valid['has_exposure'] == 0],
        'Gain exposure (0->1)': panel_valid[panel_valid['exposure_change'] == 1],
        'Lose exposure (1->0)': panel_valid[panel_valid['exposure_change'] == -1],
        'Stay exposed (1->1)': panel_valid[panel_valid['exposure_change'] == 0]
                               .loc[panel_valid['has_exposure'] == 1],
    }

    transition_rows = []
    for label, sub in transitions.items():
        if len(sub) > 0:
            row = {
                'transition': label,
                'N': len(sub),
                'mean_income_change': sub['iwyn_change'].mean(),
                'sd_income_change': sub['iwyn_change'].std(),
                'median_income_change': sub['iwyn_change'].median(),
            }
            transition_rows.append(row)
            logger.info(f"    {label}: N={row['N']:,}, mean change={row['mean_income_change']:,.0f}")

    if transition_rows:
        transition_df = pd.DataFrame(transition_rows)
        trans_path = os.path.join(tables_dir, 'income_changes_by_exposure_transition.csv')
        transition_df.to_csv(trans_path, index=False)
        logger.info(f"  Saved: {trans_path}")


def _analyze_employment(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """Analyze employment status (wstat) by exposure."""
    logger.info("\n--- 2.2: Employment Status (wstat) ---")

    # Distribution of wstat by exposure
    wstat_valid = df.loc[df['wstat'].notna()].copy()
    cross_tab = pd.crosstab(
        wstat_valid['wstat'],
        wstat_valid['has_exposure'],
        margins=True,
        normalize='columns'
    )
    cross_tab_counts = pd.crosstab(
        wstat_valid['wstat'],
        wstat_valid['has_exposure'],
        margins=True,
    )

    cross_tab_path = os.path.join(tables_dir, 'employment_status_by_exposure_pct.csv')
    cross_tab.to_csv(cross_tab_path)
    logger.info(f"  Saved: {cross_tab_path}")

    counts_path = os.path.join(tables_dir, 'employment_status_by_exposure_counts.csv')
    cross_tab_counts.to_csv(counts_path)
    logger.info(f"  Saved: {counts_path}")

    # Chi-square test
    contingency = pd.crosstab(wstat_valid['wstat'], wstat_valid['has_exposure'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
        logger.info(f"  Chi-square test: chi2={chi2:.4f}, p={p_val:.6f}, dof={dof}")

    # Bar chart of employment status by exposure
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create plot-ready crosstab without margins
    cross_tab_plot = pd.crosstab(
        wstat_valid['wstat'], wstat_valid['has_exposure'], normalize='columns'
    )
    cross_tab_plot.columns = ['Not Exposed', 'Exposed']
    cross_tab_plot.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Employment Status (wstat)')
    ax.set_ylabel('Proportion')
    ax.set_title('Employment Status Distribution by AI Exposure')
    ax.legend(title='Exposure Status')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, 'employment_status_by_exposure.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    # Transition matrix: employment changes conditional on exposure changes
    logger.info("  Employment transition matrix:")
    panel = df.loc[df['wstat'].notna(), ['idpers', 'year', 'wstat', 'has_exposure']].copy()
    panel = panel.sort_values(['idpers', 'year'])
    panel['wstat_lag'] = panel.groupby('idpers')['wstat'].shift(1)
    panel['exposure_lag'] = panel.groupby('idpers')['has_exposure'].shift(1)

    panel_valid = panel.dropna(subset=['wstat_lag', 'exposure_lag'])

    for exp_change, label in [(0, 'No exposure change'), (1, 'Gained exposure'), (-1, 'Lost exposure')]:
        sub = panel_valid[
            (panel_valid['has_exposure'] - panel_valid['exposure_lag']) == exp_change
        ]
        if len(sub) > 5:
            trans_matrix = pd.crosstab(
                sub['wstat_lag'].astype(int),
                sub['wstat'].astype(int),
                normalize='index'
            )
            trans_path = os.path.join(tables_dir, f'wstat_transition_{label.replace(" ", "_").lower()}.csv')
            trans_matrix.to_csv(trans_path)
            logger.info(f"    {label}: N={len(sub):,} transitions, saved to {trans_path}")


def _analyze_job_security(df: pd.DataFrame, var: str, tables_dir: str, figures_dir: str):
    """Analyze job security variable by exposure."""
    logger.info(f"\n--- 2.3: Job Security ({var}) ---")

    exposed = df.loc[df['has_exposure'] == 1, var].dropna()
    not_exposed = df.loc[df['has_exposure'] == 0, var].dropna()

    logger.info(f"  Exposed: N={len(exposed):,}, mean={exposed.mean():.3f}")
    logger.info(f"  Not exposed: N={len(not_exposed):,}, mean={not_exposed.mean():.3f}")

    if len(exposed) > 1 and len(not_exposed) > 1:
        t_stat, p_val = stats.ttest_ind(exposed, not_exposed, equal_var=False)
        logger.info(f"  Welch t-test: t={t_stat:.4f}, p={p_val:.6f}")

        result_df = pd.DataFrame([{
            'variable': var,
            'exposed_N': len(exposed),
            'exposed_mean': exposed.mean(),
            'exposed_sd': exposed.std(),
            'not_exposed_N': len(not_exposed),
            'not_exposed_mean': not_exposed.mean(),
            'not_exposed_sd': not_exposed.std(),
            'difference': exposed.mean() - not_exposed.mean(),
            't_statistic': t_stat,
            'p_value': p_val,
        }])
        result_path = os.path.join(tables_dir, f'job_security_{var}_by_exposure.csv')
        result_df.to_csv(result_path, index=False)
        logger.info(f"  Saved: {result_path}")

    # Box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = df.loc[df[var].notna()].copy()
    plot_data['Exposure Status'] = plot_data['has_exposure'].map({0: 'Not Exposed', 1: 'Exposed'})
    sns.boxplot(data=plot_data, x='Exposure Status', y=var, ax=ax)
    ax.set_title(f'Job Security ({var}) by AI Exposure Status')
    ax.set_ylabel(f'Job Security ({var})')
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, f'job_security_{var}_by_exposure.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")


# =============================================================================
# Section 3: Political Outcomes EDA
# =============================================================================

def section_3_political_outcomes(df: pd.DataFrame, output_dir: str):
    """
    Analyze political outcomes by AI exposure.

    For continuous (ordinal) outcomes: mean comparison, OLS, within-person FE regression
    For categorical (pp19): party share comparison, conditional changes

    Produces:
      - OLS and FE regression tables for each continuous outcome
      - Distribution comparisons
      - Left-right deep dive
      - Vote choice analysis
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 3: POLITICAL OUTCOMES EDA")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    # -------------------------------------------------------------------------
    # 3.1: Continuous political outcomes (OLS + FE)
    # -------------------------------------------------------------------------
    _analyze_continuous_political(df, tables_dir, figures_dir)

    # -------------------------------------------------------------------------
    # 3.2: Left-right placement deep dive (pp10)
    # -------------------------------------------------------------------------
    if 'pp10' in df.columns:
        _leftright_deep_dive(df, tables_dir, figures_dir)

    # -------------------------------------------------------------------------
    # 3.3: Vote choice analysis (pp19)
    # -------------------------------------------------------------------------
    if 'pp19' in df.columns:
        _vote_choice_analysis(df, tables_dir, figures_dir)


def _analyze_continuous_political(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """OLS and FE regressions for each continuous political outcome."""
    logger.info("\n--- 3.1: Continuous Political Outcomes ---")

    all_results = []

    for var, label in POLITICAL_OUTCOMES_CONTINUOUS.items():
        if var not in df.columns:
            logger.info(f"  Skipping {var}: not in data")
            continue

        valid = df.loc[df[var].notna() & df[PRIMARY_EXPOSURE].notna()].copy()
        n_valid = len(valid)
        n_exposed_valid = (valid['has_exposure'] == 1).sum()

        if n_valid < 30 or n_exposed_valid < 5:
            logger.info(f"  Skipping {var}: insufficient data (N={n_valid}, exposed={n_exposed_valid})")
            continue

        logger.info(f"\n  {get_var_label(var)}: N={n_valid:,}, exposed={n_exposed_valid:,}")

        # Mean comparison
        exposed_mean = valid.loc[valid['has_exposure'] == 1, var].mean()
        not_exposed_mean = valid.loc[valid['has_exposure'] == 0, var].mean()
        logger.info(f"    Mean: exposed={exposed_mean:.3f}, not exposed={not_exposed_mean:.3f}, "
                    f"diff={exposed_mean - not_exposed_mean:.3f}")

        # --- OLS with year FE ---
        try:
            valid['year_factor'] = valid['year'].astype(str)
            year_dummies = pd.get_dummies(valid['year_factor'], prefix='yr', drop_first=True,
                                          dtype=float)
            X_ols = pd.concat([valid[[PRIMARY_EXPOSURE]], year_dummies], axis=1)
            X_ols = sm.add_constant(X_ols)
            y_ols = valid[var].astype(float)

            model_ols = sm.OLS(y_ols, X_ols).fit(cov_type='HC1')
            coef_ols = model_ols.params[PRIMARY_EXPOSURE]
            se_ols = model_ols.bse[PRIMARY_EXPOSURE]
            p_ols = model_ols.pvalues[PRIMARY_EXPOSURE]
            r2_ols = model_ols.rsquared

            logger.info(f"    OLS (year FE): coef={coef_ols:.4f}, se={se_ols:.4f}, "
                        f"p={p_ols:.4f}{significance_stars(p_ols)}, R2={r2_ols:.4f}")
        except Exception as e:
            logger.warning(f"    OLS failed for {var}: {e}")
            coef_ols = se_ols = p_ols = r2_ols = np.nan

        # --- Within-person FE regression ---
        coef_fe, se_fe, p_fe, r2_fe = _run_person_fe_regression(
            valid, var, PRIMARY_EXPOSURE
        )
        if not np.isnan(coef_fe):
            logger.info(f"    Person FE: coef={coef_fe:.4f}, se={se_fe:.4f}, "
                        f"p={p_fe:.4f}{significance_stars(p_fe)}, R2={r2_fe:.4f}")
        else:
            logger.info(f"    Person FE: could not estimate (insufficient within-person variation)")

        all_results.append({
            'variable': var,
            'label': label,
            'N': n_valid,
            'N_exposed': n_exposed_valid,
            'mean_exposed': exposed_mean,
            'mean_not_exposed': not_exposed_mean,
            'mean_difference': exposed_mean - not_exposed_mean,
            'ols_coef': coef_ols,
            'ols_se': se_ols,
            'ols_p': p_ols,
            'ols_r2': r2_ols,
            'fe_coef': coef_fe,
            'fe_se': se_fe,
            'fe_p': p_fe,
            'fe_r2': r2_fe,
        })

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df['ols_stars'] = results_df['ols_p'].apply(lambda p: significance_stars(p) if pd.notna(p) else '')
        results_df['fe_stars'] = results_df['fe_p'].apply(lambda p: significance_stars(p) if pd.notna(p) else '')
        results_df['label'] = results_df['variable'].apply(get_var_label)
        results_path = os.path.join(tables_dir, 'political_outcomes_regressions.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"\n  Saved: {results_path}")


def _run_person_fe_regression(df: pd.DataFrame, outcome_var: str,
                               exposure_var: str) -> Tuple[float, float, float, float]:
    """
    Run within-person (fixed effects) regression.

    Uses linearmodels PanelOLS if available, otherwise demeans manually.

    Returns:
        (coefficient, standard_error, p_value, r_squared)
        Returns (NaN, NaN, NaN, NaN) if regression cannot be estimated.
    """
    panel = df[['idpers', 'year', outcome_var, exposure_var]].dropna().copy()

    # Need persons with variation in exposure
    person_var = panel.groupby('idpers')[exposure_var].std()
    persons_with_variation = person_var[person_var > 0].index
    panel_var = panel[panel['idpers'].isin(persons_with_variation)]

    if len(panel_var) < 10 or len(persons_with_variation) < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    if HAS_LINEARMODELS:
        try:
            panel_indexed = panel_var.set_index(['idpers', 'year'])
            # Add year dummies
            year_dummies = pd.get_dummies(panel_indexed.index.get_level_values('year'),
                                          prefix='yr', drop_first=True, dtype=float)
            year_dummies.index = panel_indexed.index
            exog = pd.concat([panel_indexed[[exposure_var]], year_dummies], axis=1)

            mod = PanelOLS(panel_indexed[outcome_var], exog, entity_effects=True,
                          check_rank=False)
            res = mod.fit(cov_type='clustered', cluster_entity=True)
            return (res.params[exposure_var], res.std_errors[exposure_var],
                    res.pvalues[exposure_var], res.rsquared_within)
        except Exception as e:
            logger.debug(f"  linearmodels PanelOLS failed: {e}, falling back to demeaned OLS")

    # Fallback: demeaned OLS
    try:
        # Demean by person
        for col in [outcome_var, exposure_var]:
            person_means = panel_var.groupby('idpers')[col].transform('mean')
            panel_var[f'{col}_dm'] = panel_var[col] - person_means

        # Add demeaned year dummies
        year_dummies = pd.get_dummies(panel_var['year'], prefix='yr', drop_first=True, dtype=float)
        year_dummies.index = panel_var.index
        panel_var = pd.concat([panel_var, year_dummies], axis=1)

        # Demean year dummies by person
        yd_cols = year_dummies.columns.tolist()
        for col_yd in yd_cols:
            pm = panel_var.groupby('idpers')[col_yd].transform('mean')
            panel_var[f'{col_yd}_dm'] = panel_var[col_yd] - pm

        yd_dm_cols = [f'{c}_dm' for c in yd_cols]
        X_dm = panel_var[[f'{exposure_var}_dm'] + yd_dm_cols]
        # No constant in demeaned FE regression (demeaning removes it by construction)
        y_dm = panel_var[f'{outcome_var}_dm']

        model = sm.OLS(y_dm, X_dm).fit(cov_type='cluster',
                                         cov_kwds={'groups': panel_var['idpers']})
        coef = model.params[f'{exposure_var}_dm']
        se = model.bse[f'{exposure_var}_dm']
        p = model.pvalues[f'{exposure_var}_dm']
        r2 = model.rsquared

        return (coef, se, p, r2)

    except Exception as e:
        logger.debug(f"  Demeaned OLS failed: {e}")
        return (np.nan, np.nan, np.nan, np.nan)


def _leftright_deep_dive(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """Deep dive into left-right self-placement (pp10) by exposure."""
    logger.info("\n--- 3.2: Left-Right Placement Deep Dive (pp10) ---")

    valid = df.loc[df['pp10'].notna()].copy()
    n_valid = len(valid)
    logger.info(f"  Valid observations: {n_valid:,}")

    # Distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (exp_val, label) in zip(axes, [(0, 'Not Exposed'), (1, 'Exposed')]):
        sub = valid.loc[valid['has_exposure'] == exp_val, 'pp10']
        if len(sub) > 0:
            ax.hist(sub, bins=np.arange(-0.5, 11.5, 1), density=True,
                    edgecolor='black', alpha=0.7)
            ax.axvline(sub.mean(), color='red', linestyle='--',
                       label=f'Mean: {sub.mean():.2f}')
            ax.set_xlabel('Left (0) - Right (10)')
            ax.set_ylabel('Density')
            ax.set_title(f'{label} (N={len(sub):,})')
            ax.legend()
            ax.set_xlim(-0.5, 10.5)

    fig.suptitle('Left-Right Self-Placement by AI Exposure Status', fontsize=14)
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, 'leftright_distribution_by_exposure.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    # Trend over time
    lr_by_year = valid.groupby(['year', 'has_exposure'])['pp10'].agg(
        ['mean', 'count', 'sem']
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    for exp_val, label, color in [(0, 'Not Exposed', 'steelblue'), (1, 'Exposed', 'coral')]:
        sub = lr_by_year[lr_by_year['has_exposure'] == exp_val]
        if len(sub) > 0:
            ax.plot(sub['year'], sub['mean'], marker='o', label=label, color=color)
            ax.fill_between(sub['year'],
                            sub['mean'] - 1.96 * sub['sem'],
                            sub['mean'] + 1.96 * sub['sem'],
                            alpha=0.2, color=color)
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Left-Right Placement')
    ax.set_title('Left-Right Self-Placement Trends by AI Exposure Status')
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, 'leftright_trends_by_exposure.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    lr_trends_path = os.path.join(tables_dir, 'leftright_trends_by_exposure.csv')
    lr_by_year.to_csv(lr_trends_path, index=False)
    logger.info(f"  Saved: {lr_trends_path}")


def _vote_choice_analysis(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """Analyze vote choice (pp19) by exposure. pp19 is CATEGORICAL - no means."""
    logger.info("\n--- 3.3: Vote Choice Analysis (pp19) ---")

    valid = df.loc[df['pp19'].notna()].copy()
    valid['pp19'] = valid['pp19'].astype(int)
    n_valid = len(valid)
    logger.info(f"  Valid observations: {n_valid:,}")

    if n_valid == 0:
        logger.warning("  No valid pp19 data, skipping")
        return

    # Party shares by exposure status (column-normalized crosstab)
    party_shares = pd.crosstab(
        valid['pp19'],
        valid['has_exposure'],
        normalize='columns'
    )
    party_counts = pd.crosstab(valid['pp19'], valid['has_exposure'])

    shares_path = os.path.join(tables_dir, 'vote_choice_shares_by_exposure.csv')
    party_shares.to_csv(shares_path)
    logger.info(f"  Saved: {shares_path}")

    counts_path = os.path.join(tables_dir, 'vote_choice_counts_by_exposure.csv')
    party_counts.to_csv(counts_path)
    logger.info(f"  Saved: {counts_path}")

    # Chi-square test
    if party_counts.shape[0] > 1 and party_counts.shape[1] > 1:
        chi2, p_val, dof, expected = stats.chi2_contingency(party_counts)
        logger.info(f"  Chi-square test: chi2={chi2:.4f}, p={p_val:.6f}, dof={dof}")

    # Bar chart of top party shares
    fig, ax = plt.subplots(figsize=(12, 7))
    # Show top 10 party codes by overall frequency
    top_parties = party_counts.sum(axis=1).nlargest(10).index
    plot_shares = party_shares.loc[party_shares.index.isin(top_parties)].copy()
    if plot_shares.shape[1] == 2:
        plot_shares.columns = ['Not Exposed', 'Exposed']
    plot_shares.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Party Code (pp19)')
    ax.set_ylabel('Share')
    ax.set_title('Vote Choice by AI Exposure Status (Top 10 Parties)')
    ax.legend(title='Exposure Status')
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    fig_path = os.path.join(figures_dir, 'vote_choice_by_exposure.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"  Saved: {fig_path}")

    # Vote choice changes conditional on exposure changes
    logger.info("  Vote choice transitions conditional on exposure change:")
    panel = valid[['idpers', 'year', 'pp19', 'has_exposure']].copy()
    panel = panel.sort_values(['idpers', 'year'])
    panel['pp19_lag'] = panel.groupby('idpers')['pp19'].shift(1)
    panel['exposure_lag'] = panel.groupby('idpers')['has_exposure'].shift(1)

    panel_valid = panel.dropna(subset=['pp19_lag', 'exposure_lag']).copy()
    panel_valid['party_changed'] = (panel_valid['pp19'] != panel_valid['pp19_lag']).astype(int)
    panel_valid['exposure_changed'] = panel_valid['has_exposure'] - panel_valid['exposure_lag']

    # Party switch rate by exposure change
    switch_rates = panel_valid.groupby(
        panel_valid['exposure_changed'].map({-1: 'Lost exposure', 0: 'No change', 1: 'Gained exposure'})
    )['party_changed'].agg(['mean', 'count']).reset_index()
    switch_rates.columns = ['exposure_transition', 'party_switch_rate', 'N']

    switch_path = os.path.join(tables_dir, 'vote_switch_rate_by_exposure_change.csv')
    switch_rates.to_csv(switch_path, index=False)
    logger.info(f"  Saved: {switch_path}")

    for _, row in switch_rates.iterrows():
        logger.info(f"    {row['exposure_transition']}: switch rate={row['party_switch_rate']:.3f}, "
                    f"N={row['N']:,.0f}")


# =============================================================================
# Section 4: Panel Analysis
# =============================================================================

def section_4_panel_analysis(df: pd.DataFrame, output_dir: str):
    """
    Panel-level analysis: switchers, event studies, within-person variation.

    Produces:
      - Switcher identification
      - Event study around first exposure
      - Within-person variation statistics
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 4: PANEL ANALYSIS")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    # -------------------------------------------------------------------------
    # 4.1: Identify switchers
    # -------------------------------------------------------------------------
    logger.info("\n--- 4.1: Identifying Switchers ---")

    person_exposure = df.groupby('idpers').agg(
        n_years=('year', 'count'),
        n_exposed_years=('has_exposure', 'sum'),
        ever_exposed=('exposed_ever', 'max'),
        min_year=('year', 'min'),
        max_year=('year', 'max'),
    ).reset_index()

    # Switchers: persons who have SOME years exposed and SOME years not
    person_exposure['is_switcher'] = (
        (person_exposure['n_exposed_years'] > 0) &
        (person_exposure['n_exposed_years'] < person_exposure['n_years'])
    ).astype(int)

    # Always exposed (in all observed years)
    person_exposure['always_exposed'] = (
        (person_exposure['n_exposed_years'] > 0) &
        (person_exposure['n_exposed_years'] == person_exposure['n_years'])
    ).astype(int)

    # Never exposed
    person_exposure['never_exposed'] = (person_exposure['n_exposed_years'] == 0).astype(int)

    n_switchers = person_exposure['is_switcher'].sum()
    n_always = person_exposure['always_exposed'].sum()
    n_never = person_exposure['never_exposed'].sum()
    n_total_persons = len(person_exposure)

    logger.info(f"  Total persons: {n_total_persons:,}")
    logger.info(f"  Switchers (some exposed, some not): {n_switchers:,} ({100*n_switchers/n_total_persons:.2f}%)")
    logger.info(f"  Always exposed: {n_always:,}")
    logger.info(f"  Never exposed: {n_never:,}")

    person_summary_path = os.path.join(tables_dir, 'person_exposure_summary.csv')
    person_exposure.to_csv(person_summary_path, index=False)
    logger.info(f"  Saved: {person_summary_path}")

    # Summary table
    switcher_summary = pd.DataFrame([{
        'category': 'Never exposed',
        'N_persons': n_never,
        'pct': 100 * n_never / n_total_persons,
    }, {
        'category': 'Switcher',
        'N_persons': n_switchers,
        'pct': 100 * n_switchers / n_total_persons,
    }, {
        'category': 'Always exposed',
        'N_persons': n_always,
        'pct': 100 * n_always / n_total_persons,
    }])
    switcher_path = os.path.join(tables_dir, 'switcher_categories.csv')
    switcher_summary.to_csv(switcher_path, index=False)
    logger.info(f"  Saved: {switcher_path}")

    # -------------------------------------------------------------------------
    # 4.2: Event study around first exposure
    # -------------------------------------------------------------------------
    logger.info("\n--- 4.2: Event Study (First Exposure) ---")

    _event_study(df, person_exposure, tables_dir, figures_dir)

    # -------------------------------------------------------------------------
    # 4.3: Within-person variation statistics
    # -------------------------------------------------------------------------
    logger.info("\n--- 4.3: Within-Person Variation ---")

    _within_person_variation(df, tables_dir)


def _event_study(df: pd.DataFrame, person_exposure: pd.DataFrame,
                 tables_dir: str, figures_dir: str):
    """
    Event study: outcomes before/after first exposure year.
    """
    # Identify first year of exposure for each person
    exposed_persons = df.loc[df['has_exposure'] == 1, ['idpers', 'year']].copy()
    if len(exposed_persons) == 0:
        logger.warning("  No exposed person-years, skipping event study")
        return

    first_exposure = exposed_persons.groupby('idpers')['year'].min().reset_index()
    first_exposure.columns = ['idpers', 'first_exposure_year']

    # Merge back to full panel
    df_event = df.merge(first_exposure, on='idpers', how='inner')
    df_event['event_time'] = df_event['year'] - df_event['first_exposure_year']

    # Restrict to reasonable event window
    event_window = 5  # +/- 5 years around first exposure
    df_event = df_event[(df_event['event_time'] >= -event_window) &
                        (df_event['event_time'] <= event_window)]

    logger.info(f"  Persons with first exposure: {first_exposure['idpers'].nunique():,}")
    logger.info(f"  Event window: [{-event_window}, +{event_window}] years")
    logger.info(f"  Person-years in event window: {len(df_event):,}")

    # Outcome variables for event study
    event_outcomes = {}
    if 'iwyn' in df.columns:
        event_outcomes['iwyn'] = 'Income (iwyn)'
    for var, label in POLITICAL_OUTCOMES_CONTINUOUS.items():
        if var in df.columns:
            event_outcomes[var] = label

    for var, label in event_outcomes.items():
        valid_event = df_event.loc[df_event[var].notna()]
        if len(valid_event) < 20:
            continue

        event_means = valid_event.groupby('event_time')[var].agg(
            ['mean', 'count', 'sem']
        ).reset_index()

        # Save table
        event_table_path = os.path.join(tables_dir, f'event_study_{var}.csv')
        event_means.to_csv(event_table_path, index=False)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(event_means['event_time'], event_means['mean'], marker='o', color='steelblue')
        ax.fill_between(
            event_means['event_time'],
            event_means['mean'] - 1.96 * event_means['sem'],
            event_means['mean'] + 1.96 * event_means['sem'],
            alpha=0.2, color='steelblue'
        )
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='First exposure')
        ax.set_xlabel('Years Relative to First AI Exposure')
        ax.set_ylabel(f'Mean {label}')
        ax.set_title(f'Event Study: {label} Around First AI Exposure')
        ax.legend()

        # Annotate sample sizes
        for _, row_data in event_means.iterrows():
            ax.annotate(f"n={int(row_data['count'])}",
                       (row_data['event_time'], row_data['mean']),
                       textcoords='offset points', xytext=(0, 10),
                       fontsize=7, ha='center', alpha=0.6)

        fig.tight_layout()
        fig_path = os.path.join(figures_dir, f'event_study_{var}.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Event study {var}: saved {fig_path}")


def _within_person_variation(df: pd.DataFrame, tables_dir: str):
    """Compute within-person variation statistics for exposure and outcomes."""

    # Exposure within-person variation
    person_exp_stats = df.groupby('idpers')[PRIMARY_EXPOSURE].agg(
        ['mean', 'std', 'min', 'max', 'count']
    ).reset_index()
    person_exp_stats.columns = ['idpers', 'mean_exposure', 'sd_exposure',
                                 'min_exposure', 'max_exposure', 'n_years']

    # Focus on persons with any exposure
    with_exposure = person_exp_stats[person_exp_stats['max_exposure'] > 0]
    n_with_var = (with_exposure['sd_exposure'] > 0).sum()

    logger.info(f"  Persons with any exposure: {len(with_exposure):,}")
    logger.info(f"  Persons with within-person variation in exposure: {n_with_var:,}")

    if len(with_exposure) > 0:
        logger.info(f"  Mean within-person sd of exposure: {with_exposure['sd_exposure'].mean():.6f}")
        logger.info(f"  Mean within-person range: {(with_exposure['max_exposure'] - with_exposure['min_exposure']).mean():.6f}")

    # Within-person variation for outcomes
    outcome_vars_for_variation = ['iwyn']
    for var in POLITICAL_OUTCOMES_CONTINUOUS:
        if var in df.columns:
            outcome_vars_for_variation.append(var)

    variation_rows = []
    for var in outcome_vars_for_variation:
        if var not in df.columns:
            continue

        person_var_stats = df.groupby('idpers')[var].agg(['mean', 'std', 'count'])

        # Total variance
        total_var = df[var].dropna().var()
        # Between-person variance
        between_var = person_var_stats['mean'].var()
        # Within-person variance (average of individual variances)
        within_var = (person_var_stats['std'] ** 2).mean()

        if total_var > 0:
            # Note: for unbalanced panels, between + within may not equal total
            pct_b = 100 * between_var / total_var
            pct_w = 100 * within_var / total_var
            variation_rows.append({
                'variable': var,
                'total_variance': total_var,
                'between_person_variance': between_var,
                'within_person_variance': within_var,
                'pct_between': pct_b,
                'pct_within': pct_w,
                'pct_sum': pct_b + pct_w,
                'note': 'unbalanced panel: pct_between + pct_within may differ from 100%',
            })
            logger.info(f"  {var}: between={pct_b:.1f}%, within={pct_w:.1f}% "
                        f"(sum={pct_b + pct_w:.1f}%)")

    if variation_rows:
        var_df = pd.DataFrame(variation_rows)
        var_path = os.path.join(tables_dir, 'within_person_variance_decomposition.csv')
        var_df.to_csv(var_path, index=False)
        logger.info(f"  Saved: {var_path}")


# =============================================================================
# Section 5: Intensive Margin Analysis
# =============================================================================

def section_5_intensive_margin(df: pd.DataFrame, output_dir: str, job_security_var: Optional[str] = None):
    """
    Intensive margin analysis: among exposed workers, how does the LEVEL of
    AI exposure relate to outcomes?

    This complements the extensive margin (binary exposed/not) in Sections 2-4
    by exploiting continuous variation in hampole_ai_exposure_avg.
    """
    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    logger.info("\n" + "=" * 70)
    logger.info("SECTION 5: INTENSIVE MARGIN ANALYSIS")
    logger.info("=" * 70)

    # Restrict to person-years with non-zero exposure
    df_exposed = df.loc[df[PRIMARY_EXPOSURE] > 0].copy()
    n_exposed = len(df_exposed)
    n_persons_exposed = df_exposed['idpers'].nunique()
    logger.info(f"  Working sample: {n_exposed:,} person-years, {n_persons_exposed:,} persons")
    logger.info(f"  Exposure range: {df_exposed[PRIMARY_EXPOSURE].min():.4f} - "
                f"{df_exposed[PRIMARY_EXPOSURE].max():.4f}")
    logger.info(f"  Exposure mean: {df_exposed[PRIMARY_EXPOSURE].mean():.4f}, "
                f"sd: {df_exposed[PRIMARY_EXPOSURE].std():.4f}")

    if n_exposed < 50:
        logger.warning("  Insufficient exposed observations for intensive margin analysis. Skipping.")
        return

    # -------------------------------------------------------------------------
    # 5.1: Exposure quantile descriptives
    # -------------------------------------------------------------------------
    logger.info("\n--- 5.1: Exposure Quantile Analysis ---")

    # Create exposure terciles among exposed
    df_exposed['exposure_tercile'] = pd.qcut(
        df_exposed[PRIMARY_EXPOSURE], q=3, labels=['Low', 'Medium', 'High'],
        duplicates='drop'
    )

    # Outcome means by tercile
    continuous_outcomes = ['iwyn', 'pp10', 'pp13', 'pp17', 'pp22', 'pp16', 'pp15']
    if job_security_var and job_security_var in df_exposed.columns:
        continuous_outcomes.append(job_security_var)

    outcome_labels = {
        'iwyn': 'Income (net)',
        'pp10': 'Left-right (0-10)',
        'pp13': 'Social benefits pref.',
        'pp17': 'Taxes high income pref.',
        'pp22': 'Gender equality pref.',
        'pp16': 'Environment pref.',
        'pp15': 'Foreigners equal chances',
    }
    if job_security_var:
        outcome_labels[job_security_var] = 'Job security'

    tercile_rows = []
    for var in continuous_outcomes:
        if var not in df_exposed.columns:
            continue
        for tercile in ['Low', 'Medium', 'High']:
            sub = df_exposed.loc[df_exposed['exposure_tercile'] == tercile, var].dropna()
            if len(sub) > 0:
                tercile_rows.append({
                    'variable': var,
                    'label': outcome_labels.get(var, var),
                    'tercile': tercile,
                    'N': len(sub),
                    'mean': sub.mean(),
                    'sd': sub.std(),
                    'median': sub.median(),
                })

    if tercile_rows:
        tercile_df = pd.DataFrame(tercile_rows)
        tercile_path = os.path.join(tables_dir, 'intensive_margin_tercile_means.csv')
        tercile_df.to_csv(tercile_path, index=False)
        logger.info(f"  Saved: {tercile_path}")

        # Log key comparisons
        for var in continuous_outcomes:
            var_data = tercile_df[tercile_df['variable'] == var]
            if len(var_data) == 3:
                low_mean = var_data.loc[var_data['tercile'] == 'Low', 'mean'].values[0]
                high_mean = var_data.loc[var_data['tercile'] == 'High', 'mean'].values[0]
                logger.info(f"    {var}: Low tercile={low_mean:.3f}, High tercile={high_mean:.3f}, "
                            f"diff={high_mean - low_mean:.3f}")

    # Tercile bar chart for income
    if 'iwyn' in df_exposed.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        income_by_tercile = df_exposed.groupby('exposure_tercile')['iwyn'].agg(['mean', 'sem']).reindex(
            ['Low', 'Medium', 'High']
        )
        bars = ax.bar(income_by_tercile.index, income_by_tercile['mean'],
                       yerr=1.96 * income_by_tercile['sem'],
                       capsize=5, color=['#4575b4', '#ffffbf', '#d73027'],
                       edgecolor='black', alpha=0.85)
        ax.set_xlabel('AI Exposure Tercile (among exposed)')
        ax.set_ylabel('Mean Yearly Income (CHF)')
        ax.set_title('Income by AI Exposure Intensity')
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, 'intensive_income_by_tercile.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")

    # -------------------------------------------------------------------------
    # 5.2: Continuous regressions (among exposed only)
    # -------------------------------------------------------------------------
    logger.info("\n--- 5.2: Continuous Regressions (Exposed Only) ---")

    regression_rows = []
    for var in continuous_outcomes:
        if var not in df_exposed.columns:
            continue

        valid = df_exposed[[PRIMARY_EXPOSURE, var, 'idpers', 'year']].dropna()
        n_valid = len(valid)
        if n_valid < 30:
            logger.info(f"  {var}: insufficient observations ({n_valid}), skipping")
            continue

        # --- OLS with year FE (among exposed) ---
        try:
            year_dummies = pd.get_dummies(valid['year'].astype(str), prefix='yr',
                                           drop_first=True, dtype=float)
            X_ols = pd.concat([valid[[PRIMARY_EXPOSURE]].reset_index(drop=True),
                                year_dummies.reset_index(drop=True)], axis=1)
            X_ols = sm.add_constant(X_ols)
            y_ols = valid[var].reset_index(drop=True).astype(float)

            model_ols = sm.OLS(y_ols, X_ols).fit(cov_type='HC1')
            coef_ols = model_ols.params[PRIMARY_EXPOSURE]
            se_ols = model_ols.bse[PRIMARY_EXPOSURE]
            p_ols = model_ols.pvalues[PRIMARY_EXPOSURE]
            r2_ols = model_ols.rsquared
            logger.info(f"  {get_var_label(var)} OLS (exposed only): coef={coef_ols:.4f}, "
                        f"se={se_ols:.4f}, p={p_ols:.4f}{significance_stars(p_ols)}, N={n_valid}")
        except Exception as e:
            logger.warning(f"  {get_var_label(var)} OLS failed: {e}")
            coef_ols = se_ols = p_ols = r2_ols = np.nan

        # --- Person FE (among exposed) ---
        coef_fe, se_fe, p_fe, r2_fe = _run_person_fe_regression(
            valid, var, PRIMARY_EXPOSURE
        )
        if not np.isnan(coef_fe):
            logger.info(f"  {get_var_label(var)} Person FE (exposed only): coef={coef_fe:.4f}, "
                        f"se={se_fe:.4f}, p={p_fe:.4f}{significance_stars(p_fe)}")
        else:
            logger.info(f"  {get_var_label(var)} Person FE (exposed only): insufficient within-person variation")

        regression_rows.append({
            'variable': var,
            'label': outcome_labels.get(var, var),
            'sample': 'exposed_only',
            'N': n_valid,
            'ols_coef': coef_ols,
            'ols_se': se_ols,
            'ols_p': p_ols,
            'ols_r2': r2_ols,
            'fe_coef': coef_fe,
            'fe_se': se_fe,
            'fe_p': p_fe,
            'fe_r2': r2_fe,
        })

    # -------------------------------------------------------------------------
    # 5.3: Full sample continuous regressions
    # -------------------------------------------------------------------------
    logger.info("\n--- 5.3: Continuous Regressions (Full Sample) ---")

    for var in continuous_outcomes:
        if var not in df.columns:
            continue

        valid = df[[PRIMARY_EXPOSURE, var, 'idpers', 'year']].dropna().copy()
        # Fill NaN exposure with 0 (unexposed)
        valid[PRIMARY_EXPOSURE] = valid[PRIMARY_EXPOSURE].fillna(0)
        n_valid = len(valid)
        if n_valid < 30:
            continue

        # --- OLS with year FE (full sample, continuous exposure) ---
        try:
            year_dummies = pd.get_dummies(valid['year'].astype(str), prefix='yr',
                                           drop_first=True, dtype=float)
            X_ols = pd.concat([valid[[PRIMARY_EXPOSURE]].reset_index(drop=True),
                                year_dummies.reset_index(drop=True)], axis=1)
            X_ols = sm.add_constant(X_ols)
            y_ols = valid[var].reset_index(drop=True).astype(float)

            model_ols = sm.OLS(y_ols, X_ols).fit(cov_type='HC1')
            coef_ols = model_ols.params[PRIMARY_EXPOSURE]
            se_ols = model_ols.bse[PRIMARY_EXPOSURE]
            p_ols = model_ols.pvalues[PRIMARY_EXPOSURE]
            r2_ols = model_ols.rsquared
            logger.info(f"  {get_var_label(var)} OLS (full sample): coef={coef_ols:.4f}, "
                        f"se={se_ols:.4f}, p={p_ols:.4f}{significance_stars(p_ols)}, N={n_valid}")
        except Exception as e:
            logger.warning(f"  {get_var_label(var)} OLS failed: {e}")
            coef_ols = se_ols = p_ols = r2_ols = np.nan

        # --- Person FE (full sample, continuous) ---
        coef_fe, se_fe, p_fe, r2_fe = _run_person_fe_regression(
            valid, var, PRIMARY_EXPOSURE
        )
        if not np.isnan(coef_fe):
            logger.info(f"  {get_var_label(var)} Person FE (full sample): coef={coef_fe:.4f}, "
                        f"se={se_fe:.4f}, p={p_fe:.4f}{significance_stars(p_fe)}")
        else:
            logger.info(f"  {get_var_label(var)} Person FE (full sample): insufficient within-person variation")

        regression_rows.append({
            'variable': var,
            'label': outcome_labels.get(var, var),
            'sample': 'full_sample',
            'N': n_valid,
            'ols_coef': coef_ols,
            'ols_se': se_ols,
            'ols_p': p_ols,
            'ols_r2': r2_ols,
            'fe_coef': coef_fe,
            'fe_se': se_fe,
            'fe_p': p_fe,
            'fe_r2': r2_fe,
        })

    if regression_rows:
        reg_df = pd.DataFrame(regression_rows)
        # Add significance stars and human-readable labels
        reg_df['ols_stars'] = reg_df['ols_p'].apply(lambda p: significance_stars(p) if pd.notna(p) else '')
        reg_df['fe_stars'] = reg_df['fe_p'].apply(lambda p: significance_stars(p) if pd.notna(p) else '')
        reg_df['label'] = reg_df['variable'].apply(get_var_label)
        reg_path = os.path.join(tables_dir, 'intensive_margin_regressions.csv')
        reg_df.to_csv(reg_path, index=False)
        logger.info(f"\n  Saved: {reg_path}")

    # -------------------------------------------------------------------------
    # 5.4: Scatter plots — outcome vs continuous exposure
    # -------------------------------------------------------------------------
    logger.info("\n--- 5.4: Scatter/Binscatter Plots ---")

    key_outcomes = ['iwyn', 'pp10']
    if job_security_var and job_security_var in df_exposed.columns:
        key_outcomes.append(job_security_var)

    for var in key_outcomes:
        if var not in df_exposed.columns:
            continue
        plot_data = df_exposed[[PRIMARY_EXPOSURE, var]].dropna()
        if len(plot_data) < 20:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: raw scatter with LOESS/regression line
        ax1 = axes[0]
        ax1.scatter(plot_data[PRIMARY_EXPOSURE], plot_data[var],
                    alpha=0.15, s=8, color='steelblue')
        # Add OLS fit line
        z = np.polyfit(plot_data[PRIMARY_EXPOSURE], plot_data[var], 1)
        x_line = np.linspace(plot_data[PRIMARY_EXPOSURE].min(),
                              plot_data[PRIMARY_EXPOSURE].max(), 100)
        ax1.plot(x_line, np.polyval(z, x_line), color='red', linewidth=2,
                 label=f'OLS slope={z[0]:.2f}')
        ax1.set_xlabel('AI Exposure (continuous)')
        ax1.set_ylabel(outcome_labels.get(var, var))
        ax1.set_title(f'{outcome_labels.get(var, var)} vs AI Exposure')
        ax1.legend()

        # Right panel: binscatter (decile means)
        ax2 = axes[1]
        plot_data['exposure_decile'] = pd.qcut(
            plot_data[PRIMARY_EXPOSURE], q=10, duplicates='drop'
        )
        bin_means = plot_data.groupby('exposure_decile').agg(
            exposure_mean=(PRIMARY_EXPOSURE, 'mean'),
            outcome_mean=(var, 'mean'),
            outcome_se=(var, 'sem'),
            n=(var, 'count')
        ).reset_index()
        ax2.errorbar(bin_means['exposure_mean'], bin_means['outcome_mean'],
                      yerr=1.96 * bin_means['outcome_se'],
                      fmt='o-', capsize=4, color='coral', markersize=8)
        ax2.set_xlabel('AI Exposure (decile mean)')
        ax2.set_ylabel(f'Mean {outcome_labels.get(var, var)}')
        ax2.set_title(f'{outcome_labels.get(var, var)} — Binscatter by Exposure Decile')

        fig.tight_layout()
        fig_path = os.path.join(figures_dir, f'intensive_scatter_{var}.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")

    # -------------------------------------------------------------------------
    # 5.5: Within-person exposure intensity changes
    # -------------------------------------------------------------------------
    logger.info("\n--- 5.5: Within-Person Exposure Intensity Changes ---")

    panel = df_exposed[['idpers', 'year', PRIMARY_EXPOSURE, 'iwyn']].copy()
    panel = panel.sort_values(['idpers', 'year'])
    panel['exposure_lag'] = panel.groupby('idpers')[PRIMARY_EXPOSURE].shift(1)
    panel['exposure_change'] = panel[PRIMARY_EXPOSURE] - panel['exposure_lag']
    panel['income_change'] = panel.groupby('idpers')['iwyn'].diff()

    panel_valid = panel.dropna(subset=['exposure_change', 'income_change']).copy()
    n_transitions = len(panel_valid)
    logger.info(f"  Person-year transitions with both exposure and income change: {n_transitions:,}")

    if n_transitions >= 20:
        # Correlation between exposure change and income change
        corr, p_corr = stats.pearsonr(panel_valid['exposure_change'],
                                       panel_valid['income_change'])
        logger.info(f"  Correlation(delta_exposure, delta_income): r={corr:.4f}, p={p_corr:.4f}")

        # Binscatter: income change vs exposure change
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create bins of exposure change
        panel_valid['change_bin'] = pd.qcut(
            panel_valid['exposure_change'].rank(method='first'), q=10, duplicates='drop'
        )
        bin_means = panel_valid.groupby('change_bin').agg(
            exp_change_mean=('exposure_change', 'mean'),
            inc_change_mean=('income_change', 'mean'),
            inc_change_se=('income_change', 'sem'),
            n=('income_change', 'count')
        ).reset_index()

        ax.errorbar(bin_means['exp_change_mean'], bin_means['inc_change_mean'],
                      yerr=1.96 * bin_means['inc_change_se'],
                      fmt='o-', capsize=4, color='steelblue', markersize=8)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Change in AI Exposure')
        ax.set_ylabel('Change in Income (CHF)')
        ax.set_title(f'Income Change vs Exposure Change (r={corr:.3f}, N={n_transitions:,})')
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, 'intensive_delta_income_vs_delta_exposure.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")

        # Save summary table
        change_summary = panel_valid.groupby(
            panel_valid['exposure_change'].apply(
                lambda x: 'Decreased' if x < -0.01 else ('Increased' if x > 0.01 else 'Stable')
            )
        ).agg(
            N=('income_change', 'count'),
            mean_income_change=('income_change', 'mean'),
            median_income_change=('income_change', 'median'),
            mean_exposure_change=('exposure_change', 'mean'),
        ).reset_index()
        change_summary.columns = ['exposure_direction', 'N', 'mean_income_change',
                                   'median_income_change', 'mean_exposure_change']
        change_path = os.path.join(tables_dir, 'intensive_exposure_income_changes.csv')
        change_summary.to_csv(change_path, index=False)
        logger.info(f"  Saved: {change_path}")

        for _, row in change_summary.iterrows():
            logger.info(f"    {row['exposure_direction']}: N={row['N']:,}, "
                        f"mean income change={row['mean_income_change']:,.0f} CHF")


# =============================================================================
# Summary
# =============================================================================

def print_summary(df: pd.DataFrame, output_dir: str):
    """Print final summary of the EDA to console."""
    logger.info("\n" + "=" * 70)
    logger.info("EDA SUMMARY")
    logger.info("=" * 70)

    n_total = len(df)
    n_persons = df['idpers'].nunique()
    n_exposed_py = df['has_exposure'].sum()
    n_exposed_persons = df.loc[df['has_exposure'] == 1, 'idpers'].nunique()
    year_min = df['year'].min()
    year_max = df['year'].max()

    logger.info(f"  Total person-years: {n_total:,}")
    logger.info(f"  Unique persons: {n_persons:,}")
    logger.info(f"  Year range: {year_min}-{year_max}")
    logger.info(f"  Person-years with non-zero exposure: {n_exposed_py:,} ({100*n_exposed_py/n_total:.2f}%)")
    logger.info(f"  Persons ever exposed: {n_exposed_persons:,} ({100*n_exposed_persons/n_persons:.2f}%)")
    logger.info(f"  Output directory: {output_dir}")

    # Count output files
    n_tables = len([f for f in os.listdir(os.path.join(output_dir, 'tables'))
                    if f.endswith('.csv')])
    n_figures = len([f for f in os.listdir(os.path.join(output_dir, 'figures'))
                     if f.endswith('.png')])
    logger.info(f"  Tables generated: {n_tables}")
    logger.info(f"  Figures generated: {n_figures}")
    logger.info("=" * 70)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_single_level(df: pd.DataFrame, output_dir: str, level_code: str,
                      job_security_var: Optional[str] = None):
    """
    Run the full EDA for one exposure level.

    Sets PRIMARY_EXPOSURE to the appropriate suffixed column, recreates
    has_exposure/exposed_ever indicators, and runs all analysis sections.

    Args:
        df: Full SHP DataFrame (loaded once, reused across levels)
        output_dir: Base output directory (level subfolder will be created)
        level_code: Exposure level code (e.g., 'foy', 'oy', 'f')
        job_security_var: Job security variable name
    """
    global PRIMARY_EXPOSURE

    suffix = EXPOSURE_LEVEL_SUFFIXES[level_code]
    col_name = f'hampole_ai_exposure_avg{suffix}'
    label = EXPOSURE_LEVEL_LABELS[level_code]

    # Validate column exists
    if col_name not in df.columns:
        logger.warning(f"Skipping level '{level_code}' ({label}): column '{col_name}' not found in data")
        return

    # Set the module-level PRIMARY_EXPOSURE for this run
    PRIMARY_EXPOSURE = col_name

    # Create output subdirectory for this level
    level_dir = os.path.join(output_dir, level_code)
    tables_dir = os.path.join(level_dir, 'tables')
    figures_dir = os.path.join(level_dir, 'figures')
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Add file handler for this level's log
    log_path = os.path.join(level_dir, 'stage_7_shp_eda.log')
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info(f"STAGE 7 SHP EDA — Level: {label} ({level_code})")
    logger.info(f"Exposure column: {col_name}")
    logger.info("=" * 70)

    # Recreate exposure indicators for this level
    df['has_exposure'] = (df[PRIMARY_EXPOSURE].fillna(0) > 0).astype(int)
    n_exposed = df['has_exposure'].sum()
    logger.info(f"  has_exposure=1: {n_exposed:,} person-years ({100*n_exposed/len(df):.2f}%)")

    ever_exposed = df.loc[df['has_exposure'] == 1, 'idpers'].unique()
    df['exposed_ever'] = df['idpers'].isin(ever_exposed).astype(int)
    n_ever = df['exposed_ever'].sum()
    logger.info(f"  exposed_ever=1: {n_ever:,} person-years from {len(ever_exposed):,} persons")

    # Run all sections
    section_1_descriptive_stats(df, level_dir, job_security_var)
    section_2_economic_outcomes(df, level_dir, job_security_var)
    section_3_political_outcomes(df, level_dir)
    section_4_panel_analysis(df, level_dir)
    section_5_intensive_margin(df, level_dir, job_security_var)
    print_summary(df, level_dir)

    logger.info(f"\nLevel '{level_code}' ({label}) complete.")

    # Remove the file handler so next level gets its own
    logger.removeHandler(fh)
    fh.close()


def run_pipeline(args):
    """Execute the full SHP EDA pipeline for one or more exposure levels."""

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE 7 SHP: Exploratory Data Analysis")
    logger.info("=" * 70)
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Exposure levels: {args.exposure_levels}")

    # -------------------------------------------------------------------------
    # Step 1: Load and clean data (once, shared across all levels)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOAD AND CLEAN DATA")
    logger.info("=" * 70)

    df = load_and_clean_data(args.input)

    # Determine job security variable
    job_security_var = _select_job_security_variable(df, log=False)

    # -------------------------------------------------------------------------
    # Step 2: Run EDA for each requested exposure level
    # -------------------------------------------------------------------------
    for level_code in args.exposure_levels:
        if level_code not in EXPOSURE_LEVEL_SUFFIXES:
            logger.error(f"Unknown exposure level: '{level_code}'. "
                         f"Valid levels: {list(EXPOSURE_LEVEL_SUFFIXES.keys())}")
            continue

        logger.info("\n" + "#" * 70)
        logger.info(f"# RUNNING LEVEL: {level_code} ({EXPOSURE_LEVEL_LABELS[level_code]})")
        logger.info("#" * 70)

        run_single_level(df, args.output_dir, level_code, job_security_var)

    logger.info("\n" + "=" * 70)
    logger.info("ALL LEVELS COMPLETE")
    logger.info("=" * 70)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command-line arguments."""
    base_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Stage 7 SHP: Exploratory Data Analysis of AI Exposure and Individual Outcomes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python3 stage_7_shp_eda.py \\
      --input Data/shp_exposure/shp_exposure_isco4d_fallback.csv \\
      --output-dir Data/shp_eda/

  # With absolute paths
  python3 stage_7_shp_eda.py \\
      --input /path/to/shp_exposure_isco4d_fallback.csv \\
      --output-dir /path/to/shp_eda/
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default=str(base_dir / 'Data' / 'shp_exposure' / 'shp_exposure_isco4d_fallback.csv'),
        help='Path to SHP exposure fallback file (default: Data/shp_exposure/shp_exposure_isco4d_fallback.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(base_dir / 'Data' / 'shp_eda'),
        help='Output directory for tables and figures (default: Data/shp_eda/)'
    )

    parser.add_argument(
        '--exposure-levels',
        type=str,
        nargs='+',
        default=['foy'],
        help='Exposure level(s) to analyze. Options: foy, fo, oy, o, fy, f, all. '
             'Use "all" for all 6 levels. Can specify multiple: --exposure-levels foy oy fy. '
             '(default: foy)'
    )

    args = parser.parse_args()

    # Expand "all" to all 6 levels
    if 'all' in args.exposure_levels:
        args.exposure_levels = list(EXPOSURE_LEVEL_SUFFIXES.keys())

    return args


if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)
