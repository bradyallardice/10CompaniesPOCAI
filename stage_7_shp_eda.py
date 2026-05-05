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

Analysis Sections:
  1. Data Structure & Diagnostics
  2. Pairwise Exploration (Descriptive Only)
  3. Structured Regression Grid (Models A/B/C)
  4. Collective Summary (coefficient heatmaps, sign consistency)
  5. Multiple Testing Adjustment (FDR correction)
  6. Intensive Margin (among exposed only)
  7. Panel Dynamics (switchers, event studies)
  8. Stacked Long-Difference (causal, firm AI adoption)

Usage:
    python3 stage_7_shp_eda.py \\
        --output-dir Data/shp_eda/

    # Use fallback file (hierarchical 4d->3d->2d ISCO matching)
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
    from statsmodels.stats.multitest import multipletests
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

# Exposure level definitions: maps level code -> column suffix from Stage 6 SHP
EXPOSURE_LEVEL_SUFFIXES = {
    'foy': '_foy',    # Firm x Occupation x Year
    'fo':  '_fo',     # Firm x Occupation (time-invariant)
    'oy':  '_oy',     # Occupation x Year
    'o':   '_o',      # Occupation (time-invariant)
    'fy':  '_fy',     # Firm x Year
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

# Selectable base exposure metrics: shorthand -> column base name (level suffix appended separately).
# These are a 2x2: {hampole, binary} x {intensity-adjusted, not intensity-adjusted}.
# The level dimension (foy, oy, etc.) is controlled by --exposure-levels and is orthogonal.
EXPOSURE_METRIC_BASES = {
    'hampole':      'hampole_ai_exposure_avg',     # Hampole share × log(1+N_apps) intensity
    'hampole_base': 'hampole_occupation_exposure',  # Hampole share only (pre-intensity)
    'binary':       'binary_ai_exposure_avg',       # Binary exposure × log(1+N_apps) intensity
    'binary_base':  'binary_occupation_exposure',   # Binary exposure only (pre-intensity)
}

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
    'iwyn': 'Yearly work income, net (CHF, higher=more)',
    'wstat': 'Working/employment status',
    'pw86': 'Job insecurity (1=not worried ... 4=very worried, higher=less secure)',
    'pw86a': 'Job insecurity (1=not worried ... 4=very worried, higher=less secure)',
    'pp10': 'Left-right placement (1=left ... 10=right, higher=more right)',
    'pp13': 'Social benefits pref. (1=less ... 3=more, higher=more pro-welfare)',
    'pp17': 'Tax high incomes pref. (1=reduce ... 3=increase, higher=more redistributive)',
    'pp22': 'Gender equality pref. (1=gone too far ... 10=not enough, higher=more pro-equality)',
    'pp16': 'Environment vs. growth (1=protect env ... 3=promote growth, higher=more pro-growth)',
    'pp15': 'Chances for foreigners (1=equal chances ... 3=better for Swiss, higher=more nativist)',
    'pp19': 'Vote choice / party preference',
    'age': 'Age',
    'sex': 'Sex (0=male, 1=female)',
    'educat': 'Education level',
    'edcat': 'Education category',
}


SHORT_VARIABLE_LABELS = {
    'iwyn':  'Work income',
    'wstat': 'Employment status',
    'pw86':  'Job insecurity',
    'pw86a': 'Job insecurity',
    'pp10':  'Left-right',
    'pp13':  'Pro-welfare',
    'pp17':  'Redistributive',
    'pp22':  'Gender equality',
    'pp16':  'Pro-growth',
    'pp15':  'Anti-immigrant',
    'pp19':  'Party preference',
    'age':   'Age',
    'sex':   'Sex',
}


def get_var_label(var: str) -> str:
    """Return full human-readable label for a variable, or the variable name itself."""
    return VARIABLE_LABELS.get(var, var)


def get_var_short_label(var: str) -> str:
    """Return concise label for use in graph titles and axes."""
    return SHORT_VARIABLE_LABELS.get(var, var)


# Economic outcome variables
ECONOMIC_OUTCOMES = {
    'iwyn': 'Yearly work income, net (CHF, higher=more)',
    'wstat': 'Working/employment status',
}

# Job security: pw86a exists in some SHP versions with more waves; pw86 is the fallback
JOB_SECURITY_CANDIDATES = ['pw86a', 'pw86']

# Political outcome variables treated as continuous in OLS/FE regressions.
# pp10 and pp22 are 10-point scales — treating as continuous is standard in
# applied economics. pp13, pp17, pp16, pp15 are 3-point scales — OLS is a
# reasonable first approximation but ordered logit/probit would be more
# appropriate; coefficients should be interpreted with that caveat.
POLITICAL_OUTCOMES_CONTINUOUS = {
    'pp10': 'Left-right placement (1=left ... 10=right, higher=more right)',
    'pp13': 'Social benefits pref. (1=less ... 3=more, higher=more pro-welfare) [3-cat ordinal]',
    'pp17': 'Tax high incomes pref. (1=reduce ... 3=increase, higher=more redistributive) [3-cat ordinal]',
    'pp22': 'Gender equality pref. (1=gone too far ... 10=not enough, higher=more pro-equality)',
    'pp16': 'Environment vs. growth (1=protect env ... 3=promote growth, higher=more pro-growth) [3-cat ordinal]',
    'pp15': 'Chances for foreigners (1=equal chances ... 3=better for Swiss, higher=more nativist) [3-cat ordinal]',
}

# Categorical political variable (do NOT compute means)
POLITICAL_OUTCOMES_CATEGORICAL = {
    'pp19': 'Vote choice / party preference',
}

# Continuous outcomes for the regression grid (Section 3)
# These are all outcomes suitable for OLS/FE regressions
REGRESSION_OUTCOMES = ['iwyn', 'pw86', 'pp10', 'pp13', 'pp17', 'pp22', 'pp16', 'pp15']

# Control variables for regression models
CONTROL_VARS = {
    'age': 'continuous',       # Age in years
    'sex': 'binary',           # 1=male, 2=female in SHP -> recode to 0/1
    'educat': 'ordinal',       # Education level (ordinal, use as continuous)
}


# =============================================================================
# Data Loading and Cleaning
# =============================================================================

def load_and_clean_data(input_file: str) -> pd.DataFrame:
    """
    Load an SHP exposure file and recode missing values.

    Args:
        input_file: Path to SHP exposure CSV (e.g. shp_exposure_isco4d.csv)

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

    # Validate required columns (match_level is optional -- only in fallback file)
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

    # --- Recode negative values for control variables (age, sex, education) ---
    for ctrl_var in ['age', 'sex', 'educat', 'edcat']:
        if ctrl_var in df.columns:
            n_negative = (df[ctrl_var] < 0).sum()
            if n_negative > 0:
                df.loc[df[ctrl_var] < 0, ctrl_var] = np.nan
                logger.info(f"  {ctrl_var}: recoded {n_negative:,} negative values to NaN")

    # --- Recode sex to 0/1 dummy (SHP: 1=male, 2=female -> 0=male, 1=female) ---
    if 'sex' in df.columns:
        n_before = df['sex'].notna().sum()
        df['sex'] = df['sex'].map({1: 0, 2: 1})
        n_after = df['sex'].notna().sum()
        logger.info(f"  sex: recoded 1=male->0, 2=female->1 ({n_after:,} valid obs)")
        if n_before != n_after:
            logger.warning(f"  sex: {n_before - n_after:,} values were not 1 or 2 and became NaN")

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
        logger.info("  (match_level column not present -- single-level file)")

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


def _get_education_var(df: pd.DataFrame) -> Optional[str]:
    """
    Determine which education variable is available: educat preferred, edcat fallback.

    Returns:
        Variable name, or None if neither exists.
    """
    for var in ['educat', 'edcat']:
        if var in df.columns and df[var].notna().sum() > 0:
            return var
    return None


def _get_regression_outcomes(df: pd.DataFrame, job_security_var: Optional[str]) -> List[str]:
    """
    Build list of continuous outcomes available in df for regression grid.
    Uses REGRESSION_OUTCOMES as base, substitutes the correct job security var.
    """
    outcomes = []
    for var in REGRESSION_OUTCOMES:
        # Substitute pw86 with the actual job security variable
        actual_var = job_security_var if var in JOB_SECURITY_CANDIDATES else var
        if actual_var and actual_var in df.columns:
            if actual_var not in outcomes:
                outcomes.append(actual_var)
    return outcomes


def _build_all_outcome_vars(df: pd.DataFrame, job_security_var: Optional[str]) -> List[str]:
    """
    Build the full list of outcome variables (economic + political, both
    continuous and categorical) that are present in df.
    """
    outcome_vars = []
    for var in ECONOMIC_OUTCOMES:
        if var in df.columns:
            outcome_vars.append(var)
    if job_security_var and job_security_var in df.columns:
        if job_security_var not in outcome_vars:
            outcome_vars.append(job_security_var)
    for var in POLITICAL_OUTCOMES_CONTINUOUS:
        if var in df.columns:
            outcome_vars.append(var)
    for var in POLITICAL_OUTCOMES_CATEGORICAL:
        if var in df.columns:
            outcome_vars.append(var)
    return outcome_vars


# =============================================================================
# Regression Helper
# =============================================================================

def _run_person_fe_regression(df: pd.DataFrame, outcome_var: str,
                               exposure_var: str) -> Tuple[float, float, float, float, int]:
    """
    Run within-person (fixed effects) regression.

    Uses linearmodels PanelOLS if available, otherwise demeans manually.

    Returns:
        (coefficient, standard_error, p_value, r_squared, n_actual)
        n_actual is the number of observations actually used in the regression
        (after dropping NaNs and persons without within-person exposure variation).
        Returns (NaN, NaN, NaN, NaN, 0) if regression cannot be estimated.
    """
    panel = df[['idpers', 'year', outcome_var, exposure_var]].dropna().copy()

    # Need persons with variation in exposure
    person_var = panel.groupby('idpers')[exposure_var].std()
    persons_with_variation = person_var[person_var > 0].index
    panel_var = panel[panel['idpers'].isin(persons_with_variation)]

    # n_actual is the sample that enters the regression (NaN-dropped + variation-filtered)
    n_actual = len(panel_var)

    if n_actual < 10 or len(persons_with_variation) < 2:
        return (np.nan, np.nan, np.nan, np.nan, 0)

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
                    res.pvalues[exposure_var], res.rsquared_within, n_actual)
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
        # R² from demeaned OLS is the within-R² (variance explained among deviations
        # from person means). It is NOT comparable to the overall R² reported by
        # Models A and B, and will typically be much lower.
        r2 = model.rsquared

        return (coef, se, p, r2, n_actual)

    except Exception as e:
        logger.debug(f"  Demeaned OLS failed: {e}")
        return (np.nan, np.nan, np.nan, np.nan, 0)


def _run_ols_with_year_fe(df_valid: pd.DataFrame, outcome_var: str,
                           exposure_var: str) -> Tuple[float, float, float, float, int]:
    """
    Model A: OLS with year fixed effects only.
    Returns: (coef, se, p, r2, N)
    """
    try:
        year_dummies = pd.get_dummies(df_valid['year'].astype(str), prefix='yr',
                                       drop_first=True, dtype=float)
        X = pd.concat([df_valid[[exposure_var]].reset_index(drop=True),
                        year_dummies.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X)
        y = df_valid[outcome_var].reset_index(drop=True).astype(float)

        model = sm.OLS(y, X).fit(cov_type='HC1')
        return (model.params[exposure_var], model.bse[exposure_var],
                model.pvalues[exposure_var], model.rsquared, len(y))
    except Exception as e:
        logger.debug(f"  OLS year FE failed for {outcome_var}: {e}")
        return (np.nan, np.nan, np.nan, np.nan, len(df_valid))


def _run_ols_with_controls(df_valid: pd.DataFrame, outcome_var: str,
                            exposure_var: str, edu_var: Optional[str]) -> Tuple[float, float, float, float, int]:
    """
    Model B: OLS with year FE + controls (age, sex, education).
    Returns: (coef, se, p, r2, N)
    """
    try:
        # Build controls list
        control_cols = []
        for ctrl in ['age', 'sex']:
            if ctrl in df_valid.columns and df_valid[ctrl].notna().sum() > 0:
                control_cols.append(ctrl)
        if edu_var and edu_var in df_valid.columns and df_valid[edu_var].notna().sum() > 0:
            control_cols.append(edu_var)

        # Drop rows with missing controls
        cols_needed = [exposure_var, outcome_var, 'year'] + control_cols
        valid = df_valid[cols_needed].dropna().copy()

        if len(valid) < 30:
            return (np.nan, np.nan, np.nan, np.nan, len(valid))

        year_dummies = pd.get_dummies(valid['year'].astype(str), prefix='yr',
                                       drop_first=True, dtype=float)
        X = pd.concat([valid[[exposure_var] + control_cols].reset_index(drop=True),
                        year_dummies.reset_index(drop=True)], axis=1)
        X = sm.add_constant(X)
        y = valid[outcome_var].reset_index(drop=True).astype(float)

        model = sm.OLS(y, X).fit(cov_type='HC1')
        return (model.params[exposure_var], model.bse[exposure_var],
                model.pvalues[exposure_var], model.rsquared, len(y))
    except Exception as e:
        logger.debug(f"  OLS with controls failed for {outcome_var}: {e}")
        return (np.nan, np.nan, np.nan, np.nan, len(df_valid))


# =============================================================================
# Section 1: Data Structure & Diagnostics
# =============================================================================

def section_1_data_structure(df: pd.DataFrame, output_dir: str,
                              job_security_var: Optional[str] = None):
    """
    Section 1: Data Structure & Diagnostics.

    Produces:
      - Summary stats for ALL outcome variables in one table
      - Missingness patterns (outcome x exposure combinations)
      - Exposure distribution (histogram + quantiles)
      - Coverage table: N, N_exposed, N_valid per outcome
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 1: DATA STRUCTURE & DIAGNOSTICS")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    # Build full list of outcome variables
    outcome_vars = _build_all_outcome_vars(df, job_security_var)
    categorical_vars = set(POLITICAL_OUTCOMES_CATEGORICAL.keys()) | {'wstat'}

    # --- 1.1: Summary statistics for ALL outcomes in one table ---
    logger.info("\n--- 1.1: Summary Statistics (All Outcomes) ---")
    summary_rows = []

    for var in outcome_vars:
        if var not in df.columns:
            continue
        series = df[var].dropna()
        if var in categorical_vars:
            row = {
                'variable': var,
                'label': get_var_label(var),
                'type': 'categorical',
                'N': len(series),
                'mean': np.nan,
                'sd': np.nan,
                'min': series.min(),
                'p25': np.nan,
                'median': np.nan,
                'p75': np.nan,
                'max': series.max(),
                'pct_missing': 100 * df[var].isna().sum() / len(df),
                'n_unique': series.nunique(),
            }
            logger.info(f"  {get_var_label(var)} ({var}): N={row['N']:,}, "
                        f"{row['n_unique']} unique values, categorical")
        else:
            row = {
                'variable': var,
                'label': get_var_label(var),
                'type': 'continuous',
                'N': len(series),
                'mean': series.mean(),
                'sd': series.std(),
                'min': series.min(),
                'p25': series.quantile(0.25),
                'median': series.median(),
                'p75': series.quantile(0.75),
                'max': series.max(),
                'pct_missing': 100 * df[var].isna().sum() / len(df),
                'n_unique': series.nunique(),
            }
            logger.info(f"  {get_var_label(var)} ({var}): N={row['N']:,}, "
                        f"mean={row['mean']:.3f}, sd={row['sd']:.3f}")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(tables_dir, 'summary_stats_all_outcomes.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  Saved: {summary_path}")

    # --- 1.2: Missingness patterns ---
    logger.info("\n--- 1.2: Missingness Patterns ---")
    missingness_rows = []
    for var in outcome_vars:
        if var not in df.columns:
            continue
        has_outcome = df[var].notna()
        has_exp = df[PRIMARY_EXPOSURE].notna() & (df[PRIMARY_EXPOSURE] > 0)

        missingness_rows.append({
            'variable': var,
            'label': get_var_label(var),
            'n_total': len(df),
            'n_valid_outcome': has_outcome.sum(),
            'n_missing_outcome': (~has_outcome).sum(),
            'pct_missing_outcome': 100 * (~has_outcome).sum() / len(df),
            'n_with_exposure': has_exp.sum(),
            'n_valid_outcome_and_exposure': (has_outcome & has_exp).sum(),
            'n_valid_outcome_no_exposure': (has_outcome & ~has_exp).sum(),
        })

    missingness_df = pd.DataFrame(missingness_rows)
    miss_path = os.path.join(tables_dir, 'missingness_patterns.csv')
    missingness_df.to_csv(miss_path, index=False)
    logger.info(f"  Saved: {miss_path}")
    for _, row in missingness_df.iterrows():
        logger.info(f"  {row['variable']}: {row['n_valid_outcome']:,} valid, "
                    f"{row['n_valid_outcome_and_exposure']:,} with exposure, "
                    f"{row['pct_missing_outcome']:.1f}% missing")

    # --- 1.3: Exposure distribution ---
    logger.info("\n--- 1.3: Exposure Distribution ---")

    # Full distribution (including zeros)
    all_exp = df[PRIMARY_EXPOSURE].fillna(0)
    n_zero = (all_exp == 0).sum()
    n_nonzero = (all_exp > 0).sum()
    logger.info(f"  Total person-years: {len(all_exp):,}")
    logger.info(f"  Zero exposure: {n_zero:,} ({100*n_zero/len(all_exp):.1f}%)")
    logger.info(f"  Non-zero exposure: {n_nonzero:,} ({100*n_nonzero/len(all_exp):.1f}%)")

    # Quantiles of non-zero exposure
    nonzero_exp = df.loc[df[PRIMARY_EXPOSURE] > 0, PRIMARY_EXPOSURE]
    if len(nonzero_exp) > 0:
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        exp_quantiles = {f'p{int(q*100):02d}': nonzero_exp.quantile(q) for q in quantiles}
        exp_quantiles['N'] = len(nonzero_exp)
        exp_quantiles['mean'] = nonzero_exp.mean()
        exp_quantiles['sd'] = nonzero_exp.std()
        exp_quantiles['min'] = nonzero_exp.min()
        exp_quantiles['max'] = nonzero_exp.max()

        exp_q_df = pd.DataFrame([exp_quantiles])
        exp_q_path = os.path.join(tables_dir, 'exposure_distribution_quantiles.csv')
        exp_q_df.to_csv(exp_q_path, index=False)
        logger.info(f"  Saved: {exp_q_path}")
        logger.info(f"  Non-zero exposure: mean={exp_quantiles['mean']:.6f}, "
                    f"median={exp_quantiles['p50']:.6f}, max={exp_quantiles['max']:.6f}")

        # Histogram
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: full distribution
        ax1 = axes[0]
        ax1.hist(all_exp, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel(f'{PRIMARY_EXPOSURE}')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Full Distribution (N={len(all_exp):,}, {n_nonzero:,} non-zero)')
        ax1.set_yscale('log')

        # Right: non-zero only
        ax2 = axes[1]
        ax2.hist(nonzero_exp, bins=50, edgecolor='black', alpha=0.7, color='coral')
        ax2.set_xlabel(f'{PRIMARY_EXPOSURE}')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Non-Zero Only (N={len(nonzero_exp):,})')
        ax2.axvline(nonzero_exp.median(), color='red', linestyle='--',
                    label=f'Median: {nonzero_exp.median():.6f}')
        ax2.legend()

        fig.suptitle('AI Exposure Distribution', fontsize=14)
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, 'exposure_distribution.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")

    # --- 1.4: Coverage table ---
    logger.info("\n--- 1.4: Coverage Table ---")
    coverage_rows = []
    for var in outcome_vars:
        if var not in df.columns:
            continue
        has_outcome = df[var].notna()
        has_exp = df[PRIMARY_EXPOSURE].notna() & (df[PRIMARY_EXPOSURE] > 0)

        coverage_rows.append({
            'variable': var,
            'label': get_var_label(var),
            'N_total': len(df),
            'N_valid': has_outcome.sum(),
            'N_exposed': has_exp.sum(),
            'N_valid_and_exposed': (has_outcome & has_exp).sum(),
            'pct_valid': 100 * has_outcome.sum() / len(df),
            'pct_exposed': 100 * has_exp.sum() / len(df),
        })

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_path = os.path.join(tables_dir, 'coverage_table.csv')
    coverage_df.to_csv(coverage_path, index=False)
    logger.info(f"  Saved: {coverage_path}")


# =============================================================================
# Section 2: Pairwise Exploration (Descriptive Only)
# =============================================================================

def section_2_pairwise_exploration(df: pd.DataFrame, output_dir: str,
                                    job_security_var: Optional[str] = None):
    """
    Section 2: Pairwise Exploration (Descriptive Only).

    NO p-values in this section -- purely descriptive.

    Produces:
      - Correlation matrix: exposure vs all continuous outcomes (Spearman)
      - Heatmap figure + CSV
      - Means by exposure status for all outcomes (one unified table)
      - Cross-tabs for categorical variables (wstat, pp19)
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 2: PAIRWISE EXPLORATION (DESCRIPTIVE ONLY)")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    outcome_vars = _build_all_outcome_vars(df, job_security_var)
    categorical_vars = set(POLITICAL_OUTCOMES_CATEGORICAL.keys()) | {'wstat'}

    # --- 2.1: Correlation matrix (Spearman) ---
    logger.info("\n--- 2.1: Correlation Matrix (Spearman) ---")

    continuous_outcomes = [v for v in outcome_vars if v not in categorical_vars and v in df.columns]
    corr_vars = [PRIMARY_EXPOSURE] + continuous_outcomes

    # Build correlation matrix using Spearman (appropriate for ordinal outcomes)
    corr_data = df[corr_vars].dropna()
    if len(corr_data) > 10:
        corr_matrix = corr_data.corr(method='spearman')

        # Save CSV
        corr_path = os.path.join(tables_dir, 'correlation_matrix_spearman.csv')
        corr_matrix.to_csv(corr_path)
        logger.info(f"  Saved: {corr_path} (N={len(corr_data):,} complete cases)")

        # Log exposure correlations
        for var in continuous_outcomes:
            r = corr_matrix.loc[PRIMARY_EXPOSURE, var]
            logger.info(f"  Spearman r({get_var_label(var)}, exposure) = {r:.4f}")

        # Heatmap
        # Use human-readable labels for axes
        label_map = {v: get_var_short_label(v) for v in corr_vars}
        label_map[PRIMARY_EXPOSURE] = 'AI Exposure'
        corr_display = corr_matrix.rename(index=label_map, columns=label_map)

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_display, annot=True, fmt='.3f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True, ax=ax,
                    linewidths=0.5, cbar_kws={'label': 'Spearman r'})
        ax.set_title('Spearman Correlation Matrix: Exposure vs Outcomes', fontsize=14)
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, 'correlation_heatmap_spearman.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")
    else:
        logger.warning("  Insufficient complete cases for correlation matrix")

    # --- 2.2: Means by exposure status (unified table) ---
    logger.info("\n--- 2.2: Means by Exposure Status ---")
    means_rows = []
    for var in outcome_vars:
        if var not in df.columns:
            continue
        if var in categorical_vars:
            continue  # Categorical vars handled in cross-tabs below

        for exp_val, group_label in [(0, 'Not exposed'), (1, 'Exposed')]:
            sub = df.loc[df['has_exposure'] == exp_val, var].dropna()
            if len(sub) == 0:
                continue
            means_rows.append({
                'variable': var,
                'label': get_var_label(var),
                'group': group_label,
                'N': len(sub),
                'mean': sub.mean(),
                'sd': sub.std(),
                'median': sub.median(),
            })

    means_df = pd.DataFrame(means_rows)
    means_path = os.path.join(tables_dir, 'means_by_exposure_status.csv')
    means_df.to_csv(means_path, index=False)
    logger.info(f"  Saved: {means_path}")

    # Log the differences
    for var in outcome_vars:
        if var in categorical_vars or var not in df.columns:
            continue
        exposed = df.loc[df['has_exposure'] == 1, var].dropna()
        not_exposed = df.loc[df['has_exposure'] == 0, var].dropna()
        if len(exposed) > 0 and len(not_exposed) > 0:
            diff = exposed.mean() - not_exposed.mean()
            logger.info(f"  {get_var_label(var)}: exposed={exposed.mean():.3f}, "
                        f"not exposed={not_exposed.mean():.3f}, diff={diff:.3f}")

    # --- 2.3: Cross-tabs for categorical variables ---
    logger.info("\n--- 2.3: Cross-Tabs for Categorical Variables ---")

    for var in categorical_vars:
        if var not in df.columns:
            continue

        valid = df.loc[df[var].notna()].copy()
        if len(valid) == 0:
            continue

        # Counts cross-tab
        cross_counts = pd.crosstab(valid[var], valid['has_exposure'], margins=True)
        counts_path = os.path.join(tables_dir, f'crosstab_{var}_counts.csv')
        cross_counts.to_csv(counts_path)

        # Proportions cross-tab (column-normalized)
        cross_pct = pd.crosstab(valid[var], valid['has_exposure'],
                                normalize='columns')
        pct_path = os.path.join(tables_dir, f'crosstab_{var}_proportions.csv')
        cross_pct.to_csv(pct_path)

        logger.info(f"  {get_var_label(var)} ({var}): saved counts and proportions cross-tabs")

        # Bar chart for top categories
        top_cats = cross_counts.drop('All', errors='ignore').sum(axis=1).nlargest(10).index
        plot_data = cross_pct.loc[cross_pct.index.isin(top_cats)].copy()
        if plot_data.shape[1] >= 2:
            plot_data.columns = ['Not Exposed', 'Exposed']
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_data.plot(kind='bar', ax=ax, edgecolor='black', alpha=0.8)
            ax.set_xlabel(get_var_short_label(var))
            ax.set_ylabel('Proportion')
            ax.set_title(f'{get_var_short_label(var)} by AI Exposure Status')
            ax.legend(title='Exposure Status')
            plt.xticks(rotation=45, ha='right')
            fig.tight_layout()
            fig_path = os.path.join(figures_dir, f'crosstab_{var}_barplot.png')
            fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
            plt.close(fig)
            logger.info(f"  Saved: {fig_path}")


# =============================================================================
# Section 3: Structured Regression Grid
# =============================================================================

def section_3_regression_grid(df: pd.DataFrame, output_dir: str,
                               job_security_var: Optional[str] = None) -> pd.DataFrame:
    """
    Section 3: Structured Regression Grid.

    For each continuous outcome, runs three models across three margin panels:
      Extensive margin: binary has_exposure (0/1), full sample
      Continuous margin: continuous PRIMARY_EXPOSURE score, full sample
      Intensive margin:  continuous PRIMARY_EXPOSURE score, exposed-only subsample

    Within each panel:
      - Model A: outcome ~ exposure + year_dummies (pooled OLS)
      - Model B: outcome ~ exposure + year_dummies + age + sex + education (OLS with controls)
      - Model C: outcome ~ exposure | person FE + year FE (within-person)

    Results include a 'margin' column identifying the panel.
    Returns:
        DataFrame with all regression results (used by Sections 4 and 5).
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 3: STRUCTURED REGRESSION GRID")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    edu_var = _get_education_var(df)

    if edu_var:
        logger.info(f"  Education variable: {edu_var}")
    else:
        logger.warning("  No education variable found (educat/edcat), Model B will exclude education")

    # Control variable summary
    for ctrl in ['age', 'sex', edu_var]:
        if ctrl and ctrl in df.columns:
            n_valid = df[ctrl].notna().sum()
            logger.info(f"  Control {ctrl}: {n_valid:,} valid observations")

    outcomes = _get_regression_outcomes(df, job_security_var)
    logger.info(f"  Outcomes to analyze: {outcomes}")

    # Three panels: extensive (binary, full), continuous (score, full), intensive (score, exposed-only)
    MARGINS = [
        ('extensive',  'has_exposure',   False),  # binary exposed/unexposed, full sample
        ('continuous', PRIMARY_EXPOSURE, False),  # continuous score, full sample
        ('intensive',  PRIMARY_EXPOSURE, True),   # continuous score, exposed-only
    ]

    all_results = []

    for var in outcomes:
        if var not in df.columns:
            logger.info(f"  Skipping {var}: not in data")
            continue

        # Base sample: outcome non-null and PRIMARY_EXPOSURE non-null
        base_valid = df.loc[df[var].notna() & df[PRIMARY_EXPOSURE].notna()].copy()
        n_valid = len(base_valid)
        n_exposed = (base_valid['has_exposure'] == 1).sum()

        if n_valid < 30 or n_exposed < 5:
            logger.info(f"  Skipping {get_var_label(var)}: insufficient data "
                        f"(N={n_valid}, exposed={n_exposed})")
            continue

        logger.info(f"\n  {get_var_label(var)} ({var}): N={n_valid:,}, exposed={n_exposed:,}")

        for margin_name, exp_var, exposed_only in MARGINS:
            valid = base_valid[base_valid['has_exposure'] == 1].copy() if exposed_only else base_valid
            margin_n = len(valid)

            if margin_n < 30:
                logger.info(f"    [{margin_name}] Skipping: insufficient N={margin_n}")
                continue

            logger.info(f"    [{margin_name}] N={margin_n:,}, exposure_var={exp_var}")

            # --- Model A: Pooled OLS with year FE ---
            coef_a, se_a, p_a, r2_a, n_a = _run_ols_with_year_fe(valid, var, exp_var)
            if not np.isnan(coef_a):
                logger.info(f"      Model A (OLS + year FE): coef={coef_a:.4f}, se={se_a:.4f}, "
                            f"p={p_a:.4f}{significance_stars(p_a)}, R2={r2_a:.4f}, N={n_a}")
            else:
                logger.info(f"      Model A (OLS + year FE): failed to estimate")

            all_results.append({
                'outcome': var,
                'outcome_label': get_var_label(var),
                'margin': margin_name,
                'model_type': 'A_ols_yearfe',
                'N': n_a,
                'coef': coef_a,
                'se': se_a,
                'p': p_a,
                'stars': significance_stars(p_a) if pd.notna(p_a) else '',
                'r2': r2_a,
                'r2_type': 'overall',
            })

            # --- Model B: OLS with year FE + controls ---
            coef_b, se_b, p_b, r2_b, n_b = _run_ols_with_controls(valid, var, exp_var, edu_var)
            if not np.isnan(coef_b):
                logger.info(f"      Model B (OLS + controls): coef={coef_b:.4f}, se={se_b:.4f}, "
                            f"p={p_b:.4f}{significance_stars(p_b)}, R2={r2_b:.4f}, N={n_b}")
            else:
                logger.info(f"      Model B (OLS + controls): failed to estimate")

            all_results.append({
                'outcome': var,
                'outcome_label': get_var_label(var),
                'margin': margin_name,
                'model_type': 'B_ols_controls',
                'N': n_b,
                'coef': coef_b,
                'se': se_b,
                'p': p_b,
                'stars': significance_stars(p_b) if pd.notna(p_b) else '',
                'r2': r2_b,
                'r2_type': 'overall',
            })

            # --- Model C: Person FE ---
            # n_c comes from inside the function: persons with non-NaN data AND
            # within-person exposure variation. It is smaller than the pre-filter
            # sample and is the actual regression sample.
            coef_c, se_c, p_c, r2_c, n_c = _run_person_fe_regression(valid, var, exp_var)
            if not np.isnan(coef_c):
                logger.info(f"      Model C (Person FE): coef={coef_c:.4f}, se={se_c:.4f}, "
                            f"p={p_c:.4f}{significance_stars(p_c)}, R2={r2_c:.4f} (within)")
            else:
                logger.info(f"      Model C (Person FE): insufficient within-person variation")

            all_results.append({
                'outcome': var,
                'outcome_label': get_var_label(var),
                'margin': margin_name,
                'model_type': 'C_person_fe',
                'N': n_c,
                'coef': coef_c,
                'se': se_c,
                'p': p_c,
                'stars': significance_stars(p_c) if pd.notna(p_c) else '',
                'r2': r2_c,
                # Within-R²: variance explained among deviations from person means.
                # NOT comparable to the overall R² in Models A and B.
                'r2_type': 'within (not comparable to A/B)',
            })

    # Save full grid
    results_df = pd.DataFrame(all_results)
    if len(results_df) > 0:
        grid_path = os.path.join(tables_dir, 'regression_grid_all_results.csv')
        results_df.to_csv(grid_path, index=False)
        logger.info(f"\n  Saved full regression grid: {grid_path}")
        logger.info(f"  Total models estimated: {len(results_df)}")
    else:
        logger.warning("  No regression results produced")

    return results_df


# =============================================================================
# Section 4: Collective Summary
# =============================================================================

def section_4_collective_summary(df: pd.DataFrame, regression_results: pd.DataFrame,
                                  output_dir: str, job_security_var: Optional[str] = None):
    """
    Section 4: Collective Summary.

    Produces:
      - Coefficient heatmap: outcomes (rows) x model types (columns)
      - Sign consistency table
      - Effect size summary (standardized coefficients)
      - Model fit comparison (R2)
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 4: COLLECTIVE SUMMARY")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    if len(regression_results) == 0:
        logger.warning("  No regression results available for collective summary. Skipping.")
        return

    # --- 4.1: Coefficient heatmap (one per margin) ---
    logger.info("\n--- 4.1: Coefficient Heatmap ---")

    col_order = ['A_ols_yearfe', 'B_ols_controls', 'C_person_fe']
    col_labels = {
        'A_ols_yearfe': 'Model A\n(OLS + Year FE)',
        'B_ols_controls': 'Model B\n(OLS + Controls)',
        'C_person_fe': 'Model C\n(Person FE)',
    }
    margin_titles = {
        'extensive':  'Extensive Margin (binary exposure)',
        'continuous': 'Continuous Margin (full sample)',
        'intensive':  'Intensive Margin (exposed only)',
    }

    for margin_name in ['extensive', 'continuous', 'intensive']:
        margin_res = regression_results[regression_results['margin'] == margin_name]
        if len(margin_res) == 0:
            continue
        pivot_coef = margin_res.pivot_table(index='outcome', columns='model_type', values='coef')
        cols = [c for c in col_order if c in pivot_coef.columns]
        pivot_display = pivot_coef[cols].rename(
            index=lambda v: get_var_short_label(v),
            columns=col_labels
        )

        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_display) * 0.8)))
        sns.heatmap(pivot_display, annot=True, fmt='.4f', cmap='RdBu_r',
                    center=0, linewidths=0.5, ax=ax, cbar_kws={'label': 'Coefficient'})
        ax.set_title(f'Regression Coefficients — {margin_titles[margin_name]}', fontsize=14)
        ax.set_ylabel('Outcome')
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, f'coefficient_heatmap_{margin_name}.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved: {fig_path}")

    # --- 4.2: Sign consistency table (per margin) ---
    logger.info("\n--- 4.2: Sign Consistency ---")

    sign_rows = []
    for outcome_label in regression_results['outcome_label'].unique():
      for margin_name in regression_results['margin'].unique():
        sub = regression_results[
            (regression_results['outcome_label'] == outcome_label) &
            (regression_results['margin'] == margin_name)
        ]
        coefs = sub['coef'].dropna()
        if len(coefs) == 0:
            continue

        signs = coefs.apply(lambda x: '+' if x > 0 else ('-' if x < 0 else '0'))
        n_positive = (coefs > 0).sum()
        n_negative = (coefs < 0).sum()
        n_models = len(coefs)
        consistent = 'Yes' if (n_positive == n_models or n_negative == n_models) else 'No'

        sign_rows.append({
            'outcome_label': outcome_label,
            'outcome': sub['outcome'].iloc[0],
            'margin': margin_name,
            'n_models': n_models,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'sign_consistent': consistent,
            'signs': ', '.join(signs.values),
        })
        logger.info(f"  [{margin_name}] {outcome_label}: {consistent} "
                    f"(+:{n_positive}, -:{n_negative}, signs={', '.join(signs.values)})")

    sign_df = pd.DataFrame(sign_rows)
    sign_path = os.path.join(tables_dir, 'sign_consistency.csv')
    sign_df.to_csv(sign_path, index=False)
    logger.info(f"  Saved: {sign_path}")

    # --- 4.3: Effect size summary (standardized coefficients) ---
    logger.info("\n--- 4.3: Standardized Effect Sizes ---")

    # SD of each exposure variable (extensive uses has_exposure, others use PRIMARY_EXPOSURE)
    sd_by_margin = {
        'extensive':  df['has_exposure'].std(),
        'continuous': df[PRIMARY_EXPOSURE].std(),
        'intensive':  df.loc[df['has_exposure'] == 1, PRIMARY_EXPOSURE].std(),
    }

    std_rows = []
    for _, row in regression_results.iterrows():
        outcome_var = row['outcome']
        sd_exposure = sd_by_margin.get(row['margin'], df[PRIMARY_EXPOSURE].std())
        if outcome_var in df.columns:
            sd_outcome = df[outcome_var].std()
            if pd.notna(row['coef']) and sd_outcome > 0 and sd_exposure > 0:
                beta_std = row['coef'] * (sd_exposure / sd_outcome)
            else:
                beta_std = np.nan
        else:
            beta_std = np.nan

        # Standardized betas WITHIN a margin are comparable across outcomes.
        # Standardized betas ACROSS margins are NOT directly comparable: the
        # extensive margin uses SD(binary indicator) ≤ 0.5 by construction,
        # while the continuous and intensive margins use SD of the raw exposure
        # score which can be much larger. Cross-margin comparisons are invalid.
        margin_note = {
            'extensive':  'SD of binary indicator (≤0.5); cross-margin comparison invalid',
            'continuous': 'SD of continuous exposure score; cross-margin comparison invalid',
            'intensive':  'SD of exposure score among exposed only; cross-margin comparison invalid',
        }.get(row['margin'], '')
        std_rows.append({
            'outcome': row['outcome'],
            'outcome_label': row['outcome_label'],
            'margin': row['margin'],
            'model_type': row['model_type'],
            'coef_raw': row['coef'],
            'coef_standardized': beta_std,
            'sd_exposure': sd_exposure,
            'sd_outcome': df[outcome_var].std() if outcome_var in df.columns else np.nan,
            'comparability_note': margin_note,
        })

    std_df = pd.DataFrame(std_rows)
    std_path = os.path.join(tables_dir, 'standardized_effect_sizes.csv')
    std_df.to_csv(std_path, index=False)
    logger.info(f"  Saved: {std_path}")

    for _, row in std_df.iterrows():
        if pd.notna(row['coef_standardized']):
            logger.info(f"  [{row['margin']}] {row['outcome_label']} ({row['model_type']}): "
                        f"beta_std={row['coef_standardized']:.4f}")

    # --- 4.4: Model fit comparison (R2, per margin) ---
    logger.info("\n--- 4.4: Model Fit Comparison (R2) ---")

    for margin_name in ['extensive', 'continuous', 'intensive']:
        margin_res = regression_results[regression_results['margin'] == margin_name]
        if len(margin_res) == 0:
            continue
        pivot_r2 = margin_res.pivot_table(index='outcome_label', columns='model_type', values='r2')
        cols = [c for c in col_order if c in pivot_r2.columns]
        pivot_r2 = pivot_r2[cols]
        r2_path = os.path.join(tables_dir, f'model_fit_r2_{margin_name}.csv')
        pivot_r2.to_csv(r2_path)
        logger.info(f"  [{margin_name}] Saved: {r2_path}")
        for idx, row in pivot_r2.iterrows():
            vals = ', '.join([f"{c}={row[c]:.4f}" for c in pivot_r2.columns if pd.notna(row[c])])
            logger.info(f"    {idx}: {vals}")


# =============================================================================
# Section 5: Multiple Testing Adjustment
# =============================================================================

def section_5_multiple_testing(regression_results: pd.DataFrame, output_dir: str):
    """
    Section 5: Multiple Testing Adjustment.

    Applies Benjamini-Hochberg FDR correction across all p-values from Section 3.

    Produces:
      - Table: original p, adjusted p, significant before/after correction
      - Summary count of results surviving FDR correction
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 5: MULTIPLE TESTING ADJUSTMENT (FDR)")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')

    if len(regression_results) == 0:
        logger.warning("  No regression results for FDR correction. Skipping.")
        return

    # Filter to rows with valid p-values
    valid_results = regression_results.dropna(subset=['p']).copy()
    n_tests = len(valid_results)
    logger.info(f"  Total tests: {n_tests}")

    if n_tests == 0:
        logger.warning("  No valid p-values for FDR correction. Skipping.")
        return

    # Apply Benjamini-Hochberg FDR correction
    reject_bh, p_adjusted, _, _ = multipletests(
        valid_results['p'].values, alpha=0.05, method='fdr_bh'
    )

    valid_results['p_original'] = valid_results['p']
    valid_results['p_adjusted_fdr'] = p_adjusted
    valid_results['sig_original_05'] = (valid_results['p_original'] < 0.05).astype(int)
    valid_results['sig_adjusted_05'] = reject_bh.astype(int)
    valid_results['stars_original'] = valid_results['p_original'].apply(significance_stars)
    valid_results['stars_adjusted'] = valid_results['p_adjusted_fdr'].apply(significance_stars)

    # Save full FDR table
    fdr_cols = ['outcome', 'outcome_label', 'margin', 'model_type', 'coef', 'se',
                'p_original', 'stars_original', 'p_adjusted_fdr', 'stars_adjusted',
                'sig_original_05', 'sig_adjusted_05']
    fdr_df = valid_results[fdr_cols].copy()
    fdr_path = os.path.join(tables_dir, 'fdr_correction_results.csv')
    fdr_df.to_csv(fdr_path, index=False)
    logger.info(f"  Saved: {fdr_path}")

    # Summary
    n_sig_original = valid_results['sig_original_05'].sum()
    n_sig_adjusted = valid_results['sig_adjusted_05'].sum()

    logger.info(f"\n  FDR Correction Summary:")
    logger.info(f"    Total tests: {n_tests}")
    logger.info(f"    Significant at p<0.05 (unadjusted): {n_sig_original} ({100*n_sig_original/n_tests:.1f}%)")
    logger.info(f"    Significant at p<0.05 (FDR-adjusted): {n_sig_adjusted} ({100*n_sig_adjusted/n_tests:.1f}%)")

    # Log each result
    for _, row in fdr_df.iterrows():
        status = "SURVIVES" if row['sig_adjusted_05'] else "does not survive"
        logger.info(f"    {row['outcome_label']} ({row['model_type']}): "
                    f"p={row['p_original']:.4f}{row['stars_original']}, "
                    f"p_adj={row['p_adjusted_fdr']:.4f}{row['stars_adjusted']} "
                    f"-- {status} FDR")

    # Summary table
    summary_df = pd.DataFrame([{
        'total_tests': n_tests,
        'sig_unadjusted_05': n_sig_original,
        'sig_fdr_adjusted_05': n_sig_adjusted,
        'pct_surviving_fdr': 100 * n_sig_adjusted / n_tests if n_tests > 0 else 0,
        'method': 'Benjamini-Hochberg',
        'alpha': 0.05,
    }])
    summary_path = os.path.join(tables_dir, 'fdr_correction_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"  Saved: {summary_path}")


# =============================================================================
# Section 6: Intensive Margin Analysis
# =============================================================================

def section_6_intensive_margin(df: pd.DataFrame, output_dir: str,
                                job_security_var: Optional[str] = None):
    """
    Section 6: Intensive Margin Analysis.

    Among exposed workers only: how does the LEVEL of exposure relate to outcomes?

    Produces:
      - Tercile analysis (means by exposure tercile)
      - Regression grid (Models A/B/C) restricted to exposed subsample
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 6: INTENSIVE MARGIN ANALYSIS")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

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

    # --- 6.1: Exposure tercile descriptives ---
    logger.info("\n--- 6.1: Exposure Tercile Analysis ---")

    df_exposed['exposure_tercile'] = pd.qcut(
        df_exposed[PRIMARY_EXPOSURE], q=3, labels=['Low', 'Medium', 'High'],
        duplicates='drop'
    )

    outcomes = _get_regression_outcomes(df, job_security_var)
    tercile_rows = []
    for var in outcomes:
        if var not in df_exposed.columns:
            continue
        for tercile in ['Low', 'Medium', 'High']:
            sub = df_exposed.loc[df_exposed['exposure_tercile'] == tercile, var].dropna()
            if len(sub) > 0:
                tercile_rows.append({
                    'variable': var,
                    'label': get_var_label(var),
                    'tercile': tercile,
                    'N': len(sub),
                    'mean': sub.mean(),
                    'sd': sub.std(),
                    'median': sub.median(),
                })

    if tercile_rows:
        tercile_df = pd.DataFrame(tercile_rows)
        tercile_path = os.path.join(tables_dir, 'intensive_tercile_means.csv')
        tercile_df.to_csv(tercile_path, index=False)
        logger.info(f"  Saved: {tercile_path}")

        for var in outcomes:
            var_data = tercile_df[tercile_df['variable'] == var]
            if len(var_data) == 3:
                low_mean = var_data.loc[var_data['tercile'] == 'Low', 'mean'].values[0]
                high_mean = var_data.loc[var_data['tercile'] == 'High', 'mean'].values[0]
                logger.info(f"    {get_var_label(var)}: Low={low_mean:.3f}, High={high_mean:.3f}, "
                            f"diff={high_mean - low_mean:.3f}")

    # Tercile bar chart for income
    if 'iwyn' in df_exposed.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        income_by_tercile = df_exposed.groupby('exposure_tercile')['iwyn'].agg(
            ['mean', 'sem']
        ).reindex(['Low', 'Medium', 'High'])
        if income_by_tercile.notna().all().all():
            ax.bar(income_by_tercile.index, income_by_tercile['mean'],
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
        else:
            plt.close(fig)

    # --- 6.2: Regression grid (Models A/B/C) on exposed subsample ---
    logger.info("\n--- 6.2: Regression Grid (Exposed Only) ---")

    edu_var = _get_education_var(df)
    intensive_results = []

    for var in outcomes:
        if var not in df_exposed.columns:
            continue

        valid = df_exposed.loc[df_exposed[var].notna() & df_exposed[PRIMARY_EXPOSURE].notna()].copy()
        n_valid = len(valid)
        if n_valid < 30:
            logger.info(f"  {get_var_label(var)}: insufficient data ({n_valid}), skipping")
            continue

        logger.info(f"\n  {get_var_label(var)} ({var}): N={n_valid:,}")

        # Model A: OLS + year FE (exposed only)
        coef_a, se_a, p_a, r2_a, n_a = _run_ols_with_year_fe(valid, var, PRIMARY_EXPOSURE)
        if not np.isnan(coef_a):
            logger.info(f"    Model A (exposed): coef={coef_a:.4f}, se={se_a:.4f}, "
                        f"p={p_a:.4f}{significance_stars(p_a)}, N={n_a}")

        intensive_results.append({
            'outcome': var,
            'outcome_label': get_var_label(var),
            'model_type': 'A_ols_yearfe',
            'sample': 'exposed_only',
            'N': n_a,
            'coef': coef_a, 'se': se_a, 'p': p_a,
            'stars': significance_stars(p_a) if pd.notna(p_a) else '',
            'r2': r2_a,
        })

        # Model B: OLS + controls (exposed only)
        coef_b, se_b, p_b, r2_b, n_b = _run_ols_with_controls(
            valid, var, PRIMARY_EXPOSURE, edu_var
        )
        if not np.isnan(coef_b):
            logger.info(f"    Model B (exposed): coef={coef_b:.4f}, se={se_b:.4f}, "
                        f"p={p_b:.4f}{significance_stars(p_b)}, N={n_b}")

        intensive_results.append({
            'outcome': var,
            'outcome_label': get_var_label(var),
            'model_type': 'B_ols_controls',
            'sample': 'exposed_only',
            'N': n_b,
            'coef': coef_b, 'se': se_b, 'p': p_b,
            'stars': significance_stars(p_b) if pd.notna(p_b) else '',
            'r2': r2_b,
        })

        # Model C: Person FE (exposed only)
        coef_c, se_c, p_c, r2_c, n_c_intensive = _run_person_fe_regression(valid, var, PRIMARY_EXPOSURE)
        if not np.isnan(coef_c):
            logger.info(f"    Model C (exposed): coef={coef_c:.4f}, se={se_c:.4f}, "
                        f"p={p_c:.4f}{significance_stars(p_c)}")
        else:
            logger.info(f"    Model C (exposed): insufficient within-person variation")

        intensive_results.append({
            'outcome': var,
            'outcome_label': get_var_label(var),
            'model_type': 'C_person_fe',
            'sample': 'exposed_only',
            'N': n_c_intensive,
            'coef': coef_c, 'se': se_c, 'p': p_c,
            'stars': significance_stars(p_c) if pd.notna(p_c) else '',
            'r2': r2_c,
        })

    if intensive_results:
        intensive_df = pd.DataFrame(intensive_results)
        intensive_path = os.path.join(tables_dir, 'intensive_regression_grid.csv')
        intensive_df.to_csv(intensive_path, index=False)
        logger.info(f"\n  Saved: {intensive_path}")


# =============================================================================
# Section 7: Panel Dynamics
# =============================================================================

def section_7_panel_dynamics(df: pd.DataFrame, output_dir: str,
                              job_security_var: Optional[str] = None):
    """
    Section 7: Panel Dynamics.

    Produces:
      - Switcher identification and summary
      - Event study around first exposure
      - Within-person transitions and variation statistics
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 7: PANEL DYNAMICS")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    # --- 7.1: Identify switchers ---
    logger.info("\n--- 7.1: Identifying Switchers ---")

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

    # --- 7.2: Event study around first exposure ---
    logger.info("\n--- 7.2: Event Study (First Exposure) ---")

    _event_study(df, person_exposure, tables_dir, figures_dir)

    # --- 7.3: Within-person variation statistics ---
    logger.info("\n--- 7.3: Within-Person Variation ---")

    _within_person_variation(df, tables_dir)

    # --- 7.4: Within-person income transitions ---
    logger.info("\n--- 7.4: Within-Person Income Transitions ---")

    if 'iwyn' in df.columns:
        _within_person_income_changes(df, tables_dir, figures_dir)
    else:
        logger.warning("  Variable 'iwyn' not available, skipping income transitions")

    # --- 7.5: Job-switch × exposure-change matrix ---
    logger.info("\n--- 7.5: Job-Switch × Exposure-Change Matrix ---")

    _job_switch_exposure_matrix(df, tables_dir)

    # --- 7.6: Causal event study — job switchers ---
    logger.info("\n--- 7.6: Causal Event Study (Job Switchers) ---")

    _job_switcher_causal_event_study(df, tables_dir, figures_dir)

    # --- 7.7: Causal event study — firm AI adoption for stayers ---
    logger.info("\n--- 7.7: Causal Event Study (Firm Adoption, Stayers) ---")

    _firm_adoption_event_study(df, tables_dir, figures_dir)


def _firm_adoption_event_study(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """
    Section 7.7: Event study around firm AI adoption for stayers.

    This is the preferred causal design over §7.6. The treatment (firm AI
    adoption) is external to the worker: the firm decides to adopt AI, and
    the worker is affected regardless of their own choices.

    Population: Persons who stay at the same employer (pw18 ∈ {1, 4}).
      pw18 coding: 1=changed job (same employer), 2=changed employer (same job),
                   3=changed both, 4=no change. So {1,4} = same employer.
    Event: First year a stayer has positive AI exposure (previous year was
           0 or NaN — i.e., not yet positive). This relaxes the strict
           0→>0 requirement to include cases where the firm first appears
           in our job-ad data with AI exposure.
    Treated (D=1): Stayers whose firm ever has positive AI exposure.
    Control (D=0): Stayers whose firm never has positive AI exposure
                   (exposure is always 0 or NaN across all observed years).

    Two analyses per outcome:
      Descriptive: mean outcome by event_time for treated only (no natural
                   event_time for never-adopting controls), normalized to t=-1.
      Regression:  DiD — β_k from y_it = Σ_{k≠-1} β_k(D_i × 1[event_time=k])
                   + person_FE + year_FE, with never-adopting stayers as control.
                   Control observations have D_i=0 so all interaction terms are
                   0 for them; they contribute only to person and year FEs.
    """
    needed_cols = ['idpers', 'year', 'pw18', PRIMARY_EXPOSURE]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        logger.warning(f"  Skipping firm adoption event study: missing columns {missing}")
        return

    work = df[needed_cols].copy()
    work['pw18'] = work['pw18'].where(work['pw18'] > 0, np.nan)
    work = work.sort_values(['idpers', 'year'])
    work['exposure_lag'] = work.groupby('idpers')[PRIMARY_EXPOSURE].shift(1)

    # --- Diagnostics: N at each filter step ---
    n_total_persons = work['idpers'].nunique()
    # pw18: 1=changed job (same employer), 2=changed employer (same job),
    #       3=changed both, 4=no change. Stayers = {1, 4} (same employer).
    stayer_codes = {1, 4}
    ever_stayer_ids = set(work.loc[work['pw18'].isin(stayer_codes), 'idpers'].unique())
    n_ever_stayers = len(ever_stayer_ids)
    n_with_any_exposure = work.loc[
        work['idpers'].isin(ever_stayer_ids) & work[PRIMARY_EXPOSURE].notna(),
        'idpers'
    ].nunique()
    logger.info(f"  Diagnostic: {n_total_persons:,} total persons")
    logger.info(f"  Diagnostic: {n_ever_stayers:,} ever-stayers (pw18 in {{1,4}})")
    logger.info(f"  Diagnostic: {n_with_any_exposure:,} ever-stayers with any non-NaN exposure")

    # --- Treated: stayers where exposure first becomes positive ---
    # Relaxed definition: previous year was 0 or NaN (not yet positive),
    # current year is >0, while staying at same employer.
    adoption_rows = work[
        (work['pw18'].isin(stayer_codes)) &
        (work[PRIMARY_EXPOSURE] > 0) &
        ((work['exposure_lag'].isna()) | (work['exposure_lag'] == 0))
    ]
    first_adoption = (
        adoption_rows.groupby('idpers')['year'].min()
        .reset_index()
        .rename(columns={'year': 'adoption_year'})
    )
    first_adoption['treated'] = 1

    # --- Control: stayers whose exposure was never positive ---
    # Includes persons with all-NaN exposure (firm not in our job-ad data)
    # as well as confirmed-zero (firm in data but no AI applications).
    treated_ids = set(first_adoption['idpers'].unique())
    candidate_control_ids = ever_stayer_ids - treated_ids

    person_max_exp = (
        work[work['idpers'].isin(candidate_control_ids)]
        .groupby('idpers')[PRIMARY_EXPOSURE].max()
    )
    # Never-positive: max is 0 (confirmed zero) or NaN (no exposure data)
    never_positive_ids = set(
        person_max_exp[(person_max_exp == 0) | person_max_exp.isna()].index
    )
    n_confirmed_zero = int((person_max_exp == 0).sum())
    n_all_nan = int(person_max_exp.isna().sum())

    n_treated = len(first_adoption)
    n_control = len(never_positive_ids)
    logger.info(f"  Treated stayers (first positive exposure): {n_treated:,}")
    logger.info(f"  Control stayers (never positive): {n_control:,} "
                f"({n_confirmed_zero:,} confirmed-zero + {n_all_nan:,} all-NaN)")

    if n_treated < 10:
        logger.warning(f"  Too few treated stayers ({n_treated}), skipping")
        return
    if n_control < 10:
        logger.warning(f"  Too few control stayers ({n_control}), skipping")
        return

    # --- Build event panel ---
    event_window = 5

    # Treated panel: ±event_window years around adoption
    treated_panel = df[df['idpers'].isin(treated_ids)].merge(
        first_adoption[['idpers', 'adoption_year', 'treated']], on='idpers'
    ).copy()
    treated_panel['event_time'] = treated_panel['year'] - treated_panel['adoption_year']
    treated_panel = treated_panel[
        (treated_panel['event_time'] >= -event_window) &
        (treated_panel['event_time'] <= event_window)
    ]

    # Control panel: all available observations (no natural event_time;
    # event_time=0 is a placeholder — harmless because D_i=0 makes all
    # interaction terms zero in the regression)
    control_panel = df[df['idpers'].isin(never_positive_ids)].copy()
    control_panel['treated'] = 0
    control_panel['event_time'] = 0  # placeholder

    logger.info(f"  Event window: [±{event_window} years] around adoption")
    logger.info(f"  Treated person-years in window: {len(treated_panel):,}")
    logger.info(f"  Control person-years: {len(control_panel):,}")

    # Save summary table
    summary = treated_panel.groupby(['event_time', 'treated']).size().reset_index(name='n_person_years')
    summary_path = os.path.join(tables_dir, 'adoption_event_summary.csv')
    summary.to_csv(summary_path, index=False)
    logger.info(f"  Saved: {summary_path}")

    # --- Outcomes ---
    event_outcomes = {}
    if 'iwyn' in df.columns:
        event_outcomes['iwyn'] = get_var_label('iwyn')
    job_sec_var = next((v for v in JOB_SECURITY_CANDIDATES if v in df.columns), None)
    if job_sec_var:
        event_outcomes[job_sec_var] = get_var_label(job_sec_var)
    for var, lbl in POLITICAL_OUTCOMES_CONTINUOUS.items():
        if var in df.columns:
            event_outcomes[var] = lbl

    for var, lbl in event_outcomes.items():
        if var not in treated_panel.columns:
            continue
        n_valid = treated_panel[var].notna().sum()
        if n_valid < 20:
            logger.info(f"  [{var}] Too few treated observations ({n_valid}), skipping")
            continue

        logger.info(f"  [{var}] Running firm adoption event study (N treated={n_valid:,})")

        # Part A: descriptive — treated trajectory only, normalized to t=-1
        _adoption_means_plot(treated_panel, var, lbl, tables_dir, figures_dir)

        # Part B: DiD regression with never-adopting stayers as control
        reg_df = pd.concat([
            treated_panel[['idpers', 'year', 'event_time', 'treated', var]],
            control_panel[['idpers', 'year', 'event_time', 'treated', var]],
        ], ignore_index=True)
        _adoption_regression_plot(reg_df, var, lbl, tables_dir, figures_dir)


def _adoption_means_plot(treated_panel: pd.DataFrame, var: str, label: str,
                          tables_dir: str, figures_dir: str):
    """
    Descriptive: mean outcome for treated stayers by event_time, normalized
    to 0 at t=-1. Shows the unconditional trajectory around firm AI adoption.
    Control stayers have no natural event_time so are excluded from this plot.
    """
    valid = treated_panel.loc[treated_panel[var].notna()].copy()

    ref_mean = valid.loc[valid['event_time'] == -1, var].mean()
    if pd.isna(ref_mean):
        logger.warning(f"    [{var}] No observations at t=-1, skipping means plot")
        return

    stats = valid.groupby('event_time')[var].agg(
        mean='mean', sem='sem', count='count'
    ).reset_index()
    stats['mean_norm'] = stats['mean'] - ref_mean

    table_path = os.path.join(tables_dir, f'adoption_event_means_{var}.csv')
    stats.to_csv(table_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(stats['event_time'], stats['mean_norm'], marker='o', color='steelblue')
    ax.fill_between(
        stats['event_time'],
        stats['mean_norm'] - 1.96 * stats['sem'],
        stats['mean_norm'] + 1.96 * stats['sem'],
        alpha=0.2, color='steelblue'
    )
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Firm AI adoption (t=0)')
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5, label='Reference (t=-1)')
    ax.set_xlabel('Years Relative to Firm AI Adoption')
    ax.set_ylabel(f'Change in Mean {get_var_short_label(var)} (normalized to t=-1)')
    ax.set_title(f'Firm Adoption Event Study (Descriptive, Treated Stayers Only): '
                 f'{get_var_short_label(var)}')
    ax.legend(fontsize=9)

    for _, row_data in stats.iterrows():
        ax.annotate(f"n={int(row_data['count'])}",
                    (row_data['event_time'], row_data['mean_norm']),
                    textcoords='offset points', xytext=(0, 10),
                    fontsize=6, ha='center', alpha=0.6)

    fig.tight_layout()
    fig_path = os.path.join(figures_dir, f'adoption_event_means_{var}.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"    [{var}] Saved means plot: {fig_path}")


def _adoption_regression_plot(reg_df: pd.DataFrame, var: str, label: str,
                               tables_dir: str, figures_dir: str):
    """
    DiD regression: β_k from y_it = Σ_{k≠-1} β_k(D_i × 1[event_time=k])
    + person_FE + year_FE, estimated via demeaned OLS.

    Treated (D=1): stayers at AI-adopting firms, event_time = year - adoption_year.
    Control (D=0): never-adopting stayers; all D_i × 1(event_time=k) = 0
    regardless of their event_time placeholder, so they contribute only to
    person FE and year FE — tightening the counterfactual.

    β_k measures the average treatment effect on treated (ATT) at event_time k
    relative to k=-1, after removing person-level time-invariant differences
    and common year shocks.
    """
    valid = reg_df.loc[reg_df[var].notna()].copy()

    # Only use event_times from the treated group for interaction dummies
    treated_times = sorted(valid.loc[valid['treated'] == 1, 'event_time'].unique())
    ref_k = -1
    interact_times = [k for k in treated_times if k != ref_k]

    if len(interact_times) < 2:
        logger.warning(f"    [{var}] Too few event_time values for regression, skipping")
        return

    n_treated_persons = valid[valid['treated'] == 1]['idpers'].nunique()
    n_control_persons = valid[valid['treated'] == 0]['idpers'].nunique()
    if n_treated_persons < 5 or n_control_persons < 5:
        logger.warning(f"    [{var}] Too few persons: treated={n_treated_persons}, "
                       f"control={n_control_persons}, skipping")
        return

    # D_i × 1(event_time=k) — zero for all control rows by construction
    for k in interact_times:
        col = f'interact_k{k:+d}'
        valid[col] = (valid['treated'] * (valid['event_time'] == k)).astype(float)

    interact_cols = [f'interact_k{k:+d}' for k in interact_times]

    year_dummies = pd.get_dummies(valid['year'], prefix='yr', drop_first=True, dtype=float)
    year_dummies.index = valid.index
    yd_cols = year_dummies.columns.tolist()
    valid = pd.concat([valid, year_dummies], axis=1)

    all_x_cols = interact_cols + yd_cols

    try:
        for col in [var] + all_x_cols:
            person_means = valid.groupby('idpers')[col].transform('mean')
            valid[f'{col}_dm'] = valid[col] - person_means

        X_dm = valid[[f'{c}_dm' for c in all_x_cols]]
        y_dm = valid[f'{var}_dm']

        complete = X_dm.notna().all(axis=1) & y_dm.notna()
        X_dm = X_dm[complete]
        y_dm = y_dm[complete]
        groups = valid.loc[complete, 'idpers']

        if len(y_dm) < 20:
            logger.warning(f"    [{var}] Too few observations after demeaning ({len(y_dm)}), skipping")
            return

        model = sm.OLS(y_dm, X_dm).fit(
            cov_type='cluster', cov_kwds={'groups': groups}
        )

        interact_dm_cols = [f'interact_k{k:+d}_dm' for k in interact_times]
        coefs = pd.DataFrame({
            'event_time': interact_times,
            'coef': [model.params[c] for c in interact_dm_cols],
            'se':   [model.bse[c]    for c in interact_dm_cols],
        })
        ref_row = pd.DataFrame({'event_time': [ref_k], 'coef': [0.0], 'se': [0.0]})
        coefs = pd.concat([coefs, ref_row], ignore_index=True).sort_values('event_time')
        coefs['ci_lo'] = coefs['coef'] - 1.96 * coefs['se']
        coefs['ci_hi'] = coefs['coef'] + 1.96 * coefs['se']

        # Pre-trend F-test
        pre_cols = [f'interact_k{k:+d}_dm' for k in interact_times if k < ref_k]
        pre_trend_p = np.nan
        pre_trend_note = ''
        if len(pre_cols) >= 2:
            try:
                f_test = model.f_test([f'{c} = 0' for c in pre_cols])
                pre_trend_p = float(f_test.pvalue)
                pre_trend_note = f'Pre-trend F-test p={pre_trend_p:.3f}'
                logger.info(f"    [{var}] {pre_trend_note}")
            except Exception:
                pass

        coefs['pre_trend_f_p'] = pre_trend_p
        table_path = os.path.join(tables_dir, f'adoption_event_coefs_{var}.csv')
        coefs.to_csv(table_path, index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(coefs['event_time'], coefs['coef'], marker='o', color='darkgreen')
        ax.fill_between(coefs['event_time'], coefs['ci_lo'], coefs['ci_hi'],
                        alpha=0.2, color='darkgreen')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Firm AI adoption (t=0)')
        ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
        ax.set_xlabel('Years Relative to Firm AI Adoption')
        ax.set_ylabel(f'β_k  (ATT relative to never-adopting stayers, ref=t=-1)')
        title = (f'Firm Adoption DiD Event Study: {get_var_short_label(var)}\n'
                 f'Treated={n_treated_persons} persons, Control={n_control_persons} persons')
        if pre_trend_note:
            title += f'\n{pre_trend_note}'
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig_path = os.path.join(figures_dir, f'adoption_event_coefs_{var}.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"    [{var}] Saved coefficient plot: {fig_path}")

    except Exception as e:
        logger.warning(f"    [{var}] Regression failed: {e}")


def _switcher_means_plot(event_df: pd.DataFrame, var: str, label: str,
                         tables_dir: str, figures_dir: str):
    """
    Part A of the switcher event study: descriptive means plot.

    Plots mean outcome by event_time for treated (exposure increased at switch)
    and control (exposure stable/decreased) switchers, both series normalized
    to 0 at t=-1. Pre-trends visible at t<0; divergence at t>=0 shows effect.
    """
    valid = event_df.loc[event_df[var].notna()].copy()

    # Need the reference period (t=-1) for each group to normalize
    ref = valid[valid['event_time'] == -1].groupby('treated')[var].mean()
    if ref.empty or valid['event_time'].nunique() < 2:
        logger.warning(f"    [{var}] Insufficient event-time variation for means plot, skipping")
        return

    stats = valid.groupby(['event_time', 'treated'])[var].agg(
        mean='mean', sem='sem', count='count'
    ).reset_index()

    # Normalize to t=-1 within each group
    stats['ref_mean'] = stats['treated'].map(ref)
    stats['mean_norm'] = stats['mean'] - stats['ref_mean']

    # Save table
    table_path = os.path.join(tables_dir, f'switcher_event_means_{var}.csv')
    stats.to_csv(table_path, index=False)

    # Minimum N per group check
    for grp, grp_label in [(1, 'Treated'), (0, 'Control')]:
        grp_data = stats[stats['treated'] == grp]
        if len(grp_data) < 2:
            logger.warning(f"    [{var}] {grp_label} group has <2 event_time periods, skipping plot")
            return

    # Plot
    colors = {1: 'steelblue', 0: 'darkorange'}
    group_labels = {1: 'Exposure increased (treated)', 0: 'Exposure stable/decreased (control)'}

    fig, ax = plt.subplots(figsize=(10, 6))
    for grp in [0, 1]:
        grp_data = stats[stats['treated'] == grp].sort_values('event_time')
        if grp_data.empty:
            continue
        ax.plot(grp_data['event_time'], grp_data['mean_norm'],
                marker='o', color=colors[grp], label=group_labels[grp])
        ax.fill_between(
            grp_data['event_time'],
            grp_data['mean_norm'] - 1.96 * grp_data['sem'],
            grp_data['mean_norm'] + 1.96 * grp_data['sem'],
            alpha=0.15, color=colors[grp]
        )

    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Job switch (t=0)')
    ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
    ax.set_xlabel('Years Relative to Job Switch')
    ax.set_ylabel(f'Change in Mean {get_var_short_label(var)} (normalized to t=-1)')
    ax.set_title(f'Switcher Event Study (Descriptive): {get_var_short_label(var)}')
    ax.legend(fontsize=9)

    # Annotate n per group at each event_time
    for _, row_data in stats.iterrows():
        offset = 8 if row_data['treated'] == 1 else -14
        ax.annotate(f"n={int(row_data['count'])}",
                    (row_data['event_time'], row_data['mean_norm']),
                    textcoords='offset points', xytext=(0, offset),
                    fontsize=6, ha='center',
                    color=colors[row_data['treated']], alpha=0.7)

    fig.tight_layout()
    fig_path = os.path.join(figures_dir, f'switcher_event_means_{var}.png')
    fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
    plt.close(fig)
    logger.info(f"    [{var}] Saved means plot: {fig_path}")


def _switcher_regression_plot(event_df: pd.DataFrame, var: str, label: str,
                               tables_dir: str, figures_dir: str):
    """
    Part B of the switcher event study: DiD regression coefficient plot.

    Regression: y_it = Σ_{k≠-1} β_k × (D_i × 1(event_time=k)) + person_FE + year_FE
    where D_i = 1 for treated switchers (exposure increased at switch).

    β_k is the differential effect on treated vs control at event_time=k,
    relative to k=-1. Uses demeaned OLS with clustered SEs by person.

    Pre-trend test: joint F-test on k<0 coefficients (expect p>0.05 for valid design).
    """
    valid = event_df.loc[event_df[var].notna()].copy()

    event_times = sorted(valid['event_time'].unique())
    ref_k = -1
    interact_times = [k for k in event_times if k != ref_k]

    if len(interact_times) < 2:
        logger.warning(f"    [{var}] Too few event_time values for regression, skipping")
        return

    # Minimum persons per group
    n_treated = valid[valid['treated'] == 1]['idpers'].nunique()
    n_control = valid[valid['treated'] == 0]['idpers'].nunique()
    if n_treated < 5 or n_control < 5:
        logger.warning(f"    [{var}] Too few persons: treated={n_treated}, control={n_control}, skipping")
        return

    # Build D_i × 1(event_time=k) interaction regressors
    for k in interact_times:
        col = f'interact_k{k:+d}'
        valid[col] = (valid['treated'] * (valid['event_time'] == k)).astype(float)

    interact_cols = [f'interact_k{k:+d}' for k in interact_times]

    # Year dummies
    year_dummies = pd.get_dummies(valid['year'], prefix='yr', drop_first=True, dtype=float)
    year_dummies.index = valid.index
    yd_cols = year_dummies.columns.tolist()
    valid = pd.concat([valid, year_dummies], axis=1)

    all_x_cols = interact_cols + yd_cols

    try:
        # Person-demean outcome and all regressors
        for col in [var] + all_x_cols:
            person_means = valid.groupby('idpers')[col].transform('mean')
            valid[f'{col}_dm'] = valid[col] - person_means

        X_dm = valid[[f'{c}_dm' for c in all_x_cols]]
        y_dm = valid[f'{var}_dm']

        # Drop rows where any column is NaN after demeaning
        complete = X_dm.notna().all(axis=1) & y_dm.notna()
        X_dm = X_dm[complete]
        y_dm = y_dm[complete]
        groups = valid.loc[complete, 'idpers']

        if len(y_dm) < 20:
            logger.warning(f"    [{var}] Too few observations after demeaning ({len(y_dm)}), skipping")
            return

        model = sm.OLS(y_dm, X_dm).fit(
            cov_type='cluster', cov_kwds={'groups': groups}
        )

        # Extract interaction coefficients only (not year dummies)
        interact_dm_cols = [f'interact_k{k:+d}_dm' for k in interact_times]
        coefs = pd.DataFrame({
            'event_time': interact_times,
            'coef': [model.params[c] for c in interact_dm_cols],
            'se':   [model.bse[c]    for c in interact_dm_cols],
        })
        # Add reference row (k=-1, β=0 by construction)
        ref_row = pd.DataFrame({'event_time': [ref_k], 'coef': [0.0], 'se': [0.0]})
        coefs = pd.concat([coefs, ref_row], ignore_index=True).sort_values('event_time')
        coefs['ci_lo'] = coefs['coef'] - 1.96 * coefs['se']
        coefs['ci_hi'] = coefs['coef'] + 1.96 * coefs['se']

        # Pre-trend F-test: joint significance of k < -1 coefficients
        pre_cols = [f'interact_k{k:+d}_dm' for k in interact_times if k < ref_k]
        pre_trend_p = np.nan
        pre_trend_note = ''
        if len(pre_cols) >= 2:
            try:
                f_test = model.f_test([f'{c} = 0' for c in pre_cols])
                pre_trend_p = float(f_test.pvalue)
                pre_trend_note = f'Pre-trend F-test p={pre_trend_p:.3f}'
                logger.info(f"    [{var}] {pre_trend_note}")
            except Exception:
                pass

        coefs['pre_trend_f_p'] = pre_trend_p
        table_path = os.path.join(tables_dir, f'switcher_event_coefs_{var}.csv')
        coefs.to_csv(table_path, index=False)

        # Coefficient plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(coefs['event_time'], coefs['coef'], marker='o', color='steelblue')
        ax.fill_between(coefs['event_time'], coefs['ci_lo'], coefs['ci_hi'],
                        alpha=0.2, color='steelblue')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Job switch (t=0)')
        ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
        ax.set_xlabel('Years Relative to Job Switch')
        ax.set_ylabel(f'β_k  (differential effect on treated vs control, ref=t=-1)')
        title = f'Switcher DiD Event Study: {get_var_short_label(var)}'
        if pre_trend_note:
            title += f'\n{pre_trend_note}'
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)

        fig.tight_layout()
        fig_path = os.path.join(figures_dir, f'switcher_event_coefs_{var}.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"    [{var}] Saved coefficient plot: {fig_path}")

    except Exception as e:
        logger.warning(f"    [{var}] Regression failed: {e}")


def _job_switcher_causal_event_study(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """
    Section 7.6: Causal event study among job switchers.

    Identification: Among persons who switch employers (pw18 ∈ {2, 3}; 2=changed
    employer same job, 3=changed both employer and job), exposure at the new firm
    varies quasi-randomly. Treatment (D_i=1) is defined as PRIMARY_EXPOSURE
    increasing at the first switch (exposure_t > exposure_{t-1}).
    Control (D_i=0) is exposure stable or decreased at the switch.

    Two analyses per outcome:
      Part A (descriptive): mean outcomes by event_time × treatment, normalized to t=-1.
      Part B (regression):  β_k from y_it = Σ_{k≠-1} β_k(D_i × 1[event_time=k])
                            + person_FE + year_FE, estimated via demeaned OLS.
    """
    needed_cols = ['idpers', 'year', 'pw18', PRIMARY_EXPOSURE]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        logger.warning(f"  Skipping switcher event study: missing columns {missing}")
        return

    work = df[needed_cols].copy()
    work['pw18'] = work['pw18'].where(work['pw18'] > 0, np.nan)

    # Lag exposure within person to compute change at each transition
    work = work.sort_values(['idpers', 'year'])
    work['exposure_lag'] = work.groupby('idpers')[PRIMARY_EXPOSURE].shift(1)
    work['exposure_delta'] = work[PRIMARY_EXPOSURE] - work['exposure_lag']

    # Identify first job switch per person (pw18 ∈ {2, 3})
    switches = work[work['pw18'].isin([2, 3]) &
                   work[PRIMARY_EXPOSURE].notna() &
                   work['exposure_lag'].notna()].copy()

    if len(switches) == 0:
        logger.warning("  No qualifying job switches with valid exposure data, skipping")
        return

    first_switch = switches.groupby('idpers').first().reset_index()[
        ['idpers', 'year', 'exposure_delta']
    ].rename(columns={'year': 'switch_year'})

    # Treatment: exposure increased at first switch
    first_switch['treated'] = (first_switch['exposure_delta'] > 0).astype(int)

    n_treated = first_switch['treated'].sum()
    n_control = (first_switch['treated'] == 0).sum()
    logger.info(f"  Job switchers with valid exposure: {len(first_switch):,}")
    logger.info(f"  Treated (exposure increased): {n_treated:,}")
    logger.info(f"  Control (exposure stable/decreased): {n_control:,}")

    if n_treated < 10 or n_control < 10:
        logger.warning(f"  Insufficient switchers (need ≥10 per group): "
                       f"treated={n_treated}, control={n_control}. Skipping.")
        return

    # Build event-time panel for all switchers (full df, not just switch rows)
    event_df = df.merge(first_switch[['idpers', 'switch_year', 'treated']], on='idpers', how='inner')
    event_df['event_time'] = event_df['year'] - event_df['switch_year']

    event_window = 5
    event_df = event_df[
        (event_df['event_time'] >= -event_window) &
        (event_df['event_time'] <= event_window)
    ].copy()

    logger.info(f"  Event window: [±{event_window} years] around first switch")
    logger.info(f"  Person-years in event window: {len(event_df):,}")

    # Save switcher summary table
    summary = event_df.groupby(['event_time', 'treated']).size().reset_index(name='n_person_years')
    summary_path = os.path.join(tables_dir, 'switcher_event_summary.csv')
    summary.to_csv(summary_path, index=False)
    logger.info(f"  Saved: {summary_path}")

    # Outcomes to analyze
    event_outcomes = {}
    if 'iwyn' in df.columns:
        event_outcomes['iwyn'] = get_var_label('iwyn')
    job_sec_var = next((v for v in JOB_SECURITY_CANDIDATES if v in df.columns), None)
    if job_sec_var:
        event_outcomes[job_sec_var] = get_var_label(job_sec_var)
    for var, lbl in POLITICAL_OUTCOMES_CONTINUOUS.items():
        if var in df.columns:
            event_outcomes[var] = lbl

    for var, lbl in event_outcomes.items():
        n_valid = event_df[var].notna().sum()
        if n_valid < 20:
            logger.info(f"  [{var}] Too few observations ({n_valid}), skipping")
            continue

        logger.info(f"  [{var}] Running switcher event study (N={n_valid:,})")
        _switcher_means_plot(event_df, var, lbl, tables_dir, figures_dir)
        _switcher_regression_plot(event_df, var, lbl, tables_dir, figures_dir)


def _job_switch_exposure_matrix(df: pd.DataFrame, tables_dir: str):
    """
    2×2-style matrix of job mobility × exposure change at the person-year transition level.

    Rows:  pw18 employer/job-change status (1=changed job same employer,
           2=changed employer same job, 3=changed both, 4=no change)
    Cols:  whether foy exposure changed year-on-year (changed / unchanged)

    Only includes transitions where both dimensions are observed (pw18 ∈ {1,2,3,4}
    and exposure is non-NaN in both t and t-1).
    """
    needed = ['idpers', 'year', 'pw18', PRIMARY_EXPOSURE]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        logger.warning(f"  Skipping job-switch matrix: missing columns {missing}")
        return
    if 'pw18' not in df.columns:
        logger.warning("  Skipping job-switch matrix: pw18 not in data")
        return

    work = df[needed].copy()

    # Recode negative SHP sentinels to NaN
    work['pw18'] = work['pw18'].where(work['pw18'] > 0, np.nan)

    # Year-on-year exposure change within person
    work = work.sort_values(['idpers', 'year'])
    work['exposure_lag'] = work.groupby('idpers')[PRIMARY_EXPOSURE].shift(1)
    work['exposure_changed'] = (
        (work[PRIMARY_EXPOSURE] != work['exposure_lag'])
        & work[PRIMARY_EXPOSURE].notna()
        & work['exposure_lag'].notna()
    )

    # pw18: 1=changed job (same employer), 2=changed employer (same job),
    #       3=changed both, 4=no change
    valid = work[
        work['pw18'].isin([1, 2, 3, 4])
        & work[PRIMARY_EXPOSURE].notna()
        & work['exposure_lag'].notna()
    ].copy()

    if len(valid) < 10:
        logger.warning(f"  Too few valid transitions ({len(valid)}) for job-switch matrix, skipping")
        return

    logger.info(f"  Valid person-year transitions: {len(valid):,} ({valid['idpers'].nunique():,} persons)")

    pw18_labels = {
        1: 'Changed job (same employer)',
        2: 'Changed employer (same job)',
        3: 'Changed both',
        4: 'No change',
    }
    valid['job_label'] = valid['pw18'].map(pw18_labels)
    valid['exp_label'] = valid['exposure_changed'].map(
        {True: 'Exposure changed', False: 'Exposure unchanged'}
    )

    # Count matrix
    counts = pd.crosstab(valid['job_label'], valid['exp_label'], margins=True, margins_name='Total')

    # Percentage of all valid transitions
    pct = pd.crosstab(valid['job_label'], valid['exp_label'], normalize='all').mul(100).round(1)

    # Row-percentage (what share of each job-mobility category had exposure change)
    row_pct = pd.crosstab(valid['job_label'], valid['exp_label'], normalize='index').mul(100).round(1)

    logger.info(f"\n  Counts:\n{counts.to_string()}")
    logger.info(f"\n  % of all transitions:\n{pct.to_string()}")
    logger.info(f"\n  Row % (within job-mobility category):\n{row_pct.to_string()}")

    # Save all three tables
    counts.to_csv(os.path.join(tables_dir, 'job_switch_exposure_matrix_counts.csv'))
    pct.to_csv(os.path.join(tables_dir, 'job_switch_exposure_matrix_pct_total.csv'))
    row_pct.to_csv(os.path.join(tables_dir, 'job_switch_exposure_matrix_pct_row.csv'))
    logger.info(f"  Saved: job_switch_exposure_matrix_counts/pct_total/pct_row.csv")


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
        event_outcomes['iwyn'] = get_var_label('iwyn')
    job_sec_var = next((v for v in JOB_SECURITY_CANDIDATES if v in df.columns), None)
    if job_sec_var:
        event_outcomes[job_sec_var] = get_var_label(job_sec_var)
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
        ax.set_ylabel(f'Mean {get_var_short_label(var)}')
        ax.set_title(f'Event Study: {get_var_short_label(var)} Around First AI Exposure')
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
        logger.info(f"  Event study {get_var_label(var)}: saved {fig_path}")


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

    # Within-person variation for outcomes (variance decomposition)
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
            pct_b = 100 * between_var / total_var
            pct_w = 100 * within_var / total_var
            variation_rows.append({
                'variable': var,
                'label': get_var_label(var),
                'total_variance': total_var,
                'between_person_variance': between_var,
                'within_person_variance': within_var,
                'pct_between': pct_b,
                'pct_within': pct_w,
                'pct_sum': pct_b + pct_w,
                # Between component is unweighted (each person contributes one
                # person-mean regardless of how many waves they appear in).
                # Within component is the unweighted average of per-person variances.
                # In an unbalanced panel these two components do not add to total
                # variance, so pct_sum ≠ 100%. Use pct_sum as a diagnostic: large
                # deviations from 100 indicate high imbalance (e.g. many single-wave
                # persons inflate between-variance relative to total).
                'note': (
                    'Unbalanced panel: between and within components are unweighted '
                    'averages, so pct_between + pct_within ≠ 100%. '
                    'pct_sum < 100 means single-wave persons reduce estimated within-variance; '
                    'pct_sum > 100 means persons with many waves inflate between-variance.'
                ),
            })
            logger.info(f"  {get_var_label(var)}: between={pct_b:.1f}%, within={pct_w:.1f}% "
                        f"(sum={pct_b + pct_w:.1f}%)")

    if variation_rows:
        var_df = pd.DataFrame(variation_rows)
        var_path = os.path.join(tables_dir, 'within_person_variance_decomposition.csv')
        var_df.to_csv(var_path, index=False)
        logger.info(f"  Saved: {var_path}")


def _within_person_income_changes(df: pd.DataFrame, tables_dir: str, figures_dir: str):
    """Analyze within-person income changes conditional on exposure changes."""
    logger.info("  Analyzing within-person income changes:")

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


# =============================================================================
# Section 8: Stacked Long-Difference (Causal)
# =============================================================================

def section_8_stacked_long_diff(df: pd.DataFrame, output_dir: str,
                                 job_security_var: Optional[str] = None):
    """
    Section 8: Stacked Long-Difference Estimator for Firm AI Adoption.

    For each adoption cohort g (year of first positive AI exposure while staying),
    builds a clean sub-experiment:
      - Treated: stayers first exposed in year g
      - Control: stayers not yet treated by year g+L (clean counterfactual)
      - Long difference: ΔY_i = Y_{g+L} - Y_{g-1}
    Stacks all cohort-specific long-differences and estimates:
      ΔY_ig = α_g + β × D_i + ε_ig   (cohort FE + treated indicator)

    Runs for multiple horizons L = 1, 2, 3 to show how effects build over time.
    Clustered SEs at the person level.

    pw18 coding: 1=changed job (same employer), 2=changed employer (same job),
                 3=changed both, 4=no change. Stayers = {1, 4}.
    """
    logger.info("\n" + "=" * 70)
    logger.info("SECTION 8: STACKED LONG-DIFFERENCE (CAUSAL)")
    logger.info("=" * 70)

    tables_dir = os.path.join(output_dir, 'tables')
    figures_dir = os.path.join(output_dir, 'figures')

    needed_cols = ['idpers', 'year', 'pw18', PRIMARY_EXPOSURE]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        logger.warning(f"  Skipping stacked long-diff: missing columns {missing}")
        return

    # --- Identify adoption cohorts among stayers ---
    work = df[['idpers', 'year', 'pw18', PRIMARY_EXPOSURE]].copy()
    work['pw18'] = work['pw18'].where(work['pw18'] > 0, np.nan)
    work = work.sort_values(['idpers', 'year'])
    work['exposure_lag'] = work.groupby('idpers')[PRIMARY_EXPOSURE].shift(1)

    stayer_codes = {1, 4}
    # First year of positive exposure while staying (lag was 0 or NaN)
    adoption_rows = work[
        (work['pw18'].isin(stayer_codes)) &
        (work[PRIMARY_EXPOSURE] > 0) &
        ((work['exposure_lag'].isna()) | (work['exposure_lag'] == 0))
    ]
    first_adoption = (
        adoption_rows.groupby('idpers')['year'].min()
        .reset_index()
        .rename(columns={'year': 'adoption_year'})
    )
    treated_ids = set(first_adoption['idpers'].unique())

    # All stayer person-years (for control pool)
    ever_stayer_ids = set(work.loc[work['pw18'].isin(stayer_codes), 'idpers'].unique())

    n_treated = len(treated_ids)
    logger.info(f"  Adoption cohorts identified: {n_treated:,} treated stayers")
    logger.info(f"  Total stayer pool: {len(ever_stayer_ids):,} persons")

    if n_treated < 10:
        logger.warning(f"  Too few treated stayers ({n_treated}), skipping")
        return

    cohort_counts = first_adoption['adoption_year'].value_counts().sort_index()
    logger.info(f"  Cohorts by year:\n{cohort_counts.to_string()}")

    # --- Build outcomes list ---
    outcomes = {}
    if 'iwyn' in df.columns:
        outcomes['iwyn'] = get_var_short_label('iwyn')
    if job_security_var is None:
        job_security_var = next((v for v in JOB_SECURITY_CANDIDATES if v in df.columns), None)
    if job_security_var and job_security_var in df.columns:
        outcomes[job_security_var] = get_var_short_label(job_security_var)
    for var in POLITICAL_OUTCOMES_CONTINUOUS:
        if var in df.columns:
            outcomes[var] = get_var_short_label(var)

    if not outcomes:
        logger.warning("  No valid outcome variables found, skipping")
        return

    # --- Run for multiple horizons ---
    horizons = [1, 2, 3]
    all_results = []

    for L in horizons:
        logger.info(f"\n  --- Horizon L={L} (ΔY = Y_{{g+{L}}} - Y_{{g-1}}) ---")

        stacked_rows = []

        for g, cohort_df in first_adoption.groupby('adoption_year'):
            cohort_treated_ids = set(cohort_df['idpers'].unique())

            # Control for cohort g: stayers not yet treated by year g+L
            # (their adoption_year > g+L, or they are never treated)
            not_yet_treated = ever_stayer_ids - treated_ids  # never treated
            late_treated = set(
                first_adoption.loc[
                    first_adoption['adoption_year'] > g + L, 'idpers'
                ].unique()
            )
            control_ids_g = not_yet_treated | late_treated

            if len(control_ids_g) < 5:
                logger.info(f"    Cohort {g}: too few controls ({len(control_ids_g)}), skipping")
                continue

            # For treated: need Y at g-1 and g+L
            for idp in cohort_treated_ids:
                person_data = df[df['idpers'] == idp]
                y_pre = person_data.loc[person_data['year'] == g - 1]
                y_post = person_data.loc[person_data['year'] == g + L]
                if len(y_pre) == 0 or len(y_post) == 0:
                    continue
                row = {'idpers': idp, 'cohort': g, 'treated': 1}
                for var in outcomes:
                    pre_val = y_pre[var].values[0] if var in y_pre.columns else np.nan
                    post_val = y_post[var].values[0] if var in y_post.columns else np.nan
                    row[f'{var}_pre'] = pre_val
                    row[f'{var}_post'] = post_val
                    row[f'{var}_diff'] = post_val - pre_val if pd.notna(pre_val) and pd.notna(post_val) else np.nan
                stacked_rows.append(row)

            # For controls: same g-1 and g+L
            for idp in control_ids_g:
                person_data = df[df['idpers'] == idp]
                y_pre = person_data.loc[person_data['year'] == g - 1]
                y_post = person_data.loc[person_data['year'] == g + L]
                if len(y_pre) == 0 or len(y_post) == 0:
                    continue
                row = {'idpers': idp, 'cohort': g, 'treated': 0}
                for var in outcomes:
                    pre_val = y_pre[var].values[0] if var in y_pre.columns else np.nan
                    post_val = y_post[var].values[0] if var in y_post.columns else np.nan
                    row[f'{var}_pre'] = pre_val
                    row[f'{var}_post'] = post_val
                    row[f'{var}_diff'] = post_val - pre_val if pd.notna(pre_val) and pd.notna(post_val) else np.nan
                stacked_rows.append(row)

        if not stacked_rows:
            logger.warning(f"    L={L}: No valid observations after stacking, skipping")
            continue

        stacked = pd.DataFrame(stacked_rows)
        n_t = stacked['treated'].sum()
        n_c = (stacked['treated'] == 0).sum()
        n_cohorts = stacked['cohort'].nunique()
        logger.info(f"    Stacked dataset: {len(stacked):,} obs, "
                    f"{n_t:,} treated, {n_c:,} control, {n_cohorts} cohorts")

        # Save stacked dataset
        stacked_path = os.path.join(tables_dir, f'stacked_long_diff_L{L}.csv')
        stacked.to_csv(stacked_path, index=False)
        logger.info(f"    Saved: {stacked_path}")

        # --- Estimate β for each outcome ---
        for var, var_label in outcomes.items():
            diff_col = f'{var}_diff'
            valid = stacked[[diff_col, 'treated', 'cohort', 'idpers']].dropna(subset=[diff_col])

            n_valid_t = valid['treated'].sum()
            n_valid_c = (valid['treated'] == 0).sum()
            if n_valid_t < 5 or n_valid_c < 5:
                logger.info(f"    [{var}] L={L}: too few obs (treated={n_valid_t}, "
                           f"control={n_valid_c}), skipping")
                continue

            # ΔY_ig = α_g + β × D_i + ε_ig
            cohort_dummies = pd.get_dummies(valid['cohort'], prefix='cohort',
                                            drop_first=True, dtype=float)
            X = pd.concat([
                valid[['treated']].reset_index(drop=True),
                cohort_dummies.reset_index(drop=True)
            ], axis=1)
            y = valid[diff_col].reset_index(drop=True)
            groups = valid['idpers'].reset_index(drop=True)

            try:
                model = sm.OLS(y, sm.add_constant(X)).fit(
                    cov_type='cluster', cov_kwds={'groups': groups}
                )
                beta = model.params['treated']
                se = model.bse['treated']
                p = model.pvalues['treated']
                ci_lo = beta - 1.96 * se
                ci_hi = beta + 1.96 * se

                all_results.append({
                    'horizon': L,
                    'outcome': var,
                    'outcome_label': var_label,
                    'beta': beta,
                    'se': se,
                    'p': p,
                    'ci_lo': ci_lo,
                    'ci_hi': ci_hi,
                    'stars': significance_stars(p),
                    'n_treated': int(n_valid_t),
                    'n_control': int(n_valid_c),
                    'n_cohorts': int(valid['cohort'].nunique()),
                    'mean_diff_treated': valid.loc[valid['treated'] == 1, diff_col].mean(),
                    'mean_diff_control': valid.loc[valid['treated'] == 0, diff_col].mean(),
                })

                logger.info(f"    [{var}] L={L}: β={beta:.4f} (SE={se:.4f}){significance_stars(p)} "
                           f"N_t={int(n_valid_t)}, N_c={int(n_valid_c)}")

            except Exception as e:
                logger.warning(f"    [{var}] L={L}: regression failed: {e}")

    if not all_results:
        logger.warning("  No results from any horizon, skipping output")
        return

    # --- Save combined results table ---
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(tables_dir, 'stacked_long_diff_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"\n  Saved combined results: {results_path}")

    # --- Coefficient-by-horizon plot for each outcome ---
    for var, var_label in outcomes.items():
        var_results = results_df[results_df['outcome'] == var]
        if len(var_results) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            var_results['horizon'], var_results['beta'],
            yerr=1.96 * var_results['se'],
            marker='o', color='darkgreen', capsize=5, linewidth=2, markersize=8
        )
        ax.axhline(0, color='grey', linestyle=':', alpha=0.5)
        ax.set_xlabel('Horizon L (years after adoption)')
        ax.set_ylabel(f'β (ATT on Δ{var_label})')
        ax.set_xticks(horizons)

        # Annotate with N and stars
        for _, row in var_results.iterrows():
            ax.annotate(
                f"N={row['n_treated']+row['n_control']}{row['stars']}",
                (row['horizon'], row['beta']),
                textcoords='offset points', xytext=(0, 12),
                fontsize=8, ha='center', alpha=0.7
            )

        ax.set_title(f'Stacked Long-Difference: {var_label}\n'
                     f'β = effect of firm AI adoption on ΔY (cohort FE, clustered SE)',
                     fontsize=10)
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, f'stacked_long_diff_{var}.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  [{var}] Saved horizon plot: {fig_path}")

    # --- Summary heatmap: outcomes × horizons ---
    if len(results_df) >= 3:
        pivot = results_df.pivot(index='outcome_label', columns='horizon', values='beta')
        stars_pivot = results_df.pivot(index='outcome_label', columns='horizon', values='stars')

        fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) * 0.6)))
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdBu_r', center=0,
                    linewidths=0.5, ax=ax, cbar_kws={'label': 'β (ATT)'})

        # Overlay stars
        for i, outcome in enumerate(pivot.index):
            for j, h in enumerate(pivot.columns):
                star = stars_pivot.loc[outcome, h] if pd.notna(stars_pivot.loc[outcome, h]) else ''
                if star:
                    ax.text(j + 0.5, i + 0.75, star, ha='center', va='center',
                            fontsize=10, color='black', fontweight='bold')

        ax.set_xlabel('Horizon L (years)')
        ax.set_ylabel('')
        ax.set_title('Stacked Long-Difference: ATT by Outcome × Horizon', fontsize=11)
        fig.tight_layout()
        fig_path = os.path.join(figures_dir, 'stacked_long_diff_heatmap.png')
        fig.savefig(fig_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT)
        plt.close(fig)
        logger.info(f"  Saved heatmap: {fig_path}")


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
    tables_path = os.path.join(output_dir, 'tables')
    figures_path = os.path.join(output_dir, 'figures')
    n_tables = len([f for f in os.listdir(tables_path) if f.endswith('.csv')]) if os.path.isdir(tables_path) else 0
    n_figures = len([f for f in os.listdir(figures_path) if f.endswith('.png')]) if os.path.isdir(figures_path) else 0
    logger.info(f"  Tables generated: {n_tables}")
    logger.info(f"  Figures generated: {n_figures}")
    logger.info("=" * 70)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_single_level(df: pd.DataFrame, output_dir: str, level_code: str,
                      exposure_metric: str = 'hampole',
                      job_security_var: Optional[str] = None):
    """
    Run the full EDA for one exposure level and one exposure metric.

    Sets PRIMARY_EXPOSURE to the appropriate suffixed column, recreates
    has_exposure/exposed_ever indicators, and runs all analysis sections.

    Args:
        df: Full SHP DataFrame (loaded once, reused across levels)
        output_dir: Base output directory; outputs go to {output_dir}/{metric}/{level_code}/
        level_code: Exposure level code (e.g., 'foy', 'oy', 'f')
        exposure_metric: Key into EXPOSURE_METRIC_BASES selecting which base column to use
        job_security_var: Job security variable name
    """
    global PRIMARY_EXPOSURE

    suffix = EXPOSURE_LEVEL_SUFFIXES[level_code]
    base_col = EXPOSURE_METRIC_BASES[exposure_metric]
    col_name = f'{base_col}{suffix}'
    label = EXPOSURE_LEVEL_LABELS[level_code]

    # Validate column exists
    if col_name not in df.columns:
        logger.warning(f"Skipping level '{level_code}' ({label}), metric '{exposure_metric}': "
                       f"column '{col_name}' not found in data")
        return

    # Set the module-level PRIMARY_EXPOSURE for this run
    PRIMARY_EXPOSURE = col_name

    # Create output subdirectory: {output_dir}/{metric}/{level_code}/
    level_dir = os.path.join(output_dir, exposure_metric, level_code)
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
    logger.info(f"STAGE 7 SHP EDA -- Level: {label} ({level_code}), Metric: {exposure_metric}")
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
    section_1_data_structure(df, level_dir, job_security_var)
    section_2_pairwise_exploration(df, level_dir, job_security_var)
    regression_results = section_3_regression_grid(df, level_dir, job_security_var)
    section_4_collective_summary(df, regression_results, level_dir, job_security_var)
    section_5_multiple_testing(regression_results, level_dir)
    section_6_intensive_margin(df, level_dir, job_security_var)
    section_7_panel_dynamics(df, level_dir, job_security_var)
    section_8_stacked_long_diff(df, level_dir, job_security_var)
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
    logger.info(f"Exposure metric: {args.exposure_metric} ({EXPOSURE_METRIC_BASES[args.exposure_metric]})")
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

        run_single_level(df, args.output_dir, level_code, args.exposure_metric, job_security_var)

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
Metrics (--exposure-metric) and levels (--exposure-levels) are independent dimensions.

Examples:
  # Default: Hampole intensity-adjusted metric, foy level
  # Output: Data/shp_eda/hampole/foy/
  python3 stage_7_shp_eda.py --output-dir Data/shp_eda/

  # Pre-intensity Hampole (no log-intensity scaling), foy level
  # Output: Data/shp_eda/hampole_base/foy/
  python3 stage_7_shp_eda.py --output-dir Data/shp_eda/ --exposure-metric hampole_base

  # Binary metric, occupation×year level
  # Output: Data/shp_eda/binary/oy/
  python3 stage_7_shp_eda.py --output-dir Data/shp_eda/ --exposure-metric binary --exposure-levels oy

  # Run all 4 metrics at foy level (run separately, each gets its own folder)
  # Outputs: Data/shp_eda/hampole/foy/, Data/shp_eda/hampole_base/foy/, etc.
  python3 stage_7_shp_eda.py --output-dir Data/shp_eda/ --exposure-metric hampole      --exposure-levels foy
  python3 stage_7_shp_eda.py --output-dir Data/shp_eda/ --exposure-metric hampole_base --exposure-levels foy
  python3 stage_7_shp_eda.py --output-dir Data/shp_eda/ --exposure-metric binary       --exposure-levels foy
  python3 stage_7_shp_eda.py --output-dir Data/shp_eda/ --exposure-metric binary_base  --exposure-levels foy
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default=str(base_dir / 'Data' / 'shp_exposure' / 'shp_exposure_isco4d.csv'),
        help='Path to SHP exposure file. Options: shp_exposure_isco4d.csv (default, exact 4d matches), '
             'shp_exposure_isco4d_fallback.csv (hierarchical 4d→3d→2d), '
             'shp_exposure_isco3d.csv, shp_exposure_isco2d.csv'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(base_dir / 'Data' / 'shp_eda'),
        help='Output directory for tables and figures (default: Data/shp_eda/)'
    )

    parser.add_argument(
        '--exposure-metric',
        type=str,
        choices=list(EXPOSURE_METRIC_BASES.keys()),
        default='hampole',
        help=(
            'Which of the 4 exposure metrics to use (2x2: method x intensity). '
            'Independent of --exposure-levels. '
            'hampole: Hampole share × log(1+N_apps) [default]; '
            'hampole_base: Hampole share only, pre-intensity; '
            'binary: binary exposure × log(1+N_apps); '
            'binary_base: binary exposure only, pre-intensity. '
            'Outputs go to {output_dir}/{metric}/{level}/ so each metric has its own folder.'
        )
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
