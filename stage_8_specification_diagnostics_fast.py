"""
SHP PANEL SPECIFICATION DIAGNOSTICS - STREAMLINED VERSION
==========================================================

Fits 3 specifications on key outcomes (income + subset of political vars)
using memory-efficient approaches.

Output: Regression tables, validity scores, income reversal diagnostics
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from datetime import datetime
import json
import logging

import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path('/Users/bradyallardice/Dropbox/Allardice/KurerAllardice2024/10CompaniesPOCAI/Data')
OUTPUT_DIR = DATA_DIR / 'Testing/stage_8/specification_diagnostics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
OUTCOME_VARS = {
    'economic': ['outcome_log_income', 'outcome_job_insecurity'],
    'political': ['outcome_leftright', 'outcome_nativism'],  # Just 2 for speed
}

EXPOSURE_VAR = 'hampole_ai_exposure_avg_foy'
CONTROLS = ['age_centered', 'age_squared', 'female', 'education']

# ============================================================================
# LOAD & PREPARE
# ============================================================================

logger.info("Loading SHP panel data...")
df = pd.read_csv(DATA_DIR / 'shp_panel_prepared.csv')

# Create FE indicators
if 'firm_year' not in df.columns:
    df['firm_year'] = df['firm_id'].astype(str) + '_' + df['year'].astype(str)
if 'occ_year' not in df.columns:
    df['occ_year'] = df['isco08_4d'].astype(str) + '_' + df['year'].astype(str)
if 'person_year' not in df.columns:
    df['person_year'] = df['idpers'].astype(str) + '_' + df['year'].astype(str)

df['firm'] = df['firm_id'].astype(str)
df['occupation'] = df['isco08_4d'].astype(str)

logger.info(f"Data shape: {df.shape}")
logger.info(f"Persons: {df['idpers'].nunique()}, Firm-years: {df['firm_year'].nunique()}")

# ============================================================================
# SPECIFICATION FITTING FUNCTION
# ============================================================================

def fit_specifications(outcome_var: str) -> dict:
    """Fit all 3 specs for a given outcome. Returns results dict."""

    # Get subset with complete cases
    cols_needed = ['idpers', 'firm_year', 'occ_year', 'person_year', 'firm',
                   'occupation', 'year', EXPOSURE_VAR, outcome_var] + CONTROLS
    df_sub = df[cols_needed].dropna()

    if len(df_sub) < 100:
        logger.warning(f"Sample too small for {outcome_var}: N={len(df_sub)}")
        return None

    logger.info(f"{outcome_var}: N={len(df_sub)}")

    results = {}
    controls_str = ' + '.join(CONTROLS)

    # SPEC 1: Additive FE
    try:
        logger.info(f"  Spec 1 (Additive FE)...")
        formula = f"{outcome_var} ~ C(idpers) + C(firm) + C(year) + C(occupation) + {EXPOSURE_VAR} + {controls_str}"
        m1 = smf.ols(formula, data=df_sub).fit(cov_type='cluster', cov_kwds={'groups': df_sub['idpers']})
        results['spec1'] = m1
        logger.info(f"    R²={m1.rsquared:.4f}, DoF={m1.df_resid}")
    except Exception as e:
        logger.error(f"    FAILED: {e}")

    # SPEC 2: Firm-Year + Occ-Year FE
    try:
        logger.info(f"  Spec 2 (Firm-Year + Occ-Year FE)...")
        formula = f"{outcome_var} ~ C(idpers) + C(firm_year) + C(occ_year) + {EXPOSURE_VAR} + {controls_str}"
        m2 = smf.ols(formula, data=df_sub).fit(cov_type='cluster', cov_kwds={'groups': df_sub['idpers']})
        results['spec2'] = m2
        logger.info(f"    R²={m2.rsquared:.4f}, DoF={m2.df_resid}")
    except Exception as e:
        logger.error(f"    FAILED: {e}")

    # SPEC 3: Saturated FE
    try:
        logger.info(f"  Spec 3 (Saturated FE)...")
        formula = f"{outcome_var} ~ C(idpers) + C(year) + C(firm_year) + C(occ_year) + C(person_year) + {EXPOSURE_VAR} + {controls_str}"
        m3 = smf.ols(formula, data=df_sub).fit(cov_type='cluster', cov_kwds={'groups': df_sub['idpers']})
        results['spec3'] = m3
        logger.info(f"    R²={m3.rsquared:.4f}, DoF={m3.df_resid}")
    except Exception as e:
        logger.error(f"    FAILED: {e}")

    return results


# ============================================================================
# DIAGNOSTICS & VALIDITY SCORING
# ============================================================================

def diagnose_results(results: dict, outcome_var: str) -> dict:
    """Run diagnostics on fitted specifications."""

    diag = {'outcome': outcome_var}

    # Extract coefficient info
    for spec_name in ['spec1', 'spec2', 'spec3']:
        if spec_name not in results:
            diag[spec_name] = {'status': 'FAILED'}
            continue

        res = results[spec_name]

        # Check if exposure var is in params
        if EXPOSURE_VAR not in res.params.index:
            diag[spec_name] = {'status': 'NO_EXPOSURE_COEF'}
            continue

        coef = res.params[EXPOSURE_VAR]
        se = res.bse[EXPOSURE_VAR]
        pval = res.pvalues[EXPOSURE_VAR]
        ci_lower, ci_upper = res.conf_int().loc[EXPOSURE_VAR]

        # Assess validity
        n_obs = res.nobs
        df_resid = res.df_resid
        df_ratio = df_resid / n_obs

        if df_ratio > 0.3:
            power = 'GOOD'
        elif df_ratio > 0.1:
            power = 'MODERATE'
        else:
            power = 'TIGHT'

        # Determine confidence rating
        if pval < 0.05:
            sig = '**'
            sig_rate = 'YES'
        elif pval < 0.10:
            sig = '*'
            sig_rate = 'MARGINAL'
        else:
            sig = ''
            sig_rate = 'NO'

        if power == 'GOOD' and sig_rate == 'YES':
            rating = 'HIGH'
        elif power in ['MODERATE', 'GOOD'] or sig_rate in ['YES', 'MARGINAL']:
            rating = 'MEDIUM'
        else:
            rating = 'LOW'

        diag[spec_name] = {
            'status': 'OK',
            'coef': coef,
            'se': se,
            'pval': pval,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'sig': sig,
            'n_obs': n_obs,
            'df_resid': df_resid,
            'df_ratio': df_ratio,
            'power': power,
            'significance': sig_rate,
            'rsquared': res.rsquared,
            'validity_rating': rating,
        }

    return diag


def score_specification(spec_name: str, diagnostics: dict) -> dict:
    """Generate validity score and recommendation for a spec."""

    if spec_name not in diagnostics or diagnostics[spec_name]['status'] != 'OK':
        return {
            'spec': spec_name,
            'rating': 'UNKNOWN',
            'reason': 'Model failed or no coefficient',
            'recommendation': 'N/A',
        }

    spec_diag = diagnostics[spec_name]

    rating = spec_diag['validity_rating']

    # Generate recommendation
    if spec_name == 'spec1':
        rec = "Conservative baseline. Good for robustness, weak causal identification."
    elif spec_name == 'spec2':
        rec = "**RECOMMENDED MAIN SPEC.** Best causal identification (firm-year + occ-year FE control for time-varying shocks)."
    else:  # spec3
        rec = "Robustness check only. Very tight DoF, check convergence."

    return {
        'spec': spec_name,
        'rating': rating,
        'power': spec_diag['power'],
        'significance': spec_diag['significance'],
        'reason': f"{spec_diag['power']} power, {spec_diag['significance']} significance",
        'recommendation': rec,
    }


# ============================================================================
# INCOME REVERSAL ANALYSIS
# ============================================================================

def analyze_income_reversal(df_full: pd.DataFrame) -> dict:
    """
    Diagnose sign flip in income coefficient.

    Simple sequence:
    1. No controls
    2. +Controls
    3. +Additive FE
    4. +Firm-Year + Occ-Year FE
    """
    logger.info("\n" + "="*80)
    logger.info("INCOME REVERSAL ANALYSIS")
    logger.info("="*80)

    df_inc = df_full[['idpers', 'firm_id', 'isco08_4d', 'year', EXPOSURE_VAR, 'outcome_log_income'] + CONTROLS].dropna()
    logger.info(f"Income sample: N={len(df_inc)}")

    # Create FE indicators
    df_inc['firm_year'] = df_inc['firm_id'].astype(str) + '_' + df_inc['year'].astype(str)
    df_inc['occ_year'] = df_inc['isco08_4d'].astype(str) + '_' + df_inc['year'].astype(str)
    df_inc['firm'] = df_inc['firm_id'].astype(str)
    df_inc['occupation'] = df_inc['isco08_4d'].astype(str)

    results = {}
    controls_str = ' + '.join(CONTROLS)

    # Model 1: Unadjusted
    m1 = smf.ols(f"outcome_log_income ~ {EXPOSURE_VAR}", data=df_inc).fit()
    results['m1_unadjusted'] = {
        'coef': m1.params[EXPOSURE_VAR],
        'se': m1.bse[EXPOSURE_VAR],
        'pval': m1.pvalues[EXPOSURE_VAR],
    }
    logger.info(f"M1 (unadjusted): coef={results['m1_unadjusted']['coef']:.6f}")

    # Model 2: Controls only
    m2 = smf.ols(f"outcome_log_income ~ {EXPOSURE_VAR} + {controls_str}", data=df_inc).fit()
    results['m2_controls'] = {
        'coef': m2.params[EXPOSURE_VAR],
        'se': m2.bse[EXPOSURE_VAR],
        'pval': m2.pvalues[EXPOSURE_VAR],
    }
    logger.info(f"M2 (controls): coef={results['m2_controls']['coef']:.6f}")

    # Model 3: Additive FE
    try:
        m3 = smf.ols(f"outcome_log_income ~ C(idpers) + C(firm) + C(year) + C(occupation) + {EXPOSURE_VAR} + {controls_str}",
                      data=df_inc).fit(cov_type='cluster', cov_kwds={'groups': df_inc['idpers']})
        results['m3_additive'] = {
            'coef': m3.params[EXPOSURE_VAR],
            'se': m3.bse[EXPOSURE_VAR],
            'pval': m3.pvalues[EXPOSURE_VAR],
        }
        logger.info(f"M3 (additive FE): coef={results['m3_additive']['coef']:.6f}")
    except Exception as e:
        logger.error(f"M3 failed: {e}")

    # Model 4: Firm-Year + Occ-Year FE
    try:
        m4 = smf.ols(f"outcome_log_income ~ C(idpers) + C(firm_year) + C(occ_year) + {EXPOSURE_VAR} + {controls_str}",
                      data=df_inc).fit(cov_type='cluster', cov_kwds={'groups': df_inc['idpers']})
        results['m4_firm_occ_year'] = {
            'coef': m4.params[EXPOSURE_VAR],
            'se': m4.bse[EXPOSURE_VAR],
            'pval': m4.pvalues[EXPOSURE_VAR],
        }
        logger.info(f"M4 (firm-year + occ-year FE): coef={results['m4_firm_occ_year']['coef']:.6f}")
    except Exception as e:
        logger.error(f"M4 failed: {e}")

    # Check for sign flip
    coefs = [v['coef'] for v in results.values()]
    signs = [np.sign(c) for c in coefs]
    if len(set(signs)) > 1:
        logger.warning("*** SIGN FLIP DETECTED ***")
        results['sign_flip'] = True
    else:
        results['sign_flip'] = False

    return results


# ============================================================================
# EXPORT RESULTS
# ============================================================================

def export_results(all_results: dict, all_diagnostics: dict):
    """Export all results to CSV/JSON."""

    # Regression table
    table_rows = []
    for outcome_var, results in all_results.items():
        if results is None:
            continue

        for spec_name in ['spec1', 'spec2', 'spec3']:
            if spec_name not in results:
                continue

            res = results[spec_name]
            if EXPOSURE_VAR not in res.params.index:
                continue

            coef = res.params[EXPOSURE_VAR]
            se = res.bse[EXPOSURE_VAR]
            pval = res.pvalues[EXPOSURE_VAR]
            ci_lower, ci_upper = res.conf_int().loc[EXPOSURE_VAR]

            sig = '***' if pval < 0.01 else ('**' if pval < 0.05 else ('*' if pval < 0.10 else ''))

            table_rows.append({
                'Outcome': outcome_var,
                'Specification': spec_name,
                'Coefficient': f"{coef:.6f}{sig}",
                'Std.Error': f"{se:.6f}",
                'P-value': f"{pval:.4f}",
                'CI_Lower': f"{ci_lower:.6f}",
                'CI_Upper': f"{ci_upper:.6f}",
                'N': res.nobs,
                'DoF_Resid': res.df_resid,
                'R²': f"{res.rsquared:.4f}",
            })

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUTPUT_DIR / 'regression_tables.csv', index=False)
    logger.info(f"Saved: regression_tables.csv")

    # Validity scorecard
    scorecard_rows = []
    for outcome_var, diag in all_diagnostics.items():
        for spec_name in ['spec1', 'spec2', 'spec3']:
            score = score_specification(spec_name, diag)
            scorecard_rows.append({
                'Outcome': outcome_var,
                'Specification': spec_name,
                'Validity_Rating': score['rating'],
                'Power': score.get('power', ''),
                'Significance': score.get('significance', ''),
                'Recommendation': score['recommendation'],
            })

    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv(OUTPUT_DIR / 'validity_scorecard.csv', index=False)
    logger.info(f"Saved: validity_scorecard.csv")

    # Diagnostics detail
    diag_json = {}
    for outcome_var, diag in all_diagnostics.items():
        diag_json[outcome_var] = {k: str(v) for k, v in diag.items()}

    with open(OUTPUT_DIR / 'detailed_diagnostics.json', 'w') as f:
        json.dump(diag_json, f, indent=2, default=str)
    logger.info(f"Saved: detailed_diagnostics.json")


# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("SHP SPECIFICATION DIAGNOSTICS (STREAMLINED)")
    logger.info("="*80)

    all_outcomes = OUTCOME_VARS['economic'] + OUTCOME_VARS['political']
    all_results = {}
    all_diagnostics = {}

    # Fit models for each outcome
    for outcome_var in all_outcomes:
        logger.info(f"\nProcessing {outcome_var}...")
        results = fit_specifications(outcome_var)

        if results is None:
            logger.warning(f"Skipped {outcome_var}")
            continue

        all_results[outcome_var] = results
        all_diagnostics[outcome_var] = diagnose_results(results, outcome_var)

    # Income reversal analysis
    income_analysis = analyze_income_reversal(df)

    # Export results
    export_results(all_results, all_diagnostics)

    # Save income analysis
    with open(OUTPUT_DIR / 'income_reversal.json', 'w') as f:
        json.dump({k: str(v) for k, v in income_analysis.items()}, f, indent=2, default=str)
    logger.info(f"Saved: income_reversal.json")

    # Print summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY: RECOMMENDED SPECIFICATIONS")
    logger.info("="*80)

    for outcome_var, diag in all_diagnostics.items():
        logger.info(f"\n{outcome_var}:")
        for spec_name in ['spec1', 'spec2', 'spec3']:
            score = score_specification(spec_name, diag)
            logger.info(f"  {spec_name}: {score['rating']} - {score['recommendation']}")

    logger.info(f"\n{'='*80}")
    logger.info(f"All results saved to: {OUTPUT_DIR}")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()

