#!/usr/bin/env python3
"""
Stage 9b: First-Differenced Panel Estimation

Specification (per docs spec §4–§5):
  d_Y_{i,t} = alpha_t + beta1 * d_exp_bar_{o,t} + beta2 * d_exp_dev_{i,f,o,t} + e_{i,t}

  - alpha_t: year fixed effects (C(year))
  - SEs: two-way clustered on (idpers, firm_id)
  - No worker covariates (d_age = 1 for consecutive pairs → absorbed)

Three lag specifications per outcome:
  - contemp:     RHS = [d_exp_bar_ot,      d_exp_dev]
  - lag1:        RHS = [d_exp_bar_ot_lag1, d_exp_dev_lag1]
  - distributed: RHS = [contemp + lag1]    (joint test of dynamic effects)

Three continuous outcomes:
  - d_outcome_job_insecurity
  - d_outcome_log_income
  - d_outcome_hours_worked

Inputs:
  Data/shp_panel_first_diff.csv (from stage_9a)

Outputs:
  Data/stage9b_results_long.csv (long-format coef table)
  Data/stage9b_estimation_log.txt
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from linearmodels.iv import IV2SLS

project_root = Path(__file__).parent
data_dir = project_root / "Data"
panel_file = data_dir / "shp_panel_first_diff.csv"
output_results = data_dir / "stage9b_results_long.csv"
log_file = data_dir / "stage9b_estimation_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

OUTCOMES = ['d_outcome_job_insecurity', 'd_outcome_log_income', 'd_outcome_hours_worked']

SPECS = {
    'contemp':     ['d_exp_bar_ot',      'd_exp_dev'],
    'lag1':        ['d_exp_bar_ot_lag1', 'd_exp_dev_lag1'],
    'distributed': ['d_exp_bar_ot',      'd_exp_dev',
                    'd_exp_bar_ot_lag1', 'd_exp_dev_lag1'],
}


def construct_lags(df):
    """Within-person lag1 of treatment Δs, only when prior obs is a consecutive year."""
    df = df.sort_values(['idpers', 'year']).reset_index(drop=True).copy()
    year_prev = df.groupby('idpers')['year'].shift(1)
    prev_consecutive = (df['year'] - year_prev == 1)

    for col in ['d_exp_bar_ot', 'd_exp_dev']:
        lagged = df.groupby('idpers')[col].shift(1)
        df[f'{col}_lag1'] = lagged.where(prev_consecutive)

    n_lag1 = df['d_exp_bar_ot_lag1'].notna().sum()
    logger.info(f"Lag1 treatment defined for {n_lag1:,} person-years")
    return df


def fit_one(data, outcome_col, rhs_cols, spec_name):
    """Fit OLS with year FE and two-way clustered SEs on (idpers, firm_id)."""
    needed = [outcome_col, 'year', 'idpers', 'firm_id'] + rhs_cols
    sub = data[needed].dropna().copy()
    if len(sub) == 0:
        logger.warning(f"  EMPTY SAMPLE for {outcome_col} / {spec_name}; skipping")
        return None

    sub['firm_id'] = sub['firm_id'].astype('int64')
    sub['idpers'] = sub['idpers'].astype('int64')
    sub['year'] = sub['year'].astype('int64')

    rhs_str = ' + '.join(rhs_cols)
    formula = f"{outcome_col} ~ 1 + C(year) + {rhs_str}"

    try:
        mod = IV2SLS.from_formula(formula, sub)
        res = mod.fit(cov_type='clustered', clusters=sub[['idpers', 'firm_id']])
    except Exception as e:
        logger.error(f"  FIT FAILED for {outcome_col} / {spec_name}: {e}")
        return None

    n_obs = int(res.nobs)
    n_persons = sub['idpers'].nunique()
    n_firms = sub['firm_id'].nunique()
    n_years = sub['year'].nunique()

    logger.info(f"  {outcome_col} | {spec_name}: n={n_obs:,} persons={n_persons:,} "
                f"firms={n_firms:,} years={n_years}")

    rows = []
    for term in rhs_cols:
        if term not in res.params.index:
            logger.warning(f"    term {term} not in result params; skipping")
            continue
        rows.append({
            'outcome': outcome_col,
            'spec': spec_name,
            'term': term,
            'estimate': res.params[term],
            'std_error': res.std_errors[term],
            't_stat': res.tstats[term],
            'p_value': res.pvalues[term],
            'ci_lower': res.conf_int().loc[term, 'lower'],
            'ci_upper': res.conf_int().loc[term, 'upper'],
            'n_obs': n_obs,
            'n_persons': n_persons,
            'n_firms': n_firms,
            'n_years': n_years,
        })
    return rows


def main():
    logger.info("=" * 70)
    logger.info("STAGE 9B: FIRST-DIFFERENCED PANEL ESTIMATION")
    logger.info("=" * 70)
    try:
        if not panel_file.exists():
            raise FileNotFoundError(f"Panel input not found: {panel_file}")

        df = pd.read_csv(panel_file, low_memory=False)
        logger.info(f"Loaded panel: {len(df):,} rows, {df['idpers'].nunique():,} persons")

        df = construct_lags(df)

        all_rows = []
        for outcome in OUTCOMES:
            logger.info(f"\n--- Outcome: {outcome} ---")
            for spec_name, rhs_cols in SPECS.items():
                rows = fit_one(df, outcome, rhs_cols, spec_name)
                if rows:
                    all_rows.extend(rows)

        if not all_rows:
            raise RuntimeError("No regressions produced output; check logs.")

        results = pd.DataFrame(all_rows)
        results.to_csv(output_results, index=False)

        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        for outcome in OUTCOMES:
            logger.info(f"\n{outcome}")
            sub = results[results['outcome'] == outcome]
            for _, r in sub.iterrows():
                sig = '***' if r['p_value'] < 0.01 else ('**' if r['p_value'] < 0.05 else
                                                         ('*' if r['p_value'] < 0.10 else ''))
                logger.info(f"  [{r['spec']:>11}] {r['term']:>22}: "
                            f"{r['estimate']:>+.5f} ({r['std_error']:.5f}) {sig:>3}  "
                            f"n={r['n_obs']:,}")

        logger.info(f"\n✓ Saved: {output_results} ({len(results)} rows)")
        logger.info("=" * 70)
        return 0

    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
