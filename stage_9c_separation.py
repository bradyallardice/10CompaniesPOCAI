#!/usr/bin/env python3
"""
Stage 9c: Separation Analysis (Forward-Looking Binary Outcome)

Specification (per docs spec §6):
  Pr(Sep_{i,t+1} = 1) = alpha_t + beta1 * d_exp_bar_{o,t} + beta2 * d_exp_dev_{i,f,o,t} + e_{i,t}

  Note: outcome is a level (not first-differenced); forward-looking by construction
  (separation_t1 = 1 if firm_id changes between t and t+1, defined in stage_8a).

  Models:
    - LPM (linearmodels OLS, two-way clusters on idpers × firm_id)  ← primary
    - Logit (statsmodels, one-way cluster on idpers)                 ← robustness

  Three lag specs per model, same as stage_9b:
    - contemp:     [d_exp_bar_ot,      d_exp_dev]
    - lag1:        [d_exp_bar_ot_lag1, d_exp_dev_lag1]
    - distributed: contemp + lag1

Inputs:
  Data/shp_panel_first_diff.csv

Outputs:
  Data/stage9c_results_long.csv
  Data/stage9c_estimation_log.txt
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
import warnings

project_root = Path(__file__).parent
data_dir = project_root / "Data"
panel_file = data_dir / "shp_panel_first_diff.csv"
output_results = data_dir / "stage9c_results_long.csv"
log_file = data_dir / "stage9c_estimation_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

OUTCOME = 'separation_t1'
SPECS = {
    'contemp':     ['d_exp_bar_ot',      'd_exp_dev'],
    'lag1':        ['d_exp_bar_ot_lag1', 'd_exp_dev_lag1'],
    'distributed': ['d_exp_bar_ot',      'd_exp_dev',
                    'd_exp_bar_ot_lag1', 'd_exp_dev_lag1'],
}


def construct_lags(df):
    df = df.sort_values(['idpers', 'year']).reset_index(drop=True).copy()
    year_prev = df.groupby('idpers')['year'].shift(1)
    prev_consecutive = (df['year'] - year_prev == 1)
    for col in ['d_exp_bar_ot', 'd_exp_dev']:
        df[f'{col}_lag1'] = df.groupby('idpers')[col].shift(1).where(prev_consecutive)
    return df


def estimation_sample(df, rhs_cols):
    needed = [OUTCOME, 'year', 'idpers', 'firm_id'] + rhs_cols
    sub = df[needed].dropna().copy()
    sub['firm_id'] = sub['firm_id'].astype('int64')
    sub['idpers'] = sub['idpers'].astype('int64')
    sub['year'] = sub['year'].astype('int64')
    return sub


def fit_lpm(sub, rhs_cols, spec_name):
    rhs_str = ' + '.join(rhs_cols)
    formula = f"{OUTCOME} ~ 1 + C(year) + {rhs_str}"
    try:
        mod = IV2SLS.from_formula(formula, sub)
        res = mod.fit(cov_type='clustered', clusters=sub[['idpers', 'firm_id']])
    except Exception as e:
        logger.error(f"  LPM FIT FAILED [{spec_name}]: {e}")
        return None
    return res


def fit_logit(sub, rhs_cols, spec_name):
    """Logit with one-way cluster on idpers (statsmodels limitation)."""
    rhs_str = ' + '.join(rhs_cols)
    formula = f"{OUTCOME} ~ 1 + C(year) + {rhs_str}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = sm.formula.logit(formula=formula, data=sub)
            res = mod.fit(cov_type='cluster', cov_kwds={'groups': sub['idpers']}, disp=0, maxiter=200)
    except Exception as e:
        logger.error(f"  LOGIT FIT FAILED [{spec_name}]: {e}")
        return None
    return res


def extract_rows(res, model_name, spec_name, rhs_cols, sub):
    if res is None:
        return []
    n_obs = int(res.nobs)
    n_persons = sub['idpers'].nunique()
    n_firms = sub['firm_id'].nunique()
    n_events = int(sub[OUTCOME].sum())
    rows = []
    for term in rhs_cols:
        if term not in res.params.index:
            continue
        if model_name == 'lpm':
            est = res.params[term]
            se = res.std_errors[term]
            t = res.tstats[term]
            p = res.pvalues[term]
            ci_l = res.conf_int().loc[term, 'lower']
            ci_u = res.conf_int().loc[term, 'upper']
        else:  # logit (statsmodels)
            est = res.params[term]
            se = res.bse[term]
            t = res.tvalues[term]
            p = res.pvalues[term]
            ci = res.conf_int().loc[term].values
            ci_l, ci_u = ci[0], ci[1]
        rows.append({
            'model': model_name,
            'spec': spec_name,
            'term': term,
            'estimate': est,
            'std_error': se,
            't_stat': t,
            'p_value': p,
            'ci_lower': ci_l,
            'ci_upper': ci_u,
            'n_obs': n_obs,
            'n_persons': n_persons,
            'n_firms': n_firms,
            'n_events': n_events,
        })
    return rows


def main():
    logger.info("=" * 70)
    logger.info("STAGE 9C: SEPARATION ANALYSIS")
    logger.info("=" * 70)
    try:
        if not panel_file.exists():
            raise FileNotFoundError(f"Panel input not found: {panel_file}")

        df = pd.read_csv(panel_file, low_memory=False)
        logger.info(f"Loaded panel: {len(df):,} rows, {df['idpers'].nunique():,} persons")
        df = construct_lags(df)

        all_rows = []
        for spec_name, rhs_cols in SPECS.items():
            sub = estimation_sample(df, rhs_cols)
            if len(sub) == 0:
                logger.warning(f"Empty sample for spec {spec_name}; skipping")
                continue
            n_events = int(sub[OUTCOME].sum())
            n_obs = len(sub)
            base_rate = n_events / n_obs if n_obs > 0 else 0
            logger.info(f"\n--- Spec: {spec_name} ---")
            logger.info(f"  n_obs={n_obs:,} persons={sub['idpers'].nunique():,} "
                        f"firms={sub['firm_id'].nunique():,} events={n_events} "
                        f"base_rate={base_rate*100:.2f}%")

            res_lpm = fit_lpm(sub, rhs_cols, spec_name)
            all_rows.extend(extract_rows(res_lpm, 'lpm', spec_name, rhs_cols, sub))

            res_logit = fit_logit(sub, rhs_cols, spec_name)
            all_rows.extend(extract_rows(res_logit, 'logit', spec_name, rhs_cols, sub))

        if not all_rows:
            raise RuntimeError("No regressions produced output; check logs.")

        results = pd.DataFrame(all_rows)
        results.to_csv(output_results, index=False)

        logger.info("\n" + "=" * 70)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 70)
        for model in ['lpm', 'logit']:
            label = ('LPM (two-way cluster idpers × firm_id)' if model == 'lpm'
                    else 'LOGIT (one-way cluster idpers)')
            logger.info(f"\n{label}")
            sub_res = results[results['model'] == model]
            for _, r in sub_res.iterrows():
                sig = ('***' if r['p_value'] < 0.01 else
                       '**'  if r['p_value'] < 0.05 else
                       '*'   if r['p_value'] < 0.10 else '')
                logger.info(f"  [{r['spec']:>11}] {r['term']:>22}: "
                            f"{r['estimate']:>+.5f} ({r['std_error']:.5f}) {sig:>3}  "
                            f"n={r['n_obs']:,} events={r['n_events']}")

        logger.info(f"\n✓ Saved: {output_results} ({len(results)} rows)")
        logger.info("=" * 70)
        return 0

    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
