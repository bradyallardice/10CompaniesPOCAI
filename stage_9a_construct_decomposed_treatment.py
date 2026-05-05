#!/usr/bin/env python3
"""
Stage 9a: Construct Decomposed AI Exposure Treatment for First-Differenced Panel

Purpose:
  Merge Stage 5 firm-occupation-year AI exposure onto the SHP analysis panel,
  decompose into occupation-year (national) and firm-deviation components, and
  construct within-worker first-differences for both treatment and outcomes.

Decomposition:
  Exp_{i,f,o,t} = Exp_bar_{o,t} + Exp_dev_{i,f,o,t}
  where Exp_bar_{o,t}  = mean of hampole_ai_exposure_avg across all firms
                         in the exposure database for (isco08_4d, year)
        Exp_dev        = hampole_ai_exposure_avg − Exp_bar_{o,t}

Note: the in-sample mean is the chosen benchmark because Stage 5's
hampole_occupation_exposure is firm-specific (a task is "exposed" at
firm f only if f's AI apps match it), so no truly pure occupation-year
measure exists in the pipeline. See conversation log for justification.

First differences (within-person, only when consecutive years):
  d_exp_bar_ot, d_exp_dev, d_outcome_*

Inputs:
  Data/shp_panel_prepared.csv (from stage_8a)
  Data/Testing/stage_5/full_sample/skip_ce/isco_occupation_year_exposure_core_tasks_pct_05_ce_0.0.csv

Output:
  Data/shp_panel_first_diff.csv
  Data/shp_panel_first_diff_log.txt
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
data_dir = project_root / "Data"
panel_file = data_dir / "shp_panel_prepared.csv"
exposure_file = (
    data_dir / "Testing" / "stage_5" / "full_sample" / "skip_ce"
    / "isco_occupation_year_exposure_core_tasks_pct_05_ce_0.0.csv"
)
output_file = data_dir / "shp_panel_first_diff.csv"
log_file = data_dir / "shp_panel_first_diff_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

OUTCOMES = ['outcome_job_insecurity', 'outcome_log_income', 'outcome_hours_worked']
TREATMENT = ['exp_bar_ot', 'exp_dev']


def validate_inputs():
    for f in [panel_file, exposure_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required input not found: {f}")
    logger.info(f"Inputs validated.")


def load_panel():
    cols = [
        'idpers', 'year', 'firm_id', 'company_id', 'isco08_4d',
        'age', 'female',
    ] + OUTCOMES + ['separation_t1']
    df = pd.read_csv(panel_file, usecols=lambda c: c in cols, low_memory=False)
    df['firm_id'] = df['firm_id'].astype('Int64')
    df['isco08_4d'] = pd.to_numeric(df['isco08_4d'], errors='coerce').astype('Int64')
    df['year'] = df['year'].astype('int32')
    logger.info(f"Loaded panel: {len(df):,} rows, {df['idpers'].nunique():,} persons, "
                f"years {df['year'].min()}–{df['year'].max()}")
    return df


def load_exposure():
    cols = ['company_id', 'isco08_4d', 'year', 'hampole_ai_exposure_avg']
    df = pd.read_csv(exposure_file, usecols=cols)
    logger.info(f"Loaded exposure: {len(df):,} firm-occ-year rows")

    df['firm_id'] = ((df['company_id'].astype('int64') + 13) * 13).astype('Int64')
    df['isco08_4d'] = pd.to_numeric(df['isco08_4d'], errors='coerce').astype('Int64')
    df['year'] = df['year'].astype('int32')

    # Validate uniqueness of merge keys
    n_dup = df.duplicated(['firm_id', 'isco08_4d', 'year']).sum()
    if n_dup > 0:
        raise ValueError(f"Exposure has {n_dup:,} duplicate (firm_id, isco08_4d, year) rows")
    logger.info(f"  ✓ Unique on (firm_id, isco08_4d, year)")

    # Compute occupation-year mean (Exp_bar_{o,t}) across all firms in the database
    occ_year_mean = (
        df.groupby(['isco08_4d', 'year'])['hampole_ai_exposure_avg']
        .agg(['mean', 'std', 'size'])
        .reset_index()
        .rename(columns={'mean': 'exp_bar_ot', 'std': 'exp_bar_ot_sd_within_cell', 'size': 'n_firms_oy'})
    )
    logger.info(f"  ✓ Computed Exp_bar_{{o,t}}: {len(occ_year_mean):,} (isco, year) cells")

    return df, occ_year_mean


def merge_and_decompose(panel, exposure, occ_year_mean):
    merge_cols = ['firm_id', 'isco08_4d', 'year']
    keep_from_exp = merge_cols + ['hampole_ai_exposure_avg']

    merged = panel.merge(
        exposure[keep_from_exp],
        on=merge_cols, how='left', indicator=True, validate='m:1'
    )

    n_total = len(merged)
    n_match = (merged['_merge'] == 'both').sum()
    n_panel_with_keys = panel[merge_cols].notna().all(axis=1).sum()
    logger.info(f"Firm-level exposure merge:")
    logger.info(f"  - Panel rows: {n_total:,}")
    logger.info(f"  - Panel rows with all merge keys present: {n_panel_with_keys:,}")
    logger.info(f"  - Matched (firm in DB for that occ-year): {n_match:,} "
                f"({100*n_match/n_total:.1f}% of panel, "
                f"{100*n_match/n_panel_with_keys:.1f}% of rows with keys)")
    merged = merged.drop(columns=['_merge'])

    # Merge occ-year mean (defined whenever (isco, year) has any firm in the exposure DB)
    merged = merged.merge(occ_year_mean, on=['isco08_4d', 'year'], how='left')
    n_with_oy_mean = merged['exp_bar_ot'].notna().sum()
    logger.info(f"Occupation-year mean merge:")
    logger.info(f"  - Panel rows with exp_bar_ot defined: {n_with_oy_mean:,} "
                f"({100*n_with_oy_mean/n_total:.1f}% of panel)")

    # Decomposition: deviation only defined when firm-level exposure is present
    merged['exp_dev'] = merged['hampole_ai_exposure_avg'] - merged['exp_bar_ot']

    return merged


def first_differences(df):
    df = df.sort_values(['idpers', 'year']).reset_index(drop=True).copy()

    year_prev = df.groupby('idpers')['year'].shift(1)
    consecutive = (df['year'] - year_prev == 1)

    cols_to_diff = OUTCOMES + TREATMENT
    for col in cols_to_diff:
        prev = df.groupby('idpers')[col].shift(1)
        diff = df[col] - prev
        df[f'd_{col}'] = diff.where(consecutive)

    df['consecutive_with_prev'] = consecutive.fillna(False)
    return df


def run_diagnostics(df):
    logger.info("=" * 70)
    logger.info("DIAGNOSTICS")
    logger.info("=" * 70)

    # |F_(o,t)| distribution among matched rows
    foy = df.dropna(subset=['hampole_ai_exposure_avg'])['n_firms_oy']
    if len(foy) > 0:
        logger.info(f"\n|F_(o,t)| (firms per occupation-year cell, among matched panel rows):")
        for q in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            logger.info(f"  - p{int(q*100):>3}: {foy.quantile(q):.0f}")
        logger.info(f"  - mean: {foy.mean():.1f}")

    # Treatment distributions (levels)
    logger.info(f"\nTreatment levels (matched rows only):")
    for col in TREATMENT:
        v = df[col].dropna()
        if len(v) > 0:
            logger.info(f"  {col:>15}: n={len(v):>6,} mean={v.mean():>9.5f} "
                        f"sd={v.std():>9.5f} min={v.min():>9.5f} max={v.max():>9.5f}")

    # Treatment first differences
    logger.info(f"\nTreatment first-differences:")
    for col in [f'd_{c}' for c in TREATMENT]:
        v = df[col].dropna()
        if len(v) > 0:
            pct_zero = (v == 0).mean() * 100
            logger.info(f"  {col:>17}: n={len(v):>6,} mean={v.mean():>10.6f} "
                        f"sd={v.std():>9.6f} %zero={pct_zero:>5.1f}")

    # Outcome first differences
    logger.info(f"\nOutcome first-differences:")
    for col in [f'd_{o}' for o in OUTCOMES]:
        v = df[col].dropna()
        if len(v) > 0:
            logger.info(f"  {col:>27}: n={len(v):>6,} mean={v.mean():>9.5f} sd={v.std():>9.5f}")

    # Estimation sample sizes
    logger.info(f"\nEstimation samples (rows with d_outcome AND d_exp_bar_ot AND d_exp_dev all non-missing):")
    for o in OUTCOMES:
        d_o = f'd_{o}'
        mask = df[[d_o, 'd_exp_bar_ot', 'd_exp_dev']].notna().all(axis=1)
        n_obs = mask.sum()
        n_pers = df.loc[mask, 'idpers'].nunique()
        logger.info(f"  {d_o:>27}: {n_obs:>6,} person-years, {n_pers:>5,} unique persons")

    # Separation sample
    sep_mask = df[['separation_t1', 'd_exp_bar_ot', 'd_exp_dev']].notna().all(axis=1)
    n_sep_obs = sep_mask.sum()
    n_sep_pers = df.loc[sep_mask, 'idpers'].nunique()
    n_sep_event = (df.loc[sep_mask, 'separation_t1'] == 1).sum()
    logger.info(f"\nSeparation analysis sample (separation_t1 forward-looking):")
    logger.info(f"  - {n_sep_obs:,} person-years, {n_sep_pers:,} persons, {n_sep_event:,} separation events")


def main():
    logger.info("=" * 70)
    logger.info("STAGE 9A: DECOMPOSED TREATMENT CONSTRUCTION")
    logger.info("=" * 70)
    try:
        validate_inputs()
        panel = load_panel()
        exposure, occ_year_mean = load_exposure()
        merged = merge_and_decompose(panel, exposure, occ_year_mean)
        final = first_differences(merged)
        run_diagnostics(final)

        final.to_csv(output_file, index=False)
        size_mb = output_file.stat().st_size / (1024 ** 2)
        logger.info(f"\n✓ Saved: {output_file} ({size_mb:.1f} MB, {len(final):,} rows, {final.shape[1]} cols)")
        logger.info("=" * 70)
        return 0
    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
