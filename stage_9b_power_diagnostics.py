#!/usr/bin/env python3
"""
Stage 9b Power Diagnostics: Post-hoc minimum detectable effect (MDE) analysis.

For each coefficient in stage9b_results_long.csv, compute the smallest effect
size that would have been detected at α=0.05 with 80% power, and translate
it into interpretable units:
  - raw units
  - per-1-SD-of-treatment effect on outcome
  - per-1-SD-of-treatment effect in outcome-SD units (fully standardized)

MDE = SE × (z_{0.975} + z_{0.80}) = SE × (1.96 + 0.84) ≈ 2.8 × SE
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent
data_dir = project_root / "Data"
panel_file = data_dir / "shp_panel_first_diff.csv"
results_file = data_dir / "stage9b_results_long.csv"
output_file = data_dir / "stage9b_power_diagnostics.csv"

MDE_MULTIPLIER = 1.96 + 0.84  # two-sided α=0.05, power=0.80


def construct_lags(df):
    df = df.sort_values(['idpers', 'year']).reset_index(drop=True).copy()
    year_prev = df.groupby('idpers')['year'].shift(1)
    prev_consecutive = (df['year'] - year_prev == 1)
    for col in ['d_exp_bar_ot', 'd_exp_dev']:
        df[f'{col}_lag1'] = df.groupby('idpers')[col].shift(1).where(prev_consecutive)
    return df


def compute_sds(df):
    """Pool SDs across the full panel (small sample-specific differences ignored)."""
    cols = ['d_exp_bar_ot', 'd_exp_dev', 'd_exp_bar_ot_lag1', 'd_exp_dev_lag1',
            'd_outcome_job_insecurity', 'd_outcome_log_income', 'd_outcome_hours_worked']
    return {c: df[c].std() for c in cols}


def main():
    df = pd.read_csv(panel_file, low_memory=False)
    df = construct_lags(df)
    sds = compute_sds(df)

    res = pd.read_csv(results_file)
    res['mde'] = res['std_error'] * MDE_MULTIPLIER
    res['treatment_sd'] = res['term'].map(sds)
    res['outcome_sd'] = res['outcome'].map(sds)

    # Effect of 1-SD change in treatment, in raw outcome units
    res['est_per_sd_treat'] = res['estimate'] * res['treatment_sd']
    res['mde_per_sd_treat'] = res['mde'] * res['treatment_sd']

    # Same, in outcome-SD units (fully standardized)
    res['est_per_sd_treat_in_outcome_sd'] = res['est_per_sd_treat'] / res['outcome_sd']
    res['mde_per_sd_treat_in_outcome_sd'] = res['mde_per_sd_treat'] / res['outcome_sd']

    res.to_csv(output_file, index=False)

    # Pretty-print
    print("=" * 100)
    print(f"POWER DIAGNOSTICS (α=0.05, power=0.80, MDE = SE × {MDE_MULTIPLIER:.2f})")
    print("=" * 100)
    print(f"{'outcome':<28}{'spec':<13}{'term':<22}"
          f"{'est':>8}{'SE':>8}{'MDE':>8}  "
          f"{'est/SD_t':>9}{'MDE/SD_t':>9}  "
          f"{'est/σY':>8}{'MDE/σY':>8}")
    print("-" * 100)
    for _, r in res.iterrows():
        print(f"{r['outcome']:<28}{r['spec']:<13}{r['term']:<22}"
              f"{r['estimate']:>+8.4f}{r['std_error']:>8.4f}{r['mde']:>8.4f}  "
              f"{r['est_per_sd_treat']:>+9.4f}{r['mde_per_sd_treat']:>9.4f}  "
              f"{r['est_per_sd_treat_in_outcome_sd']:>+8.4f}{r['mde_per_sd_treat_in_outcome_sd']:>8.4f}")

    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE")
    print("=" * 100)
    print(f"• MDE = smallest effect detectable at α=0.05 / 80% power for this SE")
    print(f"• est/SD_t, MDE/SD_t = effect/MDE of a 1-SD change in the treatment, in raw outcome units")
    print(f"• est/σY, MDE/σY = same as above but in outcome-SD units (fully standardized)")
    print()
    print("Treatment SDs (Δ exposure):")
    for c in ['d_exp_bar_ot', 'd_exp_dev', 'd_exp_bar_ot_lag1', 'd_exp_dev_lag1']:
        print(f"  {c}: σ = {sds[c]:.5f}")
    print("Outcome SDs (Δ outcomes):")
    for c in ['d_outcome_job_insecurity', 'd_outcome_log_income', 'd_outcome_hours_worked']:
        print(f"  {c}: σ = {sds[c]:.5f}")
    print(f"\n✓ Saved: {output_file}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
