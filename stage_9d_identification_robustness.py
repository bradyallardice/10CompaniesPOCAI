#!/usr/bin/env python3
"""
Stage 9d: Identification Robustness Checks — Job Insecurity

Tests the robustness of the job insecurity result (Spec A: matched-only) to:
  1. Sample definition: matched-only (exposure > 0) vs full X28-linked (exposure >= 0)
  2. Firm size control (pw85, 9-category self-reported employer size)
  3. Firm hiring activity control (log total job postings per firm-year)
  4. Both controls simultaneously
  5. Placebo leads test: current vs future AI exposure

All specs use:
  - Outcome:    outcome_job_insecurity (pw86, 1–4 scale, levels)
  - Treatment:  hampole_ai_exposure_avg_foy (firm-occ-year, from Stage 6 SHP)
  - FE:         Person + 3-digit ISCO × Year (within-demeaning, memory-efficient)
  - SE:         One-way cluster on idpers
  - Controls:   age_centered, age_squared, female, education

Key design notes:
  - Zero-fill (exposure == 0) = firm linked via X28 company_id but no AI apps detected
    for that occ-year. All zero-fill firms ARE in the X28 database by construction.
  - Matched-only (exposure > 0) = firm has confirmed positive AI adoption in that occ-year.
  - Within-demeaning used in place of dummy variables to avoid memory issues.

Inputs:
  Data/shp_panel_prepared.csv
  Data/firm_ai_summary_report.csv
  Data/shp_exposure/shp_exposure_isco4d.csv  (for pw85 firm size)

Outputs:
  Data/stage9d_results_long.csv
  Data/stage9d_estimation_log.txt
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
data_dir     = project_root / "Data"
panel_file   = data_dir / "shp_panel_prepared.csv"
firm_file    = data_dir / "firm_ai_summary_report.csv"
shp_file     = data_dir / "shp_exposure" / "shp_exposure_isco4d.csv"
output_file  = data_dir / "stage9d_results_long.csv"
log_file     = data_dir / "stage9d_estimation_log.txt"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

EXPOSURE = 'hampole_ai_exposure_avg_foy'
OUTCOME  = 'outcome_job_insecurity'
CONTROLS = ['age_centered', 'age_squared', 'female', 'education']


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data():
    for f in [panel_file, firm_file, shp_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required input not found: {f}")

    logger.info("Loading panel data...")
    cols = ['idpers', 'firm_id', 'isco08_4d', 'year', EXPOSURE, OUTCOME] + CONTROLS
    panel = pd.read_csv(panel_file, low_memory=False,
                        usecols=lambda c: c in cols)
    panel['firm_id'] = panel['firm_id'].astype('Int64')
    panel['year']    = panel['year'].astype('int64')
    logger.info(f"  Panel: {len(panel):,} rows, {panel['idpers'].nunique():,} persons")

    logger.info("Loading firm report (hiring activity)...")
    firm = pd.read_csv(firm_file, usecols=['company_id', 'year', 'total_unique_job_ads'])
    firm['firm_id']      = ((firm['company_id'].astype('int64') + 13) * 13)
    firm['log_job_ads']  = np.log1p(firm['total_unique_job_ads'])
    firm['year']         = firm['year'].astype('int64')
    firm = firm[['firm_id', 'year', 'log_job_ads']]
    logger.info(f"  Firm report: {len(firm):,} firm-year rows")

    logger.info("Loading SHP firm size (pw85)...")
    shp = pd.read_csv(shp_file, usecols=['idpers', 'year', 'pw85'], low_memory=False)
    shp['firm_size'] = shp['pw85'].where(shp['pw85'] > 0)
    shp = shp[['idpers', 'year', 'firm_size']]
    logger.info(f"  pw85 valid responses: {shp['firm_size'].notna().sum():,}")

    return panel, firm, shp


def build_analysis_frame(panel, firm, shp):
    """Merge all controls and construct occ-year FE identifier."""
    df = panel.copy()
    df['firm_id_int'] = pd.to_numeric(df['firm_id'], errors='coerce').astype('Int64')
    df['isco3d']   = (pd.to_numeric(df['isco08_4d'], errors='coerce') // 10).astype(str)
    df['occ_year'] = df['isco3d'] + '_' + df['year'].astype(str)

    df = df.merge(shp,  on=['idpers', 'year'],    how='left')
    df = df.merge(firm, on=['firm_id_int', 'year'], how='left',
                  left_on=['firm_id_int', 'year'], right_on=['firm_id', 'year'])
    df = df.drop(columns=['firm_id_y'], errors='ignore')

    # Sample flags
    df['matched'] = df[EXPOSURE] > 0               # confirmed AI adoption
    df['x28_linked'] = df[EXPOSURE].notna()        # in X28 DB (exposure >= 0)

    logger.info("\nSample breakdown:")
    logger.info(f"  exposure > 0  (matched, positive AI):         {df['matched'].sum():>7,}")
    logger.info(f"  exposure == 0 (X28-linked, zero AI):          {((df[EXPOSURE]==0) & df['x28_linked']).sum():>7,}")
    logger.info(f"  exposure NaN  (no firm or occ match):         {df[EXPOSURE].isna().sum():>7,}")

    return df


# ── Estimation ────────────────────────────────────────────────────────────────

def within_demean(data, groups, value_cols):
    """Sequential within-group demeaning (Frisch-Waugh FE absorption)."""
    out = data.copy()
    for g in groups:
        for v in value_cols:
            out[v] = out[v] - out.groupby(g)[v].transform('mean')
    return out


def fit_ols(sub, rhs_cols, label):
    """OLS with person + 3d-occ-year FE via demeaning, one-way cluster on idpers."""
    needed = ['idpers', 'occ_year', OUTCOME] + rhs_cols
    s = sub[needed].dropna().copy()
    if len(s) < 50:
        logger.warning(f"  Sample too small for [{label}]: N={len(s)}")
        return None, s

    FE = ['idpers', 'occ_year']
    dm = within_demean(s, FE, [OUTCOME] + rhs_cols)
    X  = sm.add_constant(dm[rhs_cols])
    res = sm.OLS(dm[OUTCOME], X).fit(
        cov_type='cluster', cov_kwds={'groups': s['idpers']})
    return res, s


def extract_row(res, s, label, sample_label, rhs_cols):
    rows = []
    for term in rhs_cols:
        if term not in res.params.index:
            continue
        rows.append({
            'spec':       label,
            'sample':     sample_label,
            'term':       term,
            'estimate':   res.params[term],
            'std_error':  res.bse[term],
            't_stat':     res.tvalues[term],
            'p_value':    res.pvalues[term],
            'ci_lower':   res.conf_int().loc[term, 0],
            'ci_upper':   res.conf_int().loc[term, 1],
            'n_obs':      len(s),
            'n_persons':  s['idpers'].nunique(),
        })
    return rows


# ── Placebo leads test ────────────────────────────────────────────────────────

def build_leads(df):
    df = df.sort_values(['idpers', 'year']).reset_index(drop=True)
    year_next = df.groupby('idpers')['year'].shift(-1)
    consecutive = (year_next - df['year'] == 1)
    df['exposure_lead1'] = df.groupby('idpers')[EXPOSURE].shift(-1).where(consecutive)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("STAGE 9D: IDENTIFICATION ROBUSTNESS — JOB INSECURITY")
    logger.info("=" * 70)

    panel, firm, shp = load_data()
    df = build_analysis_frame(panel, firm, shp)
    df = build_leads(df)

    # Restrict to rows with non-missing outcome + base controls + exposure defined
    base_mask = df[[OUTCOME] + CONTROLS + [EXPOSURE]].notna().all(axis=1)
    full  = df[base_mask].copy()                  # full X28-linked (exposure >= 0)
    matched = full[full['matched']].copy()        # matched-only (exposure > 0)

    logger.info(f"\nEstimation frames:")
    logger.info(f"  Full X28-linked: N={len(full):,}")
    logger.info(f"  Matched-only:    N={len(matched):,}")

    all_rows = []

    # ── Main robustness table ─────────────────────────────────────────────────
    specs = [
        ('(1) Matched-only, baseline',            matched, [EXPOSURE] + CONTROLS),
        ('(2) Full sample, baseline',              full,    [EXPOSURE] + CONTROLS),
        ('(3) Full + firm size',                   full,    [EXPOSURE, 'firm_size'] + CONTROLS),
        ('(4) Full + log(job ads)',                full,    [EXPOSURE, 'log_job_ads'] + CONTROLS),
        ('(5) Full + firm size + log(job ads)',    full,    [EXPOSURE, 'firm_size', 'log_job_ads'] + CONTROLS),
    ]

    logger.info("\n" + "=" * 70)
    logger.info("MAIN ROBUSTNESS REGRESSIONS")
    logger.info("Person + 3d-ISCO×Year FE | one-way cluster on idpers")
    logger.info("=" * 70)

    for label, sub, rhs in specs:
        sample_label = 'matched' if 'Matched' in label else 'full'
        res, s = fit_ols(sub, rhs, label)
        if res is None:
            continue
        b  = res.params[EXPOSURE]
        se = res.bse[EXPOSURE]
        p  = res.pvalues[EXPOSURE]
        sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
        logger.info(f"  {label:<45}  N={len(s):>6,}  "
                    f"beta={b:>+8.4f}  SE={se:.4f}  p={p:.3f} {sig}")
        all_rows.extend(extract_row(res, s, label, sample_label, rhs))

    # ── Placebo leads test ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PLACEBO LEADS TEST (current vs future exposure)")
    logger.info("=" * 70)

    lead_specs = [
        ('Matched: current only',         matched, [EXPOSURE] + CONTROLS),
        ('Matched: horse race',            matched, [EXPOSURE, 'exposure_lead1'] + CONTROLS),
        ('Matched: lead only (placebo)',   matched, ['exposure_lead1'] + CONTROLS),
        ('Full: current only',            full,    [EXPOSURE] + CONTROLS),
        ('Full: horse race',              full,    [EXPOSURE, 'exposure_lead1'] + CONTROLS),
        ('Full: lead only (placebo)',     full,    ['exposure_lead1'] + CONTROLS),
    ]

    for label, sub, rhs in lead_specs:
        sample_label = 'matched' if 'Matched' in label else 'full'
        res, s = fit_ols(sub, rhs, label)
        if res is None:
            continue
        b_ai   = res.params.get(EXPOSURE, np.nan)
        b_lead = res.params.get('exposure_lead1', np.nan)
        p_ai   = res.pvalues.get(EXPOSURE, np.nan)
        p_lead = res.pvalues.get('exposure_lead1', np.nan)
        logger.info(f"  {label:<40}  N={len(s):>6,}  "
                    f"beta_current={b_ai:>+8.4f} (p={p_ai:.3f})  "
                    f"beta_lead={b_lead:>+8.4f} (p={p_lead:.3f})")
        all_rows.extend(extract_row(res, s, label, sample_label, rhs))

    if not all_rows:
        raise RuntimeError("No regressions produced output; check logs.")

    results = pd.DataFrame(all_rows)
    results.to_csv(output_file, index=False)
    logger.info(f"\n✓ Saved: {output_file} ({len(results)} rows)")
    logger.info("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
