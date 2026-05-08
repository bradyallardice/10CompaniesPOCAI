#!/usr/bin/env python3
"""
Stage 9d: Identification Robustness Checks — Economic Outcomes

For each outcome, tests robustness of the AI exposure effect to:
  1. Sample definition: matched-only (exposure > 0) vs full X28-linked (exposure >= 0)
  2. Firm size control (pw85, 9-category self-reported employer size)
  3. Firm hiring activity control (log total job postings per firm-year)
  4. Both controls simultaneously
  5. Placebo leads test: current vs future AI exposure

Outcomes:
  - outcome_job_insecurity  (pw86, 1–4 scale)
  - outcome_log_income      (log(iwyn + 1), CHF)
  - outcome_hours_worked    (pw77, hours/week, 1–99)
  - separation_t1           (binary LPM, forward-looking firm change)

All specs use:
  - Treatment:  hampole_ai_exposure_avg_foy (firm-occ-year, Stage 6 SHP)
  - FE:         Person + 3-digit ISCO × Year (within-demeaning, memory-efficient)
  - SE:         One-way cluster on idpers
  - Controls:   age_centered, age_squared, female, education

Key design notes:
  - Zero-fill (exposure == 0) = firm linked via X28 company_id but no AI apps detected
    for that occ-year. All zero-fill firms ARE in the X28 database by construction.
  - Matched-only (exposure > 0) = firm has confirmed positive AI adoption in that occ-year.
  - Within-demeaning avoids dummy variable memory issues.
  - Placebo leads: if future exposure predicts current outcome, suspect selection.

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
CONTROLS = ['age_centered', 'age_squared', 'female', 'education',
            'firm_size', 'public_sector']
OUTCOMES = {
    # Economic
    'outcome_job_insecurity':  'Job Insecurity (1-4)',
    'outcome_log_income':      'Log Income',
    'outcome_hours_worked':    'Hours Worked',
    'separation_t1':           'Separation (LPM)',
    # Labor market perceptions
    'unemp_risk':              'Perceived Unemployment Risk (0-10)',
    'job_satisfaction':        'Job Satisfaction Overall (0-10)',
    'employer_change':         'Employer Change (binary, LPM)',
    'restructuring':           'Firm Restructuring (binary, LPM)',
    # Job satisfaction sub-dimensions (all 0-10)
    'jobsat_income':           'Job Satisfaction: Income (0-10)',
    'jobsat_conditions':       'Job Satisfaction: Work Conditions (0-10)',
    'jobsat_atmosphere':       'Job Satisfaction: Atmosphere (0-10)',
    'jobsat_tasks':            'Job Satisfaction: Interest in Tasks (0-10)',
    'jobsat_workload':         'Job Satisfaction: Workload (0-10, to 2021)',
    # Work quality
    'work_intensity':          'Work Intensity/Pace (0-10)',
    'work_stress':             'Work Stress (binary, LPM)',
    'work_autonomy':           'Work Autonomy (1-3)',
    # Wellbeing
    'depression_anxiety':      'Depression/Anxiety Frequency (0-10)',
    'life_satisfaction':       'Life Satisfaction (0-10)',
    # Political — continuous annual coverage
    'leftright':               'Left-Right Self-Placement (0-10)',
    # Political — rotating battery (2014, 2017, 2020, 2023 only; low power)
    'nativism':                'Nativism / Opp. for Foreigners (1-3)',
    'welfare':                 'Welfare State Support (1-3)',
    'redistributive':          'Redistribution Support (1-3)',
    'gender_equality':         'Gender Equality (1-10)',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    for f in [panel_file, firm_file, shp_file]:
        if not f.exists():
            raise FileNotFoundError(f"Required input not found: {f}")

    logger.info("Loading panel data...")
    # Raw panel political columns — cleaned versions derived in build_analysis_frame
    raw_political = ['outcome_leftright', 'outcome_nativism', 'outcome_welfare',
                     'outcome_redistributive', 'outcome_gender_equality']
    want = ['idpers', 'firm_id', 'isco08_4d', 'year', EXPOSURE] + raw_political + CONTROLS + list(OUTCOMES.keys())
    panel = pd.read_csv(panel_file, low_memory=False, usecols=lambda c: c in want)
    panel['firm_id'] = panel['firm_id'].astype('Int64')
    panel['year']    = panel['year'].astype('int64')
    logger.info(f"  Panel: {len(panel):,} rows, {panel['idpers'].nunique():,} persons")

    logger.info("Loading firm report (hiring activity)...")
    firm = pd.read_csv(firm_file, usecols=['company_id', 'year', 'total_unique_job_ads'])
    firm['firm_id']     = ((firm['company_id'].astype('int64') + 13) * 13)
    firm['log_job_ads'] = np.log1p(firm['total_unique_job_ads'])
    firm['year']        = firm['year'].astype('int64')
    firm = firm[['firm_id', 'year', 'log_job_ads']]
    logger.info(f"  Firm report: {len(firm):,} firm-year rows")

    logger.info("Loading SHP auxiliary variables...")
    shp_cols = ['idpers', 'year', 'pw85', 'pw32', 'noga2m', 'pw101', 'pw228', 'pw18', 'pw602',
                'pw603', 'pw604', 'pw91', 'pc17', 'pc44',
                'pw92', 'pw93', 'pw94', 'pw229', 'pw230']
    shp = pd.read_csv(shp_file, usecols=lambda c: c in shp_cols, low_memory=False)

    def _pos(col):
        s = pd.to_numeric(shp[col], errors='coerce')
        return s.where(s >= 0)

    def _pos_nonzero(col):
        s = pd.to_numeric(shp[col], errors='coerce')
        return s.where(s > 0)

    def _binary(col, yes_codes, no_codes):
        s = pd.to_numeric(shp[col], errors='coerce')
        return np.where(s.isin(yes_codes), 1, np.where(s.isin(no_codes), 0, np.nan))

    # Firm size: pw85 categories 1-9; negative = missing
    shp['firm_size'] = _pos_nonzero('pw85')

    # Public sector: pw32 1=private, 2=public; negative = missing
    pw32 = pd.to_numeric(shp['pw32'], errors='coerce')
    shp['public_sector'] = (pw32 == 2).astype(float).where(pw32 > 0)

    # Industry code: noga2m 1-17 sectors; negative = missing
    noga = pd.to_numeric(shp['noga2m'], errors='coerce')
    shp['noga2m_clean'] = noga.where(noga > 0).astype('Int64')

    # Perceived unemployment risk: pw101 scale 0-10
    shp['unemp_risk'] = _pos('pw101')

    # Overall job satisfaction: pw228 scale 0-10
    shp['job_satisfaction'] = _pos('pw228')

    # Employer change: pw18 in {2,3} = employer changed; {1,4} = no employer change
    shp['employer_change'] = _binary('pw18', yes_codes=[2, 3], no_codes=[1, 4])

    # Restructuring: pw602 1=yes, 2=no
    shp['restructuring'] = _binary('pw602', yes_codes=[1], no_codes=[2])

    # Work intensity/pace: pw603 scale 0-10; negative = missing
    shp['work_intensity'] = _pos('pw603')

    # Work stress: pw604 binary; 1=yes (stressed), 2=no
    shp['work_stress'] = _binary('pw604', yes_codes=[1], no_codes=[2])

    # Work autonomy: pw91 ordinal 1-3 (higher = more autonomy); negative = missing
    shp['work_autonomy'] = _pos_nonzero('pw91')

    # Depression/anxiety frequency: pc17 scale 0-10
    shp['depression_anxiety'] = _pos('pc17')

    # Life satisfaction: pc44 scale 0-10
    shp['life_satisfaction'] = _pos('pc44')

    # Job satisfaction sub-dimensions: pw92-pw94, pw229, pw230 all 0-10
    shp['jobsat_income']     = _pos('pw92')
    shp['jobsat_conditions'] = _pos('pw93')
    shp['jobsat_atmosphere'] = _pos('pw94')
    shp['jobsat_tasks']      = _pos('pw229')
    shp['jobsat_workload']   = _pos('pw230')

    keep_cols = ['idpers', 'year', 'firm_size', 'public_sector', 'noga2m_clean',
                 'unemp_risk', 'job_satisfaction',
                 'employer_change', 'restructuring', 'work_intensity', 'work_stress',
                 'work_autonomy', 'depression_anxiety', 'life_satisfaction',
                 'jobsat_income', 'jobsat_conditions', 'jobsat_atmosphere',
                 'jobsat_tasks', 'jobsat_workload']
    shp = shp[keep_cols]

    for c in keep_cols[2:]:
        logger.info(f"  {c}: {shp[c].notna().sum():,} valid")

    return panel, firm, shp


def build_analysis_frame(panel, firm, shp):
    df = panel.copy()
    df['firm_id_int'] = pd.to_numeric(df['firm_id'], errors='coerce').astype('Int64')
    df['isco3d']      = (pd.to_numeric(df['isco08_4d'], errors='coerce') // 10).astype(str)
    df['occ_year']    = df['isco3d'] + '_' + df['year'].astype(str)

    df = df.merge(shp, on=['idpers', 'year'], how='left')
    firm_r = firm.rename(columns={'firm_id': 'firm_id_int'})
    df = df.merge(firm_r, on=['firm_id_int', 'year'], how='left')

    # Clean negative SHP missing codes in panel-sourced political outcomes
    lr = pd.to_numeric(df['outcome_leftright'], errors='coerce')
    df['leftright'] = lr.where(lr >= 0)

    # Rotating battery political outcomes: valid range is > 0 (1-3 or 1-10 scales)
    for raw_col, clean_col in [('outcome_nativism',       'nativism'),
                                ('outcome_welfare',        'welfare'),
                                ('outcome_redistributive', 'redistributive'),
                                ('outcome_gender_equality','gender_equality')]:
        s = pd.to_numeric(df[raw_col], errors='coerce')
        df[clean_col] = s.where(s > 0)

    df['matched']   = df[EXPOSURE] > 0
    df['x28_linked'] = df[EXPOSURE].notna()

    logger.info("\nSample breakdown (all rows):")
    logger.info(f"  exposure > 0  (matched, positive AI):         {df['matched'].sum():>7,}")
    logger.info(f"  exposure == 0 (X28-linked, zero AI):          {((df[EXPOSURE]==0) & df['x28_linked']).sum():>7,}")
    logger.info(f"  exposure NaN  (no firm or occ match):         {df[EXPOSURE].isna().sum():>7,}")

    # Lead exposure (t+1), consecutive years only
    df = df.sort_values(['idpers', 'year']).reset_index(drop=True)
    year_next = df.groupby('idpers')['year'].shift(-1)
    consecutive = (year_next - df['year'] == 1)
    df['exposure_lead1'] = df.groupby('idpers')[EXPOSURE].shift(-1).where(consecutive)

    return df


# ── Estimation ────────────────────────────────────────────────────────────────

def within_demean(data, groups, value_cols):
    out = data.copy()
    for g in groups:
        for v in value_cols:
            out[v] = out[v] - out.groupby(g)[v].transform('mean')
    return out


def fit_ols(sub, outcome, rhs_cols):
    FE = ['idpers', 'occ_year']
    needed = FE + [outcome] + [c for c in rhs_cols if c in sub.columns]
    s = sub[needed].dropna().copy()
    if len(s) < 50:
        return None, s
    dm = within_demean(s, FE, [outcome] + rhs_cols)
    X  = sm.add_constant(dm[rhs_cols])
    res = sm.OLS(dm[outcome], X).fit(
        cov_type='cluster', cov_kwds={'groups': s['idpers']})
    return res, s


def extract_rows(res, s, outcome, label, sample_label, rhs_cols):
    rows = []
    for term in rhs_cols:
        if term not in res.params.index:
            continue
        rows.append({
            'outcome':    outcome,
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("STAGE 9D: IDENTIFICATION ROBUSTNESS — ECONOMIC OUTCOMES")
    logger.info("=" * 70)

    panel, firm, shp = load_data()
    df = build_analysis_frame(panel, firm, shp)

    all_rows = []

    for outcome, outcome_label in OUTCOMES.items():
        logger.info(f"\n{'=' * 70}")
        logger.info(f"OUTCOME: {outcome_label} ({outcome})")
        logger.info(f"Person + 3d-ISCO×Year FE | one-way cluster on idpers")
        logger.info(f"{'=' * 70}")

        # Build outcome-specific estimation frames
        base_mask = df[[outcome] + CONTROLS + [EXPOSURE]].notna().all(axis=1)
        full    = df[base_mask].copy()
        matched = full[full['matched']].copy()
        logger.info(f"  Full X28-linked: N={len(full):,} | Matched-only: N={len(matched):,}")

        main_specs = [
            ('(1) Matched-only, baseline',          matched, [EXPOSURE] + CONTROLS),
            ('(2) Full sample, baseline',            full,    [EXPOSURE] + CONTROLS),
            ('(3) Full + firm size',                 full,    [EXPOSURE, 'firm_size'] + CONTROLS),
            ('(4) Full + log(job ads)',              full,    [EXPOSURE, 'log_job_ads'] + CONTROLS),
            ('(5) Full + firm size + log(job ads)', full,    [EXPOSURE, 'firm_size', 'log_job_ads'] + CONTROLS),
        ]

        logger.info(f"\n  {'Spec':<45} {'N':>7}  {'beta_AI':>9}  {'SE':>7}  {'p':>6}")
        logger.info(f"  {'-'*75}")
        for label, sub, rhs in main_specs:
            samp = 'matched' if 'Matched' in label else 'full'
            res, s = fit_ols(sub, outcome, rhs)
            if res is None:
                logger.warning(f"  {label}: insufficient sample (N={len(s)})")
                continue
            b  = res.params.get(EXPOSURE, np.nan)
            se = res.bse.get(EXPOSURE, np.nan)
            p  = res.pvalues.get(EXPOSURE, np.nan)
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
            logger.info(f"  {label:<45} {len(s):>7,}  {b:>+9.4f}  {se:>7.4f}  {p:>6.3f} {sig}")
            all_rows.extend(extract_rows(res, s, outcome, label, samp, rhs))

        lead_specs = [
            ('Matched: current only',       matched, [EXPOSURE] + CONTROLS),
            ('Matched: horse race',          matched, [EXPOSURE, 'exposure_lead1'] + CONTROLS),
            ('Matched: lead only (placebo)', matched, ['exposure_lead1'] + CONTROLS),
            ('Full: current only',          full,    [EXPOSURE] + CONTROLS),
            ('Full: horse race',            full,    [EXPOSURE, 'exposure_lead1'] + CONTROLS),
            ('Full: lead only (placebo)',   full,    ['exposure_lead1'] + CONTROLS),
        ]

        logger.info(f"\n  Placebo leads:")
        logger.info(f"  {'Spec':<40} {'N':>7}  {'beta_current':>13}  {'beta_lead':>10}")
        logger.info(f"  {'-'*75}")
        for label, sub, rhs in lead_specs:
            samp = 'matched' if 'Matched' in label else 'full'
            res, s = fit_ols(sub, outcome, rhs)
            if res is None:
                continue
            b_curr = res.params.get(EXPOSURE, np.nan)
            b_lead = res.params.get('exposure_lead1', np.nan)
            p_curr = res.pvalues.get(EXPOSURE, np.nan)
            p_lead = res.pvalues.get('exposure_lead1', np.nan)
            logger.info(f"  {label:<40} {len(s):>7,}  "
                        f"{b_curr:>+13.4f} (p={p_curr:.3f})  "
                        f"{b_lead:>+10.4f} (p={p_lead:.3f})")
            all_rows.extend(extract_rows(res, s, outcome, label, samp, rhs))

    if not all_rows:
        raise RuntimeError("No regressions produced output; check logs.")

    results = pd.DataFrame(all_rows)
    results.to_csv(output_file, index=False)
    logger.info(f"\n{'=' * 70}")
    logger.info(f"✓ Saved: {output_file} ({len(results)} rows)")
    logger.info("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
