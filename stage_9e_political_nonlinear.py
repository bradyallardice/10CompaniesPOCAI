#!/usr/bin/env python3
"""
Stage 9e: Political Outcomes, Non-linearities, Interactions, Heterogeneity, and Firm Hiring

Baseline spec (all person-level modules):
  - Sample:    matched-only (hampole_ai_exposure_avg_foy > 0)
  - FE:        Person + 3-digit ISCO × Year (within-demeaning)
  - SE:        One-way cluster on idpers
  - Treatment: hampole_ai_exposure_avg_foy

Modules:
  A. Political outcomes — full table
     Annual (strong power): vote_svp, vote_sp_gps, leftright, social_trust
     Rotating battery:      welfare, redistributive, nativism, trust_govt,
                            democracy_sat, political_efficacy, eu_opinion
  B. Non-linearity tests (job insecurity + vote_svp + redistribution)
     1. Quadratic: exposure + exposure²
     2. Tercile dummies: T2, T3 vs T1 among matched
  C. Interactions with economic fear
     Moderators (centered at matched-sample mean):
       - job_insecurity (1–4): fear of job loss
       - unemp_risk (pw101, 0–10): perceived unemployment risk
     Outcomes: vote_svp, vote_sp_gps, leftright, redistributive, welfare
  D. Subgroup stability
     Splits: gender, education (high ≥ 7 vs low ≤ 6), period (pre/post 2018)
     Outcomes: outcome_job_insecurity, vote_svp, redistributive
  E. Firm-level future hiring (firm × year panel)
     Outcome: log(job_ads at t+1), Δlog(job_ads)
     Treatment: pct_ai_ads_cumulative (main), firm_ai_exposure (robustness)
     FE: firm + year; SE: cluster on company_id

Inputs:
  Data/shp_panel_prepared.csv
  Data/firm_ai_summary_report.csv
  Data/shp_exposure/shp_exposure_isco4d.csv

Outputs:
  Data/stage9e_results_long.csv    (modules A–D)
  Data/stage9e_firm_hiring.csv     (module E)
  Data/stage9e_estimation_log.txt
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
out_person   = data_dir / "stage9e_results_long.csv"
out_firm     = data_dir / "stage9e_firm_hiring.csv"
log_file     = data_dir / "stage9e_estimation_log.txt"

for f in [panel_file, firm_file, shp_file]:
    if not f.exists():
        raise FileNotFoundError(f"Required input not found: {f}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

EXPOSURE = 'hampole_ai_exposure_avg_foy'
# All controls are time-varying and pre-determined relative to current AI exposure.
# female/education dropped — time-invariant, fully absorbed by person FE.
# firm_size and log_job_ads lagged (t-1) to avoid conditioning on outcomes caused by treatment.
CONTROLS = ['age_centered', 'age_squared',
            'contract_perm',      # permanent contract (pw36==2 vs fixed-term)
            'part_time',          # part-time (pw39==1 vs full-time)
            'computer_work',      # uses computer at work (pw607) — tech-intensity proxy
            'firm_size',          # current-year employer size (pw85)
            'public_sector',      # public employer (pw32==2 vs private)
            'log_job_ads_lag1']   # lagged firm hiring volume (log job ads at t-1)

# Controls without lagged firm variables — used in Module F robustness comparison.
CONTROLS_NO_FIRM_LAG = ['age_centered', 'age_squared',
                        'contract_perm', 'part_time', 'computer_work',
                        'firm_size', 'public_sector']
PERSON_FE  = 'idpers'
OCC_YEAR_FE = 'occ_year'
IND_YEAR_FE = 'ind_year'   # industry (NOGA2M 17-sector) × year — third FE


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    logger.info("Loading panel data...")
    panel_want = [
        'idpers', 'firm_id', 'isco08_4d', 'year', EXPOSURE,
        'outcome_job_insecurity', 'outcome_leftright', 'outcome_nativism',
        'outcome_welfare', 'outcome_redistributive', 'outcome_gender_equality',
        'pp10', 'pp13', 'pp15', 'pp17', 'pp22',
        'separation_t1',
        'female', 'education',   # kept for subgroup splits in Module D, not regression controls
    ] + CONTROLS
    panel = pd.read_csv(panel_file, low_memory=False,
                        usecols=lambda c: c in panel_want)
    panel['firm_id'] = panel['firm_id'].astype('Int64')
    panel['year']    = panel['year'].astype('int64')
    logger.info(f"  Panel: {len(panel):,} rows, {panel['idpers'].nunique():,} persons")

    logger.info("Loading SHP auxiliary variables (extended political + fear moderators)...")
    shp_want = [
        'idpers', 'year',
        # Fear moderators
        'pw101',
        # Vote intention (numeric coding: 3=SP, 4=SVP, 9=GPS)
        'pp19',
        # Political — rotating / annual
        'pp02', 'pp03', 'pp04', 'pp14', 'pp45',
        # Firm-level controls (lagged in build_analysis_frame)
        'pw85',   # self-reported employer size (1-9)
        # Individual job controls (time-varying, pre-determined)
        'pw36',   # contract type: 1=fixed-term, 2=permanent
        'pw39',   # working time: 1=part-time, 2=full-time
        'pw607',  # computer use at work: 1=yes, 2=no
    ]
    shp = pd.read_csv(shp_file, low_memory=False,
                      usecols=lambda c: c in shp_want)
    shp['year'] = pd.to_numeric(shp['year'], errors='coerce').astype('Int64')

    # Construct vote variables from numeric pp19 (stage_8a construction was broken)
    # pp19 valid codes: 1=FDP, 2=CVP, 3=SP, 4=SVP, 9=GPS, 50=no party, 51=wouldn't vote, 52=spoiled
    pp19 = pd.to_numeric(shp['pp19'], errors='coerce')
    pp19_valid = pp19.where(pp19 > 0)   # drops negative missing codes (-1, -2, -3)

    def _vote(codes):
        v = pp19_valid.isin(codes) if isinstance(codes, list) else (pp19_valid == codes)
        return v.astype(float).where(pp19_valid.notna())

    # Main parties — disaggregated
    shp['vote_svp']       = _vote(4)           # SVP/UDC right-populist
    shp['vote_sp']        = _vote(3)           # SP Social Democrats
    shp['vote_fdp']       = _vote(1)           # FDP Liberals (center-right)
    shp['vote_cvp']       = _vote(2)           # CVP/PDC Christian Democrats (center)
    shp['vote_glp']       = _vote([11, 20])    # GLP Green Liberals (11=old code, 20=new)
    shp['vote_bdp']       = _vote(21)          # BDP Conservative Democrats
    shp['vote_no_party']  = _vote([50, 51])    # No party / wouldn't vote (disengagement)
    # Legacy grouped (kept for Module C/D compatibility)
    shp['vote_sp_gps']    = _vote([3, 9])

    def _pos(col):
        s = pd.to_numeric(shp[col], errors='coerce')
        return s.where(s >= 0)            # 0–10 scale: keep 0

    def _pos_nonzero(col):
        s = pd.to_numeric(shp[col], errors='coerce')
        return s.where(s > 0)             # 1–3 scale: drop 0 (missing code)

    shp['unemp_risk']         = _pos('pw101')          # 0–10
    shp['democracy_sat']      = _pos('pp02')            # 0–10
    shp['political_efficacy'] = _pos('pp03')            # 0–10
    shp['trust_govt']         = _pos('pp04')            # 0–10
    shp['eu_opinion']         = _pos_nonzero('pp14')    # 1–3
    shp['social_trust']       = _pos('pp45')            # 0–10
    shp['computer_work'] = pd.to_numeric(shp['pw607'], errors='coerce').apply(
                               lambda x: 1 if x == 1 else (0 if x == 2 else np.nan))
    pw85 = pd.to_numeric(shp['pw85'], errors='coerce')
    shp['firm_size'] = pw85.where(pw85 > 0)         # 1-9 ordinal; will be lagged in build_analysis_frame

    pw36 = pd.to_numeric(shp['pw36'], errors='coerce')
    shp['contract_perm'] = (pw36 == 2).astype(float).where(pw36 > 0)  # 1=permanent, 0=fixed-term

    pw39 = pd.to_numeric(shp['pw39'], errors='coerce')
    shp['part_time'] = (pw39 == 1).astype(float).where(pw39 > 0)      # 1=part-time, 0=full-time

    keep = ['idpers', 'year', 'unemp_risk', 'democracy_sat', 'political_efficacy',
            'trust_govt', 'eu_opinion', 'social_trust',
            'firm_size', 'contract_perm', 'part_time', 'computer_work',
            'vote_svp', 'vote_sp', 'vote_fdp', 'vote_cvp', 'vote_glp', 'vote_bdp',
            'vote_no_party', 'vote_sp_gps']
    shp = shp[keep]
    for c in keep[2:]:
        logger.info(f"  {c}: {shp[c].notna().sum():,} valid")
    logger.info(f"  vote_svp base rate: {shp['vote_svp'].mean():.3f} | "
                f"vote_sp_gps base rate: {shp['vote_sp_gps'].mean():.3f}")

    logger.info("Loading firm report...")
    firm = pd.read_csv(firm_file)
    firm['year'] = firm['year'].astype('int64')
    logger.info(f"  Firm report: {len(firm):,} firm-year rows, "
                f"{firm['company_id'].nunique():,} companies")

    return panel, shp, firm


def build_analysis_frame(panel, shp, firm):
    df = panel.copy()
    df['isco3d']   = (pd.to_numeric(df['isco08_4d'], errors='coerce') // 10).astype(str)
    df['occ_year'] = df['isco3d'] + '_' + df['year'].astype(str)

    df = df.merge(shp, on=['idpers', 'year'], how='left')

    # Build firm-level log_job_ads with lag at firm×year level (avoids endogeneity)
    firm_ctrl = firm[['company_id', 'year', 'total_unique_job_ads']].copy()
    firm_ctrl['firm_id_int'] = ((firm_ctrl['company_id'].astype('int64') + 13) * 13)
    firm_ctrl['log_job_ads']  = np.log1p(firm_ctrl['total_unique_job_ads'])
    firm_ctrl = firm_ctrl.sort_values(['firm_id_int', 'year']).reset_index(drop=True)
    yr_prev_firm = firm_ctrl.groupby('firm_id_int')['year'].shift(1)
    consec_firm  = (firm_ctrl['year'] - yr_prev_firm == 1)
    firm_ctrl['log_job_ads_lag1'] = (firm_ctrl.groupby('firm_id_int')['log_job_ads']
                                     .shift(1).where(consec_firm))
    df['firm_id_int'] = pd.to_numeric(df['firm_id'], errors='coerce').astype('Int64')
    df = df.merge(firm_ctrl[['firm_id_int', 'year', 'log_job_ads_lag1']],
                  on=['firm_id_int', 'year'], how='left')

    df = df.sort_values(['idpers', 'year']).reset_index(drop=True)

    # Clean political variables already in panel (negative = missing code)
    for raw, clean, use_pos in [
        ('outcome_job_insecurity', 'job_insecurity', False),  # 1-4
        ('outcome_leftright',      'leftright',      True),   # 0-10
        ('outcome_nativism',       'nativism',       False),  # 1-3
        ('outcome_welfare',        'welfare',        False),  # 1-3
        ('outcome_redistributive', 'redistributive', False),  # 1-3
        ('outcome_gender_equality','gender_equality', True),  # 0-10
    ]:
        s = pd.to_numeric(df[raw], errors='coerce') if raw in df.columns else pd.Series(dtype=float)
        df[clean] = s.where(s >= 0) if use_pos else s.where(s > 0)

    df['matched'] = df[EXPOSURE] > 0

    logger.info("\nSample breakdown:")
    logger.info(f"  exposure > 0  (matched):  {df['matched'].sum():>7,}")
    logger.info(f"  exposure == 0 (zero-fill): {((df[EXPOSURE]==0) & df[EXPOSURE].notna()).sum():>7,}")
    logger.info(f"  exposure NaN:              {df[EXPOSURE].isna().sum():>7,}")

    # Log education distribution
    edu = df.loc[df['matched'], 'education'].dropna()
    logger.info(f"\n  Education distribution (matched): {edu.value_counts().sort_index().to_dict()}")

    # Log control coverage in matched sample
    matched = df[df['matched']]
    for ctrl in CONTROLS:
        if ctrl in df.columns:
            n_valid = matched[ctrl].notna().sum()
            pct = 100 * n_valid / len(matched)
            logger.info(f"  Control {ctrl}: {n_valid:,}/{len(matched):,} valid ({pct:.1f}%)")

    return df


# ── Estimation helpers ─────────────────────────────────────────────────────────

def within_demean(data, groups, value_cols):
    out = data.copy()
    for g in groups:
        for v in value_cols:
            out[v] = out[v] - out.groupby(g)[v].transform('mean')
    return out


def fit_ols(sub, outcome, rhs_cols):
    FE = [PERSON_FE, OCC_YEAR_FE]
    needed = FE + [outcome] + [c for c in rhs_cols if c in sub.columns]
    s = sub[needed].dropna().copy()
    if len(s) < 50:
        return None, s
    dm = within_demean(s, FE, [outcome] + rhs_cols)
    X  = sm.add_constant(dm[rhs_cols])
    res = sm.OLS(dm[outcome], X).fit(
        cov_type='cluster', cov_kwds={'groups': s[PERSON_FE]})
    return res, s


def make_row(res, s, outcome, label, term, module):
    if term not in res.params.index:
        return None
    return {
        'module':     module,
        'outcome':    outcome,
        'spec':       label,
        'term':       term,
        'estimate':   res.params[term],
        'std_error':  res.bse[term],
        't_stat':     res.tvalues[term],
        'p_value':    res.pvalues[term],
        'ci_lower':   res.conf_int().loc[term, 0],
        'ci_upper':   res.conf_int().loc[term, 1],
        'n_obs':      len(s),
        'n_persons':  s[PERSON_FE].nunique(),
    }


def log_result(label, res, s, term=EXPOSURE):
    if res is None:
        logger.info(f"  {label:<55} N too small")
        return
    b  = res.params.get(term, np.nan)
    se = res.bse.get(term, np.nan)
    p  = res.pvalues.get(term, np.nan)
    sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '   '))
    logger.info(f"  {label:<55} N={len(s):>6,}  β={b:>+8.4f}  SE={se:>7.4f}  p={p:>6.3f} {sig}")


# ── Module A: Political outcomes ───────────────────────────────────────────────

def module_a(df):
    logger.info("\n" + "=" * 70)
    logger.info("MODULE A: POLITICAL OUTCOMES — FULL TABLE")
    logger.info("Spec: matched-only | Person + 3d-ISCO×Year FE | cluster idpers")
    logger.info("=" * 70)

    matched = df[df['matched']].copy()

    # Log within-person vote variation as diagnostic
    for v in ['vote_svp', 'vote_sp_gps']:
        sub = matched.dropna(subset=[v])
        wp = sub.groupby('idpers')[v].nunique()
        switchers = (wp > 1).sum()
        logger.info(f"  {v}: N={len(sub):,}, switchers={switchers} ({100*switchers/len(wp):.1f}% of persons) "
                    f"— person FE identifies only within-person switches")

    # Vote diagnostics
    vote_vars = ['vote_svp', 'vote_sp', 'vote_fdp', 'vote_cvp', 'vote_glp', 'vote_bdp', 'vote_no_party']
    for v in vote_vars:
        sub = matched.dropna(subset=[v])
        wp = sub.groupby('idpers')[v].nunique()
        switchers = (wp > 1).sum()
        base = sub[v].mean()
        logger.info(f"  {v}: N={len(sub):,}, base={base:.3f}, switchers={switchers} ({100*switchers/len(wp):.1f}%)")

    # Annual outcomes — disaggregated by party
    # NOTE: ~6-11% within-person switching per party; person FE identifies only switchers
    annual = [
        ('vote_svp',      'Vote SVP/UDC — right-populist (binary)'),
        ('vote_sp',       'Vote SP — Social Democrats (binary)'),
        ('vote_fdp',      'Vote FDP — Liberals/center-right (binary)'),
        ('vote_cvp',      'Vote CVP/PDC — Christian Dem/center (binary)'),
        ('vote_glp',      'Vote GLP — Green Liberals (binary)'),
        ('vote_bdp',      'Vote BDP — Conservative Dem (binary)'),
        ('vote_no_party', 'Vote no party / wouldn\'t vote (disengagement)'),
        ('leftright',     'Left-Right Self-Placement (0-10, annual)'),
        ('social_trust',  'Social Trust (pp45, 0-10, from 2002)'),
    ]
    # Rotating battery (annual 1999-2009, then every 3 years — lower power)
    rotating = [
        ('welfare',            'Social Spending Support (pp13, 1-3)'),
        ('redistributive',     'Redistribution Support (pp17, 1-3)'),
        ('nativism',           'Nativism / Opp. for Foreigners (pp15, 1-3)'),
        ('gender_equality',    'Gender Equality (pp22, 0-10)'),
        ('trust_govt',         'Trust in Federal Government (pp04, 0-10)'),
        ('democracy_sat',      'Satisfaction with Democracy (pp02, 0-10)'),
        ('political_efficacy', 'Political Efficacy (pp03, 0-10)'),
        ('eu_opinion',         'EU Opinion (pp14, 1-3)'),
    ]

    rows = []
    logger.info("\n  ANNUAL (strong power):")
    logger.info(f"  {'Outcome':<50} {'N':>7}  {'beta_AI':>9}  {'SE':>7}  {'p':>6}")
    logger.info(f"  {'-'*80}")
    for outcome, label in annual:
        res, s = fit_ols(matched, outcome, [EXPOSURE] + CONTROLS)
        log_result(label, res, s)
        if res is not None:
            r = make_row(res, s, outcome, 'baseline', EXPOSURE, 'A')
            if r: rows.append(r)

    logger.info("\n  ROTATING BATTERY (lower power — waves 1999-2009 annual, then 2011/14/17/20/23):")
    logger.info(f"  {'Outcome':<50} {'N':>7}  {'beta_AI':>9}  {'SE':>7}  {'p':>6}")
    logger.info(f"  {'-'*80}")
    for outcome, label in rotating:
        res, s = fit_ols(matched, outcome, [EXPOSURE] + CONTROLS)
        log_result(label, res, s)
        if res is not None:
            r = make_row(res, s, outcome, 'baseline', EXPOSURE, 'A')
            if r: rows.append(r)

    return rows


# ── Module B: Non-linearities ──────────────────────────────────────────────────

def module_b(df):
    logger.info("\n" + "=" * 70)
    logger.info("MODULE B: NON-LINEARITIES (quadratic + tercile dummies)")
    logger.info("=" * 70)

    matched = df[df['matched']].copy()

    # Tercile cutpoints from matched distribution
    t33, t67 = matched[EXPOSURE].quantile([1/3, 2/3])
    matched['exp_t2'] = ((matched[EXPOSURE] >  t33) & (matched[EXPOSURE] <= t67)).astype(float)
    matched['exp_t3'] = (matched[EXPOSURE] >  t67).astype(float)
    logger.info(f"\n  Exposure tercile cutpoints (matched): T1≤{t33:.4f} | T2≤{t67:.4f} | T3>{t67:.4f}")
    logger.info(f"  N per tercile: T1={( matched[EXPOSURE] <= t33).sum():,}  "
                f"T2={(matched['exp_t2']==1).sum():,}  T3={(matched['exp_t3']==1).sum():,}")

    outcomes_b = [
        ('job_insecurity',  'Job Insecurity (1-4)'),
        ('vote_svp',        'Vote SVP/UDC (binary)'),
        ('vote_sp_gps',     'Vote SP/GPS (binary)'),
        ('redistributive',  'Redistribution Support (1-3)'),
    ]

    rows = []
    for outcome, label in outcomes_b:
        logger.info(f"\n  Outcome: {label}")

        # Quadratic
        matched['exp2'] = matched[EXPOSURE] ** 2
        res_q, s_q = fit_ols(matched, outcome, [EXPOSURE, 'exp2'] + CONTROLS)
        log_result(f"  Quadratic (exposure²)", res_q, s_q)
        if res_q is not None:
            for term in [EXPOSURE, 'exp2']:
                r = make_row(res_q, s_q, outcome, 'quadratic', term, 'B')
                if r: rows.append(r)

        # Tercile dummies (T1 = reference)
        res_t, s_t = fit_ols(matched, outcome, ['exp_t2', 'exp_t3'] + CONTROLS)
        if res_t is not None:
            b2  = res_t.params.get('exp_t2', np.nan)
            b3  = res_t.params.get('exp_t3', np.nan)
            p2  = res_t.pvalues.get('exp_t2', np.nan)
            p3  = res_t.pvalues.get('exp_t3', np.nan)
            logger.info(f"    Tercile dummies  N={len(s_t):>6,}  "
                        f"T2 β={b2:>+7.4f}(p={p2:.3f})  T3 β={b3:>+7.4f}(p={p3:.3f})  [T1=ref]")
            for term in ['exp_t2', 'exp_t3']:
                r = make_row(res_t, s_t, outcome, 'tercile_dummies', term, 'B')
                if r: rows.append(r)

    return rows


# ── Module C: Interactions with economic fear ──────────────────────────────────

def module_c(df):
    logger.info("\n" + "=" * 70)
    logger.info("MODULE C: INTERACTIONS WITH ECONOMIC FEAR")
    logger.info("  Moderator 1: job insecurity (1-4) — fear of job loss")
    logger.info("  Moderator 2: perceived unemployment risk pw101 (0-10)")
    logger.info("  Interaction: exposure × moderator_centered")
    logger.info("=" * 70)

    matched = df[df['matched']].copy()

    # Center moderators at matched-sample mean (excluding missing)
    for mod_raw, mod_centered in [('job_insecurity', 'ji_c'), ('unemp_risk', 'ur_c')]:
        mu = matched[mod_raw].mean(skipna=True)
        matched[mod_centered] = matched[mod_raw] - mu
        logger.info(f"\n  {mod_raw}: mean={mu:.3f}, centered → {mod_centered}")

    matched['ix_ji'] = matched[EXPOSURE] * matched['ji_c']   # exposure × job_insecurity
    matched['ix_ur'] = matched[EXPOSURE] * matched['ur_c']   # exposure × unemp_risk

    outcomes_c = [
        ('vote_svp',       'Vote SVP/UDC'),
        ('vote_sp_gps',    'Vote SP/GPS'),
        ('vote_sp',        'Vote SP'),
        ('leftright',      'Left-Right (0-10)'),
        ('redistributive', 'Redistribution Support'),
        ('welfare',        'Social Spending Support'),
        ('job_insecurity', 'Job Insecurity (1-4)'),
    ]

    rows = []
    for outcome, label in outcomes_c:
        logger.info(f"\n  Outcome: {label}")

        # Interaction with job insecurity
        rhs_ji = [EXPOSURE, 'ji_c', 'ix_ji'] + CONTROLS
        res_ji, s_ji = fit_ols(matched, outcome, rhs_ji)
        if res_ji is not None:
            b_main = res_ji.params.get(EXPOSURE, np.nan)
            b_ix   = res_ji.params.get('ix_ji', np.nan)
            p_main = res_ji.pvalues.get(EXPOSURE, np.nan)
            p_ix   = res_ji.pvalues.get('ix_ji', np.nan)
            logger.info(f"    × job_insecurity  N={len(s_ji):>6,}  "
                        f"β_exp={b_main:>+7.4f}(p={p_main:.3f})  "
                        f"β_ix={b_ix:>+7.4f}(p={p_ix:.3f})")
            for term in [EXPOSURE, 'ji_c', 'ix_ji']:
                r = make_row(res_ji, s_ji, outcome, 'interaction_job_insecurity', term, 'C')
                if r: rows.append(r)

        # Interaction with unemployment risk
        rhs_ur = [EXPOSURE, 'ur_c', 'ix_ur'] + CONTROLS
        res_ur, s_ur = fit_ols(matched, outcome, rhs_ur)
        if res_ur is not None:
            b_main = res_ur.params.get(EXPOSURE, np.nan)
            b_ix   = res_ur.params.get('ix_ur', np.nan)
            p_main = res_ur.pvalues.get(EXPOSURE, np.nan)
            p_ix   = res_ur.pvalues.get('ix_ur', np.nan)
            logger.info(f"    × unemp_risk      N={len(s_ur):>6,}  "
                        f"β_exp={b_main:>+7.4f}(p={p_main:.3f})  "
                        f"β_ix={b_ix:>+7.4f}(p={p_ix:.3f})")
            for term in [EXPOSURE, 'ur_c', 'ix_ur']:
                r = make_row(res_ur, s_ur, outcome, 'interaction_unemp_risk', term, 'C')
                if r: rows.append(r)

    return rows


# ── Module D: Subgroup stability ───────────────────────────────────────────────

def module_d(df):
    logger.info("\n" + "=" * 70)
    logger.info("MODULE D: SUBGROUP STABILITY")
    logger.info("  Splits: gender | education (≥7 vs ≤6) | period (pre/post 2018)")
    logger.info("=" * 70)

    matched = df[df['matched']].copy()

    edu_high = matched['education'] >= 7   # university / applied sciences
    edu_low  = matched['education'] <= 6

    splits = [
        ('Full matched',    matched,                          'full'),
        ('Male',            matched[matched['female'] == 0],  'male'),
        ('Female',          matched[matched['female'] == 1],  'female'),
        ('High education',  matched[edu_high],                'edu_high'),
        ('Low education',   matched[edu_low],                 'edu_low'),
        ('Pre-2018',        matched[matched['year'] < 2018],  'pre2018'),
        ('Post-2018',       matched[matched['year'] >= 2018], 'post2018'),
    ]

    outcomes_d = [
        ('job_insecurity',  'Job Insecurity (1-4)'),
        ('vote_svp',        'Vote SVP/UDC'),
        ('vote_sp',         'Vote SP'),
        ('vote_fdp',        'Vote FDP'),
        ('vote_glp',        'Vote GLP'),
        ('redistributive',  'Redistribution Support'),
    ]

    rows = []
    for outcome, label in outcomes_d:
        logger.info(f"\n  Outcome: {label}")
        logger.info(f"  {'Split':<22} {'N':>7}  {'beta_AI':>9}  {'SE':>7}  {'p':>6}")
        logger.info(f"  {'-'*55}")
        for split_label, sub, split_key in splits:
            res, s = fit_ols(sub, outcome, [EXPOSURE] + CONTROLS)
            log_result(split_label, res, s)
            if res is not None:
                r = make_row(res, s, outcome, f'subgroup_{split_key}', EXPOSURE, 'D')
                if r: rows.append(r)

    return rows


# ── Module E: Firm-level future hiring ─────────────────────────────────────────

def module_e(firm):
    logger.info("\n" + "=" * 70)
    logger.info("MODULE E: FIRM-LEVEL FUTURE HIRING")
    logger.info("  Unit: firm × year | FE: firm + year | SE: cluster company_id")
    logger.info("  Outcome 1: log(job_ads at t+1) — future hiring level")
    logger.info("  Outcome 2: Δlog(job_ads) = log(t+1) - log(t) — hiring growth rate")
    logger.info("  No current log control: firm FE absorbs persistent level differences.")
    logger.info("  Treatment: pct_ai_ads_cumulative (main), firm_ai_exposure, log_ai_apps")
    logger.info("=" * 70)

    df = firm.copy()
    df = df.sort_values(['company_id', 'year']).reset_index(drop=True)

    df['log_job_ads']      = np.log1p(df['total_unique_job_ads'])
    df['log_ai_apps']      = np.log1p(df['total_ai_apps_all'])

    # Lead outcomes (t+1): require consecutive year
    df['year_next']          = df.groupby('company_id')['year'].shift(-1)
    consecutive              = (df['year_next'] - df['year'] == 1)
    df['log_job_ads_lead1']  = df.groupby('company_id')['log_job_ads'].shift(-1).where(consecutive)
    df['delta_log_job_ads']  = df['log_job_ads_lead1'] - df['log_job_ads']

    df['year_str'] = df['year'].astype(str)

    logger.info(f"\n  Firm-year panel: {len(df):,} rows, {df['company_id'].nunique():,} firms")
    logger.info(f"  With lead outcome: {df['log_job_ads_lead1'].notna().sum():,}")

    def fit_firm_ols(sub, outcome, treat_cols, control_cols):
        fe = ['company_id', 'year_str']
        needed = fe + [outcome] + treat_cols + control_cols
        s = sub[[c for c in needed if c in sub.columns]].dropna().copy()
        if len(s) < 30:
            return None, s
        # Within-demean by firm and year
        rhs = treat_cols + [c for c in control_cols if c in s.columns]
        dm = s.copy()
        for g in fe:
            for v in [outcome] + rhs:
                dm[v] = dm[v] - dm.groupby(g)[v].transform('mean')
        X   = sm.add_constant(dm[rhs])
        res = sm.OLS(dm[outcome], X).fit(
            cov_type='cluster', cov_kwds={'groups': s['company_id']})
        return res, s

    treat_pairs = [
        ('pct_ai_ads_cumulative', 'AI adoption (pct_ai_ads_cumulative)'),
        ('firm_ai_exposure',      'Task-based exposure (firm_ai_exposure)'),
        ('log_ai_apps',           'Log total AI apps'),
    ]

    rows = []
    for outcome, out_label in [
        ('log_job_ads_lead1', 'Log job ads at t+1'),
        ('delta_log_job_ads', 'Δlog job ads'),
    ]:
        logger.info(f"\n  Outcome: {out_label}")
        logger.info(f"  {'Treatment':<50} {'N':>6}  {'beta':>9}  {'SE':>7}  {'p':>6}")
        logger.info(f"  {'-'*80}")
        for treat, treat_label in treat_pairs:
            res, s = fit_firm_ols(df, outcome, [treat], [])
            if res is None:
                logger.info(f"  {treat_label:<50}  N too small")
                continue
            b  = res.params.get(treat, np.nan)
            se = res.bse.get(treat, np.nan)
            p  = res.pvalues.get(treat, np.nan)
            sig = '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else '   '))
            logger.info(f"  {treat_label:<50} {len(s):>6,}  {b:>+9.4f}  {se:>7.4f}  {p:>6.3f} {sig}")
            rows.append({
                'module':    'E',
                'outcome':   outcome,
                'spec':      out_label,
                'term':      treat,
                'estimate':  b,
                'std_error': se,
                'p_value':   p,
                'n_obs':     len(s),
                'n_firms':   s['company_id'].nunique(),
            })

    return rows


# ── Module F: Robustness — sample selection vs. control effect ─────────────────

def module_f(df):
    """
    Decomposes the gap between the identification robustness memo (redistribution
    p=0.116, N=2,147) and Module A results (redistribution p=0.017, N=1,510).

    Three specs for redistribution, welfare, and nativism:

      F1 — CONTROLS_NO_FIRM_LAG, full available rotating-battery sample
           Replicates the identification robustness memo spec as closely as possible.
           N ≈ 2,100.  If significant here, the memo's p=0.116 was a different run.

      F2 — CONTROLS (full with firm lags), full available rotating-battery sample
           This is what Module A already reports.  N ≈ 1,510.

      F3 — CONTROLS_NO_FIRM_LAG, restricted to the N=1,510 Module A sample
           Same observations as F2, but drops firm lag controls.
           Isolates whether significance in F2 comes from:
             (a) the firm lags absorbing confounders → β stable, SE smaller
             (b) sample selection (who has non-missing firm lags) → β changes
           If β(F2) ≈ β(F3): significance is from controls, not selection.
           If β(F2) ≠ β(F3): significance is from sample selection.
    """
    logger.info("\n" + "=" * 70)
    logger.info("MODULE F: ROBUSTNESS — SAMPLE SELECTION vs. CONTROL EFFECT")
    logger.info("  Rotating battery: redistribution, welfare, nativism")
    logger.info("  F1: no firm lags, full sample  (~N=2,100)")
    logger.info("  F2: full controls, full sample  (~N=1,510)  [= Module A]")
    logger.info("  F3: no firm lags, restricted to F2 sample  (~N=1,510)")
    logger.info("=" * 70)

    matched = df[df['matched']].copy()

    # Flag observations with non-missing log_job_ads_lag1 (the only remaining lagged control)
    has_firm_lags = matched['log_job_ads_lag1'].notna()
    matched['has_firm_lags'] = has_firm_lags

    outcomes_f = [
        ('redistributive', 'Redistribution Support (pp17, 1-3)'),
        ('welfare',        'Social Spending Support (pp13, 1-3)'),
        ('nativism',       'Nativism / Opp. Foreigners (pp15, 1-3)'),
    ]

    specs = [
        ('F1_no_firm_lag_full',       matched,                         CONTROLS_NO_FIRM_LAG, 'F1: no firm lags, full sample'),
        ('F2_full_controls_full',     matched,                         CONTROLS,             'F2: full controls, full sample'),
        ('F3_no_firm_lag_restricted', matched[matched['has_firm_lags']], CONTROLS_NO_FIRM_LAG, 'F3: no firm lags, restricted to F2 sample'),
    ]

    rows = []
    for outcome, out_label in outcomes_f:
        logger.info(f"\n  Outcome: {out_label}")
        logger.info(f"  {'Spec':<45} {'N':>7}  {'beta_AI':>9}  {'SE':>7}  {'p':>6}")
        logger.info(f"  {'-'*75}")
        for spec_key, sub, ctrl, spec_label in specs:
            res, s = fit_ols(sub, outcome, [EXPOSURE] + ctrl)
            log_result(spec_label, res, s)
            if res is not None:
                r = make_row(res, s, outcome, spec_key, EXPOSURE, 'F')
                if r:
                    r['spec_label'] = spec_label
                    rows.append(r)

    logger.info("\n  Interpretation guide:")
    logger.info("  β(F1) ≈ β(F2) ≈ β(F3) → controls and sample both irrelevant; robust result")
    logger.info("  β(F2) ≈ β(F3) but β(F1) ≠ β(F2) → sample selection drives the difference")
    logger.info("  β(F2) ≠ β(F3) → firm lag controls change estimate within same sample (omitted variable)")

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 70)
    logger.info("STAGE 9E: POLITICAL OUTCOMES, NON-LINEARITIES, INTERACTIONS,")
    logger.info("          HETEROGENEITY, AND FIRM HIRING")
    logger.info("=" * 70)

    panel, shp, firm = load_data()
    df = build_analysis_frame(panel, shp, firm)

    all_rows    = []
    all_rows   += module_a(df)
    all_rows   += module_b(df)
    all_rows   += module_c(df)
    all_rows   += module_d(df)
    all_rows   += module_f(df)
    firm_rows   = module_e(firm)

    if not all_rows:
        raise RuntimeError("Modules A–D produced no output; check logs.")

    results = pd.DataFrame(all_rows)
    results.to_csv(out_person, index=False)
    logger.info(f"\n✓ Saved person-level results: {out_person} ({len(results)} rows)")

    if firm_rows:
        pd.DataFrame(firm_rows).to_csv(out_firm, index=False)
        logger.info(f"✓ Saved firm hiring results: {out_firm} ({len(firm_rows)} rows)")

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 9E COMPLETE")
    logger.info("=" * 70)
    return 0


if __name__ == '__main__':
    sys.exit(main())
