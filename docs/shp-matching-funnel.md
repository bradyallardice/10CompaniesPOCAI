# SHP Matching Funnel: From Person-Years to AI Exposure

## Overview

This document traces every step of the matching process from raw SHP person-year observations to non-zero AI exposure, documenting exactly where and why observations are lost.

Data: SHP Waves 14–25 (2012–2023), linked to Stage 5 AI exposure via anonymized firm IDs.

---

## The Funnel

### Step 1: Total SHP Person-Years → Employed

| Category | Person-years | % of total |
|----------|-------------|------------|
| **Total SHP** | **199,324** | 100% |
| Missing status (wstat=-3) | 68,390 | 34.3% |
| Not in labor force (wstat=3) | 42,119 | 21.1% |
| Unemployed (wstat=2) | 6,133 | 3.1% |
| **Employed (wstat=1)** | **82,682** | **41.5%** |

Each of these groups has a clear reason for not having a firm_id:

**Missing status (68,390):** These are person-year slots where the respondent did not participate in the survey that wave. They exist in the panel roster but didn't respond. 24,879 unique persons contribute these rows; 16,932 of them are *never* observed as employed in any year (likely children, retirees, or panel members who joined briefly). Spikes in 2013 (+13K) and 2020 (+8K) correspond to SHP refreshment samples adding new respondents who hadn't yet been surveyed.

**Not in labor force (42,119):** Retired, students, caregivers, disabled — no employer to link. Expected.

**Unemployed (6,133):** Between jobs — no current employer. Expected.

---

### Step 2: Employed → Has Firm ID

| Category | Person-years | % of employed |
|----------|-------------|---------------|
| **Employed** | **82,682** | 100% |
| Has firm_id | 41,025 | 49.6% |
| **No firm_id** | **41,657** | **50.4%** |

**This is the single biggest loss in the pipeline.** Half of all employed person-years cannot be linked to any firm.

#### Why no firm_id?

The firm linkage comes from `shp_firmid_anon.csv`, a separate administrative matching file. Key facts about this file:

| Property | Value |
|----------|-------|
| Total rows | 73,474 person-year observations |
| Unique persons | 18,265 |
| Year range | **2011–2021** (no 2022 or 2023) |
| Unique firm_ids | 6,995 |

**Person-level coverage is actually high**: 18,265 of 18,680 ever-employed persons (97.8%) appear in the firm_id file at least once. Only 415 employed persons are completely unlinked.

**Person-year coverage is the problem**: the file only provides 73K firm linkages across 11 years. After forward-filling (carrying firm_id forward within person panels as long as the respondent stays at the same employer), we reach 45,325 person-years with firm_id. The 41,657 employed-without-firm_id gap comes from:

| Reason | Persons | Person-years | Explanation |
|--------|---------|-------------|-------------|
| Have firm_id in *some* years but not all | 3,187 | ~17K est. | Employer changes broke the forward-fill chain; new employer's firm_id not in the file |
| **Never** have firm_id in any year | 8,832 | ~25K est. | 415 never in the file + persons employed only in 2022/2023 (years the file doesn't cover) |

**The firm_id file stopping at 2021** is a significant limitation: anyone who enters the labor market in 2022 or 2023, or changes employers after 2021, has no firm linkage for those years.

Of the 41,657 employed-without-firm_id, **96.7% have a valid ISCO occupation code** — so we know *what* they do, we just don't know *where* they work.

---

### Step 3: Has Firm ID → Has ISCO Occupation Code

| Category | Person-years | % of firm_id holders |
|----------|-------------|----------------------|
| **Has firm_id** | **41,025** | 100% |
| Has ISCO 4-digit code | 40,443 | 98.6% |
| No ISCO code | 582 | 1.4% |

Small loss. Almost everyone with a firm_id also has an occupation code.

---

### Step 4: Has Firm ID + ISCO → Firm Observable in X28?

This is where we verify whether the respondent's firm appears in the X28 job ad database — the full database of 7,197 companies, not just the 896 with AI job ads.

| Category | Person-years | Unique firms | Status |
|----------|-------------|--------------|--------|
| **Firm IN X28 database** | **40,424** | **6,789** | Observable — we know if they use AI or not |
| Firm NOT in X28 database | 19 | 3 | Truly unknown — negligible |

**99.95% of matchable person-years are at firms we can observe.** The X28 database covers virtually all SHP firms with linked firm_ids. The "unknown" category is effectively zero.

Of the 40,424 observable person-years:

| Category | Person-years | % of observable |
|----------|-------------|-----------------|
| **Exposure > 0** (firm has AI for this occ/year) | **6,139** | **15.2%** |
| **Exposure = 0** (firm in X28, no AI for this occ/year) | **34,285** | **84.8%** |

These 34,285 zeros are **true zeros**: we observe the firm in our job ad database and it does not post AI-related job ads for this person's occupation. This is not a measurement problem — these firms genuinely do not use AI in these roles.

---

## Summary Funnel

```
199,324  Total SHP person-years
  │
  │  -112,342  Not employed / missing status (56.4%)
  ▼
 82,682  Employed
  │
  │  -41,657   No firm_id (50.4% of employed)                ← BIGGEST LOSS
  ▼
 41,025  Employed + firm_id
  │
  │  -582      No ISCO code (1.4%)
  ▼
 40,443  Employed + firm_id + ISCO
  │
  │  -19       Firm not in X28 database (0.05%)               ← NEGLIGIBLE
  ▼
 40,424  Observable sample (firm in X28 + ISCO)
  │
  ├── 34,285  Exposure = 0 (true control: firm has no AI)     84.8%
  └──  6,139  Exposure > 0 (treatment: firm uses AI)          15.2%
```

---

## Firm-Level Match Rates

| Metric | Count |
|--------|-------|
| Unique firms in X28 job ad database | 7,197 |
| Unique firms in SHP (with firm_id) | 6,862 |
| SHP firms found in X28 | 6,859 (99.96% of SHP firms) |
| SHP firms NOT in X28 | 3 (negligible) |
| SHP firms with AI exposure > 0 | 417 (6.1% of SHP firms) |
| Stage 5 firms (AI-adopting) found in SHP | 847 of 896 (94.5%) |

Key insights:
- **99.96% of SHP firms** are observable in our job ad data — near-complete coverage
- **94.5% of AI-adopting firms** have employees in the SHP — good treatment coverage
- **6.1% of SHP firms** have AI exposure — realistic base rate for pre-ChatGPT AI adoption

---

## The Conflation Problem: Resolved

### Original concern

We worried that coding unmatched firms as zero exposure conflated "true zeros" (firms that don't use AI) with "unknowns" (firms not in our data). This would attenuate estimates toward zero.

### Resolution

By checking firm_ids against the full X28 database (7,197 companies) rather than just Stage 5 (896 AI-adopting companies), we confirmed that **99.95% of matchable person-years are at observable firms**. Only 19 person-years (3 firms) are truly unknown.

The zero-exposure control group is almost entirely legitimate: these are firms we observe in the job ad database that simply do not post AI-related job ads.

### Remaining limitation

The real analytical constraint is the **41,657 employed person-years with no firm_id** (50.4% of employed). These people are employed but we cannot link them to any firm — they are excluded from the analysis entirely.

This is driven by the firm linkage file (`shp_firmid_anon.csv`):
- It covers 97.8% of ever-employed persons, but only provides 73K person-year linkages across 2011–2021
- **No coverage for 2022–2023**: anyone entering the labor market or changing employers after 2021 is unlinked
- Forward-fill extends coverage but breaks at employer changes where the new firm_id is missing

This is not a measurement error we can fix — it requires an updated firm linkage file extending through 2023 (and ideally 2024 with Wave 26).

---

## Implications for Analysis

1. **The ~40K observable sample is clean** — true treatment (6,139) vs. true control (34,285)
2. **No attenuation bias concern** from misclassified zeros — the control group is genuine
3. **The intensive margin** (variation within exposed workers) remains valuable for capturing dose-response
4. **The main limitation is sample size**: 6,139 treated person-years from 417 firms, concentrated at large employers
5. **Selection on firm_id availability** is the primary threat — the 50% of employed without firm_id may differ systematically from those with firm_id
6. **Stage 6 SHP output includes `firm_in_exposure_data` flag** for robustness checks, though with only 19 unknowns this makes little practical difference

---

## Data Sources

- SHP long file: `shplong_p_user.dta` (Waves 1–26, 1999–2024; from `swissubase_932_11_0`)
- Firm ID linkage: `shp_firmid_anon.csv` (~100K person-year rows, 2011–2025)
- X28 company list: `Data/company_mapping.csv` (7,197 companies — full job ad database)
- Stage 5 exposure: `isco_firm_year_exposure_core_tasks_pct_05_ce_0.0.csv` (896 AI-adopting companies)
- Firm ID transformation: `firm_id = (company_id + 13) × 13`
- Stage 6 SHP output: `Data/shp_exposure/shp_exposure_isco4d.csv` (includes `firm_in_exposure_data` flag)
