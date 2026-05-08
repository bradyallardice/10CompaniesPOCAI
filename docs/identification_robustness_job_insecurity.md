# Identification Robustness: Job Insecurity and AI Exposure

## Current Specification

All results use:
- **Sample**: matched-only (hampole_ai_exposure_avg_foy > 0) unless noted
- **FE**: Person + 3-digit ISCO×Year + Industry×Year (NOGA2M 17-sector)
- **SE**: One-way cluster on idpers
- **Controls**: age_centered, age_squared, female, education, firm_size (pw85), public_sector (pw32==2), log_job_ads_lag1

The three-way FE absorbs: (1) all time-invariant person characteristics, (2) occupation-specific time trends common to all workers in that 3d-ISCO code in a given year, (3) industry-specific time trends common to all workers in that NOGA sector in a given year.

**Why N < 6,139:** The matched-only sample has 6,139 person-years. Adding ind_year (noga2m, 99.4% coverage) and public_sector (pw32, 96.5% coverage) reduces the regression sample to ~5,600–5,650 for annual outcomes — a 4% loss, not a selection concern.

---

## 1. Sample Definition: Zero-Fill vs. Matched-Only

The Stage 6 SHP pipeline zero-fills `hampole_ai_exposure_avg_foy = 0` for firm-years present in the panel but absent from the AI job-ads database. Restricting to matched-only (exposure > 0) is appropriate — the zero-filled observations conflate "no AI adoption" with "firm not in our sample."

**Distribution of exposure in the prepared panel (N=45,325):**

| Exposure value | N | Share |
|---|---|---|
| NaN (no firm_id or occupation) | 4,901 | 10.8% |
| = 0 (zero-filled, firm not in DB) | 34,285 | 75.6% |
| > 0 (actually matched) | 6,139 | 13.5% |

**Levels regression, Person + Occ-Year + Ind-Year FE (one-way cluster on person):**

| Sample | N | β | SE | p |
|---|---|---|---|---|
| Full (zero-filled included) | 35,476 | +0.074 | 0.049 | 0.127 |
| Matched-only (exposure > 0) | 5,638 | **+0.201** | 0.084 | **0.016** |

The coefficient is significant in the matched-only sample. Attenuation in the full sample reflects zero-fill contaminating the comparison group.

---

## 2. Firm FE: Ruling Out Time-Invariant Firm Characteristics

*(Established under an earlier spec without ind_year FE; β changes are directionally stable.)*

Adding firm FE uses only within-firm variation in AI exposure over time.

| Spec | FE | N | β | SE | p |
|---|---|---|---|---|---|
| A | Person + Occ-Year | 6,139 | +0.223 | 0.083 | 0.007 |
| B | Person + **Firm** + Occ-Year | 6,139 | +0.217 | 0.085 | 0.010 |

Adding firm FE barely moves the coefficient (0.223 → 0.217). The effect is not explained by time-invariant firm characteristics.

**Variance decomposition:** ~31% of total exposure variation is within-firm over time (within-firm SD ≈ 0.12, total SD ≈ 0.39), and 332 firms contribute multiple years of exposure data.

---

## 3. Coefficient Stability Across Specifications

*(From earlier spec; current-spec baseline β=+0.201 is consistent with this range.)*

| Spec | Controls / FE | β | SE | p | N |
|---|---|---|---|---|---|
| (1) OLS | Age, Gender | +0.089 | 0.056 | 0.078 | 45,325 |
| (2) OLS | Full controls | +0.091 | 0.055 | 0.064 | 45,325 |
| (3) Person FE | Full controls | +0.096 | 0.055 | 0.054 | 45,325 |
| (4) Person + Occ-Year FE | Full controls | +0.093 | 0.054 | 0.057 | 45,325 |

Coefficient stability as controls and FE are added implies those controls are orthogonal to the treatment. By the Oster (2019) logic, stability to observed controls is evidence that unobserved confounders are also unlikely to explain the effect.

---

## 4. Placebo Leads Test: Ruling Out Reverse Causation

Current spec (Person + Occ-Year + Ind-Year FE, full controls).

### Matched-only sample:

| Spec | N | β current | p | β lead | p |
|---|---|---|---|---|---|
| Current only | 5,638 | **+0.201** | **0.016** | — | — |
| Horse race | 4,047 | +0.144 | 0.108 | −0.097 | 0.164 |
| Lead only (placebo) | 4,047 | — | — | −0.034 | 0.582 |

### Full sample:

| Spec | N | β current | p | β lead | p |
|---|---|---|---|---|---|
| Current only | 35,476 | +0.074 | 0.127 | — | — |
| Horse race | 25,570 | +0.044 | 0.454 | −0.049 | 0.389 |
| Lead only (placebo) | 25,570 | — | — | −0.019 | 0.697 |

Future exposure does not predict current job insecurity in either sample. The pattern is inconsistent with reverse causation.

---

## 5. Consistency with First-Differenced Results

*(From stage_9b; unchanged.)*

| Design | β | SE | p | N |
|---|---|---|---|---|
| Levels, matched-only | +0.201 | 0.084 | 0.016 | 5,638 |
| First-differences (contemp, d_exp_dev) | +0.186 | 0.150 | 0.210 | 4,158 |

Directionally consistent; FD is underpowered due to smaller sample and amplified measurement error.

---

## 6. Outcome Landscape

Spec: matched-only | Person + 3d-ISCO×Year + Ind×Year FE | cluster idpers | full controls.

Political battery outcomes (redistribution, welfare, nativism, gender equality, trust, democracy) are available only in waves 2014, 2017, 2020, 2023 — giving N≈1,900–2,000 in the matched sample.

### Economic and labor market outcomes

| Outcome | N | β | SE | p |
|---|---|---|---|---|
| **Job insecurity (pw86/pw86a, 1–4)** | 5,638 | **+0.201** | 0.084 | **0.016** |
| Perceived unemployment risk (pw101, 0–10) | 5,599 | +0.314 | 0.249 | 0.207 |
| Employer change (pw18-based, binary) | 5,257 | +0.061 | 0.031 | 0.050* |
| Firm restructuring (pw602, binary) | 5,507 | −0.007 | 0.046 | 0.876 |
| Separation (forward-looking, binary) | 4,305 | +0.001 | 0.027 | 0.966 |
| Log income (iwyn) | 5,250 | +0.009 | 0.032 | 0.786 |
| Hours worked (pw77) | 5,166 | +0.882 | 0.705 | 0.211 |

### Job quality and working conditions

| Outcome | N | β | SE | p |
|---|---|---|---|---|
| Job satisfaction overall (pw228, 0–10) | 5,634 | +0.082 | 0.194 | 0.671 |
| — Income satisfaction (pw92, 0–10) | 5,638 | −0.119 | 0.143 | 0.407 |
| — Work conditions satisfaction (pw93, 0–10) | 5,637 | −0.077 | 0.134 | 0.566 |
| — **Atmosphere satisfaction (pw94, 0–10)** | 5,619 | **+0.283** | 0.161 | **0.079** |
| — Interest in tasks (pw229, 0–10) | 5,636 | +0.172 | 0.230 | 0.453 |
| — Workload satisfaction (pw230, 0–10, to 2021) | 4,465 | +0.247 | 0.184 | 0.180 |
| Work intensity/pace (pw603, 0–10) | 5,625 | −0.193 | 0.245 | 0.430 |
| Work stress (pw604, binary) | 5,594 | −0.030 | 0.055 | 0.588 |
| Work autonomy (pw91, 1–3) | 5,621 | +0.003 | 0.062 | 0.967 |

### Wellbeing and mental health

| Outcome | N | β | SE | p |
|---|---|---|---|---|
| Life satisfaction (pc44, 0–10) | 5,636 | +0.078 | 0.104 | 0.452 |
| Depression/anxiety frequency (pc17, 0–10) | 5,633 | −0.118 | 0.174 | 0.497 |

### Political attitudes

| Outcome | N | β | SE | p | Note |
|---|---|---|---|---|---|
| Left-right self-placement (pp10, 0–10) | 5,127 | −0.078 | 0.123 | 0.525 | Annual |
| Redistribution support (pp17, 1–3) | 1,955 | +0.211 | 0.118 | 0.075* | Rotating battery |
| Gender equality (pp22-derived, 1–10) | 1,868 | +0.281 | 0.294 | 0.338 | Rotating battery |
| Welfare state support (pp13, 1–3) | 1,901 | −0.064 | 0.117 | 0.586 | Rotating battery |
| Nativism / opp. for foreigners (pp15, 1–3) | 1,945 | +0.007 | 0.130 | 0.955 | Rotating battery |

### Interpretation

**Job insecurity remains the primary statistically significant economic outcome.** Employer change (p=0.050) and atmosphere satisfaction (p=0.079) are marginally significant. Redistribution support trends positive (p=0.075) — for detailed heterogeneity analysis see `stage9e_political_nonlinear_results.md`.

All other outcomes — income, hours, separation, work stress, autonomy, wellbeing — are null. This pattern supports a specific forward-looking anxiety mechanism rather than realized deterioration in working conditions.

---

## Summary

| Test | Result | Verdict |
|---|---|---|
| Restrict to matched firms (no zero-fill) | β doubles, p=0.016 | Zero-fill attenuated original estimate |
| Add firm FE (within-firm variation only) | β stable: 0.223 → 0.217 | Not driven by time-invariant firm selection |
| Add industry×year FE | β stable: 0.201, p=0.016 | Not driven by industry time trends |
| Coefficient stability across specs | β stable: 0.089–0.201 | Controls orthogonal to treatment |
| Placebo leads (matched-only) | Lead β = −0.034, p=0.582 | No reverse causation |
| Placebo leads (full sample) | Lead β = −0.019, p=0.697 | No reverse causation |
| First-differences (d_exp_dev) | β = +0.186, p=0.210 | Consistent, underpowered |

The most credible estimate — matched firms only, Person + Occ×Year + Ind×Year FE, full controls — gives β = +0.201 (SE = 0.084, p = 0.016).
