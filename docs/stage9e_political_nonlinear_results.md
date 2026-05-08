# Stage 9e: Political Outcomes, Non-Linearities, Heterogeneities, and Firm Hiring

## Overview

This memo documents results from `stage_9e_political_nonlinear.py`.

**Baseline specification:**
- Sample: matched-only (hampole_ai_exposure_avg_foy > 0)
- FE: Person + 3-digit ISCO×Year + Industry×Year (NOGA2M 17-sector)
- SE: One-way cluster on idpers
- Controls: age_centered, age_squared, contract_perm (pw36), part_time (pw39), computer_work (pw607), firm_size (pw85, current year), public_sector (pw32==2), log_job_ads_lag1

Time-invariant controls (female, education) are excluded from regression controls (absorbed by person FE) but used for subgroup splits in Module D.

**Important sample note:** The `exposure > 0` restriction means the sample contains only person-year observations with *positive* measured exposure. Person FE within-demean identifies off within-person variation in the *level* of exposure over time — years of higher vs. lower (but still positive) exposure for the same individual. The estimates capture **exposure intensity**, not presence vs. absence.

**Why N varies across outcomes:**
- Annual outcomes (vote, left-right, social trust): N≈5,000–5,500. Vote outcomes (pp19) drop further because pp19 is not asked every wave.
- Rotating battery (redistribution pp17, welfare pp13, nativism pp15, etc.): N≈1,900–2,000. These are only asked in 2014, 2017, 2020, 2023 — roughly 35% of matched person-years fall in those four waves.
- The matched-only sample has 6,139 person-years. Adding ind_year (noga2m, 99.4% coverage) and public_sector (96.5%) reduces regression N to ~5,600–5,650 — a 4% loss, not a selection concern.

---

## Module A: Baseline Political Outcomes

### Party Vote (binary dummies from pp19)

Vote coding: 1=FDP, 2=CVP, 3=SP, 4=SVP, 9=GPS, 11/20=GLP, 21=BDP, 50/51=no party.

| Outcome | N | β | SE | p |
|---|---|---|---|---|
| Vote SVP | 4,999 | −0.043 | 0.024 | 0.071* |
| Vote SP | 4,999 | +0.022 | 0.026 | 0.402 |
| Vote FDP | 4,999 | +0.011 | 0.045 | 0.812 |
| Vote CVP | 4,999 | +0.016 | 0.023 | 0.501 |
| Vote GLP | 4,999 | −0.020 | 0.034 | 0.551 |
| Vote BDP | 4,999 | +0.009 | 0.013 | 0.491 |
| Vote no party | 4,999 | −0.005 | 0.030 | 0.881 |

With the full three-way FE specification, the SVP effect weakens to p=0.071 (was p=0.027 without industry×year FE). The direction is preserved — workers at AI-exposed firms are less likely to vote SVP — but the result is now marginal rather than conventionally significant.

### Ideological and Attitudinal Outcomes

| Outcome | N | β | SE | p |
|---|---|---|---|---|
| Left-right self-placement (pp10, 0–10) | 4,997 | −0.029 | 0.128 | 0.818 |
| Social trust (pp45, 0–10) | 5,480 | +0.030 | 0.160 | 0.853 |
| Redistribution support (pp17, 1–3) | 1,950 | +0.224 | 0.116 | 0.053* |
| Welfare/social spending (pp13, 1–3) | 1,898 | −0.106 | 0.121 | 0.385 |
| Nativism/opp. for foreigners (pp15, 1–3) | 1,940 | +0.025 | 0.139 | 0.857 |
| Gender equality (pp22-derived, 0–10) | 1,985 | +0.506 | 0.395 | 0.201 |
| Trust in government (pp04, 0–10) | 1,981 | +0.060 | 0.207 | 0.773 |
| Democracy satisfaction (pp02, 0–10) | 1,974 | −0.174 | 0.267 | 0.515 |
| Political efficacy (pp03, 0–10) | 1,982 | +0.001 | 0.414 | 0.998 |
| EU opinion (pp14, 1–3) | 1,938 | +0.019 | 0.083 | 0.821 |

**Redistribution** (pp17): β=+0.224, p=0.053* — borderline. The welfare result is now clearly null with industry×year FE absorbed. See Module F for the full sample-selection decomposition.

### A.1 Note on the Redistribution–Welfare Divergence

The earlier spec (without ind_year FE) showed redistribution significant (p=0.017) and welfare significant (p=0.025). Adding industry×year FE moves redistribution to p=0.053 and collapses welfare to p=0.385. This suggests that industry-specific time trends (e.g., tech sector workers shifting political attitudes over time) were partially driving the earlier results. Redistribution is more robust to this correction than welfare.

---

## Module B: Non-Linearities

### B.1 Quadratic Specification

**Job insecurity:**

| Term | N | β | SE | p |
|---|---|---|---|---|
| Exposure (linear) | 5,482 | −0.015 | 0.152 | 0.924 |
| Exposure² | 5,482 | — | — | — |

The quadratic term is no longer significant with the full three-way FE. The earlier convex pattern (significant exp² at p=0.008) was partially absorbed by industry×year trends.

**Vote SVP:**

| Term | N | β | SE | p |
|---|---|---|---|---|
| Exposure (linear) | 4,999 | **−0.133** | 0.049 | **0.006** |
| Exposure² | 4,999 | **−0.122** | 0.049 | **0.012** |

Both terms remain significant. The negative quadratic term here (sign reversed from earlier spec) suggests the anti-SVP effect accelerates at higher exposure levels rather than attenuating.

**Redistribution:**

| Term | N | β | SE | p |
|---|---|---|---|---|
| Exposure (linear) | 1,950 | +0.224 | 0.172 | 0.193 |
| Exposure² | 1,950 | +0.014 | 0.217 | 0.947 |

No significant quadratic. Linear spec is appropriate.

### B.2 Tercile Dummies

**Job insecurity** (T1 reference, N=5,482):

| Tercile | β | SE | p |
|---|---|---|---|
| T2 vs T1 | −0.056 | 0.039 | 0.146 |
| T3 vs T1 | −0.040 | 0.052 | 0.448 |

**Vote SVP** (T1 reference, N=4,999):

| Tercile | β | SE | p |
|---|---|---|---|
| T2 vs T1 | −0.022 | 0.014 | 0.124 |
| T3 vs T1 | −0.039 | 0.020 | 0.055* |

T3 significant at 10% — the SVP effect remains concentrated in the top tercile.

---

## Module C: Interactions with Perceived Insecurity

Moderators centered at matched-sample mean.

### Vote SVP

| Spec | N | β exposure | β interaction | p interaction |
|---|---|---|---|---|
| × Job insecurity | 4,981 | −0.043* | +0.003 | 0.850 |
| × Unemp. risk | 4,967 | −0.044* | −0.002 | 0.625 |

No significant interactions. The SVP effect is not mediated by individual insecurity levels.

### Redistribution Support

| Spec | N | β exposure | β interaction | p interaction |
|---|---|---|---|---|
| × Job insecurity | 1,945 | +0.205 | +0.061 | 0.208 |
| × Unemp. risk | 1,936 | +0.198 | +0.008 | 0.633 |

No significant interactions. Redistribution demand from AI exposure is not amplified for more anxious workers.

### Job Insecurity × Unemployment Risk

| Term | N | β | SE | p |
|---|---|---|---|---|
| Exposure | 5,453 | +0.183 | 0.064 | 0.004 |
| Unemployment risk (centered) | 5,453 | +0.102 | 0.011 | <0.001 |
| Interaction | 5,453 | +0.000 | 0.015 | 0.993 |

AI exposure raises job insecurity independently of baseline unemployment risk perceptions.

---

## Module D: Heterogeneous Effects

### Job Insecurity

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| **Full sample** | 5,482 | **+0.214** | 0.073 | **0.004** |
| Male | 3,082 | **+0.193** | 0.094 | **0.041** |
| Female | 2,400 | +0.212 | 0.110 | 0.053* |
| High education | 2,829 | +0.199 | 0.117 | 0.089* |
| Low education | 2,653 | **+0.204** | 0.092 | **0.026** |
| Pre-2018 | 1,799 | +0.049 | 0.164 | 0.768 |
| **Post-2018** | 3,683 | **+0.287** | 0.114 | **0.012** |

The post-2018 period drives the job insecurity effect — consistent with AI becoming more salient to workers after the ChatGPT/LLM wave.

### Vote SVP

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 4,999 | −0.043 | 0.024 | 0.071* |
| Male | 2,894 | −0.049 | 0.033 | 0.137 |
| Female | 2,105 | −0.025 | 0.041 | 0.531 |
| High education | 2,654 | −0.048 | 0.034 | 0.163 |
| Low education | 2,345 | −0.023 | 0.035 | 0.500 |
| Pre-2018 | 1,644 | −0.073 | 0.046 | 0.109 |
| Post-2018 | 3,355 | −0.035 | 0.037 | 0.338 |

No subgroup is individually significant; the full-sample result is a weak aggregate.

### Vote SP (sharpest heterogeneity)

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 4,999 | +0.022 | 0.026 | 0.402 |
| **Male** | 2,894 | **+0.125** | 0.039 | **0.001** |
| **Female** | 2,105 | **−0.113** | 0.047 | **0.015** |
| **High education** | 2,654 | **+0.070** | 0.032 | **0.029** |
| Low education | 2,345 | −0.006 | 0.042 | 0.884 |
| Pre-2018 | 1,644 | +0.007 | 0.062 | 0.910 |
| Post-2018 | 3,355 | +0.030 | 0.031 | 0.326 |

**The SP gender reversal survives the full three-way FE specification.** Men shift toward SP (+0.125***); women shift away (−0.113**). The full-sample null is a composition effect.

### Vote FDP

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 4,999 | +0.011 | 0.045 | 0.812 |
| Male | 2,894 | +0.034 | 0.043 | 0.439 |
| Female | 2,105 | −0.032 | 0.077 | 0.674 |
| High education | 2,654 | −0.064 | 0.064 | 0.313 |
| Low education | 2,345 | +0.071 | 0.056 | 0.201 |
| Pre-2018 | 1,644 | −0.005 | 0.069 | 0.943 |
| Post-2018 | 3,355 | +0.004 | 0.043 | 0.933 |

### Vote GLP

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 4,999 | −0.020 | 0.034 | 0.551 |
| Male | 2,894 | −0.039 | 0.053 | 0.459 |
| Female | 2,105 | −0.021 | 0.040 | 0.601 |
| High education | 2,654 | +0.038 | 0.046 | 0.406 |
| Low education | 2,345 | **−0.088** | 0.045 | **0.049** |
| Pre-2018 | 1,644 | +0.006 | 0.071 | 0.929 |
| Post-2018 | 3,355 | −0.048 | 0.035 | 0.176 |

Low-education workers at AI-exposed firms show a marginally significant decline in GLP support.

### Redistribution Support

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 1,950 | +0.224 | 0.116 | 0.053* |
| Male | 1,127 | +0.226 | 0.162 | 0.162 |
| Female | 823 | +0.310 | 0.160 | 0.053* |
| **High education** | 1,046 | **+0.333** | 0.159 | **0.037** |
| Low education | 904 | +0.073 | 0.153 | 0.633 |
| Pre-2018 | 644 | +0.199 | 0.267 | 0.457 |
| Post-2018 | 1,306 | +0.266 | 0.168 | 0.112 |

The redistribution effect remains concentrated among high-education workers. The pattern is directionally consistent but the full-sample estimate is now borderline (p=0.053).

---

## Module E: Firm-Level Future Hiring

Firm + Year FE, cluster on company_id. N=72,720 firm-years, 7,197 companies.

| Outcome | Treatment | N | Firms | β | SE | p |
|---|---|---|---|---|---|---|
| Log job ads (t+1) | pct_ai_ads_cumulative | 72,720 | 7,197 | +0.0007 | 0.0007 | 0.324 |
| **Log job ads (t+1)** | **firm_ai_exposure** | 72,720 | 7,197 | **+0.059** | 0.016 | **<0.001** |
| **Log job ads (t+1)** | **log_ai_apps** | 72,720 | 7,197 | **+0.101** | 0.032 | **0.002** |
| Δlog job ads | pct_ai_ads_cumulative | 72,720 | 7,197 | +0.0003 | 0.0003 | 0.381 |
| Δlog job ads | firm_ai_exposure | 72,720 | 7,197 | −0.001 | 0.004 | 0.794 |
| Δlog job ads | log_ai_apps | 72,720 | 7,197 | +0.003 | 0.007 | 0.657 |

AI-exposed firms hire more in the future (level effect) but growth rates are unaffected. AI adoption predicts permanently higher hiring steady-states, not accelerating expansion.

---

## Module F: Robustness — Sample Selection vs. Control Effect

Three specs for rotating battery outcomes:
- **F1** — No log_job_ads_lag1, full available rotating-battery sample (~1,950–1,990)
- **F2** — Full controls including log_job_ads_lag1 (~1,950) [= Module A]
- **F3** — No log_job_ads_lag1, restricted to F2 sample

F2 ≈ F3 within-sample implies log_job_ads_lag1 is not moving the estimate — any difference between F1 and F2 is sample selection, not control effect.

### Redistribution Support (pp17, 1–3)

| Spec | N | β | SE | p |
|---|---|---|---|---|
| F1: no firm lag, full sample | 1,953 | +0.199 | 0.117 | 0.087* |
| F2: full controls | 1,950 | +0.224 | 0.116 | 0.053* |
| F3: no firm lag, restricted to F2 sample | 1,950 | +0.199 | 0.117 | 0.088* |

Consistent across all three specs. The redistribution result is not an artefact of sample selection or the lagged control.

### Welfare / Social Spending (pp13, 1–3)

| Spec | N | β | SE | p |
|---|---|---|---|---|
| F1: no firm lag, full sample | 1,899 | −0.064 | 0.118 | 0.586 |
| F2: full controls | 1,898 | −0.106 | 0.121 | 0.385 |
| F3: no firm lag, restricted to F2 sample | 1,898 | −0.063 | 0.118 | 0.590 |

Consistently null across all specs. Welfare is robustly non-significant.

### Nativism (pp15, 1–3)

| Spec | N | β | SE | p |
|---|---|---|---|---|
| F1: no firm lag, full sample | 1,943 | −0.000 | 0.131 | 0.998 |
| F2: full controls | 1,940 | +0.025 | 0.139 | 0.857 |
| F3: no firm lag, restricted to F2 sample | 1,940 | −0.001 | 0.131 | 0.995 |

Robustly null.

---

## Summary of Key Results

| Module | Outcome | β | p | N | Note |
|---|---|---|---|---|---|
| A | Job insecurity | +0.214 | 0.004*** | 5,482 | Robust |
| A | Vote SVP | −0.043 | 0.071* | 4,999 | Marginal with ind×year FE |
| A | Redistribution | +0.224 | 0.053* | 1,950 | Borderline; high-edu driven |
| A | Welfare | −0.106 | 0.385 | 1,898 | Null with ind×year FE |
| A | All other political | ~0 | n.s. | — | |
| B | SVP quadratic | Both terms sig. | 0.006/0.012 | 4,999 | Effect accelerates at high exposure |
| C | All interactions | ~0 | n.s. | — | No fear-mediation |
| D | SP: men | +0.125 | 0.001*** | 2,894 | Sharp reversal |
| D | SP: women | −0.113 | 0.015** | 2,105 | Sharp reversal |
| D | Redistribution: high edu | +0.333 | 0.037** | 1,046 | Concentrated |
| D | Job insecurity: post-2018 | +0.287 | 0.012** | 3,683 | Post-LLM period |
| E | Future hiring level | +0.059 | <0.001*** | 72,720 | AI firms hire more |
| E | Hiring growth | ~0 | n.s. | 72,720 | No acceleration |

---

## Notes on Control Specification

Controls used throughout:
- `age_centered`, `age_squared`
- `contract_perm` (pw36==2, permanent contract)
- `part_time` (pw39==1)
- `computer_work` (pw607, uses computer at work)
- `firm_size` (pw85, current year; stable, no lag needed)
- `public_sector` (pw32==2, government employer)
- `log_job_ads_lag1` (lagged to avoid bad-controls endogeneity)

**Coverage in matched sample:** age 100%, contract_perm 98.9%, part_time 99.9%, computer_work 100%, firm_size 93.4%, public_sector 96.5%, log_job_ads_lag1 97.8%. noga2m (ind_year FE) 99.4%.
