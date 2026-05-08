# Stage 9e: Political Outcomes, Non-Linearities, Heterogeneities, and Firm Hiring

## Overview

This memo documents results from `stage_9e_political_nonlinear.py`. The baseline specification throughout is the matched-only sample (hampole_ai_exposure_avg_foy > 0) with Person FE + 3-digit ISCO×Year FE, one-way clustering on idpers. Controls are time-varying pre-determined variables: age (centered + squared), contract type (permanent vs. fixed-term), part-time status, computer use at work, lagged firm size (t-1), and lagged log job ads (t-1). Time-invariant controls (female, education) are dropped as absorbed by person FE. This matches the specification in `identification_robustness_job_insecurity.md`.

**Important sample note:** The `exposure > 0` restriction means the sample contains only person-year observations with *positive* measured exposure. Person FE within-demean then identifies off within-person variation in the *level* of exposure over time — years of higher exposure versus years of lower (but still positive) exposure for the same individual. The comparison group is not "workers at matched firms in years with zero AI adoption"; those observations are excluded. The estimates capture the effect of **exposure intensity**, not presence versus absence of exposure.

**Why N varies across outcomes:**
- Annual outcomes (vote, left-right, social trust): N≈4,150–4,530. Drop to 4,172 for vote outcomes because pp19 is not asked every wave.
- Rotating battery (redistribution pp17, welfare pp13, nativism pp15, etc.): N≈1,450–1,530. These are only asked in 2014, 2017, 2020, 2023 — roughly 35% of matched person-years fall in those four waves.
- The lagged firm controls (firm_size_lag1 at 74.6% coverage) further reduce N relative to specs without them. This sample selection is the subject of Module F.

---

## Module A: Baseline Political Outcomes

Spec: matched-only | Person + 3d-ISCO×Year FE | cluster idpers | full controls (including lagged firm variables).

### Party Vote (binary dummies from pp19)

Vote coding: 1=FDP, 2=CVP, 3=SP, 4=SVP, 9=GPS, 11/20=GLP, 21=BDP, 50/51=no party.

| Outcome | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| **Vote SVP** | 4,172 | 1,249 | **−0.055** | 0.025 | **0.027** |
| Vote SP | 4,172 | 1,249 | +0.015 | 0.025 | 0.540 |
| Vote FDP | 4,172 | 1,249 | −0.011 | 0.049 | 0.818 |
| Vote CVP | 4,172 | 1,249 | +0.002 | 0.027 | 0.941 |
| Vote GLP | 4,172 | 1,249 | +0.002 | 0.029 | 0.938 |
| Vote BDP | 4,172 | 1,249 | +0.006 | 0.016 | 0.725 |
| Vote no party | 4,172 | 1,249 | −0.000 | 0.035 | 0.996 |

**SVP is the only vote outcome with a statistically significant effect.** Workers at AI-exposed firms are less likely to vote SVP (right-populist). The null on all other parties implies the SVP effect is not a general rightward-leftward shift but specific defection from the right-populist option.

### Ideological and Attitudinal Outcomes

| Outcome | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Left-right self-placement (pp10, 0–10) | 4,151 | 1,230 | +0.040 | 0.141 | 0.777 |
| Social trust (pp45, 0–10) | 4,529 | 1,311 | −0.017 | 0.174 | 0.921 |
| **Redistribution support (pp17, 1–3)** | 1,510 | 925 | **+0.288** | 0.121 | **0.017** |
| **Welfare/social spending (pp13, 1–3)** | 1,468 | 900 | **−0.327** | 0.146 | **0.025** |
| Nativism/opp. for foreigners (pp15, 1–3) | 1,493 | 915 | −0.041 | 0.166 | 0.803 |
| Gender equality (pp22-derived, 0–10) | 1,529 | 935 | +0.418 | 0.446 | 0.348 |
| Trust in government (pp04, 0–10) | 1,527 | 931 | +0.013 | 0.243 | 0.956 |
| Democracy satisfaction (pp02, 0–10) | 1,524 | 932 | +0.052 | 0.353 | 0.883 |
| Political efficacy (pp03, 0–10) | 1,527 | 932 | −0.285 | 0.472 | 0.545 |
| EU opinion (pp14, 1–3) | 1,497 | 917 | +0.060 | 0.102 | 0.560 |

**Significant findings:**
- **Redistribution** (pp17): β=+0.288 (p=0.017) — workers at AI-exposed firms want higher taxes on the wealthy. Robustness discussed in Module F.
- **Welfare/social spending** (pp13): β=−0.327 (p=0.025) — the same workers *oppose* increasing current social spending. Robustness discussed in Module F.

### A.1 The Redistribution–Welfare Paradox

pp17 asks about redistribution through taxation of high incomes; pp13 asks about social spending direction. The opposing signs suggest these are not the same underlying preference. One reading: AI-exposed workers — who tend to be at larger, more formal-sector firms — want more progressive taxation but are skeptical of traditional welfare spending programs. This "productivist" welfare state preference is documented in the comparative welfare state literature: formal-sector workers want redistribution through the tax code rather than expansion of benefit programs.

---

## Module B: Non-Linearities

### B.1 Quadratic Specification

We add `exposure²` to the baseline. N is the same as Module A for each outcome.

**Job insecurity:**

| Term | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Exposure (linear) | 4,533 | 1,313 | −0.136 | 0.160 | 0.395 |
| Exposure² | 4,533 | 1,313 | **+0.201** | 0.076 | **0.008** |

The significant quadratic term with insignificant linear term implies a convex (accelerating) relationship. At moderate exposure levels effects are muted; at high exposure, job insecurity rises sharply.

**Vote SVP:**

| Term | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Exposure (linear) | 4,172 | 1,249 | **−0.187** | 0.057 | **0.001** |
| Exposure² | 4,172 | 1,249 | **+0.073** | 0.022 | **0.001** |

Both terms significant. Negative linear + positive quadratic: the anti-SVP effect is strongest at intermediate exposure levels but weakens at very high exposure, consistent with high-exposure workers shifting to abstention rather than any specific party.

**Redistribution:**

| Term | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Exposure (linear) | 1,510 | 925 | +0.278 | 0.264 | 0.291 |
| Exposure² | 1,510 | 925 | +0.006 | 0.115 | 0.961 |

No significant quadratic — the redistribution effect is approximately linear.

### B.2 Tercile Dummies

T1 = lowest tercile (omitted reference). N is the same as Module A for each outcome.

**Job insecurity:**

| Tercile | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| T2 vs T1 | 4,533 | 1,313 | −0.064 | 0.039 | 0.105 |
| T3 vs T1 | 4,533 | 1,313 | −0.043 | 0.056 | 0.448 |

Neither significant — the quadratic result above suggests the effect is concentrated at the very top of the continuous distribution, which terciles do not fully capture.

**Vote SVP:**

| Tercile | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| T2 vs T1 | 4,172 | 1,249 | −0.024 | 0.015 | 0.117 |
| T3 vs T1 | 4,172 | 1,249 | **−0.059** | 0.022 | **0.006** |

The SVP effect is concentrated in the top tercile — only the highest-exposure workers systematically abandon SVP.

---

## Module C: Interaction with Perceived Insecurity (Moderators)

Moderators centered at matched-sample mean. Outcomes: vote_svp, vote_sp, vote_sp_gps, leftright, redistributive, welfare, job_insecurity.

### Vote SVP

| Spec | N | Persons | β exposure | β interaction | p interaction |
|---|---|---|---|---|---|
| × Job insecurity (centered) | 4,158 | 1,247 | −0.057* | +0.012 | 0.441 |
| × Unemp. risk (centered) | 4,148 | 1,243 | −0.057** | +0.000 | 0.935 |

No significant interaction. The SVP effect from exposure is not amplified for workers who feel more job insecure or perceive higher unemployment risk.

### Redistribution Support

| Spec | N | Persons | β exposure | β interaction | p interaction |
|---|---|---|---|---|---|
| × Job insecurity | 1,506 | 922 | +0.329*** | −0.060 | 0.226 |
| × Unemp. risk | 1,496 | 914 | +0.302** | −0.005 | 0.762 |

No significant interaction. Redistribution demand from AI exposure does not depend on individual-level fear perceptions.

### Welfare Support

| Spec | N | Persons | β exposure | β interaction | p interaction |
|---|---|---|---|---|---|
| × Job insecurity | 1,466 | 899 | −0.357** | +0.023 | 0.734 |
| × Unemp. risk | 1,458 | 892 | −0.285* | −0.033 | 0.179 |

No significant interaction.

### Job Insecurity × Unemployment Risk

| Term | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Exposure | 4,502 | 1,309 | +0.190 | 0.066 | 0.004 |
| Unemployment risk (centered) | 4,502 | 1,309 | +0.101 | 0.011 | <0.001 |
| Interaction | 4,502 | 1,309 | +0.002 | 0.015 | 0.902 |

Unemployment risk is a strong independent predictor of job insecurity, but the interaction is zero. AI exposure raises job insecurity independently of whether the worker already perceives high unemployment risk.

**Interpretation across Module C:** The absence of fear-mediation interactions implies AI exposure operates through a structural rather than subjective perception channel. Workers at AI-exposed firms feel more insecure and want more redistribution regardless of their baseline anxiety levels.

---

## Module D: Heterogeneous Effects (Gender and Education)

### Job Insecurity

| Subgroup | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Full sample | 4,533 | 1,313 | **+0.225** | 0.082 | **0.006** |
| Male | 2,544 | 715 | +0.181 | 0.113 | 0.108 |
| Female | 1,989 | 598 | **+0.258** | 0.120 | **0.032** |
| High education | 2,356 | 679 | +0.211 | 0.133 | 0.112 |
| Low education | 2,177 | 658 | +0.186 | 0.106 | 0.080 |
| Pre-2018 | 1,471 | 599 | +0.190 | 0.182 | 0.297 |
| Post-2018 | 3,062 | 1,071 | +0.226 | 0.137 | 0.098 |

Broadly similar across subgroups. Full-sample significance reflects power from pooling; subgroup estimates are directionally consistent.

### Vote SVP

| Subgroup | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Full sample | 4,172 | 1,249 | **−0.055** | 0.025 | **0.027** |
| Male | 2,409 | 694 | −0.050 | 0.034 | 0.143 |
| Female | 1,763 | 555 | −0.073 | 0.043 | 0.092 |
| High education | 2,228 | 665 | −0.062 | 0.035 | 0.079 |
| Low education | 1,944 | 605 | −0.042 | 0.038 | 0.266 |
| Pre-2018 | 1,351 | 567 | −0.068 | 0.057 | 0.232 |
| Post-2018 | 2,821 | 1,018 | −0.006 | 0.039 | 0.871 |

The SVP effect is largely concentrated in the **pre-2018 period**. The post-2018 coefficient collapses to near-zero — the anti-SVP effect may have attenuated as AI adoption became more widespread and normalized.

### Vote SP (sharpest heterogeneity in dataset)

| Subgroup | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Full sample | 4,172 | 1,249 | +0.015 | 0.025 | 0.540 |
| **Male** | 2,409 | 694 | **+0.110** | 0.034 | **0.001** |
| **Female** | 1,763 | 555 | **−0.120** | 0.051 | **0.019** |
| High education | 2,228 | 665 | **+0.073** | 0.028 | **0.010** |
| Low education | 1,944 | 605 | −0.038 | 0.048 | 0.431 |
| Pre-2018 | 1,351 | 567 | −0.010 | 0.080 | 0.903 |
| Post-2018 | 2,821 | 1,018 | +0.011 | 0.033 | 0.734 |

**The full-sample null (+0.015) masks a near-perfect cancellation:** men at AI-exposed firms shift strongly toward SP (+0.110***); women shift equally strongly away (−0.120**). The aggregate is a composition effect, not a true zero. The education split reinforces the male pattern: high-education workers drive the SP positive effect.

### Vote FDP

| Subgroup | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Full sample | 4,172 | 1,249 | −0.011 | 0.049 | 0.818 |
| Male | 2,409 | 694 | +0.015 | 0.043 | 0.727 |
| Female | 1,763 | 555 | −0.063 | 0.089 | 0.476 |
| High education | 2,228 | 665 | −0.080 | 0.069 | 0.245 |
| Low education | 1,944 | 605 | +0.033 | 0.058 | 0.571 |
| Pre-2018 | 1,351 | 567 | +0.024 | 0.070 | 0.729 |
| Post-2018 | 2,821 | 1,018 | −0.026 | 0.048 | 0.591 |

### Vote GLP

| Subgroup | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Full sample | 4,172 | 1,249 | +0.002 | 0.029 | 0.938 |
| Male | 2,409 | 694 | −0.019 | 0.042 | 0.646 |
| Female | 1,763 | 555 | +0.021 | 0.045 | 0.642 |
| High education | 2,228 | 665 | +0.026 | 0.038 | 0.487 |
| Low education | 1,944 | 605 | −0.023 | 0.048 | 0.624 |
| Pre-2018 | 1,351 | 567 | +0.089 | 0.089 | 0.320 |
| Post-2018 | 2,821 | 1,018 | −0.041 | 0.034 | 0.228 |

### Redistribution Support

| Subgroup | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| Full sample | 1,510 | 925 | **+0.288** | 0.121 | **0.017** |
| Male | 853 | 522 | +0.247 | 0.170 | 0.145 |
| Female | 657 | 403 | **+0.393** | 0.159 | **0.014** |
| **High education** | 797 | 491 | **+0.552** | 0.152 | **<0.001** |
| Low education | 713 | 448 | −0.108 | 0.173 | 0.533 |
| Pre-2018 | 583 | 458 | +0.224 | 0.350 | 0.523 |
| Post-2018 | 927 | 701 | +0.344 | 0.222 | 0.122 |

**The redistribution effect is entirely concentrated among high-education workers** (β=+0.552, p<0.001). Low-education workers show a small negative (insignificant). Consistent with "enlightened self-interest": higher-educated workers at AI firms are more likely to understand that AI productivity gains flow to capital and demand redistribution accordingly.

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

**Interpretation:** Firms with higher AI exposure hire more in the future (level effect, β=+0.059***) but growth rates are unaffected. AI adoption predicts permanently higher hiring steady-states, not accelerating expansion. This rules out a simple labor-displacement story at the firm level — AI-adopting firms are not shrinking their workforces.

---

## Module F: Robustness — Sample Selection vs. Control Effect

This module decomposes the discrepancy between the identification robustness memo (redistribution β=+0.177, p=0.116, N=2,147, no lagged firm controls) and Module A (β=+0.288, p=0.017, N=1,510, full controls). Three specs:

- **F1** — No lagged firm controls, full available rotating-battery sample (~2,100). Closest to the identification robustness memo spec.
- **F2** — Full controls including firm lags, full available sample (~1,500). Same as Module A.
- **F3** — No lagged firm controls, restricted to the F2 sample (those with non-missing firm lags). Isolates whether significance in F2 is from the controls absorbing confounders vs. from sample selection.

If β(F2) ≈ β(F3): the firm lags are not changing the estimate — significance is from **sample selection**.
If β(F2) ≠ β(F3): the firm lags are absorbing a confounder within the same sample — significance is from **omitted variable bias** being corrected.

### Redistribution Support (pp17, 1–3)

| Spec | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| F1: no firm lags, full sample | 2,126 | — | +0.161 | 0.111 | 0.146 |
| F2: full controls, full sample | 1,510 | 925 | **+0.288** | 0.121 | **0.017** |
| F3: no firm lags, restricted to F2 sample | 1,510 | — | **+0.269** | 0.119 | **0.023** |

**Verdict: sample selection.** β(F2) ≈ β(F3) (+0.288 vs +0.269) — the firm lag controls barely change the estimate within the same 1,510 observations. The jump from F1 to F2/F3 is driven entirely by *who* survives the firm-lag non-missingness filter. Workers with consecutive-year employment at larger firms have a genuinely stronger redistribution response to AI exposure. The broader rotating-battery sample (F1) contains workers with more intermittent employment where the redistribution effect is weaker or noisier.

### Welfare / Social Spending (pp13, 1–3)

| Spec | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| F1: no firm lags, full sample | 2,056 | — | −0.048 | 0.111 | 0.669 |
| F2: full controls, full sample | 1,468 | 900 | **−0.327** | 0.146 | **0.025** |
| F3: no firm lags, restricted to F2 sample | 1,468 | — | −0.216 | 0.137 | 0.116 |

**Verdict: both sample selection and controls.** β(F2) ≠ β(F3) within the same 1,468 observations (−0.327 vs −0.216). The firm lag controls are absorbing positive confounding — without them, the negative welfare effect is attenuated even in the selected sample. The welfare result is therefore less robust: it depends on both who is in the sample and on controlling for lagged firm activity.

### Nativism / Opp. for Foreigners (pp15, 1–3)

| Spec | N | Persons | β | SE | p |
|---|---|---|---|---|---|
| F1: no firm lags, full sample | 2,117 | — | +0.009 | 0.127 | 0.947 |
| F2: full controls, full sample | 1,493 | 915 | −0.041 | 0.166 | 0.803 |
| F3: no firm lags, restricted to F2 sample | 1,493 | — | −0.064 | 0.149 | 0.669 |

Null across all three specs. The nativism non-result is robust.

### Summary of robustness verdicts

| Outcome | F1 significant? | F2 significant? | F3 significant? | Verdict |
|---|---|---|---|---|
| Redistribution | No (p=0.146) | Yes (p=0.017) | Yes (p=0.023) | Sample selection |
| Welfare | No (p=0.669) | Yes (p=0.025) | No (p=0.116) | Sample selection + controls |
| Nativism | No | No | No | Robust null |

**Implication for the paper:** The redistribution result should be presented with the caveat that it holds within the subset of workers with stable consecutive-year employment at firms in the dataset. This is a selected but substantively coherent subgroup — precisely the workers most exposed to firm-level AI adoption over time. The welfare result is more fragile and should be presented as exploratory, contingent on the full control set.

---

## Notes on Control Specification

The control set used throughout (CONTROLS) is:
- `age_centered`, `age_squared` (time-varying, continuous)
- `contract_perm` (pw36==2, permanent contract, time-varying)
- `part_time` (pw39==1, part-time worker, time-varying)
- `computer_work` (pw607, uses computer at work, time-varying)
- `firm_size_lag1` (pw85 at t-1, lagged to avoid bad-controls endogeneity)
- `log_job_ads_lag1` (log firm job ads at t-1, lagged for same reason)

`female` and `education` are excluded from regression controls (absorbed by person FE / time-invariant) but used for subgroup splits in Module D.

**Control coverage in matched sample:** contract_perm 98.9%, part_time 99.9%, computer_work 100.0%, log_job_ads_lag1 97.8%, firm_size_lag1 74.6%. The 74.6% coverage on firm_size_lag1 (requires consecutive years and non-missing pw85) is the primary driver of N reduction relative to the identification robustness memo spec.
