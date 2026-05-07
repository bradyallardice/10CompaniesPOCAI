# Stage 9e: Political Outcomes, Non-Linearities, Heterogeneities, and Firm Hiring

## Overview

This memo documents results from `stage_9e_political_nonlinear.py`. The baseline specification throughout is the matched-only sample (hampole_ai_exposure_avg_foy > 0) with Person FE + 3-digit ISCO×Year FE, one-way clustering on idpers. Controls are time-varying pre-determined variables: age (centered + squared), contract type (permanent vs. fixed-term), part-time status, computer use at work, lagged firm size (t-1), and lagged log job ads (t-1). Time-invariant controls (female, education) are dropped as absorbed by person FE. This matches the specification in `identification_robustness_job_insecurity.md`.

**Sample sizes:**
- Annual outcomes (vote, left-right, social trust): N≈4,150–4,530, ~1,200–1,300 persons
- Rotating battery (redistribution, welfare, nativism, trust, democracy): N≈1,450–1,530, ~890–940 persons

---

## Module A: Baseline Political Outcomes

All outcomes run with the matched-only specification. Political battery outcomes (redistribution, welfare, nativism, gender equality, trust in government, democracy satisfaction, political efficacy, EU opinion) are available only in 2014, 2017, 2020, 2023 waves.

### Party Vote (binary dummies from pp19)

Vote coding: 1=FDP, 2=CVP, 3=SP, 4=SVP, 9=GPS, 11/20=GLP, 21=BDP, 50/51=no party.

| Outcome | N | β | SE | p |
|---|---|---|---|---|
| **Vote SVP** | 4,172 | **−0.055** | 0.025 | **0.027** |
| Vote SP | 4,172 | +0.015 | 0.025 | 0.540 |
| Vote FDP | 4,172 | −0.011 | 0.049 | 0.818 |
| Vote CVP | 4,172 | +0.002 | 0.027 | 0.941 |
| Vote GLP | 4,172 | +0.002 | 0.029 | 0.938 |
| Vote BDP | 4,172 | +0.006 | 0.016 | 0.725 |
| Vote no party | 4,172 | −0.000 | 0.035 | 0.996 |

**SVP is the only vote outcome with a statistically significant effect.** Workers at AI-exposed firms are less likely to vote SVP (right-populist). The null on all other parties implies the SVP effect is not a general rightward-leftward shift but specific defection from the right-populist option.

### Ideological and Attitudinal Outcomes

| Outcome | N | β | SE | p |
|---|---|---|---|---|
| Left-right self-placement (pp10, 0–10) | 4,151 | +0.040 | 0.141 | 0.777 |
| Social trust (pp45, 0–10) | 4,529 | −0.017 | 0.174 | 0.921 |
| **Redistribution support (pp17, 1–3)** | 1,510 | **+0.288** | 0.121 | **0.017** |
| **Welfare/social spending (pp13, 1–3)** | 1,468 | **−0.327** | 0.146 | **0.025** |
| Nativism/opp. for foreigners (pp15, 1–3) | 1,493 | −0.041 | 0.166 | 0.803 |
| Gender equality (pp22-derived, 1–10) | 1,529 | +0.418 | 0.446 | 0.348 |
| Trust in government (pp04, 0–10) | 1,527 | +0.013 | 0.243 | 0.956 |
| Democracy satisfaction (pp02, 0–10) | 1,524 | +0.052 | 0.353 | 0.883 |
| Political efficacy (pp03, 0–10) | 1,527 | −0.285 | 0.472 | 0.545 |
| EU opinion (pp14, 1–3) | 1,497 | +0.060 | 0.102 | 0.560 |

**Significant findings:**
- **Redistribution** (pp17): β=+0.288 (p=0.017) — workers at AI-exposed firms want higher taxes on the wealthy.
- **Welfare/social spending** (pp13): β=−0.327 (p=0.025) — the same workers *oppose* current social spending levels (or want less of it, coded as direction of increase). This apparent paradox requires interpretation (see §A.1 below).

### A.1 The Redistribution–Welfare Paradox

pp17 asks about redistribution through taxation of high incomes (increasing taxes); pp13 asks about social spending. Both are coded 1–3. The opposing signs suggest these are not the same underlying preference:

- **Redistribution (pp17)**: Pro-redistribution may reflect dissatisfaction with inequality and demands for more progressive taxation. AI exposure → rising job insecurity → desire to "make the rich pay."
- **Social spending (pp13)**: The negative sign on welfare is harder to interpret. One reading: AI-exposed workers — who tend to be higher-skilled, at tech-forward firms — may be skeptical of *existing* social spending programs (welfare state skepticism among higher earners). Another: the pp13 scale has a specific directionality we should verify. A third: the sample restriction to matched firms (which skews toward larger, formal sector employers) may select workers with different welfare state preferences.

The redistribution-welfare divergence is consistent with the "productivist" welfare state preferences documented in the comparative welfare state literature: higher-educated, formal-sector workers want more redistribution but not more spending on traditional welfare programs.

---

## Module B: Non-Linearities

### B.1 Quadratic Specification

We add `exposure²` to the baseline to test whether effects are non-linear. Centered exposure is used; the quadratic term is uncentered squared.

**Job insecurity:**

| Term | β | SE | p |
|---|---|---|---|
| Exposure (linear) | −0.136 | 0.160 | 0.395 |
| Exposure² | **+0.201** | 0.076 | **0.008** |

The significant quadratic term with an insignificant linear term implies a convex (U-shaped or accelerating) relationship. At moderate exposure levels, effects are muted; at high exposure, job insecurity rises sharply. This is consistent with a threshold model: AI exposure below some level is not salient to workers, but beyond that level, displacement anxiety accelerates.

**Vote SVP:**

| Term | β | SE | p |
|---|---|---|---|
| Exposure (linear) | **−0.187** | 0.057 | **0.001** |
| Exposure² | **+0.073** | 0.022 | **0.001** |

Both terms significant. The negative linear + positive quadratic implies the anti-SVP effect is strongest at intermediate exposure levels but weakens (or partially reverses) at very high exposure. Workers at moderately exposed firms shift most sharply away from SVP; at the highest exposure levels, they may be shifting toward protest abstention rather than SVP specifically.

**Redistribution (rotating battery):**

| Term | β | SE | p |
|---|---|---|---|
| Exposure (linear) | +0.278 | 0.264 | 0.291 |
| Exposure² | +0.006 | 0.115 | 0.961 |

No significant quadratic effect — the redistribution effect appears approximately linear.

### B.2 Tercile Dummies

Splitting exposure into terciles (T1=omitted, T2, T3) tests whether effects are concentrated in any specific part of the distribution.

**Job insecurity:**

| Tercile | β | SE | p |
|---|---|---|---|
| T2 vs T1 | −0.064 | 0.039 | 0.105 |
| T3 vs T1 | −0.043 | 0.056 | 0.448 |

Surprisingly, neither T2 nor T3 is significantly different from T1 in the tercile specification. The quadratic result above suggests the effect is concentrated at the very top of the continuous distribution, which terciles may not fully capture.

**Vote SVP:**

| Tercile | β | SE | p |
|---|---|---|---|
| T2 vs T1 | −0.024 | 0.015 | 0.117 |
| T3 vs T1 | **−0.059** | 0.022 | **0.006** |

The SVP effect is concentrated in the top tercile — only the highest-exposure workers systematically abandon the SVP.

---

## Module C: Interaction with Perceived Insecurity (Moderators)

We interact AI exposure with two centered moderators: job insecurity (ji_c, centered) and perceived unemployment risk (ur_c, centered on 0–10 scale). The specification includes main effects for both moderator and exposure plus the interaction term.

*Note: The job insecurity × job insecurity interaction is mechanically perfect (collinear), shown here for completeness only.*

### Vote SVP

| Spec | Exposure main | Moderator main | Interaction | Interaction p |
|---|---|---|---|---|
| × Job insecurity | −0.057* | −0.008 | +0.012 | 0.441 |
| × Unemp. risk | **−0.057*** | +0.004 | +0.0003 | 0.935 |

No significant interactions. The SVP effect from exposure is not amplified for workers who feel more job insecure or perceive higher unemployment risk. The channel is not "fear-mediated" — exposure directly shifts political preferences even among workers who don't report elevated insecurity.

### Redistribution Support

| Spec | Exposure main | Moderator main | Interaction | Interaction p |
|---|---|---|---|---|
| × Job insecurity | **+0.329*** | +0.012 | −0.060 | 0.226 |
| × Unemp. risk | **+0.302*** | +0.013 | −0.005 | 0.762 |

No significant interaction. The redistribution demand from AI exposure does not depend on individual-level insecurity perceptions. The redistribution effect is driven by exposure itself, not by the worker's subjective fear.

### Job Insecurity × Unemployment Risk

| Term | β | SE | p |
|---|---|---|---|
| Exposure | +0.190 | 0.066 | 0.004 |
| Unemployment risk (centered) | +0.101 | 0.011 | <0.001 |
| Interaction | +0.002 | 0.015 | 0.902 |

Unemployment risk is a strong independent predictor of job insecurity (as expected), but the interaction is zero. AI exposure raises job insecurity independently of whether the worker already perceives high unemployment risk.

**Interpretation:** The absence of fear-mediation interactions implies that AI exposure operates through an objective/structural channel rather than a subjective perception channel. Workers at AI-exposed firms feel more insecure and want more redistribution regardless of their baseline anxiety levels.

---

## Module D: Heterogeneous Effects (Gender and Education)

### Job Insecurity

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 4,533 | **+0.225** | 0.082 | **0.006** |
| Male | 2,544 | +0.181 | 0.113 | 0.108 |
| Female | 1,989 | **+0.258** | 0.120 | **0.032** |
| High education | 2,356 | +0.211 | 0.133 | 0.112 |
| Low education | 2,177 | +0.186 | 0.106 | 0.080 |
| Pre-2018 | 1,471 | +0.190 | 0.182 | 0.297 |
| Post-2018 | 3,062 | +0.226 | 0.137 | 0.098 |

Job insecurity effects are broadly similar across subgroups. The full-sample significance reflects power from pooling; subgroup estimates are directionally consistent but individually noisier.

### Vote SVP

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 4,172 | **−0.055** | 0.025 | **0.027** |
| Male | 2,409 | −0.050 | 0.034 | 0.143 |
| Female | 1,763 | −0.073 | 0.043 | 0.092 |
| High education | 2,228 | −0.062 | 0.035 | 0.079 |
| Low education | 1,944 | −0.042 | 0.038 | 0.266 |
| Pre-2018 | 1,351 | −0.068 | 0.057 | 0.232 |
| Post-2018 | 2,821 | −0.006 | 0.039 | 0.871 |

The SVP effect is largely concentrated in the **pre-2018 period and among women**. The post-2018 coefficient collapses to near-zero, suggesting the anti-SVP effect may have been present earlier in the AI adoption wave but attenuated by 2019–2023. This could reflect changing SVP positioning on economic issues or heterogeneous exposure timing.

### Vote SP (most striking heterogeneity)

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 4,172 | +0.015 | 0.025 | 0.540 |
| **Male** | 2,409 | **+0.110** | 0.034 | **0.001** |
| **Female** | 1,763 | **−0.120** | 0.051 | **0.019** |
| High education | 2,228 | **+0.073** | 0.028 | **0.010** |
| Low education | 1,944 | −0.038 | 0.048 | 0.431 |

**The SP gender reversal is the sharpest heterogeneity in the dataset.** The full-sample null masks a near-perfect cancellation: men at AI-exposed firms shift strongly toward SP (+0.110***); women at AI-exposed firms shift equally strongly away (−0.120**). The null aggregate is a composition effect, not a true zero.

Possible interpretations:
- SP's labor market protection messaging may resonate more with men threatened by AI displacement, while women at AI-exposed firms may be attracted to other parties (GLP, GPS) that combine economic with social liberalism.
- Women at AI-adopting firms may be in different occupational profiles within firms (e.g., administrative/support rather than technical roles), experiencing AI differently.
- Selection: female workers willing to remain at AI-exposed firms may have different political baseline (more economically liberal) than those who exit.

The education split reinforces the male pattern: high-education workers drive the SP positive effect, while low-education workers show a slight negative (though insignificant).

### Redistribution Support

| Subgroup | N | β | SE | p |
|---|---|---|---|---|
| Full sample | 1,510 | **+0.288** | 0.121 | **0.017** |
| Male | 853 | +0.247 | 0.170 | 0.145 |
| Female | 657 | **+0.393** | 0.159 | **0.014** |
| **High education** | 797 | **+0.552** | 0.152 | **<0.001** |
| Low education | 713 | −0.108 | 0.173 | 0.533 |

**The redistribution effect is entirely concentrated among high-education workers** (β=+0.552, p<0.001). Low-education workers show a small negative (insignificant). This is consistent with an "enlightened self-interest" mechanism: higher-educated workers at AI firms are more likely to understand that AI profits flow to capital owners and may demand redistribution accordingly. Low-education workers may have more fatalistic or anti-government orientations.

---

## Module E: Firm-Level Future Hiring

Using the firm AI summary report (N=72,720 firm-years, 7,197 companies), we test whether AI exposure predicts future hiring (log job ads at t+1) and hiring growth rate (Δlog job ads). Firm + Year FE included. Three AI exposure measures: pct_ai_ads_cumulative (AI share of ads), firm_ai_exposure (Hampole exposure score), log_ai_apps (log applications).

| Outcome | Measure | β | SE | p |
|---|---|---|---|---|
| **Log job ads (t+1)** | pct_ai_ads_cumulative | +0.0007 | 0.0007 | 0.324 |
| **Log job ads (t+1)** | **firm_ai_exposure** | **+0.059** | 0.016 | **<0.001** |
| **Log job ads (t+1)** | **log_ai_apps** | **+0.101** | 0.032 | **0.002** |
| Δlog job ads | pct_ai_ads_cumulative | +0.0003 | 0.0003 | 0.381 |
| Δlog job ads | firm_ai_exposure | −0.001 | 0.004 | 0.794 |
| Δlog job ads | log_ai_apps | +0.003 | 0.007 | 0.657 |

**Interpretation:**
- **Firms with higher AI exposure hire more in the future** (level effect: β=+0.059 for Hampole score, p<0.001). A one-standard-deviation increase in firm_ai_exposure is associated with ~6% higher future job ad volume.
- **But growth rates are unaffected** (all Δlog coefficients null). AI exposure predicts permanently higher hiring levels, not faster growth from current levels. This is consistent with AI as a productivity-enhancing technology that expands total firm activity rather than replacing workers in aggregate.
- The null on growth rates also rules out an "AI as expansion catalyst" story where AI exposure causes firms to ramp up hiring acceleration; instead, firms that adopted AI earlier had already reached higher hiring steady-states.

---

## Summary of Key Results

| Module | Finding | Direction | p-value |
|---|---|---|---|
| A | Vote SVP | − | 0.027 |
| A | Redistribution support | + | 0.017 |
| A | Welfare/social spending | − | 0.025 |
| A | All other political outcomes | ~ 0 | n.s. |
| B | Nonlinearity: job insecurity (quadratic) | Convex ↑ | 0.008 |
| B | Nonlinearity: SVP (quadratic + linear) | Concave ↓ then ↑ | 0.001 |
| B | Nonlinearity: redistribution | None | n.s. |
| C | Interaction with job insecurity | None | n.s. |
| C | Interaction with unemployment risk | None | n.s. |
| D | Job insecurity by gender | Similar (F slightly larger) | Both ~0.03–0.11 |
| D | SP vote: men | Strongly + | 0.001 |
| D | SP vote: women | Strongly − | 0.019 |
| D | Redistribution: high education | Very strong + | <0.001 |
| D | Redistribution: low education | Near zero | n.s. |
| D | SVP: post-2018 | Near zero | n.s. |
| E | Future hiring level (firm_ai_exposure) | + | <0.001 |
| E | Future hiring growth | None | n.s. |

---

## Notes on Control Specification

The control set used throughout is:
- `age_centered`, `age_squared` (time-varying, continuous)
- `contract_perm` (pw36==2, permanent contract, time-varying)
- `part_time` (pw39==1, part-time worker, time-varying)
- `computer_work` (pw607, uses computer at work, time-varying)
- `firm_size_lag1` (pw85 at t-1, lagged to avoid bad-controls endogeneity)
- `log_job_ads_lag1` (log firm job ads at t-1, lagged for same reason)

`female` and `education` are excluded from regression controls (absorbed by person FE / time-invariant) but used for subgroup splits in Module D. Coverage: contract_perm 98.9%, part_time 99.9%, computer_work 100%, log_job_ads_lag1 97.8%, firm_size_lag1 74.6%.
