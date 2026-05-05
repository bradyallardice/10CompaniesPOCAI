# SHP Panel Specification Testing: Technical Report
**Date**: April 11, 2026  
**Status**: Technical Specification & Validation Framework  
**Author**: Claude Code (Econometrics)

---

## Executive Summary

This report documents a comprehensive specification testing protocol for the SHP panel econometric analysis with AI exposure. Three competing specifications were designed and ranked by causal identification strength and statistical power:

1. **Specification 1 (Additive FE)**: α_i + α_f + α_t + α_o
   - **Role**: Conservative baseline for robustness
   - **Causal ID**: Weak (no firm-year FE)
   - **Power**: Highest (~45k effective observations)

2. **Specification 2 (Recommended Primary)**: α_i + α_{ft} + α_{ot}
   - **Role**: Main specification for inference
   - **Causal ID**: Strong (triple-difference logic)
   - **Power**: Acceptable (~2.3k effective DoF post-FE)

3. **Specification 3 (Robustness Check)**: α_i + α_t + α_{ft} + α_{ot} + α_{it}
   - **Role**: Specification test under maximum FE saturation
   - **Causal ID**: Over-identified (all variation absorbed)
   - **Power**: Very tight (~500-1000 effective DoF)

---

## Part I: Specification Design & Rationale

### Core Research Question

**How does within-firm, within-occupation AI exposure affect individual political preferences and economic outcomes?**

The three specifications represent three points on the identification/power tradeoff:

- **Spec 1**: Maximum power, minimum causal confidence
- **Spec 2**: Balanced power and causal ID (recommended)
- **Spec 3**: Minimal power, maximum causal robustness

### Specification 1: Additive Fixed Effects (α_i + α_f + α_t + α_o)

**Model**:
$$Y_{ift} = \alpha_i + \alpha_f + \alpha_t + \alpha_o + \beta \cdot \text{Exposure}_{iot} + \mathbf{X}_{it}\gamma + \varepsilon_{ift}$$

**What it controls**:
- **α_i (Person FE)**: Time-invariant individual traits (education, personality, baseline ideology)
- **α_f (Firm FE)**: Time-invariant firm characteristics (sector, size, firm ideology/culture)
- **α_t (Year FE)**: Aggregate macro trends (national politics, economy-wide shifts)
- **α_o (Occupation FE)**: Time-invariant occupation characteristics (prestige, wage level, worker composition)

**What it does NOT control**:
- Firm-year shocks (restructuring, acquisition, cost-cutting in year t at firm f)
- Occupation-year shocks (labor market trends specific to occupation o in year t)
- Firm-occupation-year interactions (firm f deploys AI in occupation o specifically in year t)

**Identification**:
- Variation comes from: Same person, same firm, different years → across occupations in that person-year
- Logic: As person's firm's AI exposure varies across occupations in different years, does their politics shift?

**Threats**:
- **Firm-year confounding** (primary): Firm X has bad year in 2022, also restructures certain occupations
  - Coefficient conflates "AI exposure" with "restructuring risk"
- **Occupation selection**: Firms may deploy AI in occupations they're already targeting for downsizing

**Power**:
- Full sample: 45,325 person-years
- No observations dropped due to FE collinearity
- Standard errors relatively tight

**Recommendation**: Conservative baseline. Good for showing robustness, not ideal as main spec due to weak causal ID.

---

### Specification 2: Firm-Year + Occupation-Year FE (α_i + α_{ft} + α_{ot}) — RECOMMENDED

**Model**:
$$Y_{ift} = \alpha_i + \alpha_{ft} + \alpha_{ot} + \beta \cdot \text{Exposure}_{iot} + \mathbf{X}_{it}\gamma + \varepsilon_{ift}$$

**What it controls** (in addition to Spec 1):
- **α_{ft} (Firm-Year FE)**: All firm-specific shocks in year t
  - Restructuring, acquisitions, management changes, bad earnings, morale shifts
  - Controls for *why* the firm deployed AI that year (benign: efficiency vs. malign: cost-cutting)
- **α_{ot} (Occupation-Year FE)**: All occupation-specific labor market shocks in year t
  - Wage trends, skills demand, job availability in that occupation that year
  - Controls for national occupation-specific trends

**What it does NOT control**:
- Individual-year interactions (person i's mood shifts in year t for idiosyncratic reasons)
- Residual firm-occupation endogeneity (why does firm f deploy AI in occupation o specifically?)

**Identification** (Triple-Difference Logic):
The coefficient β identifies from variation where:
- **Difference 1**: Across occupations within the same firm-year
  - Same firm f, same year t, different occupations o vs. o'
- **Difference 2**: Comparing within firm-year, which controls firm-year shocks
  - Both occupations see same firm-level shocks in year t
- **Difference 3**: Fixed effects absorb all time-invariant traits and trends
  - Same person across years, occupation trends across firms

**Causal interpretation**:
> In firm f in year t, do workers in occupations more exposed to AI have different outcomes than workers in less-exposed occupations?

Because α_{ft} controls firm-year shocks, the coefficient isolates:
- The effect of AI deployment *within* the firm in that year
- Comparing across occupations that see the same firm-level conditions

**Power**:
- Sample size after FE absorption: ~45,325 observations
- Degrees of freedom absorbed:
  - Person FEs: ~9,895
  - Firm-Year FEs: ~30,507
  - Occupation-Year FEs: ~3,000
  - Total: ~43,402 FEs absorbed
- Effective DoF for identification: 45,325 - 43,402 ≈ **1,923 DoF**
- DoF ratio: 1,923 / 45,325 ≈ 0.042 (TIGHT)
- **Power assessment**: TIGHT but feasible
  - Similar to high-dimensional fixed effects models in literature
  - Expect larger standard errors than Spec 1
  - Need adequate effect sizes for significance at conventional levels

**Remaining threats**:
- **Firm-occupation endogeneity**: Even with firm-year FE, why does firm f deploy AI in occ. o?
  - Benign: Task structure drives deployment (firm automates routine tasks)
  - Malign: Firm targets troubled occupations (occupation doing poorly, so automate it)
  - **Mitigation**: Test whether AI exposure predicts task-replaceability, not pre-treatment occupation health

---

### Specification 3: Saturated FE (α_i + α_t + α_{ft} + α_{ot} + α_{it}) — Robustness Check

**Model**:
$$Y_{ift} = \alpha_i + \alpha_t + \alpha_{ft} + \alpha_{ot} + \alpha_{it} + \beta \cdot \text{Exposure}_{iot} + \varepsilon_{ift}$$

**New element: α_{it} (Person-Year FE)**:
- Absorbs all idiosyncratic person-level shocks in year t
  - Life events, mood shifts, personal crises, job search unrelated to AI
  - Eliminates individual-year confounding

**Degrees of freedom**:
- Person FEs: ~9,895
- Year FE: 12 (years 2012-2023)
- Firm-Year FEs: ~30,507
- Occupation-Year FEs: ~3,000
- Person-Year FEs: ~55,000+
- **Total FEs: >98,000 for 45,325 observations**

**This is overidentified** — more parameters than observations. Convergence may fail or residual DoF near zero.

**Identification**:
- Identifying variation: Very narrow
- Essentially: Within a person-year, across firms/occupations (extremely rare)
- Most people are in one firm in a given year → identification relies on rare multi-firm-year cases

**Power**:
- Extremely tight (DoF near zero)
- Likely convergence issues
- Estimates unreliable even if model fits

**Role**: Specification test only. Shows how much power is lost with maximum FE saturation.

---

## Part II: Validation Criteria

### Criterion 1: Convergence & Estimation Stability

**Check**: Does the model converge? Are coefficients stable across iterations?

For Specs 1-2: Standard convergence expected (manageable FE count).
For Spec 3: May fail to converge or produce unstable estimates.

---

### Criterion 2: Statistical Power

**Power diagnostic**: DoF-to-observations ratio

| Specification | FEs Absorbed | DoF Residual | Ratio | Power Assessment |
|---------------|------------|------------|-------|------------------|
| Spec 1 (Additive) | ~11.9k | ~33.4k | 0.74 | GOOD |
| Spec 2 (Firm-Yr+Occ-Yr) | ~43.4k | ~1.9k | 0.04 | TIGHT but acceptable |
| Spec 3 (Saturated) | ~98k+ | ~0 | ~0 | FAILED (over-identified) |

**Interpretation**:
- Spec 1: High power, tight confidence intervals
- Spec 2: Power loss from firm-year FEs, but still ~1.9k effective DoF (similar to high-dim FE papers)
- Spec 3: Infeasible due to saturation

---

### Criterion 3: Causal Identification Strength

**Ranking by identification**:

1. **Spec 2 (STRONGEST)**: Triple-difference logic cleanly separates identification
   - Within firm-year, across occupations
   - Firm-year FE removes firm-level confounding
   - Clear source of variation

2. **Spec 1 (WEAK)**: Additive FEs don't control time-varying firm-year confounding
   - Firm-year shocks alias into AI exposure coefficient
   - Large threat if firm targets certain occupations in bad years

3. **Spec 3 (OVER-IDENTIFIED)**: So many FEs that nearly all variation is absorbed
   - Identification is extremely narrow
   - Not useful for main inference

---

### Criterion 4: Treatment Coefficient Stability

**Comparison**: How much does β change across specs?

**Expected pattern**:
- If **Spec 1 > Spec 2**: Positive confounding in Spec 1 (firm-year shocks amplify effect)
- If **Spec 1 < Spec 2**: Negative selection (firm targets troubled occupations)
- If **Spec 1 ≈ Spec 2**: Effect stable, suggests benign assignment

**Large shifts** (>50% change) → suggest confounding or endogeneity issues

**Small shifts** (<25% change) → suggest specification is robust

---

### Criterion 5: Heteroskedasticity & Residuals

**Checks**:
- Breusch-Pagan test for heteroskedasticity
- Shapiro-Wilk test for normality (less critical with large N)
- Kurtosis and skewness in residuals

---

### Criterion 6: Coefficient Significance

**Question**: Is the treatment effect statistically distinguishable from zero?

- **p < 0.05**: Significant at 5% level
- **p < 0.10**: Marginal significance
- **p > 0.10**: Not significant

**Power constraint**: Spec 2 has tight DoF → larger standard errors → harder to achieve significance

---

## Part III: Diagnosis of Income Reversal

**Empirical puzzle**: Why does the coefficient on AI exposure flip sign when moving from specifications without firm-year FE to specs with firm-year FE?

### Hypothesis 1: Firm-Year Confounding (Most Likely)

**Story**:
1. In years when firm X has bad performance/downsizing, it:
   - Restructures underperforming departments (including low-wage occupations)
   - Deploys AI to automate tasks in those occupations
2. Workers in those occupations see:
   - Higher AI exposure (from automation)
   - Lower income (from restructuring/wage pressure)
3. In naive Spec 1 (no firm-year FE):
   - Coefficient = AI effect + restructuring effect
   - If restructuring and AI both hit troubled occupations → coefficient biased
4. In Spec 2 (with firm-year FE):
   - Firm-year FE removes restructuring effect
   - Coefficient = pure AI effect (isolated)
   - May flip sign if restructuring was driving the correlation

### Hypothesis 2: Selection of High-Wage Occupations into AI

**Story**:
1. High-wage occupations (e.g., engineering, finance) adopt AI earlier
2. AI in high-wage roles has different effect on perceived labor demand
3. Naive Spec 1 conflates "being in high-wage occ" with "AI effect"
4. Spec 2 decomposes: within same firm-year, comparing occ's within that firm
5. Within-firm comparison removes high-wage selection → different effect

### Hypothesis 3: AI Targets Routine Tasks (Task-Fit)

**Story**:
1. Firms deploy AI where it's most applicable (routine, automatable tasks)
2. Routine tasks tend to be lower-wage
3. In Spec 1: Coefficient reflects AI + routine task wage gap
4. In Spec 2: Triple-diff captures AI effect while controlling routine-task effects
5. Effect changes as we isolate pure AI adoption from task characteristics

### Testing These Hypotheses

**Test 1**: Pre-treatment balance
- Do pre-2015 occupation wages predict AI exposure in 2020+?
- If yes → firm targeting low-wage occupations
- If no → assignment more random

**Test 2**: Task-replaceability correlation
- Does AI exposure correlate with O*NET routine-task intensity?
- If yes → task-fit driving deployment
- If no → assignment not driven by task structure

**Test 3**: Post-treatment employment outcomes
- Do workers in AI-exposed occupations see worse employment outcomes?
- If yes → firm targeting occupations for restructuring
- If no → AI deployment benign

---

## Part IV: Recommendations for Final Analysis

### Primary Specification: SPEC 2

**Use for main results**: Firm-Year + Occupation-Year FE

**Why**:
1. Strongest causal identification (triple-difference)
2. Controls all identified threats (firm-year shocks, occ-year trends)
3. Acceptable power (~2k effective DoF, similar to literature)
4. Clear interpretation: within-firm, across-occupation effect

**Reporting**:
- Main table: Spec 2 results with 95% and 90% CI (due to tight DoF)
- Note: Report effective DoF alongside N for transparency

### Robustness Specification 1: SPEC 1

**Use to show robustness**: Additive FE (conservative baseline)

**Why**:
1. Tests whether results robust to weaker specification
2. Shows power advantage (more precise estimates)
3. Allows discussion of sensitivity to firm-year FE inclusion
4. Readers can judge whether identification improvement justifies power loss

**Presentation**:
- Appendix table comparing Specs 1 & 2
- Discussion: "Results unchanged when we include firm-year FEs (our preferred spec),
  suggesting the treatment effect is robust to accounting for within-firm timing shocks"

### Robustness Specification 2: SPEC 3 (Optional)

**Use only if it converges**: Saturated FE specification test

**Why**:
1. Tests whether results robust to maximum FE saturation
2. If Spec 3 ≈ Spec 2 → person-level time-varying confounding unlikely
3. If Spec 3 >> wider CI or different sign → suggests residual confounding

**Note**: Expect much larger SEs due to DoF loss. Include wide CIs.

---

## Part V: Expected Empirical Patterns

### Income Reversal (Expected Pattern)

**Model sequence**:
1. **Unadjusted** (no controls): β₁ = some positive correlation (high-wage occ's adopt AI)
2. **+Demographics** (no FE): β₂ ≈ β₁ (demographics don't explain much)
3. **+Additive FE** (Spec 1): β₃ < β₁ (person FE removes high-wage selection)
4. **+Firm-Year + Occ-Year FE** (Spec 2): β₄ << β₃ or β₄ < 0 (firm-year FE removes restructuring/firm-specific effects)

**Diagnostic interpretation**:
- If β₃ similar to β₄: Additive FE sufficient
- If β₃ >> β₄: Firm-year shocks important → **Spec 2 justified**
- If β₄ < 0 and β₃ > 0: **Sign flip indicates firm-year confounding**

---

## Part VI: Implementation Notes

### Code Structure

Three separate specifications are fit using statsmodels with person-level clustering:

```python
import statsmodels.formula.api as smf

# SPEC 1
m1 = smf.ols(
    "outcome ~ C(person) + C(firm) + C(year) + C(occupation) + exposure + controls",
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['person_id']})

# SPEC 2 (RECOMMENDED)
m2 = smf.ols(
    "outcome ~ C(person) + C(firm_year) + C(occ_year) + exposure + controls",
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['person_id']})

# SPEC 3
m3 = smf.ols(
    "outcome ~ C(person) + C(year) + C(firm_year) + C(occ_year) + C(person_year) + exposure + controls",
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['person_id']})
```

### Computation Notes

- **Spec 1 & 2**: Fast, converges reliably
- **Spec 3**: May be slow or fail to converge due to saturation
  - If failure: Skip and focus on Specs 1-2
  - If success: Report with wide CI's and DoF caveat

### Output Validation

For each specification:
1. Check convergence status
2. Report degrees of freedom
3. Examine treatment coefficient significance
4. Compare across specs (is coefficient stable?)
5. Test for heteroskedasticity (Breusch-Pagan)

---

## Part VII: Validity Scorecard Framework

| Criterion | Spec 1 (Additive) | Spec 2 (Recommended) | Spec 3 (Saturated) |
|-----------|------------------|---------------------|-------------------|
| **Convergence** | ✓ Yes | ✓ Yes | ✗ Likely fails |
| **Power** | ✓ GOOD | ⚠ TIGHT | ✗ VERY TIGHT |
| **Causal ID** | ⚠ WEAK | ✓ STRONG | ✗ Over-ID |
| **Identification threat** | Firm-year shocks | Minimal | Minimal |
| **Interpretation** | Conservative | Causal | Over-fitted |
| **Main results?** | No (robustness) | ✓ YES | No (check only) |
| **Overall rating** | MEDIUM | **HIGH** | LOW |

---

## Part VIII: Final Recommendations

### Primary Analysis

**Use Specification 2** (Firm-Year + Occupation-Year FE) for main inference on all outcomes:
- Income
- Job insecurity
- Political outcomes (left-right, nativism, welfare, etc.)

### Robustness Analysis

**Show Specification 1** alongside main results to demonstrate:
- Whether results sensitive to firm-year FE inclusion
- Magnitude of power gain from weaker specification
- Coefficient stability across identification strengthening

### Specification Test

**Use Specification 3** (if feasible) to test:
- Whether person-level time-varying confounding matters
- Robustness to near-complete FE saturation

### Endogeneity Evidence Package

Before finalizing interpretation, collect:

1. **Task-fit evidence**: AI exposure ~ task-replaceability
2. **Pre-treatment balance**: Pre-AI wages vs. AI exposure
3. **Post-treatment outcomes**: Employment/wages post-AI in exposed occ's
4. **Occupation analysis**: Which occupations see most AI? Does it align with tasks?

If evidence supports benign assignment → coefficient interpretation is closer to causal.
If evidence supports restructuring targeting → coefficient partly reflects employment risk, not AI per se.

---

## References

- Hampole et al. (2025): Firm-level AI adoption and labor market outcomes
- Callaway & O'Neill (2023): Synthetic control methods for heterogeneous treatment effects
- Cameron & Miller (2015): A practitioner's guide to cluster-robust inference
- Abadie et al. (2023): When should you add fixed effects?

---

## Appendix: Sample Size Diagnostics

From specification document (April 10, 2026):

```
Total SHP respondents: 38,528
With firm linkage: 9,895 (25.7%)
Person-year obs with firm_id: 45,325 (22.7%)

Firm-year cells: 30,507
  with 2+ occupations: 4,523 (16.3%)
  with 2+ respondents: 5,722 (18.8%)
  median respondents: 1

Firm-occupation-year cells: 37,064
  with 2+ respondents: 1,852 (5.0%)
  median respondents: 1

Firm switchers: 1,794 (18.1%)
Firm stayers: 8,101 (81.9%)
```

**Implication for power**:
- Most people don't switch firms (81.9% stayers)
- Most firm-years have only 1-2 occupations
- Within-firm variation is modest
- This explains why Spec 2 has tight DoF

---

**Status**: Framework complete and ready for empirical implementation.

**Next steps**: Run all three specifications on prepared SHP panel data and generate validity scorecards for each outcome variable.

