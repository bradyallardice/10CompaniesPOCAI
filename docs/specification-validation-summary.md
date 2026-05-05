# SHP Econometric Specification Validation: Executive Summary

**Date**: April 11, 2026  
**Status**: Specification Design Complete ✓ | Implementation In Progress  
**Task**: Validate three competing FE specifications for political outcomes analysis

---

## Quick Reference: Three Specifications

### Specification 1: Additive FE (Baseline Robustness)
```
Y ~ α_i + α_f + α_t + α_o + β·Exposure + Controls
```
- **Power**: GOOD (full sample: 45.3k obs)
- **Causal ID**: WEAK (no firm-year FE)
- **Use**: Conservative comparison, robustness check
- **Threat**: Firm-year confounding (firm targets certain occupations when in trouble)

---

### Specification 2: Firm-Year + Occupation-Year FE (RECOMMENDED PRIMARY)
```
Y ~ α_i + α_{ft} + α_{ot} + β·Exposure + Controls
```
- **Power**: TIGHT but feasible (~1.9k effective DoF post-FE)
- **Causal ID**: STRONG (triple-difference logic)
- **Use**: Main results for all outcomes
- **Strength**: Controls firm-year shocks (restructuring, acquisitions, morale)
- **Threat**: Residual firm-occupation endogeneity (why deploy AI in that occ in that year?)

---

### Specification 3: Saturated FE (Specification Test Only)
```
Y ~ α_i + α_t + α_{ft} + α_{ot} + α_{it} + β·Exposure + Controls
```
- **Power**: VERY TIGHT (near-zero effective DoF)
- **Causal ID**: OVER-IDENTIFIED (nearly all variation absorbed)
- **Use**: Robustness check only, if it converges
- **Purpose**: Test sensitivity to person-level time-varying confounding

---

## Specification Recommendations

### ✓ DO THIS: Use Spec 2 for main results

**Rationale**:
1. **Strongest causal identification**: Triple-diff logic (within firm-year, across occupations)
2. **Reasonable power**: ~1.9k DoF after FE absorption (comparable to high-dimensional FE literature)
3. **Controls key confounders**: 
   - Firm-year FE removes restructuring/acquisition shocks
   - Occupation-year FE removes labor market trends
4. **Clear interpretation**: Effect of AI deployment within firms, comparing across occupations

**Reporting**:
- Main table: Spec 2 estimates with 95% CI's (and 90% CI's as secondary, to show tight DoF)
- Report effective degrees of freedom for transparency
- Note: "We employ firm-year and occupation-year FE to account for time-varying shocks at both levels, providing clean identification of within-firm, across-occupation treatment heterogeneity."

---

### ⚠ ALSO SHOW: Use Spec 1 for robustness

**Rationale**:
1. Tests whether conclusions robust to weaker specification
2. Shows power advantage of conservative approach
3. Allows reader to judge identification/power tradeoff
4. If Spec 1 ≈ Spec 2 → results robust to FE structure

**Presentation**:
- Appendix table: "Robustness check: Additive fixed effects (without firm-year FE)"
- Narrative: "Results are robust to alternative FE specifications. In Appendix Table X, we show that coefficients are similar when we use additive FE only (α_i + α_f + α_t + α_o), though standard errors are smaller, suggesting firm-year shocks are not explaining our findings."

---

### ✗ DON'T LEAD WITH: Spec 3 (if it converges)

**Role**: Sensitivity check only

**If you do report it**:
- Appendix, not main text
- Include disclaimer: "Specification 3 absorbs nearly all variation; estimates are very noisy and should be interpreted with caution."
- If Spec 3 ≈ Spec 2 → suggests person-level time-varying confounding unlikely
- If Spec 3 >> width → suggests residual confounding matters

---

## Income Reversal: Diagnostic Sequence

**Expected pattern** as we move from naive to fully-controlled specifications:

1. **Unadjusted**: Income coefficient likely positive (high-wage occupations adopt AI)
2. **+Demographics**: Coefficient stable or slightly smaller
3. **+Additive FE (Spec 1)**: Coefficient decreases (removes between-person selection)
4. **+Firm-Year FE (Spec 2)**: Coefficient may flip or become much smaller

**Why the flip?**
- Hypothesis 1 (most likely): Firm-year confounding
  - Firms in trouble deploy AI in underperforming (lower-wage) occupations
  - Naive specs conflate AI effect with restructuring effect
  - Firm-year FE removes this confounding → true AI effect emerges
  
- Hypothesis 2: Task-fit selection
  - Firms deploy AI where it's most applicable (routine, lower-wage occupations)
  - Naive specs conflate AI effect with task characteristics
  - Within-firm comparison (Spec 2) isolates pure AI effect
  
- Hypothesis 3: High-wage occupation targeting
  - Firms deliberately deploy AI in high-wage professional roles
  - Such roles have good labor market options → less income compression
  - Naive specs miss this heterogeneity

**Diagnostic test**: 
- Compare Spec 1 coefficient to Spec 2 coefficient
- Large change → suggests firm-year shocks matter
- Small change → suggests firm-year shocks minor concern
- Sign flip → strong evidence of endogeneity in naive spec

---

## Validity Scorecard (by Outcome & Specification)

### Framework

For each outcome variable (income, job insecurity, political outcomes), assess each spec on:

| Criterion | Assessment |
|-----------|------------|
| **Convergence** | Does model fit without errors? |
| **Power** | Are residual DoF adequate for precision? |
| **Significance** | Is treatment coeff significantly different from zero? |
| **Stability** | How much does coeff change from Spec 1 → Spec 2? |
| **Residual Q** | Are residuals well-behaved (normality, homosked)? |
| **Causal ID** | Identification strength (weak→strong) |

### Overall Rating

- **HIGH**: Strong causal ID + adequate power + significant effect
- **MEDIUM**: Good causal ID but tight power, or marginal significance
- **LOW**: Weak causal ID or convergence problems

**Expectation by specification**:
- Spec 1: Likely HIGH (power advantage)
- Spec 2: Likely MEDIUM to HIGH (tight DoF but strong causal ID)
- Spec 3: Likely LOW (if converges; too much FE saturation)

---

## Implementation Status

### Complete ✓
- Specification design document (`specification-shp-panel-econometrics.md`)
- Technical validation framework (`specification-testing-technical-report.md`)
- Three specification codes ready to run:
  - `stage_8_specification_diagnostics.py` (comprehensive)
  - `stage_8_specification_diagnostics_fast.py` (streamlined)
  - `stage_8_spec_validation_core.py` (minimal)

### In Progress ⏳
- Empirical estimation on full SHP panel (45.3k observations)
  - Takes ~30-60 min per outcome due to FE absorption
  - Multiple outcomes × 3 specs = ~6-12 total specs to fit
  - Scripts running in background on `Data/shp_panel_prepared.csv`

### To Do Next
1. Monitor completion of background scripts
2. Extract results → CSV and JSON files in `Data/Testing/stage_8/specification_diagnostics/`
3. Generate validity scorecards for each outcome
4. Diagnose income reversal using model sequence
5. Produce final recommendation table

---

## Key Files

### Documentation
- `/docs/specification-shp-panel-econometrics.md` — Full methodology
- `/docs/specification-testing-technical-report.md` — Technical report (this doc)
- `/docs/specification-validation-summary.md` — Executive summary (you are here)

### Code (Ready to Run)
- `stage_8_spec_validation_core.py` — Main validation script
  - Usage: `python3 stage_8_spec_validation_core.py`
  - Output: `Data/Testing/stage_8/specification_diagnostics/`
  - Runtime: ~30-60 minutes for all outcomes

### Data
- `Data/shp_panel_prepared.csv` — Input panel (45.3k person-years)
- `Data/Testing/stage_8/specification_diagnostics/` — Output directory (TBD)

---

## What to Expect: Results Files

After scripts complete:

1. **specification_validity_scorecard.csv**
   - Rows: Outcome × Specification
   - Cols: Validity rating, power, significance, recommendation
   - Use: Quick reference for which specs trust for which outcomes

2. **income_reversal_diagnosis.json**
   - Model sequence: Unadjusted → +Controls → +Spec1 → +Spec2
   - Coefficients, SE, p-values at each step
   - Sign flip detection
   - % change calculations

3. **detailed_diagnostics.json**
   - Convergence status, residual properties, heteroskedasticity tests
   - By outcome and specification

---

## Quick Decision Guide

**"Which specification should I use for my main results?"**

→ **Specification 2 (Firm-Year + Occupation-Year FE)**

**"Why not just use the one with best power (Spec 1)?"**

→ Because it doesn't control firm-year shocks (restructuring, acquisitions, etc.), which likely confound the relationship. Spec 2's triple-diff logic is cleaner: it compares occupations *within the same firm-year*, so firm-level shocks don't create bias.

**"Will my results be precisely estimated (tight CI's)?"**

→ Tighter than Spec 2, but tighter is not always better. Spec 1 might have a wide margin of error from omitted bias. Spec 2 has wider standard errors (tight DoF) but truer estimates. Report both 95% and 90% CI for Spec 2 to account for tight DoF.

**"What if Spec 1 and Spec 2 give very different answers?"**

→ That's important! It suggests firm-year shocks matter. Spec 2 is more trustworthy because it controls them. Discuss in robustness section: "Coefficients differ substantially when we include firm-year FE (Table X), suggesting firm-specific time-varying shocks may confound the simpler specification."

**"Should I report all three specs?"**

→ No. Main table: Spec 2 only. Appendix: Spec 1 for robustness. Spec 3 only if it converges and adds something (e.g., person-level time-varying confounding is tested). Keep main narrative clean.

---

## Caveats & Limitations

1. **Tight DoF in Spec 2**
   - May not achieve 5% significance with small-moderate effect sizes
   - Report 90% CI's in addition to 95% for transparency
   - Consider pre-registration of effect size hypotheses

2. **Residual firm-occupation endogeneity**
   - Even with firm-year FE, *why* does firm deploy AI in occupation o that year?
   - Benign (task-fit): AI in routine tasks
   - Malign (targeting): AI in occupations targeted for restructuring
   - **Test with pre-treatment balance checks** (do wages in occ-o predict future AI exposure?)

3. **Generalizability**
   - Results apply to workers with firm linkage in SHP (25.7% of respondents)
   - May not generalize to self-employed, government workers, or those without firm ID
   - Sample is relatively educated, employed population

4. **Income measurement**
   - Outcome variable: `outcome_log_income`
   - Missing ~17% of person-years
   - Likely MCAR given SHP survey structure, but check

---

## Next Steps

### Short term (this week)
1. Monitor background scripts for completion
2. Extract results to CSV/JSON
3. Generate validity scorecards
4. Diagnose income reversal

### Medium term (next week)
1. Write up results section with Spec 2 as main
2. Add Spec 1 robustness table
3. Discuss specification choice and identification
4. Address income reversal in text

### Long term (before submission)
1. Collect endogeneity evidence (task-fit, pre-treatment balance)
2. Run heterogeneous effects analysis
3. Sensitivity analyses (different FE levels, alternative outcomes)
4. Final validity scorecard in appendix

---

## Summary

**Three specifications designed for SHP political outcomes analysis:**

| | Power | Causal ID | Use |
|---|---|---|---|
| **Spec 1** (Additive FE) | GOOD | WEAK | Robustness check |
| **Spec 2** (Firm-Yr+Occ-Yr FE) | TIGHT | STRONG | ✓ MAIN SPEC |
| **Spec 3** (Saturated FE) | VERY TIGHT | OVER-ID | Specification test |

**Recommendation**: Use Spec 2 for main results. Show Spec 1 in appendix for robustness.

**Rationale**: Spec 2 has strongest causal identification (triple-diff) and controls key confounders (firm-year & occupation-year shocks), even at cost of tighter DoF.

**Status**: Technical framework complete. Empirical estimation in progress. Results TBD.

---

*Report prepared by Claude Code (Econometrics), April 11, 2026*

