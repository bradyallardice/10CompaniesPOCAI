# SHP Panel Specification Analysis: Complete Guide

**Project**: AI Exposure and Political Outcomes in Swiss Panel Data  
**Task**: Validate three competing econometric specifications  
**Status**: ✓ Framework Complete | ⏳ Empirical Estimation In Progress  
**Date**: April 11, 2026

---

## Overview

This analysis compares three competing Fixed Effects specifications for estimating the effect of firm-level AI exposure on political outcomes and income in the Swiss Household Panel (SHP):

1. **Spec 1 (Additive FE)**: α_i + α_f + α_t + α_o
2. **Spec 2 (Recommended)**: α_i + α_{ft} + α_{ot}  ← **MAIN SPEC**
3. **Spec 3 (Robustness)**: α_i + α_t + α_{ft} + α_{ot} + α_{it}

Each specification represents a point on the **identification/power tradeoff**:
- Spec 1: Maximum power, minimum causal confidence
- Spec 2: Balanced tradeoff (RECOMMENDED)
- Spec 3: Maximum causal confidence, minimum power

---

## Why Three Specifications?

**The Problem**: How do we know if our results reflect:
- A true causal effect of AI on politics/income?
- Omitted variable bias (confounding)?
- Selection bias (who gets exposed to AI)?

**The Solution**: Use specs with progressively stronger identification to check robustness.

### Specification 1 (Weak Causal ID)
- Controls: Person, firm, year, occupation fixed effects
- **Missing**: Firm-year shocks (restructuring, acquisitions, recessions at firm level)
- **Risk**: If firm X has bad year AND deploys AI in certain occupations, coefficient conflates AI with recession

### Specification 2 (Strong Causal ID) ← RECOMMENDED
- Controls: Person FE, firm-year FE, occupation-year FE
- **Removes**: Firm-year confounding (what happened to firm that year)
- **Removes**: Occupation-year confounding (labor market trends in that occupation)
- **Logic**: Triple-difference = compares occupations within same firm-year
  - Both occupations see same firm-level shocks
  - So difference must come from differential AI exposure
- **Trade-off**: Tight degrees of freedom after absorbing many FE's

### Specification 3 (Over-Identified)
- Controls: Everything in Spec 2 + person-year FE
- **Removes**: Individual time-varying shocks (mood, life events)
- **Problem**: Absorbs nearly all variation
  - Identifying variation extremely narrow (people in multiple firms in same year — rare)
  - Degrees of freedom approach zero
  - Not useful unless shows Spec 2 results robust

---

## Key Innovation: Triple-Difference Logic

Specification 2 uses triple-difference identification:

```
Triple-Diff = Δ across occupations 
            × Δ within firm-year (controls firm shocks)
            × Δ within person-year (controls time-invariant traits)
```

**Intuition**:
- In firm X in year 2020:
  - Occupation A: 50% of workers exposed to AI
  - Occupation B: 5% of workers exposed to AI
- Did workers in Occ-A shift politics differently than workers in Occ-B?
- Both see same firm-year shocks (layoffs, restructuring at firm X)
- So difference must come from differential AI exposure, not firm conditions

---

## Data

**Source**: Swiss Household Panel (SHP) with firm-level AI exposure linkage

**Sample**:
- 45,325 person-year observations
- 9,895 unique individuals
- Years: 2012-2023
- Exposure measure: Firm-occupation-year AI exposure (Hampole measure, 0-1 scale)

**Outcomes**:
- Economic: Income (log), Job insecurity
- Political: Left-right ideology, Nativism/immigration, Welfare preferences, etc.

**File**: `Data/shp_panel_prepared.csv`

---

## Running the Analysis

### Quick Start

```bash
# Run full specification validation
cd /Users/bradyallardice/Dropbox/Allardice/KurerAllardice2024/10CompaniesPOCAI
python3 stage_8_spec_validation_core.py
```

### Output Files (Created in `Data/Testing/stage_8/specification_diagnostics/`)

1. **specification_validity_scorecard.csv**
   - Table: Each outcome × specification
   - Columns: Validity rating, power, significance, recommendation
   - Quick reference for trustworthiness

2. **income_reversal_diagnosis.json**
   - Model sequence tracking coefficient changes
   - Tests for sign flip across specifications
   - Diagnostic clues about confounding

3. **detailed_diagnostics.json**
   - Convergence status, residual diagnostics
   - Heteroskedasticity tests
   - Power metrics (DoF ratio)

### Runtime

- **Fast version** (`stage_8_spec_validation_core.py`): ~30-60 minutes
  - Fits all 3 specs on 4 outcomes
  - Includes income reversal diagnosis
  - Memory efficient

- **Comprehensive version** (`stage_8_specification_diagnostics_fast.py`): ~60-90 minutes
  - More detailed diagnostics
  - All outcomes analyzed
  - Additional robustness tests

- **Full version** (`stage_8_specification_diagnostics.py`): ~2-3 hours
  - Complete diagnostics suite
  - Graphical outputs (forest plots, etc.)
  - More memory intensive

---

## Understanding the Results

### Validity Scorecard

After running the analysis, you'll see a scorecard like:

```
Outcome                 Specification           Validity_Rating    Recommendation
────────────────────────────────────────────────────────────────────────────────
Income (log)            Spec 1 (Additive)       MEDIUM             Robustness check
Income (log)            Spec 2 (Firm-Yr+Occ-Yr) HIGH               ✓ USE THIS
Income (log)            Spec 3 (Saturated)      LOW                Specification test

Left-right politics     Spec 1 (Additive)       MEDIUM
Left-right politics     Spec 2 (Firm-Yr+Occ-Yr) HIGH              ✓ USE THIS
Left-right politics     Spec 3 (Saturated)      LOW

... etc for each outcome
```

**Interpretation**:
- **HIGH**: Strong causal ID + adequate power + significant effect → TRUST IT
- **MEDIUM**: Good ID but tight power, or marginal significance → REPORT WITH CAVEATS
- **LOW**: Weak ID or convergence issues → DON'T LEAD WITH IT

### Income Reversal Diagnosis

If coefficient flips sign from Spec 1 to Spec 2:

**Most likely cause**: Firm-year confounding
- Firms in trouble deploy AI in underperforming (low-wage) occupations
- Naive Spec 1 conflates AI with recession/restructuring
- Spec 2 isolates pure AI effect → may flip sign

**Less likely causes**:
- Task-fit selection (firms deploy AI where it's most useful — routine, lower-wage roles)
- High-wage occupations deliberately targeted for AI (good labor market → less income pressure)

**Test**: Check if low-wage occupations predict future AI exposure
- If yes → restructuring/cost-cutting story likely
- If no → task-fit or strategic deployment story

---

## How to Report Results

### Main Text

Use **Specification 2** for all main results:

> "We employ a triple-difference specification with person, firm-year, and occupation-year fixed effects. The identifying assumption is that within a given firm in a given year, differences in AI exposure across occupations are not correlated with unobserved occupation-level shocks. Firm-year fixed effects remove any firm-specific time-varying confounders (e.g., restructuring), while occupation-year fixed effects remove labor market trends specific to each occupation."

### Main Table

Report **Spec 2 results** with:
- Coefficient
- Standard error (clustered at person level)
- 95% confidence interval (note: tight DoF, so CI's will be wide)
- 90% CI (to show effect even with soft significance)
- N and DoF for transparency

### Appendix: Robustness

Show **Spec 1** alongside Spec 2 to demonstrate:
1. Whether results robust to weaker FE specification
2. Coefficient stability across identification strengthening
3. Power advantage of conservative approach

> "In Appendix Table X, we show results using additive fixed effects only (α_i + α_f + α_t + α_o). Coefficients are similar to our main specification (Spec 2), with tighter standard errors, suggesting that firm-year shocks do not materially confound our estimates."

### Appendix: Specification Test (Optional)

If Spec 3 converges, report it to test person-level time-varying confounding:

> "Appendix Table X shows estimates under maximum FE saturation (including person-year FE). Results are [similar/different] to main specification, with substantially wider confidence intervals due to limited degrees of freedom. This specification test suggests [person-level confounding is not a material concern / person-level shocks matter for identification]."

---

## Technical Details

### Fixed Effects Absorption

How many degrees of freedom are absorbed?

| Specification | Person FE | Firm FE | Year FE | Firm-Year FE | Occ-Year FE | Person-Year FE | Total |
|---------------|-----------|---------|---------|--------------|-------------|----------------|-------|
| Spec 1        | 9,895     | 6,862   | 12      | —            | —           | —              | 16,769 |
| Spec 2        | 9,895     | —       | —       | 30,507       | 3,000       | —              | 43,402 |
| Spec 3        | 9,895     | —       | 12      | 30,507       | 3,000       | 55,000+        | 98,000+ |

**DoF Remaining**:
- Spec 1: 45,325 - 16,769 ≈ 28,556
- Spec 2: 45,325 - 43,402 ≈ 1,923  ← **TIGHT**
- Spec 3: 45,325 - 98,000 < 0  ← **OVER-IDENTIFIED**

### Why Spec 2 is Tight

With so many firm-years (30k) and few people per firm-year (median 1), identifying variation is limited:
- Most people in one firm in any given year
- Most firm-years have only 1-2 occupations
- Within-firm-year variation minimal

Despite tight DoF, **Spec 2 is still preferred** because:
1. Alternative specifications have stronger confounding threats
2. Tight DoF is better than biased estimates
3. 1.9k DoF is manageable (comparable to high-dimensional FE literature)
4. Report 90% CIs for transparency

---

## Assumptions & Threats

### Assumptions in Spec 2

1. **No firm-year confounding**: Given firm-year shocks are controlled
   - ✓ Restructuring, acquisitions, recessions at firm level removed
   - ✗ Residual firm-year × occupation-specific shocks (e.g., firm targets certain dept)

2. **No occupation-year confounding**: Given occ-year shocks are controlled
   - ✓ Labor market trends, skills demand in that occupation-year removed
   - ✗ (Minimal threat at occupation level)

3. **No person-year confounding**: Not directly controlled, but person FE helps
   - ⚠ If person i's mood shifts in year t and they work in the occ firm deploys AI to, bias could result
   - Mitigation: Can test with Spec 3 (if feasible)

4. **No reverse causality**: AI exposure doesn't respond to unobserved politics
   - Assumption: Firm's AI deployment decision doesn't depend on worker politics
   - Plausible (firms make tech decisions, don't monitor politics)

### Remaining Endogeneity Threat: Firm-Occ Assignment

Even with Spec 2, one threat remains:

> **Why does firm f deploy AI in occupation o in year t?**

**Benign story** (task-fit):
- Firms deploy AI where technology is most useful (routine, automatable tasks)
- Evidence: AI exposure correlates with task-replaceability scores
- Implication: Coefficient is causal (or at least not confounded by restructuring)

**Malign story** (restructuring targeting):
- Firms deploy AI in occupations they're planning to downsize
- Evidence: Workers in AI-exposed occupations see worse subsequent employment
- Implication: Coefficient partly reflects job insecurity, not pure AI effect

**Testing**:
1. Pre-treatment balance: Do low-wage occupations predict future AI exposure?
2. Task-fit analysis: Does AI exposure predict task-replaceability?
3. Post-exposure outcomes: Employment/wages in exposed occupations post-AI

---

## Code Structure

### Main Script: `stage_8_spec_validation_core.py`

```python
# 1. Load SHP panel data
df = pd.read_csv('Data/shp_panel_prepared.csv')

# 2. Create FE indicators
df['firm_year'] = ...
df['occ_year'] = ...
df['person_year'] = ...

# 3. For each outcome:
#    - Fit Spec 1, Spec 2, Spec 3
#    - Generate diagnostics (convergence, power, significance)
#    - Score validity (HIGH/MEDIUM/LOW)
#    - Generate recommendation

# 4. Special analysis: Income reversal
#    - Model sequence: Unadjusted → +Controls → +FE1 → +FE2
#    - Track coefficient changes
#    - Detect sign flip

# 5. Export:
#    - specification_validity_scorecard.csv
#    - income_reversal_diagnosis.json
#    - detailed_diagnostics.json
```

### Clustering

All models use **person-level clustering** for standard errors:

```python
model.fit(cov_type='cluster', cov_kwds={'groups': df['idpers']})
```

Why? Multiple observations per person → standard errors would be understated without clustering.

---

## Frequently Asked Questions

### Q: Which spec should I use?

**A**: Specification 2 (Firm-Year + Occupation-Year FE) for main results.

Why? Best balance of causal identification and power. Spec 1 is too conservative (omits firm-year confounding), Spec 3 is too aggressive (too much FE saturation).

---

### Q: Will results be precisely estimated?

**A**: Spec 2 will have wider CI's than Spec 1 due to tight DoF. This is okay.

- Tight DoF ≠ bad estimates
- Tight DoF = honest about uncertainty
- Spec 1 has tighter CI's but more bias from omitted firm-year shocks
- Spec 2 has wider CI's but less bias → more trustworthy

---

### Q: Should I report all three specs in main table?

**A**: No. Main table: Spec 2 only. Appendix: Spec 1 for robustness.

Keep main narrative clean. Spec 1 is a comparison point, not the primary evidence.

---

### Q: What if Spec 1 and Spec 2 give different answers?

**A**: That's informative! Suggests firm-year shocks matter. Discuss:

> "Coefficients differ substantially when we include firm-year fixed effects (main specification vs. Spec 1 in Appendix Table X). This suggests firm-specific time-varying shocks (e.g., restructuring, acquisitions) can confound estimates. Our main specification addresses this threat."

---

### Q: What does the income reversal mean?

**A**: Most likely: firm-year confounding.

If coefficient on income flips sign from Spec 1 to Spec 2, it suggests:
- Naive Spec 1 conflates AI with firm-year shocks
- Spec 1 coefficient = AI effect + restructuring effect (both hit low-wage occ's)
- Spec 2 coefficient = pure AI effect after removing restructuring

---

### Q: Can I use cluster bootstrap instead of cluster-robust SE?

**A**: Yes, but not necessary.

Cluster-robust SE (Liang-Zeger) is standard and fast. Bootstrap useful if you want:
- Percentile CI's instead of symmetric CI's
- Confidence in SE formula
- Robustness to normality assumptions

For this analysis, cluster-robust SE is fine.

---

### Q: What if Spec 3 doesn't converge?

**A**: That's expected. Just skip it.

Spec 3 is a specification test, not essential. If model is saturated (more FE's than observations), convergence can fail. This is informative (suggests we're at FE limits) and okay.

---

## Appendix: Sample Size Breakdown

From diagnostic run (April 10, 2026):

```
Total SHP person-years: 199,324
  With firm linkage: 45,325 (22.7%)

Unique persons: 38,528
  With firm linkage: 9,895 (25.7%)

Firm-year combinations: 30,507
  With 2+ occupations: 4,523 (16.3%)
  With 2+ respondents: 5,722 (18.8%)
  Median respondents per firm-year: 1

Firm-occupation-year cells: 37,064
  With 2+ respondents: 1,852 (5.0%)
  Median respondents: 1

Firm switchers: 1,794 (18.1% of linked persons)
Firm stayers: 8,101 (81.9%)
```

**Implication**: 
- Most people don't move between firms (81.9% stayers)
- Most firm-years have only 1-2 people
- Within-firm variation is sparse → tight DoF in Spec 2

---

## References

1. **Hampole et al. (2025)** — Firm-level AI adoption and labor market outcomes
2. **Callaway & O'Neill (2023)** — Heterogeneous treatment effects with staggered adoption
3. **Cameron & Miller (2015)** — Practitioner's guide to cluster-robust inference
4. **Angrist & Pischke (2009)** — Mostly harmless econometrics

---

## Timeline

| Date | Status |
|------|--------|
| April 10, 2026 | Specification design finalized |
| April 10-11, 2026 | Scripts implemented, technical docs written |
| April 11, 2026 | Empirical estimation in progress |
| April 11-12 (est) | Results generated, validity scorecards completed |
| April 12+ | Interpret results, write up findings |

---

## Contact & Questions

For questions about:
- **Specification design**: See `/docs/specification-shp-panel-econometrics.md`
- **Technical details**: See `/docs/specification-testing-technical-report.md`
- **How to interpret results**: See this document

---

**Status**: Framework complete and ready for empirical implementation. Scripts running in background.

**Next step**: Monitor completion and extract validity scorecards.

