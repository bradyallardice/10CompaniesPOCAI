# STRESS TEST PRIORITY MATRIX
**April 10, 2026**

## Quick Reference: Which Tests to Run First

| Priority | Test Name | Threat It Addresses | Effort | Time | Impact on Credibility |
|----------|-----------|-------------------|--------|------|---------------------|
| **🔴 CRITICAL** | Lead-lag specification | Reverse causality, measurement error, unobserved trends | 20 lines code | 30 min | HIGH: Pass/fail for causal inference |
| **🔴 CRITICAL** | Gender interaction formal test | Statistical significance of gender difference | 10 lines code | 15 min | HIGH: Main finding validity |
| **🟠 HIGH** | Task-fit regression | Benign vs. malign firm-occ assignment | 30 lines code | 45 min | HIGH: Endogeneity magnitude |
| **🟠 HIGH** | Pre-treatment balance | Selection on unobservables concern | 20 lines code | 30 min | MEDIUM: Supports benign story |
| **🟠 HIGH** | COVID exclusion | Confounding from pandemic shock | 10 lines code | 20 min | MEDIUM: Robustness |
| **🟡 MEDIUM** | Occupational plausibility | Measurement validity check | 15 lines code | 30 min | MEDIUM: Face validity |
| **🟡 MEDIUM** | Percentile sensitivity (Stage 4) | Measurement threshold fragility | Requires Stage 4 rerun | 2-4 hours | MEDIUM: Measurement robustness |
| **🟡 MEDIUM** | Sample representativeness | Selection bias magnitude | 30 lines code + data | 1 hour | MEDIUM: Generalizability |
| **🔵 LOW** | Alternative exposure measure (occupational level) | Specification sensitivity | 1-2 hours | 1-2 hours | LOW: Secondary robustness |
| **🔵 LOW** | Manual validation sample | Ground truth measurement | 10-20 hours | 1-2 weeks | HIGH: But expensive; do if publishing |

---

## WHAT TO DO THIS WEEK (Priority 1-2)

### Today (30 minutes)
1. **Lead-lag specification**
   ```python
   # Add to stage_8b_robustness_checks.py
   # Y_it = β_{-2}*Exp_{it+2} + β_{-1}*Exp_{it+1} + β_0*Exp_it + 
   #        β_1*Exp_{it-1} + β_2*Exp_{it-2} + FE + ε
   # Report all coefficients with 95% CI
   # PASS: β_{-2}, β_{-1} not significant
   # FAIL: β_{-2}, β_{-1} significant → reverse causality problem
   ```

2. **Gender interaction test**
   ```python
   # Y = α_i + α_ft + α_ot + β₁*Exposure + β₂*Female + β₃*Exposure*Female + ε
   # Test: β₃ = 0?
   # PASS: β₃ significant → gender difference is real, not sampling variance
   # FAIL: β₃ not significant → differences might be noise
   ```

### This Week (2-4 hours cumulative)
3. **Task-fit evidence**
   - Regress AI exposure on O*NET routine-task intensity
   - Should find: positive correlation if benign assignment
   - Should find: no pre-treatment correlation if not targeting troubled occupations

4. **COVID robustness** (20 minutes)
   - Exclude 2020-2021, re-run main spec
   - Compare coefficient magnitude to full-sample result

5. **Occupational plausibility** (30 minutes)
   - Cross-tab mean AI exposure by occupation
   - Inspect: Are software developers/data scientists high? Janitors/retail low?

---

## INTERPRETATION GUIDE

### If Lead-Lag Test FAILS (β_{-2}, β_{-1} significant)

**What it means**: Future exposure predicts current outcomes → major problem

**Action**: 
- Revise causal claim to "correlational"
- Investigate measurement error (are job ads dated incorrectly?)
- Check for unobserved trends correlated with both exposure and outcomes

**Publication**: Demote results to suggestive/correlational; acknowledge reverse causality threat

---

### If Gender Interaction Test FAILS (β₃ not significant)

**What it means**: Gender difference might be sampling variation

**Action**:
- Report descriptively: "Males show -0.43 [±0.22], females show -0.01 [±0.18]"
- Note: Large overlapping CIs suggest differences could be noise
- Don't over-interpret gender heterogeneity

**Publication**: Weaken gender difference claim; focus on main average effects

---

### If Task-Fit Regression FAILS (AI exposure negatively correlated with routine-task intensity)

**What it means**: Firm AI deployment doesn't follow task structure → likely targeting (malign story)

**Action**:
- Investigate: Which occupations get high AI exposure despite low routine intensity?
- Possible explanation: Firms targeting high-wage occupations for cost reduction
- Interpret wage effects with caution: could reflect restructuring, not pure AI effect

**Publication**: Acknowledge endogeneity threat; note 30-40% of wage effect could be selection

---

### If COVID Exclusion FAILS (coefficient drops >30%)

**What it means**: COVID confounding is substantial

**Action**:
- Pandemic created widespread anxiety and employment uncertainty
- Difficult to isolate AI effect from COVID shock
- Consider pre-2020 results as more credible

**Publication**: Report both pre-COVID and full-sample results; note robustness caveat

---

## DECISION TREE FOR YOUR NEXT STEPS

```
START: Do you want to publish these results?
  │
  ├─→ YES, top journal (AER, Econometrica, JHR)
  │   └─→ Run all CRITICAL + HIGH priority tests
  │       └─→ Invest in manual validation sample (1-2 weeks)
  │       └─→ Lead-lag must PASS
  │       └─→ Task-fit evidence must be collected
  │       └─→ Target: 8-10 weeks to robust publication
  │
  ├─→ YES, field journal (Labor, Demography, Polit. Behavior)
  │   └─→ Run all CRITICAL priority tests (1-2 weeks)
  │       └─→ Lead-lag must PASS
  │       └─→ Gender interaction must PASS
  │       └─→ Honest limitations section required
  │       └─→ Target: 3-4 weeks to submission
  │
  ├─→ YES, working paper / preprint only
  │   └─→ Run CRITICAL tests (1 week)
  │       └─→ Lead-lag specification minimum
  │       └─→ Clear caveats in abstract
  │       └─→ Label "preliminary" or "suggestive"
  │       └─→ Target: 1-2 weeks to posting
  │
  └─→ NO, internal/stakeholder report only
      └─→ Document what you have
          └─→ Add honest limitations section
          └─→ Don't make strong causal claims
          └─→ Ready now
```

---

## SPECIFIC REGRESSION SPECIFICATIONS TO ADD

### Test 1: Lead-Lag (Paste into stage_8b_robustness_checks.py)

```python
def run_lead_lag_specification(df, outcome_var='outcome_left_right'):
    """
    Test for reverse causality / unobserved trends
    Y_it = β_{-2}*Exp_{it+2} + β_{-1}*Exp_{it+1} + β_0*Exp_it + 
           β_1*Exp_{it-1} + β_2*Exp_{it-2} + person_FE + firm_year_FE + occ_year_FE + ε
    """
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    
    df = df.sort_values(['idpers', 'year']).copy()
    
    # Create leads and lags
    df['exp_lead2'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(-2)
    df['exp_lead1'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(-1)
    df['exp_lag1'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(1)
    df['exp_lag2'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(2)
    
    # De-mean by firm_year
    cols_to_dm = ['hampole_ai_exposure_avg_foy', 'exp_lead2', 'exp_lead1', 
                  'exp_lag1', 'exp_lag2', outcome_var]
    for col in cols_to_dm:
        df[col + '_dm'] = df.groupby('firm_year')[col].transform(lambda x: x - x.mean())
    
    # De-mean by person
    for col in cols_to_dm:
        df[col + '_dm_dm'] = df.groupby('idpers')[col + '_dm'].transform(lambda x: x - x.mean())
    
    # Fit model
    data_clean = df[[c for c in df.columns if '_dm_dm' in c or c == 'idpers']].dropna()
    
    y = data_clean[outcome_var + '_dm_dm']
    X = data_clean[['hampole_ai_exposure_avg_foy_dm_dm', 'exp_lead2_dm_dm', 'exp_lead1_dm_dm',
                    'exp_lag1_dm_dm', 'exp_lag2_dm_dm']]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_clean['idpers']})
    
    # Extract results
    results = {
        'lag2': (model.params['exp_lag2_dm_dm'], model.pvalues['exp_lag2_dm_dm']),
        'lag1': (model.params['exp_lag1_dm_dm'], model.pvalues['exp_lag1_dm_dm']),
        'current': (model.params['hampole_ai_exposure_avg_foy_dm_dm'], 
                   model.pvalues['hampole_ai_exposure_avg_foy_dm_dm']),
        'lead1': (model.params['exp_lead1_dm_dm'], model.pvalues['exp_lead1_dm_dm']),
        'lead2': (model.params['exp_lead2_dm_dm'], model.pvalues['exp_lead2_dm_dm'])
    }
    
    # Print results
    print("\nLEAD-LAG SPECIFICATION RESULTS")
    print(f"Outcome: {outcome_var}")
    print(f"Sample size: {len(y)}")
    print("\nCoefficient (p-value):")
    for lag, (coef, pval) in results.items():
        sig = "*" if pval < 0.05 else ""
        print(f"  {lag:8s}: {coef:8.4f} ({pval:.3f}) {sig}")
    
    print("\nINTERPRETATION:")
    print("  PASS: lead1, lead2 NOT significant (p > 0.05)")
    print("  FAIL: lead1, lead2 significant → reverse causality / unobserved trends")
    
    return results, model

# Call it:
# lead_lag_results = run_lead_lag_specification(df, outcome_var='outcome_left_right')
```

### Test 2: Gender Interaction

```python
def test_gender_interaction(df, outcome_var='outcome_left_right'):
    """
    Formal test: Y = α + β₁*Exp + β₂*Female + β₃*Exp*Female + FE + ε
    H0: β₃ = 0 (no gender difference)
    """
    df = df.copy()
    df['exp_x_female'] = df['hampole_ai_exposure_avg_foy'] * df['female']
    
    # De-mean by firm_year
    for col in ['hampole_ai_exposure_avg_foy', 'female', 'exp_x_female', outcome_var]:
        df[col + '_dm'] = df.groupby('firm_year')[col].transform(lambda x: x - x.mean())
    
    # De-mean by person
    for col in ['hampole_ai_exposure_avg_foy', 'female', 'exp_x_female', outcome_var]:
        df[col + '_dm_dm'] = df.groupby('idpers')[col + '_dm'].transform(lambda x: x - x.mean())
    
    # Fit
    data_clean = df[[c for c in df.columns if '_dm_dm' in c or c == 'idpers']].dropna()
    y = data_clean[outcome_var + '_dm_dm']
    X = data_clean[['hampole_ai_exposure_avg_foy_dm_dm', 'female_dm_dm', 'exp_x_female_dm_dm']]
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': data_clean['idpers']})
    
    print("\nGENDER INTERACTION TEST")
    print(f"Outcome: {outcome_var}")
    print(f"\nβ₃ (Exp × Female):")
    print(f"  Coefficient: {model.params['exp_x_female_dm_dm']:.4f}")
    print(f"  SE: {model.bse['exp_x_female_dm_dm']:.4f}")
    print(f"  p-value: {model.pvalues['exp_x_female_dm_dm']:.3f}")
    print(f"  95% CI: [{model.conf_int().loc['exp_x_female_dm_dm', 0]:.4f}, "
          f"{model.conf_int().loc['exp_x_female_dm_dm', 1]:.4f}]")
    
    if model.pvalues['exp_x_female_dm_dm'] < 0.05:
        print("\n✓ PASS: Gender difference is statistically significant (p < 0.05)")
    else:
        print("\n✗ FAIL: Gender difference NOT significant (p ≥ 0.05)")
        print("    → Overlapping CIs suggest difference might be sampling variance")
    
    return model
```

---

## SUCCESS CRITERIA FOR EACH TEST

### Lead-Lag: PASS/FAIL Rubric
- **PASS**: β_{-2} p > 0.05 AND β_{-1} p > 0.05
- **FAIL**: Either β_{-2} OR β_{-1} p < 0.05
- **What to publish**: If PASS, say "Lead-lag specification rules out reverse causality"; if FAIL, say "Results may reflect reverse causality or unobserved trends (see Appendix X)"

### Gender Interaction: PASS/FAIL Rubric
- **PASS**: β₃ (Exp × Female) p < 0.05
- **FAIL**: β₃ p ≥ 0.05
- **What to publish**: If PASS, "Significant gender heterogeneity in effects"; if FAIL, "Gender differences in point estimates but not statistically significant"

### Task-Fit: PASS/FAIL Rubric
- **PASS**: Correlation(AI_Exposure, Routine_Task_Intensity) > 0 and p < 0.05
- **NEUTRAL**: Correlation positive but p > 0.05
- **FAIL**: Correlation ≤ 0
- **What to publish**: If PASS, "AI deployment aligns with task structure (benign assignment)"; if FAIL, "AI deployment does not correlate with task structure (endogeneity concern)"

---

## EXPECTED RESULTS (Based on Your Findings)

| Test | Expected Outcome | Why | Implication |
|------|-----------------|-----|-------------|
| Lead-lag | PASS (β_lead insignificant) | Effects are contemporaneous, not reverse-causal | Credibility ↑↑ |
| Gender interaction | PASS (β₃ significant) | Gender difference is real, not noise | Credibility ↑ |
| Task-fit regression | PASS (Corr > 0, sig) | Benign assignment story supported | Credibility ↑ |
| COVID exclusion | PASS (coef similar pre-2020) | COVID not major confounder | Credibility ↑ |
| Occupational plausibility | PASS (software eng. >> janitors) | Face validity good | Credibility ↑ |

If all PASS: Credibility moves from MEDIUM → MEDIUM-HIGH
If any FAIL: Credibility drops; need deeper investigation

---

## NEXT STEPS

1. **This week**: Run lead-lag + gender interaction tests (1-2 hours)
   - If both PASS → move to publishing phase
   - If either FAIL → stop and investigate

2. **Next week**: Run task-fit, COVID, occupational plausibility tests (2-3 hours)
   - Build evidence package for endogeneity narrative
   - Document findings in appendix

3. **If results PASS tier 1-2 tests**: You're ready for journal submission with honest limitations section

4. **If results FAIL any CRITICAL test**: Revise causal claims to correlational; investigate root cause

---

**END OF PRIORITY MATRIX**

Prepared: April 10, 2026  
Status: ACTIONABLE
