# Identification Validation Roadmap
## Critical Tests to Validate Causal Interpretations

**Date**: April 10, 2026  
**Purpose**: Explicit validation checklist for income reversal analysis  
**Status**: Pre-validation (tests not yet executed)

---

## Part 1: Validating Firm-Year FE Wage Effects

The FY-FE wage loss (-8.5%) is our primary causal estimate for worker impact. But it rests on two identifying assumptions:

1. **Firm × Occupation assignment is task-driven (benign), not targeting-driven (malign)**
2. **Person-level time-varying shocks don't confound the effect**

Both must be validated before claiming causal inference.

---

### Test 1A: Task-Fit Analysis (CRITICAL)

**What we're testing**: Does AI exposure predict O*NET task replaceability?

**The benign story**: 
- Firms deploy AI in occupations with high routine-task intensity
- Assignment is driven by task structure, not by targeting workers for layoffs
- Causal interpretation of FY-FE is credible

**The malign story**:
- Firms deploy AI in occupations they want to downsize
- Assignment is endogenous to firm restructuring plans
- FY-FE coefficient conflates AI exposure with employment risk

**How to test**:

```python
# Merge stage_5 exposure with O*NET task characteristics
exposure = pd.read_csv("Data/isco_firm_occupation_year_exposure.csv")
onet_tasks = pd.read_csv("Data/onet_task_characteristics.csv")  # routine, cognitive, manual, etc.

# Regression 1: Do AI-exposed occupations have higher routine-task intensity?
result1 = sm.ols(
    "hampole_ai_exposure_avg ~ routine_task_intensity + cognitive_intensity + manual_intensity",
    data=merged_data
).fit()

# Expected: positive coefficient on routine_task_intensity (AI goes to routine occupations)
# Interpretation: If strong positive, supports benign assignment
```

**Success criteria**:
- [ ] Coefficient on routine-task intensity: Positive and statistically significant (p < 0.05)
- [ ] Magnitude: Economically meaningful (not trivial)
- [ ] R-squared: Reasonable fit (>0.20 suggests task structure explains substantial variation)
- [ ] Other task types: Negative or small coefficients (AI not going to complex tasks)

**If test passes**: "AI deployment is task-driven, supporting benign assignment hypothesis"

**If test fails** (negative/null coefficient on routine tasks):
- Malign story may be plausible
- FY-FE wage loss could reflect restructuring, not pure AI effect
- Add robustness check: "Wage loss among occupations with high routine-task intensity vs. low"
- Interpretation becomes: "AI-coincident wage loss" not "purely causal AI effect"

---

### Test 1B: Pre-Treatment Balance (IMPORTANT)

**What we're testing**: Do pre-AI occupational characteristics predict whether firms deploy AI in them?

**The benign story**:
- Firm chooses occupation for AI based on 2020+ task structure
- Pre-2017 occupation characteristics are *not* predictive of 2020+ AI deployment
- Assignment is not selecting based on pre-existing occupation health

**The malign story**:
- Firm targets occupations that were already struggling (low wages, declining employment, low growth)
- Pre-2017 characteristics *are* predictive of 2020+ AI deployment
- Assignment is endogenous to occupational trajectory

**How to test**:

```python
# Create pre-treatment (2012-2016) occupation characteristics
pre_treatment = shp_data[shp_data['year'] < 2017].groupby('isco08_4d').agg({
    'wage': 'median',
    'employment_rate': 'mean',
    'employment_change': 'mean'  # growth rate
})

# Merge with 2020+ AI exposure
exposure_2020_plus = exposure[exposure['year'] >= 2020]
merged_balance = pre_treatment.merge(exposure_2020_plus, on='isco08_4d')

# Regression: Does pre-treatment occupation health predict post-treatment exposure?
result_balance = sm.ols(
    "hampole_ai_exposure_avg ~ wage_2012_16 + employment_rate_2012_16 + employment_growth_2012_16",
    data=merged_balance
).fit()

# Expected: no relationship (coefficients ~0 and not significant)
# If negative relationships: firms targeting low-wage/declining occupations
```

**Success criteria**:
- [ ] Coefficient on pre-treatment wage: Not significantly different from zero (p > 0.10)
- [ ] Coefficient on pre-treatment employment rate: Not significantly negative
- [ ] Coefficient on pre-treatment employment growth: Not significantly negative
- [ ] Joint F-test: Cannot reject that all coefficients = 0 (p > 0.20)

**If test passes**: "Firms are not targeting occupations for AI deployment based on pre-existing health; assignment is likely benign"

**If test fails** (negative relationship on pre-treatment wages):
- Suggests malign targeting: "Firms deploy AI in low-wage occupations"
- FY-FE wage loss could be selecting for already-vulnerable occupations
- Robustness check: "Control for pre-treatment wage level in main regression"
- Interpretation: "Wage loss concentrated in already-vulnerable occupations"

---

### Test 1C: Post-Treatment Employment Outcomes (CONFIRMATORY)

**What we're testing**: Do AI-exposed occupations experience employment disruption?

**The benign story**:
- AI augments tasks, reallocates workers to higher-value work
- No increase in separations or hiring freezes
- Wage loss reflects task recomposition, not employment loss

**The malign story**:
- AI is cover for restructuring, leads to layoffs
- Higher separations in AI-exposed occupations
- Wage loss reflects employment risk, not pure task effects

**How to test**:

```python
# Merge exposure with employment outcomes
outcomes = shp_data.groupby(['isco08_4d', 'year']).agg({
    'employment': 'count',  # number employed in occupation
    'separation': 'sum',    # workers who left occupation
    'hiring': 'sum'         # new workers entering
})

merged_outcomes = exposure.merge(outcomes, on=['isco08_4d', 'year'])

# Regression: Does higher AI exposure predict separations?
result_sep = sm.ols(
    "separation_rate ~ hampole_ai_exposure_avg + year + isco08_4d",
    data=merged_outcomes
).fit()

# Expected: coefficient ~0 (no effect) or small positive
# If strongly positive: suggests employment disruption
```

**Success criteria**:
- [ ] Coefficient on exposure: Positive but small (0.05-0.15 on separation rate)
- [ ] P-value: Not significant (p > 0.10)
- [ ] Hiring rate: Not significantly reduced by exposure
- [ ] Alternative interpretation: Task reallocation, not employment loss

**If test passes**: "No evidence of AI-driven employment disruption; wage loss reflects task shifts, not layoffs"

**If test fails** (high coefficient on separations):
- Suggests employment risk accompanies AI deployment
- Wage loss may reflect "insurance loss" (employment risk) not pure task effect
- Robustness check: "Wage loss larger in high-separation occupations"

---

## Part 2: Validating Firm-Year FE Person-Level Confounding (Spec 3)

The second key assumption for FY-FE is that person × year shocks don't confound. Spec 3 tests this.

---

### Test 2A: Spec 3 Robustness Check (IMPORTANT)

**What we're testing**: Does wage effect survive person-year FE specification?

**The expected result**:
- FY-FE effect: -8.5% (main estimate)
- Spec 3 effect: -6% to -8% (modest attenuation, still significant)
- If Spec 3 effect disappears: person-level confounding is major

**How to run**:

```python
# Load panel data
df = pd.read_csv("Data/shp_panel_prepared.csv")

# Create person-year identifier
df['person_year'] = df['idpers'].astype(str) + '_' + df['year'].astype(str)

# Spec 3: Firm-Year FE + Occupation-Year FE + Person-Year FE + Person FE
import statsmodels.formula.api as smf

spec_3_model = smf.ols(
    "log_income ~ C(idpers) + C(firm_year) + C(occ_year) + C(person_year) + "
    "hampole_ai_exposure_avg + age + gender + employment_status",
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['idpers']})

# Extract coefficient
beta_spec3 = spec_3_model.params['hampole_ai_exposure_avg']
se_spec3 = spec_3_model.bse['hampole_ai_exposure_avg']
pval_spec3 = spec_3_model.pvalues['hampole_ai_exposure_avg']
```

**Success criteria**:
- [ ] Coefficient negative: β < 0 (same direction as FY-FE)
- [ ] Reasonable attenuation: -6% to -8% (not completely different)
- [ ] Still significant at 10%: p < 0.10 (passes robustness)
- [ ] Standard error: Not implausibly large (would indicate collinearity)
- [ ] Model converges: No singularity or numerical issues

**If test passes**: 
- "Effect is robust to person-level time-varying confounding"
- "Causal interpretation of FY-FE wage loss is credible"

**If test fails**:
- Effect disappears (β ≈ 0): "Person-level shocks confound FY-FE; effect not causal"
- Model doesn't converge: "Too few degrees of freedom; FY-FE is appropriate level of granularity"
- SE becomes huge: "Can't separate person-year shocks from exposure; power too low"

---

### Test 2B: Placebo Test (EXPLORATORY)

**What we're testing**: Does pre-treatment outcome variable predict exposure?

**The logic**: If exposure is truly exogenous, pre-treatment values should not be predictive. If they are, selection is present.

**How to test**:

```python
# Use 2012-2016 wages as pre-treatment baseline
pre_2017 = df[df['year'] < 2017].groupby(['idpers', 'isco08_4d']).agg({'log_income': 'mean'})

# Merge with 2017+ exposure
post_2016 = df[df['year'] >= 2017].drop_duplicates(['idpers', 'year'])
merged_placebo = pre_2017.merge(post_2016, on=['idpers', 'isco08_4d'])

# Regression: Does pre-treatment wage predict exposure?
result_placebo = sm.ols(
    "hampole_ai_exposure_avg ~ log_income + year",
    data=merged_placebo
).fit()

# Expected: coefficient on log_income ~0 and not significant
# If positive: high-wage workers get more AI (selection into high-skill)
# If negative: low-wage workers get more AI (targeting vulnerable)
```

**Success criteria**:
- [ ] Coefficient on pre-treatment wage: Not significantly different from zero (p > 0.10)
- [ ] If significant: Interpret direction (who's selected into AI?)
- [ ] Magnitude: Small compared to treatment effect

**If test passes**: "No pre-treatment selection visible; exogeneity assumption more plausible"

**If test fails**: Document selection pattern and adjust interpretation accordingly

---

## Part 3: Validating Occupation-Year FE Political Effects

The occupation-year FE specification for political outcomes rests on the assumption that firm-level selection doesn't bias the political effect.

---

### Test 3A: Within-Firm Political Specification (IMPORTANT)

**What we're testing**: Do political effects survive firm-year FE specification?

**The benign story** (OY-FE is appropriate):
- Political ideology responds to occupation-level labor market vulnerability
- Firm characteristics (wage level, industry) are less relevant for political response
- OY-FE political effect survives FY-FE specification (minimal change)

**The malign story** (firm selection confounds):
- Political shift reflects benefits of working at high-wage firms
- FY-FE political effect is much smaller or zero
- OY-FE coefficient is biased by firm-level selection

**How to test**:

```python
# Fit political outcome with Firm-Year FE (FY-FE) logic
# Instead of Occupation-Year FE

fy_fe_politics = smf.ols(
    "left_right_placement ~ C(idpers) + C(firm_year) + "
    "hampole_ai_exposure_avg + age + gender + employment_status",
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['idpers']})

# Compare with existing OY-FE results
# β_oy_fe = -0.252 (p = 0.067†)

beta_fy_fe_politics = fy_fe_politics.params['hampole_ai_exposure_avg']
```

**Success criteria**:
- [ ] Coefficient negative: β < 0 (same direction as OY-FE)
- [ ] Similar magnitude: -0.15 to -0.30 (within ~30% of OY-FE)
- [ ] Statistical significance: p < 0.15 (still reasonably detected)
- [ ] If OY-FE & FY-FE agree: Firm selection is not major confound

**If test passes**: "Political effect is robust to specification; occupation-level mechanism confirmed"

**If test fails** (FY-FE effect disappears or flips sign):
- "Political response may reflect firm benefits, not occupational exposure"
- "OY-FE result should be interpreted with caution; firm selection may bias"
- "Use FY-FE for final political estimates if OY-FE effect disappears"

---

### Test 3B: Occupational Heterogeneity (EXPLORATORY)

**What we're testing**: Do political effects differ by occupation task structure?

**The logic**: If political response is to occupational vulnerability, it should be stronger in routine occupations more vulnerable to automation.

**How to test**:

```python
# Merge exposure with O*NET task characteristics
merged_het = df.merge(onet_tasks, on='isco08_4d')

# Interaction: AI exposure × routine-task intensity
result_het = smf.ols(
    "left_right_placement ~ C(idpers) + C(occ_year) + "
    "hampole_ai_exposure_avg * routine_task_intensity + "
    "age + gender + employment_status",
    data=merged_het
).fit(cov_type='cluster', cov_kwds={'groups': merged_het['idpers']})

# Expected: positive interaction (stronger political response in routine occupations)
```

**Success criteria**:
- [ ] Main effect (AI exposure): Negative (leftward shift)
- [ ] Interaction: Positive and significant (effect stronger for routine occupations)
- [ ] Interpretation: "Workers in automatable occupations respond more strongly"

**If test passes**: "Political response is calibrated to occupational automation risk"

**If test fails**: "Political response is general, not occupation-specific"

---

## Part 4: Priority and Sequencing

### Tier 1 (Must Do Before Finalizing Causal Claims)

1. **Test 1A: Task-fit analysis** (Does exposure predict routine tasks?)
   - Time estimate: 2-4 hours
   - Impact: Validates benign assignment hypothesis
   - Status: **CRITICAL MISSING VALIDATION**

2. **Test 2A: Spec 3 robustness** (Does wage effect survive person-year FE?)
   - Time estimate: 3-5 hours (computation + debugging)
   - Impact: Validates person-level confounding is not major
   - Status: **CRITICAL MISSING VALIDATION**

### Tier 2 (Should Do Before Publication)

3. **Test 1B: Pre-treatment balance** (Do pre-2017 characteristics predict 2020+ exposure?)
   - Time estimate: 2-3 hours
   - Impact: Confirms assignment is not based on pre-existing occupation health
   - Status: **IMPORTANT CONFIRMATION**

4. **Test 3A: Within-firm political spec** (Do politics survive FY-FE?)
   - Time estimate: 1-2 hours
   - Impact: Validates that occupation-level mechanism is appropriate for politics
   - Status: **IMPORTANT CONFIRMATION**

### Tier 3 (Nice to Have, Exploratory)

5. **Test 1C: Post-treatment employment** (Higher separations in exposed occupations?)
6. **Test 2B: Placebo test** (Does pre-treatment outcome predict exposure?)
7. **Test 3B: Occupational heterogeneity** (Stronger effect in routine occupations?)

---

## Part 5: Interpretation Framework

Once validations are complete, use this framework to interpret results:

### If All Tier 1 Tests Pass

**Causal interpretation is strong:**
- "AI exposure causes 8.5% wage loss within firms"
- "Effect is robust to person-level confounding (Spec 3)"
- "Assignment is task-driven, not targeting-driven (task-fit test)"
- "Political shift reflects occupation-level vulnerability (OY-FE appropriate)"

**Policy language:**
- "Workers bear real economic costs from AI adoption"
- "Vulnerable populations need direct income support"
- "Political mobilization is endogenous response to threat perception"

### If Task-Fit Test Fails (Exposure ≠ Routine Tasks)

**Interpretation weakens:**
- "AI exposure may reflect restructuring (malign assignment)"
- "Wage loss could confound causal effect with employment risk"
- "Conservative interpretation: wage loss among occupations targeted for restructuring"

**Robustness required:**
- Subset analysis: "Wage loss larger in high-routine-task occupations" (should be true if benign)
- Add controls: Include pre-treatment occupation health in main regression

### If Spec 3 Test Fails (Effect Disappears)

**Major interpretation problem:**
- "Person-level time-varying shocks confound FY-FE estimate"
- "Cannot claim causal interpretation of -8.5%"
- "Effect may reflect person-year specific events (life events, mood)"

**Alternative estimate:**
- If Spec 3 still shows negative effect (just noisier): Use Spec 3 as primary, report FY-FE as supplementary
- If Spec 3 is zero: Report OY-FE as only viable estimate (acknowledge selection bias)

### If Pre-Treatment Balance Fails (Low Wages Predict Exposure)

**Interpretation adds nuance:**
- "Firms deploy AI in lower-wage occupations"
- "Wage loss may partly reflect targeting of already-vulnerable occupations"
- "Effect is concentrated among populations with fewer opportunities"

**Robustness required:**
- Control for pre-treatment wage in main regression
- Subset analysis by pre-treatment occupation health
- Focus policy language on vulnerability of affected workers

---

## Summary Table: Tests × Interpretation

| Test | Status | Interpretation | Action |
|---|---|---|---|
| Task-fit | If PASS | Benign assignment | Proceed with causal language |
| Task-fit | If FAIL | Possible targeting | Add pre-treatment controls |
| Spec 3 | If PASS | Robust to person confounds | FY-FE is primary estimate |
| Spec 3 | If FAIL | Person shocks matter | Use Spec 3 as primary |
| Pre-balance | If PASS | Not selecting vulnerable | Clean assignment |
| Pre-balance | If FAIL | Targeting low-wage occ | Document selection pattern |
| Within-firm politics | If PASS | Occupation mechanism | OY-FE appropriate for politics |
| Within-firm politics | If FAIL | Firm effects matter | Reconsider specification |

---

## Checklist Before Publication

### Validations Required
- [ ] Task-fit analysis complete
- [ ] Spec 3 estimates produced
- [ ] Pre-treatment balance tested
- [ ] Within-firm political specification run

### Documentation Required
- [ ] Interpretation narrative (benign vs. malign assignment)
- [ ] Robustness evidence (all four tests)
- [ ] Limitation discussion (remaining threats to identification)
- [ ] Sample and specification diagnostic table

### Reporting Checklist
- [ ] Main results: FY-FE wages, OY-FE politics
- [ ] Robustness: Spec 1, Spec 3 (if converges)
- [ ] Heterogeneity: Gender, age, education subgroups
- [ ] Validation evidence: Task-fit, balance, post-treatment outcomes
- [ ] Appendix: Full specification, model diagnostics, sensitivity tests

---

**Status**: Validation roadmap complete. Ready to execute tests in priority order.  
**Next Steps**: Run Tier 1 tests (task-fit, Spec 3) before finalizing causal claims for publication.
