# Statistical Critique: Political Outcomes Analysis (Stage 8d/8e)
## AI Exposure and Political Preferences in Swiss Household Panel

**Reviewer**: Quantitative economist / econometrician  
**Date**: 2026-04-10  
**Sample**: 45,325 person-year observations  
**Specification**: Person FE + Occupation-Year FE (de-meaned OLS)

---

## SUMMARY JUDGMENT

**Overall Assessment: CONDITIONAL APPROVAL WITH SIGNIFICANT CAVEATS**

**Model specification is technically sound in its execution** but has **critical identification concerns** that limit causal interpretation. The de-meaned fixed effects approach is correctly implemented, and clustering choices are appropriate. However, the inability to rule out selection bias, combined with borderline significance levels and modest effect sizes, means results should be interpreted as **suggestive evidence** rather than causal effects. The analysis is well-executed within its constraints but those constraints are substantial.

---

## 1. MODEL SPECIFICATION

### Technical Implementation: ✓ CORRECT

The specification implements a two-way fixed effects (2FE) model via within-group demeaning:

**Specification**: Y_{pit} = α_i + δ_{ot} + β·Exposure_{ot} + γ·Controls_{pit} + ε_{pit}

Where:
- **α_i** = Person FE (idpers, accounts for time-invariant person heterogeneity)
- **δ_{ot}** = Occupation-Year FE (isco08_4d × year, controls for macro occupational trends)
- **β** = AI exposure effect (primary coefficient of interest)
- **Controls** = Age (centered), gender, employment status
- **Clustering** = By person (accounts for repeated observations)

**Implementation details examined**:
1. ✓ De-meaning procedure is correctly executed (lines 62-77 in stage_8e_all_political_outcomes.py):
   - First demean by occ_year groups
   - Then demean by person
   - Results in time-demeaned and person-demeaned data
   - Maintains consistent sample (good_idx filtering for missing/inf values)

2. ✓ Clustering is appropriate:
   - Clusters by person (idpers)
   - Accounts for serial correlation within person panels
   - Correct implementation of statsmodels cluster covariance

3. ✓ Data quality checks are in place:
   - Filters for NaN, Inf values before estimation
   - Minimum sample size checks (n_obs > 50)
   - Reasonable exclusion criteria

### Specification Concerns: ⚠️ IMPORTANT ISSUES

#### 1. **Selection Bias: The Core Threat to Identification**

**The critical problem**: The exposure measure (Hampole AI exposure at firm-occupation-year level) is **not randomly assigned**. Workers with pre-existing leftist preferences may sort into AI-exposed occupations, or firms may adopt AI in response to worker preferences.

**What the specification controls for**:
- Time-invariant person characteristics (via person FE)
- Macro occupational trends (via occupation-year FE)

**What the specification CANNOT control for**:
- Time-varying selection: Workers with changing political views sorting into/out of AI-exposed positions
- Reverse causality: Political preferences influencing occupational choices
- Firm-level selection: Firms adopting AI in response to workforce composition

**Example threat scenario**:
- Year t: Worker develops leftist views (independent of AI)
- Year t+1: Worker seeks employment in tech sector (coincidentally high-AI exposure)
- Analysis observes: AI exposure → leftward shift
- Truth: Political shift preceded occupational change

**Test that should be run**: Lead/lag analysis
- Does exposure in year t predict politics in year t+1?
- Do lead values of exposure (future exposure) predict current politics?
- If yes, suggests reverse causality or selection

**Verdict**: Selection bias is plausible and cannot be ruled out with current specification.

---

#### 2. **Multicollinearity and Identification**

**Concern**: With both person FE and occupation-year FE, the main identifying variation is:

Y_{pit} - Ȳ_i - Ȳ_{ot} = β(Exposure_{ot} - Ō_i - Ō_{ot}) + (Controls - Controls_bar) + ε

The exposure variation comes primarily from **year-to-year changes within occupation-person cells**.

**Questions about this variation**:
1. **How much variation remains** after both demeaning operations?
   - Need to report: variance of demeaned exposure
   - Without this, hard to assess whether identification is strong or weak

2. **Is the variation exogenous?**
   - Occupation-year FE controls for macro shocks (e.g., industry-wide AI adoption)
   - But doesn't control for firm-specific shocks that might correlate with worker politics
   - Example: Tech firm adopts AI + attracts politically similar workers

**Variance of exposure measure not reported**. This is a critical diagnostic.

---

#### 3. **De-Meaning vs. Fixed Effects: Technical Validity Check**

The de-meaned OLS approach is mathematically equivalent to least squares dummy variable (LSDV) estimation. **This is correct** — there's no specification error here. However:

**Potential issue with degree-of-freedom calculation**:
- Code uses statsmodels OLS without explicitly accounting for dropped dummies
- De-meaned approach implicitly removes FE parameters from model
- Clustering by person should account for this, but worth verifying that standard errors are conservative

**Recommendation**: Compare reported SEs to LSDV regression SEs to validate. The fact that all effects are marginally significant or just crossing threshold (p ≈ 0.067) makes SE calculation critical.

---

### Specification Strengths

✓ **Correctly specified for observational data**:
- Two-way FE is standard for panel data with unit and time heterogeneity
- De-meaned approach is valid and computationally efficient

✓ **Appropriate control inclusion**:
- Age, gender, employment status are demographic controls (not colliders)
- Correctly centered (age_centered) to maintain interpretation

✓ **Clustering choice is defensible**:
- Accounts for multiple observations per person
- May be too conservative if main source of variation is occupational
- But better to be conservative (inflate SE) than anti-conservative

---

## 2. ASSUMPTION VIOLATIONS

### Linearity Assumption

**Status**: Plausible but untested

**Concern**: Political preferences and AI exposure may have nonlinear relationships:
- Threshold effects: No effect until exposure crosses critical level, then rapid shift
- Saturation effects: Effect diminishes at high exposure levels
- Interaction effects: Effect of exposure differs dramatically by gender/age

**Evidence from data**:
- Gender heterogeneity IS present (males: β=-0.43**, females: β=-0.01**)
- This suggests functional form may not be linear across groups
- But interactions are estimated separately, not tested

**Question**: Are the reported coefficients stable across model specifications?
- No robustness to nonlinear specifications reported
- No polynomial terms or splines tested
- No sensitivity to outliers (winsorization)

**Recommendation**: Test linearity by:
1. Fitting polynomial terms (exposure²)
2. Spline estimation in exposure
3. Quantile regression to test effect heterogeneity across the distribution

---

### Homoskedasticity Assumption

**Status**: Likely violated but unclear severity

**Concern**: Political preferences may have heterogeneous variance:
- Workers in unstable occupations (high exposure, high turnover) may have higher residual variance
- Self-reported political preferences may have measurement error that correlates with exposure

**What we observe**:
- Robust standard errors (clustered) account for some heteroskedasticity
- But don't address whether heteroskedasticity is systematic

**Not reported**:
- Tests for heteroskedasticity (Breusch-Pagan, White)
- Variance of residuals by exposure quintiles
- Whether SEs differ meaningfully between robust and standard

**Verdict**: Unknown. Clustering helps but doesn't fully address this.

---

### Serial Correlation

**Status**: Addressed via clustering, but potentially incompletely

**The issue**:
- Person is clustered unit, so obs within person may be correlated
- But clustering only addresses correlation within person at one time
- Not clear if there's year-to-year correlation after FE removal

**Example problem**: If worker has stable unobserved leftism that varies over time, demeaning removes the mean but residuals could still be correlated across years.

**Clustering handles this partially** but not comprehensively. A Durbin-Watson test or residual autocorrelation test would help.

---

### Selection Problems

**Status**: ⚠️ LIKELY PRESENT

Beyond selection into occupations, there's **sample selection at work**:

1. **Firm linkage missingness** (from known-issues.md):
   - Only ~23% of SHP sample has firm linkage
   - Analysis restricted to employed with firm_id
   - Excluded: Self-employed, firm-unlinked employed, unemployed

2. **Occupational coverage**:
   - Only ISCO-08 coded occupations included
   - May oversample large structured occupations, undersample small/informal

3. **Panel dropout**:
   - SHP has differential attrition
   - Workers in unstable occupations (high AI exposure) may selectively exit panel
   - Creates survivorship bias

**Impact on results**:
- Estimates apply only to employed workers in formally structured occupations
- Generalization to "all workers" or "typical worker" is limited
- May overstate effects if AI-exposed workers who remain in panel are self-selected

**Not tested**: Sensitivity of results to inclusion/exclusion of attrition-prone groups.

---

## 3. RESULT INTERPRETATION

### Effect Magnitudes: Appropriate Interpretation?

**Reported for Left-Right Placement (primary finding)**:
- Coefficient: β = -0.252
- SE: 0.137
- P-value: 0.067† (just crossing 10% threshold)
- Scale: 1-10 (where 1=left, 10=right)

**Substantive magnitude claim** (from summary):
> "Shift from center-slightly-right (5.0) to moderate-left (4.75)"  
> "This is 11.6% of a standard deviation (SD=2.18)"

**Evaluation: ✓ CORRECT interpretation, but with qualifications**

Math checks out:
- 0.252 / 2.18 = 11.6% of SD ✓
- On 10-point scale, 0.25-point shift is modest but politically meaningful

**However**:
1. Effect size is **moderate-to-small** by conventional standards
   - Cohen's d = 0.115 (small effect)
   - Explains < 1% of variance in outcome alone

2. **Precision is borderline**:
   - p = 0.067 means 6.7% Type I error rate at stated threshold
   - 95% CI: [-0.252 ± 1.96×0.137] = [-0.521, +0.016]
   - CI includes zero! Effect could be null.

3. **Interpretation should emphasize uncertainty**:
   - Current summary states effect as fact
   - Better phrasing: "We find suggestive evidence (p=0.067) of a 0.25-point leftward shift..."

**Critical omission**: Confidence intervals are calculated in code (lines 102-103) but **NOT reported in results tables or summary**. This is a major transparency gap.

---

### Mediation Analysis: Partial or Complete?

**Finding**: Direct effect (AI → politics controlling for job insecurity) remains large:
- Total effect: -0.252
- Direct effect: -0.248
- Difference: -0.004 (trivial)

**Interpretation given**: Job insecurity is "not the primary mechanism"

**Evaluation: ✓ CORRECT but incomplete**

The interpretation is right: insecurity doesn't fully mediate. But analysis could go further:

1. **Missing counterfactual**: What is the indirect effect coefficient?
   - Need: coef(insecurity) × coef(exposure → insecurity)
   - Partial report given: coef(insecurity) = -0.0372, p=0.1185
   - This seems wrong — negative coefficient on insecurity as predictor of leftward shift?
   - Would expect positive (more insecurity → more leftward)

2. **Interpretation puzzle**:
   - Females show strong insecurity response (β=+0.155, p=0.053)
   - But NO political response (β=-0.012, p=0.958)
   - Males show weak insecurity response (β=+0.070, p=0.227)
   - But STRONG political response (β=-0.430, p=0.013)

   **This pattern contradicts the mediation analysis**. If insecurity mediated the effect, we'd expect:
   - Strong insecurity → strong political response
   - Weak insecurity → weak political response

   Instead we see the **opposite for males and females**. This suggests:
   - Gender is a critical moderator (insecurity affects politics differently by gender)
   - Simple mediation model is misspecified
   - Need interaction: Gender × Insecurity → Politics

**Verdict**: Mediation analysis is correctly executed but interpretation is incomplete. The gender contradiction deserves deeper investigation.

---

### Gender Heterogeneity: What Explains It?

**Findings**:
- Males: -0.430*** (p=0.013) leftward shift
- Females: -0.012 (p=0.958) no shift
- Difference: -0.418 (4.3% of scale)

**Discussion provided** is speculative (occupational sorting, selection, differential threat perception). These are plausible but untested.

**What should be done**:
1. **Formal interaction test**: Estimate β₁(Gender) + β₂(Gender × Exposure) directly
   - Code only estimates separately by subgroup
   - Interaction test would quantify statistical significance of difference

2. **Occupational composition test**:
   - If female AI-exposed workers are in different occupations (tech vs. routine)
   - Can test by: Controlling for occupation dummies (not just FE)
   - Examine distribution of females across occupations

3. **Mechanism exploration**:
   - Does gender difference persist if you control for wage changes? (economic threat)
   - Does it persist if you control for union membership? (political mobilization)
   - Does it persist if you control for education? (ideology correlation)

**Current state**: Hypotheses presented but not tested. This is honest about limitations, which is good, but leaves the gender puzzle unsolved.

---

## 4. POWER AND PRECISION

### Sample Size and Power Calculation

**Sample**: N = 45,325 person-year observations

**Key questions about power**:
1. Is this sample adequate to detect the reported effects?
2. What is the minimum detectable effect (MDE) size?
3. For non-significant results (p > 0.10), what's the power to detect economically meaningful effects?

**Quick power calculation** (ballpark):

For a two-tailed test with α=0.05, β=0.10 (90% power):
- Effect size needed for political outcome (SD≈2.18): ~0.06 points (0.03 SDs)
- With N=45,325 and clustering by person (~2-3 obs per person on average)
  - Effective N ≈ 45,325 / 2.5 ≈ 18,000
  - Can detect effects of ~0.01-0.02 SDs with high power

**Interpretation**: 
- ✓ Sample IS adequately powered to detect the observed effects (|β| ≈ 0.11-0.2 SDs)
- ✓ Power is **not** the main limitation

---

### Precision: The Real Problem

**What's reported**:
- Point estimates: ✓ Provided for all outcomes
- Standard errors: ✓ Provided (clustered)
- P-values: ✓ Provided
- 95% Confidence intervals: ✗ **NOT reported in summary tables**

**Why this matters**:
- P-value of 0.067 is ambiguous (could say "suggests" or "inconclusive")
- CI width tells us if estimate is precise or noisy
- If CI is [-0.52, +0.02], effect could easily be zero or negative

**For main finding (left-right, β=-0.252, SE=0.137)**:
- 95% CI: [-0.521, +0.017]
- Effect is **not precisely estimated**
- Reasonable confidence that true effect is ≤ 0.5 points, but could be 0

**Critical gap**: Summary document should report:
```
Left-Right Placement: β = -0.252 (95% CI: [-0.521, +0.017], p=0.067)
```
Instead of:
```
Left-Right Placement: β = -0.252 (p=0.067)
```

---

### Power for Null Results

**Non-significant effects reported** (all political outcomes except left-right):
- Nativism: β=-0.207, p=0.191
- Redistributive: β=-0.179, p=0.202
- Welfare: β=-0.084, p=0.609
- Gender equality: β=-0.040, p=0.902

**Questions**:
1. Are these true nulls or just imprecise estimates?
2. Can we rule out economically meaningful effects?

**Power analysis** (for p=0.10 threshold):
- For Nativism (β=-0.207, SE=0.162, p=0.191):
  - 95% CI: [-0.524, +0.110]
  - Cannot rule out |β| > 0.5 points
  - This is substantial uncertainty

**Conclusion**: For non-significant outcomes, wide CIs mean we **cannot conclude effects are zero**. We simply have low power to detect them in these subgroups/outcomes.

---

## 5. SPECIFICATION ROBUSTNESS

### What Robustness Checks Should Be Run?

The current analysis presents results from one specification. Robustness is tested by varying:
1. Model structure (FE variants)
2. Sample (subgroups, alternative exclusions)
3. Outcome construction (scaling, recoding)
4. Exposure measure (different aggregation levels)

### Checks NOT Performed (but should be)

#### 1. **Alternative Fixed Effects Structures**
- [ ] **OLS only** (no FE): Tests if FE are absorbing all variation
- [ ] **Person FE only** (no occ-year): Tests if occupation trends matter
- [ ] **Firm-Year FE only** (no person): Tests firm-level identification
- [ ] **First-differences** (FD): Alternative to within-FE, tests trending selection

**Why important**: Different FE structures make different causal assumptions. If results flip across specs, identification is fragile.

#### 2. **Sample Robustness**
- [ ] **Balanced panel only**: Restricts to persons appearing in all years (eliminates entry/exit bias)
- [ ] **With vs. without attrition-prone occupations**: Tests survivorship bias
- [ ] **By firm size**: Tests if effect varies with firm size (larger firms may adopt AI differently)
- [ ] **By wage terciles**: Tests if effect varies with worker skill

#### 3. **Exposure Measure Variations**
Current analysis uses: **hampole_ai_exposure_avg_foy** (firm-occupation-year)

Should test:
- [ ] **Binary exposure** vs. continuous: Does continuous measure matter?
- [ ] **Different aggregation levels**: Firm-only, Occupation-only, Firm-Year, etc.
- [ ] **With/without intensity adjustment**: log(1+N_apps) is included, but what if you exclude it?
- [ ] **Cross-encoder refined** vs. percentile-based: Stage 4 has multiple match methods

#### 4. **Outcome Robustness**
- [ ] **Alternative political scales**: Reverse-code to check if direction is robust
- [ ] **Winsorized outcomes**: Remove outliers to test if extreme responses drive effects
- [ ] **Categorical outcomes**: Use ordered logit instead of linear (tests functional form)

#### 5. **Critical: Lead/Lag Tests**
- [ ] **Does year t+1 exposure predict year t politics?** (reverse causality check)
- [ ] **Does year t exposure better predict year t+1 politics vs. year t?** (lag identification)
- [ ] **Do lags of political outcomes help identify long-run vs. short-run effects?**

This is the **most important robustness check** for causal interpretation.

#### 6. **Model Specification Tests**
- [ ] **Linearity**: Polynomial exposure terms
- [ ] **Heteroskedasticity**: Breusch-Pagan test; robust vs. standard SEs comparison
- [ ] **Residual autocorrelation**: Durbin-Watson within persons
- [ ] **Functional form**: Spline terms in exposure

### What IS Reported

From the summary and code review:
- ✓ Gender heterogeneity (subgroup analysis)
- ✓ Age heterogeneity (subgroup analysis)
- ✓ Mediation analysis (testing mechanism)
- ✓ Multiple political outcomes (testing specificity)

**Grade: PARTIAL** (some heterogeneity tested, but not structural robustness)

---

## 6. SPECIFICATION STRENGTH ASSESSMENT

### What Would Break This Analysis?

**Critical vulnerabilities** (in order of likelihood):

1. **Selection on unobservables** (HIGH RISK)
   - If workers with pre-existing leftism sort into AI occupations
   - OR if firms adopt AI in response to workforce ideology
   - Lead/lag test would expose this

2. **Measurement error in exposure** (MEDIUM RISK)
   - If AI exposure measure is noisy
   - Could attenuate coefficients toward zero
   - Attenuation bias would make effects seem smaller than true causal effect

3. **Occupational sorting by gender** (MEDIUM RISK)
   - If female AI-exposed workers are in tech (high-skill, not threatened)
   - Male AI-exposed workers in routine automation (threatened)
   - Controls for occupation likely capture this, but not tested directly

4. **Attrition bias** (MEDIUM RISK)
   - If workers with strong political reactions exit panel
   - Creates survivor bias in remaining sample
   - Limits generalizability

5. **Specification bias** (LOW RISK, given FE structure)
   - Wrong functional form
   - Omitted interactions (already addressing with gender/age)
   - These would likely show in residuals or cross-spec variation

---

## 7. KEY QUESTIONS FOR THE ANALYST

1. **On identification**: 
   - Can you run a lead/lag analysis to test whether exposure in year t+1 predicts politics in year t?
   - If significant, this suggests reverse causality or selection.

2. **On precision**:
   - Why are confidence intervals not reported in the main summary?
   - Can you add 95% CI to all reported coefficients?

3. **On gender difference**:
   - Have you tested the interaction (Gender × Exposure) formally?
   - What are the separate coefficients for males and females in a single specification?
   - Does occupational composition differ dramatically between males and females?

4. **On mediation**:
   - The coefficient on job insecurity in the direct effect spec is negative (-0.037).
   - This seems counterintuitive (higher insecurity → more rightward?).
   - Can you clarify this sign?

5. **On robustness**:
   - How stable are results across different FE structures (Person-only, Occ-Year-only, FD)?
   - What happens if you use the binary exposure measure instead of continuous?
   - Have you tested for violations of key assumptions (linearity, homoskedasticity)?

6. **On power**:
   - For null results (Nativism, Welfare, Gender), what's the width of 95% CI?
   - Can you calculate and report the minimum detectable effect size (MDE)?

7. **On sample**:
   - How does attrition vary by AI exposure level?
   - How would results change if restricted to balanced panel?
   - What's the effect if you stratify by firm size or worker skill?

---

## 8. SYNTHESIS: STRENGTHS AND WEAKNESSES

### Key Strengths ✓

1. **Technically sound implementation**
   - Two-way FE via demeaning is correctly executed
   - Clustering is appropriate and conservative
   - Large sample size (N=45,325)

2. **Comprehensive heterogeneity analysis**
   - Gender differences clearly documented (striking female null vs. male effect)
   - Age effects tested
   - Multiple outcomes examined

3. **Transparent reporting**
   - Clear specification language
   - Honest about limitations (sample selection, causality questions)
   - Good summary of substantive interpretation

4. **Appropriate caution on interpretation**
   - Discusses selection bias concern
   - Notes that firm linkage missingness limits generalizability
   - Doesn't overstate causal claims

### Critical Weaknesses ✗

1. **Selection bias unaddressed**
   - Main identification threat is unobserved selection
   - No lead/lag test to check reverse causality
   - No pre/post exposure analysis to rule out anticipatory selection

2. **Missing confidence intervals in reporting**
   - Primary findings reported with p-values only
   - No uncertainty bands shown
   - For borderline significant effects (p≈0.067), CI width is critical

3. **Incomplete mediation analysis**
   - Gender moderation suggests direct effect ≠ total effect, but different mechanism by gender
   - Doesn't explain why females experience insecurity but no political shift
   - Needs interaction specification

4. **Limited robustness testing**
   - Only subgroup analysis and one outcome test mediation
   - No checks of FE structure, sample, or measure
   - No functional form tests (linearity, heteroskedasticity)

5. **Specification fragility unassessed**
   - Only one primary specification reported
   - "What breaks this?" analysis would strengthen paper
   - Cross-specification variation would reveal robustness

---

## 9. RECOMMENDATIONS FOR IMPROVEMENT

### Priority 1: Confidence Intervals (ESSENTIAL)

**Action**: Add 95% CIs to all reported results.

**Current**: "β = -0.252, p=0.067"  
**Better**: "β = -0.252 (95% CI: [-0.521, +0.017], p=0.067)"

**Why**: For borderline significant effects, width of CI determines if result is substantively meaningful.

---

### Priority 2: Lead/Lag Analysis (CRITICAL FOR CAUSALITY)

**Action**: Test whether:
1. Exposure_{t+1} predicts Politics_t (reverse causality)
2. Exposure_{t} better predicts Politics_{t+1} than Politics_t (lagged effect)

**Interpretation guide**:
- If Lead is significant: Reverse causality or selection likely
- If Lag is significant: Effect is more causal (exposure precedes outcome change)
- If both: Complex dynamic process, requires lagged dependent variable specification

---

### Priority 3: Gender-Mediation Interaction (IMPORTANT FOR MECHANISM)

**Action**: Instead of:
```
Direct effect: AI → Politics | Insecurity
```

Estimate:
```
Direct effect split by gender:
  Males: AI → Politics | Insecurity (should be large)
  Females: AI → Politics | Insecurity (should be near zero)
```

**Reason**: Current analysis shows females respond to insecurity differently than males, but doesn't model this interaction.

---

### Priority 4: Functional Form Tests (RECOMMENDED)

**Action**: Estimate:
1. Exposure squared term: Y = β₁ Exposure + β₂ Exposure²
2. Spline in exposure (terciles or quartiles)
3. Compare to linear model via F-test

**Interpretation**: If β₂ is significant, linear model is inadequate.

---

### Priority 5: Specification Robustness (IMPORTANT)

**Action**: Create table comparing:

| Specification | Left-Right β | 95% CI | N | 
|---|---|---|---|
| Current (Person FE + OY-FE) | -0.252 | [-0.521, +0.017] | 45,325 |
| Person FE only | ? | ? | ? |
| OY-FE only | ? | ? | ? |
| First-differences | ? | ? | ? |
| OLS only | ? | ? | ? |

**Interpretation**: If coefficients vary dramatically, identification is fragile.

---

## 10. FINAL VERDICT

### Summary Judgment: CONDITIONALLY VALID

**Can these results support causal claims about AI exposure → political shift?**

**Answer**: NOT YET. The analysis is **well-executed exploratory research** that documents associations, but has **significant identification concerns** that prevent strong causal claims.

---

### What We Can Conclude (With Confidence)

1. **Correlation**: Workers in AI-exposed occupations show a 0.25-point leftward political shift in this sample (p=0.067, imprecise)
2. **Heterogeneity**: Effect is concentrated among males (β=-0.43, p=0.013), absent for females (β=-0.01, p=0.96)
3. **Partial mechanisms**: Job insecurity explains some but not all of the effect, and gender moderates the relationship

---

### What We CANNOT Conclude (Yet)

1. **Causality**: Whether AI exposure CAUSES political shifts or selection drives both
2. **Generalizability**: Results apply to employed workers in formal occupations (23% of SHP)
3. **Completeness**: Whether identified mechanisms are the primary drivers

---

### Path Forward

**To strengthen causal claims**, implement in this order:
1. Lead/lag analysis (test reverse causality)
2. Report confidence intervals (assess precision)
3. Test functional form (ensure linearity)
4. Compare FE structures (assess robustness)
5. Investigate gender-mediation interaction (understand why females don't respond)

**If all of these support current findings**, causal interpretation becomes much stronger. If any breaks (e.g., lead is highly significant), interpretation must be substantially revised.

---

## APPENDIX: Statistical Details

### Variance Explained

For left-right outcome:
- Reported β = -0.252 (marginal effect of exposure)
- Exposure SD: Not reported (CRITICAL OMISSION)
- Outcome SD: ~2.18
- R² of simple bivariat: ~0.013 (exposure alone explains 1.3% of variance)

**This is small but not negligible** for a policy-relevant outcome. However, with clustered SEs and person FE, within-person R² is what matters, which is likely higher.

### Serial Correlation After FE Removal

The de-meaned approach removes person means, but residuals could still be serially correlated within person over time. Proper test:

```
ρ_lag1 = Corr(ε_it, ε_i,t-1 | person)
```

If significant, clustering accounts for it, but residual structure should be documented.

### Clustering Rationale

Three alternatives considered:
1. **By person** (chosen): Appropriate if main concern is repeated obs within person
2. **By occupation**: Appropriate if main concern is general equilibrium effects in occupations
3. **Two-way** (person + occ-year): Conservative but may over-correct

**Chosen approach (person) is appropriate** given the specification has person FE (removes person effects) and occ-year FE (removes occupational effects). Residual dependence is likely within-person.

---

## CONCLUSION

This analysis is **competently executed research on an important question**, with proper implementation of econometric methods but insufficient robustness checks to support strong causal claims. The combination of:

- **Well-documented associations** (AI exposure ↔ leftward shift)
- **Striking heterogeneity** (strong male effect, zero female effect)
- **Moderate effect sizes** (11-12% of SD, borderline significance)
- **Unresolved identification concerns** (selection, reverse causality)

...suggests **suggestive evidence of a real phenomenon** but not conclusive proof of causality.

**Recommended conclusion language**: 
> "This analysis provides suggestive evidence that AI exposure is associated with leftward political shifts, particularly among male workers. However, selection bias and reverse causality cannot be ruled out without additional tests (lead-lag analysis, etc.). Results apply to formal-sector workers with documented firm linkage and should not be generalized to the full working population."

This is more defensible than either "no causal effect" or "causal effect confirmed."
