# CRITICAL REVIEW MEMO: AI Exposure and Worker Outcomes
**Date**: April 10, 2026  
**Reviewer Role**: Devil's advocate challenging identification, measurement, and causal inference  
**Overall Credibility Assessment**: **MEDIUM** (with significant caveats)

---

## EXECUTIVE SUMMARY

Your empirical strategy has three distinct pieces:
1. **AI exposure measurement**: Extract from job ads → match to O*NET tasks → firm-level aggregation
2. **Sample linkage**: 22.7% of SHP workers linked to firms via anonymized firm IDs
3. **Panel regressions**: Triple-diff FE specs identifying within-firm, within-occupation treatment effects

**The core tension**: You've found a striking income reversal (wage loss within firms, wage gains across firms) and political realignment effects, but the identification relies on assumptions that are difficult to verify with available data. The causal story is plausible but faces three major threats:

1. **AI measurement may conflate marketing with actual task exposure** (job ads may not represent real work)
2. **Sample selection is severe** (75% structural firm_id missingness; biased toward large firms)
3. **Residual firm-occupation confounding** (why does firm X deploy AI in occupation Y in year t? Could reflect restructuring targets)

Below I systematically attack each pillar.

---

## SECTION 1: THE AI EXPOSURE MEASUREMENT PROBLEM

### 1.1 Job Ads vs. Actual Task Exposure

**The threat**: Your AI exposure measure comes entirely from keywords matched in job advertisements. This conflates two very different phenomena:
- **Stated**: What firms claim they need (marketing, signaling, aspirational job descriptions)
- **Actual**: What workers actually do with AI (implementation, integration, daily practice)

**Evidence of divergence**:

| Mechanism | Direction | Plausibility |
|-----------|-----------|-------------|
| Firms oversell AI need to attract talent | Upward bias | High (tech skills premium is real) |
| Firms use old job descriptions (not updated for AI adoption) | Downward bias | Medium (legacy hiring processes) |
| Small firms use AI without posting job ads | Downward bias | Medium (silent adoption) |
| Job ads mention AI as aspirational/future need, not current | Upward bias | High (common in tech recruitment) |

**You cannot measure this bias from the data** because you have no independent measure of actual AI use in occupations.

**Specific red flags**:
- 157 AI keywords in your multilingual list, many are very broad ("AI engineer", "data scientist", "predictive analytics", "language model")
- Job ads may mention multiple AI concepts without any being central to actual task (e.g., "AI-powered" software that workers merely use, not develop)
- No validation: You don't have a ground truth sample of jobs you manually coded to validate that keyword matches represent true AI exposure

**Recommended stress-test**:
```python
# Placebo test: Run leads/lags of exposure
# Specification: Y_{it} = β_(-2)*Exposure_{it-2} + β_(-1)*Exposure_{it-1} + 
#                        β_(0)*Exposure_{it} + β_(1)*Exposure_{it+1} + FE + ε

# If you find significant effects on FUTURE exposure (year t+2 predicting year t outcome),
# this suggests either:
# a) Reverse causality (outcomes causing exposure assignment), OR
# b) Measurement error (job ads not capturing true timing of AI adoption)
```

**Estimated bias direction**: Likely UPWARD (job ads overstate AI prevalence). This would bias wage effects toward zero, so your negative wage effects are probably understated in magnitude.

---

### 1.2 Crosswalk Chain Fragility

Your exposure measure requires perfect alignment across four sequential steps:
```
Job Ads → Keywords → Stage 3 (LLM extraction) → Stage 4 (O*NET matching) → 
Stage 5 (Firm aggregation) → Stage 6 (Link to workers)
```

**Failure modes at each step**:

1. **Keywords → Job Ads**: False positives and false negatives
   - Your Stage 0 starts with ~14.7M jobs, reduces to 107K via keyword matching
   - You then filter via "false positive classification" but this is subjective
   - **Question**: How many jobs mention "AI" in marketing language but have no actual AI exposure?
   - **You reported**: 351 NaN values in matched_keywords were removed
   - **This suggests**: ~0.3% of matched jobs had empty keyword fields (catching some false positives)
   - **But**: This is filtering at the wrong level (empty keywords, not false positives)

2. **Stage 3 (LLM extraction)**: Three-step LLM pipeline with reprocessing needed
   - Step 1 (O3): F1 = 0.7817 (not great; ~22% error rate)
   - Step 2 (GPT-5-mini): F1 = 0.8442 (best performer)
   - Step 3 (O3): F1 = 0.9194 (after JSON fixes)
   - **Problem**: You're not comparing to ground truth on your actual data, only on manual sample of 199 items
   - **Scale uncertainty**: How does F1 = 0.82 average error rate translate to systematic bias in downstream analysis?

3. **Stage 4 (O*NET similarity)**: Similarity threshold choice drives everything
   - **Current choice**: 95th percentile threshold
   - **Never tested**: What happens at 90th, 85th, 80th percentiles? (robustness check missing)
   - **The problem**: You're selecting high-similarity matches, but O*NET task statements are very general
   - **Example**: "AI software engineer" matches to O*NET task "Uses computers to process information" (true but not specific)
   - **You should check**: Does exposure measure change dramatically with percentile threshold?

4. **Stage 5 (Firm aggregation)**: Hampole weighting scheme
   - Share-of-applications approach = (# AI apps matching task / total # AI apps in firm-year) × task importance
   - **Assumption**: All AI applications in a firm-year equally exposed to that occupation's workers
   - **Unrealistic**: A firm may have AI applications in data science (never touches production workers in manufacturing)
   - **Alternative**: Your binary method is more conservative but also included

**Recommended stress-test**:
```python
# Sensitivity to measurement:
# 1. Re-run analysis excluding jobs with AI keywords in title (often less real)
# 2. Re-run with higher percentile thresholds (90th, 85th)
# 3. Cross-tabulate AI exposure by occupation; does it match labor economist expectations?
#    (E.g., software engineers should have high AI exposure, janitors should have zero)
```

---

### 1.3 Missing Validation Against Reality

**Most critical gap**: You have no validation dataset.

You don't have:
- A sample of jobs you can manually review and code for true AI exposure
- Comparison to actual firm AI investments (from 10-K filings, patent data, etc.)
- Time-use survey data showing how much time workers actually spend with AI
- Occupational surveys asking workers about AI use in their jobs

**What you would need to validate**:
1. **Construct validity**: Do higher-exposure occupations report more AI use in worker surveys?
2. **Predictive validity**: Do occupations flagged as AI-exposed have faster wage growth/decline than occupations not flagged?
3. **External validity**: Do industry-level patterns match independent AI adoption data?

**Without this, your measure is unvalidated**. The political and wage effects could be artifacts of measurement error rather than true AI exposure.

**Risk assessment**: If true AI exposure is measured with classical error, your regression coefficients are biased toward zero (except for the political effects if they respond to perceived threat, not actual exposure). But if measurement error is non-classical (systematic bias), you could get seriously confounded estimates.

---

## SECTION 2: THE SAMPLE SELECTION PROBLEM

### 2.1 Severe Firm_ID Missingness (75% Structural)

You've documented this in known-issues.md, so you know it's bad. Let me quantify the threat:

**The facts**:
- 22.7% of SHP person-years have firm_id (45,325 / 199,324)
- Only 26% of SHP respondents ever appear in firm linkage file
- Missingness is ~74-80% every year, including 2012-2021 when linkage file is available
- Root cause: Anonymization restrictions (likely small firms, self-employed excluded)

**The causal threat**:
Selection into firm linkage is **NOT random** across workers. If selection is correlated with:
- **Worker characteristics** (education, age, gender): Bias in treatment effects
- **Occupations**: Bias in occupation-level effects (e.g., AI effects will be concentrated in large-firm occupations)
- **Firm characteristics** (wage level, innovation): Bias in firm-level heterogeneity

**Your evidence on selection**:
- Large firms overrepresented (75% of person-years in firms with 2+ occupations are in firms with 30+ respondents)
- Probably white-collar occupations overrepresented (small firms = service, blue collar)
- Probably higher-wage workers (large firms pay more)

**Quantifying the bias**:

You found:
- Firm-Year FE wage effect: -8.5% (negative)
- Occupation-Year FE wage effect: +10.9% (positive)
- Difference: 19.35 percentage points

**Question**: Is this true compositional effect, or artifact of selection?

**Test**: If you had random sample, you should see same results. But if sample is selected on firm type:
- Small firms may suppress wages differently than large firms (more paternalistic, informal arrangements)
- Selection could reverse the sign entirely

**Your subgroup analysis showed**:
- Women: -17.6% wage loss (FY-FE)
- Older workers: -15.8% wage loss (FY-FE)

**Critical question**: Are women and older workers **overrepresented** in your firm-linked sample? If they're underrepresented, the true wage effects in the full sample could be much larger.

**You should test**:
```python
# Compare demographics between firm-linked and non-linked samples
# firm_linked = (firm_id is not null)
# Compare: age, gender, education, occupation distribution

# If women/older/less educated are UNDERREPRESENTED in firm-linked sample,
# then wage effects are selection-biased
```

**Recommended bound**: Assume worst-case selection scenario:
- Workers with firm_id are from high-wage firms (10-20% wage premium)
- AI exposure affects large firms more than small firms (plausible)
- True wage effect on full sample could be 1.5-2x larger in magnitude

**Implication**: Political effects are similarly selected. You're measuring response to AI exposure **among large-firm workers**, not representative population.

---

### 2.2 Post-FE Degrees of Freedom Are Tight

You've documented:
- Effective DoF after FEs: ~2,325 (for Spec 2 with firm-year FE + person FE)
- This is feasible but creates three problems:

1. **Standard error inflation**: With tight DoF, clustering standard errors become less reliable (Angrist-Pischke recommendation: need at least 50-100 clusters, you have ~10k persons but each person contributes ~4.5 observations)

2. **Multiple testing problem**: You're running 5 political outcome variables + 1 wage outcome + robustness checks
   - Without correction, false discovery rate is ~20-30%
   - Even with Bonferroni correction, you lose power

3. **Weak instrument concerns** (if any IV approach): If you were to use instruments (which you're not, but should consider), tight DoF means weak instrument bias is a serious concern

**Your reported results**:
- Left-right placement: p = 0.067 (marginally significant)
- Male left-right: p = 0.013 (significant)
- Income loss: p = 0.030 (significant)

**The DoF problem**: With 2,325 effective DoF, you have low power for small effect sizes. Your effects are moderate size, so you can detect them, but:
- Confidence intervals are wide (you report 90% and 95% CIs separately, good practice)
- Any subgroup analysis has 3-5x lower power

**Recommended sensitivity check**:
```python
# Wild bootstrap CI (Cameron, Gelbach, Miller approach)
# Standard asymptotics may be unreliable with tight DoF
# Run wild cluster bootstrap at person level to get robust CIs
```

---

### 2.3 Heterogeneous Effects on Thin Ice

You report heterogeneous effects by:
- Gender (male vs. female)
- Age (old vs. young)
- Employment status

**Sample size by subgroup**:
- Males: ~22k person-years
- Females: ~23k person-years
- Old (>67th percentile): ~15k person-years
- Young (<33rd percentile): ~15k person-years

**Problem**: Subgroup analysis reduces sample by 3-5x, so effective DoF drops to 400-700 per subgroup.

**Why this matters**: Your key finding is gender difference in political response (males shift left, females don't).

- Male left-right: β = -0.430, p = 0.013, N_eff ≈ 7,500 person-years
- Female left-right: β = -0.012, p = 0.958, N_eff ≈ 7,500 person-years

**The issue**: These could be genuinely different effects, BUT gender × AI exposure interaction has lower power.

**You should test**:
```python
# Formal interaction test:
# Y = α + β₁*Exposure + β₂*Female + β₃*Exposure*Female + FE + ε
# H0: β₃ = 0 (no gender difference)

# If you reject H0 with p < 0.05, gender difference is real
# If p > 0.05, you cannot rule out that differences are due to sampling variability
```

---

## SECTION 3: IDENTIFICATION AND CONFOUNDING

### 3.1 The Firm-Occupation Assignment Problem (Core Threat)

Your recommended spec (SPEC 2) uses firm-year FE + occupation-year FE + person FE. This identifies:

**Variation**: Workers in same firm, same year, different occupations → different AI exposures

**Logic**: "In firm F in year T, do workers in occupation O₁ (high AI exposure) differ from workers in occupation O₂ (low AI exposure)?"

**This is excellent identification... IF**:

The firm's decision to deploy AI in occupation O₁ in year T is **exogenous** to that occupation's unobserved characteristics.

**The threat**: Firm-occupation assignment violates this.

**Two competing stories**:

| Story | Mechanism | Evidence | Bias Direction |
|-------|-----------|----------|-----------------|
| **Benign (Task-tech fit)** | Firm deploys AI where tasks are routine/automatable | AI exposure ↔ O*NET routine-task intensity | If true: causal effect valid |
| **Malign (Restructuring)** | Firm targets AI to occupations it's downsizing/attacking | AI exposure ↔ prior occupation trouble | If true: confounds wage effect |

**You've documented both as possible** but haven't tested them systematically.

**Testing the benign story**:
```python
# Regression 1: Does AI exposure correlate with task structure?
# AI_exposure ~ routine_task_intensity + abstract_task_intensity + ...
# Should find: positive on routine, negative on abstract

# Regression 2: Pre-treatment occupation health vs. AI exposure
# For occupations pre-2015 (before major AI wave):
# Does prior average wage, employment growth predict later AI exposure?
# Should find: NO correlation if assignment is random
```

**You documented**:
- Wage loss concentrated in women and older workers
- Wage loss is ~8.5% in magnitude

**If malign story is true**:
- Firms may have targeted occupations with surplus female labor (lower bargaining power)
- Wage loss isn't causal AI effect; it's selection into restructuring-targeted occupations
- True causal effect of AI could be zero or positive

**My assessment**: Probably 60% benign, 40% malign.
- Benign: Tech deployment follows task structure, makes sense economically
- Malign: Firms use technology adoption as cover for restructuring (documented in organizational behavior literature)

**Confidence in your wage effects**: MEDIUM

If malign story is 40% true, your wage effect is 40% confounded, meaning true causal effect is closer to -5% than -8.5%.

---

### 3.2 The Political Selection Problem

You find strong leftward political shift driven by AI exposure. But:

**Reverse causality threat**: 
- Workers with pre-existing leftist views may self-select into AI-exposed occupations (tech, finance have left-wing labor forces)
- OR workers in AI-exposed firms may be driven LEFT by other firm characteristics (tech companies tend to be left-leaning employers)
- Your firm-year FE controls for firm-wide sentiment, but not firm×occupation×year sentiment

**Your occupation-year FE approach assumes**:
- Within a given occupation-year, workers across firms are comparable
- But AI-exposed firms within an occupation may be systematically different (left-leaning tech firms vs. right-leaning manufacturers)

**Test for selection**:
```python
# Lead-lag specification:
# Y_it = β₀ + β₁*Exposure_{it} + β₂*Exposure_{it+1} + β₃*Exposure_{it+2} + FE + ε_it

# If workers self-select LEFT into AI-exposed firms:
# Should find: β₂, β₃ > 0 (future exposure predicts current political preferences)
# OR negative lagged effects (past exposure predicts current outcomes)

# This would indicate reverse causality / selection
```

**My assessment**: Political effects are more suspect than wage effects because:
1. Harder to rule out pre-treatment ideology
2. Occupational sorting is known to be strong (e.g., tech workers are already left-leaning)
3. Media/information effects: Workers may respond to AI news coverage, not actual exposure

**Confidence in political effects**: LOW-MEDIUM

---

### 3.3 The Omitted Variable Problem: Individual × Year Interactions

Your Spec 2 includes person FE, firm-year FE, occupation-year FE. What it doesn't include:

**Person-year interactions** (Spec 3): 
- These would capture idiosyncratic life events (divorce, health crisis, retirement planning) that shift mood and politics
- You tried this (Spec 3) and found power collapsed, but didn't report whether coefficients changed

**You should have**:
- Reported Spec 3 coefficients even if imprecise
- Test: Do Spec 2 and Spec 3 coefficients move substantially? If yes, omitted person-year shocks are important

**Example threat**: 
- Person i turns 60 in year 2020 (thinking about retirement)
- Same year, their firm deploys AI in their occupation
- They shift left (really due to retirement anxiety, not AI exposure)
- Spec 2 attributes it to AI, Spec 3 would (try to) control it

---

## SECTION 4: THE INCOME REVERSAL MYSTERY (And What It Reveals)

### 4.1 Your Explanation is Self-Aware But Unverifiable

You correctly identify that:
- **Firm-Year FE**: Within-firm wage effect is -8.5% (causal)
- **Occupation-Year FE**: Across-firm effect is +10.9% (selection)

**Your mechanism**: High-wage firms adopt AI → positive selection across firms → masquerades as positive wage effect.

**But you cannot verify this** because you don't have a direct measure of which firms adopted AI and when.

**What you should test**:
```python
# Decomposition: Firm selection vs. task selection
# 1. Does AI exposure predict firm size? (Should be positive if large firms adopt more)
# 2. Within occupations, do high-wage firms have more AI exposure?
# 3. Do AI-exposed occupations differ in baseline wage from non-exposed?

# If all three are true, then selection mechanism is confirmed
```

### 4.2 The Gender × AI Wage Effect Needs Explanation

You found:
- **Women**: -17.6% wage loss (FY-FE, p = 0.026)
- **Men**: -1.7% wage loss (FY-FE, p = 0.68, not significant)

**Three explanations**:

| Explanation | Evidence Needed | Plausibility |
|---|---|---|
| **Women concentrated in routine tasks** | AI-exposed occupations have higher female share | Medium (some truth but not complete story) |
| **Women have lower bargaining power** | Occupation-level wage-setting power analysis | Medium (union rates, mobility) |
| **Occupational sorting** | Women in AI companies are different (higher skilled) than women elsewhere | Medium |

**You didn't test**: Does AI exposure predict employment loss differently for men and women?

If AI-exposed occupations see female employment growth, then wage loss for women isn't about job loss (men also losing jobs but wages holding). This would support bargaining power story.

**Critical test**:
```python
# Employment change analysis
# outcome = employment_change_{it} (did person stay employed year t to t+1?)
# Does AI exposure predict job loss for women > men?

# If yes, wage loss is about job destruction + compositional shift
# If no, wage loss is about within-occupation skill downgrading
```

---

## SECTION 5: STRUCTURAL THREATS TO VALIDITY

### 5.1 The "Fake Placebo" Problem in Your Design

You haven't run lead/lag tests. This is critical because:

**Lead test**: Does FUTURE exposure (year t+2, t+3) predict CURRENT outcomes (year t)?
- If yes: Measurement error, reverse causality, or unobserved trends
- If no: More confidence in contemporaneous effects

**Lag test**: Does PAST exposure (year t-2) predict current outcomes better than current exposure?
- If yes: Effects are slow-moving / accumulation
- If no: Effects are immediate

**You should run**:
```python
# Full lead-lag specification
Y_it = Σ_k β_k * Exposure_{it+k} + FE + ε_it
# k ∈ {-2, -1, 0, 1, 2}

# Expected pattern if valid:
# β_{-2} ≈ β_{-1} ≈ 0 (no effects on future)
# β_{0} significant (current effect)
# β_{1}, β_{2} may be significant (lagged effects from persistence)

# If β_{-1}, β_{-2} significant, you have a problem
```

This is a **critical missing test** that takes 5 minutes to run.

---

### 5.2 Time-Varying Confounders in Unobserved Dimension

Your specifications absorb:
- Firm-year shocks (firm-year FE)
- Occupation-year shocks (occupation-year FE)
- Person-invariant traits (person FE)

**They don't absorb**:
- Firm-occupation-year interactions (firm deciding to downsize call center in year t specifically)
- Individual-year interactions (person's mood shift in year t due to life events)
- Cohort × calendar time effects (e.g., older workers in 2020 more anxious due to age + COVID, not AI)

**The threat**: If any of these time-varying confounders correlate with AI exposure, you're biased.

**Example**: COVID-19 in 2020-2021
- Caused high anxiety (especially older workers)
- Remote work adopted (maybe looks like AI to job posters)
- Firms adapted quickly (some laid off, some retained workers)

**You should test**:
```python
# COVID-robustness check
# Re-run specifications excluding 2020-2021
# Do main effects hold?

# If effects disappear, COVID confounding is probable
# If effects similar, probably not COVID-driven
```

---

## SECTION 6: SPECIFIC STRESS TESTS (Priority Order)

### Test 1: LEAD SPECIFICATION (CRITICAL - 5 minutes)
```python
# Y_it = β_{-1}*Exp_{it+1} + β_0*Exp_it + β_1*Exp_{it-1} + FE + ε
# If β_{-1} significant, you have a major problem
```

### Test 2: EXCLUSION RESTRICTIONS FOR GENDER DIFFERENCE
```python
# Interaction test: Y = α + β₁*Exp + β₂*Female + β₃*Exp*Female + ...
# Does β₃ = 0? (Gender difference significant)
# Compare SE(β₃) to SE(β₁) to assess precision
```

### Test 3: PERCENTILE SENSITIVITY (Stage 4)
```python
# Re-run Spec 2 with different O*NET similarity thresholds
# Thresholds: 95th (current), 90th, 85th, 80th percentile
# Plot coefficient as function of threshold
# If coefficient varies wildly, measurement fragile
```

### Test 4: SAMPLE REPRESENTATIVENESS
```python
# Compare demographics: firm-linked vs. non-linked samples
# Variables: age, gender, education, occupation, firm size
# If substantial differences, document as sample limitation
```

### Test 5: REMOVE COVID YEARS
```python
# Exclude 2020-2021, re-run Spec 2
# Do main effects hold magnitude and significance?
```

### Test 6: REMOVE RECENT YEARS
```python
# Exclude 2019+ (when AI became mainstream)
# Do effects exist in 2012-2018 only?
# If effects only post-2018, measurement error likely (noisier job ads, media hype)
```

### Test 7: OCCUPATIONAL PLAUSIBILITY
```python
# Cross-tabulate: AI exposure by occupation code
# Do software engineers, data scientists get highest AI exposure?
# Do janitors, retail workers get zero?
# If implausible patterns, measurement needs review
```

---

## SECTION 7: CAUSAL CREDIBILITY ASSESSMENT

### Summary of Threats

| Threat | Severity | Direction | Mitigation Feasible? |
|--------|----------|-----------|------------------|
| **AI measure = job ads only (not actual use)** | HIGH | Upward bias (overstates exposure) | No (need external validation) |
| **Crosswalk chain fragility** | MEDIUM | Could go either way | Yes (percentile sensitivity test) |
| **No validation dataset** | HIGH | Unknown direction | No (would need new data collection) |
| **Firm_id selection (75% structural missing)** | HIGH | Biases toward large-firm effects | Partial (sensitivity analysis on subgroups) |
| **Tight DoF post-FE** | MEDIUM | Inflates standard errors | No (design limitation) |
| **Firm-occupation endogeneity** | MEDIUM | Could bias wage effect by ±30% | Partial (task-fit + pre-treatment balance tests) |
| **Reverse causality in politics** | MEDIUM | Could reverse sign or reduce effect | Partial (lead/lag test) |
| **Missing person-year interactions** | MEDIUM | Could reduce effect size 20-30% | Partial (Spec 3 comparison) |

### Causal Credibility by Outcome

**INCOME EFFECTS**: **MEDIUM credibility**
- Robust heterogeneity (women, older workers)
- Within-firm variation reduces selection bias
- BUT: firm-occupation endogeneity unresolved, measurement unvalidated
- **Likely range of true effect**: -5% to -12% (vs. reported -8.5%)
- **Confidence**: 60% confident effect is real and causal

**POLITICAL EFFECTS**: **MEDIUM-LOW credibility**
- Strong heterogeneity (men >> women, old > young) is interesting
- BUT: Occupational sorting, reverse causality harder to rule out
- Occupation-year FE doesn't fully control for firm selection within occupation
- **Likely range of true effect**: -0.1 to -0.3 (vs. reported -0.25)
- **Confidence**: 50% confident effect is causal (vs. selection/sorting)

**JOB INSECURITY EFFECTS**: **MEDIUM credibility**
- More direct mechanism (job loss → anxiety)
- Spec 2 controls firm-year shocks well
- BUT: Still subject to same firm-occupation endogeneity
- **Confidence**: 65% confident effect is real

---

## SECTION 8: WHAT WOULD IMPROVE CREDIBILITY (RANK ORDER)

### Tier 1: Can do immediately (0-2 weeks)

1. **Lead/lag specification** (Test 1 above)
2. **Gender interaction formal test** (Test 2)
3. **COVID exclusion** (Test 5)
4. **Occupational plausibility check** (Test 7)
5. **Task-fit regression**: Does AI exposure correlate with O*NET routine-task intensity?
6. **Pre-treatment balance**: Do pre-2015 occupation wages predict post-2015 AI exposure?

### Tier 2: Moderate effort (1-4 weeks)

1. **Percentile sensitivity** (Test 3) → requires re-running Stage 4
2. **Sample representativeness** (Test 4) → merge census/admin data if available
3. **Spec 3 detailed comparison**: Report Spec 3 coefficients even if noisy
4. **Subgroup robustness**: Split by firm size, industry if available
5. **Alternative AI measures**: Occupational-level (simple average) vs. firm-occupation interaction

### Tier 3: Hard but valuable (2-8 weeks)

1. **Validation dataset**: Manually code sample of 500-1000 jobs for true AI exposure, compare to keyword measure
2. **Alternative external validation**: Cross-tabulate your AI exposure against Bureau of Labor Statistics AI adoption statistics
3. **Longitudinal wage analysis**: Do occupations flagged as AI-exposed see different wage growth than non-exposed? (independent of your firm-level measure)
4. **Reversed exposure measure**: Instead of exposure → outcomes, do outcomes predict exposure? (rules out reverse causality)

---

## SECTION 9: FINAL VERDICT

### What You've Done Well

1. **Transparent about limitations**: Your memos document firm_id missingness, subgroup power issues, specification choices clearly
2. **Multiple specifications**: Showing Spec 1, 2, 3 alternatives is good practice
3. **Heterogeneous effects**: Reporting gender, age subgroups catches real variation
4. **Honest about income reversal**: Not sweeping under the rug that results change by spec

### What Needs Work

1. **No validation of AI measure** against external ground truth
2. **No lead/lag tests** (most basic placebo test missing)
3. **No firm-occupation endogeneity evidence** (benign vs. malign story untested)
4. **Sample selection** documented but not sensitivity-tested
5. **Measurement fragility** (percentile thresholds untested)

### Confidence Levels (with caveats)

| Claim | Confidence | Key Caveat |
|-------|-----------|-----------|
| "AI exposure causes wage loss of ~8%" | 60% | Within large firms; selection bias possible; could be 5-12% |
| "Wage loss larger for women/older" | 75% | Heterogeneity seems robust; subgroup sample size adequate |
| "AI exposure causes leftward political shift of ~0.25 points" | 50% | Could be selection/sorting; need lead/lag test |
| "Male political response driven by AI, not job insecurity alone" | 55% | Need mediation analysis more formally |

### Publishable? With Conditions

**If you're aiming for:**
- **Top-5 journal** (Nature, Science, PNAS, AER): Need Tier 1 tests minimum, preferably Tier 2
- **Top field journal** (Labor, sociology, political science): Tier 1 tests + honest limitations section adequate
- **Policy/applied outlet** (SSRN, working papers): Current state publishable with caveats clearly stated

**You should add one sentence to every empirical claim**:
> "These effects are identified within large firms with available firm linkage data (~23% of sample). The exposure measure relies on job advertisements matched to O*NET tasks; effects may be biased if job ads misrepresent actual AI task content."

---

## CONCLUSION: CREDIBILITY SCORECARD

| Dimension | Score | Status |
|-----------|-------|--------|
| **Measurement Validity** | 5/10 | Unvalidated; job ads may not match actual work |
| **Sample Representativeness** | 6/10 | Severe selection (75% structural missing); documented but not tested |
| **Identification Strategy** | 7/10 | Triple-diff logic sound; firm-occupation endogeneity unresolved |
| **Statistical Power** | 7/10 | Adequate for main effects; low for subgroups |
| **Robustness Evidence** | 5/10 | Missing critical tests (leads, sensitivity, validation) |
| **Transparency** | 9/10 | Excellent documentation of limitations |
| **Overall Causal Credibility** | 6/10 | **MEDIUM** — Results likely real but magnitudes uncertain |

**Bottom line**: Your findings are **plausibly causal** for income effects (~60% confident) and **likely confounded** for political effects (~50% confident of causality, 50% confident of selection/sorting). The income reversal itself is revealing and suggests real within-firm mechanisms. But the measurement is fundamentally unvalidated, and the sample selection is severe. Tier 1 tests (2 weeks work) would substantially increase credibility. Current state is solid for a working paper or field journal, but needs Tier 1+2 before top-5 aspirations.

---

## APPENDIX: SYNTAX FOR RECOMMENDED TESTS

All tests can be executed in your existing Python/statsmodels pipeline. Add to `stage_8b_robustness_checks.py`:

```python
# Test 1: Lead-Lag Specification
def run_lead_lag_test(df, outcome_var):
    """
    Lead-lag specification:
    Y_it = β_{-2}*Exp_{it+2} + β_{-1}*Exp_{it+1} + β_0*Exp_it + 
           β_1*Exp_{it-1} + β_2*Exp_{it-2} + FE + ε_it
    """
    # Create leads/lags
    df['exp_lead2'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(-2)
    df['exp_lead1'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(-1)
    df['exp_lag1'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(1)
    df['exp_lag2'] = df.groupby('idpers')['hampole_ai_exposure_avg_foy'].shift(2)
    
    # Fit with firm-year FE via de-meaning
    # ... (de-mean logic from Spec 2) ...
    
    # Report all lags/leads with 95% CI
    # If β_{-1}, β_{-2} significant, reverse causality likely

# Test 2: Gender Interaction
def test_gender_interaction(df, outcome_var):
    """Formal test: Does Exp*Female coefficient differ from zero?"""
    # Y = α_i + α_ft + α_ot + β₁*Exp + β₂*Female + β₃*Exp*Female + ε
    # H0: β₃ = 0
    
# Test 3: Percentile Sensitivity
def percentile_sensitivity_test(df):
    """
    Re-run Stage 4 matching with different thresholds
    Plot coefficient as function of percentile threshold
    """
    thresholds = [0.80, 0.85, 0.90, 0.95]
    results = []
    for pct in thresholds:
        # Load Stage 4 output with different threshold
        # Run Spec 2
        # Record coefficient
    # Plot
    
# Test 4: COVID Exclusion
def covid_robustness(df, outcome_var):
    """Exclude 2020-2021, re-run Spec 2"""
    df_pre_covid = df[df['year'] < 2020]
    # Run Spec 2 on df_pre_covid
    # Compare coefficient to full-sample Spec 2
    
# Test 7: Occupational Plausibility
def occupational_plausibility_check(df):
    """
    Cross-tabulate mean exposure by 4-digit ISCO code
    Check: Are software developers > janitors?
    """
    exposure_by_occ = df.groupby('isco08_4d')['hampole_ai_exposure_avg_foy'].agg(['mean', 'count'])
    # Merge with ISCO titles
    # Inspect top and bottom occupations
    # Plot distribution
```

---

**END OF MEMO**

Date: April 10, 2026  
Prepared by: Critical Review Analysis  
Status: FINAL DRAFT
