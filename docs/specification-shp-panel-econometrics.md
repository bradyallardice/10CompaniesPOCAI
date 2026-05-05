# SHP Panel Econometric Specification: AI Exposure and Political Outcomes

**Date**: April 10, 2026  
**Analysis Phase**: SHP panel regression design with interacted fixed effects  
**Status**: Specification finalized based on sample size diagnostics

---

## Research Question

How does within-firm, within-occupation AI exposure affect individual political preferences and vote choice? Specifically, do workers in occupations more exposed to AI within their firm become more politically conservative, more authoritarian, or more likely to support anti-immigration policies?

---

## Data Sample

**Total SHP respondents**: 38,528 unique persons  
**Respondents with firm linkage**: 9,895 (25.7%)  
**Person-year observations with firm_id**: 45,325 (22.7% of 199,324 total)  
**Year range**: 2012–2023

### Sample Size Power Analysis

Based on diagnostic run (April 10, 2026):

| Dimension | Count | % Effective |
|-----------|-------|------------|
| **Firm-Year Cells** | 30,507 | 100% |
| ├─ with 2+ occupations | 4,523 | 16.3% |
| ├─ with 3+ occupations | 1,698 | 6.1% |
| ├─ with 2+ respondents | 5,722 | 18.8% |
| └─ median respondents | 1 | — |
| **Firm-Occ-Year Cells** | 37,064 | — |
| ├─ with 2+ respondents | 1,852 | 5.0% |
| └─ median respondents | 1 | — |
| **Firm Switchers** | 1,794 | 18.1% of 9,895 |
| **Occupation-Year Coverage** | 71,703 | 36% non-zero |

**Key constraint**: 75% of firm-year cells have 1 person; 81.9% of people never switch firms. This limits power for certain specifications but does not eliminate identification.

---

## Three Candidate Specifications

### SPEC 1: Additive Fixed Effects (Baseline)

**Model**:
$$Y_{ift} = \alpha_i + \alpha_f + \alpha_t + \alpha_o + \beta \cdot \text{Exposure}_{iot} + \varepsilon_{ift}$$

**Fixed effects**:
- $\alpha_i$: Person FE (absorbs time-invariant individual traits)
- $\alpha_f$: Firm FE (absorbs time-invariant firm effects)
- $\alpha_t$: Year FE (absorbs aggregate macro trends)
- $\alpha_o$: Occupation FE (absorbs time-invariant occupation traits)

**What it absorbs**:
- Individual traits (education, personality, political ideology pre-treatment)
- Permanent firm characteristics (firm size, sector, ideology)
- Year-wide trends (national political shifts, macro economy)
- Occupation-wide baseline (e.g., sales occupations are different from engineering)

**What it does NOT absorb**:
- Time-varying firm-year shocks (firm-specific recessions, acquisitions, restructuring)
- Time-varying firm-occupation interactions (this firm specifically deployed AI in this occupation in this year)
- Any individual × year interactions

**Identification**:
- Variation: Across-occupation differences within persons over time
- Logic: Same person, same firm, different years — as firm's AI exposure varies across occupations, does person's politics shift?

**Threats to identification**:
- **Firm-year confounders**: If firm X had a bad year in 2022 and *also* deployed AI in certain occupations that year (targeting underperforming departments), the coefficient on exposure could conflate "AI exposure" with "working in a troubled department"
- **Selection into firms**: People self-sort into firms; people in firms deploying AI may be different from those in non-deploying firms

**Power**: HIGHEST. Uses all 45,325 person-years. Every person-year contributes.

**Status**: Conservative baseline. Good for robustness, less ideal as main spec.

---

### SPEC 2: Firm-Year FE + Occupation-Year FE (RECOMMENDED PRIMARY SPEC)

**Model**:
$$Y_{ift} = \alpha_i + \alpha_{ft} + \alpha_{ot} + \beta \cdot \text{Exposure}_{iot} + \varepsilon_{ift}$$

**Fixed effects**:
- $\alpha_i$: Person FE (time-invariant individual heterogeneity)
- $\alpha_{ft}$: Firm-Year FE (firm-specific year-level shocks)
- $\alpha_{ot}$: Occupation-Year FE (occupation-specific year-level shocks)

**What it absorbs**:
- Everything Spec 1 absorbs, PLUS:
  - Firm-level time-varying shocks: restructuring, layoffs, acquisition, bad earnings, management change at firm $f$ in year $t$
  - Occupation-level time-varying shocks: nationwide labor market trends for occupation $o$ in year $t$, technological shifts in occupation $o$ that year
  - All firm-wide sentiment in year $t$ (good or bad morale, announcements, news)

**What it does NOT absorb**:
- Individual × year interactions (person's mood shifts in year $t$ for idiosyncratic reasons)
- Firm × person × year three-way interactions

**Identification**:
- Variation: **Within firm-year, across occupations**
- Logic: In firm $f$ in year $t$, are workers in occupations more exposed to AI politically different from workers in less-exposed occupations in the same firm-year?
- This is **triple-difference** logic:
  - Diff 1: Across occupations (exposed vs. not exposed within firm)
  - Diff 2: Within same firm-year (both see same firm-level shocks)
  - Diff 3: Fixed effects absorb time-invariant traits and occupation/firm trends

**Threats to identification**:
- **Residual firm × occupation endogeneity**: Even after absorbing firm-year shocks, why does firm $f$ deploy AI in occupation $o$ in year $t$ specifically?
  - Benign story: Firm deploys AI where it's most useful (task structure drives deployment)
  - Malign story: Firm deploys AI in occupations it's targeting for restructuring
  - Mitigation: Show AI deployment predicts task-replaceability, not pre-treatment occupation health

- **Individual × year confounders**: If person $i$'s mood shifts in year $t$ and person works in the occupation that the firm happens to be deploying AI in that year, bias could result
  - Mitigation: Limited by sample size but manageable; test with person × year FE as robustness check (see Spec 3)

**Power**: 
- Effective sample: 4,523 firm-years with 2+ occupations (16.3% of 30,507 total)
- Number of DoF absorbed: ~30k firm-year FEs + ~3k occupation-year FEs + ~9.9k person FEs ≈ 43k
- Remaining DoF for identification: 45,325 - 43,000 = ~2,325
- Feasible but tight; expect larger standard errors than Spec 1

**Status**: **RECOMMENDED MAIN SPECIFICATION**
- Strongest causal identification story
- Acceptable power given the constraint
- Clean triple-difference logic
- Transparent assumptions

---

### SPEC 3: Triple-Diff with Person-Year FE (ROBUSTNESS/EXPLORATION)

**Model**:
$$Y_{ift} = \alpha_i + \alpha_t + \alpha_{ft} + \alpha_{ot} + \alpha_{it} + \beta \cdot \text{Exposure}_{iot} + \varepsilon_{ift}$$

**Fixed effects**:
- $\alpha_i$: Person FE
- $\alpha_t$: Year FE
- $\alpha_{ft}$: Firm-Year FE
- $\alpha_{ot}$: Occupation-Year FE
- $\alpha_{it}$: Person-Year FE (new)

**What the addition of $\alpha_{it}$ does**:
- Absorbs all time-varying person-level shocks (mood, life events, unobserved job search, personal political shifts)
- Means you identify *only* off within-person-year variation across occupations/firms
- But almost all people in a given year are in a single firm, so this mostly replicates Spec 2 absorbing residual variation

**When to use**:
- Robustness check: "Do results hold if we're even more aggressive about person-level confounders?"
- Exploration: Understand how much power is lost with maximum FE specification

**Power**: LOWEST. 
- Nearly all DoF absorbed
- Identifying variation: firm-year-occupation cells where same person appears in multiple occupations (extremely rare)
- Likely to yield very noisy estimates or convergence issues

**Status**: Secondary robustness check only. Not primary spec.

---

## Endogeneity Narrative: Firm × Occupation Assignment

Even with firm-year FE, a central threat remains: **Why does firm $f$ deploy AI in occupation $o$ in year $t$?**

This is not a flaw in the research design — it's inherent to the data. But we can evaluate whether the "benign" or "malign" explanation is more plausible.

### Benign Story (Task-Technology Fit)

**Assumption**: Firms deploy AI where the technology matches task characteristics of the occupation.

**Evidence to collect**:
- Does AI exposure correlate with O\*NET task-replaceability scores?
- Within firms, do AI-exposed occupations have higher concentration of routine, automatable tasks?
- Across firms, do occupations with high routine-task intensity see higher AI exposure?

**If true**: Exposure assignment is driven by task structure, not by whether the occupation was "in trouble." Coefficient is causal.

**How to test**:
```python
# Regression 1: AI exposure ~ O*NET task characteristics
# Should find strong positive correlation with routine-task intensity

# Regression 2: Pre-treatment (2012) occupation health ~ AI exposure (2020+)
# Should find NO relationship — firms are not targeting troubled occupations
```

### Malign Story (Restructuring Targeting)

**Assumption**: Firms deploy AI in occupations they're planning to restructure or downsize.

**Evidence to collect**:
- Do AI-exposed occupations in a firm have worse subsequent employment outcomes?
- Do AI-exposed occupations see higher wage pressure or hiring freezes?
- Pre-treatment occupation health predicts AI exposure assignment?

**If true**: Exposure is endogenous; workers in targeted occupations may shift politics due to job insecurity, not due to AI itself.

**How to test**:
```python
# Regression 1: Subsequent employment loss ~ AI exposure
# If positive, suggests targeting

# Regression 2: Pre-treatment firm-occupation wages ~ AI exposure
# If lower-wage occupations get more AI, suggests cost-cutting motivation
```

### Recommended Evidence Package

Before finalizing main results, collect:

1. **Task-fit analysis**: Correlation between AI exposure and O\*NET task metrics
2. **Pre-treatment balance**: Does pre-AI-deployment occupation health predict exposure?
3. **Post-exposure outcomes**: Employment, wages, hiring in exposed occupations post-AI
4. **Occupation-level analysis**: Which occupations see highest AI exposure? Does it align with task structure?

If evidence supports benign story → coefficient interpretation is causal (or at least not obviously confounded by restructuring).

If evidence supports malign story → need to acknowledge: coefficient conflates AI exposure with employment risk; consider robustness checks.

---

## Recommended Specification Choice

### Primary: SPEC 2 (Firm-Year FE + Occupation-Year FE)

**Why**:
- Strongest causal narrative (triple-difference)
- Acceptable power for inference (feasible with ~2.3k DoF after absorbing FEs)
- Clear, defensible identification assumptions
- Mirrors Hampole et al. design (which also uses firm-year FEs)

**Implementation**:
```python
# Pseudo-code
import statsmodels.formula.api as smf

# Firm-year FE: Create indicator for each firm×year combination
# Occupation-year FE: Create indicator for each occupation×year combination
# Person FE: Individual fixed effects (built into regression)

spec_2 = smf.ols(
    "outcome ~ C(idpers) + C(firm_year) + C(occ_year) + exposure",
    data=shp_panel
).fit()
```

**Standard errors**: Cluster at person level (multiple observations per person)

**Reporting**:
- Main table: Spec 2 results
- Robustness: Spec 1 (additive), Spec 3 (person-year FE)
- Endogeneity discussion: Task-fit evidence, pre-treatment balance

---

### Secondary: SPEC 1 (Additive FE) as Robustness

**Use case**: 
- Conservative comparison (baseline)
- Shows power advantage if results robust
- Tests sensitivity to confounding assumptions

**Expect**: Larger coefficient if malign confounding; smaller if benign

---

### Tertiary: SPEC 3 (Person-Year FE) as Specification Check

**Use case**:
- Test whether results survive maximum FE specification
- Explore robustness to person-level time-varying confounders

**Expect**: Noisy estimates, wider confidence intervals (power loss)

---

## Implementation Roadmap

### Phase 1: Data Preparation
- [ ] Load SHP panel with exposure measures (ISCO 4-digit)
- [ ] Create firm-year indicators
- [ ] Create occupation-year indicators
- [ ] Construct outcome variables (vote choice, policy preferences, authoritarianism)
- [ ] Define person-level cluster variable for SE

### Phase 2: Baseline Estimates
- [ ] Run Spec 2 (primary) with employment status controls
- [ ] Run Spec 1 (robustness) for comparison
- [ ] Examine coefficient magnitudes, SEs, significance

### Phase 3: Endogeneity Evidence
- [ ] Merge O\*NET task characteristics with exposure data
- [ ] Regress AI exposure on task-replaceability scores
- [ ] Check pre-treatment balance (does past occupation health predict exposure?)
- [ ] Document evidence narrative

### Phase 4: Robustness Checks
- [ ] Spec 3 (person-year FE) — specification check
- [ ] Alternative outcome measures
- [ ] Subgroup analysis (by firm size, sector, age, education)
- [ ] Sensitivity to occupation-year FE specification (test using coarser occupation codes)

### Phase 5: Reporting
- [ ] Main results table (Specs 1, 2, 3)
- [ ] Endogeneity narrative with evidence
- [ ] Appendix: Power diagnostics, alternative specs, robustness

---

## Key Assumptions and Caveats

### Assumption 1: Firm-Year FE Absorbs All Firm-Wide Confounders

**What it buys**: Any shock or decision that affects all workers at firm $f$ in year $t$ equally is removed.

**What it doesn't buy**: Shocks specific to a firm-occupation pair (e.g., "firm decided to automate the call center specifically").

**Mitigation**: Provide evidence that occupation-specific deployment decisions are driven by task structure, not restructuring targets.

### Assumption 2: Occupation-Year FE Absorbs All Occupation-Wide Trends

**What it buys**: Nationwide labor market shocks specific to occupation $o$ in year $t$ are removed.

**Mitigation**: Robust to cross-national variation; applies at country level where SHP is located (Switzerland).

### Assumption 3: No Residual Individual × Year Confounding

**What it means**: Person $i$'s unobserved mood/sentiment doesn't change in year $t$ in a way that correlates with which occupations get AI in that year.

**Threat**: If person $i$ becomes a different person in year $t$ (life event, personal crisis) and happens to work in the occupation the firm deploys AI in, bias results.

**Mitigation**: Limited by sample size; can partially test with person-year FE (Spec 3), though power cost is high.

---

## Sample Size and Power Implications

**Conservative estimate** (using Spec 2):

- **Effective DoF for identification**: ~2,325 (after absorbing 43k FEs)
- **Number of firm-years with identifying variation**: 4,523
- **Expected precision**: Larger SEs than Spec 1; may need large effect sizes for significance
- **Recommendation**: Report 90% CIs in addition to 95% CIs; interpret magnitudes relative to domain knowledge

**Example**:
- If true effect of AI exposure on conservatism is 0.05 on a 0-10 scale, power is moderate
- If true effect is 0.01, power may be insufficient to detect at 5% significance
- Report both point estimate and CI for transparency

---

## References and Related Work

- **Hampole et al. (2025)**: Firm-level AI adoption and labor market outcomes (Hampole et al. use firm × occupation × year and occupation × year interactions)
- **Abadie et al. (2023)**: When should you use fixed effects vs. random effects in panel data?
- **Cameron & Miller (2015)**: Bootstrap and asymptotics in cluster-robust inference

---

## Specification Decision Log

**2026-04-10 (Today)**
- Completed sample size diagnostic on SHP + exposure merged data
- Found: 75% of firm-years have only 1 respondent; 81.9% of people never switch firms
- Decided: Firm-year FE is feasible but power-constrained (16.3% of firm-years have 2+ occupations)
- **Selected**: Spec 2 (firm-year FE + occupation-year FE) as primary
- **Rationale**: Best balance of identification (triple-difference logic) and power (acceptable DoF)

---

## Appendix A: Creating Fixed Effects Indicators in Python

```python
import pandas as pd

# Load panel data
df = pd.read_csv("shp_panel_with_exposure.csv")

# Create firm-year identifier
df['firm_year'] = df['firm_id'].astype(str) + '_' + df['year'].astype(str)

# Create occupation-year identifier  
df['occ_year'] = df['isco08_4d'].astype(str) + '_' + df['year'].astype(str)

# Verify counts
print(f"Unique persons: {df['idpers'].nunique()}")
print(f"Unique firm-years: {df['firm_year'].nunique()}")
print(f"Unique occupation-years: {df['occ_year'].nunique()}")
print(f"Total rows: {len(df)}")

# Fit model with statsmodels
import statsmodels.formula.api as smf

# Spec 2: Firm-year FE + Occupation-year FE + Person FE
model = smf.ols(
    "outcome ~ C(idpers) + C(firm_year) + C(occ_year) + exposure + controls",
    data=df
).fit(cov_type='cluster', cov_kwds={'groups': df['idpers']})

print(model.summary())
```

---

## Appendix B: Diagnostic Output (April 10, 2026)

**From: shp_sample_size_diagnostic.py**

```
Total rows: 199,324
Unique persons: 38,528
Rows with firm_id: 45,325 (22.7%)
Unique persons with firm_id: 9,895

Firm-year cells: 30,507
  - with 2+ respondents: 5,722 (18.8%)
  - with 2+ occupations: 4,523 (16.3%)
  - median respondents: 1
  - mean respondents: 1.49

Firm-occupation-year cells: 37,064
  - with 2+ respondents: 1,852 (5.0%)
  - median respondents: 1
  - mean respondents: 1.09

Person switchers (2+ firms): 1,794 (18.1%)
Person stayers (1 firm): 8,101 (81.9%)
```

---

**Status**: This specification is approved for implementation. Proceed to Phase 1 (data preparation) and Phase 2 (baseline estimates).
