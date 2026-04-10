# Robustness Testing Roadmap: Income Reversal Analysis

## Testing Strategy: Prioritized Decision Tree

```
FINDING: FY-FE coefficient = -0.085 (wage loss from AI exposure)

Question 1: Is this a REAL effect or SELECTION artifact?
│
├─→ TEST 1A: Pre-treatment wage trends
│    │ Plot wage growth in years t-3 to t-1 for workers treated in year t
│    │ Compare to never-treated workers
│    └─→ Diverge? → Selection present → Estimate bounds (Altonji)
│         Parallel? → More confidence in causation → Proceed to Q2
│
└─→ TEST 1B: Placebo test  
     │ Assign fake exposure dates 1-2 years before actual
     │ Re-estimate FY-FE with false treatment
     └─→ Effect appears in pre-treatment? → Selection present
         No pre-treatment effect? → Supports causation → Proceed to Q2

Question 2: WHY does wage loss occur? (Mechanism)
│
├─→ TEST 2A: Job title tracking
│    │ Within each firm, track job title transitions
│    │ Classify: No change | Downgrade | Exit | Entry
│    │ Re-estimate FY-FE separately by transition type
│    │
│    ├─→ No title change, wage loss → Task displacement or bargaining
│    ├─→ Downgrade, wage loss → Occupational sorting/demotions
│    ├─→ High exit rate → Firm contraction or worker dissatisfaction
│    └─→ All mechanisms visible?
│
└─→ TEST 2B: Wage decomposition
     │ Use Oaxaca-Blinder to decompose wage loss into:
     │   - Within-title wage change (bargaining/hours)
     │   - Composition change (title downgrade)
     │   - Returns to characteristics
     │
     ├─→ Most loss within-title? → Bargaining or hours
     └─→ Most loss between-title? → Downgrading

Question 3: WHO is vulnerable? (Heterogeneity)
│
├─→ TEST 3A: Full heterogeneous effects
│    │ Estimate β(AI exposure) for each cell:
│    │   Education (HS, Vocational, Bachelor, Graduate)
│    │   × Sector (Manufacturing, Finance, IT, Health, Services, Other)
│    │   × Firm size (SME, Large)
│    │
│    └─→ Heat map output: Identify cells with
│         (large negative β, large n, high AI adoption)
│         These are politically salient vulnerability clusters
│
└─→ TEST 3B: Occupational differences
     │ Does wage loss differ by occupational task structure?
     │ Hypothesis: "Routine-cognitive" jobs lose more than "analytical"
     │
     └─→ Merge O*NET task content; stratify by routine-intensity

Question 4: Do workers PERCEIVE and RESPOND politically? (Political Bridge)
│
├─→ TEST 4A: Attribution experiment
│    │ Survey vignette randomization:
│    │ "You experience 10% wage loss due to: [AI | Globalization | Firm]"
│    │ Outcome: Policy demand for each attribution
│    │
│    ├─→ Demand differs by attribution? → Narrative matters
│    │   If yes: use this framing in future mobilization
│    │   If no: wage loss alone drives demand (frame-insensitive)
│    │
│    └─→ Does demand differ by demographic (age, gender, education)?
│         → Identifies politically active constituencies
│
└─→ TEST 4B: Panel dynamics of political response
     │ In SHP, estimate political outcome (vote choice, ideology, policy demand)
     │ as function of: contemporaneous wage loss, lagged wage loss, cumulative loss
     │
     ├─→ Immediate response (T+0)? → Fast political reaction (salience)
     ├─→ Delayed response (T+1 to T+3)? → Slow learning/processing
     └─→ Cumulative response? → Political demand builds over years
```

---

## Test Specifications and Expected Results

### TEST 1A: Pre-Treatment Wage Trends

**What to do:**
```python
# For each worker, identify year T when they first have AI exposure > 0
# Plot mean ln(wage) in years T-3, T-2, T-1, T, T+1, T+2

df_treated = df[df['hampole_ai_exposure_avg_foy'] > 0].copy()
first_ai_year = df_treated.groupby('idpers')['year'].min().reset_index()
first_ai_year.rename(columns={'year': 'treatment_year'}, inplace=True)

# For each worker, compute relative year: year - treatment_year
df_treated = df_treated.merge(first_ai_year, on='idpers')
df_treated['relative_year'] = df_treated['year'] - df_treated['treatment_year']

# For workers with relative year in [-3, 2], get mean wage
trend = df_treated[df_treated['relative_year'].isin(range(-3, 3))].groupby(
    'relative_year'
)['outcome_log_income'].mean()

# Plot: Wage trend should be flat from -3 to -1, then drop at 0
# If drops start at -3, selection is present
```

**Expected result (if causal):**
- Flat wage trend in years T-3 to T-1
- Wage drop begins at T or T+1
- Drop persists through T+2

**Expected result (if selection):**
- Wage already declining in T-3 to T-1
- Drop continues or accelerates at T
- Workers sorted by prior low wage, not AI treatment

**Action:**
- If parallel: Confidence in causation increases → Proceed to mechanism tests
- If diverge: Selection is present → Estimate bounds using Altonji et al. (2005)
  - "Selection parameter": How much stronger must unobserved selection be than observed selection to explain away effect?
  - If small ratio: Effect is robust to unobserved selection
  - If large ratio: Effect is fragile

---

### TEST 1B: Placebo Test (False Exposure Dates)

**What to do:**
```python
df_placebo = df.copy()

# For each worker, assign FAKE treatment date 1 year before actual
df_placebo['placebo_treatment_year'] = df_placebo.groupby('idpers')[
    'year'
].transform(lambda x: x.min() - 1)

# Create fake exposure indicator: 1 if year >= placebo_treatment_year, 0 else
df_placebo['fake_exposure'] = (
    df_placebo['year'] >= df_placebo['placebo_treatment_year']
).astype(int)

# Re-estimate FY-FE with fake exposure
# Should be: β(fake_exposure) ≈ 0, NOT significant

# Then also estimate with actual exposure in same regression
# β(actual_exposure) - β(fake_exposure) = true causal effect
```

**Expected result (if causal):**
- β(fake_exposure) ≈ 0 or small and insignificant
- β(actual_exposure) ≈ -0.085 as before
- Difference is significant (true effect exists before placebo)

**Expected result (if selection):**
- β(fake_exposure) ≈ -0.04 or more (wage loss appears in pre-treatment period)
- Suggests workers were already on lower trajectory
- True causal effect is zero (or much smaller)

**Action:**
- If placebo is zero: Supports causation → Proceed
- If placebo is large: Selection is present → Bound the effect

---

### TEST 2A: Job Title Transitions

**What to do:**
```python
# Within each firm, for each person-year pair (t, t+1):
# Classify job title transition

# You need: job title variable in SHP (may need to construct from employment data)
# Or: Link to job postings by firm-person-year to infer job content

df_transitions = df.sort_values(['idpers', 'year']).copy()
df_transitions['title_next'] = df_transitions.groupby('idpers')['job_title'].shift(-1)

# Classify transitions
df_transitions['transition_type'] = 'no_change'
df_transitions.loc[df_transitions['job_title'] != df_transitions['title_next'], 'transition_type'] = 'changed'

# Within-firm wage percentile to classify downgrade:
wage_percentile_within_firm = df_transitions.groupby('firm_year')[
    'outcome_log_income'
].transform(lambda x: x.rank(pct=True))

# If worker's wage percentile drops >20pp: downgrade
df_transitions['downgrade'] = wage_percentile_within_firm.shift(1) - wage_percentile_within_firm > 0.20

df_transitions['transition_type'] = 'downgrade'

# Now re-estimate FY-FE separately by transition type:
for trans_type in ['no_change', 'downgrade', 'exit']:
    df_sub = df_transitions[df_transitions['transition_type'] == trans_type]
    # FY-FE regression
    # Report: β(AI exposure | transition_type)
```

**Expected results:**

If **task displacement mechanism:**
- β(AI | no_change) ≈ -0.085 (all loss from same-title wage cut)
- β(AI | downgrade) ≈ 0 (workers who downgrade already selected)

If **occupational downgrading mechanism:**
- β(AI | downgrade) ≈ -0.10 (large loss, from demotion)
- β(AI | no_change) ≈ -0.03 (small loss, residual)

If **bargaining/hours mechanism:**
- β(AI | no_change) ≈ -0.085 (all loss from negotiation weakness)
- β(AI | downgrade) ≈ -0.05 to -0.10 (some from demotion, some from bargaining)

**Action:**
- Pattern shows which mechanism dominates
- Politically interpret based on mechanism:
  - Displacement → "technology policy" frame
  - Downgrading → "firm discrimination" frame
  - Bargaining → "labor power" frame

---

### TEST 3A: Heterogeneous Effects Heat Map

**What to do:**
```python
# Create interaction terms
df['edu_sector_size'] = (
    df['education_level'] + '_' + df['sector'] + '_' + df['firm_size_cat']
)

# Fit FY-FE regression separately by group
results_het = {}
for group in df['edu_sector_size'].unique():
    df_group = df[df['edu_sector_size'] == group]
    # FY-FE regression
    # β_group = coefficient on AI exposure for this group
    results_het[group] = {
        'coef': β_group,
        'se': se_group,
        'n': n_group,
        'mean_exposure': exposure_group.mean()
    }

# Output: DataFrame with columns
# education, sector, size, coef, se, n, mean_exposure

# Create heat map: coef by education × sector (averaging across size)
pivot_table(results_het, index='education', columns='sector', values='coef')

# Color: Red (large negative) to White (zero) to Green (positive)

# Highlight cells with:
# - |coef| > 0.10 (large effect)
# - n > 100 (sufficient sample)
# - mean_exposure > 0.05 (actually exposed to AI)
```

**Expected results:**
You should see a pattern like:

|  | Manufacturing | Finance | IT Services | Health | Retail |
|---|---|---|---|---|---|
| High School | **-0.18** | -0.12 | -0.08 | -0.05 | **-0.20** |
| Vocational | -0.10 | -0.08 | -0.03 | -0.02 | -0.15 |
| Bachelor | -0.06 | -0.04 | +0.02 | +0.01 | -0.08 |
| Graduate | -0.02 | +0.01 | +0.05 | +0.02 | -0.03 |

(Example: High school in manufacturing/retail hit hardest; graduate degree in IT/finance protected or unaffected)

**Political interpretation:**
Cells in bold (large negative, large n) are:
1. Politically vulnerable (clear harm)
2. Numerous (politically consequential)
3. Potentially mobilizable (concentrated harm)

---

### TEST 4A: Attribution Experiment

**Survey Design:**

```
Imagine you experience a 10% wage loss over two years. 
The loss is due to: [RANDOMIZED]

A) Your company started using AI software that automated 
   many of your routine tasks. You now do lower-value work.

B) Global competition forced your company to cut costs. 
   Your wages were reduced to stay competitive.

C) Your company became very profitable but reduced wages 
   to increase shareholder returns.

Given this cause, how much would you support each policy?
(1 = Strongly against, 5 = Strongly support)

- Regulation of AI in the workplace
- Government-funded retraining programs
- Stronger labor unions
- Minimum wage increases
- Profit-sharing at companies
- Universal basic income
```

**Expected results:**

| Policy | AI Cause | Competition | Greed | Difference |
|---|---|---|---|---|
| AI regulation | 4.2 | 2.5 | 2.1 | 2.1** |
| Retraining | 4.1 | 3.9 | 2.8 | 1.3** |
| Unions | 3.5 | 3.8 | 4.3 | -0.8 |
| Min wage | 3.2 | 3.5 | 4.1 | -0.9 |
| Profit-sharing | 2.8 | 3.2 | 4.5 | -1.7** |

**Interpretation:**
- AI attribution uniquely drives support for AI regulation and retraining
- Greed attribution drives support for redistribution (profit-sharing, min wage)
- Different narratives → different political demands

**Action:** Use this to understand which narrative frames will resonate with affected workers.

---

### TEST 4B: Temporal Dynamics of Political Response

**Specification:**
```python
# Political outcome (in year t): 
# - Vote choice (left vs right)
# - Self-reported ideology (left-right scale)
# - Policy demand (union support, AI regulation, redistribution)

political_outcome_t = (
    α + 
    β0 * wage_loss_t +           # Contemporaneous
    β1 * wage_loss_t-1 +         # Lagged 1 year
    β2 * wage_loss_t-2 +         # Lagged 2 years
    γ * cumulative_wage_loss_t + # Cumulative to date
    δ * controls +
    ε
)

# Cluster by person; control for person FE and year FE
```

**Expected results (if immediate response):**
- β0 >> β1 >> β2
- Politics respond to contemporaneous wage loss
- Suggests high political salience/awareness

**Expected results (if delayed response):**
- β0 ≈ 0, β1 significant, β2 small
- Political response delayed by 1 year
- Suggests learning lag or organizing time

**Expected results (if cumulative):**
- γ significant, βτ small
- Response to total damage, not annual loss
- Political anger builds over time

**Action:** Informs campaign timing and political organization strategy.

---

## Quick Decision Tree: Which Tests Matter Most?

```
If your goal is: ACADEMIC VALIDITY
→ Priority: TEST 1A (pre-trends) + TEST 1B (placebo) + TEST 2A (mechanism)
   Reason: Robustness to selection and mechanism clarity are essential

If your goal is: POLICY IMPACT
→ Priority: TEST 3A (vulnerability map) + TEST 4A (attribution experiment)
   Reason: Need to know who is harmed and how to frame policy

If your goal is: POLITICAL PREDICTION
→ Priority: TEST 3A (vulnerability map) + TEST 4B (temporal dynamics)
   Reason: Need to know who mobilizes and on what timeline

If your goal is: EVERYTHING (complete story)
→ Do them in this order:
   1. TEST 1A + 1B (determines credibility of all downstream findings)
   2. TEST 2A (determines political narrative)
   3. TEST 3A (determines political coalition)
   4. TEST 4A + 4B (validates political story)
```

---

## Rough Timeline and Resource Estimate

| Test | Effort | Timeline | Data Needed |
|---|---|---|---|
| 1A (pre-trends) | 2-3 days | 1 week | Current data (SHP + exposure) |
| 1B (placebo) | 1-2 days | 3-5 days | Current data |
| 2A (job titles) | 1-2 weeks | 2-3 weeks | SHP employment history OR job postings linkage |
| 3A (heterogeneity) | 3-5 days | 1-2 weeks | Current data + education/sector variables |
| 4A (attribution) | 4-6 weeks | 8-12 weeks | Survey design, recruitment, fielding, analysis |
| 4B (temporal) | 2-3 days | 1 week | Current data (already have panel structure) |

**Minimum viable robustness (2-3 weeks):**
- 1A + 1B (selection robustness)
- 2A (mechanism)
- 3A (heterogeneity)

**Full story (4-5 months):**
- All tests above
- Includes survey fielding for attribution experiment

