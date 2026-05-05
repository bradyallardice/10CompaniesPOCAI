# Methodological Critique: Income Reversal Analysis
## Summary Report for Brady

---

## What I Did

You asked me to play the role of a methodological critic and identify gaps between your econometric findings (AI exposure predicts wage loss) and a valid political economy interpretation (wage loss causes political demand for policy change).

I've written three documents analyzing the income reversal analysis:

1. **`docs/methodological_critique_income_reversal.md`** (850 words)
   - Full critique identifying 5 critical gaps
   - Political economy interpretation requirements
   - Proposed empirical tests with expected results
   - Comprehensive reference for research design

2. **`docs/critique_executive_summary.md`** (1,100 words)
   - Quick version of gap analysis
   - Why each gap matters for political interpretation
   - Prioritized list of what to test next
   - Timeline and resource estimates

3. **`docs/robustness_testing_roadmap.md`** (1,200 words)
   - Detailed specifications for each test
   - Decision tree: What to test depending on your goals
   - Code examples (Python specs) for implementation
   - Expected results patterns for each scenario

All three are in `/docs/` and committed to git.

---

## The Five Critical Gaps (Executive Version)

### Gap 1: Mechanism Ambiguity (HIGHEST PRIORITY)
**Your finding:** 8.5% wage loss for exposed workers (17.6% for women, 15.8% for older workers)

**The problem:** Same coefficient consistent with 5 different mechanisms:
- Task displacement (AI automates worker's tasks)
- Occupational downgrading (worker demoted to lower-wage role)
- Bargaining power loss (worker loses wage negotiations)
- Hours reduction (worker works fewer hours)
- Selection on unobservables (low-ability workers assigned to AI roles)

**Why it matters:** Each mechanism implies different political demand:
- Displacement → retraining, education, tech regulation
- Downgrading → union strength, occupational protections, firm accountability
- Bargaining loss → labor movement, working-class solidarity
- Hours loss → labor standards, benefits preservation
- Selection → no real grievance

**Test:** Track job titles pre/post AI adoption within firms. Decompose wage loss into within-title (bargaining/hours) vs. between-title (downgrading) components.

---

### Gap 2: Selection on Unobservables (THREATENS EVERYTHING)
**Your finding:** Firm-Year FE controls for firm heterogeneity, but can't rule out time-varying worker selection

**The problem:** Firms might preferentially assign low-unobserved-ability workers to AI-exposed roles
- Example: Firm identifies data entry as routine, hires less-motivated workers, then automates
- You observe wage loss (real), but cause is worker quality, not AI treatment
- Same coefficient, different interpretation

**Why it matters:** If selection explains your result, all downstream analyses (heterogeneous effects, political outcomes) are unreliable.

**Test:** 
1. Plot pre-treatment wage trends for workers who will be exposed to AI (3 years before)
2. Run placebo test: assign fake exposure dates 1-2 years before actual
3. Estimate Altonji bounds on robustness to unobserved selection

**Expected:** If causal, trends should be flat pre-treatment and wage loss appears at treatment date. If selection, wage loss already appears before treatment.

---

### Gap 3: AI Exposure Measurement is Coarse
**Your finding:** AI exposure from job ads linked to firm×occupation×year level

**The problem:**
- Job ads lag actual implementation (1-2 year lag)
- Ads measure demand for AI roles, not displacement of existing tasks
- Discrete (post ads vs. don't), not continuous (task intensity)
- Only 6,139 of 45,325 person-years have exposure > 0; median = 0
- You're mostly comparing "firm posted AI ads" vs. "didn't"

**Why it matters:** If exposure is mismeasured, you're conflating task automation with firm selection into high-wage sector.

**Test:** 
1. Link job postings to individual respondents by role (not just occupation)
2. Vary exposure lag (t, t+1, t+2) and test effect persistence
3. Control for firm posting volume (large firms post more)

---

### Gap 4: Incomplete Heterogeneous Effects
**Your finding:** Gender and age splits (women: -17.6%, men: -1.7%; old: -15.8%, young: +2.3%)

**The problem:** Missing dimensions:
- Education/skill level (matters for retraining capacity)
- Sector (manufacturing AI ≠ finance AI ≠ retail AI)
- Firm size (SME flexibility ≠ large firm flexibility)

**Why it matters:** You need to identify politically vulnerable *clusters* (education × sector × size) to predict which groups will mobilize.

**Test:** Estimate β(AI exposure) for every cell in education × sector × firm-size matrix. Create heat map showing which groups suffer largest losses with sufficient sample size.

---

### Gap 5: Sample Selection Bias
**Your finding:** 37,279 person-years with complete data from SHP

**The problem:**
- ~75% of SHP respondents have no firm_id (structural missing data)
- Only 8-13% of SHP respondents per year are linked to firms
- Unknown who is missing: self-employed? Small-firm workers? Temp workers?
- Sample is likely biased toward large, structured employers

**Why it matters:** 
- If large firms (high-wage, AI-heavy) are over-represented, wage loss estimate is for large-firm workers
- If small firms (low-wage, routine-task-heavy) are under-represented, you're missing the most vulnerable
- Generalizability to full Swiss labor force is unclear

**Test:** 
1. Compare your sample to full SHP (unlinked) and Swiss census data
2. Check if exposure is correlated with firm size in your sample
3. Use inverse-probability weighting to adjust for selection

---

## What You Should Do Next (Prioritized)

### Phase 1: Robustness to Selection (Urgent — 2-3 weeks)
1. **Pre-trend check** (3-5 days)
   - Plot wage growth for future-treated workers 3 years before AI adoption
   - If trends parallel to untreated workers, selection is less likely
   
2. **Placebo test** (3-5 days)
   - Assign fake exposure 1-2 years before actual
   - Re-estimate with false treatment date
   - If wage loss appears in pre-treatment, selection is present
   
3. **Altonji bounds** (3-5 days)
   - Estimate robustness of wage loss to unobserved selection
   - Report: "Effect is robust if unobserved selection is < X times as strong as observed selection"

**Why first:** If selection explains your result, everything else is unreliable. Do this before moving forward.

---

### Phase 2: Mechanism Identification (Essential — 2-3 weeks)
1. **Job title tracking** (2-3 weeks)
   - Within each firm, classify worker transitions pre/post AI
   - Decompose: no title change | downgrade | exit
   - Re-estimate FY-FE separately by transition type
   
**Expected patterns:**
- If displacement: wage loss is in no-title-change group
- If downgrading: wage loss is in downgrade group
- If both: split the effect

**Why:** Political narrative depends entirely on this. Same wage loss means different policy demands depending on mechanism.

---

### Phase 3: Vulnerable Populations (Important — 1-2 weeks)
1. **Heterogeneous effects heat map** (1-2 weeks)
   - Estimate β(AI exposure) for every cell in education × sector × firm-size
   - Color code: red (large negative) to white (zero)
   - Highlight cells with (large effect, large sample, high exposure)
   
**Why:** Identifies politically consequential vulnerability clusters that will mobilize.

---

### Phase 4: Political Bridge (Critical — 4-6 weeks)
1. **Attribution experiment** (4-6 weeks)
   - Survey: "You lose 10% wage due to [AI | globalization | firm greed]"
   - Outcome: Policy demand (AI regulation, unions, retraining, redistribution)
   - Test: Does demand differ by attribution?
   
2. **Temporal dynamics** (1 week, uses existing SHP data)
   - Political outcome (vote, ideology, policy demand) vs. lagged wage loss
   - Test: Does response appear immediately (T+0) or delayed (T+1, T+2)?
   
**Why:** Without this, you have a wage coefficient but not a political story.

---

## Key Takeaways

### The Good News
Your econometric analysis is careful and the results are interesting:
- Income reversal properly diagnosed (compositional effect)
- Firm-Year FE specification is appropriate for within-firm causal effects
- Heterogeneous effects (by gender, age) suggest mechanism clues
- 8.5% wage loss is economically meaningful (CHF 486/month median wage)

### The Challenge
The coefficient alone doesn't tell the political story:
- You have a wage effect, not yet a political effect
- You know *that* workers suffer losses, not *why*
- You know *who* (women, older workers), but not *which sectors* or *education levels*
- You haven't measured political attribution or political demand

### The Path Forward
Do the tests in order:
1. Robustness to selection (makes or breaks credibility)
2. Mechanism (determines political narrative)
3. Heterogeneous vulnerability (identifies political coalition)
4. Political attribution + outcomes (validates story)

You're not starting from scratch. You have a solid empirical foundation. You just need to bridge from labor economics to political economy with explicit tests of the linking mechanisms.

---

## Files Created (All Committed to Git)

1. **`docs/methodological_critique_income_reversal.md`**
   - Full 850-word critique
   - 5 gaps with evidence
   - Proposed tests with expected results
   - Political economy interpretation requirements

2. **`docs/critique_executive_summary.md`**
   - Quick-read version
   - Priority decision tree
   - Timeline and resources

3. **`docs/robustness_testing_roadmap.md`**
   - Detailed test specifications
   - Decision tree by research goal
   - Python code examples
   - Expected result patterns

---

## Questions to Ask Yourself

1. **Mechanism:** Do you have access to job titles in SHP employment data? If yes, title-tracking test is feasible (2-3 weeks). If no, you'd need to proxy from job postings linkage (harder).

2. **Selection robustness:** How confident are you that the wage loss is causal? If unsure, do pre-trend check first (should take 1 week).

3. **Heterogeneity:** Does your SHP data include education variables? If yes, heat map is easy (1 week). If no, you'd need to merge external education data.

4. **Political outcomes:** Do you have vote choice / ideology measures in SHP? If yes, temporal dynamics test is trivial (1 week). If no, you'd need external data.

5. **Survey vs. observational:** Would an attribution experiment (4-6 weeks, requires fieldwork) be feasible, or would you prefer to infer political response from SHP panel data alone?

These questions determine the order and scope of your next work.

---

**Bottom line:** Your income reversal analysis is solid labor economics. To make it political economy, you need to explicitly test why workers lose wages (mechanism), confirm the effect is causal (selection), identify who is vulnerable (heterogeneity), and show that wage losses translate to political demand (attribution + outcomes). The tests are specified in the roadmap. Prioritize in order: selection → mechanism → heterogeneity → politics.

