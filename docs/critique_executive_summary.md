# Executive Summary: Methodological Critique of Income Reversal Analysis

## Quick Version (Elevator Pitch)

Your income reversal analysis is econometrically sound but politically incomplete. You've proven that **AI exposure predicts wage loss within firms** (especially for women and older workers), but you haven't proven **why** this happens or **how** it translates to political demand. Three evidence gaps are blocking interpretation as a political economy story:

1. **You don't know the mechanism:** Is it task displacement, occupational downgrading, bargaining power loss, or hours reduction? Different mechanisms imply different political demands.
2. **You can't rule out selection:** Could low-ability workers self-select into AI roles? Your pre-treatment trends are untested.
3. **You haven't measured political outcomes:** You have wage loss. You need political attribution and policy demand to complete the story.

## Most Critical Issue: Mechanism Ambiguity

**The Problem:**
You observe an 8.5% wage loss in aggregate (17.6% for women, 15.8% for older workers) under Firm-Year FE. But the same coefficient is consistent with:

- **Scenario A (Task Displacement):** AI automates the worker's primary tasks. Worker reassigned to lower-value activities within same job title. Result: same title, lower wage.
- **Scenario B (Occupational Downgrading):** Worker is moved to a lower-wage job title within firm (e.g., from "Senior Developer" to "Technical Support"). Result: new title, lower wage.
- **Scenario C (Bargaining Erosion):** Worker retains all tasks but loses wage negotiations because AI makes them more substitutable. Result: same title, same tasks, lower wage.
- **Scenario D (Hours Reduction):** Worker works fewer hours for same hourly wage. Result: same title, same wage, fewer hours.
- **Scenario E (Selection):** Lower-ability workers sort into AI-exposed roles. Wage loss reflects their ability, not AI treatment. Result: no causal effect on workers.

**Why This Matters for Politics:**
- A: Demand for retraining + anger at technology
- B: Demand for occupational protections + firm wage discrimination anger  
- C: Demand for union strengthening + working-class solidarity
- D: Demand for labor standards (minimum hours, benefit preservation)
- E: No real grievance (outcome reflects ability)

**Your Current Evidence:**
You can't distinguish these because you only observe:
- Person wage loss
- Firm AI exposure
- Person demographics (age, gender)

You don't observe:
- Individual job titles (pre/post AI)
- Task content (what actually changed?)
- Hours worked (hours reduction vs. wage loss?)
- Worker transitions (did they move to different title?)

**The Test You Need:**
Track job titles within firms before/after AI adoption. Re-estimate the wage loss separately for:
- Workers with unchanged titles (rules out downgrading; suggests displacement/bargaining/hours)
- Workers with downgraded titles (occupational sorting)
- Workers who exit firms (firm contraction vs. displacement)

This decomposition is critical for political prognosis.

---

## Second Critical Issue: Selection on Unobservables

**The Problem:**
Your Firm-Year FE specification controls for observable worker characteristics (age, gender, employment status) and implicitly controls for time-invariant firm characteristics. But it cannot rule out time-varying selection: firms may preferentially assign low-unobserved-ability workers to AI-exposed roles.

**Example:**
Suppose a firm identifies that data entry is routine and hires less-motivated/lower-skilled workers for data entry roles. Then the firm announces AI will automate data entry and reassigns those workers. In your data:
- Workers assigned to "AI exposure" (data entry pre-AI)
- Workers experience wage loss (downgrade to lower role post-AI)
- You estimate FY-FE effect: wage loss = -8.5%
- But true causal effect is zero (workers were always low-ability)

**Your Current Evidence:**
You can't distinguish selection from causation because you only observe:
- Person wages (before/after AI exposure assignment)
- AI exposure assignment

You don't observe:
- Pre-treatment wage trends (do future AI-exposed workers already earn less before AI?)
- False exposure dates (placebo test)
- Worker exit patterns (do treated workers leave at higher rates?)

**The Test You Need:**
1. **Pre-trend check:** For workers who will experience AI exposure in year T, plot their wage growth in years T-3, T-2, T-1. If they already had lower wage growth before AI, selection is present.
2. **Placebo test:** Randomly assign "fake" AI exposure dates 1-2 years before actual adoption. Re-estimate FY-FE with false treatment dates. If wage loss appears in pre-treatment period, selection is driving your results.
3. **Bounds estimation:** Use Altonji et al. (2005) selection-on-observables ratio to quantify robustness. If small changes in unobserved selection assumptions dramatically change your coefficient, the result is fragile.

---

## Third Critical Issue: Measurement of AI Exposure

**The Problem:**
Your AI exposure measure comes from job advertisements linked to firms. This is at least one (or two) steps removed from actual AI implementation:

1. **Time lag:** Firms post AI-required jobs *after* internal adoption. 1-2 year lag is plausible.
2. **Measurement of demand, not displacement:** A firm posting 5 "AI Engineer" positions doesn't tell you whether 10 existing engineers had tasks automated.
3. **Selection into advertising:** High-status firms advertise AI heavily (signal innovation); low-status firms adopt quietly.
4. **Discrete vs. continuous:** Your exposure distribution shows:
   - Median exposure = 0.0000
   - 75th percentile = 0.0000  
   - Only 6,139/45,325 person-years (13.5%) have exposure > 0
   
   This means you're mostly comparing "firm posted AI ads" vs. "firm didn't," not "worker's tasks changed" vs. "didn't."

**Political Consequence:**
If exposure is mismeasured, you may be confounding:
- "Firm adopts AI" (your measure)
- "Worker's tasks actually change" (true treatment)

If many firms post AI ads without actually changing worker tasks, your coefficient reflects signaling effects or selection into high-wage firms (good for worker) confounded with task automation (bad for worker).

**The Tests You Need:**
1. **Individual-level exposure:** If you have job postings data and can link to respondents by title + firm (not just occupation + firm), estimate exposure at individual level. Heterogeneity within occupation would validate that exposure varies meaningfully.
2. **Timing variation:** Lag exposure measure (t, t+1, t+2) and test whether effects persist. If true task displacement, the lag structure is informative (immediate vs. delayed damage).
3. **Firm volume control:** Control for firm's total job posting volume (large firms post more). If coefficient shrinks when you control posting volume, exposure is partly a "firm size" effect (not AI).

---

## Political Bridge Gap: No Political Outcomes Yet

**Current State:**
You have wage loss coefficients. You don't have political outcomes.

Your paper plan bridges from labor market effects (wage loss) to political outcomes (vote, ideology, policy demand). This is essential but not yet empirical.

**The Missing Steps:**
1. **Worker attribution:** Does the worker who experiences wage loss *attribute* it to AI? Or to "the boss cut my pay"? Or to "the economy is bad"? 
   - SHP probably doesn't ask this
   - Survey experiment could address it (show vignettes with different attributed causes; ask policy demand)

2. **Political response timing:** When does wage loss translate to political demand?
   - Immediately (T+0)?
   - After cumulative losses (T+3)?
   - Only after workers organize (event-driven)?
   
   Your data enables this test if you estimate political outcomes lagged relative to wage loss.

3. **Remedy selectivity:** Do different mechanisms demand different policies?
   - If task displacement: retraining, education, technology investment  
   - If downgrading: union strength, wage floors, occupational licensing
   - If bargaining loss: unions, profit-sharing, redistribution
   
   Your heterogeneous effects (by gender, age) might proxy for mechanism (women/older workers in different task structures), but it's speculative.

**The Tests You Need:**
1. **Survey experiment (external):** Show SHP respondents vignettes: "You experience a 10% wage loss because [AI / globalization / firm greed / bad business]." Elicit policy demand. Does demand differ by attribution?
2. **Timing analysis (use SHP panel):** Estimate political outcome (vote, ideology) as function of lagged wage loss. Does it appear at T+1, T+2, T+3?
3. **Heterogeneous political response:** Do women and older workers (who suffer larger wage losses) have different political responses? If yes, suggests mechanism-specific political mobilization.

---

## What You Should Do Next (Priority Order)

### Phase 1 (Robustness to Selection): Urgent
- [ ] Plot pre-treatment wage trends for workers who will experience AI exposure
- [ ] Run placebo test with false exposure dates 1-2 years before actual
- [ ] Estimate Altonji bounds to quantify robustness to unobserved selection

**Why:** If selection on unobservables explains your wage loss, everything downstream (heterogeneous effects, political analysis) is unreliable.

### Phase 2 (Mechanism): Essential for Political Interpretation
- [ ] Track job title transitions within firms before/after AI adoption
- [ ] Decompose FY-FE wage loss into:
  - Within-title wage change (bargaining/hours)
  - Between-title transition wage (downgrading)
  - Exit rates
- [ ] Re-estimate heterogeneous effects by mechanism

**Why:** Same wage loss coefficient means different political demands depending on mechanism.

### Phase 3 (Heterogeneous Vulnerability): Important for Political Targeting
- [ ] Estimate heat map: education × sector × firm size heterogeneity
- [ ] Identify cells with (large negative effect, large sample, high AI adoption)
- [ ] These are politically salient vulnerability clusters

**Why:** You need to know which demographic/sector groups will mobilize politically.

### Phase 4 (Political Bridge): Critical for Completeness
- [ ] Design survey experiment on wage loss attribution
- [ ] Estimate lags of wage loss on political outcomes (vote, ideology, policy demand)
- [ ] Test whether political response differs by mechanism or demographic group

**Why:** Without this, you don't have evidence that wage loss *causes* political demand.

---

## Bottom Line

Your econometric analysis is careful and the results are interesting. The income reversal is a sophisticated finding about compositional effects. But it remains a *labor market* finding, not yet a *political economy* finding.

**Current state:** "AI exposure predicts wage loss within firms."

**Needed for political story:** "Workers experiencing wage loss due to [mechanism] attribute harm to AI and demand [policy]."

The gap between these is large but bridgeable with the tests above. Priority order is:

1. **Select on unobservables** (threatens everything downstream)
2. **Mechanism decomposition** (determines political demand type)
3. **Heterogeneous vulnerability** (determines political coalition)
4. **Political outcomes** (validates the whole story)

Don't rush to political analysis until 1-2 are addressed.

