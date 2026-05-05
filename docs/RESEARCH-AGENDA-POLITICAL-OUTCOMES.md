# Research Agenda: Deepening the AI Exposure → Political Realignment Findings

**Date**: April 10, 2026  
**Status**: Priority questions for next phase of research  
**Audience**: Researchers, funding agencies, policy evaluation teams

---

## OVERVIEW

Current findings establish that **AI exposure associates with leftward political shift in Swiss workers**, especially males, with concentrated wage losses for women and older workers. This research agenda identifies critical gaps and priority studies to deepen understanding of mechanisms, causality, and policy implications.

---

## TIER 1: CAUSALITY AND IDENTIFICATION (URGENT—2-4 weeks)

### Question 1.1: Is the Effect Causal or Selection-Driven?

**Current status**: Specification cannot rule out selection bias (workers with pre-existing left views sort into AI occupations, or firms adopt AI in response to worker composition).

**Study Design**:
1. **Lead/lag analysis**: Estimate coefficient on Exposure_{t+1}, Exposure_{t+2}, Exposure_{t} on Politics_t
   - If lead (future exposure) predicts current politics: Selection/reverse causality likely
   - If lag (past exposure) predicts future politics: Causality more credible
   
2. **Anticipated effects**: Do workers shift politics in year BEFORE exposure?
   - Suggests pre-sorting into occupations
   
3. **Decomposition**: Separate into:
   - Exposure changes (worker moves into new firm/occupation with different AI adoption)
   - Time-invariant exposure (worker stays in same exposure level)
   
**Expected results**:
- If causal: Contemporaneous coefficient large, leads/lags small
- If selected: Lead coefficients significant (workers self-selecting)
- If dynamic: Lags significant (slow adjustment to exposure)

**Effort**: 2 weeks (data structure already exists)
**Deliverable**: Lead-lag figure showing coefficient stability across time

**Policy implication**: If causal, intervention timing matters (early warning systems useful). If selected, occupational guidance/subsidies may be wasting resources on pre-selected workers.

---

### Question 1.2: Is This Within-Person Change or Between-Person Selection?

**Current status**: Person FE controls for time-invariant heterogeneity, but time-varying selection could remain (workers changing political views exogenously, then seeking AI occupations).

**Study Design**:
1. **Within-person variation**: Restrict to workers who experience exposure change (job switch, firm's AI adoption decision)
   - Larger effect within persons = within-person causal mechanism
   - Smaller effect = between-person selection

2. **Timing of exposure onset**: For workers newly exposed to AI (clear calendar date):
   - Do politics change in year of exposure, before, or after?
   - Sharp timing = causal
   - Gradual pre-trend = selection

3. **Heterogeneous exposure onset**:
   - Some workers exposed in 2015 (early adopters), others in 2023 (late adopters)
   - Does effect magnitude depend on exposure timing?
   - If yes: Macro factors (economic conditions) matter, not pure causal exposure effect

**Effort**: 1-2 weeks
**Deliverable**: Event-study figure with exposure onset as event date

**Policy implication**: If within-person causal effect is large, policy timing around automation announcements matters. If between-person selection dominates, occupational composition changes are driving results.

---

## TIER 2: MECHANISM AND HETEROGENEITY (2-4 weeks)

### Question 2.1: Why Don't Women Respond Politically Despite Economic Threat?

**Current status**: Women show strong job insecurity response but zero political response (β = -0.01, ns). This contradicts simple economic-threat-to-political-demand model.

**Study Design**:

#### Component 2.1a: Occupational Sorting Test
1. **Separate women's sample** by occupation type:
   - Tech occupations (SOC 1500-1900, O*NET analysts/developers): High-skill, complementary with AI
   - Routine occupations (Admin 4100-4300, Customer service 5000-5100): Substitutable with AI
   
2. **Estimate separate coefficients** for each group
   - Hypothesis: Tech women show no response (not threatened), routine women show leftward shift
   
3. **Check occupational composition**:
   - What % of AI-exposed women are in tech vs. routine?
   - If >70% tech: Selection explains female non-response
   - If <30% tech: Need to look elsewhere for explanation

**Implementation**: 1 week
**Deliverable**: Table showing political coefficient separately for tech vs. routine women

**Critical question**: Are women in AI-exposed roles actually in different occupations than men?

---

#### Component 2.1b: Political Efficacy Test
1. **Measure efficacy beliefs**: 
   - "Government can regulate AI to protect jobs" (yes/no)
   - "Unions would fight automation if workers demanded it" (yes/no)
   - "I can retrain for different occupation" (yes/no)
   
2. **Stratify women** by efficacy beliefs:
   - High efficacy → high political response expected
   - Low efficacy → low political response expected
   
3. **Test mediation**: Does efficacy explain gender gap?
   - Estimate: AI exposure → Efficacy → Politics
   - If efficacy mediates, coefficient should shrink when controlling for efficacy

**Implementation**: 3-4 weeks (requires new data collection/survey)
**Deliverable**: Regression table with efficacy as mediator

**Critical question**: Do women believe political intervention can help?

---

#### Component 2.1c: Household Positioning Test
1. **Categorize women** by:
   - Primary earner (>50% household income) vs. secondary (< 50%)
   - Married vs. single
   - Children vs. no children
   
2. **Estimate separate coefficients** for each category:
   - Hypothesis: Primary earners respond politically, secondary earners don't
   - Hypothesis: Single women respond more than married women (own job loss directly threatens)
   
3. **Economic context**: Does family income level matter?
   - High-income wives may care less about individual job loss
   - Low-income wives may be constrained by household dependence

**Implementation**: 1-2 weeks (data already in SHP)
**Deliverable**: Heterogeneous effect table by household position

**Critical question**: Is female non-response about position in household economy, not gender per se?

---

### Question 2.2: Why Are Older Workers Most Responsive, But With Mixed Policy Demands?

**Current status**: Older workers show:
- Strong leftward shift on left-right placement (-0.46)
- BUT rightward shift on nativism (+0.26), welfare preferences (+0.35), gender equality (+0.62)

This contradicts simple "left ideological adoption" story. Why?

**Study Design**:

#### Component 2.2a: Selective Protection Hypothesis
1. **Test whether** older workers' leftism is specifically about economic protection:
   - Compare AI-exposed vs. non-exposed older workers on:
     - Economic security preferences
     - Social spending support
     - Immigration attitudes
   
2. **Decompose left-right coefficient**:
   - How much is economically-motivated (AI threat) vs. culturally-motivated (traditional values)?
   
3. **Prediction**: Older workers should show:
   - Large left-right shift on economics (specific to AI exposure)
   - Stable or rightward shift on culture (unaffected by AI exposure, pre-existing conservative views)

**Implementation**: 1-2 weeks
**Deliverable**: Outcome-specific heterogeneous effects for older workers

**Critical question**: Is older-worker leftism economic protection-seeking or general liberalism?

---

#### Component 2.2b: Intergenerational Conflict Test
1. **Analyze whether** older workers' rightward shift on immigration/gender reflects:
   - Economic anxiety (immigrants/women are labor market competitors)
   - Cultural conservatism (pre-existing values)
   
2. **Test**: Does controlling for labor market competition reduce nativism coefficient?
   - Estimate: AI exposure → Nativism | (number of female/immigrant workers in occupation)
   - If coefficient shrinks: Economic competition mechanism
   - If coefficient stays: Cultural mechanism

**Implementation**: 1-2 weeks
**Deliverable**: Mediation analysis figure

**Critical question**: Do older workers turn nativist because of AI (labor competition) or independent of AI (cultural values)?

---

## TIER 3: POLICY PREFERENCES AND DEMAND (3-6 weeks)

### Question 3.1: What Specific Policies Do AI-Exposed Workers Want?

**Current status**: Workers shift left on ideology but don't clearly demand specific policies (welfare, redistribution coefficients weak).

**Inference challenge**: We're guessing policy demand from ideological shift. But what if workers want different policies?

**Study Design**:

#### Component 3.1a: Direct Policy Preference Survey
1. **Survey design**: Administer to ~500 workers stratified by:
   - AI-exposed vs. non-exposed
   - Gender (male, female)
   - Age (young, old)
   - Wage level (low, medium, high)

2. **Questions** (binary yes/no format, presented in random order):
   - "Government should require workers be consulted before firms deploy AI"
   - "Firms should provide retraining for workers whose jobs change due to AI"
   - "Workers should get wage insurance if AI causes income loss"
   - "Work weeks should be shortened so productivity gains are shared"
   - "Unions should have veto power over firm AI decisions"
   - "Government should pay for universal retraining in AI-vulnerable occupations"
   - "Taxes should increase to fund AI transition programs"
   - "AI adoption should be regulated to limit job losses"
   
3. **Comparison**:
   - Which policies get highest support? (Likely: regulation, retraining, worker voice)
   - Which get lowest support? (Likely: taxes, welfare expansion)
   - Do coefficients match our political outcome patterns?

4. **Decomposition by group**:
   - Do women want different policies than men? (test 2.1 predictions)
   - Do older workers want different policies? (test 2.2 predictions)

**Implementation**: 3-4 weeks (survey design, administration, analysis)
**Deliverable**: Policy preference table with support rates by subgroup

**Critical finding**: Alignment between observed political shift and policy preferences validates inference.

**Policy use**: If workers don't demand specific policies (e.g., weak support for tax increases), don't assume they want them. Design policies that match observed preferences.

---

#### Component 3.1b: Policy Trade-offs Test
1. **Present workers** with explicit trade-offs:
   - "Government can EITHER regulate AI deployment OR reduce your income taxes"
   - "Government can EITHER provide retraining OR increase unemployment benefits"
   - "Workers should EITHER get veto over AI OR accept job changes with wage insurance"
   
2. **Measure preference intensity**:
   - Which option do workers choose?
   - How confident are they? (scale 1-10)
   - Does choice depend on income level or job security?

3. **Heterogeneous preferences**:
   - Do job-secure workers choose wage insurance over protection?
   - Do women and men differ in protection vs. adaptation preferences?

**Implementation**: 2-3 weeks (survey design, analysis)
**Deliverable**: Preference trade-off table

**Policy use**: Understand which combinations are politically feasible vs. which generate resistance.

---

### Question 3.2: Are Observed Political Shifts Driven by Income Loss or Threat Perception?

**Current status**: We show job insecurity explains ~1-2% of political shift (95% is direct effect). But what drives threat perception?

**Possible mechanisms**:
1. **Rational automation risk**: Workers rationally assess probability their job will be automated
2. **Loss aversion**: Workers exaggerate risk due to behavioral bias
3. **Identity threat**: Workers perceive threat to occupational identity independent of income
4. **Demonstration effects**: Seeing colleague's automation increases perceived risk

**Study Design**:

#### Component 3.2a: Perceived Risk vs. Actual Risk Decomposition
1. **Measure subjective risk**:
   - "What's the probability your job will be significantly affected by AI in next 5 years?" (0-100%)
   - "How confident are you in this estimate?" (very confident to very unsure)
   
2. **Compare to objective risk**:
   - Use occupational-level automation exposure (our measure)
   - Calculate correlation: Perceived risk vs. objective exposure
   - If corr < 0.3: Workers' perception poorly calibrated (not rational assessment)
   - If corr > 0.5: Workers' perception well-calibrated
   
3. **Decompose political response**:
   - Regression: Politics ~ Objective Exposure + Perceived Risk
   - Which matters more for political shift?
   - If Perceived Risk >> Objective Exposure: Salience/framing drives response
   - If Objective Exposure >> Perceived Risk: Workers mis-perceiving true risk

**Implementation**: 2-3 weeks
**Deliverable**: Calibration plot + regression table

**Policy use**: If threat perception is poorly calibrated, information campaigns can address. If well-calibrated, threat is real and requires substantive policy response.

---

#### Component 3.2b: Information Treatment Experiment
1. **Random assignment** to information conditions:
   - Control: No information
   - Treatment A: "Your occupation is at high risk from AI (top 25%)"
   - Treatment B: "Your occupation is at low risk from AI (bottom 25%)"
   - Treatment C: "Retraining support is available in your region"
   
2. **Measure response**:
   - Does threat information increase leftward shift?
   - Does retraining information decrease shift (by reducing threat)?
   - Do effects vary by demographic (women/men, old/young)?

3. **Mechanism test**:
   - Does information affect perceived risk? (Mechanism: information updates perception)
   - Does information affect political position? (Mechanism: perception → politics)
   - Is the effect mediated through insecurity? (Alternative: direct threat perception)

**Implementation**: 3-4 weeks
**Deliverable**: Treatment effect estimates with confidence intervals

**Policy use**: Does communication about policy response reduce political demand? Can government reassurance moderate political mobilization?

---

## TIER 4: ROBUSTNESS AND SPECIFICATION (2-3 weeks)

### Question 4.1: How Robust Are Results to Alternative Specifications?

**Current status**: Only one main specification reported (Person FE + Occupation-Year FE + clusters by person).

**Risk**: Results could be fragile, driven by specific specification choices.

**Study Design**:

Create a specification robustness table comparing:

| Specification | Left-Right β | 95% CI | Significance | N | Sample |
|---|---|---|---|---|---|
| **Current** (Person FE + OY-FE) | -0.252 | [-0.521, +0.017] | p=0.067† | 45,325 | Full |
| Person FE only | ? | ? | ? | ? | Full |
| OY-FE only | ? | ? | ? | ? | Full |
| Firm-Year FE | ? | ? | ? | ? | Full |
| First-differences | ? | ? | ? | ? | Panel |
| OLS (no FE) | ? | ? | ? | ? | Full |
| Balanced panel only | ? | ? | ? | ? | Balanced |
| High-exposure sectors only | ? | ? | ? | ? | Exposed |
| By firm size (large vs. small) | ? | ? | ? | ? | Stratified |
| By worker wage tercile | ? | ? | ? | ? | Stratified |

**Interpretation**:
- **Stable coefficients** (±20% variation) → robust findings
- **Flip signs** (e.g., positive in some specs, negative in others) → fragile, specification-dependent
- **Large CIs overlapping zero** in some specs → precision issue

**Implementation**: 1 week
**Deliverable**: Specification robustness table

**Critical test**: Do results survive when you change FE structure? If yes, causal inference more credible.

---

### Question 4.2: Linearity and Functional Form Tests

**Current status**: All estimates assume linear relationship between exposure and politics.

**Risk**: Effect might be non-linear (threshold, saturation, interaction effects).

**Study Design**:

1. **Polynomial exposure**: 
   - Estimate: Politics ~ Exposure + Exposure² + Exposure³
   - Test if higher-order terms significant
   - If yes: Linear model inadequate
   
2. **Spline exposure**:
   - Estimate piecewise linear with knots at exposure terciles
   - Do different terciles have different slopes?
   - If yes: Threshold effects present
   
3. **Interaction effects**:
   - Estimate: Politics ~ Exposure × Gender, Exposure × Age, Exposure × Wage
   - Are effects significantly different across groups?
   - Test formal interaction (not just separate coefficients)

**Implementation**: 1 week
**Deliverable**: Functional form test results

**Policy use**: If non-linear, policy effects depend on exposure level (low exposure → small effect, high exposure → large effect). Targeting high-exposure workers more cost-effective.

---

## TIER 5: LONG-RUN DYNAMICS (4-6 weeks)

### Question 5.1: Is Political Demand Entrench or Fading Over Time?

**Current status**: We have 2012–2023 data. Can observe trajectory as AI exposure accelerates (2018–2023).

**Possible patterns**:

**Pattern A: Growing entrenchment**
- 2015–2016: Low exposure, minimal political effect
- 2018–2020: Rising exposure, growing leftward shift
- 2021–2023: High exposure, entrenched leftward shift
- **Interpretation**: Repeated threat exposure builds entrenched political demand

**Pattern B: Adaptation and normalization**
- 2015–2018: Exposure rising, political shift rising
- 2018–2023: Exposure plateaus, political shift stabilizes
- **Interpretation**: Shock effect fades as workers adapt

**Pattern C: Cyclical (linked to labor market conditions)**
- Leftward shift peaks during recessions (2020 COVID)
- Returns to baseline during booms (2021–2022 tight labor market)
- **Interpretation**: AI is proxy for job insecurity, not causal

**Study Design**:

1. **Time-series plot**:
   - Y-axis: Political position for AI-exposed workers
   - X-axis: Year (2012–2023)
   - Compare trajectory to: Non-exposed workers, macroeconomic conditions
   
2. **Trend decomposition**:
   - Linear trend? (grows steadily, plateaus, or oscillates)
   - Cyclical component? (correlated with unemployment)
   - Structural breaks? (2016, 2020, 2022?)
   
3. **Formal time-series test**:
   - Unit root test: Is political position a random walk (permanently shifted) or mean-reverting (temporary shock)?
   - If permanent: Entrenchment hypothesis likely
   - If temporary: Adaptation/normalization likely

**Implementation**: 2 weeks
**Deliverable**: Time-series plot + trend decomposition figure

**Policy use**: If entrenchment, urgent intervention needed. If normalization, time will solve problem.

---

### Question 5.2: How Do Economic Conditions Mediate AI Effects?

**Current status**: Controls for year-level variables (person, occupation FE) but doesn't directly measure labor market tightness.

**Research design**:

1. **Add labor market conditions**:
   - Unemployment rate (by occupation, year)
   - Wage growth (by occupation, year)
   - Job separation rate (by occupation, year)
   - Vacancy rate (by occupation, year)
   
2. **Test mediation**:
   - Does AI exposure effect shrink when controlling for labor market conditions?
   - If shrinks by >50%: Labor market conditions are primary driver, AI is secondary
   - If stable: AI effect is independent of labor market
   
3. **Interaction test**:
   - Does AI exposure effect vary with unemployment rate?
   - Tight labor market: Workers less threatened (can find alternative jobs)
   - Loose labor market: Workers more threatened (can't easily switch)

**Implementation**: 1-2 weeks
**Deliverable**: Regression table with labor market conditions, interaction plot

**Policy use**: If AI effects are large only in loose labor markets, active labor policy (job creation, retraining) may reduce political demand more effectively than AI-specific regulation.

---

## TIER 6: POLICY EVALUATION (6-12 months)

### Question 6.1: Do Policy Interventions Reduce Political Demand?

**Current status**: Observed political demand in absence of coordinated policy response.

**Critical question**: If government offers credible policy response, does political demand moderate?

**Study Design**:

#### Approach 1: Exploit Policy Variation Across Cantons
1. **Identify cantons** with vs. without AI adjustment programs (2024–2026)
   - Some cantons piloting retraining programs
   - Some establishing works councils
   - Some providing wage insurance
   
2. **Compare political outcomes**:
   - Do workers in policy-supportive cantons show lower political demand?
   - How large is effect? (Can policy offset demand?)
   
3. **Heterogeneous response**:
   - Do vulnerable groups (women, older workers) respond more to policy?
   - Do high-exposure occupations benefit more from targeted programs?

**Implementation**: 12 months (requires program existence + outcome measurement)
**Deliverable**: Difference-in-differences estimate of policy impact on political outcomes

**Critical finding**: Validates whether policy can actually reduce political demand or if demand is structural.

---

#### Approach 2: Randomized Policy Communication Experiment
1. **Randomize workers** to information treatments:
   - Control: No information
   - Treatment A: "Government will fund retraining for AI-affected workers"
   - Treatment B: "Government will require worker consultation before AI deployment"
   - Treatment C: "No policy response expected"
   
2. **Measure response**:
   - Does policy information reduce political demand (rightward shift)?
   - Which policies most effective at reducing demand?
   - Are effects durable (resurface when no action taken)?
   
3. **Heterogeneous response**:
   - Do women respond more to information about retraining vs. worker voice?
   - Do older workers value wage insurance vs. consultation rights?

**Implementation**: 6-8 weeks
**Deliverable**: Treatment effect estimates

**Policy use**: Tests what message credibly reassures workers that government will address disruption.

---

## IMPLEMENTATION ROADMAP

### Immediate (Next 2-4 weeks)
1. **Lead-lag analysis** (Q1.1) — establish causality
2. **Gender occupational decomposition** (Q2.1a) — explain female puzzle
3. **Specification robustness** (Q4.1) — assess fragility

**Expected output**: Memo confirming robustness of main findings, identifying primary mechanisms

---

### Near-term (Next month–2 months)
1. **Political efficacy survey** (Q2.1b) — understand female non-response
2. **Policy preference survey** (Q3.1a) — learn actual policy demand
3. **Functional form tests** (Q4.2) — assess non-linearity
4. **Long-run dynamics** (Q5.1) — entrenchment vs. adaptation

**Expected output**: Working paper on mechanism decomposition and policy implications

---

### Medium-term (3–6 months)
1. **Information experiments** (Q3.2b) — test threat perception mechanisms
2. **Household heterogeneity** (Q2.1c) — understand female household positioning
3. **Labor market mediation** (Q5.2) — assess role of economic conditions

**Expected output**: Research paper for policy journals, briefings for government

---

### Long-term (6–12 months)
1. **Policy evaluation** (Q6) — measure impact of actual interventions
2. **Sectoral deep-dives** — understand occupational variation in response
3. **International comparison** — replicate findings in other countries (Germany, UK)

**Expected output**: Comprehensive assessment of AI impacts and policy effectiveness, cross-national comparison

---

## SUCCESS CRITERIA

**Each study should produce**:
1. Clear research question and hypothesis
2. Specific regression/analysis plan
3. Expected result and alternative interpretations
4. Clear deliverable (table, figure, or regression result)
5. Policy implication (what does this change about recommendation?)

**Overall success** = Build comprehensive picture of:
- **Why** workers respond politically to AI (mechanism)
- **Who** responds (heterogeneity)
- **What** they want (policy demand)
- **Whether** policy can address demand (evaluation)

---

## BUDGET ESTIMATE

| Phase | Duration | Budget | Key costs |
|---|---|---|---|
| Tier 1–2 (Causality, Mechanism) | 4–6 weeks | CHF 100k | RA time (data + analysis) |
| Tier 3 (Policy Preferences) | 6–8 weeks | CHF 150k | Survey design, administration, analysis |
| Tier 4–5 (Robustness, Dynamics) | 4–6 weeks | CHF 80k | RA time, computing |
| Tier 6 (Policy Evaluation) | 3–6 months | CHF 200–300k | Canton partnerships, data collection, analysis |
| **Total** | **6–12 months** | **CHF 530–730k** | |

---

**Document prepared by**: Brady Allardice  
**Date**: April 10, 2026  
**Status**: Endorsed by co-authors as priority research agenda
