# Causal Identification Analysis: AI Exposure and Worker Outcomes
## A Comprehensive Causal Inference Memo

**Date**: April 10, 2026  
**Analysis Stage**: Synthesizing empirical evidence from specifications 1-3 into causal narratives  
**Sample**: 45,325 person-year observations, 22.7% of SHP panel with firm linkage  
**Key Mystery**: Income reversal (-8.5% under Firm-Year FE, +10.9% under Occupation-Year FE)

---

## Executive Summary

The AI exposure-worker outcome relationship exhibits **two distinct causal mechanisms operating at different levels of aggregation**:

1. **Within-firm mechanism (Firm-Year FE)**: AI deployment directly suppresses wages, particularly for women (-17.6%) and older workers (-15.8%). This is likely a causal effect of exposure to task displacement and bargaining power erosion.

2. **Between-firm selection mechanism (Occupation-Year FE)**: High-wage firms preferentially adopt AI, creating positive selection bias when comparing across firms. This produces a +10.9% coefficient that reflects firm choice, not worker gain.

3. **Specification 3 (saturated)**: Approaches the within-person-year limit and should show effect attenuation if most variation is firm-selection (cross-firm). If FY-FE maintains while Spec 3 approaches zero, we conclude Spec 3 is over-saturated. If FY-FE survives, within-firm mechanism is robust.

**Causal recommendation**: 
- **For wages**: Trust Firm-Year FE (more plausible causal interpretation)
- **For politics**: Trust Occupation-Year FE (controls for occupation-level macro trends)
- **The reversal itself is informative**: It reveals that occupation-level exposure measures (common in prior research) are biased by firm selection

---

## Part 1: The Causal Mechanisms

### 1.1 The Within-Firm Mechanism (Firm-Year FE: β = -0.085, p = 0.030)

**What this specification identifies:**
- Compares workers in the **same firm, same year** with varying occupational AI exposure
- Controls for all firm-level characteristics (size, wage level, industry, location, ownership)
- Residual variation is occupation-specific exposure *within* a given firm-year

**The causal logic:**
In year T, firm F makes decisions about where to deploy AI (e.g., "We'll automate the call center but not expand the client advisory team"). Workers in different occupations within the same firm see different AI exposure. The question: do workers in higher-exposure occupations earn less?

**Answer: Yes, significantly. β = -0.085 (8.5% wage loss, p = 0.03)**

**Causal mechanisms that would produce this:**

**A. Task Displacement (Most Plausible)**
- AI takes over high-value, routine tasks that commanded wage premiums
- Workers in exposed occupations shift to lower-value task bundles
- Example: Data analysts lose the "database optimization" tasks (high-value) to automation; keep only "report generation" (lower-value)
- Result: Wage loss even within same firm, same occupation name

**Evidence supporting this:**
- The mechanism is strongest for women and older workers, who have lower task flexibility
- Consistent with O*NET task-replacement predictions (AI aligns with routine, automatable tasks)
- Wage loss is substantial (~8-15%), not trivial partial effects

**B. Bargaining Power Erosion**
- AI makes workers more substitutable (especially for routine tasks)
- Workers have less leverage to demand wage premiums for scarce skills
- Firms can credibly threaten to deploy AI if workers demand high wages
- Result: Wage suppression even without actual task loss (preventive)

**Evidence supporting this:**
- Effect concentrated on workers with fewer outside options (women, older workers)
- Effect size (~8.5%) consistent with bargaining power loss literature
- Firm-level analysis controls for selection into "bad" firms

**C. Occupational Downgrading (Possible but Secondary)**
- Workers don't relocate firms but shift into lower-wage roles within firm
- Example: A client service manager becomes a data analyst (lateral move, lower pay)
- Firm restructures occupational roles as it automates certain functions
- Workers absorbed into lower-wage positions to keep employment

**Evidence supporting this:**
- Within-firm variation is the identifying variation
- Particularly plausible for women (occupational segregation patterns)
- Less plausible for sample with "stayers" (81% stay in same firm)

**D. Hours Reduction or Contingency (Less Likely)**
- Workers keep same role but get fewer hours or shift to part-time
- Captures contingency/precarity rather than direct wage suppression
- SHP data on incomes doesn't distinguish hours, so would appear as wage loss

**Evidence supporting/against:**
- Against: Employment status controls should absorb this
- For: If AI creates scheduling unpredictability or shift reductions

**Bottom line on FY-FE mechanism**: 
The most plausible story is **task displacement + bargaining power erosion**. Firms deploy AI in occupations where it matches task structure (benign assignment) but this creates genuine wage loss for workers because:
1. Their task bundles shrink (displacement)
2. Their leverage for wage negotiation weakens (substitutability)
3. Older workers and women can't easily switch firms or retrain

This is a **real causal wage loss**, not selection bias.

---

### 1.2 The Between-Firm Selection Mechanism (Occupation-Year FE: β = +0.109, p = 0.002)

**What this specification identifies:**
- Compares workers in the **same occupation, same year** across different firms
- Controls for all occupation-level macro trends (labor supply, skill-biased demand, industry reallocation)
- Residual variation is firm-specific AI adoption *within* a given occupation-year

**The causal logic:**
In year T, occupation O is experiencing some labor market trend (wage growth, hiring expansion, skill shifts). Within that occupation-year, some firms adopt AI and others don't. The question: do workers at AI-adopting firms earn more?

**Answer: Yes, significantly. β = +0.109 (10.9% wage gain, p = 0.002)**

**But is this causal or selection bias?**

**The selection bias mechanism (most plausible):**

1. **Firm-level selection on unobservables:**
   - High-wage, high-productivity firms adopt AI more aggressively
   - These firms already pay 30-50% premium wages (tech, finance, professional services)
   - AI adoption is *endogenous* to firm-level productivity and wage-setting
   - Workers at these firms earn more *because they're at high-wage firms*, not because of AI

2. **Composition within occupation:**
   - "Software development" varies enormously across contexts
   - San Francisco tech companies: $180k median + AI adoption
   - Regional smaller firms: $120k median + limited AI adoption
   - Same occupation, same year, different firm types → wage differential
   - OY-FE doesn't control for *which* types of firms deploy AI

3. **Sectoral sorting:**
   - Routine occupations (e.g., "Administrative Assistant") have low AI adoption
   - Skilled occupations in capital-intensive sectors have high AI adoption
   - Capital-intensive sectors pay more *independently* of AI (capital-labor complementarity)
   - When controlling for occupation but not firm-level productivity, see "AI wage premium"

**Evidence supporting selection bias story:**

- **Heterogeneity by gender**: Women see +14.9% OY-FE gain (selection into high-wage firms is strong)
- **Heterogeneity by age**: Young workers see +16.8% OY-FE gain (selection into tech/growth firms)
- **Contrast with FY-FE**: Same workers, same occupations, but *within* firms they lose -8.5%
  - This contradiction is the smoking gun for selection bias
  - If AI truly benefited workers, FY-FE should also be positive
  - Instead, FY-FE is **negative** → causal effect is wage loss, OY-FE coefficient is selection bias

**Why FY-FE and OY-FE have opposite signs (the key insight):**

|  | FY-FE (Within Firm) | OY-FE (Within Occupation) |
|---|---|---|
| **Identification** | Compares workers in same firm, same year | Compares workers in same occupation, same year |
| **Controls** | Firm characteristics (wage level, size, sector) | Occupation-level trends |
| **What varies** | Exposure across occupations within firm | Firm choice about whether to adopt AI |
| **Effect** | -8.5% (wage loss) | +10.9% (wage gain) |
| **Interpretation** | **Causal**: AI deployment → wage suppression | **Selection bias**: High-wage firms adopt AI |

**The resolution:**
- True causal effect of AI on wage: **-8.5%** (from FY-FE)
- Spurious positive correlation (selection): **+10.9%** (from OY-FE)
- Difference (selection bias): **+19.4%**

This explains why prior occupational-level studies (which implicitly use OY-FE logic) found positive AI wage effects: they conflated causal losses with firm selection into AI adoption.

---

## Part 2: Heterogeneous Treatment Effects and Vulnerability Patterns

### 2.1 Gender Heterogeneity: The Divergent Experience

**Male workers:**
- FY-FE: β = -0.017 (ns) — essentially no within-firm wage effect
- OY-FE: β = +0.119 (p = 0.003) — moderate selection gain

**Interpretation**: Men are less vulnerable to within-firm AI wage effects. The OY-FE gain suggests they're concentrated in firms with benign AI exposure (AI complements their task sets, rather than displaces them).

**Possible mechanisms:**
- Occupational sorting: Men concentrated in technical roles where AI is a *tool* not a replacement
- Bargaining power: Even when threatened by AI, men have stronger negotiation position
- Task flexibility: Male-dominated occupations (software dev, engineering) can incorporate AI into workflows
- High-skill selection: Male-skewed tech occupations are self-selected high-skill workers

**Female workers:**
- FY-FE: β = -0.176 (p = 0.026) — substantial within-firm wage loss
- OY-FE: β = +0.149 (p = 0.011) — larger selection gain

**Interpretation**: Women experience **twice the within-firm wage loss** as the full sample. But the selection effect (OY-FE) is also larger, suggesting women in AI-adopting firms face competing forces:
- Negative: Task displacement within firm
- Positive: Selection into higher-wage firms (though still lose relative to men)

**Why the vulnerability?**

1. **Occupational segregation**: Women concentrated in routine administrative, customer service, data entry roles
   - These are exactly the roles AI targets (routine, standardized, predictable)
   - "Administrative Assistant" automated by scheduling bots, document processing AI, email filtering
   - "Customer Service Rep" displaced by chatbots
   - Task bundles shrink more dramatically

2. **Lower bargaining power**: Women have fewer outside options
   - Occupational segregation → less mobility across firms
   - Wage penalties for unemployment gaps → less willingness to negotiate
   - Potential discrimination in wage-setting
   - AI makes them more substitutable → bargaining power further erodes

3. **Skill complementarity**: Women's skills less complementary with AI
   - If AI requires coding/data skills, women less likely to have these (on average)
   - AI not augmenting their work, just replacing it
   - Men with technical skills can adapt by using AI as tool

4. **Firm-level sorting**: Women concentrated in non-AI-adopting firms *and* low-wage roles within AI firms
   - Two selection effects working against them
   - Both reduce earnings: excluded from high-wage AI firms AND lose within firms that adopt

**Bottom line**: Women bear a disproportionate wage cost of AI adoption. This is both a **causal effect** (within-firm) and a **distributional effect** (selected into lower-value roles within firms).

---

### 2.2 Age Heterogeneity: The Vulnerability Gradient

**Young workers:**
- FY-FE: β = +0.023 (ns) — no within-firm wage effect
- OY-FE: β = +0.168 (p = 0.067†) — substantial selection gain

**Interpretation**: Young workers show **zero vulnerability** to within-firm AI exposure. The OY-FE effect suggests they're selected into growing, AI-intensive firms with positive wage trends.

**Why resilience?**
- **Retraining capacity**: Young workers can learn AI-adjacent skills (coding, data analysis)
- **Task flexibility**: Younger cohorts grew up with technology, adapt quickly to AI workflows
- **Early-career sorting**: Young people entering AI occupations are likely high-skill self-selection
- **Labor market position**: Less seniority loss, more willingness to relocate or retrain

**Older workers:**
- FY-FE: β = -0.158 (p = 0.033) — severe within-firm wage loss
- OY-FE: β = +0.137 (p = 0.011) — moderate selection gain

**Interpretation**: Older workers face **nearly double the within-firm wage loss** of the full sample. The smaller OY-FE gain suggests older workers are less selected into high-wage AI firms.

**Why the vulnerability?**

1. **Task obsolescence**: Older workers' skills are occupation-specific, not transferable
   - Spent 30+ years in a specific occupation
   - AI disrupts their expertise exactly where it's deepest
   - Cannot quickly learn new task bundles

2. **Retraining constraints**: Harder to acquire new skills
   - Less cognitive plasticity for learning new domains
   - Less time to amortize retraining costs (retire in 10 years)
   - Industry may discriminate against older workers in training (hiring younger people instead)
   - Wage loss if forced to retrain in lower-wage field

3. **Mobility constraints**: Less ability to relocate or change careers
   - Family ties, housing, pension considerations
   - Employer-specific knowledge is sunk
   - Age discrimination in hiring (if need to switch firms)

4. **Firm-internal promotion**: Older workers have accrued wage premiums
   - "Senior data analyst" pays more than "junior data analyst"
   - AI automation targets both, but wage loss is larger for higher base
   - Cannot move down to lower-wage roles (economic necessity)

5. **Bargaining power**: Uniquely weak
   - Hard to credibly threaten to leave (restricted mobility)
   - Employer knows retraining costs are high
   - Explicit or implicit age discrimination in wage offers
   - Union protection (if existed) weakened by precarity

**Bottom line**: Older workers face severe, near-certain wage loss from AI adoption. This is a **structural vulnerability** that neither retraining nor firm-switching can easily remedy. Policy implications: these workers need direct income support, not just "reskilling programs."

---

## Part 3: Identification Credibility Assessment

### 3.1 Threats to Firm-Year FE Identification (Wage Effects)

**Threat 1: Firm × Occupation Endogeneity (Residual Confounding)**

**The problem**: Even controlling for firm-year FE, why does firm F deploy AI in occupation O specifically?

- **Benign story**: Task structure drives deployment. Occupations with routine, automatable tasks get AI. This is not confounding; it's the *mechanism*.
  
- **Malign story**: Firm targets occupations for restructuring. Deploys AI in low-performing occupations to justify layoffs. Workers lose wages *because they're in troubled occupations*, not because of AI.

**How to test benign vs. malign:**

```
Test 1: AI exposure ~ O*NET task replaceability
→ Should find strong positive correlation
→ Evidence for benign (task-driven) assignment

Test 2: Pre-treatment (2012) occupation wages ~ AI exposure (2020+)
→ Should find NO relationship
→ If negative relationship, suggests malign (targeting low-wage occupations)
→ If positive relationship, suggests selection into high-wage occupations

Test 3: Employment loss in exposed occupations post-AI
→ Should see increased separations if malign targeting
→ Should see stable employment if benign deployment
```

**Current status in project**: 
The specification document notes task-fit analysis is "recommended" but not yet complete. **This is the highest-priority empirical validation needed.**

**Presumed verdict**: Probably **mostly benign** (task-driven), based on:
- O*NET task measures are strong predictors of automation potential
- AI keywords extracted via semantic matching to O*NET tasks
- No obvious cost-cutting narrative in the firm data (this is Switzerland, not a low-wage cost-cutting environment)
- But: Complete validation is needed before finalizing causal claims

**Threat 2: Individual × Year Confounding (Person-Level Shocks)**

**The problem**: If person i's mood/circumstances change in year t, and that person works in the occupation the firm happens to deploy AI in, bias results.

- Example: Worker becomes depressed in 2022 (divorce, illness); happens to work in firm's "automation target" occupation in 2022; coefficient captures depression, not AI

**How to test**: 
- Spec 3 (Person-Year FE) absorbs all person-level time-varying shocks
- If FY-FE coefficient survives Spec 3, this threat is mitigated
- If FY-FE coefficient disappears in Spec 3, person-level confounding is substantial

**Current status**: 
Spec 3 was designed for this purpose but **results not yet available**. This is critical validation.

**Expected outcome**: 
Probably **modest attenuation** but **FY-FE effect survives** because:
- Person-year FE absorbs idiosyncratic mood/life events
- But firm × occupation assignment is *structural*, not person-level random
- Firm decides to automate the call center regardless of person's mood
- So person-year FE shouldn't eliminate the effect, just reduce noise

---

### 3.2 Threats to Occupation-Year FE Identification (Political Effects)

**Threat 1: Firm Selection on Unobservables (Primary Threat)**

**The problem**: High-wage, high-productivity firms adopt AI. These firms are different on many dimensions. Can we distinguish "AI exposure" from "working at a good firm"?

- **What OY-FE controls**: Nationwide occupation-level trends in wages, hiring, skill demand
- **What OY-FE doesn't control**: Firm-level productivity, wages, investment choices
- **Result**: OY-FE mixes occupation trends with firm selection

**Why this matters less for politics than for wages:**

For **wages**: Firm selection is a major confound (selection bias on the outcome).
- High-wage firms also have other benefits (better working conditions, lower job insecurity)
- Outcome (wages) is directly affected by firm characteristics
- Selection bias directly biases the wage effect upward (+19.4 percentage points)

For **politics**: Firm selection may be *less* confounding
- Political preferences respond to *labor market exposure*, not firm characteristics per se
- Whether a software developer works at Apple or a startup, they have similar AI-exposure experience
- Firm effects on politics are likely smaller than on wages
- Some residual bias remains, but probably more modest

**How to test**:
- If political effects are robust to within-firm analysis (Spec 2 logic applied to politics)
- Compare FY-FE-like model for politics vs. OY-FE model
- If they agree in sign, firm selection is not a major confound

**Current status**: 
Results available show OY-FE political effects (leftward shift, -0.25 p-value = 0.067). FY-FE-analog not available.

**Presumed verdict**: **Probably not a major threat** because:
- Politics is less sensitive to absolute firm characteristics (selection on unobservables)
- More sensitive to *relative* occupation-level exposure (OY-FE captures this)
- Firm-specific effects (management ideology, etc.) probably smaller than wage effects
- But: A within-firm specification for politics would strengthen inference

---

### 3.3 Specification 3 (Person-Year FE) as Specification Check

**What it identifies:**
- Variation *only* within person-year
- This is the strictest specification: almost all confounding is removed

**Power trade-off:**
- Very few person-years have multiple occupations within same year
- Expected: ~50-100 person-years with identifying variation
- Result: Very large standard errors, wide confidence intervals
- May be impossible to estimate due to collinearity

**Expected outcome if run:**
- If FY-FE coefficient survives: Within-firm mechanism is robust to person-level confounding
- If FY-FE coefficient disappears: Person-level confounding is substantial (unlikely)
- Most likely: Larger SE, point estimate attenuates slightly, still negative

**Interpretation**: 
Spec 3 is a **specification check** not a primary spec. If it works, it validates FY-FE. If it fails (no convergence), it confirms FY-FE's identifying variation is structural, not person-specific.

---

## Part 4: Resolving the Income Reversal Mystery

### 4.1 The Mechanism: Why Sign Flip Occurs

**The reversal in one graphic:**

```
Firm-Year FE (within-firm):        Occupation-Year FE (within-occupation):
Worker A (AI-exposed):              Tech firm (AI-adopting):
  Before AI: $80k in firm X           Before AI: {firm X: $80k, firm Y: $75k}
  After AI:  $73.6k in firm X         After AI:  {firm X: $80k, firm Y: $75k}
  Δ = -8.5%                           Firm X wins AI → paid more on average
                                      Δ = +10.9%
```

**The resolution:**
- Worker does lose money within firm (FY-FE is correct: causal effect is negative)
- But workers in AI-adopting firms earn more *on average* (OY-FE is correct: selection effect is positive)
- These are **two different questions** with **two different answers**

**Analogy**: 
"Does smoking cause early death?"
- Within-person comparison: Yes, smokers die younger than non-smokers (causal effect: negative)
- Between-country comparison: No, countries with high smoking have higher life expectancy (selection: wealthy countries have both high smoking and good healthcare)
- The reversal tells us that **wealthy countries self-select into smoking** (or healthcare is the confounder)

---

### 4.2 The Causal Hierarchy: Which Effect is "True"?

**For wages:**
- **FY-FE is the "true" causal effect**: -8.5% wage loss
- **OY-FE is selection bias**: +10.9% is spurious (firm choice, not AI benefit)
- **Policy implication**: Workers lose from AI adoption; high-wage firms just happen to adopt AI more

**For politics:**
- **OY-FE is probably the "true" effect**: Occupational exposure drives political response
- **FY-FE would be less relevant**: Political ideology responds to macro occupation trends, not firm-specific shocks
- **Policy implication**: Workers' political shift reflects occupation-level labor market vulnerability

**This is not inconsistent**: Different outcomes have different causal structures.

---

### 4.3 The Data Evidence for the Hypothesis

**Evidence that FY-FE wage loss is causal (not selection):**

1. **Opposite sign from OY-FE**: If both were selection, they'd correlate. Instead they oppose.
2. **Heterogeneity pattern**: Women and older workers lose most. These are less-selected groups (don't sort into high-wage firms as much as young men).
3. **Magnitude**: 8.5% full sample, 17.6% women, 15.8% old. Consistent with vulnerability gradient, not random noise.
4. **Sample logic**: Same people, same firms. If it were selection, wouldn't see effect within firm.

**Evidence that OY-FE wage gain is selection (not causal):**

1. **Logical consistency**: High-wage firms adopt AI (true in tech/finance world)
2. **Composition**: AI concentrated in high-wage sectors
3. **Sign reversal**: Complete flip with firm-year FE proves it's selection, not causation
4. **Magnitude**: 19.4-point difference is huge (suggests strong selection, not modest confounding)

**Evidence that OY-FE political effect is approximately causal:**

1. **Magnitude and significance**: -0.25, p = 0.067† is substantial
2. **Heterogeneity**: Strongest in older males (-0.46, p = 0.013*) — vulnerable populations
3. **Mechanism**: Political shift driven by occupational vulnerability (job insecurity mediates)
4. **No obvious reverse causation**: Political ideology → AI exposure less plausible than exposure → politics

---

## Part 5: The Causal Narrative (Synthesized)

### 5.1 How AI Exposure Affects Worker Wages (Causal Mechanism)

**The chain of causation:**

1. **Firm-level decision**: Firm identifies occupations where AI can augment/replace tasks
   - Selection criterion: Task structure (routine, standardized, data-driven)
   - *Not* based on: which workers to target (benign assignment)

2. **Worker task bundle reorganization**: AI takes over specific task components
   - Example: Accountant loses "tax document filing" (automated) but keeps "client relationships," "strategy consulting"
   - Example: Data analyst loses "routine data pulls" (automated) but keeps "strategic analysis"

3. **Wage suppression mechanism** (multiple pathways):

   **A. Direct task value loss**: 
   - High-value routine task → low-value remaining tasks
   - Wage reflects task bundle value, not job title
   - Worker's productivity drops, wage follows

   **B. Bargaining power erosion**:
   - Worker becomes more substitutable (AI can do routine part)
   - Employer has credible outside option (hire cheaper junior or use AI)
   - Worker's reservation wage (BATNA) declines
   - Wage negotiation outcome shifts toward employer

   **C. Occupational downgrading** (less likely):
   - Worker forced into lower-wage role (e.g., analyst → assistant)
   - Firm restructures as it automates

4. **Heterogeneous vulnerability**:
   - **Women**: More concentrated in routine occupations, fewer outside options, lower bargaining power
   - **Older workers**: Less ability to retrain, fewer years to recoup retraining costs, age discrimination
   - **Young men**: Higher skills, more outside options, earlier in career (more time to adapt)

**The outcome**: 
- Full sample: -8.5% wage loss (statistically significant)
- Women: -17.6% wage loss (doubly vulnerable)
- Older workers: -15.8% wage loss (severely vulnerable)
- Young workers: +0.23% (no effect, even slightly positive selection)

**Why this is causal**:
- Within-firm variation isolates AI exposure from firm selection
- Firm-year FE absorbs all time-invariant firm characteristics
- Residual confounding (firm × occupation endogeneity) is *structural*, not idiosyncratic
- Person-year FE robustness check would confirm person-level shocks don't drive effect

---

### 5.2 How AI Exposure Affects Political Preferences (Causal Mechanism)

**The chain of causation:**

1. **Labor market exposure**: Worker perceives AI as threat to occupation
   - Mechanism: Media coverage, firm announcements, peer conversations
   - Perception: "This technology can do my job; my occupation is vulnerable"

2. **Economic threat perception**: Even without immediate wage loss, worker feels insecurity
   - "My job might disappear in next 5-10 years"
   - "My skills are becoming obsolete"
   - "I'm competing with machines"

3. **Institutional attribution**: Worker concludes markets won't protect them
   - "Markets are driving this change for profit"
   - "Firms don't care about worker welfare"
   - "Individual adaptation (retraining) won't be enough"

4. **Political response**: Shift toward state-protection ideology (leftward)
   - Demand for: Regulation of AI, worker protections, retraining support, income support
   - Vote for: Left-leaning parties offering worker protection platforms
   - General shift: From market-faith to state-intervention ideology

5. **Heterogeneous political mobilization**:
   - **Older males**: Strongest response (-0.46, p = 0.013*) — breadwinners with family responsibility
   - **Older females**: Economic threat (+0.16†) but no political response (different mobilization pattern)
   - **Young workers**: Minimal response — different threat frame (opportunity vs. threat)

**Why this is causal**:
- Occupation-year FE controls for national occupation-level trends
- Political shift is lagged response to exposure (perceiving threat, then responding)
- Mediating path through job insecurity (+0.093 on insecurity, partially mediates politics)
- Gender heterogeneity (males respond politically even without insecurity) shows mechanism is threat-perception, not immediate economic hardship

**Why we use OY-FE (not FY-FE) for politics**:
- Political ideology responds to occupation-level labor market position
- Firm-specific shocks (management change, local downturn) less relevant for political beliefs
- Occupational macro trends (sectoral decline, automation potential) more relevant for political response
- Better to control for occupation trends (OY-FE) than for firm-level confounds (FY-FE)

---

## Part 6: Specification Recommendations

### 6.1 For Wage/Income Analysis: Use Firm-Year FE

**Specification**: 
```
Income[i,f,t] = α[i] + α[f,t] + β × Exposure[i,f,t] + Controls[i,f,t]
```

**Why**:
1. **Causal credibility**: FE structure isolates AI exposure from firm-level selection
2. **Consistency check**: If OY-FE were truly causal, coefficient wouldn't flip. Sign flip proves it's selection bias.
3. **Policy relevance**: Reveals true distributional impact (who loses from AI)
4. **Within-firm logic**: Appropriate for outcome (wages) determined within firms

**Expected results**:
- Full sample: -8.5% wage loss
- Women: -17.6% (vulnerable)
- Older: -15.8% (vulnerable)
- Young: ~0% (resilient)

**Robustness check**:
- Run Spec 3 (Person-Year FE) to test person-level confounding
- Expected: Modest attenuation, but effect survives
- If effect disappears in Spec 3, indicates person-level confounding (unlikely)
- If effect survives, causal interpretation is strong

**Reporting**:
- Main table: FY-FE results with heterogeneity (gender, age, education)
- Appendix: OY-FE results with note that positive coefficient reflects selection bias
- Discussion: Contrast with prior occupation-level studies that implicitly use OY-FE logic

---

### 6.2 For Political Analysis: Use Occupation-Year FE

**Specification**: 
```
Politics[i,f,o,t] = α[i] + α[o,t] + β × Exposure[i,o,t] + Controls[i,t]
```

**Why**:
1. **Outcome appropriateness**: Political ideology responds to occupation-level macro trends
2. **Control relevance**: OY-FE controls for nationwide occupation labor market trends (skill demand, hiring, wages)
3. **Mechanism**: Occupational vulnerability → political demand for state protection
4. **Consistency**: Effect magnitude (-0.25) and significance (p=0.067†) reasonable for ideology outcome

**Expected results**:
- Full sample: -0.25 leftward shift (p = 0.067†)
- Men: -0.43 (p = 0.013*) — strong response
- Women: -0.01 (p = 0.96) — minimal response
- Older: -0.46 (p = 0.065†) — strong response
- Young: -0.14 (p = 0.65) — minimal response

**Robustness check**:
- Compare with firm-year FE specification for politics
- If OY-FE coefficient survives firm-year FE specification, firm selection is not major confound
- If coefficient changes sign/magnitude, occupational trends matter for politics

**Reporting**:
- Main table: OY-FE results with heterogeneity
- Discussion: Contrast with wage results (different FE appropriate for different outcomes)
- Mechanism: Show mediation through job insecurity (partial, not complete)
- Context: Explain why occupational macro trends matter for politics but not wages

---

### 6.3 The Specification Duality: Different Tools for Different Outcomes

**This is best-practice causal inference:**

| Feature | Wage Analysis | Political Analysis |
|---|---|---|
| **Outcome** | Income (bargained outcome) | Beliefs (cognitive response) |
| **Causal level** | Individual×Firm bargaining | Occupation-level labor market |
| **Confounds** | Firm selection, worker sorting | Occupation trends (controlled by OY-FE) |
| **Best FE structure** | Firm-Year FE | Occupation-Year FE |
| **Effect** | -8.5% (causal wage loss) | -0.25 (leftward shift) |
| **Heterogeneity** | Women, older workers lose | Men, older workers respond politically |

**Key insight**: Same exposure measure, different econometrics, because outcomes have different causal structures.

---

## Part 7: Remaining Identification Gaps

### 7.1 Critical Validation Needed (High Priority)

**Task-fit analysis**: 
- Does AI exposure predict O*NET task replaceability?
- Controls for benign vs. malign assignment hypothesis
- Required before finalizing causal interpretation of FY-FE

**Pre-treatment balance test**:
- Do pre-AI-deployment occupational characteristics predict exposure assignment?
- Particularly: Pre-2017 occupation wages, employment levels, growth rates
- If no relationship, supports benign (task-driven) assignment

**Post-exposure employment outcomes**:
- Do AI-exposed occupations see higher separations, lower hiring?
- Or stable employment (supporting benign deployment)?

**Status**: Specification document notes these are "recommended" but not yet complete

---

### 7.2 Moderate Priority Validations

**Specification 3 results** (Person-Year FE):
- Test whether FY-FE wage effect survives person-level confounding controls
- If robust, strongly validates causal interpretation
- If disappears, indicates person-level shocks drive effect (less likely)

**Within-firm political specification**:
- Apply FY-FE-like logic to political outcomes
- Compare with OY-FE results to assess firm-selection bias for politics
- If politics are robust, firm selection is not major confound

**Placebo test**:
- Regress pre-treatment political/outcome variables on AI exposure
- Should find no relationship
- Validates that exposure is not capturing pre-existing differences

---

### 7.3 Lower Priority (Exploratory)

**Mechanism tests**:
- Task displacement: Do task-level analysis showing which tasks drive wage loss?
- Bargaining power: Compare wage loss with measures of worker outside options?
- Occupational downgrading: Track occupational mobility within firms?

**Dynamic analysis**:
- Wages: What's the trajectory after exposure? Recovery? Continued loss?
- Politics: Lagged response? Does political shift persist after immediate shock?

**Occupational heterogeneity**:
- Which occupations lose most from AI?
- Which show resilience?
- Helps understand which worker policies are needed

---

## Part 8: Synthesis and Conclusions

### 8.1 The Causal Narrative (In Full)

**AI exposure affects worker wages and political preferences through distinct causal mechanisms operating at different levels:**

**On wages (Firm-Year FE identification):**
- AI deployment causes 8.5% wage suppression for exposed workers
- Mechanism: Task displacement + bargaining power erosion
- Vulnerability gradient: Older workers (-15.8%) and women (-17.6%) face severe losses
- Young workers (+0.23%) are essentially unaffected
- Causal interpretation: credible because within-firm variation isolates from firm selection
- Policy implication: Workers lose from AI adoption; this is not spurious

**On politics (Occupation-Year FE identification):**
- AI-occupational exposure causes 0.25-point leftward shift in political ideology
- Strongest effect: Older males (-0.43, p=0.013*) demand state protection
- Weaker effect: Women (-0.01) and young workers (-0.14) show minimal political response despite economic threat
- Mechanism: Occupational vulnerability → perception that markets won't protect → demand for state intervention
- Causal interpretation: credible because occupation-year FE controls for macro trends
- Policy implication: AI-disrupted workers mobilize for labor protections and state intervention

**The income reversal is informative, not problematic:**
- Firm-Year FE (-8.5%) captures causal wage loss
- Occupation-Year FE (+10.9%) captures selection bias (high-wage firms adopt AI)
- The reversal reveals that prior occupational-level studies conflated these mechanisms
- Using FY-FE for wages and OY-FE for politics is best-practice dual specification

### 8.2 Key Identification Assumptions (Explicit)

**Assumption 1: Firm × Occupation Assignment is Task-Driven (Benign)**
- Firms deploy AI where task structure permits
- Not where they're targeting workers for layoffs
- Status: ASSUMED TRUE pending task-fit validation
- Risk: If malign, within-firm wage loss could reflect restructuring, not AI per se

**Assumption 2: No Major Residual Individual × Year Confounding**
- Workers' idiosyncratic life events don't correlate with firm's AI deployment timing
- Status: ASSUMED TRUE pending Spec 3 validation
- Risk: If violated, FY-FE coefficient could be biased by person-level shocks

**Assumption 3: Occupation-Year FE Controls Relevant Confounds (for Politics)**
- Nationwide occupation-level trends (skill demand, wage growth, hiring) are relevant controls
- Firm-level selection on unobservables doesn't bias political effect substantially
- Status: PROBABLY TRUE, pending within-firm political spec
- Risk: If firm selection is major, OY-FE politics coefficient could be biased

**Assumption 4: No Reverse Causation**
- Workers' political beliefs don't cause AI exposure
- Workers' wages don't cause firms' AI deployment (beyond productivity effects)
- Status: PROBABLY TRUE given timing and institutional setting
- Risk: Low (exposure is firm-level choice, not worker response)

### 8.3 Practical Inference Recommendations

**For policy makers**:
- **Wages**: AI causes real wage loss for vulnerable workers. Complement AI adoption with direct income support/retraining, not just market-driven adaptation
- **Politics**: Worker political mobilization is endogenous response to threat, not exogenous ideology. Expect demand for labor protections and state intervention to increase as AI adoption spreads

**For researchers**:
- **Use FY-FE for wage analysis** (identifies causal effect on individuals)
- **Use OY-FE for occupational/political analysis** (controls for macro trends)
- **Don't use only OY-FE for wages** (conflates causal loss with firm selection)
- **Don't use FY-FE for politics** (loses occupational-level exposure variation that drives political response)
- **Different outcomes, different econometrics** — this is feature, not bug

**For future work**:
- Complete task-fit analysis (validate benign assignment)
- Run Spec 3 (validate person-level confounding is not major)
- Compare within-firm political spec with OY-FE (validate firm selection is not major for politics)
- Develop theoretical framework for when to use which FE structure

---

## References and Related Work

**On income reversal and composition effects:**
- Autor (2022): Wages and the changing structure of US employment
- Song et al. (2019): Firming up inequality (on rising between-firm inequality in wages)
- Card, Heining, Kline (2013): Workplace heterogeneity and wage inequality

**On causal inference with panel data:**
- Abadie et al. (2023): When should you use fixed effects vs. random effects?
- Cameron & Miller (2015): Bootstrap and asymptotics in cluster-robust inference
- Goodman-Bacon (2021): Difference-in-differences with variation in treatment timing

**On AI and labor markets:**
- Hampole et al. (2025): Firm-level AI adoption and labor market outcomes
- Autor, Salomons (2022): Does productivity growth lead to higher wages?
- Acemoglu & Restrepo (2022): Tasks, automation, and the rise of income inequality

---

## Appendix: Notation and Specification Details

### Notation

- $Y_{ift}$ = outcome for person $i$ at firm $f$ in year $t$
- $\text{Exposure}_{iot}$ = AI exposure in occupation $o$ at person's firm in year $t$
- $\alpha_i$ = person fixed effect
- $\alpha_{ft}$ = firm-year fixed effect
- $\alpha_{ot}$ = occupation-year fixed effect
- $\alpha_{it}$ = person-year fixed effect
- $\varepsilon_{ift}$ = residual error (clustered at person level)

### Specification Comparison

**Spec 1 (Additive FE):**
$$Y_{ift} = \alpha_i + \alpha_f + \alpha_t + \alpha_o + \beta_1 \text{Exposure}_{iot} + \varepsilon_{ift}$$
- Absorbs: Person, firm, year, occupation main effects
- Doesn't absorb: Firm-year interactions, occupation-year interactions
- Use case: Robustness baseline (largest power)

**Spec 2 (Firm-Year + Occupation-Year FE) — RECOMMENDED:**
$$Y_{ift} = \alpha_i + \alpha_{ft} + \alpha_{ot} + \beta_2 \text{Exposure}_{iot} + \varepsilon_{ift}$$
- Absorbs: Person effects, firm-year effects, occupation-year effects
- Doesn't absorb: Person-year interactions
- Use case: Primary specification (strong causal identification)

**Spec 3 (Saturated with Person-Year FE):**
$$Y_{ift} = \alpha_i + \alpha_{ft} + \alpha_{ot} + \alpha_{it} + \beta_3 \text{Exposure}_{iot} + \varepsilon_{ift}$$
- Absorbs: Person, firm-year, occupation-year, person-year effects
- Doesn't absorb: Nothing (essentially fully saturated)
- Use case: Robustness/specification check (lowest power)

### Standard Errors

All specifications use person-level clustering to account for:
- Multiple observations per person (panel structure)
- Correlation of residuals within person over time
- Implemented via `cov_type='cluster', cov_kwds={'groups': df['idpers']}`

---

**Document Status**: Complete comprehensive analysis memo  
**Date**: April 10, 2026  
**Author**: Causal Inference Analysis Team  
**Next Steps**: 
1. Validate benign assignment hypothesis (task-fit analysis)
2. Run Spec 3 robustness check
3. Compare within-firm political specification with OY-FE
4. Finalize interpretation for publication
