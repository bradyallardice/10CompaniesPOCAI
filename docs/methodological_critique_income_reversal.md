# Methodological Critique: Income Reversal and Political Economy Implications

## Executive Summary

The income reversal analysis elegantly documents a compositional paradox: within firms, AI exposure predicts wage loss (β = -0.085); across firms, it predicts wage gain (β = +0.109). The mechanisms are correctly identified—selection of high-wage firms into AI adoption confounds occupation-level estimates—but substantial gaps remain before this evidence can support a political economy argument about how worker wage losses translate into political demands.

The problem is not econometric validity. The problem is **interpretation**: the Firm-Year FE coefficient identifies something real (wage suppression for some workers), but *what* and *who* are afflicted remains opaque. Without disentangling the mechanisms, causal channels, and affected populations, these wage effects hang disconnected from political behavior.

---

## Core Methodological Gaps

### Gap 1: Missing Mechanism Identification

**The Problem:**
The FY-FE wage loss of 8.5% is identified, but the underlying mechanism remains unobserved. Multiple incompatible mechanisms could produce the same coefficient:

1. **Task displacement**: AI automates the worker's primary tasks → worker reassigned to lower-value work (same job title, reduced task set)
2. **Occupational downgrading**: Worker moves from "Software Engineer" to "Technical Support" within same firm
3. **Bargaining erosion**: Worker retains tasks but negotiates lower wages due to AI-driven substitutability
4. **Hours reduction**: Worker keeps position but works fewer hours for same hourly wage
5. **Selection on unobservables**: Lower-ability workers self-select into AI-exposed roles (wage loss reflects prior ability, not AI)

**Why it matters for politics:**
These mechanisms have starkly different political implications:
- **Task displacement** → demand for education/retraining + blame for technology adoption
- **Downgrading** → demand for occupational protections + resentment of firm stratification
- **Bargaining loss** → demand for union strengthening + class-based grievance
- **Hours loss** → demand for labor standards + support for minimum hours guarantees
- **Selection** → no genuine grievance (low-ability workers always earned less)

**Current approach:** Firm-Year FE, with heterogeneity by gender/age. This is insufficient. You observe that women and older workers suffer larger losses (17.6% and 15.8% respectively), but you don't know why.

**Required test:** 
- Track individual job titles pre/post AI adoption within firm-year stayers
  - If titles unchanged, task displacement likely
  - If titles changed, downgrading likely
- Decompose exposure effect into:
  - Within-title wage change (bargaining/hours)
  - Between-title composition shifts (downgrading)
  - Retention/exit probability (worker exit from firm)

### Gap 2: Selection on Unobservables Remains Unaddressed

**The Problem:**
The Firm-Year FE specification controls for time-invariant firm characteristics and occupational composition, but cannot rule out time-varying worker selection into AI-exposed roles.

**Specific concern:**
Suppose firms adopt AI in roles they identify as having lower-productivity workers (e.g., "we're automating routine data entry where we have less engaged staff"). Then:
- Workers are assigned to AI exposure *because* of low unobserved productivity
- Wage loss reflects worker quality, not AI treatment effect
- Same FY-FE coefficient emerges

**Evidence suggesting this is plausible:**
- You find larger effects for older/female workers (17.6% and 15.8%)
- These groups have *lower* average reservation wages, consistent with lower bargaining power or lower outside options
- No evidence that firms preferentially assign younger/male workers away from AI

**Current approach:** Fixed effects, which assume random assignment conditional on controls. You control for age, gender, employment status—but not unobserved productivity, learning ability, or task-specific skill.

**Required test:**
- **Examine pre-treatment wage trends:** Do workers later assigned to AI exposure have systematically lower wage growth *before* adoption? 
  - If yes, selection on unobservables is driving results
  - If no, more confidence in causal interpretation
- **Placebo test:** Randomly assign "fake" AI exposure dates 1-2 years before actual adoption. Do wage effects appear in the false pre-treatment period?
- **Mechanism test:** Is worker exit from firm elevated after AI exposure assignment?
  - If workers vote with their feet, selection is likely (they knew they were low-value)
  - If exit rates unchanged, causal harm more plausible

### Gap 3: Measurement of AI Exposure is Coarse and Lagged

**The Problem:**
AI exposure is measured via job advertisements linked to firms at the occupation×year level. This is at least one layer of indirection from actual AI deployment:

1. **Job ads lag implementation:** Firms post AI-required positions *after* internal AI adoption. A 1-2 year lag is plausible.
2. **Ads measure demand, not displacement:** A firm posting 5 "AI Engineer" positions doesn't tell you whether 10 existing software engineers had tasks automated.
3. **Selection into advertising:** High-status firms advertise AI heavily to signal innovation; low-status firms adopt quietly. Measurement error is systematic, not random.
4. **Skill requirement inflation:** A firm might re-title an existing role as "AI-enabled" without changing actual tasks, just signaling to external labor market.

**Evidence of coarseness:**
- You find only 6,139 person-years with exposure > 0 (out of 45,325)
- Median exposure is 0.0000; 75th percentile is 0.0000
- Exposure is occupation×firm×year level, not individual job level

This means you're comparing:
- Workers in occupations at firms that posted AI ads (vs. didn't)
- Not: workers whose actual tasks changed (vs. didn't)

**Required test:**
- **Individual-level exposure validation:** If you have access to job postings data, link to individual respondents by job title and firm, not just aggregate. Exposure should vary within occupation.
- **Timing test:** Stagger exposure definitions (t, t+1, t+2) and test whether effects persist or decay. If true task displacement, lag structure should be informative.
- **Firm size adjustment:** Larger firms post more ads. Control for firm posting volume to separate "firm adopts AI" from "firm is large/visible."

### Gap 4: Heterogeneous Effects are Incomplete

**What you have:**
- Gender split (women: 17.6% loss; men: insignificant)
- Age split (older: 15.8% loss; younger: insignificant)

**What you're missing:**
1. **Education/skill level:** Does the wage loss differ for university-educated vs. vocational workers? This cuts to whether AI is displacing low-skill routine tasks (where low-education workers dominate) vs. high-skill tasks.
   - High-education workers may retrain into new AI-adjacent roles
   - Low-education workers may lack retraining capacity
   
2. **Sector effects:** Manufacturing AI ≠ Finance AI ≠ Retail AI. Different sectors have:
   - Different AI types (automation vs. decision support)
   - Different wage-setting institutions (manufacturing unions vs. finance individual bargaining)
   - Different worker mobility constraints
   
3. **Firm size:** SMEs vs. large firms respond differently to AI adoption:
   - SMEs may lack resources to reskill workers; large firms have internal mobility
   - Small firm wage loss could be bankruptcy/contraction; large firm loss is bargaining
   
4. **Occupational exposure variation:** Within occupations, high-wage roles may adopt AI defensively (protect skilled work) while low-wage roles adopt offensive (replace workers).

**Required test:**
- Estimate fully heterogeneous model:
  - β(AI exposure | education, sector, firm size, occupation quintile)
  - Report results in heatmap: which education × sector cells suffer losses?
- This directly informs political targeting and policy demand predictions.

### Gap 5: Sample Selection and Representativeness

**The Issue:**
Your sample includes 37,279 person-years with valid income and exposure data. But the SHP population is selective:

1. **Firm ID missingness is structural (~75%):** 
   - Only 3,130-5,735 person-years per year have a firm_id
   - This is ~8-13% of SHP respondents in any year
   - Root cause: linkage file only covers a subset (likely those who consented + were matchable)
   
2. **Who is missing?**
   - Self-employed (no firm_id assigned; excluded from sample)
   - Very small firm employees (likely excluded for anonymization)
   - Contract/temp workers (may not link to single firm)
   - Recent immigrants/movers (linkage may not work)
   
3. **Are AI-exposed and non-exposed workers equally likely to be in the sample?**
   - High-wage tech firms likely *over*-represented (more data linkage support)
   - Low-wage service firms likely *under*-represented (harder to link)
   - **If true:** Your FY-FE wage loss of 8.5% may underestimate true loss in low-wage sectors

**Evidence of bias:**
- You show 53.6% male, 46.4% female in your full SHP sample
- Swiss labor force is ~48% female
- You're slightly female-depleted (possibly due to firm linkage bias toward structured employment)

**Required test:**
- **Representativeness check:** Compare your SHP sample characteristics (age, income, education, sector) to:
  - Full SHP (unlinked)
  - Swiss Census LFS (administrative labor force data)
  - If differences substantial, use inverse-probability weighting to adjust for selection
- **Firm linkage bias check:** Does exposure vary systematically by firm size in your sample?
  - Overrepresentation of large firms → your wage loss is for large-firm workers, not representative
  - This matters for political response (large firms have different politics)

---

## Political Economy Interpretation Gaps

### The Causal Mechanism Question

You identify a wage effect, but political response depends on *perceived* causality and *attributable* blame, not econometric identification.

**The gap:**
- Econometrics identifies: exposure → wage loss
- Politics requires: worker *understands* exposure caused loss AND *attributes* loss to someone culpable

**Why it matters:**
If women see 17.6% wage loss but attribute it to "I was hired at lower wage, not that AI hurt me," there's no political demand for AI regulation. If they see it as "AI forced my downgrade," demand for regulation follows.

Your heterogeneity evidence (women/older workers hit harder) suggests a causal story, but the evidence is circonstantial. You need *qualitative validation* or *behavioral evidence* (union organizing, protest, policy demand) to confirm that workers perceive the mechanism you've identified.

### Political Outcome Measurement

**The gap:** You don't measure political outcomes yet.

Your plan is to link wage effects to political outcomes (vote, ideology, policy demand). But the linking mechanism is unclear:

1. **Time lag:** Worker experiences wage loss in year T. When do they change political preferences? T+1? T+2? Or only after cumulative losses?
2. **Attribution:** Wage loss in year T. Is it attributed to "AI" (rate 8% among Swiss public) or "globalization" (rate 67%) or "firm greed" (rate 90%)?
3. **Remedy selectivity:** Workers demand different policies based on how they perceive causality:
   - AI displacement → demand AI regulation, retraining, education
   - Firm wage suppression → demand unions, profit-sharing, redistribution
   - Globalization → demand protectionism
   
You're measuring one input (wage loss) but not the political mechanism (attribution) or outcomes (political demand).

**Required test:**
- **Survey experiment:** Show SHP respondents vignettes of wage loss with different attributed causes:
  - "Your firm introduced AI to automate your tasks"
  - "Your firm became more competitive due to global tech competition"
  - "Your firm cut wages to improve profitability"
  
  Elicit policy demand for each. Does the wage loss coefficient change?

- **Timing analysis:** Estimate political outcome as function of wage loss *with lags*:
  - Does political response appear at T+0, T+1, T+2?
  - Does it depend on cumulative loss or contemporaneous loss?

---

## Five Critical Gaps Summary

| Gap | Phenomenon | Evidence | Fix |
|---|---|---|---|
| **Mechanism** | Multiple mechanisms (displacement, downgrading, bargaining, hours, selection) could produce same FY-FE coeff | Women/older workers lose more, but mechanism unknown | Track job titles; decompose wage loss into within-title and between-title components |
| **Selection** | Firms may assign low-unobserved-ability workers to AI exposure | No pre-treatment wage trend comparison; no placebo test | Examine pre-adoption wage trends; test false exposure dates; measure worker exit |
| **Exposure measurement** | AI measured via job ads, not actual task changes; includes 1-2 year lag | Only 6,139/45,325 person-years have exposure > 0; median exposure = 0 | Link job ads to individual roles within occupation; vary lag structure; control firm posting volume |
| **Heterogeneity** | Effects split by gender/age, but not education/skill/sector/firm size | No exploration of which education × sector combinations suffer losses | Estimate heterogeneous effects across all dimensions; build heat map of vulnerability |
| **Sample bias** | ~75% of SHP have no firm_id; sample is likely biased toward large firms and structured employment | Sample selection mechanism unknown; representativeness to full Swiss labor force unclear | Compare sample to census/LFS; use IPW to adjust for selection; test firm size bias |

---

## Proposed Empirical Tests (Priority Order)

### Test 1: Pre-Treatment Wage Trends (Medium effort, high value)
**Why:** Isolates selection on unobservables from causal effect

**Method:** 
- Identify all workers who will experience AI exposure in year T
- Examine their wage growth in years T-3 to T-1
- Compare to workers who never experience exposure
- If pre-trend differs, selection is present

**Threshold:** If pre-trends differ at p < 0.05, selection bias is non-negligible. Estimate bounds on causal effect using methods from Altonji et al. (2005).

### Test 2: Mechanism Decomposition via Job Titles (High effort, very high value)
**Why:** Distinguishes displacement (task loss within role) from downgrading (occupational downgrade)

**Method:**
- Within each firm, identify workers' job titles pre- and post-AI adoption
- Classify transitions:
  - **No change:** Same title pre/post
  - **Downgrade:** Title moves to lower-wage quartile within firm
  - **Exit:** Worker leaves firm
- Re-estimate FY-FE effect separately for each group:
  - β(wage loss | no title change) — suggests task displacement or bargaining loss
  - β(wage loss | downgrade) — suggests occupational downgrading
  - β(wage loss | exit) — suggests firm contraction or worker dissatisfaction

**Expected result:** If women suffer larger losses due to downgrading (not displacement), that's a different political story (firm discrimination) than task displacement (technology disruption).

### Test 3: Heterogeneous Effects Heat Map (Medium effort, high value)
**Why:** Identifies which workers are politically vulnerable

**Method:**
```
Education × Sector × Firm Size matrix
Rows: No college, Vocational, Bachelor+
Cols: Manufacturing, Finance, Services, Other
Size: SME vs. large
Values: β(AI exposure) for each cell
```

Fill in wage loss coefficients. Focus on cells with:
- Large negative coefficients (high vulnerability)
- Large sample sizes (political salience)
- High AI adoption (mechanism is active)

**Political application:** You can then predict which groups demand which policies.

### Test 4: Political Attribution Experiment (Survey, moderate effort, critical for interpretation)
**Why:** Bridges wage effect to political response

**Method:**
- Survey SHP respondents (or representative sample)
- Randomize scenario vignettes: wage loss due to [AI / globalization / firm greed]
- Ask policy preferences for each scenario
- Analyze: Does policy demand differ by attributed cause?

**Expected finding:** If political response depends on attribution, this reveals which narrative frames will matter for political organization.

---

## Conclusion

These results suggest **task displacement or bargaining erosion within firms**, with women and older workers disproportionately affected. But to understand the political response, we need:

1. **Mechanism clarity:** Are workers displaced from tasks, downgraded to lower roles, or losing bargaining power?
2. **Selection bounds:** How much of the wage loss reflects treatment vs. selection on unobservables?
3. **Heterogeneous vulnerability:** Which education × sector combinations are politically active and will demand policy?
4. **Attribution evidence:** Do workers perceive the cause you've identified, or attribute losses to something else?
5. **Political bridge:** Do wage losses translate to political demand, and on what timeline?

The income reversal analysis demonstrates compositional sophistication and econometric care. But it remains a labor market analysis, not yet a political economy argument. The next phase must move from "AI exposure predicts wage loss" to "workers experiencing wage loss demand specific policies *because* they attribute loss to AI and believe policy X will redress harm."

Without this bridge, you have a regression coefficient. With it, you have a political story.

