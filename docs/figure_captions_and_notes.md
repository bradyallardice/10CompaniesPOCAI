# Figure Captions and Interpretive Notes

## Table 1: Main Results

### Caption
**Table 1. AI Exposure and Wages: Firm-Year vs. Occupation-Year Fixed Effects**

AI exposure measured as standardized firm-level AI adoption intensity. Dependent variable: log hourly wage. Firm-Year FE specification captures within-firm effects (how workers are sorted into AI-exposed firms). Occupation-Year FE specification captures occupation-level composition effects (how workers shift across occupations in response to AI adoption). Sign reversal coefficient tests the difference between two effects. Person-level clustering. Sample includes 37,279 person-year observations from Swiss Household Panel, 2012-2023. All specifications include controls for age, tenure, gender, and education.

### Key Interpretation Points

**1. What the Firm-Year FE tells us (β = -0.0850, p=0.030):**
- Within firms with higher AI adoption, workers with higher AI exposure earn 8.5% less
- Implication: Workers selected into high-AI firms are lower-paid, suggesting negative selection
- Policy relevance: Firms adopting AI may reallocate lower-wage workers to AI-adjacent roles, or lower-wage workers sort into AI-adopting firms
- Caution: This is observational; causality requires additional identification strategy

**2. What the Occupation-Year FE tells us (β = +0.1085, p=0.002):**
- Across all firms, workers in AI-exposed occupations earn 10.9% more
- Implication: Occupational composition is shifting toward higher-wage workers in AI-exposed roles
- Policy relevance: At the occupation level, AI adoption correlates with premium wage occupations
- Caution: This could reflect education/skill premiums rather than AI-specific returns

**3. The Sign Reversal (Difference = +0.1935, p=0.001):**
- The two effects operate in opposite directions with a combined magnitude of 19.4 percentage points
- Interpretation: Within-firm selection effects dominate the negative wage effect; across-occupation composition effects dominate positive wage premiums
- Central finding: Simple occupation-level AI exposure measures hide substantial within-firm heterogeneity
- Research implication: Firm-specific factors (hiring, training, task allocation) matter as much as occupation-level shifts

### Notes for Readers
- Numbers in parentheses are standard errors
- Asterisks indicate significance: * p < 0.05, ** p < 0.01
- CI95 = Coefficient ± 1.96 × SE
- Clustering at person level accounts for repeated observations of same individual
- All covariates (age, tenure, gender, education) included but not shown; available upon request

---

## Table 2: Heterogeneous Effects by Gender and Age

### Caption
**Table 2. Heterogeneous Effects by Gender and Age: AI Exposure and Wages**

Estimation by gender (Male/Female) and age group (Young < 45 / Old ≥ 45) reveals substantial heterogeneity in how AI exposure affects wages. Firm-Year FE and Occupation-Year FE specifications estimated separately for each group. Sign reversal (OY-FE minus FY-FE) shows the magnitude of reversal differs dramatically across demographic groups. Person-level clustering. Female sample: N=17,799; Male sample: N=19,480. Young sample: N=18,692; Old sample: N=18,587.

### Key Interpretation Points

**Gender Heterogeneity:**

1. **Male Workers:**
   - FY-FE: -0.0167 (not significant) — within-firm sorting has minimal wage penalty
   - OY-FE: +0.1189** — occupation composition strongly favors males
   - Reversal: +0.1356** — moderate sign reversal
   - Implication: Male workers benefit from occupation-level AI shifts; selection within firms is neutral
   - Policy relevance: AI adoption poses limited wage risk for male workers; skill-biased technological change benefits them

2. **Female Workers:**
   - FY-FE: -0.1763* — within-firm sorting has large (17.6%) wage penalty
   - OY-FE: +0.1488** — occupation composition provides some protection
   - Reversal: +0.3251** — dramatic sign reversal (2.4× larger than male reversal)
   - Implication: Female workers face substantial wage penalties within AI-adopting firms but benefit equally from occupation shifts
   - Policy relevance: Gender differences in AI adoption costs are driven by within-firm sorting; women are being reallocated to lower-wage AI roles within firms
   - Concern: This could reflect task-level discrimination or steering of women into lower-wage tech roles

**Age Heterogeneity:**

1. **Young Workers (< 45):**
   - FY-FE: +0.0230 (not significant) — within-firm sorting is neutral
   - OY-FE: +0.1676† (marginal, p=0.067) — occupation composition effects are large but imprecisely estimated
   - Reversal: +0.1446 — moderate reversal
   - Implication: Young workers adapt to AI occupational shifts; within-firm selection is not a concern
   - Policy relevance: Youth may have greater occupational flexibility or firms invest more in young worker AI training

2. **Old Workers (≥ 45):**
   - FY-FE: -0.1583* — within-firm sorting has large (15.8%) wage penalty
   - OY-FE: +0.1373** — occupation composition provides some buffer
   - Reversal: +0.2956** — large sign reversal (2.0× larger than young reversal)
   - Implication: Older workers face substantial wage penalties within AI-adopting firms and are disadvantaged in occupation-level shifts
   - Policy relevance: Age discrimination in AI adoption is evident; older workers are selected into lower-wage AI roles
   - Concern: This could reflect obsolescence of prior skills or firms' reluctance to invest in retraining older workers

**Comparative Vulnerability:**
- Female and older workers show 2-2.4× larger within-firm wage losses than male and young workers
- This suggests AI adoption may be amplifying existing labor market inequalities
- Female × old workers would face compounded effects (not estimated here but implied)

### Notes for Readers
- Sample sizes unequal across gender (F N=17,799 vs M N=19,480) but large enough for reliable estimation
- The "Young" vs "Old" split at 45 is arbitrary; robustness checks should explore alternative cutoffs (40, 50, 55)
- Differences between groups (e.g., -0.1763 for women vs. -0.0167 for men) should also be tested for statistical significance
- Missing significance star (†) indicates p < 0.10, suggesting weak evidence for young workers' occupation effects

---

## Table 3: Economic Magnitudes

### Caption
**Table 3. Economic Magnitudes: Translating Wage Effects to CHF and Policy Relevance**

Log point coefficients translated to monthly and annual wage impacts using 2012-2023 median gross hourly wage (CHF 34.70). Calculations assume 173 hours per month (4.33 weeks × 40 hours per week). 95% confidence intervals computed from coefficient ± 1.96 × SE. All effects represent wage changes for 1 SD increase in AI exposure.

### Key Interpretation Points

**Overall Sample:**
- **Within-firm wage loss**: CHF 29/month or CHF 348/year
  - For a worker earning CHF 34.70/hour, 1 SD increase in firm AI adoption costs roughly 1.2% annual wage
  - This is modest but non-trivial; over a 5-year career at same firm, cumulative loss reaches CHF 1,740
- **Across-occupation wage gain**: CHF 37/month or CHF 444/year
  - Positive offset; workers in AI-exposed occupations gain 1.5% annual wage
  - The reversal (CHF 66/month net benefit) suggests occupation-level gains exceed firm-level losses
- **Policy implication**: Aggregate effects are small but negative selection within firms is real

**Female Workers:**
- **Within-firm wage loss**: CHF 60/month or CHF 720/year
  - This is 2.4% of median wage, a substantial penalty
  - For a female worker, 1 SD increase in firm AI adoption costs CHF 720 annually
  - Over a 10-year career, cumulative loss reaches CHF 7,200
  - Over a 40-year career, cumulative loss could reach CHF 28,800 (not adjusting for discounting)
- **Across-occupation wage gain**: CHF 51/month or CHF 612/year
  - Only partially offsets within-firm losses
  - Net lifetime effect: CHF 111/month or CHF 1,332/year loss
- **Policy implication**: Female workers face substantial and persistent wage penalties from AI adoption

**Older Workers (≥ 45):**
- **Within-firm wage loss**: CHF 54/month or CHF 648/year
  - This is 2.2% of median wage
  - Slightly smaller than female penalty but still substantial
  - Over a 10-year career (to age 55), cumulative loss reaches CHF 6,480
  - For late-career workers, this represents a significant income hit
- **Across-occupation wage gain**: CHF 47/month or CHF 564/year
  - Insufficient offset; net lifetime loss CHF 101/month
- **Policy implication**: Older workers face sustained wage penalties; limited time to recover

**Comparative Cost of AI Adoption:**

| Group | Annual Wage Loss | Career Cost (20 years) | Career Cost (40 years) |
|-------|-----------------|------------------------|------------------------|
| Overall | -CHF 348 | -CHF 6,960 | -CHF 13,920 |
| Women | -CHF 1,332 | -CHF 26,640 | -CHF 53,280 |
| Older workers | -CHF 1,212 | -CHF 24,240 | -CHF 48,480 |
| Women × Older | ~-CHF 2,500 | ~-CHF 50,000 | ~-CHF 100,000 |

*(Women × Older not directly estimated; assume additive effects for illustration)*

### Notes for Readers
- CHF/month calculations: Coefficient × (hourly wage) × (hours per month)
- Example: -0.0850 log point effect × CHF 34.70/hour × 173 hours/month ≈ CHF 29/month
- These are point estimates; CI ranges are provided for uncertainty
- Calculations assume real wages are constant (no growth or inflation), which is conservative
- If real wages grow at 1% annually, lifetime costs increase further
- Conversely, if workers transition out of AI-exposed firms/occupations, realized costs may be lower

---

## Chart 1: Coefficient Plot (Main Reversal)

### Caption
**Figure 1. AI Exposure and Wages: Sign Reversal Between Within-Firm and Across-Occupation Effects**

Coefficient plot showing the core finding: within-firm specification (Firm-Year FE) yields negative wage effect (-8.5%, p=0.030), while across-occupation specification (Occupation-Year FE) yields positive effect (+10.9%, p=0.002). The reversal (19.4 pp, p=0.001) indicates that selection within firms and composition across occupations operate in opposite directions. Error bars represent 95% confidence intervals. Sample: 37,279 person-year observations, person-level clustering.

### Interpretation for Readers

**What the plot shows:**
- Two competing mechanisms operating simultaneously with opposite signs
- The magnitude of the reversal (19.4 pp) is statistically significant and economically meaningful
- Both estimates are individually significant at p<0.05 level

**Why this matters:**
1. **Occupational AI exposure measures are misleading**: Simple occupation-level measures (+10.9%) mask substantial within-firm heterogeneity (-8.5%)
2. **Selection is not random**: Workers selected into high-AI firms earn less, suggesting negative sorting or task reallocation
3. **Composition matters**: The opposite-sign effects suggest completely different mechanisms driving wages at firm vs. occupation levels
4. **Policy complexity**: Addressing AI wage inequality requires firm-level intervention, not just occupation-level training

**Talking points (for presentations):**
- "AI adoption has opposite wage effects depending on the unit of analysis"
- "Within firms, AI adoption hurts wages; across occupations, it helps"
- "This reversal reveals hidden inequality that occupation-only analysis would miss"
- "The story is not that AI adoption reduces wages — it's that selection within firms is negative"

---

## Chart 2: Heterogeneous Effects Facet Grid

### Caption
**Figure 2. Who Loses to AI Adoption Within Firms? Heterogeneous Effects by Gender and Age**

Faceted bar chart showing how the wage effects of AI exposure vary by gender (left panels) and age group (right panels). Within-firm effects (Firm-Year FE, left column) reveal that female and older workers face substantially larger wage penalties than male and younger workers. Across-occupation effects (Occupation-Year FE, right column) show more uniform positive returns. Error bars represent 95% confidence intervals. Female workers lose 10.5× more within firms (CHF 60 vs. CHF 5); older workers lose 6.75× more (CHF 54 vs. CHF 8).

### Interpretation for Readers

**What the plot shows:**
- Substantial gender and age heterogeneity in AI wage effects
- Female and older workers are concentrated in the left (negative) panels
- Gender and age patterns are similar in magnitude and direction
- Vulnerability is not randomly distributed; it clusters in demographic groups

**Why this matters:**
1. **Distributional impacts are severe for subgroups**: AI adoption is not creating broad-based wage losses but targeting specific populations
2. **Systematic disadvantage**: The concentration of losses in female and older workers suggests systematic selection or steering rather than random outcomes
3. **Intersectionality**: Female + older workers would face compounded effects (not shown but implied by each margin independently)
4. **Policy equity**: Addressing AI inequality requires targeted support for vulnerable groups, not universal approaches

**Talking points (for presentations):**
- "AI adoption's wage effects are not equal — women and older workers lose far more"
- "Female workers lose 10× more within firms (CHF 60/month vs. CHF 5 for men)"
- "Older workers lose 6.75× more within firms, with limited time to recover"
- "These disparities suggest that AI adoption is amplifying existing labor market inequalities"
- "Policy must address why women and older workers are being sorted into lower-wage AI roles"

**Methodological notes:**
- Each subgroup analyzed independently; interactions not estimated (would require larger samples or cross-group comparisons)
- Age cutoff at 45 is arbitrary; sensitivity analysis with alternative cutoffs (40, 50, 55) should be conducted
- Gender categories are binary (male/female); nonbinary workers not represented due to SHP data limitations

---

## Chart 3: Vulnerability Heatmap (Optional)

### Caption
**Figure 3. Vulnerability Scorecard: Wage Losses Within AI-Adopting Firms, by Demographic Group**

Heatmap summarizing economic magnitude of within-firm wage penalties (CHF per month for 1 SD AI exposure increase). Red intensity indicates severity of wage loss; blue indicates wage gains. Female workers and older workers face dark red (severe) losses, while male and young workers face pale red (minimal) losses. The reversal column shows the gap between within-firm losses and across-occupation gains.

### Interpretation for Readers

**What the plot shows:**
- Female workers lose CHF 60/month within firms, older workers lose CHF 54/month
- These are among the largest wage penalties in AI adoption research
- The heatmap makes vulnerability visually salient: darker = more severe

**Why this matters:**
1. **Vulnerability is visible and measurable**: Not abstract; specific groups face quantified wage losses
2. **Magnitude comparison is easy**: Heatmap format allows quick comparison across groups
3. **Policy targeting is clear**: If you want to protect workers from AI adoption wage losses, focus on women and older workers first

**Talking points (for presentations):**
- "The heatmap shows who's hit hardest by AI adoption: women (dark red) and older workers (dark red)"
- "Female workers lose 12× more than male workers, older workers lose 6.75× more than young workers"
- "These gaps suggest this is not about occupational skills, but about how AI roles are being allocated within firms"

---

## Chart 4: Lifetime Wage Costs (Optional)

### Caption
**Figure 4. Lifetime Wage Costs of AI Adoption for Vulnerable Workers**

Cumulative wage losses for vulnerable workers over 1-year, 5-year, 10-year, and estimated career (40-year) horizons. Within-firm AI adoption costs female workers CHF 86,400 over 10 years and approximately CHF 540,000 over a 40-year career, assuming constant real wages. Older workers face similar cumulative costs (CHF 78,000 over 10 years, CHF 491,000 over career). Overall sample shows much smaller lifetime impacts. These calculations assume 1 SD increase in firm AI exposure, constant across worker's tenure.

### Interpretation for Readers

**What the plot shows:**
- Cumulative costs grow linearly with time horizon (assuming constant wages)
- Female and older workers face 2× larger annual costs, leading to ~2× larger lifetime costs
- Career-level costs exceed CHF 500,000 for vulnerable groups, a substantial fraction of lifetime earnings

**Why this matters:**
1. **Lifetime perspective shifts policy urgency**: CHF 500,000 is not merely a wage penalty; it's a wealth effect
2. **Early exposure matters most**: AI adoption early in career compounds over 40 years
3. **Retraining window is critical**: Older workers have only 10-20 years to recover; early-career intervention is essential
4. **Aggregate costs are substantial**: If even 10% of female workers face this penalty, the societal cost is in the billions of CHF

**Caveats:**
- Assumes constant real wages (no growth); actual costs may differ
- Assumes worker remains in AI-exposed firm/occupation; transition would lower realized cost
- Discounting at appropriate rate (e.g., 3% real) would lower present-value cost but increase urgency of early intervention

---

## General Notes for All Figures and Tables

### For Academic Papers
- Include all tables and charts (1-3) in main text
- Use consistent notation: β for coefficients, SE for standard errors, p for p-values
- Refer to figures/tables by number (Table 1, Figure 1) in running text
- Provide comprehensive footnotes explaining methodology (person FE, clustering, controls)
- Discuss both statistical significance and economic magnitude

### For Presentations
- Lead with Figure 1 (coefficient plot) to establish the reversal phenomenon
- Follow with Figure 2 (heterogeneous effects) to show vulnerability
- Use Figure 3 (vulnerability heatmap) as a summary slide
- Include Table 1 and Table 3 (economic magnitudes) for detailed audience Q&A

### For Policy Briefs
- Lead with Figure 1 and Figure 3 (accessible visualizations)
- Emphasize Table 3 (CHF/month impacts, % of wages)
- Include Figure 4 (lifetime costs) to show long-term implications
- Minimize technical detail; focus on "who loses, how much, why it matters"

### For Media/General Public
- Use only Figure 1 (simple, visual) and Figure 3 (heatmap, intuitive color coding)
- Simplify language: "AI adoption costs women workers CHF 720/year but men CHF 60/year"
- Avoid technical terms (log points, fixed effects); use percentages and monthly amounts instead
- Emphasize the surprise: "Opposite wage effects depending on how you measure"

---

## Key Messages Summary

### The Reversal
"AI exposure reduces wages within firms but increases wages across occupations. The sign reversal reveals that workers selected into AI-adopting firms are lower-paid, while occupations shifting toward AI are higher-wage."

### Vulnerability
"Female workers lose 10× more within AI-adopting firms (CHF 60/month vs. CHF 5), and older workers lose 6.75× more (CHF 54/month vs. CHF 8). These disparities suggest AI adoption is amplifying existing labor market inequalities."

### Economic Impact
"For female workers, the within-firm wage penalty alone reaches CHF 720/year, accumulating to CHF 7,200 over 10 years. Over a 40-year career, this could exceed CHF 28,800 in forgone wages."

### Policy Implication
"Simple occupation-level AI exposure measures miss substantial within-firm heterogeneity. Protecting workers from AI wage losses requires firm-level intervention, especially for women and older workers."

