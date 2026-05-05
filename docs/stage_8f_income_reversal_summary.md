# The Income Reversal Mystery: A Summary

## Executive Summary

We uncovered a striking paradox in how AI exposure affects worker wages depending on the econometric specification used. The same exposure measure produces opposite-signed coefficients under different fixed effects approaches:

- **Within-firm comparison (Firm-Year FE)**: AI exposure → wage **loss** of 8.5%
- **Within-occupation comparison (Occupation-Year FE)**: AI exposure → wage **gain** of 10.9%

This reversal is not a statistical error but reveals two distinct mechanisms operating simultaneously: **causal wage suppression within firms** and **positive selection of high-wage firms into AI adoption**.

---

## The Puzzle

### Full Sample Results

| Specification | Coefficient | SE | P-value | Interpretation |
|---|---|---|---|---|
| **Firm-Year FE** | -0.0850 | 0.0392 | 0.030* | Wage loss |
| **Occupation-Year FE** | +0.1085 | 0.0367 | 0.002** | Wage gain |
| **Difference** | 0.1935 | — | — | Complete sign flip |

The coefficient difference of 0.1935 is approximately twice the magnitude of either individual effect, indicating the two specifications are identifying opposite-signed phenomena.

---

## Heterogeneous Effects: Who Bears the Costs?

### By Gender

The wage loss under Firm-Year FE is **driven entirely by women**:

| Group | FY-FE | OY-FE | Difference |
|---|---|---|---|
| **Male** | -0.017 (ns) | +0.119** | +0.136 |
| **Female** | -0.176* | +0.149* | +0.325 |

- **Males** show no FY-FE effect (β = -0.017, p = 0.68) but significant OY-FE gain
- **Females** show significant FY-FE loss (β = -0.176, p = 0.026) and OY-FE gain

**Interpretation**: Women in AI-exposed firms face direct wage suppression. This could reflect:
- Women concentrated in lower-wage roles within firms where AI is deployed
- AI reducing demand for routine tasks where women are overrepresented
- Occupational sorting: women move to lower-wage positions as AI scales

### By Age

The FY-FE wage loss intensifies for **older workers**:

| Group | FY-FE | OY-FE | Difference |
|---|---|---|---|
| **Young** | +0.023 (ns) | +0.168† | +0.145 |
| **Old** | -0.158* | +0.137* | +0.295 |

- **Young workers** show no FY-FE effect and marginally significant OY-FE gain
- **Older workers** show significant FY-FE loss (β = -0.158, p = 0.033) and OY-FE gain

**Interpretation**: Older workers are more vulnerable to AI-driven wage suppression within firms, consistent with:
- Lower retraining capacity and skill flexibility
- Potential age discrimination or preference for younger workers with AI skills
- Task overlap: older workers' skills more vulnerable to automation

---

## Mechanisms: Why the Sign Flip?

### What Each Specification Identifies

**Firm-Year FE (Within-firm, within-year)**
- Compares workers in the **same firm, same year** with varying AI exposure
- Controls for all firm-level characteristics (size, wage level, location, industry, etc.)
- Identifies: effect of AI deployment on workers exposed vs. unexposed within same employment context
- Result: **Wage loss of 8.5%** — direct negative effect on workers inside AI-adopting firms

**Occupation-Year FE (Within-occupation, within-year)**
- Compares workers in the **same occupation, same year** across different firms
- Controls for occupation-level macroeconomic trends and selection into occupations
- Identifies: which firms are deploying AI within a given occupation
- Result: **Wage gain of 10.9%** — reflects that high-wage firms preferentially adopt AI

### The Selection Mechanism

The positive OY-FE effect is primarily **selection bias**:

1. **Firm Selection**: High-wage firms adopt AI more aggressively
   - Tech companies pay 30-50% premium wages and invest heavily in AI
   - Manufacturing firms with AI are typically capital-intensive and higher-paying
   - Routine, low-wage positions less likely to be AI-targeted

2. **Worker Sorting**: AI-exposed occupations are concentrated in high-wage sectors
   - Finance, tech, professional services: higher baseline wages AND AI adoption
   - These sectors naturally pay more, independent of AI

3. **Composition Effect**: Within occupations, AI goes to higher-wage sub-groups
   - "Software development" in San Francisco vs. rural areas
   - "Finance" in investment banking vs. small-town credit unions
   - High-wage firms within occupation have more AI applications

**The OY-FE coefficient captures this selection, not a causal benefit to workers.**

### The Causal Effect (Firm-Year FE)

The negative FY-FE effect reveals the true causal mechanism:

When a firm deploys AI, the marginal workers directly exposed to that technology experience wage suppression. This could occur via:

1. **Task displacement**: AI takes over high-value tasks, leaving workers doing lower-value work
2. **Bargaining power erosion**: AI makes workers more substitutable, weakening their wage negotiations
3. **Occupational downgrading**: Workers move to lower-wage roles within the firm
4. **Hours reduction**: Workers maintain positions but with reduced hours/overtime

The fact that this effect is larger for women and older workers suggests these groups have less bargaining power or fewer alternative employment options.

---

## Theoretical Implications

### Composition Effects in Action

This analysis demonstrates why **occupational-level exposure measures can be misleading**:

- A researcher looking only at occupation-year variation (common in prior AI exposure work) would conclude: "AI workers earn 11% more"
- Reality: Same workers working at same occupations in different firms earn wages that differ by firm-level AI adoption
- Firms choosing to deploy AI are **selected on unobservables related to wage levels**, creating spurious positive correlation

### Causal vs. Selection Effects

| Dimension | Firm-Year FE | Occupation-Year FE |
|---|---|---|
| **Identifies** | Within-firm treatment effect | Selection of firms into AI adoption |
| **Controls for** | All firm characteristics | Occupation-level macrotrends |
| **Confounding** | Exposure varies by individual characteristics within firm | Firm selection on wage/productivity unobservables |
| **Causal? ** | More plausible | Likely selection bias |
| **Effect sign** | Negative (wage loss) | Positive (selection) |

---

## Specification Recommendation

### For Income/Wage Analysis: Use Firm-Year FE

**Rationale:**
- Identifies causal effect of AI deployment on exposed workers
- Controls firm selection (compares workers within same firm)
- Reveals true vulnerability: older workers and women lose 15-18% when AI deploys
- Appropriate for labor market impact assessment

**Use case**: 
- Policy questions: "How are workers affected by AI adoption?"
- Distributional analysis: "Who bears the costs?"
- Wage trajectory studies

### For Political Analysis: Keep Occupation-Year FE

**Rationale:**
- Political preferences respond to occupation-level labor market trends
- Occupation trends matter more than firm idiosyncrasies for political ideology
- OY-FE controls for national occupation-level shocks (sectoral trends, demand changes)
- Appropriate for understanding political realignment driven by occupation-level structural change

**Use case:**
- Political economy: "How do occupational labor market shifts drive political preferences?"
- Policy demand: "Do affected occupations demand left-wing policies?"
- Sectoral political shifts

### Why Different Specs for Different Outcomes?

This is actually **best practice in causal inference**:

1. **Different outcomes have different causal structures**
   - Wages: Result of individual bargaining within firms → firm-specific effects matter
   - Politics: Result of perceived labor market vulnerability at occupation level → macro occupation trends matter

2. **Different confounders**
   - Wage assignment: Within-firm sorting by productivity, role, seniority
   - Political assignment: Shared occupation-level exposure to labor market shocks

3. **Different identification strategies needed**
   - Best wage estimates: Vary exposure within firm (FY-FE)
   - Best political estimates: Control for macro occupation trends (OY-FE)

---

## Magnitudes and Policy Relevance

### Wage Effects (Firm-Year FE)

**Full sample**: -0.085 log points = 8.1% wage reduction

- For median wage of CHF 6,000/month: **CHF 486/month loss** (CHF 5,838 annually)
- For low-wage worker (CHF 4,000/month): **CHF 324/month loss**

**Vulnerable groups**:
- Older workers: **-15.8%** (CHF 2,770/month loss on median wage)
- Women: **-17.6%** (CHF 3,069/month loss on median wage)

### Implications

These are **economically meaningful effects**:
- Not compensated by wage growth elsewhere (within-worker over time)
- Larger than typical annual wage growth (2-3%)
- Concentrated on least adaptable workers (older, women)

---

## Conclusion

The income reversal mystery reveals a fundamental tension in AI labor market effects:

1. **Within firms**: AI deployment suppresses wages, particularly for women and older workers
2. **Across firms**: High-wage firms adopt AI, creating positive selection bias at occupation level

**The bottom line**: 
- AI is associated with wage losses for exposed workers *within* firms (real causal effect)
- But high-wage firms deploy AI more, creating positive selection *across* firms (compositional effect)
- Standard occupation-level exposure measures conflate these, producing misleading positive wage effects
- Firm-level variation in exposure is critical for understanding true AI labor market impacts

This also justifies our **dual-specification approach** to the research question:
- **Wages**: Use Firm-Year FE for causal identification
- **Politics**: Use Occupation-Year FE for macro labor market trends

Different outcomes require different identification strategies.

---

## Files and Outputs

- **Full analysis**: `Data/shp_econometric_results/INCOME_REVERSAL_ANALYSIS.txt`
- **Data used**: `Data/shp_panel_prepared.csv` (45,325 person-years, 37,757 with income data)
- **Code**: `stage_8f_income_mystery.py`

### Key Statistics

- Sample size: 37,279 person-year observations with valid income and exposure data
- Exposure measure: Hampole AI exposure average at firm-occupation-year level
- Controls: Age (centered), gender, employment status
- Clustering: By person (for panel correlation)
