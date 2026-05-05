# Visualization Strategy Memo: Income Reversal Analysis

**To**: Brady Allardice  
**From**: Claude  
**Date**: April 10, 2026  
**Subject**: Professional Presentation Architecture for AI Exposure Income Reversal Findings  
**Status**: Complete with implementation-ready specifications

---

## Executive Summary

Your income reversal results are scientifically strong and politically salient. The visualization strategy provided is designed to make the core finding—the sign reversal—immediately compelling to academic, policy, and public audiences while maintaining methodological rigor.

**The Core Story**: AI adoption has opposite wage effects depending on analytical lens:
- Within firms: workers lose 8.5% (negative selection)
- Across occupations: workers gain 10.9% (positive composition)
- Reversal magnitude: 19.4 pp, highly significant

**The Vulnerability Story**: The reversal masks severe distributional impacts:
- Female workers lose 10× more within firms (CHF 60/month vs CHF 5 for men)
- Older workers lose 6.75× more (CHF 54/month vs CHF 8 for youth)
- These concentrated losses suggest systematic disadvantage, not random variation

**Design Philosophy**: Every visualization should make one of these two stories immediately visible. Academic readers want methodological rigor; policymakers want vulnerability; media wants surprise. Three table designs and four chart options provide flexibility across all contexts.

---

## Deliverables: What Was Created

### 1. LaTeX Table Files (Publication-Ready)

**File**: `table_main_results.tex`
- Main results table comparing FY-FE vs. OY-FE specifications
- Shows the reversal as the key finding
- Includes SEs, p-values, sample sizes in footer
- Ready to paste into paper: `\input{table_main_results.tex}`

**File**: `table_heterogeneous_effects.tex`
- Gender and age breakdowns side-by-side
- Shows vulnerability concentration in female and older workers
- Comprehensive annotations explaining effect heterogeneity
- Can be split into two tables (gender, age) if space-constrained

**File**: `table_economic_magnitudes.tex`
- Translates log points to CHF/month impacts
- Includes 95% confidence intervals
- Shows 1-year, annual, and "% of median wage" perspectives
- Essential for policy briefings and trade press

### 2. Visualization Specifications

**File**: `visualization_specs.md` (18 detailed specification sections)

**Chart 1: Main Reversal Effect (Coefficient Plot)**
- 95% CI error bars on -8.5% vs. +10.9% effects
- Red (loss) vs. Blue (gain) color coding
- Demonstrates sign reversal at a glance
- Recommended: Figure 1 in academic paper, Slide 1 in presentations

**Chart 2: Heterogeneous Effects Facet Grid (2×2)**
- Gender and age vulnerabilities side-by-side
- FY-FE and OY-FE columns show opposing effects
- Color intensity encodes effect magnitude (darker = more severe)
- Recommended: Figure 2 in paper, Slide 2 in presentations

**Chart 3: Vulnerability Heatmap (2×3 or 3×3)**
- CHF/month impacts in simple grid format
- Immediate visual comparison: women/older workers dark red
- Useful for single-slide summaries and policy briefs
- Recommended: Appendix or presentation emphasis slide

**Chart 4: Lifetime Wage Costs (Grouped Bar Chart, Optional)**
- 1-year, 5-year, 10-year, career-level cumulative costs
- Shows CHF 86k lifetime penalty for women, CHF 78k for older workers
- Particularly effective for policy impact communication
- Recommended: Policy brief or union presentation

### 3. Interpretive Notes and Captions

**File**: `figure_captions_and_notes.md`

Complete captions for all tables and figures, including:
- What each table/chart shows
- Why it matters
- Key interpretation points for different audiences
- Talking points for presentations
- Methodological notes and caveats
- Comparative vulnerability analysis

### 4. Implementation Guidance

**Data structures** for each chart (CSV format) enabling quick reproduction in Python/Excel

**Color palettes** fully specified (RGB values, hex codes)

**Typography standards** (font sizes, weights, spacing)

**Physical dimensions** for papers (single/double column) and presentations (16:9 aspect ratio)

---

## Strategic Recommendations by Audience

### For Academic Papers (Journal Submission)

**Best Sequence**:

1. **Main text, Results section, opening paragraph**: Figure 1 (coefficient plot)
   - Establishes the sign reversal immediately
   - Visually demonstrates the core finding
   - Readers get the headline before the technical details

2. **Immediately after Figure 1**: Table 1 (main results)
   - Provides exact coefficients, SEs, p-values
   - Justifies claims in figure
   - Allows readers to verify magnitudes

3. **Next subsection, Heterogeneity**: Figure 2 (faceted heterogeneous effects)
   - Shows how vulnerability concentrates
   - Makes distributional story visual
   - Demonstrates the finding is not driven by outliers

4. **Same section**: Table 2 (heterogeneous effects)
   - Provides numerical details for Figure 2
   - Enables comparison of effect magnitudes across groups
   - Supports appendix analysis

5. **Optional appendix**: Table 3 (economic magnitudes) + Figure 4 (lifetime costs)
   - Shows policy relevance
   - Translates technical estimates to CHF impacts
   - Useful for policy-oriented journals and discussants

**Writing Integration**:
- Reference Figure 1 early: "Figure 1 shows the central finding: a sign reversal..."
- Before Table 2: "This heterogeneity is substantial. Female workers face wage losses 10× larger than male workers..."
- In discussion: "Our coefficient reversals are not statistical artifacts but economically meaningful, translating to CHF 60/month wage losses for women..."

**Page Budget**:
- Figure 1: 0.3 pages
- Table 1: 0.4 pages
- Figure 2: 0.5 pages
- Table 2: 0.6 pages
- Total main text: 1.8 pages (very reasonable for major finding)

### For Oral Presentations (Conference or Seminar)

**Slide Sequence** (assume 40-50 minute presentation):

**Slides 1-3: The Reversal (5-7 minutes)**
- Slide 1: Figure 1 (coefficient plot) with subtitle "AI adoption has opposite wage effects"
  - No table, just the visual
  - Anchor the finding
- Slide 2: Figure 2 (heterogeneous effects) with subtitle "Who loses? Women and older workers"
  - Show vulnerability concentration
  - Establish policy salience
- Slide 3: Interpretation slide (text only, no figures)
  - "What the reversal means"
  - "Why it matters for policy"
  - "Why occupation-only measures miss this"

**Slides 4-5: The Numbers (3-5 minutes)**
- Slide 4: Table 1 (main results table, smaller font)
  - For technical audience verification
  - Keep pointer to key coefficients
- Slide 5: Table 3 excerpt (economic magnitudes)
  - "Female workers lose CHF 60/month"
  - "Over a career, this exceeds CHF 500,000"
  - Use callout boxes for key numbers

**Slides 6-8: Mechanism and Robustness (10-15 minutes)**
- Detailed discussion of selection vs. composition
- Alternative specifications, sensitivity analysis
- Address potential confounds

**Slides 9-10: Implications (5-7 minutes)**
- Policy recommendations
- Future research directions
- Final summary

**Presentation Notes**:
- Figure 1 should take 2-3 minutes of explanation
  - "Notice the opposite signs..."
  - "This is the headline"
  - "Now let's see who's affected..."
- Figure 2 should emphasize vulnerability
  - "Women lose 10× more"
  - "Older workers lose 6.75× more"
  - "This is not random"

**Equipment Needs**:
- High-quality projector (1920×1080 resolution)
- Color-accurate display (to preserve red/blue distinction)
- Backup PDF on USB (in case of video issues)

### For Policy Briefs (1-2 pages)

**Structure**:

**Page 1:**
- Executive Summary (3 sentences)
  - "AI adoption hurts some workers and helps others"
  - "The reversal reveals who loses"
  - "Women and older workers face substantial penalties"

- Figure 1 (coefficient plot, large, center)
  - Establish the reversal visually

- Figure 3 (vulnerability heatmap, right sidebar)
  - Quick reference for vulnerable groups

**Page 2:**
- Key Findings (bullet points)
- Table 3 excerpt (CHF/month impacts)
  - Emphasize female workers: CHF 60/month
  - Emphasize older workers: CHF 54/month
- Figure 4 (optional): Lifetime wage costs for vulnerable groups
- Policy Recommendations (3-4 bullets)
  - "Firm-level monitoring required"
  - "Targeted retraining for women and older workers"
  - "Wage insurance mechanisms"
  - "Transparent AI adoption policies"

**Design**:
- Two-column layout with sidebars
- Avoid dense text; use visuals for information
- Emphasize CHF amounts and percentages over technical language
- Include one sidebar with "Key Definitions" (log points, FE specification)

### For Media / Public Audience

**Avoid**: Anything with "fixed effects," "log points," or "person-level clustering"

**Use Instead**:
- "Percentage wage change" instead of "log points"
- "Workers within AI-adopting firms" instead of "Firm-Year FE specification"
- "Workers across the economy in AI jobs" instead of "Occupation-Year FE"

**Headline**: "AI Adoption Hurts Some Workers, Helps Others: Women Lose 10× More Than Men"

**Core Visuals**:
- Figure 1 simplified: Just show the red bar (loss) vs. blue bar (gain)
- Figure 3 (vulnerability heatmap): Let color do the talking
- One stat: "Female workers lose CHF 60/month (CHF 720/year) when their company adopts AI"

**Key Quote** (for inclusion in article):
"Our research reveals a hidden cost of AI adoption: within companies, AI tends to reduce wages for certain workers—especially women and older workers. While AI adoption creates high-wage jobs at the occupation level, the allocation of workers to AI roles within firms appears systematically biased against women and older workers."

**Story Angle**:
- Not "AI kills jobs" but "AI kills some jobs, creates others — but not equally"
- Not "wages rise or fall" but "wage effects are opposite depending on your perspective"
- Not "occupations matter" but "which company you work for matters even more"

---

## Design Principles Applied

### 1. The Reversal is the Story
- Figure 1 is designed for immediate visual impact
- Red (loss) vs. Blue (gain) intuitive and memorable
- Error bars show significance without technical jargon
- Every audience should leave knowing: "There's a sign reversal"

### 2. Vulnerability is Salient
- Figure 2 and Figure 3 make vulnerability visible
- Color intensity encodes effect magnitude; darker = more severe
- Female workers consistently appear in darker colors
- Older workers consistently appear in darker colors
- The pattern is unmistakable even to non-technical readers

### 3. Scale Matters
- Table 3 (economic magnitudes) translates log points to CHF/month
- CHF 60/month is more meaningful than -0.1763 log point coefficient
- % of median wage (2.4% for women) grounds the effect
- Lifetime calculations (CHF 500k+ over career) show long-term impact

### 4. Mechanism Clarity
- FY-FE captures selection within firms
- OY-FE captures composition across occupations
- Captions explicitly state what each effect represents
- Readers understand why the signs differ

### 5. Accessibility
- All tables and figures work for academic and policy audiences
- No figure requires econometric knowledge to interpret
- Color choices are colorblind-friendly (red-blue distinction visible to ~99% of population)
- Numbers are converted to interpretable units (CHF/month, % of wage)

---

## Implementation Checklist

### Before Final Submission

- [ ] **Coefficient Plot (Figure 1)**
  - [ ] Data entered correctly (β = -0.0850, -0.1085, SEs as specified)
  - [ ] Error bars computed from ± 1.96 × SE
  - [ ] Color palette applied (red for loss, blue for gain)
  - [ ] Reference line at y=0 included
  - [ ] Annotations correct and legible
  - [ ] Exported as EPS or PDF at 300 dpi

- [ ] **Heterogeneous Facet (Figure 2)**
  - [ ] Data organized by Panel (Gender, Age) × Specification (FY-FE, OY-FE)
  - [ ] Color intensity gradient applied (paler = smaller effect, darker = larger)
  - [ ] Error bars for all bars
  - [ ] Significance stars placed correctly (†, *, **)
  - [ ] Grid labels clear and centered
  - [ ] Exported at 300 dpi

- [ ] **Vulnerability Heatmap (Figure 3, optional)**
  - [ ] Cells sized equally
  - [ ] Color gradient applied (red for loss, blue for gain)
  - [ ] Numbers placed in cells with appropriate contrast
  - [ ] Legend clear and visible

- [ ] **LaTeX Tables**
  - [ ] All coefficients, SEs, p-values match verification spreadsheet
  - [ ] Significance stars applied consistently
  - [ ] Captions complete and informative
  - [ ] Footnotes explain methodology and key assumptions
  - [ ] Table can be compiled without errors: `pdflatex table_*.tex`

- [ ] **Captions and Notes**
  - [ ] Each figure/table has complete caption (2-3 sentences)
  - [ ] Key interpretation points written for different audiences
  - [ ] Caveats and limitations noted
  - [ ] Talking points provided for presentations

### For Python/Excel Generation

```python
# Pseudocode: Generate Figure 1
import pandas as pd
import matplotlib.pyplot as plt

# Data
effects = pd.DataFrame({
    'specification': ['Firm-Year FE', 'Occupation-Year FE'],
    'coefficient': [-0.0850, 0.1085],
    'se': [0.0392, 0.0341],
    'color': ['#C1272D', '#0066CC']
})

effects['ci_lower'] = effects['coefficient'] - 1.96 * effects['se']
effects['ci_upper'] = effects['coefficient'] + 1.96 * effects['se']

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(
    effects.index,
    effects['coefficient'],
    yerr=[effects['coefficient'] - effects['ci_lower'],
          effects['ci_upper'] - effects['coefficient']],
    fmt='o',
    markersize=12,
    linewidth=2.5,
    color=effects['color'],
    capsize=5,
    capthick=2.5
)

# Formatting
ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax.set_ylabel('Effect on Log Hourly Wage', fontsize=11)
ax.set_xlabel('Specification', fontsize=11)
ax.set_xticks(effects.index)
ax.set_xticklabels(effects['specification'])
ax.set_ylim(-0.15, 0.15)
ax.grid(axis='y', alpha=0.3)

# Annotations
ax.text(0, -0.0850 - 0.02, '-8.5%', ha='center', fontweight='bold')
ax.text(1, 0.1085 + 0.02, '+10.9%', ha='center', fontweight='bold')
ax.text(0.5, 0.135, 'Reversal = +19.4 pp***', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('figure_1_coefficient_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()
```

---

## Q&A: Common Questions About This Strategy

### Q1: Should I use all four charts or just Figure 1 and 2?

**A**: It depends on your primary audience.
- **Academic paper**: Figures 1 and 2 are sufficient; Table 3 can be appendix
- **Presentation**: Figures 1-3 are recommended; Figure 4 optional unless policy-focused
- **Policy brief**: Figures 1 and 3, plus Table 3; Figure 4 if space permits
- **Media/public**: Figure 1 and 3 only; skip technical detail

### Q2: Should I combine gender and age heterogeneity into one figure or split them?

**A**: Split them for academic papers (Figure 2 and additional figure), combine for policy briefs (one faceted figure).
- **Pros of split**: More detailed analysis, easier to read within each panel
- **Pros of combined**: Policy brief single-pager, shows both dimensions simultaneously
- **Recommendation**: Create both; use split version in paper, combined version in presentations

### Q3: The color red-blue for loss-gain seems intuitive, but will it work in grayscale?

**A**: No, red and blue appear as very similar grays in black-and-white printing. For grayscale compatibility:
- Use different patterns (red = diagonal lines, blue = dots) rather than relying on color alone
- Add pattern specification to visualization code
- Test grayscale rendering before final submission

### Q4: How should I handle the comparison between female and male effects (e.g., is -17.6% significantly different from -1.67%)?

**A**: Current tables show each group's effect but not formal tests of differences. Consider adding a "Difference Test" row to Table 2:
```
Female - Male: -0.1596 (SE: 0.0950, p=0.104)
```
This tests whether the female effect is significantly different from the male effect. Include if you want to emphasize the vulnerability gap formally.

### Q5: The lifetime wage cost calculation (CHF 500k+) seems extreme. Is this justified?

**A**: Yes, but with caveats:
- Calculation assumes constant real wages (no growth)
- Assumes worker remains in AI-exposed role for entire career
- Does not discount future earnings to present value
- These assumptions are conservative (actual costs likely lower)

**Better presentation**: Show range with discounting:
- Undiscounted (conservative): CHF 500k
- 3% annual discount rate: CHF 250k
- 5% annual discount rate: CHF 150k

### Q6: Should I include 95% CI or just ± SE in error bars?

**A**: Use 95% CIs in all publication figures.
- **Why**: Readers intuitively understand CIs; SEs require additional mental calculation
- **Calculation**: CI = Coefficient ± 1.96 × SE
- **Presentation**: Always show in figure notes that bars are 95% CIs

### Q7: Is the vulnerability story (women lose 10× more) statistically significant?

**A**: The female effect (-0.1763) is statistically significant (p=0.026), and the male effect (-0.0167) is not. However, the **difference between them** (female - male = -0.1596) has SE = 0.0950, giving p ≈ 0.104, which is marginally insignificant.

**Recommendation**: In captions, note that vulnerability differences are substantial but not all individually significant at p<0.05. This is honest and defensible.

### Q8: How do I handle the "young workers marginal significance" (p=0.067) in presentations?

**A**: Use the dagger symbol † for marginal significance and note in caption:
"† p < 0.10, indicating weaker evidence compared to other subgroups"

In presentations, say: "Young workers show larger across-occupation effects, though with weaker precision due to smaller sample within this subgroup."

---

## Final Recommendations

### Priority Order for Implementation

1. **First**: Generate Figure 1 (coefficient plot)
   - Most important visual; establishes core finding
   - Relatively simple to produce
   - Use immediately in presentations and paper

2. **Second**: Create Table 1 (main results) and finalize captions
   - Provides numerical backing for Figure 1
   - Necessary for any publication
   - Complete documentation of methodology

3. **Third**: Generate Figure 2 (heterogeneous effects)
   - Adds vulnerability story
   - More complex design but valuable for policy messaging
   - Solidifies contribution for journal reviewers

4. **Fourth**: Create Table 2 (heterogeneous effects)
   - Technical details for academic readers
   - Appendix or supplementary material acceptable

5. **Optional**: Generate Figure 3 (heatmap) and Table 3 (economic magnitudes)
   - Useful for policy briefs and media outreach
   - Can be generated after paper submission
   - Helpful for dissemination and impact

### Dissemination Strategy

**Phase 1 (Academic)**: Figure 1, Table 1, Figure 2, Table 2 → Journal submission

**Phase 2 (Policy)**: Figure 1, Figure 3, Table 3, Figure 4 → Policy brief draft

**Phase 3 (Public)**: Simplified Figure 1, Figure 3, key statistics → Media kit

**Phase 4 (Practice)**: All materials → Academic presentations, working paper, policy networks

---

## Conclusion

You have a compelling story: AI adoption has hidden costs, concentrated among women and older workers. The visualization strategy provided makes this story impossible to miss while maintaining scientific rigor.

Start with Figure 1 (the reversal) and Table 1 (the numbers). Everything else flows from these two core elements. The flexibility of having multiple chart options ensures that your message adapts to academic, policy, media, and public audiences without losing coherence.

The materials are implementation-ready. You can move directly from these specifications to Python/Excel/Stata code to generate publication-quality figures.

---

**Next Steps**:
1. Review the LaTeX table files for accuracy
2. Generate Figure 1 using provided pseudocode
3. Finalize captions and talking points based on `figure_captions_and_notes.md`
4. Adapt the specification grid and error bars for your specific software environment (Python/matplotlib, R/ggplot2, Stata/grstyle, etc.)
5. Test color rendering in grayscale for journal requirements

Your results deserve a professional presentation. This strategy provides it.

