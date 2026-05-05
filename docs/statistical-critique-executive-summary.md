# Executive Summary: Statistical Critique of Political Outcomes Analysis

## BOTTOM LINE

**Model specification is technically sound but identification is fragile. Results should be interpreted as "suggestive evidence" rather than causal proof. Two critical gaps limit strong conclusions: (1) Selection bias cannot be ruled out; (2) Confidence intervals are not reported, hiding the precision problem.**

---

## QUICK ASSESSMENT

| Dimension | Rating | Key Finding |
|-----------|--------|-------------|
| **Technical Execution** | ✓ Good | De-meaned FE correctly implemented, clustering appropriate |
| **Sample Power** | ✓ Good | N=45,325 adequate to detect observed effects |
| **Causal Identification** | ⚠️ Weak | Selection bias unaddressed; no lead/lag tests |
| **Effect Precision** | ⚠️ Weak | Main effect (p=0.067) has CI including zero; others have wide bands |
| **Robustness Testing** | ✗ Poor | Only subgroup analysis; no alternative FE specs, sample tests, or functional form checks |
| **Transparency** | ✓ Good | Honest about limitations, honest about causality questions |

---

## THE GOOD

1. **Specification is correct**: Two-way FE with person + occupation-year demeaning properly captures within-person variation over time
2. **Clustering is appropriate**: Accounts for repeated obs within person
3. **Large, relevant sample**: 45,325 person-years with adequate power
4. **Clear heterogeneity**: Strong gender difference (males: -0.43**, females: -0.01) is striking and well-documented
5. **Transparent caveats**: Paper acknowledges selection bias, firm linkage issues, causality limits

---

## THE PROBLEMS

### 1. Selection Bias (Critical)
**The core threat**: Workers with pre-existing leftist views might sort into AI occupations, or firms might adopt AI in response to workforce composition.

**What specification controls**: Time-invariant person traits (via person FE) and macro occupation trends (via occ-year FE)

**What it doesn't control**: Time-varying selection, reverse causality, firm-level dynamics

**Test needed**: Lead/lag analysis — does year t+1 exposure predict year t politics?

### 2. Confidence Intervals Not Reported (Critical for Borderline Results)
Main finding: β = -0.252 (p = 0.067)
- 95% CI = [-0.521, +0.017]  
- **CI includes zero** — effect could be null
- No CI reported anywhere in summary

This is standard practice but **critical to mention when p≈0.067**.

### 3. Gender-Mediation Puzzle (Unsolved)
- Females: Strong insecurity response (+0.155, p=0.053) BUT zero political response (-0.012, p=0.958)
- Males: Weak insecurity response (+0.070, p=0.227) BUT strong political response (-0.430, p=0.013)

**This pattern contradicts simple mediation narrative.** Suggests gender moderates the insecurity→politics pathway, but this interaction is not formally modeled.

### 4. Limited Robustness Testing
Missing checks:
- Alternative FE structures (person-only, occ-year-only, first-differences)
- Sample variations (balanced panel, by firm size, by wage tercile)
- Functional form tests (polynomial terms, splines)
- Outcome alternatives (binary, winsorized)

---

## EFFECT SIZES: WHAT DO THEY MEAN?

| Outcome | Effect | SE | 95% CI | Interpretation |
|---------|--------|-----|--------|-----------------|
| **Left-Right** (main) | -0.252 | 0.137 | [-0.521, +0.017] | 0.25 point shift on 10-pt scale; 11.6% SD; imprecise |
| **Males (Left-Right)** | -0.430** | 0.173 | [-0.769, -0.091] | 0.43 point shift; statistically significant |
| **Females (Left-Right)** | -0.012 | 0.230 | [-0.463, +0.439] | No effect; very wide CI |
| **Nativism** | -0.207 | 0.162 | [-0.524, +0.110] | Null; wide CI masks uncertainty |
| **Welfare** | -0.084 | 0.177 | [-0.431, +0.263] | Null; wide CI |

**Key insight**: All non-male-leftright effects have CIs so wide they cannot rule out substantial effects in either direction.

---

## WHAT WOULD BREAK THIS?

**In order of likelihood**:
1. **Lead is significant** (year t+1 exposure predicts year t politics): Reverse causality or selection
2. **First-difference spec differs** (opposite sign): Specification fragility
3. **Balanced panel differs** (much smaller): Attrition bias
4. **Binary exposure gives null** (p > 0.10): Measurement noise or nonlinearity
5. **Spline terms break linearity** (β² significant): Functional form misspecification

---

## CRITICAL RECOMMENDATIONS

### MUST DO (To support causal claims)
1. **Report 95% CIs** for all main findings
2. **Run lead/lag analysis** to test reverse causality
3. **Test alternative FE structures** to assess robustness

### SHOULD DO (To strengthen argument)
4. Test functional form (polynomial, splines)
5. Investigate gender × insecurity interaction formally
6. Stratified analysis by occupation type (tech vs. routine)

### NICE TO DO (Enhancement)
7. Balanced panel check (survivorship bias)
8. Quantile regression (heterogeneity across distribution)
9. Robustness to outcome scaling (ordinal logit vs. OLS)

---

## CAUSAL INTERPRETATION: THE VERDICT

**Can we conclude: "AI exposure CAUSES leftward political shift"?**

**Current evidence**: ✗ Not yet  
**Why**: Selection bias and reverse causality cannot be ruled out  
**What's needed**: Lead/lag test + stable across specifications

**What we can say**: 
> "Workers in AI-exposed occupations are associated with 0.25-point leftward political shift (p=0.067), with substantial gender heterogeneity. Effect is partially mediated by job insecurity. However, selection bias and reverse causality cannot be ruled out. Lead/lag analysis and alternative specifications would strengthen causal claims."

---

## BOTTOM-LINE RECOMMENDATIONS

### For Publication
- Add confidence intervals to all tables
- Acknowledge CI includes zero for main finding
- Tone down causal language ("associated with" not "causes")
- Add lead/lag analysis as supplementary material

### For Future Research
- Investigate gender moderation (why females don't respond politically despite insecurity)
- Test alternative FE structures for robustness
- Link to objective employment outcomes (job loss, wages)
- Examine specific occupations and firm sizes

### For the Current Work
- This is **strong exploratory research** that identifies an important correlation and heterogeneity pattern
- Not yet sufficient for causal inference
- With additional tests (esp. lead/lag), could become causal evidence
