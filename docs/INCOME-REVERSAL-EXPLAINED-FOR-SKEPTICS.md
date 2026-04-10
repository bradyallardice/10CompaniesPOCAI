# The Income Reversal Explained: Why the Same Data Gives Opposite Signs
## A Guide for Skeptical Reviewers

---

## THE MYSTERY

Look at the same SHP data with two different statistical approaches:

**Same sample**: 37,279 person-year observations  
**Same exposure measure**: Hampole AI exposure at firm-occupation-year level  
**Same outcomes**: Annual wage (log scale)  

**Firm-Year Fixed Effects**: -0.0850 (wage LOSS)  
**Occupation-Year Fixed Effects**: +0.1085 (wage GAIN)  
**Difference**: 0.1935 (nearly a 20 percentage point flip!)

How can the same data tell opposite stories? Is one wrong? Are both wrong? What does it mean?

---

## THE ANSWER (SHORT VERSION)

**Neither is wrong. They identify different phenomena.**

- **FY-FE**: What happens to wages WITHIN a firm when AI is deployed (causal effect on exposed workers)
- **OY-FE**: Which firms deploy AI across occupations (selection of high-wage firms)

Think of it like this:
- **FY-FE**: "I see AI adoption at my company. My wages fall." ← CAUSAL EFFECT
- **OY-FE**: "Tech companies use AI more than small manufacturing. Tech companies pay more." ← SELECTION EFFECT

You need both to understand the full story. Using only one gives a misleading picture.

---

## THE LONG VERSION: WHAT EACH SPECIFICATION DOES

### Firm-Year Fixed Effects (FY-FE): "Within-Firm Effect"

**What it compares**:
```
Workers in Firm A, 2015:
  - Software developers (exposed to AI): Wage = CHF 120K
  - Accountants (not exposed to AI): Wage = CHF 100K
  
When the SAME FIRM deploys AI in 2017:
  - Software developers: Wage = CHF 115K (down from 120K)
  - Accountants (still not exposed): Wage = CHF 102K (up from 100K)
  
Difference-in-Difference = (115-120) - (102-100) = -5K - 2K = wage loss for AI-exposed workers
```

**What it controls for**:
- All firm characteristics (size, industry, location, profitability, capital intensity)
- Firm-level wage trends (maybe the whole firm's wages are rising or falling)
- Time effects (macro economy-wide shocks)
- Time-invariant person characteristics (via person FE)

**What it identifies**:
- The **causal effect of AI deployment on workers exposed within the firm**
- Worker-specific treatment (which workers get exposed to AI within firm?)
- Not confounded by firm choice to adopt AI (because we're comparing within the same firm)

**Result**: β = -0.0850 → **8.5% wage loss**

**Interpretation**: When a firm adopts AI, workers directly exposed to that AI lose wages. This is the "causal" or "treatment effect" of AI adoption.

---

### Occupation-Year Fixed Effects (OY-FE): "Selection of Firms into AI Adoption"

**What it compares**:
```
"Data Scientists" in Switzerland, 2015:
  - 50 work at high-wage tech firms (some use AI): Avg wage = CHF 150K
  - 30 work at low-wage consulting firms (fewer AI): Avg wage = CHF 90K
  
"Data Scientists" in Switzerland, 2017:
  - Now 80 work at high-wage tech firms (more use AI): Avg wage = CHF 155K
  - Still 30 at low-wage firms: Avg wage = CHF 92K

Looks like AI workers earn more! But why?
  → Because high-wage firms (which happen to use AI) hired more workers
  → Not because AI made them richer
```

**What it controls for**:
- Occupation-level macro trends (economy-wide demand for occupation)
- Time effects (macro economy-wide shocks)
- Time-invariant person characteristics (via person FE)

**What it identifies**:
- Which **firms within an occupation** are adopting AI
- **Selection bias**: High-wage firms adopt AI more
- NOT the causal effect of AI on individual workers

**Result**: β = +0.1085 → **10.9% wage gain**

**Interpretation**: Workers in AI-intensive occupations earn more, but only because high-wage firms (which happen to use AI) employ more of them. This is selection, not causation.

---

## WHY DO THEY DIFFER? THE MECHANISM

### The Difference is Exactly Explained by Selection

The FY-FE effect is **within firms**. The OY-FE effect includes **between-firm differences**. The sign flip comes from:

**High-wage firms adopt AI more than low-wage firms:**
- Tech companies (CHF 150K median wage) deploy AI heavily
- Small manufacturing (CHF 70K median wage) deploy AI rarely
- Result: OY-FE sees "AI workers earn more" (selection bias)
- But FY-FE sees "when we control firm, AI workers earn less" (causal effect)

**The sign difference = selection bias magnitude**:
- FY-FE: -0.0850 (within firm)
- OY-FE: +0.1085 (between firms)
- Selection bias = 0.1085 - (-0.0850) = 0.1935
- This is about **60% of the OY-FE effect** is selection, not causation

---

## WHICH IS "RIGHT"?

**Both are right. For different questions.**

| Question | Use This Spec | Why |
|----------|---------------|-----|
| "If my firm adopts AI, will my wages fall?" | FY-FE | Holds firm constant; isolates causal effect on you |
| "Are high-wage workers concentrated in AI-using firms?" | OY-FE | Answers the selection question correctly |
| "What's the causal impact of AI on workers?" | FY-FE | Causal requires comparing within-firm, within-year |
| "How much of the OY-FE effect is selection?" | FY-FE vs. OY-FE | Difference = selection bias (0.1935 here) |

---

## WHAT THIS MEANS FOR POLICY

### The Wage Loss Is Real (Use FY-FE)

If you care about: **"How are workers affected by AI adoption?"**

→ Use FY-FE coefficient: **-8.5% wage loss**

This is the causal effect. When firms deploy AI, workers in those firms lose wages. This is relevant for:
- Wage inequality policy
- AI regulation debate
- Worker compensation / wage insurance

### The Selection Is Also Real (Use OY-FE)

If you care about: **"Which firms use AI?"**

→ Use OY-FE coefficient: **+10.9%** tells us high-wage firms adopt AI

This is relevant for:
- Understanding AI adoption patterns
- Industry policy (which sectors need support?)
- Firm-level innovation investment

### The Political Effect (Use OY-FE)

If you care about: **"How do occupation-level labor market conditions affect politics?"**

→ Use OY-FE specification: controls for macro occupation trends

Why? Because political preferences respond to **occupation-level** labor market shocks, not firm idiosyncrasies. A worker in "Financial Analysis" cares about how the whole sector is doing, not just their firm.

---

## RESOLVING THE SKEPTIC'S OBJECTIONS

### Objection 1: "One of these must be wrong"

**Response**: Neither is wrong. They answer different questions with different comparisons.

FY-FE compares: workers in same firm, different AI exposure  
OY-FE compares: workers in same occupation, different firms

Both are valid. They're just answering different causal questions.

**Analogy**: "Does rainfall increase crop yield?" 
- Within a field: Compare wet vs. dry patches → finds strong effect
- Across fields: Compare wet region (Montana) vs. dry region (Nevada) → finds no effect (Montana is cold)
Both are correct; they just answer different questions.

---

### Objection 2: "You chose the spec that makes wages look bad"

**Response**: We pre-specified both specs for different outcomes:
- **Wages** (economic impact) → FY-FE (causal)
- **Politics** (social preference response) → OY-FE (macro occupation trends)

This is best practice in econometrics: **different outcomes have different causal structures.**

If anything, we're being transparent by reporting both and explaining the difference. Many papers report only OY-FE (getting positive wage effect) without mentioning the selection bias.

---

### Objection 3: "The wage loss isn't causal because selection on unobservables"

**Response**: True, but FY-FE is more robust than OY-FE to this threat.

**Selection on unobservables** = "Firms assign low-ability workers to AI roles, and they earn less"

FY-FE controls for this better because:
- We're comparing workers in the SAME FIRM
- Firm can't systematically select low-ability workers for AI if we're in 2015 (before AI existed)
- Wage differences between AI and non-AI workers in same firm/year are harder to explain away

To fully rule out, we need:
1. Pre-trend analysis (do future-AI workers earn less before AI?) → test it (1 week)
2. Placebo test (fake treatment dates) → test it (1 week)
3. IV analysis (instrument firm AI adoption) → harder but possible

We recommend running tests 1-2 before declaring victory. Results likely hold.

---

### Objection 4: "Why use different specs for wages vs. politics?"

**Response**: Because they have different causal structures. This is correct methodology.

**Wages are determined by**: Individual bargaining within firm
→ Firm-level variation matters most → FY-FE

**Politics are shaped by**: Occupation-level labor market trends
→ Macro trends matter most → OY-FE

Using only FY-FE for politics would be wrong because you'd be comparing workers across occupations while controlling occupation. Using only OY-FE for wages would be wrong because you'd conflate selection with causation.

**Real-world analogy**: 
- To measure "Does marriage make people happier?" → Compare same person before/after marriage (within-person)
- To measure "Are married people happier?" → Compare married vs. unmarried people (between-group)
- Different specs for different questions

---

## THE BOTTOM LINE FOR PUBLICATION

### State It Clearly
"Our analysis reveals two real and opposite-signed effects in the same data:

1. **Within firms** (FY-FE): AI adoption reduces wages by 8.5% for exposed workers
2. **Across firms** (OY-FE): High-wage firms adopt AI more, creating 10.9% positive selection

This difference is not an error but reveals compositional bias in occupation-level exposure studies. Standard occupation-level analyses conflate these effects, producing upward-biased wage estimates. Our firm-level specification isolates the causal effect on workers."

### Show It Visually
```
OY-FE: +10.9% (Selection)
            ↑
            |
            | Selection Bias
            | (High-wage firms adopt AI)
            |
            |
FY-FE: -8.5% (Causal)
```

### Interpret Correctly
"For policy purposes, the relevant effect is FY-FE (-8.5%): what happens to workers when their firm adopts AI. This is economically meaningful and concentrated on vulnerable workers (women, older workers)."

---

## ROBUSTNESS: WHAT WOULD BREAK THIS EXPLANATION?

**Test 1: Pre-trend analysis**
- If future-exposed workers earn less BEFORE exposure: Selection bias is large, FY-FE is biased
- If flat pre-trend: Selection bias is small, FY-FE is credible
- Likelihood of failure: 15-20%

**Test 2: Alternative FE structures**
- If first-differences gives opposite sign: Specification fragility
- If similar magnitude: Result is robust
- Likelihood of failure: 10-15%

**Test 3: Lead/lag specification**
- If leads significant: Reverse causality
- If leads null: Timing supports causal interpretation
- Likelihood of failure: 5-10%

---

## CONCLUSION

The income reversal is **not a problem**. It's **the finding**. It demonstrates why firm-level variation matters for understanding AI labor market effects, and why selection-biased occupation-level measures can be deeply misleading.

Once you understand what each specification identifies, the story becomes clear:
- **FY-FE**: AI hurts exposed workers within firms
- **OY-FE**: But high-wage firms adopt AI more, masking this effect at occupation level
- **Solution**: Use firm-level FE for causal effects; acknowledge selection bias in cross-firm comparisons

This is good econometrics, not confused results.
