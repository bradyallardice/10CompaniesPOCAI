# CRITICAL REVIEW: Complete Index
**April 10, 2026 | AI Exposure and Worker Outcomes Analysis**

This directory contains a comprehensive critical review of your empirical strategy, prepared from the perspective of a skeptical devil's advocate challenging identification, measurement, and causal inference.

---

## DOCUMENTS IN THIS REVIEW

### 1. **REVIEWER_EXECUTIVE_SUMMARY.md** ← START HERE
**Reading time**: 10-15 minutes  
**For**: Everyone (Brady, collaborators, potential reviewers)  
**Contains**:
- Bottom-line verdict: MEDIUM credibility (60% for income, 50% for politics)
- Three biggest threats explained in plain language
- Decision tree for publication targets
- Week-by-week work plan to boost credibility
- What's genuinely interesting in your findings (past causal claims)

**Key section**: "What You Should Do in Priority Order" (Week 1 must-dos, Week 2 supporting tests)

---

### 2. **CRITICAL_REVIEW_MEMO.md** ← TECHNICAL DEEP DIVE
**Reading time**: 45-60 minutes  
**For**: Brady (detailed analysis), co-authors, technical reviewers  
**Contains**:
- Section-by-section attack on measurement, sample, identification
- Specific threat scenarios with quantified bias directions
- Evidence you have vs. don't have for each claim
- Detailed assessment of firm-occupation endogeneity problem
- Income reversal mystery analyzed in depth
- Rigorous stress-test recommendations with code

**Key sections**:
- Section 1: AI Exposure Measurement Problem (unvalidated, confounds marketing with work)
- Section 2: Sample Selection Problem (75% firm_id missingness; biased toward large firms)
- Section 3: Identification Problem (firm-occupation assignment endogenous; benign vs. malign stories)
- Section 5: Structural Threats (lead/lag, time-varying confounders, COVID confounding)
- Section 6: Stress Tests (prioritized by feasibility + impact)
- Section 7-9: Causal Credibility Scorecard

---

### 3. **STRESS_TEST_PRIORITY_MATRIX.md** ← ACTIONABLE ROADMAP
**Reading time**: 15-20 minutes  
**For**: Brady (implementation guide), research assistants  
**Contains**:
- Quick reference table of 9 stress tests (ranked by effort and impact)
- "What to do this week" vs. "what to do next week"
- Decision tree for your next steps (publication target → work plan)
- Specific regression specs you can copy-paste into your code
- Expected results and interpretation guide
- Pass/fail rubrics for each test

**Key sections**:
- Quick reference table (pick any test, see effort + expected payoff)
- "If test FAILS..." interpretation guide for each critical test
- Copy-paste Python code for lead-lag, gender interaction, task-fit specs
- Success criteria checklist

---

## HOW TO USE THESE DOCUMENTS

### If you have 15 minutes
1. Read REVIEWER_EXECUTIVE_SUMMARY.md (first 3 sections)
2. Skim the decision tree ("What's your publication target?")
3. Look up your chosen publication target's requirements

### If you have 1 hour
1. Read REVIEWER_EXECUTIVE_SUMMARY.md completely
2. Skim CRITICAL_REVIEW_MEMO.md Sections 1-3 (threats 1-3)
3. Use STRESS_TEST_PRIORITY_MATRIX.md to plan Week 1 work

### If you have 2-3 hours (ideal for serious revision)
1. Read all three documents in order
2. Focus on CRITICAL_REVIEW_MEMO.md Sections 4-7 (firm-occupation problem, structural threats)
3. Flag specific tests you'll run in STRESS_TEST_PRIORITY_MATRIX.md
4. Plan your 2-week work schedule

### If you're responding to journal reviewers
1. Use CRITICAL_REVIEW_MEMO.md Section 7-9 (credibility scorecard, what would improve it)
2. Note which tests you'll add to address reviewer concerns
3. Use STRESS_TEST_PRIORITY_MATRIX.md to estimate effort and timeline

---

## KEY STATISTICS AT A GLANCE

**Your findings**:
- Wage loss in AI firms: -8.5% (firm-year FE) vs. +10.9% wage gain (occupation-year FE)
- Political leftward shift: -0.252 points (overall), -0.430 (men), -0.012 (women)
- Gender wage gap: Women lose 17.6%, men lose 1.7% in AI firms
- Sample size: 45,325 person-years with firm linkage (22.7% of 199,324 total)

**Credibility assessment**:
- Overall: MEDIUM (60% confidence in income effects, 50% in political effects)
- Main threat: Unvalidated AI measurement + endogenous firm-occ assignment
- Time to high credibility: 2 weeks (critical tests) to 8 weeks (publication-ready)

**Publication readiness**:
- As-is: Working paper quality (good for SSRN, NBER)
- With Week 1 tests: Field journal quality (Labor Econ, Demography)
- With Week 1+2 tests: Top journal quality (AER conditional)
- With validation sample: Top journal quality (strong candidacy)

---

## THREAT RANKING (By Severity + Likelihood)

| Rank | Threat | Severity | Likelihood | Fixability | Time |
|------|--------|----------|-----------|-----------|------|
| 1 | AI measurement unvalidated | HIGH | HIGH | Partial | 1-2 weeks |
| 2 | Firm-occ endogeneity | HIGH | MEDIUM | Partial | 2-3 hours |
| 3 | Sample selection (large firms bias) | HIGH | CERTAIN | Difficult | 1 hour |
| 4 | Lead-lag/reverse causality | MEDIUM | MEDIUM | Easy | 30 min |
| 5 | Tight post-FE DoF | MEDIUM | CERTAIN | No | N/A |
| 6 | Person-year omitted interactions | MEDIUM | MEDIUM | Partial | 2 hours |
| 7 | COVID confounding | MEDIUM | MEDIUM | Easy | 20 min |
| 8 | Percentile threshold sensitivity | MEDIUM | MEDIUM | Moderate | 2-4 hours |

---

## TESTS BY PRIORITY LEVEL

### 🔴 CRITICAL (Do this week)
- [ ] Lead-lag specification (30 min) — gates causal inference
- [ ] Gender interaction test (20 min) — checks statistical significance of heterogeneity
- [ ] Task-fit regression (1 hr) — evidence for benign vs. malign story

### 🟠 HIGH (Do next week)
- [ ] Pre-treatment balance (30 min) — selection on unobservables
- [ ] COVID robustness (20 min) — pandemic confounding
- [ ] Occupational plausibility (30 min) — face validity

### 🟡 MEDIUM (Do if time permits)
- [ ] Percentile sensitivity (Stage 4 rerun, 2-4 hrs) — measurement robustness
- [ ] Sample representativeness (1 hr) — generalizability
- [ ] Alternative exposure measures (1-2 hrs) — specification sensitivity

### 🔵 LOW (Do if aiming for top journal)
- [ ] Manual validation sample (10-20 hrs) — gold standard measurement validation
- [ ] Cross-validation with external data (1-2 wks) — external validation

---

## THE CRITICAL QUESTIONS

After reading these documents, ask yourself:

1. **On measurement**: "Would I trust job ads to tell me if a worker actually uses AI?"
   - Answer honestly: if NO, you need validation
   - If YES, explain why to next reviewer

2. **On identification**: "Why does firm X deploy AI in occupation Y in year T?"
   - Can you rule out: firm is restructuring occupation Y?
   - Can you show: AI targets routine tasks (benign) not troubled occupations (malign)?
   - Need: Task-fit evidence to support benign story

3. **On causality**: "Are my results driven by contemporaneous shocks or unobserved trends?"
   - Test: Run lead-lag specification (must PASS for causal claim)
   - If FAILS: Reframe as correlational or correlational

4. **On sample**: "Do my results apply to small-firm workers and self-employed?"
   - Answer: Probably not (sample biased to large firms)
   - How to acknowledge: "Results apply to workers in firms with firm-linkage data; sample represents ~23% of employed workforce"

---

## QUICK REFERENCE: WHAT EACH DOCUMENT ANSWERS

| Question | Document | Section |
|----------|----------|---------|
| Is my analysis causal? | EXEC_SUMMARY | Bottom line; Credibility verdict |
| What's wrong with my measure? | CRITICAL_MEMO | Section 1 |
| What's wrong with my sample? | CRITICAL_MEMO | Section 2 |
| What's wrong with my identification? | CRITICAL_MEMO | Section 3 |
| What tests should I run? | STRESS_TEST_MATRIX | All sections |
| How do I run them? | STRESS_TEST_MATRIX | Code sections |
| What should I do Week 1? | EXEC_SUMMARY / MATRIX | "What to do this week" |
| Should I publish as-is? | EXEC_SUMMARY | "Publication target" decision tree |
| What if my test FAILS? | STRESS_TEST_MATRIX | "If test FAILS..." interpretations |
| Can I fix this? | EXEC_SUMMARY + CRITICAL_MEMO | "Can you fix it?" subsections |

---

## TIMELINE PROJECTIONS

### Conservative Path (Publication = 3-4 weeks)
1. Week 1: Run critical tests (lead-lag, gender interaction, task-fit)
2. Week 2: Run supporting tests (COVID, occupational plausibility, pre-balance)
3. Weeks 3-4: Write methods, limitations, appendix; submit to field journal

### Standard Path (Publication = 6-8 weeks)
1. Weeks 1-2: Run all Tier 1 tests + start Tier 2
2. Weeks 3-4: Run Tier 2 tests; gather task-fit + pre-balance evidence
3. Weeks 5-7: Write paper; internal reviews; revisions
4. Week 8: Submit to top field journal

### Ambitious Path (Publication = 10-12 weeks)
1. Weeks 1-2: Run all critical tests
2. Weeks 3-4: Collect task-fit + pre-balance + endogeneity evidence
3. Weeks 5-6: Begin manual validation sample (code ~500 jobs)
4. Weeks 7-8: Write paper + validation appendix
5. Weeks 9-10: Internal reviews + revisions
6. Weeks 11-12: Final polish; submit to AER/top 5 journal

---

## FINAL RECOMMENDATION

**Do this immediately (today)**:
1. Read REVIEWER_EXECUTIVE_SUMMARY.md
2. Decide: Are you aiming for working paper, field journal, or top journal?
3. Use decision tree to set realistic timeline

**Do this week**:
1. Run the three critical tests (3-4 hours total)
2. If all PASS: Proceed to Tier 2 or publication
3. If any FAIL: Stop and investigate root cause

**Do next week** (if critical tests pass):
1. Run supporting tests
2. Gather endogeneity evidence (task-fit + pre-balance)
3. Start writing methods + limitations

**Do in 2-3 weeks**:
1. Complete robustness appendix
2. Honest limitations section
3. Ready for journal submission (field journal) or query (top journal)

---

## CONTACT & QUESTIONS

These documents represent a systematic, sympathetic-but-skeptical review of your work. Each threat is real and deserves attention, but none are fatal. Your findings are interesting and likely real. The question is whether they're **causal** and how confident reviewers should be.

The path forward is clear: invest 1-2 weeks in critical tests, learn what you find, and adjust confidence claims accordingly. This is what careful research looks like.

---

**Review prepared**: April 10, 2026  
**Status**: COMPLETE  
**Confidence in this assessment**: HIGH  
**Recommended first action**: Read REVIEWER_EXECUTIVE_SUMMARY.md, then run critical tests outlined in STRESS_TEST_PRIORITY_MATRIX.md
