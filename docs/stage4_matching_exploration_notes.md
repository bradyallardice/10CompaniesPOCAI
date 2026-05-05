# Stage 4 Matching Exploration Notes

**Date**: March 2026
**Context**: Exploring alternatives to the current exhaustive BGE approach for matching AI applications to O*NET tasks

---

## Background

Stage 4 takes deduplicated AI applications from Stage 3 and matches them to O*NET occupational tasks. The match quality determines everything downstream (firm×occupation exposure scores in Stage 5+). The current production approach uses exhaustive BGE cosine similarity — compute all app×task pairs, keep the top 5th percentile.

---

## Approaches Tested (on 200-app sample, core tasks)

### 1. Exhaustive BGE (current production)
- Compute cosine similarity for all 200 × 12,942 = ~2.6M pairs
- Keep top 5% by global percentile = 132,097 pairs
- No cross-encoder
- **Time (full sample ~15K apps)**: ~1-2 hours

### 2. Exhaustive BGE + Cross-Encoder (CE)
- Same as above but run CE on all 132K retained pairs
- **Time (full sample)**: ~47 hours — not feasible on laptop
- Ran this on 200-app sample as a reference/ground truth

### 3. Exhaustive OpenAI + CE
- Use OpenAI text-embedding-3-large instead of BGE
- OpenAI scores are lower on average, so far fewer pairs survive the 0.3 minimum threshold (41K vs 132K)
- CE run on all 41K pairs
- **Time (full sample)**: ~9 hours — feasible overnight
- Already ran on full sample

### 4. RAG (dense + sparse retrieval) + CE
- For each app, retrieve top-k tasks by BGE (dense) + BM25 (sparse), merge, run CE on candidates only
- Grid search over dk (dense k) ∈ {30, 50, 100, 200} and sk (sparse k) ∈ {0, 30, 50, 100}
- Best config: **dk200_sk100** (506 CE≥0.5 matches, 107/200 apps matched)
- **Time (full sample)**: ~20 hours for dk200_sk100, ~7 hours for dk50_sk50

---

## Key Finding 1: BGE Cosine Similarity is a Poor Proxy for Match Quality

The BGE-CE correlation is **0.31**. This means cosine similarity is a weak predictor of whether the CE thinks a pair is a real match. Practically:

- Exhaustive BGE top 5% = 132,097 pairs
- Of those, only **684 (0.52%) have CE≥0.5** — the other 99.5% are noise by CE standards
- But many pairs with moderate BGE scores have high CE scores, which is why RAG (which retrieves by BGE rank) misses matches

---

## Key Finding 2: RAG Recall is Limited

Even the best RAG config (dk200_sk100) only recovers ~74% of the CE≥0.5 matches found by exhaustive BGE+CE. At CE≥0.7 recall improves to ~80%. Everything RAG finds is real (near-zero false positives in the RAG set), but it's missing 20-26% of valid matches because good pairs don't always rank in the top 200 by BGE cosine.

BM25 sparse retrieval helps — at every dk level, adding sk=100 improves recall meaningfully (e.g. dk50_sk0: 294 vs dk50_sk100: 377 at CE≥0.5).

**RAG grid search results (200-app test, CE≥0.5 recall vs exhaustive+CE ground truth):**

| Config | CE Pairs (full sample) | CE Time (full) | Recall@0.5 | Recall@0.7 |
|---|---|---|---|---|
| dk200_sk100 | 4.1M | ~20 hrs | 74% | 80% |
| dk100_sk100 | 2.8M | ~13 hrs | 65% | 72% |
| dk50_sk100 | 2.1M | ~10 hrs | 55% | 64% |
| dk50_sk50 | 1.4M | ~7 hrs | 51% | 58% |
| dk30_sk0 | 450K | ~2 hrs | 34% | 41% |

---

## Key Finding 3: BGE Without CE Has Better Face Validity Downstream

Despite 99.5% "noise" at the pair level, the BGE approach produces better exposure measures at the firm×occupation level. Comparison of Stage 7 outputs (1000-company test):

| | BGE (no CE) | OpenAI (with CE) |
|---|---|---|
| Mean firm exposure | 0.127 | 0.332 |
| Top occupations | Coding clerks, Archivists, Fruit/veg preservers | Meteorologists, Payroll clerks, Assemblers |
| Between-occ variance | **57%** | 31% |
| Within-occ variance | 43% | **69%** |
| Coverage | 409 occs × 138 firms | 383 occs × 137 firms |

**BGE rankings are more intuitive** — coding clerks and archivists having high AI exposure makes sense. Meteorologists ranking #1 under the OpenAI+CE approach doesn't.

**The variance decomposition is the most telling**: BGE gives 57% between-occupation variance (occupations drive exposure, methodologically sound for the Hampole approach). OpenAI+CE gives 69% within-occupation variance, suggesting the CE filtering creates firm-specific noise rather than occupational signal.

**Why does noisy BGE produce better aggregates?**
- BGE matches each app to ~660 tasks, giving broad occupation coverage
- CE-filtered approach matches only ~3-5 tasks per app → many firm×occupation cells get zero exposure → sparse, hard-to-interpret distribution
- The "noise" pairs have low but non-zero similarity scores that, when averaged across many tasks within an occupation, produce a meaningful gradient rather than near-binary exposure

---

## Key Finding 4: OpenAI Embeddings are a Better Pre-filter Than BGE

At the bi-encoder stage (before CE), OpenAI is more efficient:
- BGE top 5%: 132K pairs, 0.52% precision (CE≥0.5)
- OpenAI ≥0.3: 41K pairs, 1.45% precision (CE≥0.5)

OpenAI sends 3x fewer pairs to CE while keeping 88% of the good matches. At CE≥0.7 the overlap is nearly perfect (304/305 pairs). So if CE filtering were the goal, OpenAI embeddings would be the better first-stage filter. But as noted above, CE filtering hurts face validity at the aggregate level.

---

## Time Estimates (Full Sample, ~15K Unique Apps)

| Approach | CE Pairs | Time | Precision | Recall | Aggregate Face Validity |
|---|---|---|---|---|---|
| Exhaustive BGE (current) | 0 | ~1-2 hrs | 0.5% | 100% | ✓ Good |
| Exhaustive BGE + CE top 1% | 1.9M | ~9 hrs | ~100% | ~90% | Unknown |
| Exhaustive OpenAI + CE | 3.1M | ~9 hrs | ~100% | 88% | ✗ Poor |
| RAG dk200_sk100 + CE | 4.1M | ~20 hrs | ~100% | 74% | Unknown |
| Exhaustive BGE + CE top 5% | 9.7M | ~47 hrs | ~100% | 100% | Unknown |

---

## Human Evaluation Set Created

To validate whether CE is actually correct at the pair level (not just assuming it is), created a stratified 200-pair evaluation set:

**File**: `Data/manual_coding_evaluation/stage4_ce_validation.csv`

- 50 apps sampled across 5 strata (high CE, moderate CE, borderline CE, no CE match, BGE-CE disagreement)
- 4 pairs per app: top-1 CE, 2nd CE, top-1 BGE (disagreement case), 1 mid-range CE
- CE score distribution: 99 pairs CE<0.1, 25 in 0.1-0.3, 20 in 0.3-0.5, 56 with CE≥0.5
- Fill `human_label` (1=match, 0=no match) and `human_notes` columns

This will let us estimate CE precision/recall against human judgment, and understand whether the CE's rejections are genuine non-matches or real matches it's too strict about.

---

## Current Status and Next Steps

**Current production approach**: Exhaustive BGE, no CE — keeps all top-5% pairs, best face validity

**Open questions**:
1. Is the CE actually right at the pair level? → needs human evaluation set coded
2. Would a lenient CE threshold (CE≥0.2 or 0.3 instead of 0.5) filter worst noise while preserving enough coverage for good aggregates?
3. Does exhaustive BGE + CE top 1% (feasible at ~9 hrs) produce better or worse aggregates than no CE?

**Decision pending**: Stick with BGE no-CE for now, revisit after human evaluation set is coded.
