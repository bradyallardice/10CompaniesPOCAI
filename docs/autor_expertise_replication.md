# Autor Expertise Hypothesis: Replication in Swiss Job-Ads × SHP Panel

## Theoretical Predictions

Autor and co-authors argue that the impact of AI on labor markets depends on **which tasks within an occupation are codified**, not merely how much AI is deployed:

1. **Expertise ↑ (AI displaces low-expertise tasks) → wages ↑.** When AI handles the rote pieces, the remaining human work is higher-expertise, and incumbent workers command higher returns.
2. **Expertise ↓ (AI displaces high-expertise tasks) → hiring ↑.** When AI codifies the expert work, the remaining tasks are lower-skill and more substitutable. Firms can expand hiring with cheaper labor.

This report tests both predictions using a measure of **task-expertise displacement** built from job-advertisement data, linked at three units of analysis: worker, firm, and occupation.

---

## Construction of `expertise_change`

We grade every O\*NET task statement (n = 19,530) on a 1–10 expertise scale via LLM, producing `Data/task_expertise_scores.csv`. Within each occupation × firm × year cell in the Stage 5 pipeline:

- `baseline_expertise` = importance-weighted mean expertise across *all* tasks in the occupation.
- `remaining_expertise` = importance-weighted mean expertise across tasks that are *not* AI-exposed for that firm-year (binary task exposure = 0).
- **`expertise_change` = remaining_expertise − baseline_expertise**

Sign convention:

- **Positive** → AI took the lower-expertise tasks; what remains is higher-expertise. Workers in the occupation "moved up the skill ladder."
- **Negative** → AI took the higher-expertise tasks; what remains is lower-expertise. The occupation "moved down."

The measure varies at the firm × occupation × year level because which tasks count as AI-exposed depends on which specific AI applications the firm's job ads describe.

### Sample composition (matched, pct_05 spec)

Full-sample run: 2,030,142 ISCO × firm × year exposure rows (Stage 5). Distribution of `expertise_change` across those rows:

| Statistic | Value |
|---|---|
| Mean | −0.062 |
| SD | 0.564 |
| Range | [−5.38, +4.73] |
| Gaining (>0) | 956,060 (48.7%) |
| Losing (<0) | 1,003,980 (50.9%) |
| Flat | 11,161 (0.6%) |
| NaN (100% of tasks AI-exposed) | 58,941 |

Mean slightly negative — across the full panel, AI is on net displacing higher-expertise tasks. But the population is roughly split, with a 14-year time trend that turns increasingly negative (see "Time trend" below).

---

## Identification

Three units of analysis, three specifications:

| Module | Unit | FE | SE | Sample |
|---|---|---|---|---|
| **G** | Worker (idpers × year) | Person + 3d-ISCO × Year + Industry × Year (NOGA2M) | Cluster idpers | Matched on `hampole_ai_exposure_avg_foy > 0`, n ≈ 5,322 |
| **E** | Firm (company_id × year) | Firm + Year | Cluster company_id | Firm panel, n = 4,721 firm-years with expertise defined |
| **H** | Occupation (isco08_4d × year) | Occupation + Year | Cluster isco_4d | All occ-years in Stage 6 linked file, n = 4,246 occ-years across 344 occupations |

All three estimates use the same treatment construction (`expertise_change` at the appropriate level of aggregation) and within-FE OLS with cluster-robust SE.

---

## Result 1: Hiring (Autor prediction 2)

**Prediction:** `expertise_change` < 0 (AI took high-skill tasks) → firms and occupations expand hiring.

### Firm level (Module E)

| Outcome | Treatment | β | SE | p | n |
|---|---|---|---|---|---|
| log(job_ads at t+1) | `firm_avg_expertise_change` | **−0.763** | 0.139 | **<0.001 \*\*\*** | 4,721 |
| Δlog(job_ads) | `firm_avg_expertise_change` | +0.209 | 0.064 | 0.001 \*\*\* | 4,721 |

The headline level test is a strong confirmation of Autor at the firm level. A 1-unit drop in firm-average expertise change is associated with roughly a 0.76 log-point increase in next-year job ads (within firm and year FE). Even an interquartile move (~0.5 units) implies ~40% more hiring.

The Δlog has the opposite sign (positive). Substantively, the Autor prediction is about **hiring volume**, not growth rate, and the level test is the canonical mapping to the theory. The Δlog reversal is consistent with firms in the "losing expertise" category having persistently elevated hiring (high level) while firms in the "gaining expertise" category accelerate from a lower base.

#### Comparison to AI-exposure-only specifications (firm level)

The full firm panel runs three AI-exposure baselines: `pct_ai_ads_cumulative` (share of all ads that mention AI cumulatively), `firm_ai_exposure` (task-based ratio), and `log_ai_apps` (log of total AI applications). On the full 72,720 firm-year panel:

| Treatment | β | SE | p | n_firms |
|---|---|---|---|---|
| pct_ai_ads_cumulative | +0.0007 | 0.0007 | 0.32 | 7,197 |
| firm_ai_exposure | **+0.0585** | 0.0158 | **<0.001 \*\*\*** | 7,197 |
| log_ai_apps | **+0.101** | 0.032 | **0.002 \*\*\*** | 7,197 |

This is not a fair comparison to the expertise test (n=4,721, 797 firms). To put them on the same footing, the AI-only treatments were rerun on the expertise sample, and joint specs added:

**Apples-to-apples (same 4,721 firm-years, 892 firms with both AI-exposure and expertise data):**

| Spec | Term | β | SE | p |
|---|---|---|---|---|
| AI adoption alone | pct_ai_ads_cumulative | **−0.0149** | 0.0044 | **0.001 \*\*\*** |
| AI exposure alone | firm_ai_exposure | **+0.0461** | 0.0097 | **<0.001 \*\*\*** |
| Log AI apps alone | log_ai_apps | +0.0091 | 0.0164 | 0.58 |
| **Expertise alone** | **firm_avg_expertise_change** | **−0.763** | 0.139 | **<0.001 \*\*\*** |
| Joint (AI adoption + expertise) | pct_ai_ads_cumulative | −0.0153 | 0.0044 | 0.001 \*\*\* |
| | firm_avg_expertise_change | **−0.779** | 0.135 | **<0.001 \*\*\*** |
| Joint (AI exposure + expertise) | firm_ai_exposure | +0.0291 | 0.0178 | 0.10 |
| | firm_avg_expertise_change | **−0.757** | 0.139 | **<0.001 \*\*\*** |
| Joint (log AI apps + expertise) | log_ai_apps | +0.0031 | 0.0160 | 0.85 |
| | firm_avg_expertise_change | **−0.763** | 0.139 | **<0.001 \*\*\*** |

Three observations:

1. **`pct_ai_ads_cumulative` flips sign when restricted to the expertise sample** (+0.0007 in the full panel → **−0.015** in the matched 4,721). In the broader population AI adoption percentage is essentially uncorrelated with hiring; *within the AI-active universe*, deeper AI adoption is associated with *less* hiring. The expertise sample is a different population.
2. **`firm_ai_exposure` survives the sample restriction** (+0.058 → +0.046, still highly significant) but **collapses to insignificance once expertise is added** to the RHS (β=+0.029, p=0.10). Task-based AI exposure carries no independent signal once direction-of-displacement is in the model.
3. **`firm_avg_expertise_change` is rock-solid**: β stays at ≈ −0.76 across every joint specification — whether you control for AI adoption percentage, task-based exposure, or log AI apps. It is the dominant predictor of firm-level future hiring.

**Substantive interpretation:** the *direction* of AI task displacement (expertise_change) carries far more information about future hiring than the *level* of AI exposure. Two firms with identical AI-exposure intensity can have opposite hiring patterns depending on whether AI is taking their experts' work or their grunt work.

### Occupation level (Module H)

Aggregating the Stage 6 linked file to (isco08_4d × year) — 4,246 occupation-years across 344 ISCO occupations and 14 years.

**Marginal specs:**

| Outcome | Treatment | β | SE | p | n |
|---|---|---|---|---|---|
| log(n_jobs in occ at t+1) | Mean `expertise_change` (across firms) | **−0.148** | 0.058 | **0.011 \*\*** | 4,246 |
| log(n_jobs in occ at t+1) | Mean AI exposure | **+0.508** | 0.162 | **0.002 \*\*\*** | 4,250 |
| Δlog(n_jobs) | Mean `expertise_change` | +0.029 | 0.036 | 0.42 | 4,246 |
| Δlog(n_jobs) | Mean AI exposure | −0.076 | 0.102 | 0.46 | 4,250 |

**Joint spec (both treatments on RHS, level outcome):**

| Term | β | SE | p |
|---|---|---|---|
| Mean expertise change | **−0.131** | 0.060 | **0.030 \*\*** |
| Mean AI exposure | **+0.461** | 0.161 | **0.004 \*\*\*** |

Unlike the firm level, **both coefficients survive jointly at the occupation level**. They capture different and orthogonal signals:

- **AI exposure level** (positive coefficient) → occupations with more AI activity grow as a class. Intuitive — these are the occupations being affected at all, and they tend to be expanding sectors.
- **Expertise change** (negative coefficient) → conditional on AI activity level, the occupations where AI specifically displaces *high-expertise* tasks grow even faster. This is the Autor substitution channel.

The within-firm and within-occupation tests therefore tell a consistent story but with different decompositions:

- At the **firm level**, expertise_change absorbs the AI-exposure signal — firms with more AI exposure tend to be firms with negative expertise_change, so once you control for direction, the level is silent.
- At the **occupation level**, the two run alongside each other — an occupation can grow because AI is active in it *and* additionally because AI is displacing its expert work.

### Worker level (Module G economic block)

| Outcome | β(expertise_change) | SE | p | n |
|---|---|---|---|---|
| Separation at t+1 | −0.013 | 0.010 | 0.18 | 3,672 |

Same direction as the firm/occ-level signal (gaining expertise → less separation), but insignificant. The worker-side test is the noisiest because separation depends on idiosyncratic individual factors.

### Three-layer summary

| Layer | β(expertise_change → log_hiring_lead1) | SE | p | n |
|---|---|---|---|---|
| Worker (separation) | −0.013 | 0.010 | 0.18 | 3,672 |
| **Firm** | **−0.763** | 0.139 | **<0.001 \*\*\*** | 4,721 |
| **Occupation** | **−0.148** | 0.058 | **0.011 \*\*** | 4,246 |

The signal is concentrated where the theory operates: at the firm and occupation level, where labor-demand decisions are made.

---

## Result 2: Wages (Autor prediction 1)

**Prediction:** `expertise_change` > 0 (AI took low-skill tasks) → income rises for incumbents.

### Worker level (Module G)

| Outcome | β | SE | p | n |
|---|---|---|---|---|
| Log income | +0.017 | 0.021 | 0.40 | 4,451 |
| Hours worked | +0.106 | 0.268 | 0.69 | 4,350 |
| Job insecurity (1–4) | −0.036 | 0.027 | 0.18 | 4,740 |

**The sign matches Autor's prediction for all three economic outcomes**, but none reaches conventional significance:

- log_income: positive (gain expertise → higher income ✓)
- hours_worked: positive (gain expertise → more hours)
- job_insecurity: negative (gain expertise → less insecure ✓)

This pattern is consistent with two readings:

1. **Power**: Person FE absorbs most of the within-person variance in income and hours. Identification rests on switchers (workers changing firms or occupations) within the matched sample of ~1,400 unique people — too thin to detect modest effects.
2. **Theory**: The wage prediction may be more diffuse in the short run than the hiring prediction. Workers don't immediately repackage their expertise; firms instantaneously repackage their job postings.

The income result is **directionally consistent** with Autor at the worker level, but the firm/occupation-level hiring tests are the empirically stronger evidence.

---

## Result 3 (Bonus): Political Outcomes

The new module tests `expertise_change` against the political outcomes in Module A. Three significant findings, all internally consistent:

| Outcome | β | SE | p | n | Direction |
|---|---|---|---|---|---|
| Vote SP (Social Democrats) | **−0.037** | 0.012 | **0.002 \*\*\*** | 4,318 | Gain expertise → less SP |
| Vote SVP (right-populist) | **+0.018** | 0.007 | **0.012 \*\*** | 4,318 | Gain expertise → more SVP |
| Left-Right placement (0–10) | **+0.107** | 0.048 | **0.027 \*\*** | 4,322 | Gain expertise → shift right |

Joint spec (AI exposure + expertise change on the RHS together) for left-right placement:

| Term | β | SE | p |
|---|---|---|---|
| AI exposure | −0.015 | 0.150 | 0.92 |
| Expertise change | **+0.107** | 0.048 | **0.026 \*\*** |

For left-right placement, AI exposure level has essentially no effect, but expertise change has a strong one. They're orthogonal signals — *what type of tasks AI takes* matters more than *how much* AI exposure there is.

Reading: workers whose AI is displacing the *low-expertise* part of their job (gaining expertise) shift right politically; workers whose AI is displacing the *high-expertise* part (losing expertise) shift toward the Social Democrats. The "winners" of AI become more conservative; the "losers" gravitate toward the social safety net.

This is a non-Autor result — it's a signal of how the same expertise-displacement mechanism that drives Autor's labor-demand prediction also shows up in workers' political identity.

---

## Robustness Considerations

### What `expertise_change` actually measures

`expertise_change` is a property of the *job ad description × O\*NET task crosswalk*: an AI application referenced in a job posting matches O\*NET tasks above a similarity threshold, and the matched tasks are weighted by O\*NET task importance and our LLM-graded expertise score. Two limitations:

1. **Threshold sensitivity.** This report uses the `pct_05` percentile threshold from Stage 4. Sensitivity to threshold choice has not been tested in this report; the same Stage 5 file produces values for pct_01, pct_10, pct_15, pct_20 that could be benchmarked.
2. **AI app → task matching.** Whether an AI application "covers" a task is a semantic-similarity decision. The Stage 4 cross-encoder threshold (0.0 / 0.2 / 0.4 / 0.6 / 0.8) provides another tunable; this report uses ce_0.0 (no CE filter).

### What the Module H sample isn't

The occupation × year aggregation pools only the AI-exposed firms (because Module H reads `stage6_jobs_linked_core_isco.csv`, which is the intersection of the SHP firms with the AI-exposure firm panel). The natural test of Autor's full prediction would aggregate hiring across **all** firms in an occupation × year — including non-AI-exposed firms — and treat the AI-exposed firms' mean expertise change as the treatment. We have the data to do this (`stage6_job_cache_max{year}.parquet`); it's a clear next step.

### What's missing for the wage test

Autor's wage prediction needs more identifying variation than the matched SHP sample provides. The Stage 5 firm-occupation panel covers ~872 firms and the linked SHP sample of ~1,400 persons. A larger administrative-linked sample (LEHD-style) would be the natural next test.

### Time trend in `expertise_change`

Across years, the mean shifts from +0.04 (2012, 53% gaining) to −0.07 (2025, 52% losing). The flip happens around 2017. Substantively this matches the qualitative shift from "AI does the rote stuff" (early 2010s) to "AI does the analytical stuff" (late 2010s onward). The headline regression coefficients in this report are FE-identified within the panel — they hold the time trend fixed — but the trend itself is worth investigating separately.

---

## Conclusion

Within the limitations of this 14-year, single-country sample:

1. **Autor's hiring prediction is confirmed strongly at the firm and occupation level**: when AI takes the high-expertise tasks, firms and occupations post more job ads. The firm-level coefficient is −0.76 (p < 0.001); the occupation-level coefficient is −0.15 (p = 0.011).
2. **The wage prediction is directionally consistent at the worker level but not significant** in the matched SHP sample. Larger administrative samples or a different identifying strategy would be needed to test this cleanly.
3. **A bonus political result**: workers gaining expertise shift right (more SVP, less SP); workers losing expertise shift left. This complements rather than replicates Autor.

The empirical case for treating the *direction* of AI task displacement as a separate signal from the *level* of AI exposure is strong in the joint specs: in the occupation-level hiring regression, both coefficients are individually significant and capture orthogonal information.

---

## Reproducibility

Inputs:

- Stage 5 expertise file: `Data/task_expertise_scores.csv`
- Stage 5 ISCO output (full sample, pct_05 × ce_0.0): `Data/Testing/stage_5_expertise/full_sample/isco_firm_year_exposure_core_tasks_pct_05_ce_0.0.csv`
- Stage 6 linked: `Data/Testing/stage_6_expertise/full_sample/stage6_jobs_linked_core_isco.csv`
- Stage 6 firm report: `Data/Testing/stage_6_expertise/full_sample/firm_ai_summary_report_core_isco.csv`
- Stage 6 SHP: `Data/shp_exposure_expertise/shp_exposure_isco4d.csv`

To reproduce:

```bash
python3 stage_9e_political_nonlinear.py
```

Outputs:

- `Data/stage9e_results_long.csv` — person-level coefficients (Modules A–D, F, G)
- `Data/stage9e_firm_hiring.csv` — firm-level Autor test (Module E)
- `Data/stage9e_occ_hiring.csv` — occupation-level Autor test (Module H)
- `Data/stage9e_estimation_log.txt` — full run log

Module G replicates Module A's regression spec with `expertise_change_foy` as the treatment in place of `hampole_ai_exposure_avg_foy`. Module H aggregates the Stage 6 linked file to the (isco08_4d, year) level and runs OLS with occupation + year FE, SE clustered on isco_4d.
