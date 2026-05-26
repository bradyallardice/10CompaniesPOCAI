# SHP Firm-ID Linkage Update: Extending to 2025

**Goal**: Refresh `shp_firmid_anon.csv` (currently covers 2011–2021) to cover 2011–2025 using the new MIS Trend employer file delivered by Thomas Kurer on 2026-05-13.

**Status as of 2026-05-14**: Our matching work is done. Thomas needs two manual steps + one re-run to ship the final linkage.

---

## Background

The R pipeline that produces `shp_firmid_anon.csv` (Thomas's [`0_anonymous_firmid_march9.R`](../../kurer_allardice_technology/code/0_anonymous_firmid_march9.R)) takes two inputs:

1. **MIS Trend employer file** — keyed by `(idpers, year, employer_name, employer_location)`. Thomas has the un-anonymized version with `idpers`; we have only the anonymized version.
2. **Firm-name → X28-company_id matching table** ([`matched_names_final_reviewed.csv`](../Data/shp_matching/matched_names_final_reviewed.csv)) — our deliverable. Maps SHP-reported employer names to X28 firm IDs.

The previous matching round (Mar 9) was built against the old MIS Trend file (W13–W23, covering 2011–2021). On 2026-05-13 Thomas dropped a new file (W13–W27, covering 2011–2025) at:

```
/Users/<user>/Dropbox/kurer_allardice_technology/data/original/mis_trend/
  SHP_W13_to_W27_Bur_Prof-20260508_anon.csv
```

This file adds 26,515 new firm-year records across W24–W27 (2022–2025), with 9,725 unique (firm_name, location) strings — 3,809 of which never appeared in 2011–2021.

---

## What we did (2026-05-14)

### Step 1: Subset the new firms

Extracted the 3,809 firm-string pairs first seen in 2022–2025 that weren't in our existing matched file:

```bash
# Output: Data/shp_matching/w24_w27_run/new_firms_w24_w27.csv (3,809 rows)
```

### Step 2: Run the matching pipeline

```bash
python3 archive/deprecated/improve_shp_firm_matching.py match-names \
  --lookup-file       Data/shp_matching/enhanced_lookup.csv \
  --manual-fixes-file Data/shp_matching/manual_fixes.csv \
  --names-file        Data/shp_matching/w24_w27_run/new_firms_w24_w27.csv \
  --output-dir        Data/shp_matching/w24_w27_run
```

Six-pass matching (manual_fix → exact_norm → token_exact → token_relaxed → fuzzy_char) produced:

| Output | Rows |
|---|---|
| `matched_names_raw.csv` (auto-matched) | 623 |
| `review_needed.csv` (borderline, needs LLM review) | 365 |
| `unmatched_names.csv` (no candidate) | 3,184 |

### Step 3: LLM review of borderline matches

```bash
python3 archive/deprecated/improve_shp_firm_matching.py llm-review \
  --review-file       Data/shp_matching/w24_w27_run/review_needed.csv \
  --output-dir        Data/shp_matching/w24_w27_run \
  --model gpt-5-mini --batch-size 20
```

Cost: $0.06. Result: 230 YES, 112 NO, 23 UNCERTAIN.

### Step 4: Compile final W24–W27 matches

```bash
python3 archive/deprecated/improve_shp_firm_matching.py compile \
  --output-dir        Data/shp_matching/w24_w27_run
```

Produced `matched_names_final.csv`: 511 new W24–W27 matches.

### Step 5: User review of LLM verdicts

I (Brady) reviewed all 365 LLM verdicts in `llm_review_for_user.csv`. Convention used:

- `Y` = accept as match (override LLM if needed)
- `N` = reject as non-match
- `U` = uncertain → punt to Thomas
- blank = agree with LLM (no override)

**Result**:
- Confirmed 213 LLM YES with no override
- **Overrode 10 LLM NO verdicts to YES** — all holding-vs-operating-company splits (Home Instead Schweiz / Holding, EMS-CHEMIE / Holding, Novartis Pharma Services / Schweiz, Haag-Streit Holding / AG, Rhenus Logistics / Port Logistics, Ringier Axel Springer Polska / Schweiz, Credit Suisse Services / AG, UBS Fund Management / AG, Weibel + Sommer Elektro / Telecom, Kuoni Mueller / Müller + Partner)
- Decided 17 LLM UNCERTAIN cases (7 Y, 10 N)
- **Punted 45 cases to Thomas** (`needs_review=True` in final file)

### Step 6: Recover W13–W23 misses with expanded manual_fixes

Audited the 13,097 unmatched firms from the Mar 9 run. About 40% had "should have matched" markers (major-employer keywords, division/branch markers, or token-overlap with matched firms).

Added **134 new manual_fixes** for verified parent firms:

| Parent | Variants recovered |
|---|---|
| Raiffeisen Gruppe | 40 |
| Université de Genève, Universität Basel, Université de Fribourg | 13 each |
| AXA Versicherungen AG | 12 |
| Universität Bern | 10 |
| Universität Zürich | 9 |
| Schweizerische Mobiliar | 8 |
| Université de Neuchâtel | 6 |
| HUG | 4 |
| (smaller: Univ-Spital Basel, Insel Gruppe, Stadt Winterthur, Gemeinde Köniz, RUAG, Swisscom, Helvetia, Post CH AG) | 1–2 each |

Re-ran `match-names` on the W13–W23 unmatched pool → recovered 140 firm-name variants.

**Critical bug avoided**: I initially had wrong `company_id` values for CHUV (pointed to Jansen AG, a steel firm), EPFL, and Universität Luzern. Verification against `enhanced_lookup.csv` caught this — CHUV/EPFL/Univ Luzern are genuinely **not in X28** and were reclassified as `known_major_parent_missing`.

### Step 7: Update master files

- `manual_fixes.csv`: 505 → 639 rows (+134); backup at `manual_fixes_pre_w24w27.csv`
- `matched_names_combined_for_thomas.csv`: 11,409 rows (combines existing_mar9 + W24–W27 + user overrides + W13–W23 recovery)
- `x28_update_candidates_for_thomas.csv`: 3,361 firms in 3 buckets

---

## Deliverables (in `Data/shp_matching/w24_w27_run/`)

| File | Rows | Purpose |
|---|---|---|
| `matched_names_combined_for_thomas.csv` | 11,409 | Ship to Thomas — input to his R script |
| `x28_update_candidates_for_thomas.csv` | 3,361 | Ship to Thomas → X28 (after pw85 filter) |
| `llm_review_for_user.csv` | 365 | Brady's LLM-verdict review (completed) |
| `matched_names_final.csv` | 511 | Just the new W24–W27 matches |
| `new_manual_fixes_proposed.csv` | 134 | The manual_fixes we added |
| `w13_w23_rematch/matched_names_raw.csv` | 140 | W13–W23 recoveries |
| `unmatched_names.csv` | 3,184 | W24–W27 no-candidate (X28 update list source) |
| `llm_{confirmed,rejected,uncertain}.csv` | 230 / 112 / 23 | LLM verdicts |

Also updated:
- `Data/shp_matching/manual_fixes.csv` (505 → 639 rows)

---

## What still needs to happen

### Thomas: 3 tasks

**1. Review 45 new flagged matches** in `matched_names_combined_for_thomas.csv`

Filter to `needs_review==True` (only 45 rows — the 334 stale Mar 9 flags have been cleared since `review_match=1` was already filled).

Composition:
- 17 cases where LLM said YES, Brady was uncertain (mostly category-rule questions like "Should all `Die Schweizerische Post AG` variants fold to Post CH AG?")
- 22 cases where LLM said NO, Brady was uncertain (mostly sibling Pflegeheim / Kirchgemeinde questions)
- 6 cases where LLM said UNCERTAIN, Brady was still uncertain

For each, fill in `review_match` with `1` (accept) or `0` (reject).

**2. Do the `pw85` size join for `x28_update_candidates_for_thomas.csv`**

This file has 3,361 firms across 3 categories:

| `x28_status` | Count | Action |
|---|---|---|
| `known_major_parent_missing` | 75 | Forward to X28 regardless of size (CHUV, EPFL, Univ Luzern — flagship institutions) |
| `llm_rejected_candidate` | 112 | Subject to pw85 filter |
| `no_candidate_found` | 3,174 | Subject to pw85 filter |

For each row, look up the SHP respondent(s) who reported that (firm_name, firm_location) in the un-anonymized MIS Trend file → pull `pw85` from the SHP person panel → take the modal value across observations → add a column `firm_size_pw85` to the file.

Then filter: forward to X28 only rows where `firm_size_pw85 ≥ 4` (= 20+ employees) OR `x28_status == known_major_parent_missing`.

Brady can't do this on our side because `idpers` was stripped from the anonymized MIS Trend file we have access to.

**3. Re-run `0_anonymous_firmid_march9.R`** with:
- New MIS Trend: `SHP_W13_to_W27_Bur_Prof-20260508_anon.csv`
- Updated matches: `matched_names_combined_for_thomas.csv` (after step 1 above)

Output: refreshed `shp_firmid_anon.csv` covering 2011–2025.

### After Thomas: Pipeline downstream

Once the new `shp_firmid_anon.csv` lands in `data/created/`:

1. Re-run [`stage_6_shp_exposure.py`](../stage_6_shp_exposure.py) to refresh the SHP-exposure linkage at 4d/3d/2d ISCO levels
2. Verify coverage gains vs. the Mar 9 baseline (expected: ~5–10k additional person-years with `firm_id`)
3. Re-run downstream stages (Stage 8/9 econometric specs) on the extended panel

**Status as of 2026-05-26**: Thomas re-ran the R script — `shp_firmid_anon.csv` now has 99,989 rows covering 2011–2025 (vs. 73,474 rows / 2011–2021 before), with 47,634 rows carrying a firm_id (vs. 35,887 before). Plus, the v11 SHP release dropped from FORS — `swissubase_932_11_0/` replaces the previous v10 folder and extends the SHP person panel through Wave 26 (2024). 2025 (Wave 27) won't be in the panel until SHP fields it (~mid-2027 beta).

### Open follow-up: residual W13–W23 unmatched

After this run, 12,957 W13–W23 firms remain unmatched. The user's hypothesis (most are <20-employee firms not worth asking X28 for) needs the same `pw85` filter as Step 2 above to verify. Deferred to a later round — current focus is on the new W24–W27 era.

---

## Key decisions logged

- **Holding/operating company splits should fold to the parent** (Brady's 10 overrides of LLM NO verdicts). Examples: `EMS-CHEMIE HOLDING AG` → `EMS-CHEMIE AG`, `Haag-Streit Holding AG` → `Haag-Streit AG`.
- **Sub-20-employee firms not worth asking X28 for** (predates this session, surfaced today). Implemented via `pw85 ≥ 4` filter (Thomas's task).
- **Major flagship institutions missing from X28** (CHUV, EPFL, Universität Luzern) should be forwarded to X28 regardless of size — these have hundreds of SHP respondents.
- **`needs_review=True` flag now refers only to this run's punts** — the 334 stale Mar 9 flags were cleared because `review_match=1` was already filled.

---

## Files referenced

- New MIS Trend input: [`SHP_W13_to_W27_Bur_Prof-20260508_anon.csv`](../../kurer_allardice_technology/data/original/mis_trend/SHP_W13_to_W27_Bur_Prof-20260508_anon.csv)
- Matching pipeline: [`archive/deprecated/improve_shp_firm_matching.py`](../archive/deprecated/improve_shp_firm_matching.py)
- Thomas's R script: [`code/0_anonymous_firmid_march9.R`](../../kurer_allardice_technology/code/0_anonymous_firmid_march9.R) (in his Dropbox)
- Stage 6 SHP linkage: [`stage_6_shp_exposure.py`](../stage_6_shp_exposure.py)
- Coverage funnel diagnostic: [`docs/shp-matching-funnel.md`](shp-matching-funnel.md)
