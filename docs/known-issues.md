# Known Issues and Data Quality Notes

## Resolved Issues

### ✅ FIXED: Stage 0 - NaN in matched_keywords

**Status**: RESOLVED (December 2025)

**Original Problem**: ~351 rows (0.33%) in `Data/ai_development_deduplicated.csv` had NaN values in the `matched_keywords` column.

**Root Cause**: The batched extraction function `_execute_batched_keyword_search()` was missing the filtering logic that exists in the regular `_execute_keyword_search()` function. Specifically:
- SQL query returns rows matching keyword patterns (including false positives)
- `_find_matching_keywords()` is applied to each row, returning `""` (empty string) for rows with no actual keywords
- The filtering step that should remove rows with empty/NaN `matched_keywords` was missing in the batched version
- Result: ~351 SQL false positives slipped through with empty `matched_keywords`, later becoming NaN

**Solution Applied**:
1. **Code Fix**: Added filtering logic to `_execute_batched_keyword_search()` (lines 921-934 in stage_0_get_job_ads.py)
   ```python
   df_filtered = df[df['matched_keywords'].notna() & (df['matched_keywords'] != '')]
   return df_filtered
   ```
2. **Data Cleanup**: Removed all NaN rows from existing pipeline outputs (December 2025):
   - `ai_development_deduplicated_custom.csv`: Removed 300 NaN rows (56,942 → 56,642)
   - `final_output_1000_sample.csv`: Removed 47 NaN rows (15,592 → 15,545)

**Future Runs**: Stage 0 will now produce clean output with no NaN values in `matched_keywords`

---

## Current Data Quality Status

### Stage 0: AI Keyword Extraction
- **Status**: ✅ Clean (post-December 2025)
- **Output**: Zero NaN values in `matched_keywords` column
- **Method**: Batched keyword matching with SQL normalization
- **Coverage**: ~107K jobs from 14.7M database

### Stage 2: Deduplication
- **Status**: ✅ Clean
- **Output**: 56,642 unique jobs (53% reduction from Stage 0)
- **Methods**: Exact dedup, false positive filtering, TF-IDF similarity
- **Quality**: No duplicates across exact match and near-duplicate detection

### Stage 3: AI Application Extraction
- **Status**: ✅ Handles malformed responses
- **Coverage**: Separate malformed files with reprocessing support
- **Quality Validation**: F1 scores tracked (0.7817 → 0.8442 → 0.9194 across steps)
- **Note**: Malformed files automatically cleaned up on successful reprocessing

### Stage 4: O*NET Similarity
- **Status**: ✅ Checkpointing system active
- **Checkpoint Types**: Content-based keys for fault tolerance
- **Recent Changes**: Salvaged checkpoint system (Dec 2025) prevents data loss

### Stage 5: Firm Exposure Calculation
- **Status**: ⚠️ Known crosswalk gaps
- **Issue**: Some X28 → ISCO → O*NET mappings incomplete
- **Status**: Expected and acceptable - holes in crosswalk data, not code errors
- **Validation**: Always report unmatched records separately

### Stage 6: Job-Exposure Linking
- **Status**: ⚠️ Known unmatch diagnostics
- **Produces**: `stage6_unmatched_jobs.csv` and `stage6_unmatched_exposures.csv`
- **Handling**: Inner join on (company_id, year, occ_code)
- **Expected**: Some jobs/exposures won't match due to crosswalk gaps

### Stage 7: Analysis
- **Status**: ✅ Comprehensive outputs
- **Generates**: Summary statistics, visualizations, detailed breakdowns
- **Testing**: Regularly tested with `Data/Testing/stage_7/1000_company_test/`

---

## Known Data Quality Issues

### Crosswalk Mapping Gaps

**Scope**: Stage 5-6 transitions
**Issue**: Not all X28 occupations map to ISCO-08 4-digit codes perfectly
**Impact**: Some firm-occupation pairs may not match with exposure data
**Status**: Expected and acceptable - data hole, not code error
**Validation**: Check `stage6_unmatched_jobs.csv` and `stage6_unmatched_exposures.csv` for diagnostics

**Example**:
```
Company A has job title "Software Developer" (X28: 1234, 5678)
- 1234 maps to ISCO 2153 (Software developers) ✓
- 5678 does not map to any ISCO code ✗
Result: Company A linked to ISCO 2153 only
```

### Stage 6 SHP: Firm ID Missingness (~75% structural)

**Scope**: Stage 6 SHP exposure linkage
**Status**: Known limitation, mostly structural (not fixable)

**Key finding** (March 2026): Firm ID missingness is ~75% across ALL years, including 2012–2021 where the firm linkage file exists. The panel stopping at wave 23 (2021) is a minor contributor.

**Breakdown by source**:

| Source | Missing firm_id | % of total missing |
|--------|----------------|--------------------|
| 2012–2021 (structural) | 127,888 | 83.0% |
| 2022–2023 (fixable with updated linkage) | 26,111 | 17.0% |
| **Total** | **153,999** | |

**Year-by-year rates** (missingness is stable at ~74–80% every year):

| Year | Total rows | Has firm_id | Missing | % missing |
|------|-----------|-------------|---------|-----------|
| 2012 | 10,963 | 2,475 | 8,488 | 77.4% |
| 2013 | 20,447 | 2,568 | 17,879 | 87.4% |
| 2014 | 18,013 | 4,062 | 13,951 | 77.4% |
| 2015 | 16,340 | 3,954 | 12,386 | 75.8% |
| 2016 | 14,957 | 3,693 | 11,264 | 75.3% |
| 2017 | 13,946 | 3,541 | 10,405 | 74.6% |
| 2018 | 13,748 | 3,599 | 10,149 | 73.8% |
| 2019 | 13,150 | 3,540 | 9,610 | 73.1% |
| 2020 | 24,712 | 5,735 | 18,977 | 76.8% |
| 2021 | 19,968 | 5,189 | 14,779 | 74.0% |
| 2022 | 17,048 | 3,839 | 13,209 | 77.5% |
| 2023 | 16,032 | 3,130 | 12,902 | 80.5% |

**Root cause**: The `shp_firmid_anon.csv` linkage file only covers a subset of SHP respondents — likely those who consented to the employer linkage or who could be matched to administrative records. Most SHP respondents do not have a firm_id in any year.

**Impact on analysis**: The ~40k person-years with firm_id (after forward-fill) represent the usable sample for firm-level exposure measures (levels 1, 2, 5, 6). Occupation-only measures (levels 3, 4) are unaffected and cover ~75k+ person-years.

**Three layers of missingness** (March 2026 decomposition):

| Layer | What | Coverage | Fixable? |
|-------|------|----------|----------|
| 1. SHP linkage file | Only 47% of SHP persons appear in the file; only 26% ever get a firm_id | **This is the bottleneck** | No — likely small-firm / self-employed exclusions for anonymization |
| 2. Forward-fill | Extends firm_id to 22.7% of person-years | Modest help | Already implemented |
| 3. X28 overlap | 99.96% of SHP firm_ids are in the X28 database (only 3 missing) | Not a problem | N/A |

**Potential improvement from updating linkage to 2024**:

An updated `shp_firmid_anon.csv` covering 2022–2024 would help two groups:

| Group | Person-years gained | Source |
|-------|-------------------|--------|
| Previously-linked persons who changed employer or have gaps | ~4,114 | Direct matches for 2022–2023 |
| Never-linked persons (optimistic, ~50% match rate) | ~11,005 | New matches if anonymization permits |

| Scenario | Gain | New coverage |
|----------|------|-------------|
| Conservative (employer-changers only) | +4,114 | 24.8% (from 22.7%) |
| Optimistic (50% of never-linked also matched) | +15,119 | 30.3% |

The 12,412 never-linked persons in 2022–2023 are likely structurally excluded from the linkage (small firms, self-employed, anonymization restrictions) and may not gain firm_ids regardless of update.

**Decomposition of missing firm_id among employed respondents** (March 2026):

Of ~100,700 employed person-years (full-time + part-time), 45% have a firm_id and 55% do not. Among the 55,393 employed person-years missing a firm_id:

| Category | Person-years | % of employed missing |
|----------|-------------|----------------------|
| Employee (should have firm) | 38,680 | 69.8% |
| Professional status unknown | 13,721 | 24.8% |
| Self-employed / independent | 2,992 | 5.4% |

Self-employment is a negligible contributor to missingness (5.4% of employed missing). The bulk (~70%) are employees who plausibly should have a firm_id but don't — likely excluded due to anonymization restrictions on small firms.

### Time-Invariant vs. Time-Variant Exposures

**Scope**: Stage 5 exposure outputs
**Default**: Time-variant (yearly exposure)
**Alternative**: Time-invariant (exposure across all years a firm is active)
**Selection**: Use `--time_var` flag in Stage 6

**Consideration**: Affects how exposure is attributed to observations in later analysis

---

## Data Validation Checklist

Before proceeding to next stage:

- [ ] **No NaN values** in critical columns (check `df.isna().sum()`)
- [ ] **Expected row counts** within reasonable bounds (e.g., Stage 0→2 should be ~50-60% reduction)
- [ ] **Column schemas match** expected format (run `df.dtypes` check)
- [ ] **No unexpected duplicates** (check `df.duplicated().sum()` on key columns)
- [ ] **Mapping coverage reported** (check logs for % successfully mapped)
- [ ] **Malformed responses handled** (confirm no `_malformed.csv` files remain)

---

## Debugging Tips

### Stage 3 Malformed Responses
If you see `custom_step2_myfile_malformed.csv`:
1. Open and inspect the malformed rows
2. Check the content in the relevant column
3. Fix issues (if possible) or update prompt
4. Rerun same stage command - it auto-detects and reprocesses

### Stage 4+ Token/Checkpoint Issues
Check for salvaged checkpoints:
```bash
ls Data/embeddings/checkpoints/*.parquet
```
Salvaged system automatically handles recovery - monitor logs for confirmations.

### Stage 6 Unmatched Records
Always inspect diagnostics:
```python
# Jobs with no matching exposure
unmatched_jobs = pd.read_csv('Data/stage6_unmatched_jobs.csv')
print(f"Unmatched: {len(unmatched_jobs)} jobs")

# Exposures with no matching jobs
unmatched_exp = pd.read_csv('Data/stage6_unmatched_exposures.csv')
print(f"Unmatched: {len(unmatched_exp)} exposure records")
```

Likely causes:
- X28 code not in crosswalk
- ISCO code has no O*NET equivalent
- Year mismatch between job and exposure data
- Typos in occupation codes

---

## Recent Improvements

### September 2025
- **Batched Processing**: Memory-efficient extraction of large datasets
- **Database Optimization**: Trigram GIN indexing for faster searches
- **Flex Processing**: 50% cost reduction on supported OpenAI models
- **Comprehensive Evaluation**: Semantic similarity with BGE embeddings
- **Multi-Model Support**: O3, GPT-5-mini, Claude-3.5-Sonnet with reasoning controls
- **Automated Retries**: Robust error handling with exponential backoff

### December 2025
- **NaN Fix**: Removed 347 NaN rows from Stage 0 outputs
- **Checkpoint System**: Content-based keys for fault tolerance
- **Salvage System**: Automatic recovery of checkpoints on restart

---

## Reporting New Issues

When reporting data quality issues:
1. **Specify stage**: Which stage did the issue appear?
2. **Row count impact**: How many rows affected?
3. **Column name**: Which column(s) have the issue?
4. **Example data**: Show 1-2 problematic rows
5. **Expected vs actual**: What should it look like?

Example issue report:
```
Stage 5 - Missing AI exposure values
- Affected: 234 firm-occupation-year combinations
- Column: hampole_ai_exposure_avg
- Root cause: Company X has no AI applications in Stage 4
- Expected: Should link to 0 (no exposure) not missing value
```
