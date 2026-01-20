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
