# Investigation: Stage 5-7 Task-App Matches Mismatch

## Problem Statement
Stage 7 is showing incorrect AI application matches for tasks. Task IDs and task text are correct, but the matched AI applications are semantically nonsensical (e.g., Task 7026 "Check equipment..." matched to "recommend personalized learning activities...").

## Root Cause Analysis

### Issue 1: `task_app_matches.csv` Gets Overwritten Multiple Times
**Location**: `stage_5_onet_to_isco_exposure.py`, line 806

```python
def step2_calculate_firm_task_exposure(self, task_app_matches: pd.DataFrame, ...):
    ...
    task_app_matches.to_csv(os.path.join(self.output_dir, 'task_app_matches.csv'), index=False)
```

**Problem**:
- Stage 5's `run_full_pipeline()` processes multiple **specifications** (different BGE percentile × CE threshold combinations)
- Each specification calls `step2_calculate_firm_task_exposure()` with filtered `spec_matches`
- Each call **overwrites** the same `task_app_matches.csv` file
- Only the **last specification processed** remains in the file
- Stage 7 then loads this single-spec file, which doesn't match what was used for other specifications

**Data Flow**:
```
Stage 4 output: task_exposure_matches_all_thresholds_*.parquet
    ↓
Stage 5 loads (line 1989): task_app_matches = full 290M+ rows with all thresholds
    ↓
Loop through specifications: for each (BGE_pct, CE_threshold):
    ├─ Filter for this spec: spec_matches = task_app_matches[task_app_matches[bge_col] & task_app_matches[ce_col]]
    ├─ Call step2 with spec_matches (line 2125-2126)
    │   └─ Saves spec_matches → task_app_matches.csv (line 806) ← OVERWRITES PREVIOUS
    └─ Next iteration...

Final state: task_app_matches.csv contains ONLY last specification's matches
```

### Issue 2: Stage 7 Expects Full Matches, Gets Partial
**Location**: `stage_7_analyze_results.py`, line 1749-1750

```python
matches_file = f"{self.stage5_dir}/task_app_matches.csv"
task_app_matches = pd.read_csv(matches_file)
```

**Problem**:
- Stage 7 loads `task_app_matches.csv` and expects it to have matches for **ALL** task-app combinations
- But it only has matches from the **last specification** that was processed in Stage 5
- When Stage 7 processes a different specification, it tries to find matches in the incomplete file
- Result: Many tasks have "no matches" even though they should have matches for the current spec

**Example Flow**:
```
Specification being processed in Stage 7: pct_20 × ce_0.6

task_app_matches.csv only has: pct_01 × ce_0.0 matches (last spec from Stage 5)

For each task:
    exposed_matches = task_app_matches[task_app_matches['pct_20'] == True]
    # pct_20 column doesn't exist or is all False!
    # Task gets marked as "no match"
```

## Why Stage 7's Matches Don't Exist in Stage 4

**Location**: `stage_7_analyze_results.py`, line 1822-1843

```python
task_matches = task_app_matches[task_app_matches['onet_task_id'] == task_id].copy()
percentile_col = self.percentile  # e.g., 'pct_05'

# Check if any match at specified percentile
if percentile_col in task_matches.columns:
    exposed_matches = task_matches[task_matches[percentile_col] == True]  # ← Filters by spec
    num_matching_at_pct = len(exposed_matches)
```

The matches shown in Stage 7 output don't exist in `app_task_matching_detail.csv` because:
1. Stage 7 loads incomplete `task_app_matches.csv` from Stage 5 (only last spec)
2. Tries to filter for current specification columns (pct_05, pct_20, etc.)
3. Gets wrong/empty results since wrong spec data is loaded
4. Falls back to showing whatever was in the incomplete file

## Architecture Problem

**Current Flow** (BROKEN):
```
Stage 5:
  - Loads full task_app_matches from Stage 4 (290M rows, all specs)
  - Processes each spec sequentially
  - OVERWRITES task_app_matches.csv with last spec only
  ↓
Stage 7:
  - Loads incomplete task_app_matches.csv (single spec)
  - Tries to use it for all specifications
  - Gets mismatched results
```

**Expected Flow** (WHAT SHOULD HAPPEN):
```
Stage 5:
  - Should save task_app_matches.csv ONCE with ALL specs/thresholds
  - OR rename per-spec versions (task_app_matches_pct_20_ce_0.6.csv)
  ↓
Stage 7:
  - Should load spec-specific file OR handle multi-spec structure
  - Should get correct matches for current specification
```

## Solution Options

### Option 1: Save One File with All Specs (Recommended)
- Move the `task_app_matches.to_csv()` call OUT of `step2_calculate_firm_task_exposure()`
- Save it ONCE in `run_full_pipeline()` with the full original `task_app_matches`
- Keep all spectral columns intact (pct_01, pct_05, pct_20, etc.)

**Benefits**:
- Stage 7 gets complete spec information
- Can filter for any specification dynamically
- Preserves all threshold information

### Option 2: Save Per-Spec Files
- Keep the current overwrite behavior but add spec identifier to filename
- Save as: `task_app_matches_pct_20_ce_0.6.csv` instead of generic `task_app_matches.csv`
- Modify Stage 7 to load the correct spec file

**Benefits**:
- Each spec file is complete for that spec
- Clear mapping between Stage 5 spec files and Stage 7 inputs

**Drawbacks**:
- More files to manage
- Stage 7 needs to construct filename from parameters

### Option 3: Don't Save to CSV at All
- Remove the `task_app_matches.to_csv()` line
- Stage 7 should load directly from Stage 4 parquet file
- Would require Stage 7 to handle parquet format

**Benefits**:
- Single source of truth (Stage 4 parquet)
- No duplication/synchronization issues

**Drawbacks**:
- Changes Stage 7 architecture
- Large parquet file (295MB+) needs to stay in memory

## Current Files Involved

**Stage 4 Output** (Source):
- `Data/stage_4/task_exposure_matches_all_thresholds_*.parquet` (~295M rows)
- Contains all spec columns: pct_01, pct_05, pct_20, ce_0.0, ce_0.2, ce_0.4, ce_0.6, ce_0.8, etc.

**Stage 5 Intermediate** (Being Saved):
- `Data/firm_year_exposure/task_app_matches.csv` (~5-100K rows depending on last spec)
- Only contains matches from LAST specification processed
- Gets overwritten each iteration

**Stage 5 Output** (Main Results):
- `Data/firm_year_exposure/isco_firm_year_exposure_*.csv`
- `Data/firm_year_exposure/onet_firm_year_exposure_*.csv`

**Stage 7 Input Expectation**:
- `Data/firm_year_exposure/task_app_matches.csv`
- Should have all spec columns for matching at specified threshold

## Validation

To confirm this issue:

1. Check when `task_app_matches.csv` was last modified:
   ```bash
   ls -l Data/firm_year_exposure/task_app_matches.csv
   ```

2. Check what specification it contains:
   ```python
   df = pd.read_csv('Data/firm_year_exposure/task_app_matches.csv')
   print(df.columns)  # Should show pct_* and ce_* columns
   print(df['pct_20'].value_counts())  # Check if any pct_20 values exist
   ```

3. Compare to Stage 4:
   ```python
   s4 = pd.read_parquet('Data/stage_4/task_exposure_matches_all_thresholds_*.parquet')
   print(s4.columns)  # Should have multiple spec columns
   ```

## Next Steps

1. **Immediate Fix**: Save full `task_app_matches` (Option 1)
   - Modify `run_full_pipeline()` to save original loaded `task_app_matches` once
   - Ensure all spec columns are preserved
   - Don't overwrite inside `step2_calculate_firm_task_exposure()`

2. **Update Stage 7**: Verify it handles multi-spec structure correctly
   - Should filter for current spec's columns dynamically
   - Should handle cases where spec columns exist or don't exist

3. **Add Validation**: Verify matches after fix
   - Confirm `task_app_matches.csv` has all spec columns
   - Verify Stage 7 output matches correspond to actual Stage 4 matches
   - Check that task-app pairs exist in `app_task_matching_detail.csv`
