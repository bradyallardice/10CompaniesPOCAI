# Stage 6 Test Suite

Comprehensive, production-grade test suite for Stage 6: Link Firm×Year Exposure to Jobs

## Overview

This test suite implements a **fail-fast, contract-driven testing approach** with deterministic fixtures, golden file comparisons, and comprehensive error handling validation.

## Test Structure

```
tests/
├── fixtures/                          # Deterministic test data
│   ├── job_postings_unified.csv       # 5 hand-crafted job records
│   ├── 240711_occupation_to_ch_isco_19.csv  # X28→ISCO crosswalk
│   ├── ESCO_to_ONET-SOC.xlsx          # ISCO→ONET crosswalk  
│   └── Data/                           # Stage 5 exposure files
│       ├── isco_firm_year_ai_exposure_time_variant.csv
│       ├── isco_firm_year_ai_exposure_time_invariant.csv
│       ├── onet_firm_year_ai_exposure_time_variant.csv
│       └── onet_firm_year_ai_exposure_time_invariant.csv
├── golden/                             # Expected outputs
│   ├── stage6_jobs_linked__jobads_isco_variant.csv
│   ├── stage6_jobs_linked__jobads_onet_variant.csv
│   ├── stage6_unmatched_jobs__jobads_isco_variant.csv
│   └── stage6_unmatched_exposures__jobads_isco_variant.csv
├── test_stage6_units.py               # Pure function tests
├── test_stage6_failfast.py            # Error handling tests
├── test_stage6_joins.py               # Core linking logic tests
├── test_stage6_cli.py                 # End-to-end CLI tests
├── pytest.ini                         # Pytest configuration (warnings→errors)
├── run_stage6_tests.sh                # Comprehensive test runner
└── README.md                          # This file
```

## Test Categories

### 1. Unit Tests (`test_stage6_units.py`)
Tests individual functions in isolation:
- ✅ X28 occupation parsing (JSON arrays, comma-separated, edge cases)
- ✅ Input validation with precise thresholds (>2% empty X28 = error)  
- ✅ Data type conversion and cleaning
- 🔄 Active year expansion logic (earliest `tst_created` onward, deletions ignored)

### 2. Fail-Fast Tests (`test_stage6_failfast.py`)
Tests error conditions and boundary cases:
- ✅ Missing crosswalk files → `FileNotFoundError`
- ✅ Unmapped X28 codes → `ValueError` (when `fail_on_warn=True`)
- ✅ Missing required columns → `ValueError` with descriptive message
- 🔄 Duplicate Stage 5 keys → `ValueError`
- ✅ Empty job expansion (0 rows) → `ValueError`
- ✅ Excessive empty X28 (>2%) → `ValueError`
- 🔄 Crosswalk drift detection (hash changes) → `ValueError`

### 3. Join Logic Tests (`test_stage6_joins.py`)
Tests core linking between jobs and exposures:
- 🔄 Crosswalk application (X28→ISCO, ISCO→ONET)
- 🔄 Inner join correctness with exact row counting
- 🔄 Unmatched diagnostics generation
- ✅ Company ID authoritative (not company name) for matching
- 🔄 Duplicate key detection and handling
- ✅ One-to-many crosswalk mappings (exploding correctly)
- ✅ Empty join results handling

### 4. CLI Tests (`test_stage6_cli.py`)
Tests complete end-to-end pipeline:
- 🔄 `job_ads × isco × time_variant` → golden file comparison
- 🔄 `job_ads × onet × time_variant` → schema validation
- 🔄 `job_ads × isco × time_invariant` → output verification  
- ✅ Missing arguments → graceful failure
- ✅ Invalid arguments → validation errors
- ✅ Help message display
- ✅ File not found → descriptive error messages

**Legend**: ✅ Implemented | 🔄 Ready for implementation once Stage 6 logic is complete

## Test Data Design

### Fixtures (Deterministic & Hand-Labeled)

**Jobs (`job_postings_unified.csv`)**:
- **Alpha AG**: Data Scientist (2020+), ML Engineer (2021+) 
- **Beta SA**: BI Analyst (2021+) with unmapped X28 code
- **Gamma GmbH**: DevOps Engineer (2020+) with empty X28 occupations

**Active-Year Rule**: Deletions ignored; titles active from earliest `tst_created` onward.

**Expected Matches**:
- Alpha/2020/Data Scientist → 2521, 2523 (exposure scores: 0.60, 0.20)
- Alpha/2021/ML Engineer → 2512 (exposure score: 0.90)
- Beta/2021/BI Analyst → **NO MATCH** (unmapped X28)
- Gamma/DevOps → **NO MATCH** (empty X28)

### Golden Files (Expected Outputs)

**`stage6_jobs_linked__jobads_isco_variant.csv`**: Exactly 3 matched rows  
**`stage6_unmatched_jobs`**: Beta SA + Gamma GmbH records  
**`stage6_unmatched_exposures`**: 1 exposure row (company=1, year=2022) with no job

## Running Tests

### Quick Test (Units + Fail-Fast)
```bash
cd tests/
pytest test_stage6_units.py test_stage6_failfast.py -v
```

### Full Test Suite
```bash
cd tests/
./run_stage6_tests.sh
```

### Specific Test Categories
```bash
pytest -m unit              # Unit tests only
pytest -m failfast          # Error handling tests  
pytest -m cli               # CLI tests only
pytest test_stage6_joins.py # Join logic tests
```

## Configuration

**`pytest.ini`** enforces fail-fast discipline:
- `filterwarnings = error` → All warnings become errors
- `--tb=short` → Concise error traces
- `--strict-markers` → Enforced test markers

## Expected Behavior

### Passing Tests
When all tests pass, Stage 6 produces:
- Correct join results matching golden files
- Proper error handling for invalid inputs  
- Deterministic output schemas
- Comprehensive diagnostic files

### Failing Tests  
Tests fail fast on:
- Schema violations (missing columns)
- Data quality issues (>2% empty X28)
- Unmapped crosswalk codes
- Duplicate keys in input data
- File not found errors
- Any warnings (treated as errors)

## CI/CD Integration

```bash
# In CI pipeline
cd tests/
pytest . --tb=short -v
# Exit code 0 = all tests pass, ready for production
# Exit code != 0 = fail build, investigate failures
```

## Next Steps

1. **Complete Stage 6 implementation** following the test contracts
2. **Run tests after each feature** to ensure correctness  
3. **Update golden files** if business logic changes
4. **Add property-based tests** with Hypothesis for randomized validation
5. **Add performance tests** for large datasets if needed

## Key Principles

- ✅ **Fail-Fast**: Any error condition stops execution immediately
- ✅ **Deterministic**: Same inputs always produce same outputs
- ✅ **Contract-Driven**: Tests define expected behavior precisely
- ✅ **Golden Files**: Explicit expected outputs for regression testing
- ✅ **Comprehensive**: Unit, integration, CLI, and error handling coverage