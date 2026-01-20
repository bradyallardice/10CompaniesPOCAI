# AI Tool Usage Extraction Pipeline

**Project**: Extract AI applications and measure occupation-level AI exposure in job advertisements
**Methodology**: Hamptone et al. approach adapted for firm-level analysis
**Language**: Python, PostgreSQL, LLM APIs (OpenAI)

---

## Quick Reference

### Documentation
- **Full methodology & analysis plan**: @docs/methodology.md
- **Complete pipeline specifications** (Stages 0-7): @docs/pipeline-specs.md
- **Cost optimization strategies**: @docs/cost-optimization.md
- **Known issues & data quality notes**: @docs/known-issues.md

### Code Standards
- **Mandatory code practices**: @.claude/rules/code-practices.md
- **Data quality validation**: @.claude/rules/data-quality.md
- **Approval requirements**: @.claude/rules/approval-gates.md

---

## Critical Rules (MANDATORY)

### Approval Gates
**REQUIRED BEFORE**: Running scripts, making API calls, modifying code, creating files, or database operations
→ See @.claude/rules/approval-gates.md for full requirements

### Code Practices
- **Fail-fast errors**: Stop on error, never silently fail or find workarounds
- **No new files**: Modify existing files only (unless explicitly requested)
- **Preserve structure**: Keep original file organization and naming
- **Use absolute paths**: Never relative paths
- **Validate everything**: Check types, missing values, ranges, assumptions
- **Understand Consequences of Changes**: When suggesting changes for a found bug, first state the risk of the changes, the downstream effects of the change, and how we will validate the changes.
- **Understand Consequences of Changes**: When suggesting changes for a found bug, first state the risk of the changes, the downstream effects of the change, and how we will validate the changes.

### Data Quality
- **No NaN values** in critical output columns (post-processing validation required)
- **Report diagnostics**: Always provide mapping coverage and unmatched records
- **Validate schemas**: Verify expected columns and data types before proceeding
- **Provide Summariess**: Always add in summaries at the end of scripts showing data quality summaries.
→ See @.claude/rules/data-quality.md for validation requirements

---

## File Structure

### Testing Organization
All testing outputs follow this structure:
```
Data/Testing/
├── stage_N/
│   ├── {test_name}/
│   │   ├── {output_files}
│   └── {test_name2}/
│       └── {output_files}
```

Common test names: `1000_company_test`, `functionality_test`, `1000_company_sample`

Example paths:
- `Data/Testing/stage_3/1000_company_test/final_output_1000_sample.csv`
- `Data/Testing/stage_5/functionality_test/isco_firm_year_exposure_core_tasks_*.csv`
- `Data/Testing/stage_7/1000_company_test/firms/*.csv` and `occupations/*.png`

### Production Files
- **Stage outputs**: `Data/stage{N}_*.csv` or `Data/stage{N}_*.parquet`
- **LLM outputs**: `Data/llm_output/*_step{N}_*.csv`
- **Embeddings**: `Data/embeddings/*.pkl`
- **Checkpoints**: `Data/embeddings/cross_encoder_checkpoints/ce_*.parquet`
- **Ground truth**: `Data/manual_coding_evaluation/*.csv`
- **Malformed responses**: `Data/*_step{N}_*_malformed.csv`
- **Job cache**: `Data/stage6_job_cache_max{year}.parquet`
- **Crosswalks**: `Data/240711_occupation_to_ch_isco_19.csv`, `Data/ESCO_to_ONET-SOC.xlsx`

---

## Current State (January 2026)

- ✅ **Post-Dec 2025**: Stage 0 produces clean output (zero NaN values in matched_keywords)
- ✅ **All stages**: Use batched processing for memory efficiency
- ✅ **Stage 3**: Three-step modular pipeline with malformed response handling
- ✅ **Stage 4**: Content-based checkpoint system prevents data loss
- ✅ **Stage 6**: Produces linked exposure data with diagnostic files
- ✅ **Stage 7**: Comprehensive analysis with summary statistics and visualizations
- ✅ **Cost optimization**: Flex processing enabled (50% cost reduction on GPT-5, O3)

---

## Stage Overview

### Stages 0-2: Data Preparation
Extract AI-related jobs from database, deduplicate, filter false positives

### Stage 3: AI Application Extraction
Three-step LLM prompting: extract → separate → filter
- Best models: O3 (Step 1), GPT-5-mini (Step 2), O3 (Step 3)
- Performance: F1 = 0.7817 → 0.8442 → 0.9194
- Handles malformed responses with reprocessing support

### Stage 4: O*NET Similarity Analysis
Deduplicate AI applications, create embeddings, calculate task similarity
- Output: AI applications matched to O*NET tasks (95th percentile threshold)

### Stage 5: Firm-Occupation Exposure
Calculate exposure measures at firm×occupation×year level
- Methods: Hampole (share-based) and Binary
- Includes intensity adjustment: log(1 + N_applications)

### Stage 6: Link to Actual Jobs
Join exposure measures to jobs in database
- Key: (company_id, year, occupation_code)
- Produces: Linked dataset + diagnostics for unmatched records

### Stage 7: Analysis & Reporting
Comprehensive breakdowns by firm, occupation, and firm×occupation pairs
- Generates: Summary statistics, visualizations, sensitivity analysis

---

## Quick Start for New Tasks

1. **Read the relevant documentation** first (see Quick Reference above)
2. **Check approval requirements** (@.claude/rules/approval-gates.md)
3. **Review code standards** (@.claude/rules/code-practices.md)
4. **Verify data quality requirements** (@.claude/rules/data-quality.md)
5. **Check known issues** (@docs/known-issues.md) for your stage
6. **Ask me questions** if anything is unclear

---

## Key Principles

**Research-Grade Code**: Correctness and transparency over cleverness and optimization

**Fail-Fast**: Errors must stop processing, not trigger workarounds or silent failures

**Explicit Over Implicit**: No auto-detection, no guessing, no hidden behavior

**Approval-Gated**: All code execution, API calls, and modifications require user approval

**Transparent Design**: Pipeline decisions documented, assumptions explicit, logic auditable

---

## Contact & Resources

- **For methodology questions**: See @docs/methodology.md
- **For technical specifications**: See @docs/pipeline-specs.md
- **For cost concerns**: See @docs/cost-optimization.md
- **For data quality issues**: See @.claude/rules/data-quality.md and @docs/known-issues.md
- **For code questions**: See @.claude/rules/code-practices.md
