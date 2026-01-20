# Stage 5: AI Task Exposure Calculation and ISCO Crosswalk

This document provides detailed technical documentation for Stage 5 of the AI exposure pipeline, which calculates firm and occupation-level AI exposure measures following Hampole et al. (2025) methodology.

---

## Overview

Stage 5 transforms AI task-application matches from Stage 4 into comprehensive exposure measures at the firm×occupation×time level. The process follows a 4-step methodology that ensures precise attribution of AI exposure to specific companies and occupations.

**Key Innovation**: Unlike typical occupation-level AI exposure measures, this approach captures **within-occupation, between-firm variation** in AI usage, enabling analysis of how the same occupation can have different AI exposure depending on which specific company the worker is employed at.

---

## Core Methodology: 4-Step Exposure Calculation

### Step 1: Task-Application Matching
- **Input**: Stage 4 top matches results (`top_5_matches.csv` or `top_10_matches.csv`)
- **Process**: Load binary exposure indicators I^95_{j,i} = 1 if AI application *i* is similar to O*NET task *j*
- **Output**: Exposed task-application pairs above similarity threshold

### Step 2: Firm-Level Task Exposure
- **Reintegrate duplicates**: Map AI applications back to ALL original job postings (not just deduplicated subset)
- **Link to firms**: Each AI application gets mapped to specific companies and years
- **Calculate exposure**: Two methods available:
  - **Hampole method**: Share of firm's AI applications that match each task
  - **Binary method**: 1 if firm uses ANY application that matches the task, 0 otherwise
- **Key principle**: A task is only considered "exposed" at a company if that exact company posted a job using the corresponding AI application

### Step 3: Occupation×Firm Level Aggregation
- **Map tasks to occupations**: Use O*NET Task Statements to link each task to occupation codes
- **Weight by importance**: Apply O*NET task importance ratings as weights
- **Calculate weighted exposure**: For each firm-occupation-year:
  ```
  Weighted_Exposure = Σ(task_exposure × importance_weight) / Σ(importance_weight)
  ```

### Step 4: AI Intensity Adjustment
- **Scale by firm AI usage**: Multiply exposure scores by log(1 + N_applications)
- **Rationale**: Firms with more AI applications should have higher exposure scores
- **Final output**: "AI Exposure Average" varying at occupation×firm×time level

---

## File Structure and Implementation

### Main Class: `TaskFirmExposurePipeline`

**Initialization Parameters:**
```python
TaskFirmExposurePipeline(
    aggregation_method="mean",     # Method to aggregate O*NET → ISCO
    time_invariant=False,          # Firm exposure time handling
    occupation_exposure="none",    # Occupation-level exposure mode
    data_dir="Data/"              # Input data directory
)
```

### Key Methods

#### Data Loading Methods
- `step1_load_task_application_matches()` - Load Stage 4 similarity results
- `load_task_statements()` - O*NET Task Statements for task→occupation mapping
- `load_task_ratings()` - O*NET Task Ratings for importance weights
- `load_esco_onet_crosswalk()` - ESCO/ISCO to O*NET mapping
- `load_job_app_mapping()` - Job application to AI task mapping
- `load_company_data()` - Company information and metadata

#### Core Processing Methods
- `step2_calculate_firm_task_exposure()` - Calculate firm-level task exposure
- `step3_calculate_occupation_firm_exposure()` - Aggregate to occupation×firm level
- `step4_apply_ai_intensity_adjustment()` - Apply AI intensity scaling
- `crosswalk_onet_to_isco()` - Convert O*NET codes to ISCO-08

#### Data Integration Methods
- `load_original_full_dataset()` - Load complete job posting dataset
- `expand_job_app_mapping_to_full_dataset()` - Reintegrate duplicate mappings
- `load_stage2_deduplication_mapping()` - Deduplication reverse mapping

---

## Input Data Requirements

### Required Files
1. **Stage 4 Output**: `top_5_matches.csv` or `top_10_matches.csv`
2. **O*NET Data**:
   - `Task Statements.xlsx` - Task descriptions and occupation codes
   - `Task Ratings.xlsx` - Task importance weights
3. **Crosswalk Data**:
   - `ESCO_to_ONET-SOC.xlsx` - ESCO/ISCO to O*NET mapping
4. **Job Data**:
   - `ai_development_training_companies_200_jobs.csv` - Original job postings
   - `job_app_mapping.csv` - Job to AI application mapping
   - `similar_duplicates_removed.csv` - Stage 2 deduplication mapping

### Optional Files
- Company data CSV for enhanced firm information
- Custom Stage 4 parquet files for detailed similarity data

---

## Command Line Interface

### Basic Usage
```bash
python3 stage_5_onet_to_isco_exposure.py --top-matches-file Data/top_5_matches.csv
```

### Advanced Options
```bash
python3 stage_5_onet_to_isco_exposure.py \
    --top-matches-file Data/top_5_matches.csv \
    --output-file Data/isco_exposure_results.csv \
    --time-invariant \
    --occupation-exposure time-variant \
    --threshold 0.6 \
    --data-dir Data/ \
    --create-sample-report \
    --n-sample 20
```

### Key Parameters

**Core Processing:**
- `--top-matches-file`: Path to Stage 4 top matches CSV
- `--output-file`: Custom output file path (auto-generated if not provided)
- `--threshold`: Cross encoder threshold for exposed tasks (default: 0.6)

**Exposure Calculation:**
- `--time-invariant`: Use time-invariant exposure (firms exposed to all their apps across all years)
- `--occupation-exposure`: 
  - `none`: Firm-level only (default)
  - `time-invariant`: All firms exposed to all apps for all time
  - `time-variant`: All firms exposed to all apps from first appearance onwards

**Data Sources:**
- `--stage4-file`: Path to Stage 4 parquet file (auto-detect if not provided)
- `--stage2-mapping-file`: Stage 2 deduplication mapping (auto-detect if not provided) 
- `--company-file`: Company data CSV (auto-detect if not provided)
- `--data-dir`: Directory containing input files (default: Data/)

**Reporting:**
- `--create-sample-report`: Generate detailed ISCO task sample report
- `--n-sample`: Number of ISCO codes to sample for report (default: 10)

---

## Output Files

### Primary Output
**ISCO Exposure CSV** - Main results file containing:
- `isco08_4d`: ISCO-08 4-digit occupation code
- `isco08_title`: ISCO occupation title
- `company_name`: Company identifier
- `year`: Time period
- `ai_exposure_average`: Final AI exposure score (0-1 scale)
- `n_applications`: Number of AI applications used by firm
- `n_tasks`: Number of exposed tasks for this occupation
- `n_unique_apps`: Number of unique AI applications

### Intermediate Files
- **O*NET Exposure**: Pre-crosswalk results with O*NET codes
- **Firm Task Exposure**: Detailed task-level exposure by firm
- **Debug Files**: Intermediate processing results for validation

### Sample Report (Optional)
When `--create-sample-report` is used:
- **ISCO Task Sample Report**: Detailed breakdown showing:
  - Sample of ISCO occupations with high exposure
  - Specific tasks contributing to exposure
  - AI applications driving the exposure
  - Company-level variation within occupations

---

## Data Integration Flow

### Deduplication Handling
1. **Forward Mapping**: Stage 2 removed ~750K duplicates from 1.5M jobs
2. **Reverse Mapping**: Stage 5 reintegrates all duplicates to ensure complete coverage
3. **Result**: AI exposure calculated for ALL original job postings, not just deduplicated subset

### Time Dimension Options

**Time-Variant (Default)**:
- Firms exposed only to AI apps they posted in each specific year
- Captures temporal evolution of AI adoption within firms
- Enables analysis of "stayers" vs. "switchers"

**Time-Invariant**:
- Firms exposed to all their AI apps across all years
- Useful for analyzing overall firm AI capability
- Less sensitive to posting timing variations

### Occupation Exposure Modes

**Firm-Level Only** (`none` - Default):
- Each firm has its own exposure based on its specific AI applications
- Maximum variation between firms within same occupation
- Recommended for individual-level analysis

**Time-Invariant Occupation** (`time-invariant`):
- All firms in occupation exposed to all AI apps ever posted for that occupation
- Reduces between-firm variation, emphasizes occupation differences
- Useful for occupation-level aggregate analysis

**Time-Variant Occupation** (`time-variant`):
- All firms exposed to all AI apps from first appearance onwards
- Balances temporal evolution with occupation-level consistency

---

## Key Features and Innovations

### 1. Firm-Specific Exposure Attribution
- **Problem**: Traditional measures assign same exposure to all firms in an occupation
- **Solution**: AI exposure varies by firm within same occupation based on actual AI usage
- **Benefit**: Enables within-occupation analysis of AI adoption heterogeneity

### 2. Temporal Precision
- **Granularity**: Exposure calculated at year level
- **Flexibility**: Multiple time-handling options for different analytical needs
- **Accuracy**: Only firms that actually posted AI-related jobs get exposure

### 3. Comprehensive Data Reintegration
- **Challenge**: Stage 2 deduplication removed ~50% of job postings
- **Solution**: Full reverse mapping ensures no data loss
- **Result**: Complete coverage of original 1.5M job dataset

### 4. ISCO Crosswalk Integration
- **Purpose**: Enable integration with Swiss survey data (uses ISCO classifications)
- **Method**: ESCO crosswalk with weighted averaging for multiple O*NET→ISCO mappings
- **Output**: Ready for survey linking at ISCO×firm×year level

---

## Error Handling and Validation

### Built-in Checks
- **File existence validation**: Auto-detects missing input files
- **Data consistency checks**: Validates UID mappings across datasets
- **Coverage reporting**: Reports data loss at each processing step
- **Exposure range validation**: Ensures exposure scores are within [0,1] bounds

### Common Issues and Solutions

**Missing Job Application Mappings**:
- **Problem**: Some Stage 3 outputs may not have corresponding job mappings
- **Solution**: Reports unmapped entries and continues with available data
- **Monitoring**: Logs unmapped counts for quality assessment

**ISCO Crosswalk Gaps**:
- **Problem**: Some O*NET codes may not map to ISCO
- **Solution**: Uses fallback ESCO titles for comprehensive coverage
- **Result**: >95% crosswalk coverage for standard occupations

**Time Period Mismatches**:
- **Problem**: Job posting dates may be inconsistent across datasets
- **Solution**: Robust date parsing with multiple format support
- **Fallback**: Uses filename or metadata date information when needed

---

## Performance and Scalability

### Processing Time
- **Small datasets** (200 jobs): ~30 seconds
- **Medium datasets** (10K jobs): ~5 minutes  
- **Large datasets** (1M+ jobs): ~30-60 minutes

### Memory Usage
- **Peak memory**: ~2-4GB for 1M job dataset
- **Optimization**: Chunked processing for very large datasets
- **Storage**: Intermediate files cleaned up automatically

### Optimization Tips
1. Use `--time-invariant` for faster processing when temporal precision not needed
2. Pre-filter input data to specific companies/time periods if analyzing subsets
3. Use parquet format for large intermediate files when available

---

## Integration with Survey Data

### Swiss Household Panel Linking
The output is specifically designed for integration with Swiss survey data:

**Matching Variables**:
- `isco08_4d`: Links to survey occupation codes
- `company_name`: Links to employer information  
- `year`: Links to survey wave timing

**Analysis Opportunities**:
1. **Within-occupation variation**: Compare workers in same occupation at different firms
2. **Firm transitions**: Track individuals changing firms within same occupation
3. **AI adoption timing**: Analyze individual responses to firm-level AI introduction
4. **Heterogeneous effects**: Estimate differential impacts by AI exposure level

### Research Applications
- **Labor economics**: Impact of AI on wages, employment, job satisfaction
- **Political economy**: AI exposure effects on political preferences and voting
- **Survey methodology**: Using AI exposure as instrument or treatment variable
- **Firm analysis**: AI adoption patterns and competitive dynamics

---

## Version History and Updates

**Current Version**: September 2025
- Comprehensive 4-step exposure calculation
- Full deduplication reintegration
- ISCO crosswalk with weighted averaging
- Multiple time-handling options
- Extensive validation and error handling

**Key Improvements from Earlier Versions**:
- Added firm-specific exposure attribution (vs. occupation-level only)
- Implemented comprehensive data reintegration pipeline
- Enhanced ISCO crosswalk with fallback title handling
- Added sample reporting functionality for validation
- Improved error handling and data validation

**Future Enhancements**:
- Support for additional crosswalk standards (SOC, NACE)
- Integration with company financial data
- Advanced weighting schemes for task importance
- Industry-specific exposure calculations

---

**Author**: Brady Allardice
**Date**: September 2025  
**Methodology**: Hampole et al. (2025) with Swiss survey integration adaptations