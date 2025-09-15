# AI Tool Usage Extraction: Allardice/Kurer Roadmap (Hamptone Approach)

This methodology follows a three-stage approach to identify AI applications and exposure in job advertisements, adapted from Hamptone et al.

---

## Overview

The approach consists of two main stages:

1. **AI keyword extraction and task identification** - Extract AI-related job postings and identify AI applications  
2. **AI exposure classification** - Map AI applications to O*NET occupational tasks for exposure analysis

---

## CURRENT IMPLEMENTED WORKFLOW (September 2025)

The following documents the current optimized workflow with batched processing, database normalization, and comprehensive evaluation systems:

### Stage 0: Batched Keyword Extraction (stage_0_get_job_ads.py)
**Function**: `extract_keyword_matching_jobs_batched()` - **PREFERRED METHOD**
- **Input**: 14.7M jobs in PostgreSQL database
- **Process**: Uses predefined multilingual keyword list to search normalized text (`content_norm`) of job postings, then extracts the original non-normalized text and other columns for matching jobs
- **Keywords**: Multilingual keyword list from `Data/ai_keywords_multilingual_v3.csv` including AI/ML terms in English, German, French, and Italian
- **Output**: `Data/batched_ai_jobs.csv` (~1.5M jobs) containing original text and metadata
- **Optimization**: Date-based batched SQL queries (6-month chunks) for memory efficiency
- **Command**: `python3 stage_0_get_job_ads.py --extract-keywords-batched`

### Stage 2: Deduplication and Translation (stage_2_deduplicate_translate.py)
**Function**: Multi-level deduplication pipeline
- **Input**: 1.5M jobs from Stage 0
- **Process**: 
  1. Remove exact duplicates
  2. Remove false positives 
  3. Remove near duplicates based on content similarity
- **Output**: 
  - `Data/similar_duplicates_removed.csv` (752K jobs, ~50% reduction)
  - Mapping file of all duplicates removed for later reintegration
- **Translation**: Optional automatic language detection and translation to English (not normally used)

---

## Stage 3: AI Application Extraction Pipeline

### Overview
Stage 3 provides a **clean, modular extraction pipeline** with independent steps that can be chained together. Each step is runnable independently with malformed response handling and reprocessing capabilities.

### Production Pipeline (Custom Processing - Default Method)

**Purpose**: Main production pipeline for extracting AI tasks from custom datasets

#### **Step-by-Step Extraction**

Each step runs independently and produces chainable outputs:

**Step 1: AI Application Extraction**
```bash
python3 stage_3_extract_ai_tasks.py custom --input-file FILE.csv --step 1 \
  --content-column content_clean --model o3 --reasoning-effort medium
```
- **Input**: Raw job advertisements
- **Output**: `ai_applications_raw` column with extracted AI applications
- **Best Model**: O3 (F1=0.7817 on training set)

**Step 2: Task Separation**  
```bash
python3 stage_3_extract_ai_tasks.py custom --input-file STEP1_OUTPUT.csv --step 2 \
  --content-column ai_applications_raw --model gpt-5-mini
```
- **Input**: AI applications from Step 1
- **Output**: `step2_output` column with separated individual tasks
- **Best Model**: GPT-5-mini (F1=0.8442, cost-effective)

**Step 3: Final Filtering**
```bash
python3 stage_3_extract_ai_tasks.py custom --input-file STEP2_OUTPUT.csv --step 3 \
  --content-column step2_output --model o3 --reasoning-effort low
```
- **Input**: Separated tasks from Step 2
- **Output**: `step3_output` column with filtered, O*NET-style tasks
- **Best Model**: O3 with low reasoning effort

#### **Malformed Response Handling**

The system automatically:
- **Detects malformed responses** (empty, invalid JSON, parsing failures)
- **Saves malformed data** to separate CSV files (`custom_step{N}_{filename}_malformed.csv`)
- **Supports reprocessing**: Fix malformed file and rerun same command
- **Auto-cleanup**: Deletes resolved malformed files after successful reprocessing

#### **Reprocessing Workflow**
```bash
# If malformed responses found:
# 1. Fix issues in: Data/custom_step2_{filename}_malformed.csv  
# 2. Rerun same command - automatically detects and processes only malformed items
python3 stage_3_extract_ai_tasks.py custom --input-file FILE.csv --step 2 \
  --content-column ai_applications_raw --model gpt-5-mini
```

### Evaluation System (For Model Testing)
**Purpose**: Iterative testing and optimization of extraction steps using hand-coded ground truth
**Process**: 
- Use hand-coded ground truth files for each step
- Run different models against ground truth to compare performance
- Save results to performance history and prompt history for comparison
- Iterate on model parameters based on results

#### Current Evaluation Results:
- **Step 1**: O3 low reasoning: F1 = 0.7817
- **Step 2**: GPT-5-mini: F1 = 0.8442 (best performance)  
- **Step 3**: O3 low reasoning with JSON fix: F1 = 0.9194
- **Evaluation Method**: Semantic similarity using BGE embeddings (threshold: 0.70)

### Ground Truth Files
- **Step 2 Ground Truth**: `Data/manual_coding_evaluation/step_2_ground_truth_28082025.csv`
  - Contains 199 manually coded AI capabilities with separated tasks
  - Ground truth data in `ground_truth_separated_tasks` column (JSON format)
  - Used for evaluating task separation performance with semantic similarity matching

---

## Stage 4: AI Task Deduplication and O*NET Similarity Analysis

### Purpose
Deduplicate AI applications and calculate similarity to O*NET occupational tasks for AI exposure analysis.

### Process
1. **Deduplicate AI Applications**: Remove duplicate tasks from Stage 3 output, saving mapping file
2. **Create Text Embeddings**: Generate embeddings for all deduplicated AI tasks and O*NET tasks  
3. **Calculate Similarities**: Compare AI task embeddings to O*NET task embeddings
4. **Determine Exposure Threshold**: Tasks above 95th percentile similarity (default) are considered AI-exposed
5. **Optional Cross Encoder**: Run subset through cross encoder for refined results
6. **Output**: CSV file mapping AI tasks to O*NET tasks for Stage 5

---

## Stage 5: Firm and Occupation AI Exposure Calculation

### Purpose  
Create AI exposure measures at firm, firm×occupation, and occupation levels following Hampole et al. (2025) methodology.

### 4-Step Process

#### Step 1: Load Task-Application Matches
- Import Stage 4 similarity results (AI applications matched to O*NET tasks above 95th percentile)
- Creates binary exposure indicators: I^95_{j,i} = 1 if AI application *i* is similar to O*NET task *j*

#### Step 2: Calculate Firm-Level Task Exposure  
- **Reintegrate duplicates**: Map AI applications back to ALL original job postings (not just deduplicated subset)
- **Link applications to firms**: Each AI application gets mapped to specific companies and years
- **Calculate firm-task exposure**: For each firm-task-year combination:
  - **Hampole method**: Share of firm's AI applications that match the task 
  - **Binary method**: 1 if firm uses ANY application that matches the task, 0 otherwise
- **Key principle**: A task is only considered "exposed" at a company if that exact company posted a job using the corresponding AI application

#### Step 3: Aggregate to Occupation×Firm Level
- **Map tasks to occupations**: Use O*NET Task Statements to link each task to occupation codes
- **Weight by importance**: Apply O*NET task importance ratings as weights
- **Calculate weighted exposure**: For each firm-occupation-year:
  - Weighted average of task exposures within that occupation
  - Formula: Σ(task_exposure × importance_weight) / Σ(importance_weight)

#### Step 4: AI Intensity Adjustment
- **Scale by firm AI usage**: Multiply exposure scores by log(1 + N_applications)
- **Rationale**: Firms with more AI applications should have higher exposure scores
- **Final output**: "AI Exposure Average" varying at occupation×firm×time level

### Cross-walk to ISCO-08
- Maps O*NET occupations to ISCO-08 codes using ESCO crosswalk
- Enables integration with Swiss survey data (which uses ISCO classifications)
- Aggregates multiple O*NET codes to single ISCO codes using weighted averaging

### Output Files
- **Firm-level exposure**: Total AI intensity by firm×year
- **Occupation-firm exposure**: AI exposure by occupation×firm×year  
- **ISCO exposure**: Final exposure scores mapped to ISCO-08 for survey linking
- **Time variants**: Both time-invariant (firm exposed to all its apps across years) and time-variant (yearly exposure) versions

### Key Innovation
Unlike typical occupation-level AI exposure measures, this approach captures **within-occupation, between-firm variation** in AI usage, enabling analysis of how the same occupation can have different AI exposure depending on which specific company the worker is employed at.

---

## Cost Optimization Features
- **Flex Processing**: 50% cost reduction on GPT-5, O3, O4-mini models
- **Intelligent Retries**: Exponential backoff for rate-limited operations
- **Model Selection**: Automatic fallback from flex to standard processing when needed
- **Batched Operations**: Memory-efficient processing of large datasets

---

## Applications and Analysis

### Data Integration
Merge AI exposure scores with Swiss Household Panel individual respondents at occupation×firm×time level.

### Analysis Dimensions

#### Within occupations between firms [and over time]:
- Descriptive analysis demonstrating importance of firm-level variation
- Justifies job vacancy data approach vs. simple occupational measures

#### Within individual over time and firm:
- How individuals respond to changing AI exposure
- Outcomes: objective/subjective economic effects, political preferences, vote choice

#### Sources of Variation:
1. **Over time**: True "stayers" within same occupation and firm
2. **Across firms**: Individual changes firm but stays within occupation  
3. **Across occupations**: Job transitions over time

---

## Implementation Guidelines

## Coding practices
**MANDATORY**: All code created should be written so that errors in processing stop the process and throw errors, rather than trying to find workarounds to ensure the code completes it run successfully. Any code that explicitly attempts to find workarounds, or auto-detect, etc. must be approved.

### Code Execution and Changes
**MANDATORY**: All code execution and changes must be approved by the user before implementation. This includes:
- Running any scripts or pipeline steps
- Modifying existing code files
- Creating new files
- Making changes to keyword lists or detection logic
- Database queries or data processing

Always consult with the user before proceeding with any code execution or modifications.

### File Management
**CRITICAL**: Never create new files. Always modify existing files instead. When improving functionality:
- Edit the original file (e.g., `stage_1_keyword_match.py`)
- Do not create new versions or variations
- Make incremental improvements to existing code
- Preserve the original file structure and naming

**CRITICAL**: Always ask before running any code that calls any API to avoid accidental costs.

### Recent Improvements (September 2025)
- **Batched Processing**: Memory-efficient extraction of large datasets
- **Database Optimization**: Trigram GIN indexing for faster searches
- **Flex Processing**: 50% cost reduction on supported OpenAI models
- **Comprehensive Evaluation**: Semantic similarity with BGE embeddings
- **Multi-Model Support**: O3, GPT-5-mini, Claude-3.5-Sonnet with reasoning controls
- **Automated Retries**: Robust error handling with exponential backoff

---

**Version**: Optimized workflow with batched processing and comprehensive evaluation
**Date**: September 2025
