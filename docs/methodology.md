# Research Methodology: AI Tool Usage Extraction

## Overview

This research follows the Hamptone et al. approach to identify AI applications and exposure in job advertisements. The methodology consists of two main stages:

1. **AI keyword extraction and task identification** - Extract AI-related job postings and identify AI applications
2. **AI exposure classification** - Map AI applications to O*NET occupational tasks for exposure analysis

---

## File Flow Reference

```
PostgreSQL Database (14.7M jobs) (Job Ads)
    ↓
Stage 0: stage_0_get_job_ads.py
    ↓
Data/ai_development.csv (107,524 rows, 351 NaN)
    ↓
Stage 2: stage_2_deduplicate_translate.py
    ↓
Data/ai_development_deduplicated_custom.csv (56,942 rows, 300 NaN)
    ↓
Stage 3: stage_3_extract_ai_tasks.py (custom pipeline)
    ↓
Data/llm_output/* (AI applications extracted)
    ↓
Stage 4: stage_4_onet_similarity.py
Stage 5: stage_5_firm_exposure.py
Stage 6: stage_6_link_exposure_to_jobs.py
```

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

## Key Innovation

Unlike typical occupation-level AI exposure measures, this approach captures **within-occupation, between-firm variation** in AI usage, enabling analysis of how the same occupation can have different AI exposure depending on which specific company the worker is employed at.

This is particularly important for:
- Understanding heterogeneous AI adoption across firms
- Identifying firm-level determinants of AI exposure
- Modeling individual-level responses to firm-specific AI adoption
