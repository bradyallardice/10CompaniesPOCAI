# AI Tool Usage Extraction: Allardice/Kurer Roadmap (Hamptone Approach) - Project Status

**Last Updated:** 2025-07-16

## Overview
This project implements a three-stage approach to identify AI applications and exposure in job advertisements, adapted from Hamptone et al. methodology. The approach analyzes job postings from 100 Swiss companies using multilingual keyword matching, hand-coding, and LLM-based task extraction.

## Project Structure
```
10CompaniesPOCAI/
├── Code/
│   ├── JobAds/               # Job advertisement processing
│   └── Tasks/                # Task extraction utilities
├── Data/
│   ├── raw_ads/             # Raw job postings data
│   ├── candidates/          # Filtered AI-related postings
│   ├── llm_output/         # LLM processing results
│   ├── keywords/           # Multilingual keyword lists
│   └── [various CSV files] # Processing outputs and translations
├── archive/
│   └── deprecated/         # Deprecated/old code files
├── tests/
│   └── validation/         # Test scripts
├── tools/
│   └── monitoring/         # Progress monitoring utilities
├── step_0_get_job_ads.py    # Stage 1: Data extraction
├── step_1_keyword_match.py  # Stage 1: Keyword matching
├── step_2_deduplicate_translate.py # Stage 2: Deduplication & translation
├── step_3_extract_ai_tasks.py     # Stage 3: AI task extraction
├── ai_pipeline.py          # Main pipeline orchestrator
├── config.env              # Configuration file
├── requirements.txt        # Python dependencies
├── CLAUDE.md              # Project methodology documentation
├── PROJECT_STATUS.md      # This status file
└── README.md              # Project overview
```

## Pipeline Implementation Status

### ❌ Stage 1: Simple Keyword-Matching ("AI Keywords")
**Focus:** Collect job ads concerned with active AI development ("AI workers")

**Tasks:**
- [ ] Load 100 company sample from PostgreSQL database
- [ ] Clean text to remove special characters
- [ ] Search text for multilingual AI keywords (German/French/Italian/English)
- [ ] Handle "ai" false positives in Italian text
- [ ] Output subsample of AI-related job ads (~1% of total)

**Expected Outcome:** Small subsample of job ads, mostly tech-intensive roles in IT development

### ❌ Stage 2: Hand-Code Selected Job Ads
**Focus:** Create an evaluation set for future use

**Tasks:**
- [ ] Load subsample of jobs from Stage 1
- [ ] Deduplicate job postings
- [ ] Translate raw text from source language to English
- [ ] Output file for hand-coding (raw translated + untranslated text)
- [ ] Hand-code job ads based on LLM instructions

**Expected Outcome:** Hand-coded job tasks file for evaluation

### ❌ Stage 3: Extract AI Tasks from Selected Job Ads
**Focus:** LLM-based identification of AI applications

**Tasks:**
- [ ] Step 1: Extract AI applications and translate to English
- [ ] Step 2: Filter and clean applications
- [ ] Step 3: Final filtering pass
- [ ] Generate harmonized task format

**Expected Outcome:** List of AI applications (~2 tasks per AI-developing job ad)

### ❌ Stage 4: Evaluate
**Focus:** Ensure model performance on evaluation set

**Tasks:**
- [ ] Compare number of tasks extracted vs manual approach
- [ ] Compare overall similarity of extracted tasks
- [ ] Manually inspect differences between manual and LLM approach

### ❌ Stage 5: Iterate Stages 3-4 Until Happy with Results
**Focus:** Refine approach based on evaluation results

**Tasks:**
- [ ] Iterate on LLM prompts and processing
- [ ] Validate against evaluation set
- [ ] Finalize methodology

### ❌ Stage 6: Repeat Steps 1-3 on Full Dataset
**Focus:** Scale approach to complete dataset

**Tasks:**
- [ ] Apply refined methodology to full dataset
- [ ] Process all job advertisements
- [ ] Generate complete AI applications dataset

### ❌ Stage 7: Classify AI Exposure at Task Level
**Focus:** Classify all occupational tasks into exposed/not-exposed categories

**Tasks:**
- [ ] Generate word embeddings for AI applications
- [ ] Calculate cosine similarity with O*NET tasks
- [ ] Apply 95th percentile threshold for exposure classification
- [ ] Calculate firm-level AI application similarity
- [ ] Generate weighted average exposure by occupation
- [ ] Merge with Swiss Household Panel data

**Expected Outcome:** Every O*NET task classified as exposed or not-exposed

## Data Metrics
- **Total Companies:** 0 (Target: 100)
- **Total Ads:** 0
- **AI Candidate Ads:** 0
- **Hand-coded Evaluation Set:** 0
- **Extracted AI Tasks:** 0
- **O*NET Tasks Classified:** 0

## Implementation Notes

### Key Requirements
- PostgreSQL database access for Swiss jobs (credentials in config.env)
- Multilingual text processing capabilities (German/French/Italian/English)
- OpenAI/Anthropic API access for LLM processing
- Embedding generation for O*NET task similarity calculation
- Integration with Swiss Household Panel data

### Multilingual Keywords List
Based on Hamptone et al. keywords, expanded for Swiss market:
- AI abbreviations: a.i., ki, k.i., ia, i.a., ai
- Artificial Intelligence in 4 languages
- Machine Learning in 4 languages  
- Deep Learning in 4 languages
- Neural Networks in 4 languages
- Natural Language Processing in 4 languages
- Computer Vision in 4 languages
- Large Language Models in 4 languages

### Expected Outcomes by Stage
- **Stage 1:** ~1% of job ads identified as AI-related
- **Stage 2:** Evaluation dataset of hand-coded tasks
- **Stage 3:** ~2 tasks per AI-developing job ad (range 1-5)
- **Stage 7:** All O*NET tasks classified for exposure analysis

## Analysis Applications

### Data Integration
Merge "AI Exposure Average" with Swiss Household Panel individual respondents at occupation×firm×time level.

### Analysis Dimensions
- **Within occupations between firms:** Firm-level variation analysis
- **Within individual over time:** Response to changing AI exposure  
- **Sources of variation:** Time, firm changes, job transitions

### Key Equations
- **Equation 23:** Firm-Level AI Application Similarity
- **Equation 24:** Weighted Average Exposure
- **Equation 26:** AI Exposure Average (occupation×firm×time)

## Known Issues & TODO Items

### Stage 6: Crosswalk Coverage Deficiencies

#### X28→ISCO Crosswalk Gaps (2.5% job loss)
- **Status:** 97.5% mapping success, 445,137 jobs unmapped
- **Root Cause:** 110 missing X28 codes (newer occupational codes not in crosswalk)
- **High-Impact Missing Codes:**
  - `11001931`: 93,746 jobs
  - `11002081`: 77,392 jobs
  - `11001544`: 72,011 jobs
  - `11000005`: 55,934 jobs
  - `11000062`: 38,196 jobs
- **TODO:** Research and create manual mappings for top 10 missing X28 codes to recover ~75% of lost jobs

#### ISCO→ONET Crosswalk Gaps (8.4% job loss)
- **Status:** 91.6% mapping success, 1,470,963 jobs unmapped
- **Root Cause:** 41 broad ISCO codes (major/sub-major groups) not in ESCO crosswalk
- **High-Impact Missing Categories:**
  - `2000` (Professionals): 182,040 jobs
  - `2300` (Teaching professionals): 179,519 jobs
  - `7200` (Metal/machinery workers): 148,900 jobs
  - `3000` (Technicians): 138,331 jobs
  - `8100` (Plant operators): 115,928 jobs
- **Decision:** Accepted as legitimate limitation - ESCO only covers specific 4-digit unit groups, not broad categories
- **Note:** Jobs coded at broad levels cannot be mapped to specific O*NET occupations

#### Overall Impact
- **Combined Coverage:** ~84% of jobs successfully mapped through both crosswalks
- **Data Quality:** Enhanced validation ensures no specific occupational codes are missing (only broad categories)
- **Recommendation:** Current coverage acceptable for analysis, but X28 updates could improve completeness

## Setup Instructions

### Prerequisites
- Python ≥ 3.10
- PostgreSQL access to job_postings table
- OpenAI/Anthropic API key (for LLM processing)
- Swiss Household Panel data access

### Installation
```bash
cd Code/
pip install -r requirements.txt
```

### Configuration
Update `Code/config.env` with:
- Database credentials
- API keys for LLM services

---
*Updated to reflect Hamptone et al. methodology adaptation*
