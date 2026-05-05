# Appendix: End-to-End Pipeline Example

## Illustrative Example: AI Exposure Measurement for Manufacturing Quality Control

To illustrate the measurement pipeline, we trace a single AI application from job advertisement through to a firm-level exposure score. This example demonstrates how AI adoption by one firm propagates exposure to workers in non-tech occupations.

### The Case: Ivoclar Vivadent AG — Automated Defect Detection

Ivoclar Vivadent AG is a Swiss manufacturer of dental and orthodontic equipment, headquartered in Schaan, Liechtenstein. In their job postings, the firm advertises positions requiring expertise in computer vision systems for automated quality inspection. The following table traces one such application through the measurement pipeline:

**Table A1: Pipeline Trace — From Job Posting to Exposure Score**

| Pipeline Stage | Description | Output for This Example |
|---|---|---|
| **Stage 0: AI Keyword Extraction** | Search 14.7M job advertisements for multilingual AI/ML keywords | Job posting identified and extracted from database |
| **Stage 2: Deduplication** | Remove exact, false positive, and near-duplicate job ads | Posting survives filters; retained in 57K unique AI job sample |
| **Stage 3a: LLM Extraction (Step 1)** | Extract raw AI capabilities and applications from job text | Raw capability: surface defect detection via computer vision |
| **Stage 3b: Task Separation (Step 2)** | Separate raw capabilities into discrete, actionable sub-tasks | Capability refined to specific task statements |
| **Stage 3c: Final Filtering (Step 3)** | Filter task statements to O*NET-compatible phrasing | **Final task: "Detect and classify surface defects on [components] from high-resolution production images for real-time quality control."** |
| **Stage 4: O*NET Similarity Matching** | Calculate embedding similarity between AI task and 19K O*NET occupational tasks | O*NET task: *"Inspect parts for surface defects"* (task ID 3992) |
| **Stage 4: Cross-Encoder Validation** | Validate semantic alignment using cross-encoder model | Cross-encoder score: **0.9946** (exceeds 0.99 threshold) |
| **Stage 4: Map to Occupation** | Identify O*NET occupation performing matched task | O*NET occupation: Electrical and Electronic Equipment Assemblers (SOC 51-2022.00) |
| **Stage 5: Exposure Score Calculation** | Aggregate firm's AI applications to firm×occupation×year level | **hampole_ai_exposure_avg = 0.0089** (55 total AI applications at firm in 2025) |

### Mechanism: Within-Firm, Between-Occupation Variation

A key feature of this approach is that AI adoption by a firm creates exposure across all occupations it employs, not just those posting AI-specific job titles. Table A2 illustrates this mechanism by showing how Ivoclar Vivadent's 55 AI applications generate non-zero exposure for multiple occupations:

**Table A2: Within-Firm Propagation of Exposure — Ivoclar Vivadent AG (2025)**

| Occupation (O*NET SOC) | Occupation Title | hampole_ai_exposure_avg | Notes |
|---|---|---|---|
| 51-2022.00 | Electrical and Electronic Equipment Assemblers | 0.0089 | Directly performs defect inspection task matched to AI application |
| 51-2081.00 | Electrical and Electronic Equipment Installers and Repairers | 0.0083 | Installation and maintenance of equipment uses similar inspection skills |
| 49-9064.00 | Watch Repairers | 0.0078 | Precision component work shares defect detection requirements |
| 17-2199.09 | Nanotechnologists | 0.0064 | Manufacturing process support roles affected by quality automation |
| 43-5081.00 | Stock Clerks and Order Fillers | 0.0041 | Inventory roles exposed through firm-wide quality standards |

**Identification Strategy**: All occupations at Ivoclar Vivadent receive non-zero exposure due to the firm's AI adoption. In contrast, assemblers at a comparable manufacturing firm that has not posted AI job advertisements would receive zero exposure for the same occupation. The variation in exposure—despite identical occupational tasks—is driven entirely by firm-level AI adoption decisions, not occupational composition or worker characteristics.

### Data Quality Notes

- All figures are sourced from verified pipeline outputs: `ai_development_training_companies_200_jobs_step1_extracted_step2_step3.csv` (LLM extraction), `task_app_matches.csv` (cross-encoder scores), and `onet_firm_occupation_year_exposure.csv` (firm-level exposures)
- Cross-encoder score of 0.9946 indicates high semantic alignment between the AI application and the matched O*NET task
- The exposure measure `hampole_ai_exposure_avg` follows the methodology described in Stage 5 of the main text, incorporating intensity adjustment via log(1 + n_ai_apps_firm_year)
- Ivoclar Vivadent appears in employment records across 2018–2025, confirming this occupation as actively hired by the firm

