<!-- .github/copilot-instructions.md - Guidance for AI coding agents working on this repo -->
# Copilot instructions — 10CompaniesPOCAI

Purpose: give AI coding agents the minimal, project-specific knowledge to be productive quickly.

- **Big picture**: This repository implements a staged pipeline that extracts AI tool usage from job ads and computes firm/occupation exposure.
  - Data flow (linear): PostgreSQL job table -> stage_0_get_job_ads.py -> stage_2_deduplicate_translate.py -> stage_3_extract_ai_tasks.py -> stage_4_onet_similarity.py -> stage_5_onet_to_isco_exposure.py -> stage_6_link_exposure_to_jobs.py
  - Outputs live in `Data/` and various `Data/*_exposure.csv` artifacts used by later stages.

- **Where to look first**:
  - CLI stages: `stage_0_get_job_ads.py`, `stage_2_deduplicate_translate.py`, `stage_3_extract_ai_tasks.py`, `stage_4_onet_similarity.py`, `stage_5_onet_to_isco_exposure.py`, `stage_6_link_exposure_to_jobs.py` (these implement the core logic and flags).
  - Pipeline orchestration: `ai_pipeline.py` (entrypoint for full runs).
  - Configuration: `config.env` / `Code/config.env` (DB creds, API keys). Never commit secrets.
  - Documentation and methodology: `CLAUDE.md`, `PROJECT_STATUS.md` (detailed design rationale and decisions).

- **Common developer workflows** (commands):
  - Install deps: `pip install -r requirements.txt`
  - Run a single stage (example): `python3 stage_0_get_job_ads.py --extract-keywords-batched`
  - Reprocess malformed LLM output (Stage 3): rerun `stage_3_extract_ai_tasks.py` with same `--step` and the fixed malformed CSV.

- **Project-specific patterns and conventions**:
  - Prefer the batched, memory-efficient versions (Stage 0 batched SQL queries and Stage 3 chainable steps).
  - LLM outputs are stored and validated; malformed responses are saved to `*_malformed.csv` and must be fixed and reprocessed (do not attempt silent auto-fixes).
  - Code should fail loudly on processing errors (raise exceptions rather than swallowing errors).
  - Caching is used (e.g., job list caches like `Data/stage6_job_cache_*.parquet`): check for caches before re-running expensive steps.
  - Cost-control flags exist (flex processing, model selection). Prefer lower-cost options unless evaluating model performance.

- **Models & costs**:
  - Preferred models per stage are documented in `CLAUDE.md` (e.g., O3 for Step 1, `gpt-5-mini` for Step 2). Follow those defaults for reproducibility.
  - Always warn or obtain explicit approval before calling paid APIs or changing model defaults.

- **Safety / repository rules (must follow)**:
  - Ask a human before running any script that will call external APIs or query the production database.
  - Do not create arbitrary new files without approval; prefer in-place edits to existing files unless requested.
  - Maintain reproducibility: include CLI flag examples and expected input/output file paths in changes.

- **Integration points & external deps**:
  - PostgreSQL database (credentials in `config.env`).
  - LLM providers (OpenAI/BGE) via API keys; usage is gated by explicit approval.
  - Crosswalk files and external mapping spreadsheets live in `Data/` (e.g., ESCO/ONET crosswalks).

- **How to handle a code change**:
  - Small bugfixes: run the targeted stage locally with a small sample and produce the same output layout.
  - Larger changes: propose design, list affected stages (0–6), and ask for permission to run full pipeline.

- **Useful quick examples** (copyable):
  - Stage 0: `python3 stage_0_get_job_ads.py --extract-keywords-batched`
  - Stage 3, step 2 re-run (task separation): `python3 stage_3_extract_ai_tasks.py custom --input-file FILE.csv --step 2 --content-column ai_applications_raw --model gpt-5-mini`
  - Stage 6 link (ISCO): `python3 stage_6_link_exposure_to_jobs.py --occ_code isco --in_dir Data --out Data/stage6_jobs_linked.csv`

If anything above is unclear or you want a different level of detail (more command examples, CI hooks, or coding style rules), tell me which section to expand.
