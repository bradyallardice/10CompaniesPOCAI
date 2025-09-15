#!/usr/bin/env python3
"""
Stage 5: Task → Occupation × Firm AI Exposure Pipeline

This stage implements a 4-step exposure calculation pipeline following Hampole et al. (2025):

Step 1: Task-Application matching (using stage 4 cross encoder results)
Step 2: Task exposure at firm level (Hampole share + Binary any-match variants)
Step 3: Occupation × Firm exposure aggregation using O*NET task importance weights
Step 4: AI intensity adjustment using log(1 + N_apps)

Then crosswalks results to ISCO-08 for integration with survey data.

Author: Claude Code Assistant  
Date: September 2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TaskFirmExposurePipeline:
    """
    Main class for calculating task → occupation × firm AI exposure following Hampole et al. (2025).
    """
    
    def __init__(self, 
                 aggregation_method: str = "mean",
                 time_invariant: bool = False,
                 occupation_exposure: str = "none",
                 data_dir: str = "Data/"):
        """
        Initialize the 4-step exposure pipeline.
        
        Args:
            aggregation_method: Method to aggregate O*NET → ISCO ("mean", "weighted_mean")
            time_invariant: If True, firms exposed to all their AI apps across all years; if False, only to apps in each specific year
            occupation_exposure: Occupation-level exposure mode ("none", "time-invariant", "time-variant")
            data_dir: Directory containing input data files
        """
        self.aggregation_method = aggregation_method
        self.time_invariant = time_invariant
        self.occupation_exposure = occupation_exposure
        self.data_dir = data_dir
        
        logger.info(f"Initialized Task → Occupation × Firm AI exposure pipeline")
        logger.info(f"  Aggregation method: {aggregation_method}")
        logger.info(f"  Firm time invariant exposure: {time_invariant}")
        logger.info(f"  Occupation exposure mode: {occupation_exposure}")
    
    def step1_load_task_application_matches(self, top_matches_file: str) -> pd.DataFrame:
        """
        Step 1: Load task-application matches from Stage 4 top matches results.
        
        Args:
            top_matches_file: Path to top_5_matches.csv file
            
        Returns:
            DataFrame with exposed task-app matches (I^95_j,i indicators)
        """
        logger.info(f"Step 1: Loading exposed task-application matches from: {top_matches_file}")
        
        matches_df = pd.read_csv(top_matches_file)
        logger.info(f"Loaded {len(matches_df):,} exposed task-application pairs")
        
        logger.info(f"  Unique tasks matched: {matches_df['onet_task_id'].nunique():,}")
        logger.info(f"  Unique AI applications: {matches_df['app_text'].nunique():,}")
        
        return matches_df
    
    def load_task_statements(self, task_statements_file: str, core_only: bool = True) -> pd.DataFrame:
        """
        Load O*NET Task Statements file to map task IDs to occupation codes.
        
        Args:
            task_statements_file: Path to Task Statements.xlsx
            core_only: If True, only include Core tasks (exclude Supplemental)
            
        Returns:
            DataFrame with task ID → O*NET code mapping
        """
        logger.info(f"Loading O*NET Task Statements from: {task_statements_file}")
        
        task_df = pd.read_excel(task_statements_file)
        logger.info(f"Loaded {len(task_df):,} task statements")
        
        if core_only:
            core_tasks = task_df[task_df['Task Type'] == 'Core'].copy()
            logger.info(f"Filtered to {len(core_tasks):,} Core tasks (excluded {len(task_df) - len(core_tasks):,} Supplemental)")
            task_df = core_tasks
        
        logger.info(f"Final dataset: {task_df['O*NET-SOC Code'].nunique():,} occupations, {task_df['Task ID'].nunique():,} tasks")
        
        return task_df
    
    def load_task_ratings(self, task_ratings_file: str) -> pd.DataFrame:
        """
        Load O*NET Task Ratings to get task importance weights.
        
        Args:
            task_ratings_file: Path to Task Ratings.xlsx
            
        Returns:
            DataFrame with task importance weights
        """
        logger.info(f"Loading O*NET Task Ratings from: {task_ratings_file}")
        
        ratings_df = pd.read_excel(task_ratings_file)
        logger.info(f"Loaded {len(ratings_df):,} task ratings")
        
        # Filter to Importance scale only
        importance_df = ratings_df[
            (ratings_df['Scale ID'] == 'IM') & 
            (ratings_df['Scale Name'] == 'Importance')
        ].copy()
        
        logger.info(f"Found {len(importance_df):,} importance ratings")
        logger.info(f"Importance rating range: {importance_df['Data Value'].min():.2f} - {importance_df['Data Value'].max():.2f}")
        
        # Keep key columns
        importance_df = importance_df[['O*NET-SOC Code', 'Task ID', 'Data Value']].copy()
        importance_df.rename(columns={'Data Value': 'importance_weight'}, inplace=True)
        
        return importance_df
    
    def load_esco_onet_crosswalk(self, esco_onet_file: str) -> pd.DataFrame:
        """
        Load ESCO to O*NET-SOC crosswalk.
        
        Args:
            esco_onet_file: Path to ESCO_to_ONET-SOC.xlsx
            
        Returns:
            DataFrame with ESCO/ISCO → O*NET mapping
        """
        logger.info(f"Loading ESCO → O*NET crosswalk from: {esco_onet_file}")
        
        # Load and clean headers
        xwalk_df = pd.read_excel(esco_onet_file, skiprows=2)
        xwalk_df.columns = ['esco_code', 'esco_title', 'onet_code', 'onet_title']
        
        # Remove header row that got mixed in
        xwalk_df = xwalk_df[xwalk_df['esco_code'] != 'ESCO/ISCO Code'].copy()
        xwalk_df = xwalk_df.dropna(subset=['esco_code', 'onet_code'])
        
        # Extract ISCO 4-digit codes from ESCO codes
        xwalk_df['isco08_4d'] = xwalk_df['esco_code'].astype(str).str[:4]
        
        logger.info(f"Loaded {len(xwalk_df):,} ESCO → O*NET mappings")
        logger.info(f"Unique ESCO codes: {xwalk_df['esco_code'].nunique():,}")
        logger.info(f"Unique O*NET codes: {xwalk_df['onet_code'].nunique():,}")
        logger.info(f"Unique ISCO 4-digit codes: {xwalk_df['isco08_4d'].nunique():,}")
        
        return xwalk_df
    
    def load_isco_titles(self) -> pd.DataFrame:
        """
        Load ISCO-08 4-digit occupation titles from ESCO crosswalk.
        Uses exact 4-digit ESCO codes as primary source, with fallback for missing codes.
        
        Returns:
            DataFrame with isco08_4d and isco08_title
        """
        logger.info("Loading ISCO-08 occupation titles from ESCO crosswalk...")
        
        esco_file = os.path.join(self.data_dir, "ESCO_to_ONET-SOC.xlsx")
        esco_df = pd.read_excel(esco_file, skiprows=2)
        esco_df.columns = ['esco_code', 'esco_title', 'onet_code', 'onet_title']
        esco_df = esco_df[esco_df['esco_code'] != 'ESCO/ISCO Code'].dropna()
        
        # Primary: Use exact 4-digit ESCO codes (these ARE ISCO titles)
        esco_df['code_length'] = esco_df['esco_code'].astype(str).str.len()
        exact_4d = esco_df[esco_df['code_length'] == 4].copy()
        primary_titles = exact_4d[['esco_code', 'esco_title']].drop_duplicates()
        primary_titles['isco08_4d'] = primary_titles['esco_code'].astype(str).str.zfill(4)
        primary_titles = primary_titles[['isco08_4d', 'esco_title']].drop_duplicates()
        primary_titles.rename(columns={'esco_title': 'isco08_title'}, inplace=True)
        
        logger.info(f"Found {len(primary_titles):,} exact 4-digit ISCO titles")
        
        # Fallback: For missing codes, use first available longer ESCO title
        esco_df['isco08_4d'] = esco_df['esco_code'].astype(str).str[:4]
        fallback_titles = esco_df.groupby('isco08_4d')['esco_title'].first().reset_index()
        fallback_titles.rename(columns={'esco_title': 'isco08_title'}, inplace=True)
        
        # Combine: primary titles + fallback for missing
        combined_titles = primary_titles.copy()
        missing_codes = set(fallback_titles['isco08_4d']) - set(primary_titles['isco08_4d'])
        if missing_codes:
            fallback_subset = fallback_titles[fallback_titles['isco08_4d'].isin(missing_codes)]
            combined_titles = pd.concat([primary_titles, fallback_subset], ignore_index=True)
            logger.info(f"Added {len(fallback_subset):,} fallback titles for missing codes")
        
        logger.info(f"Total ISCO titles available: {len(combined_titles):,}")
        return combined_titles
    
    def load_job_app_mapping(self) -> pd.DataFrame:
        """
        Load job-application mapping file to connect AI apps to specific jobs.
        
        Returns:
            DataFrame with app_text → job_uids mapping
        """
        logger.info("Loading job-application mapping...")
        
        mapping_file = os.path.join(self.data_dir, "job_app_mapping.csv")
        mapping_df = pd.read_csv(mapping_file)
        
        logger.info(f"Loaded {len(mapping_df):,} application-job mappings")
        logger.info(f"Unique AI applications: {mapping_df['app_text'].nunique():,}")
        logger.info(f"Unique job UIDs: {mapping_df['job_uids'].nunique():,}")
        
        return mapping_df
    
    def load_company_data(self, company_file: str) -> pd.DataFrame:
        """
        Load company data to map job UIDs to company information with time dimension.
        
        Args:
            company_file: Path to deduplicated AI jobs CSV file
            
        Returns:
            DataFrame with job UID → company mapping including year
        """
        logger.info(f"Loading company data from: {company_file}")
        
        company_df = pd.read_csv(company_file)
        
        # Extract year from date strings - simple string-based approach to avoid timezone parsing issues
        def extract_year_from_string(date_str):
            try:
                if pd.isna(date_str) or date_str == '':
                    return None
                # Extract first 4 characters as year (YYYY-MM-DD format)
                year = int(str(date_str)[:4])
                # Basic validation - reasonable year range
                if 2000 <= year <= 2030:
                    return year
                else:
                    return None
            except:
                return None
        
        company_df['year'] = company_df['tst_created'].apply(extract_year_from_string)
        
        # Check for missing years
        missing_years = company_df['year'].isna().sum()
        if missing_years > 0:
            missing_sample = company_df[company_df['year'].isna()]['uid'].head(5).tolist()
            missing_dates = company_df[company_df['year'].isna()]['tst_created'].head(5).tolist()
            raise ValueError(f"Data quality error: {missing_years:,} jobs have invalid tst_created dates that cannot be parsed for year extraction. "
                           f"Example job UIDs: {missing_sample}, Example dates: {missing_dates}")
        
        # Parse full datetime for those who need it (optional)
        company_df['tst_created'] = pd.to_datetime(company_df['tst_created'], errors='coerce')
        
        # Keep only relevant columns for mapping
        company_mapping = company_df[['uid', 'company_name', 'x28_industries', 'tst_created', 'year']].copy()
        company_mapping.rename(columns={'uid': 'job_uid'}, inplace=True)
        company_mapping['year'] = company_mapping['year'].astype(int)
        
        logger.info(f"Loaded {len(company_mapping):,} job-company mappings")
        logger.info(f"Unique companies: {company_mapping['company_name'].nunique():,}")
        logger.info(f"Year range: {company_mapping['year'].min()} - {company_mapping['year'].max()}")
        
        return company_mapping
    
    def load_original_full_dataset(self, original_file_path: str = None) -> pd.DataFrame:
        """
        Load the original full dataset before stage 2 deduplication to recover all jobs.
        Only loads columns needed for exposure calculations.
        
        Args:
            original_file_path: Path to batched_ai_jobs_* file before stage 2 deduplication
            
        Returns:
            DataFrame with all original jobs including those removed by stage 2 deduplication
        """
        if original_file_path is None:
            # Auto-detect the most recent batched_ai_jobs file
            # Use the Stage 2 deduplicated output as the "original" dataset 
            # This ensures we only work with AI jobs that passed false positive filtering
            dedup_file = os.path.join(self.data_dir, "ai_development_deduplicated.csv")
            if os.path.exists(dedup_file):
                original_file_path = dedup_file
            else:
                # Fallback to batched files
                batched_file = os.path.join(self.data_dir, "batched_ai_jobs.csv")
                if os.path.exists(batched_file):
                    original_file_path = batched_file
                else:
                    # Final fallback to timestamped files
                    batched_files = list(Path(self.data_dir).glob("batched_ai_jobs_*.csv"))
                    if not batched_files:
                        raise FileNotFoundError("Could not find AI jobs file. Please specify original_file_path parameter.")
                    original_file_path = str(sorted(batched_files)[-1])
        
        logger.info(f"Loading original full dataset from: {original_file_path}")
        
        # Only load columns needed for exposure calculations
        exposure_columns = [
            'uid', 'title', 'company_name', 'x28_occupations', 
            'x28_industries', 'tst_created', 'matched_keywords'
        ]
        
        original_df = pd.read_csv(original_file_path, usecols=exposure_columns)
        
        # Extract year from date strings
        def extract_year_from_string(date_str):
            try:
                if pd.isna(date_str) or date_str == '':
                    return None
                year = int(str(date_str)[:4])
                if 2000 <= year <= 2030:
                    return year
                else:
                    return None
            except:
                return None
        
        original_df['year'] = original_df['tst_created'].apply(extract_year_from_string)
        original_df = original_df[original_df['year'].notna()].copy()  # Filter out invalid years
        original_df['year'] = original_df['year'].astype(int)
        
        # Parse full datetime
        original_df['tst_created'] = pd.to_datetime(original_df['tst_created'], errors='coerce')
        
        # Rename uid to job_uid for consistency
        original_df.rename(columns={'uid': 'job_uid'}, inplace=True)
        
        logger.info(f"Loaded {len(original_df):,} original jobs from full dataset")
        logger.info(f"Unique companies: {original_df['company_name'].nunique():,}")
        logger.info(f"Year range: {original_df['year'].min()} - {original_df['year'].max()}")
        
        return original_df
    
    def load_stage2_deduplication_mapping(self, stage2_mapping_file: str = None) -> pd.DataFrame:
        """
        Load Stage 2 deduplication mapping to recover jobs removed by similarity deduplication.
        Only loads the UID columns needed for mapping.
        
        Args:
            stage2_mapping_file: Path to similar_duplicates_removed_* file
            
        Returns:
            DataFrame with kept_uid → removed_uid mapping for Stage 2 recovery
        """
        if stage2_mapping_file is None:
            # Auto-detect the most recent similar_duplicates_removed file
            stage2_files = list(Path(self.data_dir).glob("similar_duplicates_removed_*.csv"))
            if not stage2_files:
                # Check for the standard filename
                standard_file = os.path.join(self.data_dir, "similar_duplicates_removed.csv")
                if os.path.exists(standard_file):
                    stage2_mapping_file = standard_file
                else:
                    # Return None to indicate no Stage 2 mapping available
                    logger.warning("No Stage 2 deduplication mapping found. Skipping deduplication recovery.")
                    return None
            else:
                stage2_mapping_file = str(sorted(stage2_files)[-1])
        
        logger.info(f"Loading Stage 2 deduplication mapping from: {stage2_mapping_file}")
        
        # Only load the UID columns needed for mapping (memory efficient)
        mapping_columns = ['kept_uid', 'removed_uid']
        
        stage2_mapping = pd.read_csv(stage2_mapping_file, usecols=mapping_columns)
        
        logger.info(f"Loaded {len(stage2_mapping):,} Stage 2 deduplication mappings")
        logger.info(f"Unique kept jobs: {stage2_mapping['kept_uid'].nunique():,}")
        logger.info(f"Unique removed jobs: {stage2_mapping['removed_uid'].nunique():,}")
        
        return stage2_mapping
    
    def expand_job_app_mapping_to_full_dataset(self, job_app_mapping: pd.DataFrame, 
                                           original_jobs: pd.DataFrame,
                                           stage2_mapping: pd.DataFrame) -> pd.DataFrame:
        """
        Expand the job→app mapping but only JOIN using kept UIDs (Stage-2 canonical IDs).
        We still keep a provenance column listing all original UIDs that had the app.
        """
        logger.info("Expanding job-app mapping with Stage-2 UID normalization (kept UIDs only for joins)")

        # --- Build removed -> kept map
        if stage2_mapping is not None and {'kept_uid','removed_uid'}.issubset(stage2_mapping.columns):
            rem2kept = stage2_mapping.set_index('removed_uid')['kept_uid'].to_dict()
        else:
            rem2kept = {}

        def to_kept(uid: str) -> str:
            return rem2kept.get(uid, uid)

        # --- Parse Stage 4 mapping
        jam = job_app_mapping.copy()
        jam['job_uids_list'] = jam['job_uids'].fillna('').astype(str).apply(
            lambda s: [u.strip() for u in s.split('|') if u.strip()]
        )

        # Keep full provenance (removed + kept as Stage 4 saw them)
        jam['job_uids_all'] = jam['job_uids_list']

        # Canonical kept UIDs for actual joins
        jam['job_uids_kept'] = jam['job_uids_list'].apply(lambda L: sorted(set(to_kept(u) for u in L)))

        # --- Explode kept uids for join
        expanded = jam[['app_text', 'job_uids_kept', 'first_occurrence_tst_created']].explode('job_uids_kept')
        expanded = expanded.rename(columns={'job_uids_kept': 'job_uid'})

        # --- Join to (dedupbed) company data using only kept UIDs
        merged = expanded.merge(
            original_jobs, how='left', on='job_uid', validate='many_to_one'
        )

        # Warn if any KEPT ids are missing from company data
        missing_kept = merged[merged['company_name'].isna()]['job_uid'].unique().tolist()
        if missing_kept:
            logger.warning(f"{len(missing_kept)} kept UIDs not found in company data. First few: {missing_kept[:10]}")

        # Attach provenance as a pipe-joined string
        jam['provenance_job_uids'] = jam['job_uids_all'].apply(lambda L: '|'.join(L))
        prov = jam[['app_text', 'provenance_job_uids']].drop_duplicates()

        merged = merged.merge(prov, on='app_text', how='left')

        # Keep only rows that successfully joined (unchanged)
        out = merged.dropna(subset=['company_name']).copy()
        out = out[['app_text', 'job_uid', 'company_name', 'x28_occupations', 'x28_industries',
                'year', 'tst_created', 'first_occurrence_tst_created', 'provenance_job_uids']]

        logger.info(f"Final expansion (kept UIDs only): {out['job_uid'].nunique():,} jobs, {len(out):,} app-job rows")
        return out


    def step2_calculate_firm_task_exposure(self, task_app_matches: pd.DataFrame, 
                                         job_app_mapping: pd.DataFrame,
                                         original_jobs: pd.DataFrame,
                                         stage2_mapping_file: str = None) -> pd.DataFrame:
        """
        Step 2: Calculate task exposure at firm level using both Hampole and Binary variants.
        Now includes ALL jobs from original dataset, not just deduplicated subset.
        
        Args:
            task_app_matches: Task-application matches from Step 1
            job_app_mapping: AI app → job UID mapping (deduplicated)
            original_jobs: Full original dataset before deduplication
            
        Returns:
            DataFrame with task-firm exposure probabilities (both variants)
        """
        logger.info(f"Step 2: Calculating task exposure at firm level (time_invariant={self.time_invariant}, occupation_exposure={self.occupation_exposure})...")
        logger.info(f"Using expanded dataset to include all jobs removed by deduplication...")
        
        # Load Stage 2 deduplication mapping (if available)
        stage2_mapping = self.load_stage2_deduplication_mapping(stage2_mapping_file)
        
        # Always expand job-app mapping to include all original jobs 
        # The expand method handles both cases: with and without Stage 2 mapping
        expanded_job_app_mapping = self.expand_job_app_mapping_to_full_dataset(
            job_app_mapping, original_jobs, stage2_mapping
        )
        
        # Join task-app matches with expanded job mapping
        task_job_matches = task_app_matches.merge(
            expanded_job_app_mapping, 
            left_on='app_text', 
            right_on='app_text', 
            how='left'
        )
        
        # The expanded mapping already contains company and time info, so we use it directly
        task_firm_matches = task_job_matches.copy()
        
        logger.info(f"Mapped {len(task_firm_matches):,} task-app-firm relationships")
        # Only log company info if company_name column exists
        if 'company_name' in task_firm_matches.columns:
            logger.info(f"Unique firms: {task_firm_matches['company_name'].nunique():,}")
        else:
            logger.info("Company name column not available yet")
        
        # CRITICAL VALIDATION: Ensure data consistency across the pipeline
        logger.info("="*50)
        logger.info("DATA CONSISTENCY VALIDATION")
        logger.info("="*50)

        # Build removed->kept map again (same as above)
        if stage2_mapping is not None and {'kept_uid','removed_uid'}.issubset(stage2_mapping.columns):
            rem2kept = stage2_mapping.set_index('removed_uid')['kept_uid'].to_dict()
        else:
            rem2kept = {}
        to_kept = lambda u: rem2kept.get(u, u)

        # Normalize Stage-4 UID list to KEPT UIDs for a fair comparison
        stage4_jobs_kept_set = set()
        for row in job_app_mapping.itertuples():
            for u in str(row.job_uids).split('|'):
                u = u.strip()
                if not u:
                    continue
                stage4_jobs_kept_set.add(to_kept(u))

        stage4_jobs = len(stage4_jobs_kept_set)

        expanded_jobs = expanded_job_app_mapping['job_uid'].nunique()
        final_jobs_with_tasks = task_firm_matches['job_uid'].nunique()
        
        logger.info(f"Jobs with AI apps (Stage 4 mapping): {stage4_jobs:,}")
        logger.info(f"Jobs after deduplication recovery: {expanded_jobs:,}")
        logger.info(f"Jobs in final task-firm matches: {final_jobs_with_tasks:,}")
        
        # Validation checks
        if expanded_jobs < stage4_jobs:
            logger.warning(f"WARNING: Lost {stage4_jobs - expanded_jobs:,} jobs during expansion. "
                          f"This could indicate missing Stage 2 mappings.")
        
        if final_jobs_with_tasks != expanded_jobs:
            logger.warning(f"WARNING: Job count mismatch between expanded mapping ({expanded_jobs:,}) "
                          f"and final task matches ({final_jobs_with_tasks:,}). "
                          f"This could indicate issues in task-app matching.")
        
        # Check for any jobs accidentally INCLUDED without AI-task provenance
        allowed_jobs = set(expanded_job_app_mapping['job_uid'].dropna().unique())
        final_jobs   = set(task_firm_matches['job_uid'].dropna().unique())

        # Jobs that appear in final matches but weren't in the allowed mapping
        accidental_inclusions = final_jobs - allowed_jobs
        if accidental_inclusions:
            logger.error(
                f"CRITICAL ERROR: {len(accidental_inclusions)} job(s) appear in final matches "
                f"but are not present in the expanded job-app mapping (unexpected inclusion). "
                f"Examples: {list(sorted(accidental_inclusions))[:10]}"
            )
            raise ValueError("Data integrity violation: unexpected jobs included in exposure calculations")

        # Optional: jobs we expected (from mapping) but lost during the merge
        missing_expected = allowed_jobs - final_jobs
        if missing_expected:
            logger.warning(
                f"{len(missing_expected)} job(s) from the expanded mapping are missing in final matches. "
                f"First few: {list(sorted(missing_expected))[:10]}"
            )
        else:
            logger.info("✅ VALIDATION PASSED: Final jobs are a clean subset of the expanded mapping")

        
        # Get all firms, years, apps, and tasks for time-invariant mode
        all_firms = task_firm_matches['company_name'].unique()
        all_years = sorted(task_firm_matches['year'].unique())
        all_apps = task_app_matches['app_text'].unique()  # All apps across all firms
        all_tasks = task_app_matches['onet_task_id'].unique()  # All exposed tasks
        
        if self.time_invariant:
            logger.info(f"Time-invariant mode: Each firm exposed to all AI apps it has ever used, across all years")
            logger.info(f"Processing {len(all_firms)} firms across {len(all_years)} years")
        else:
            # Calculate firm-year-level AI app counts (N_f,t) - count by firm AND year
            firm_year_app_counts = task_firm_matches.groupby(['company_name', 'year'])['app_text'].nunique().reset_index()
            firm_year_app_counts.rename(columns={'app_text': 'n_ai_apps_firm_year'}, inplace=True)
            
            logger.info(f"Time-variant mode: AI apps per firm-year - Mean: {firm_year_app_counts['n_ai_apps_firm_year'].mean():.1f}, "
                       f"Max: {firm_year_app_counts['n_ai_apps_firm_year'].max():,}")
            logger.info(f"Unique firm-year combinations: {len(firm_year_app_counts):,}")
        
        # Calculate task exposure for each task-firm-year combination
        task_firm_exposure = []
        
        if self.time_invariant:
            # Time-invariant: Each firm-year is exposed to all tasks that match ANY app THAT FIRM has ever used
            for firm in all_firms:
                # Get ALL apps this specific firm has ever used across all years
                firm_apps_ever = task_firm_matches[task_firm_matches['company_name'] == firm]['app_text'].unique()
                n_firm_apps_ever = len(firm_apps_ever)
                
                if n_firm_apps_ever == 0:
                    continue  # Skip firms with no AI apps
                
                # Get all tasks that match any of this firm's apps (across all time)
                firm_app_task_matches = task_app_matches[task_app_matches['app_text'].isin(firm_apps_ever)]
                firm_exposed_tasks = firm_app_task_matches['onet_task_id'].unique()
                
                for year in all_years:
                    for task_id in firm_exposed_tasks:
                        # Count how many of THIS FIRM's apps (ever) match this task
                        task_matches_for_firm = firm_app_task_matches[firm_app_task_matches['onet_task_id'] == task_id]
                        n_matches_firm = len(task_matches_for_firm['app_text'].unique())
                        
                        # Hampole variant: share of THIS FIRM's apps (ever) that match this task
                        hampole_exposure = n_matches_firm / n_firm_apps_ever if n_firm_apps_ever > 0 else 0
                        
                        # Binary variant: 1 if any of this firm's apps (ever) matches, 0 otherwise
                        binary_exposure = 1 if n_matches_firm >= 1 else 0
                        
                        task_firm_exposure.append({
                            'company_name': firm,
                            'year': year,
                            'onet_task_id': task_id,
                            'n_ai_apps_firm_year': n_firm_apps_ever,  # Apps this firm has ever used
                            'n_task_matches_firm_year': n_matches_firm,
                            'hampole_task_exposure': hampole_exposure,
                            'binary_task_exposure': binary_exposure
                        })
        else:
            # Time-variant: Original logic - each firm-year only exposed to their own apps
            for (firm, year), group in task_firm_matches.groupby(['company_name', 'year']):
                firm_apps = group['app_text'].unique()
                n_apps_firm_year = len(firm_apps)
                
                # Get all unique tasks that have matches with any AI app at this firm-year
                tasks_at_firm_year = group['onet_task_id'].unique()
                
                for task_id in tasks_at_firm_year:
                    # Count how many AI apps at this firm-year match this task
                    task_matches_at_firm_year = group[group['onet_task_id'] == task_id]
                    n_matches = len(task_matches_at_firm_year['app_text'].unique())
                    
                    # Hampole variant: share of apps that match this task
                    hampole_exposure = n_matches / n_apps_firm_year if n_apps_firm_year > 0 else 0
                    
                    # Binary variant: 1 if any app matches, 0 otherwise
                    binary_exposure = 1 if n_matches >= 1 else 0
                    
                    task_firm_exposure.append({
                        'company_name': firm,
                        'year': year,
                        'onet_task_id': task_id,
                        'n_ai_apps_firm_year': n_apps_firm_year,
                        'n_task_matches_firm_year': n_matches,
                        'hampole_task_exposure': hampole_exposure,
                        'binary_task_exposure': binary_exposure
                    })
        
        exposure_df = pd.DataFrame(task_firm_exposure)
        
        # Apply occupation-level exposure if specified
        if self.occupation_exposure != "none":
            logger.info(f"Applying occupation-level exposure ({self.occupation_exposure}) - overriding firm-level logic...")
            exposure_df = self._apply_occupation_exposure(exposure_df, task_app_matches, task_firm_matches)
        
        logger.info(f"Calculated task-firm-year exposure: {len(exposure_df):,} task-firm-year combinations")
        if len(exposure_df) > 0:
            logger.info(f"  Hampole exposure range: {exposure_df['hampole_task_exposure'].min():.3f} - {exposure_df['hampole_task_exposure'].max():.3f}")
            logger.info(f"  Binary exposure - firm-years with exposed tasks: {(exposure_df['binary_task_exposure'] > 0).sum():,}")
            logger.info(f"  Unique firms: {exposure_df['company_name'].nunique():,}")
            logger.info(f"  Unique years: {sorted(exposure_df['year'].unique())}")
        
        return exposure_df
    
    def _apply_occupation_exposure(self, firm_exposure_df: pd.DataFrame, 
                                 task_app_matches: pd.DataFrame,
                                 task_firm_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Apply occupation-level exposure logic, overriding firm-level exposure.
        
        Args:
            firm_exposure_df: Firm-level exposure from previous calculation
            task_app_matches: Task-application matches from Step 1
            task_firm_matches: Task-app-firm relationships with years
            
        Returns:
            DataFrame with occupation-level exposure applied
        """
        logger.info(f"Occupation exposure mode: {self.occupation_exposure}")
        
        # Get all firms and years from the data
        all_firms = task_firm_matches['company_name'].unique()
        all_years = sorted(task_firm_matches['year'].unique())
        all_apps = task_app_matches['app_text'].unique()
        all_tasks = task_app_matches['onet_task_id'].unique()
        
        # Find first appearance year of each AI application (for time-variant mode)
        app_first_year = {}
        if self.occupation_exposure == "time-variant":
            for app in all_apps:
                app_years = task_firm_matches[task_firm_matches['app_text'] == app]['year'].dropna()
                if len(app_years) > 0:
                    app_first_year[app] = app_years.min()
                else:
                    app_first_year[app] = float('inf')  # Never appeared
        
        logger.info(f"Occupation exposure: All firms exposed to all {len(all_apps)} AI applications")
        logger.info(f"Processing {len(all_firms)} firms across {len(all_years)} years")
        
        # Create new occupation-level exposure
        occupation_task_exposure = []
        
        for firm in all_firms:
            for year in all_years:
                for task_id in all_tasks:
                    # Count how many apps (across ALL firms) match this task
                    task_matches_global = task_app_matches[task_app_matches['onet_task_id'] == task_id]
                    matching_apps = task_matches_global['app_text'].unique()
                    
                    if self.occupation_exposure == "time-invariant":
                        # Time-invariant: All firms exposed to all apps for all time
                        available_apps = matching_apps
                        
                    elif self.occupation_exposure == "time-variant":
                        # Time-variant: Only apps that have appeared by this year
                        available_apps = [app for app in matching_apps if app_first_year.get(app, float('inf')) <= year]
                    
                    n_available_apps = len(available_apps)
                    n_matches = len(available_apps)  # All available apps match this task
                    
                    if n_matches > 0:  # Only include if there are matching apps
                        # Hampole variant: share of ALL available apps that match this task
                        hampole_exposure = n_matches / len(all_apps) if len(all_apps) > 0 else 0
                        
                        # Binary variant: 1 if any app matches, 0 otherwise
                        binary_exposure = 1 if n_matches >= 1 else 0
                        
                        occupation_task_exposure.append({
                            'company_name': firm,
                            'year': year,
                            'onet_task_id': task_id,
                            'n_ai_apps_firm_year': len(all_apps),  # Total universe of apps
                            'n_task_matches_firm_year': n_matches,
                            'hampole_task_exposure': hampole_exposure,
                            'binary_task_exposure': binary_exposure
                        })
        
        new_exposure_df = pd.DataFrame(occupation_task_exposure)
        logger.info(f"Occupation-level exposure created {len(new_exposure_df):,} task-firm-year combinations")
        
        return new_exposure_df
    
    def step3_calculate_occupation_firm_exposure(self, task_firm_exposure: pd.DataFrame,
                                               task_statements: pd.DataFrame,
                                               task_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Step 3: Aggregate task-level exposure to occupation × firm × year level using O*NET task weights.
        
        Args:
            task_firm_exposure: Task-firm-year exposure from Step 2
            task_statements: O*NET task statements (for task → occupation mapping)
            task_ratings: O*NET task importance ratings (for weights)
            
        Returns:
            DataFrame with occupation-firm-year exposure scores (both variants)
        """
        logger.info("Step 3: Aggregating to occupation × firm × year exposure...")
        
        # Map tasks to occupations and add importance weights
        task_onet_mapping = task_statements[['Task ID', 'O*NET-SOC Code']].copy()
        task_onet_mapping.rename(columns={
            'Task ID': 'onet_task_id',
            'O*NET-SOC Code': 'onet_code'
        }, inplace=True)
        
        # Add task importance weights - handle both raw and processed formats
        if 'importance_weight' in task_ratings.columns:
            # Already processed format
            task_weights = task_ratings[['O*NET-SOC Code', 'Task ID', 'importance_weight']].copy()
        elif 'Data Value' in task_ratings.columns:
            # Raw format - process it
            task_ratings_processed = task_ratings[
                (task_ratings['Scale ID'] == 'IM') & 
                (task_ratings['Scale Name'] == 'Importance')
            ].copy()
            task_weights = task_ratings_processed[['O*NET-SOC Code', 'Task ID', 'Data Value']].copy()
            task_weights.rename(columns={'Data Value': 'importance_weight'}, inplace=True)
        else:
            raise ValueError("Task ratings must have either 'importance_weight' or 'Data Value' column")
        task_weights.rename(columns={
            'O*NET-SOC Code': 'onet_code', 
            'Task ID': 'onet_task_id'
        }, inplace=True)
        
        # Join task exposure with occupation and weight information
        task_exposure_weighted = task_firm_exposure.merge(
            task_onet_mapping, on='onet_task_id', how='left'
        ).merge(
            task_weights, on=['onet_code', 'onet_task_id'], how='left'
        )
        
        # Fill missing weights with 3.0
        task_exposure_weighted['importance_weight'] = task_exposure_weighted['importance_weight'].fillna(3.0)
        
        logger.info(f"Mapped {len(task_exposure_weighted):,} task-firm-year-occupation relationships")
        
        # Calculate weighted occupation-firm-year exposure scores
        def calc_weighted_occupation_exposure(group):
            weights = group['importance_weight']
            total_weight = weights.sum()
            
            # Hampole variant: weighted average of task exposures
            hampole_weighted = (group['hampole_task_exposure'] * weights).sum() / total_weight if total_weight > 0 else 0
            
            # Binary variant: weighted average of binary task exposures
            binary_weighted = (group['binary_task_exposure'] * weights).sum() / total_weight if total_weight > 0 else 0
            
            return pd.Series({
                'hampole_occupation_exposure': hampole_weighted,
                'binary_occupation_exposure': binary_weighted,
                'total_tasks_occupation': len(group),
                'total_importance_weight': total_weight,
                'n_ai_apps_firm_year': group['n_ai_apps_firm_year'].iloc[0]  # Same for all tasks at this firm-year
            })
        
        occupation_firm_exposure = task_exposure_weighted.groupby(['onet_code', 'company_name', 'year']).apply(
            calc_weighted_occupation_exposure
        ).reset_index()
        
        logger.info(f"Calculated occupation-firm-year exposure: {len(occupation_firm_exposure):,} occupation-firm-year combinations")
        logger.info(f"  Unique occupations: {occupation_firm_exposure['onet_code'].nunique():,}")
        logger.info(f"  Unique firms: {occupation_firm_exposure['company_name'].nunique():,}")
        logger.info(f"  Unique years: {sorted(occupation_firm_exposure['year'].unique())}")
        logger.info(f"  Hampole exposure range: {occupation_firm_exposure['hampole_occupation_exposure'].min():.3f} - {occupation_firm_exposure['hampole_occupation_exposure'].max():.3f}")
        logger.info(f"  Binary exposure range: {occupation_firm_exposure['binary_occupation_exposure'].min():.3f} - {occupation_firm_exposure['binary_occupation_exposure'].max():.3f}")
        
        return occupation_firm_exposure
    
    def step4_apply_ai_intensity_adjustment(self, occupation_firm_exposure: pd.DataFrame) -> pd.DataFrame:
        """
        Step 4: Apply AI intensity adjustment using log(1 + N_apps) scaling.
        
        Args:
            occupation_firm_exposure: Occupation-firm exposure from Step 3
            
        Returns:
            DataFrame with final AI exposure average scores
        """
        logger.info("Step 4: Applying AI intensity adjustment...")
        
        # Calculate log(1 + N_apps) adjustment factor using firm-year app counts
        final_exposure = occupation_firm_exposure.copy()
        final_exposure['log_ai_intensity'] = np.log(1 + final_exposure['n_ai_apps_firm_year'])
        
        # Apply intensity adjustment to both variants
        final_exposure['hampole_ai_exposure_avg'] = (
            final_exposure['hampole_occupation_exposure'] * final_exposure['log_ai_intensity']
        )
        final_exposure['binary_ai_exposure_avg'] = (
            final_exposure['binary_occupation_exposure'] * final_exposure['log_ai_intensity']
        )
        
        logger.info(f"Applied AI intensity adjustment to {len(final_exposure):,} occupation-firm-year combinations")
        logger.info(f"  Log intensity range: {final_exposure['log_ai_intensity'].min():.3f} - {final_exposure['log_ai_intensity'].max():.3f}")
        logger.info(f"  Hampole final exposure range: {final_exposure['hampole_ai_exposure_avg'].min():.3f} - {final_exposure['hampole_ai_exposure_avg'].max():.3f}")
        logger.info(f"  Binary final exposure range: {final_exposure['binary_ai_exposure_avg'].min():.3f} - {final_exposure['binary_ai_exposure_avg'].max():.3f}")
        
        return final_exposure
    
    def load_company_mapping(self) -> pd.DataFrame:
        """
        Load company_id to company_name mapping for Stage 6 compatibility.
        
        Returns:
            DataFrame with company_id and company_name columns
        """
        company_mapping_file = os.path.join(self.data_dir, "company_mapping.csv")
        
        if not os.path.exists(company_mapping_file):
            raise FileNotFoundError(f"Company mapping file required but not found: {company_mapping_file}")
        
        logger.info(f"Loading company mapping from: {company_mapping_file}")
        company_mapping = pd.read_csv(company_mapping_file)
        
        # Validate required columns
        if 'company_id' not in company_mapping.columns or 'company_name' not in company_mapping.columns:
            raise ValueError("Company mapping file must contain 'company_id' and 'company_name' columns")
        
        # Clean and convert types
        company_mapping['company_id'] = company_mapping['company_id'].astype(str)
        company_mapping['company_name'] = company_mapping['company_name'].astype(str)
        
        logger.info(f"✅ Loaded {len(company_mapping):,} company mappings")
        return company_mapping

    def crosswalk_onet_to_isco(self, onet_exposure: pd.DataFrame, 
                              esco_crosswalk: pd.DataFrame,
                              isco_titles: pd.DataFrame,
                              company_mapping: pd.DataFrame = None) -> pd.DataFrame:
        """
        Crosswalk O*NET occupation-firm exposure to ISCO-08 using ESCO mapping.
        
        Args:
            onet_exposure: O*NET occupation-firm exposure scores
            esco_crosswalk: ESCO → O*NET crosswalk
            isco_titles: ISCO-08 titles
            
        Returns:
            DataFrame with ISCO-08 occupation-firm exposure scores
        """
        logger.info("Crosswalking O*NET → ISCO-08...")
        
        # Add company names if not already present and mapping is provided
        if company_mapping is not None and 'company_name' not in onet_exposure.columns:
            logger.info("Adding company names to ONET exposure data...")
            onet_exposure = onet_exposure.merge(company_mapping, on='company_id', how='left')
            missing_companies = onet_exposure['company_name'].isna().sum()
            if missing_companies > 0:
                logger.warning(f"{missing_companies} company_ids could not be mapped to company names")
        
        # Map O*NET codes to ISCO codes via ESCO crosswalk
        onet_to_isco = onet_exposure.merge(
            esco_crosswalk[['onet_code', 'isco08_4d']].drop_duplicates(),
            on='onet_code',
            how='inner'
        )
        
        logger.info(f"Mapped {len(onet_to_isco):,} O*NET-firm → ISCO-firm relationships")
        logger.info(f"  Unique ISCO codes: {onet_to_isco['isco08_4d'].nunique():,}")
        logger.info(f"  Unique firms: {onet_to_isco['company_name'].nunique():,}")
        
        # Aggregate multiple O*NET codes mapping to same ISCO-firm-year combination
        # Use mean aggregation for exposure scores, sum for counts/weights
        isco_firm_exposure = onet_to_isco.groupby(['isco08_4d', 'company_name', 'year'], as_index=False).agg({
            'hampole_ai_exposure_avg': 'mean',
            'binary_ai_exposure_avg': 'mean',
            'hampole_occupation_exposure': 'mean',
            'binary_occupation_exposure': 'mean',
            'log_ai_intensity': 'mean',  # Should be same for firm-year
            'n_ai_apps_firm_year': 'mean',  # Should be same for firm-year
            'total_tasks_occupation': 'sum',
            'total_importance_weight': 'sum',
            'onet_code': 'nunique'  # Count how many O*NET codes contribute
        })
        
        # Rename the onet_code count column
        isco_firm_exposure.rename(columns={'onet_code': 'n_onet_codes_contributing'}, inplace=True)
        
        # Add ISCO titles
        isco_firm_exposure = isco_firm_exposure.merge(isco_titles, on='isco08_4d', how='left')
        
        # Reorder columns
        cols = ['isco08_4d', 'isco08_title', 'company_name', 'year'] + [col for col in isco_firm_exposure.columns 
                                                                        if col not in ['isco08_4d', 'isco08_title', 'company_name', 'year']]
        isco_firm_exposure = isco_firm_exposure[cols]
        
        logger.info(f"Final ISCO-08 firm-year exposure dataset: {len(isco_firm_exposure):,} ISCO-firm-year combinations")
        logger.info(f"  Unique ISCO occupations: {isco_firm_exposure['isco08_4d'].nunique():,}")
        logger.info(f"  Unique firms: {isco_firm_exposure['company_name'].nunique():,}")
        logger.info(f"  Unique years: {sorted(isco_firm_exposure['year'].unique())}")
        logger.info(f"  ISCO titles coverage: {isco_firm_exposure['isco08_title'].notna().sum()}/{len(isco_firm_exposure)} ({isco_firm_exposure['isco08_title'].notna().mean()*100:.1f}%)")
        
        return isco_firm_exposure
    
    def create_task_exposure_table(self, exposed_tasks: pd.DataFrame, 
                                 task_statements: pd.DataFrame,
                                 task_ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Create task-level exposure table with O*NET codes and weights.
        
        Args:
            exposed_tasks: Exposed tasks from Stage 4
            task_statements: O*NET task statements  
            task_ratings: O*NET task importance ratings
            
        Returns:
            DataFrame with columns: onet_code, onet_task_id, task_text, exposed, weight
        """
        logger.info("Creating task-level exposure table...")
        
        # Get all unique task IDs from task statements
        all_tasks = task_statements[['O*NET-SOC Code', 'Task ID', 'Task']].copy()
        all_tasks.rename(columns={
            'O*NET-SOC Code': 'onet_code',
            'Task ID': 'onet_task_id', 
            'Task': 'task_text'
        }, inplace=True)
        
        # Mark exposed tasks (from Stage 4)
        exposed_task_ids = set(exposed_tasks['onet_task_id'].unique())
        all_tasks['exposed'] = all_tasks['onet_task_id'].isin(exposed_task_ids).astype(int)
        
        # Merge with importance weights
        task_with_weights = all_tasks.merge(
            task_ratings[['O*NET-SOC Code', 'Task ID', 'importance_weight']],
            left_on=['onet_code', 'onet_task_id'],
            right_on=['O*NET-SOC Code', 'Task ID'],
            how='left'
        )
        
        # Fill missing weights with 1.0 (uniform weight)
        task_with_weights['weight'] = task_with_weights['importance_weight'].fillna(3.0)
        
        # Clean up columns
        task_exposure_df = task_with_weights[['onet_code', 'onet_task_id', 'task_text', 'exposed', 'weight']].copy()
        
        logger.info(f"Created task exposure table: {len(task_exposure_df):,} tasks")
        logger.info(f"  Exposed tasks: {task_exposure_df['exposed'].sum():,}")
        logger.info(f"  Non-exposed tasks: {(task_exposure_df['exposed'] == 0).sum():,}")
        logger.info(f"  Tasks with importance weights: {task_exposure_df['weight'].notna().sum():,}")
        
        return task_exposure_df
    
    def aggregate_tasks_to_onet_occupations(self, task_exposure: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate task-level exposure to O*NET occupation-level exposure scores.
        
        Args:
            task_exposure: Task-level exposure table
            
        Returns:
            DataFrame with columns: onet_code, exposure_score_onet, n_tasks, n_exposed_tasks
        """
        logger.info("Aggregating tasks → O*NET occupations...")
        
        # Calculate weighted exposure score for each occupation
        def calc_weighted_exposure(group):
            weights = group['weight']
            exposed = group['exposed']
            return (exposed * weights).sum() / weights.sum()
        
        onet_exposure = task_exposure.groupby('onet_code', as_index=False).agg({
            'exposed': calc_weighted_exposure,
            'onet_task_id': 'count',  # Total tasks
            'exposed': 'sum'  # Exposed tasks
        })
        
        # Fix the column naming issue by handling separately
        exposed_counts = task_exposure.groupby('onet_code')['exposed'].sum().reset_index()
        exposed_counts.rename(columns={'exposed': 'n_exposed_tasks'}, inplace=True)
        
        task_counts = task_exposure.groupby('onet_code')['onet_task_id'].count().reset_index()
        task_counts.rename(columns={'onet_task_id': 'n_tasks'}, inplace=True)
        
        exposure_scores = task_exposure.groupby('onet_code').apply(calc_weighted_exposure).reset_index()
        exposure_scores.rename(columns={0: 'exposure_score_onet'}, inplace=True)
        
        # Combine all metrics
        onet_exposure = exposure_scores.merge(task_counts, on='onet_code').merge(exposed_counts, on='onet_code')
        
        logger.info(f"Created O*NET occupation exposure: {len(onet_exposure):,} occupations")
        logger.info(f"  Exposure score range: {onet_exposure['exposure_score_onet'].min():.3f} - {onet_exposure['exposure_score_onet'].max():.3f}")
        logger.info(f"  Mean exposure score: {onet_exposure['exposure_score_onet'].mean():.3f}")
        logger.info(f"  Occupations with any exposure: {(onet_exposure['exposure_score_onet'] > 0).sum():,}")
        
        return onet_exposure
    
    def map_onet_to_isco(self, onet_exposure: pd.DataFrame, 
                        esco_crosswalk: pd.DataFrame) -> pd.DataFrame:
        """
        Map O*NET occupation exposure to ISCO-08 codes via ESCO crosswalk.
        
        Args:
            onet_exposure: O*NET occupation-level exposure
            esco_crosswalk: ESCO → O*NET crosswalk
            
        Returns:
            DataFrame with O*NET → ISCO mappings
        """
        logger.info("Mapping O*NET occupations → ISCO-08...")
        
        # Join O*NET exposure with crosswalk
        onet_to_isco = onet_exposure.merge(
            esco_crosswalk[['onet_code', 'isco08_4d']].drop_duplicates(),
            on='onet_code',
            how='inner'
        )
        
        logger.info(f"Mapped {len(onet_to_isco):,} O*NET → ISCO relationships")
        logger.info(f"  Unique O*NET codes mapped: {onet_to_isco['onet_code'].nunique():,}")
        logger.info(f"  Unique ISCO codes receiving mappings: {onet_to_isco['isco08_4d'].nunique():,}")
        
        # Check many-to-many relationships
        onet_to_isco_counts = onet_to_isco.groupby('onet_code')['isco08_4d'].nunique()
        isco_to_onet_counts = onet_to_isco.groupby('isco08_4d')['onet_code'].nunique()
        
        logger.info(f"  O*NET codes mapping to multiple ISCOs: {(onet_to_isco_counts > 1).sum():,}")
        logger.info(f"  ISCO codes receiving multiple O*NETs: {(isco_to_onet_counts > 1).sum():,}")
        
        return onet_to_isco
    
    def aggregate_onet_to_isco_exposure(self, onet_to_isco: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate O*NET exposure scores to ISCO-08 level.
        
        Args:
            onet_to_isco: O*NET → ISCO mapped exposure data
            
        Returns:
            DataFrame with ISCO-08 exposure scores
        """
        logger.info("Aggregating O*NET → ISCO-08 exposure...")
        
        if self.aggregation_method == "mean":
            # Simple mean across O*NET codes mapping to each ISCO
            isco_exposure = onet_to_isco.groupby('isco08_4d', as_index=False).agg({
                'exposure_score_onet': ['mean', 'min', 'max', 'std'],
                'onet_code': 'nunique',
                'n_tasks': 'sum',
                'n_exposed_tasks': 'sum'
            })
            
            # Flatten column names
            isco_exposure.columns = [
                'isco08_4d', 'exposure_score_isco', 'exposure_score_isco_min', 
                'exposure_score_isco_max', 'exposure_score_isco_std',
                'n_onet_per_isco', 'total_tasks', 'total_exposed_tasks'
            ]
            
        else:
            raise ValueError(f"Aggregation method '{self.aggregation_method}' not implemented")
        
        # Fill NaN std with 0 (single O*NET mapping)
        isco_exposure['exposure_score_isco_std'] = isco_exposure['exposure_score_isco_std'].fillna(0)
        
        # Sort by exposure score
        isco_exposure = isco_exposure.sort_values('exposure_score_isco', ascending=False).reset_index(drop=True)
        
        logger.info(f"Created ISCO-08 exposure scores: {len(isco_exposure):,} occupations")
        logger.info(f"  Exposure score range: {isco_exposure['exposure_score_isco'].min():.3f} - {isco_exposure['exposure_score_isco'].max():.3f}")
        logger.info(f"  Mean exposure score: {isco_exposure['exposure_score_isco'].mean():.3f}")
        logger.info(f"  ISCO codes with any exposure: {(isco_exposure['exposure_score_isco'] > 0).sum():,}")
        logger.info(f"  Mean O*NET codes per ISCO: {isco_exposure['n_onet_per_isco'].mean():.1f}")
        
        return isco_exposure
    
    def _generate_config_suffix(self) -> str:
        """
        Generate configuration suffix for output filenames based on pipeline settings.
        
        Returns:
            String suffix describing the configuration
        """
        from datetime import datetime
        
        suffix_parts = []
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        suffix_parts.append(timestamp)
        
        # Add time invariant setting
        if self.time_invariant:
            suffix_parts.append("time_invariant")
        else:
            suffix_parts.append("time_variant")
        
        # Add occupation exposure mode
        if self.occupation_exposure != "none":
            suffix_parts.append(f"occ_{self.occupation_exposure}")
        
        # Add aggregation method if not default
        if self.aggregation_method != "mean":
            suffix_parts.append(f"agg_{self.aggregation_method}")
        
        return "_" + "_".join(suffix_parts) if suffix_parts else ""
    
    def run_full_pipeline(self, 
                         top_matches_file: str = None,
                         company_file: str = None,
                         output_file: str = None,
                         stage2_mapping_file: str = None,
                         save_onet_outputs: bool = False,
                         onet_output_dir: str = None) -> pd.DataFrame:
        """
        Run the complete 4-step Task → Occupation × Firm AI Exposure pipeline.
        
        Args:
            top_matches_file: Path to top_5_matches.csv file (auto-detect if None)
            company_file: Path to company data CSV (auto-detect if None)
            output_file: Output CSV path (auto-generate if None)
            stage2_mapping_file: Path to Stage 2 deduplication mapping (auto-detect if None)
            save_onet_outputs: If True, save intermediate O*NET exposure files
            onet_output_dir: Directory for O*NET output files (defaults to data_dir)
            
        Returns:
            DataFrame with final ISCO-08 occupation-firm exposure scores
        """
        logger.info("="*60)
        logger.info("STARTING 4-STEP TASK → OCCUPATION × FIRM AI EXPOSURE PIPELINE")
        logger.info("="*60)
        
        # Set O*NET output directory
        if onet_output_dir is None:
            onet_output_dir = self.data_dir
        
        # Auto-detect top matches file if not provided
        if top_matches_file is None:
            top_matches_file = os.path.join(self.data_dir, "top_5_matches.csv")
            logger.info(f"Using top matches file: {top_matches_file}")
        
        # Auto-detect company file if not provided
        if company_file is None:
            company_file = os.path.join(self.data_dir, "ai_development_deduplicated.csv")
            logger.info(f"Using company data file: {company_file}")
        
        if save_onet_outputs:
            logger.info(f"O*NET outputs will be saved to: {onet_output_dir}")
        
        # Define other input files
        task_statements_file = os.path.join(self.data_dir, "Task Statements.xlsx")
        task_ratings_file = os.path.join(self.data_dir, "Task Ratings.xlsx")
        esco_onet_file = os.path.join(self.data_dir, "ESCO_to_ONET-SOC.xlsx")
        
        # ================================================================
        # DATA LOADING
        # ================================================================
        logger.info("\n" + "="*50)
        logger.info("DATA LOADING")
        logger.info("="*50)
        
        # Step 1: Load exposed task-application matches
        task_app_matches = self.step1_load_task_application_matches(top_matches_file)
        
        # Load supporting data
        task_statements = self.load_task_statements(task_statements_file, core_only=True)
        task_ratings = self.load_task_ratings(task_ratings_file)
        job_app_mapping = self.load_job_app_mapping()
        original_jobs = self.load_original_full_dataset(company_file)  # Load full original dataset
        esco_crosswalk = self.load_esco_onet_crosswalk(esco_onet_file)
        isco_titles = self.load_isco_titles()
        
        # Load company mapping for Stage 6 compatibility
        company_mapping = self.load_company_mapping()
        
        # ================================================================
        # 4-STEP HAMPOLE PIPELINE
        # ================================================================
        
        # Step 2: Task exposure at firm level
        logger.info("\n" + "="*50)
        logger.info("STEP 2: TASK EXPOSURE AT FIRM LEVEL")
        logger.info("="*50)
        
        task_firm_exposure = self.step2_calculate_firm_task_exposure(
            task_app_matches, job_app_mapping, original_jobs, stage2_mapping_file
        )
        
        # Step 3: Occupation × Firm exposure aggregation
        logger.info("\n" + "="*50)
        logger.info("STEP 3: OCCUPATION × FIRM EXPOSURE AGGREGATION")
        logger.info("="*50)
        
        occupation_firm_exposure = self.step3_calculate_occupation_firm_exposure(
            task_firm_exposure, task_statements, task_ratings
        )
        
        # Step 4: AI intensity adjustment
        logger.info("\n" + "="*50)
        logger.info("STEP 4: AI INTENSITY ADJUSTMENT")
        logger.info("="*50)
        
        final_onet_exposure = self.step4_apply_ai_intensity_adjustment(occupation_firm_exposure)
        
        # Add company names to ONET exposure for output compatibility
        if 'company_name' not in final_onet_exposure.columns:
            logger.info("Adding company names to ONET exposure data...")
            final_onet_exposure = final_onet_exposure.merge(company_mapping, on='company_id', how='left')
            missing_companies = final_onet_exposure['company_name'].isna().sum()
            if missing_companies > 0:
                logger.warning(f"{missing_companies} company_ids in ONET exposure could not be mapped to company names")
        
        # ================================================================
        # SAVE O*NET OUTPUTS (BEFORE ISCO CROSSWALK)
        # ================================================================
        if save_onet_outputs:
            logger.info("\n" + "="*50)
            logger.info("SAVING O*NET EXPOSURE OUTPUTS")
            logger.info("="*50)
            
            # Generate configuration suffix for filenames
            config_suffix = self._generate_config_suffix()
            
            # 1. Save O*NET Occupation-Firm-Year Exposure (primary output)
            # Add O*NET titles to firm-year exposure
            onet_titles = task_statements[['O*NET-SOC Code', 'Title']].drop_duplicates()
            onet_titles.rename(columns={'O*NET-SOC Code': 'onet_code', 'Title': 'onet_title'}, inplace=True)
            final_onet_with_titles = final_onet_exposure.merge(onet_titles, on='onet_code', how='left')
            
            # Reorder columns for better readability
            cols = ['onet_code', 'onet_title', 'company_name', 'year'] + [col for col in final_onet_with_titles.columns 
                                                                          if col not in ['onet_code', 'onet_title', 'company_name', 'year']]
            final_onet_with_titles = final_onet_with_titles[cols]
            
            time_suffix = "time_invariant" if self.time_invariant else "time_variant"
            onet_firm_file = os.path.join(onet_output_dir, f"onet_firm_year_ai_exposure_{time_suffix}{config_suffix}.csv")
            final_onet_with_titles.to_csv(onet_firm_file, index=False, encoding='utf-8')
            logger.info(f"✅ Saved O*NET firm-year exposure: {onet_firm_file}")
            
            # 2. Save O*NET Occupation-Level Exposure Summary
            task_exposure_table = self.create_task_exposure_table(task_app_matches, task_statements, task_ratings)
            onet_occupation_summary = self.aggregate_tasks_to_onet_occupations(task_exposure_table)
            
            # Add O*NET titles to the summary
            onet_titles = task_statements[['O*NET-SOC Code', 'Title']].drop_duplicates()
            onet_titles.rename(columns={'O*NET-SOC Code': 'onet_code', 'Title': 'onet_title'}, inplace=True)
            onet_occupation_summary = onet_occupation_summary.merge(onet_titles, on='onet_code', how='left')
            
            # Reorder columns to put title after code
            cols = ['onet_code', 'onet_title'] + [col for col in onet_occupation_summary.columns 
                                                  if col not in ['onet_code', 'onet_title']]
            onet_occupation_summary = onet_occupation_summary[cols]
            
            onet_summary_file = os.path.join(onet_output_dir, f"onet_occupation_exposure_summary{config_suffix}.csv")
            onet_occupation_summary.to_csv(onet_summary_file, index=False, encoding='utf-8')
            logger.info(f"✅ Saved O*NET occupation summary: {onet_summary_file}")
            
            # 3. Save O*NET Task-Level Exposure Table
            onet_task_file = os.path.join(onet_output_dir, f"onet_task_exposure{config_suffix}.csv")
            task_exposure_table.to_csv(onet_task_file, index=False, encoding='utf-8')
            logger.info(f"✅ Saved O*NET task exposure: {onet_task_file}")
            
            logger.info(f"📊 O*NET outputs summary:")
            logger.info(f"  - Firm-year combinations: {len(final_onet_exposure):,}")
            logger.info(f"  - Unique O*NET occupations: {final_onet_exposure['onet_code'].nunique():,}")
            logger.info(f"  - Unique firms: {final_onet_exposure['company_name'].nunique():,}")
            logger.info(f"  - Occupation-level summaries: {len(onet_occupation_summary):,}")
            logger.info(f"  - Task-level exposures: {len(task_exposure_table):,}")
        
        # ================================================================
        # CROSSWALK TO ISCO-08
        # ================================================================
        logger.info("\n" + "="*50)
        logger.info("CROSSWALKING O*NET → ISCO-08")
        logger.info("="*50)
        
        isco_firm_exposure = self.crosswalk_onet_to_isco(
            final_onet_exposure, esco_crosswalk, isco_titles, company_mapping
        )
        
        # ================================================================
        # SAVE RESULTS
        # ================================================================
        logger.info("\n" + "="*50)
        logger.info("SAVING RESULTS")
        logger.info("="*50)
        
        if output_file is None:
            time_suffix = "time_invariant" if self.time_invariant else "time_variant"
            output_file = os.path.join(self.data_dir, f"isco_firm_year_ai_exposure_{time_suffix}.csv")
        
        isco_firm_exposure.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved ISCO-08 occupation-firm-year AI exposure to: {output_file}")
        
        # ================================================================
        # PIPELINE SUMMARY
        # ================================================================
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Input: {len(task_app_matches):,} task-application matches from stage 4")
        logger.info(f"✅ Step 2: {len(task_firm_exposure):,} task-firm-year exposure combinations")
        logger.info(f"✅ Step 3: {len(occupation_firm_exposure):,} O*NET occupation-firm-year combinations")
        logger.info(f"✅ Step 4: Applied AI intensity adjustment with log(1 + N_apps_firm_year)")
        logger.info(f"✅ Final: {len(isco_firm_exposure):,} ISCO-08 occupation-firm-year exposure combinations")
        logger.info(f"✅ Unique ISCO occupations: {isco_firm_exposure['isco08_4d'].nunique():,}")
        logger.info(f"✅ Unique firms: {isco_firm_exposure['company_name'].nunique():,}")
        logger.info(f"✅ Unique years: {sorted(isco_firm_exposure['year'].unique())}")
        logger.info(f"✅ Output file: {output_file}")
        
        # Show top exposed ISCO-firm-year combinations (Hampole variant)
        logger.info(f"\nTop 10 highest AI exposure combinations (Hampole variant):")
        top_exposed = isco_firm_exposure.nlargest(10, 'hampole_ai_exposure_avg')
        for _, row in top_exposed.iterrows():
            logger.info(f"  {row['isco08_4d']} @ {row['company_name']} ({row['year']:.0f}): "
                       f"AI Exposure = {row['hampole_ai_exposure_avg']:.3f} "
                       f"({row['n_ai_apps_firm_year']:.0f} apps)")
        
        return isco_firm_exposure

def create_isco_task_sample_report(stage4_file: str, data_dir: str = "Data/", 
                                  n_sample: int = 10, threshold: float = 0.6,
                                  output_dir: str = "Data/"):
    """
    Create a detailed report showing sample ISCO occupations with their ESCO mappings
    and detailed task exposure breakdown.
    
    Args:
        stage4_file: Path to Stage 4 parquet file
        data_dir: Directory containing input data files
        n_sample: Number of ISCO codes to sample
        threshold: Cross encoder threshold for exposure
        output_dir: Output directory for reports
    
    Returns:
        Tuple of (detailed_report_df, summary_df)
    """
    print("="*60)
    print("ISCO TASK SAMPLE REPORT GENERATOR")
    print("="*60)
    
    # Load exposed tasks from Stage 4
    print(f"Loading exposed tasks from: {stage4_file}")
    stage4_df = pd.read_parquet(stage4_file)
    
    # Use cross_encoder_score if available, otherwise fall back to similarity
    if 'cross_encoder_score' in stage4_df.columns:
        score_column = 'cross_encoder_score'
        print(f"Using cross_encoder_score for threshold filtering")
    else:
        score_column = 'similarity'
        print(f"Using similarity score for threshold filtering (no cross_encoder_score found)")
    
    exposed_tasks = set(stage4_df[stage4_df[score_column] > threshold]['onet_task_id'].unique())
    print(f"Found {len(exposed_tasks):,} exposed task IDs ({score_column} > {threshold})")
    
    # Load data files
    print("Loading data files...")
    
    # ESCO to O*NET crosswalk
    esco_df = pd.read_excel(os.path.join(data_dir, "ESCO_to_ONET-SOC.xlsx"), skiprows=2)
    esco_df.columns = ['esco_code', 'esco_title', 'onet_code', 'onet_title']
    esco_df = esco_df[esco_df['esco_code'] != 'ESCO/ISCO Code'].dropna()
    esco_df['isco08_4d'] = esco_df['esco_code'].astype(str).str[:4]
    
    # Task statements
    tasks_df = pd.read_excel(os.path.join(data_dir, "Task Statements.xlsx"))
    
    # Task ratings for importance weights
    ratings_df = pd.read_excel(os.path.join(data_dir, "Task Ratings.xlsx"))
    importance_df = ratings_df[
        (ratings_df['Scale ID'] == 'IM') & 
        (ratings_df['Scale Name'] == 'Importance')
    ][['O*NET-SOC Code', 'Task ID', 'Data Value']].copy()
    importance_df.rename(columns={'Data Value': 'importance_weight'}, inplace=True)
    
    # ISCO titles from exact 4-digit ESCO codes
    exact_4d = esco_df[esco_df['esco_code'].astype(str).str.len() == 4].copy()
    isco_titles = exact_4d[['esco_code', 'esco_title']].drop_duplicates()
    isco_titles['isco08_4d'] = isco_titles['esco_code'].astype(str).str.zfill(4)
    isco_titles = isco_titles[['isco08_4d', 'esco_title']].drop_duplicates()
    isco_titles.rename(columns={'esco_title': 'isco08_title'}, inplace=True)
    
    print(f"Loaded {len(esco_df):,} ESCO mappings, {len(tasks_df):,} task statements")
    
    # Select sample ISCO codes
    print(f"Selecting {n_sample} sample ISCO codes...")
    isco_coverage = esco_df.groupby('isco08_4d').agg({
        'onet_code': 'nunique',
        'esco_code': 'nunique'
    }).reset_index()
    
    # Filter to codes with reasonable coverage (at least 2 O*NET codes)
    good_coverage = isco_coverage[isco_coverage['onet_code'] >= 2].copy()
    
    # Sample stratified by coverage level for diversity
    good_coverage['coverage_quartile'] = pd.qcut(good_coverage['onet_code'], 4, labels=False)
    sample_codes = []
    for quartile in range(4):
        quartile_codes = good_coverage[good_coverage['coverage_quartile'] == quartile]
        n_from_quartile = min(3, len(quartile_codes), n_sample - len(sample_codes))
        if n_from_quartile > 0:
            sampled = quartile_codes.sample(n=n_from_quartile, random_state=42)
            sample_codes.extend(sampled['isco08_4d'].tolist())
    
    # Fill remaining slots randomly if needed
    remaining_slots = n_sample - len(sample_codes)
    if remaining_slots > 0:
        available_codes = set(good_coverage['isco08_4d']) - set(sample_codes)
        if available_codes:
            additional = np.random.choice(list(available_codes), 
                                       size=min(remaining_slots, len(available_codes)), 
                                       replace=False)
            sample_codes.extend(additional)
    
    sample_codes = sample_codes[:n_sample]
    print(f"Selected {len(sample_codes)} ISCO codes: {sample_codes}")
    
    # Create detailed report
    print(f"Creating detailed report for {len(sample_codes)} ISCO codes...")
    report_data = []
    
    for i, isco_code in enumerate(sample_codes, 1):
        print(f"Processing ISCO {isco_code} ({i}/{len(sample_codes)})...")
        
        # Get ISCO title
        isco_title_row = isco_titles[isco_titles['isco08_4d'] == isco_code]
        isco_title = isco_title_row['isco08_title'].iloc[0] if len(isco_title_row) > 0 else "Title not available"
        
        # Get ESCO occupations for this ISCO
        esco_for_isco = esco_df[esco_df['isco08_4d'] == isco_code][['esco_code', 'esco_title', 'onet_code']].drop_duplicates()
        
        # Get all O*NET codes for this ISCO
        onet_codes = esco_for_isco['onet_code'].unique()
        
        # Get all tasks for these O*NET codes
        tasks_for_isco = tasks_df[tasks_df['O*NET-SOC Code'].isin(onet_codes)].copy()
        
        if len(tasks_for_isco) == 0:
            continue
        
        # Add exposure status
        tasks_for_isco['exposed'] = tasks_for_isco['Task ID'].isin(exposed_tasks)
        
        # Add importance weights
        tasks_for_isco = tasks_for_isco.merge(
            importance_df, 
            left_on=['O*NET-SOC Code', 'Task ID'],
            right_on=['O*NET-SOC Code', 'Task ID'],
            how='left'
        )
        tasks_for_isco['importance_weight'] = tasks_for_isco['importance_weight'].fillna(3.0)
        
        # Calculate exposure statistics
        total_tasks = len(tasks_for_isco)
        exposed_tasks_count = tasks_for_isco['exposed'].sum()
        exposure_rate = exposed_tasks_count / total_tasks if total_tasks > 0 else 0
        
        # Weighted exposure
        total_weight = tasks_for_isco['importance_weight'].sum()
        exposed_weight = tasks_for_isco[tasks_for_isco['exposed']]['importance_weight'].sum()
        weighted_exposure = exposed_weight / total_weight if total_weight > 0 else 0
        
        print(f"  Found {total_tasks} tasks, {exposed_tasks_count} exposed ({exposure_rate:.1%})")
        
        # Add to report
        for _, task in tasks_for_isco.iterrows():
            report_data.append({
                'isco08_4d': isco_code,
                'isco08_title': isco_title,
                'total_tasks': total_tasks,
                'exposed_tasks': exposed_tasks_count,
                'exposure_rate': exposure_rate,
                'weighted_exposure': weighted_exposure,
                'n_esco_codes': len(esco_for_isco['esco_code'].unique()),
                'n_onet_codes': len(onet_codes),
                'onet_code': task['O*NET-SOC Code'],
                'onet_title': task['Title'],
                'task_id': task['Task ID'],
                'task_text': task['Task'],
                'exposed': 'YES' if task['exposed'] else 'NO',
                'importance_weight': task['importance_weight']
            })
    
    # Create detailed report DataFrame
    report_df = pd.DataFrame(report_data)
    
    # Create summary table
    summary_data = []
    for isco_code in report_df['isco08_4d'].unique():
        isco_data = report_df[report_df['isco08_4d'] == isco_code]
        esco_for_isco = esco_df[esco_df['isco08_4d'] == isco_code][['esco_code', 'esco_title']].drop_duplicates()
        
        summary_data.append({
            'isco08_4d': isco_code,
            'isco08_title': isco_data['isco08_title'].iloc[0],
            'total_tasks': isco_data['total_tasks'].iloc[0],
            'exposed_tasks': isco_data['exposed_tasks'].iloc[0],
            'exposure_rate': isco_data['exposure_rate'].iloc[0],
            'weighted_exposure': isco_data['weighted_exposure'].iloc[0],
            'n_esco_codes': len(esco_for_isco),
            'n_onet_codes': isco_data['n_onet_codes'].iloc[0],
            'esco_codes': '; '.join(esco_for_isco['esco_code'].astype(str)),
            'esco_titles': '; '.join(esco_for_isco['esco_title'])
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save files
    detailed_file = os.path.join(output_dir, f"isco_task_sample_detailed.csv")
    summary_file = os.path.join(output_dir, f"isco_task_sample_summary.csv")
    
    report_df.to_csv(detailed_file, index=False, encoding='utf-8')
    summary_df.to_csv(summary_file, index=False, encoding='utf-8')
    
    print(f"\n✅ Detailed report saved to: {detailed_file}")
    print(f"📋 Summary table saved to: {summary_file}")
    print(f"📊 Report contains {len(report_df):,} task rows across {len(sample_codes)} ISCO occupations")
    
    # Show preview
    print(f"\nPreview of sampled ISCO occupations:")
    for _, row in summary_df.iterrows():
        print(f"  {row['isco08_4d']}: {row['isco08_title']}")
        print(f"    → {row['total_tasks']} tasks, {row['exposed_tasks']} exposed ({row['exposure_rate']:.1%})")
        print(f"    → {row['n_esco_codes']} ESCO codes, {row['n_onet_codes']} O*NET codes")
    
    return report_df, summary_df

def main():
    """Command line interface for the O*NET → ISCO exposure pipeline."""
    parser = argparse.ArgumentParser(description="O*NET Task Exposure → ISCO-08 Occupation Exposure Pipeline")
    
    parser.add_argument("--top-matches-file", type=str, 
                       help="Path to top_5_matches.csv file (auto-detect if not provided)")
    parser.add_argument("--output-file", type=str,
                       help="Output CSV file path (auto-generate if not provided)")
    parser.add_argument("--aggregation", type=str, default="mean", 
                       choices=["mean"],
                       help="Aggregation method for O*NET → ISCO (default: mean)")
    parser.add_argument("--data-dir", type=str, default="Data/",
                       help="Directory containing input data files (default: Data/)")
    parser.add_argument("--time-invariant", action="store_true",
                       help="Use time-invariant exposure (firms exposed to all their AI apps across all years)")
    parser.add_argument("--occupation-exposure", type=str, default="none",
                       choices=["none", "time-invariant", "time-variant"],
                       help="Occupation-level exposure mode: 'none' (firm-level), 'time-invariant' (all firms exposed to all apps for all time), 'time-variant' (all firms exposed to all apps from first appearance onwards)")
    parser.add_argument("--create-sample-report", action="store_true",
                       help="Create detailed ISCO task sample report")
    parser.add_argument("--n-sample", type=int, default=10,
                       help="Number of ISCO codes to sample for report (default: 10)")
    parser.add_argument("--stage4-file", type=str,
                       help="Path to Stage 4 parquet file (auto-detect if not provided)")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Cross encoder threshold for exposed tasks (default: 0.6)")
    parser.add_argument("--stage2-mapping-file", type=str,
                       help="Path to Stage 2 deduplication mapping file (auto-detect if not provided)")
    parser.add_argument("--company-file", type=str,
                       help="Path to company data CSV file (auto-detect if not provided)")
    parser.add_argument("--save-onet-outputs", action="store_true",
                       help="Save intermediate O*NET exposure files before ISCO crosswalk")
    parser.add_argument("--onet-output-dir", type=str,
                       help="Directory for O*NET output files (defaults to data-dir)")
    
    args = parser.parse_args()
    
    if args.create_sample_report:
        # Create sample report
        stage4_file = args.stage4_file
        if stage4_file is None:
            # Auto-detect stage4 parquet file from CSV filename
            if args.top_matches_file:
                csv_file = Path(args.top_matches_file)
                if csv_file.name == "top_5_matches.csv":
                    parquet_file = csv_file.parent / "ai_app_onet_top5.parquet"
                elif csv_file.name == "top_10_matches.csv":
                    parquet_file = csv_file.parent / "ai_app_onet_top10.parquet"
                else:
                    # Generic pattern - try to infer from filename
                    base_name = csv_file.stem.replace("top_", "ai_app_onet_top").replace("_matches", "")
                    parquet_file = csv_file.parent / f"{base_name}.parquet"
                
                if parquet_file.exists():
                    stage4_file = str(parquet_file)
                    print(f"Auto-detected Stage 4 file from CSV: {stage4_file}")
            
            # Fallback to glob pattern if auto-detection failed
            if stage4_file is None:
                stage4_files = sorted(Path(args.data_dir).glob("ai_app_onet_top*.parquet"))
                if not stage4_files:
                    raise FileNotFoundError("No Stage 4 output files found and cannot auto-detect from CSV filename")
                stage4_file = str(stage4_files[-1])
                print(f"Auto-detected Stage 4 file via glob: {stage4_file}")
        
        detailed_df, summary_df = create_isco_task_sample_report(
            stage4_file=stage4_file,
            data_dir=args.data_dir,
            n_sample=args.n_sample,
            threshold=args.threshold,
            output_dir=args.data_dir
        )
        
        print(f"\n✅ Sample report completed successfully!")
        print(f"📊 Generated reports for {args.n_sample} ISCO occupations")
        print(f"📋 Detailed report: {len(detailed_df):,} task-level rows")
        print(f"📋 Summary report: {len(summary_df):,} occupation-level rows")
        
    else:
        # Run main pipeline
        pipeline = TaskFirmExposurePipeline(
            aggregation_method=args.aggregation,
            time_invariant=args.time_invariant,
            occupation_exposure=args.occupation_exposure,
            data_dir=args.data_dir
        )
        
        result_df = pipeline.run_full_pipeline(
            top_matches_file=args.top_matches_file,
            company_file=args.company_file,
            output_file=args.output_file,
            stage2_mapping_file=args.stage2_mapping_file,
            save_onet_outputs=args.save_onet_outputs,
            onet_output_dir=args.onet_output_dir
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Final dataset: {len(result_df)} ISCO-08 occupations with exposure scores")

if __name__ == "__main__":
    main()