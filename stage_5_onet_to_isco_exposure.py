#!/usr/bin/env python3
"""
Stage 5: Task → Occupation × Firm AI Exposure Pipeline

This stage implements a 4-step exposure calculation pipeline following Hampole et al. (2025):

Step 1: Task-Application matching (using stage 4 cross encoder results)
Step 2: Task exposure at firm level (Hampole share + Binary any-match variants)
Step 3: Occupation × Firm exposure aggregation using O*NET task importance weights
Step 4: AI intensity adjustment using log(1 + N_apps)

Then crosswalks results to ISCO-08 for integration with survey data.

python3 stage_5_onet_to_isco_exposure.py \
    --top-matches-file task_exposure_matches_all_thresholds_openai_bge20_15_10_5_1_ce0p8_0p6_0p4_0p2_onet20_core.parquet \
    --stage-4-dir Data/Testing/stage_4/200_row_test/ \
    --stage-2-dir Data/Testing/stage_2/ \
    --save-onet-outputs \
    --output-dir Data/Testing/stage_5/200_row_test/ \
    --model openai
    --task-type core \
    --use-employment-weights
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import argparse
import gc
from datetime import datetime
import shutil

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
                 exposure_table_format: str = "exposed-firm-occ-year",
                 data_dir: str = "Data/",
                 stage_4_dir: str = "Data/Stage_4/",
                 stage_2_dir: str = "Data/Stage_2/",
                 output_dir: str = None,
                 bge_percentiles: List[float] = None,
                 ce_thresholds: List[float] = None,
                 task_type: str = 'core',
                 onet_version: int = 20,
                 use_employment_weights: bool = False,
                 apply_min_threshold: bool = True,
                 save_debug_csv: bool = True,
                 force_nonchunked: bool = False):
        """
        Initialize the multi-specification exposure pipeline.

        Args:
            aggregation_method: Method to aggregate O*NET → ISCO ("mean", "weighted_mean")
            time_invariant: If True, firms exposed to all their AI apps across all years; if False, firms exposed to apps from first appearance onwards
            exposure_table_format: Output table format for firm × occupation × year exposure:
                - "exposed-firm-occ-year": Only combinations with exposure > 0 [DEFAULT]
                - "all-firm-occ-year": All firm × occupation × year combinations (includes zeros)
                - "exposed-occ-year": Only occupation × year (same for all firms)
            data_dir: Directory containing input data files
            stage_4_dir: Directory containing Stage 4 cross-encoder results
            stage_2_dir: Directory containing Stage 2 data files
            output_dir: Directory for output files (default: Data/firm_year_exposure/). Used for both O*NET and ISCO outputs
            bge_percentiles: List of BGE percentile cutoffs (default: [20, 15, 10, 5, 1])
            ce_thresholds: List of cross-encoder thresholds (default: [0.8, 0.6, 0.4, 0.2, 0.0])
            task_type: Task type ('core' or 'all')
            onet_version: O*NET version (default: 20). BLS SOC-10→ISCO-08 crosswalk only compatible with v20
            use_employment_weights: If True, use employment-weighted aggregation for O*NET→ISCO crosswalk via 6-digit SOC
            apply_min_threshold: If True (default), filter matches by both percentile AND minimum similarity threshold (above_min_threshold column)
            save_debug_csv: If True (default), save large debug CSVs (expanded_jobs_matching.csv, task_app_matches.csv) to output_dir.
                Set False to reduce disk usage.
            force_nonchunked: If True, disable chunk-based processing paths (loads full filtered spec into memory).
                This can be faster but may increase peak memory usage substantially.
        """
        self.aggregation_method = aggregation_method
        self.time_invariant = time_invariant
        self.exposure_table_format = exposure_table_format
        self.data_dir = data_dir
        self.stage_4_dir = stage_4_dir
        self.stage_2_dir = stage_2_dir
        self.output_dir = output_dir if output_dir else os.path.join(data_dir, 'firm_year_exposure')
        self.task_type = task_type
        self.onet_version = onet_version
        self.use_employment_weights = use_employment_weights
        self.apply_min_threshold = apply_min_threshold
        self.save_debug_csv = save_debug_csv
        self.force_nonchunked = force_nonchunked

        # Validate O*NET version compatibility with BLS SOC-10 crosswalk
        if onet_version >= 25:
            raise ValueError(
                f"O*NET version {onet_version} uses SOC 2018+ taxonomy. "
                f"The BLS SOC-10 → ISCO-08 crosswalk is only compatible with O*NET v20 (SOC 2010). "
                f"Please use O*NET v20 or obtain a SOC 2018 → ISCO-08 crosswalk."
            )

        # Set default thresholds if not provided
        if bge_percentiles is None:
            bge_percentiles = [20, 15, 10, 5, 1]
        if ce_thresholds is None:
            ce_thresholds = [0.8, 0.6, 0.4, 0.2, 0.0]

        self.bge_percentiles = bge_percentiles
        self.ce_thresholds = ce_thresholds

        logger.info(f"Initialized Multi-Specification Task → Occupation × Firm AI exposure pipeline")
        logger.info(f"  Data directory: {data_dir}")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Aggregation method: {aggregation_method}")
        logger.info(f"  Firm time invariant exposure: {time_invariant}")
        logger.info(f"  Exposure table format: {exposure_table_format}")
        logger.info(f"  Task type: {task_type}")
        logger.info(f"  O*NET version: {onet_version} (SOC 2010 taxonomy)")
        logger.info(f"  O*NET→ISCO crosswalk mode: {'Employment-weighted (8d→6d→4d)' if use_employment_weights else 'Unweighted (8d→4d)'}")
        logger.info(f"  BGE percentiles: {bge_percentiles}")
        logger.info(f"  CE thresholds: {ce_thresholds}")
        logger.info(f"  Save debug CSVs: {save_debug_csv}")
        logger.info(f"  Force nonchunked: {force_nonchunked}")
    
    def step1_load_task_application_matches(self, top_matches_file: str) -> Tuple[str, pd.DataFrame]:
        """
        Step 1: Load task-application match metadata from Stage 4 unified multi-threshold file.

        Memory-efficient: Loads only filter columns (pct_*, ce_*, above_min_threshold) instead of full dataset.

        Args:
            top_matches_file: Path to task_exposure_matches_all_thresholds.parquet file

        Returns:
            Tuple of (file_path, metadata_df) where metadata_df contains only boolean filter columns
        """
        import pyarrow.parquet as pq

        logger.info(f"Step 1: Loading task-application match metadata from: {top_matches_file}")

        # Read ONLY filter columns (20-50MB instead of 15-20GB for full file)
        filter_cols = ['pct_20', 'pct_15', 'pct_10', 'pct_05', 'pct_01',
                       'ce_0.8', 'ce_0.6', 'ce_0.4', 'ce_0.2', 'ce_0.0',
                       'above_min_threshold']

        # Use PyArrow to read only filter columns
        try:
            metadata_table = pq.read_table(top_matches_file, columns=filter_cols)
            metadata_df = metadata_table.to_pandas()
        except Exception as e:
            # Fallback: try reading without above_min_threshold if it doesn't exist
            logger.warning(f"Error reading all filter columns: {e}")
            logger.warning("Trying without 'above_min_threshold' column...")
            filter_cols.remove('above_min_threshold')
            metadata_table = pq.read_table(top_matches_file, columns=filter_cols)
            metadata_df = metadata_table.to_pandas()

        logger.info(f"Loaded {len(metadata_df):,} rows of filter metadata")
        memory_mb = metadata_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Memory: {memory_mb:.1f} MB")

        # Validate filter columns
        bge_cols = [c for c in metadata_df.columns if c.startswith('pct_')]
        ce_cols = [c for c in metadata_df.columns if c.startswith('ce_')]

        if bge_cols and ce_cols:
            logger.info(f"✓ Found all threshold columns:")
            logger.info(f"  BGE percentiles: {sorted(bge_cols)}")
            logger.info(f"  CE thresholds: {sorted(ce_cols)}")
        else:
            raise ValueError(f"Threshold columns incomplete or missing: BGE={len(bge_cols)}, CE={len(ce_cols)}")

        # Get file metadata for row counts
        parquet_file = pq.ParquetFile(top_matches_file)
        total_rows = parquet_file.metadata.num_rows
        logger.info(f"  Total task-application pairs in file: {total_rows:,}")

        return top_matches_file, metadata_df

    def _load_filtered_matches(self, file_path: str, filter_mask: np.ndarray) -> pd.DataFrame:
        """
        Load only filtered rows on-demand (memory-efficient).

        Uses hybrid approach:
        - Small filtered datasets (<100M rows): PyArrow filtering (fast)
        - Large filtered datasets (≥100M rows): Pandas chunked reading (memory-safe)

        Args:
            file_path: Path to parquet file
            filter_mask: Boolean numpy array indicating which rows to load

        Returns:
            DataFrame with only filtered rows, all columns included
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        match_count = filter_mask.sum()
        logger.info(f"  Loading {match_count:,} filtered rows on-demand...")

        # Hybrid approach: chunk-based for large filtered datasets
        CHUNK_THRESHOLD = 100_000_000  # 100M rows

        if (not getattr(self, "force_nonchunked", False)) and match_count >= CHUNK_THRESHOLD:
            logger.info(f"  Using chunk-based loading (filtered dataset ≥100M rows)")
            filtered_df = self._load_filtered_matches_chunked(file_path, filter_mask)
        else:
            if getattr(self, "force_nonchunked", False) and match_count >= CHUNK_THRESHOLD:
                logger.warning(
                    "  Force-nonchunked enabled: loading large filtered dataset via PyArrow filtering "
                    f"({match_count:,} rows). This may require substantial RAM."
                )
            else:
                logger.info(f"  Using PyArrow filtering (filtered dataset <100M rows)")
            # Read table and filter using PyArrow (columnar, fast)
            table = pq.read_table(file_path)
            filtered_table = table.filter(pa.array(filter_mask))
            filtered_df = filtered_table.to_pandas()

        # Validate required columns
        required_cols = ['app_text', 'onet_task_id', 'onet_task', 'similarity',
                         'cross_encoder_score', 'ai_app_id', 'job_uid']
        missing_cols = [c for c in required_cols if c not in filtered_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        filter_pct = (match_count / len(filter_mask) * 100) if len(filter_mask) > 0 else 0
        memory_mb = filtered_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"  Loaded {len(filtered_df):,} rows ({filter_pct:.2f}% of total)")
        logger.info(f"  Memory: {memory_mb:.1f} MB")
        logger.info(f"  Unique tasks: {filtered_df['onet_task_id'].nunique():,}")
        logger.info(f"  Unique AI applications: {filtered_df['app_text'].nunique():,}")

        return filtered_df

    def _load_filtered_matches_chunked(self, file_path: str, filter_mask: np.ndarray) -> pd.DataFrame:
        """
        Load filtered rows using chunk-based processing (for large filtered datasets).

        Uses PyArrow ParquetFile to read file in row groups (chunks), applies boolean filter
        to each chunk, and accumulates results. Memory-efficient for cases where filter selects
        most/all rows.

        Args:
            file_path: Path to parquet file
            filter_mask: Boolean numpy array indicating which rows to load

        Returns:
            DataFrame with only filtered rows, all columns included
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        CHUNK_SIZE = 10_000_000  # Target 10M rows per chunk

        logger.info(f"  Reading in ~{CHUNK_SIZE:,}-row chunks using PyArrow row groups...")

        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        num_row_groups = parquet_file.num_row_groups

        logger.info(f"  File has {num_row_groups} row groups, {total_rows:,} total rows")

        filtered_chunks = []
        total_filtered = 0
        row_offset = 0

        # Read file row group by row group
        for rg_idx in range(num_row_groups):
            # Read this row group as a table
            rg_table = parquet_file.read_row_group(rg_idx)
            rg_size = len(rg_table)

            # Extract filter mask for this row group
            rg_filter = filter_mask[row_offset:row_offset + rg_size]

            # Apply filter using PyArrow
            filtered_table = rg_table.filter(pa.array(rg_filter))

            if len(filtered_table) > 0:
                # Convert to pandas and add to chunks
                chunk_filtered = filtered_table.to_pandas()
                filtered_chunks.append(chunk_filtered)
                total_filtered += len(chunk_filtered)

            # Update offset
            row_offset += rg_size

            del rg_table, filtered_table
            import gc
            gc.collect()

            if (rg_idx + 1) % 10 == 0 or rg_idx == num_row_groups - 1:
                logger.info(f"  Processed {rg_idx + 1}/{num_row_groups} row groups ({row_offset:,} rows, {total_filtered:,} filtered)")

        logger.info(f"  Processed all {num_row_groups} row groups ({total_filtered:,} rows filtered)")

        # Concatenate all filtered chunks
        if len(filtered_chunks) > 0:
            logger.info(f"  Concatenating {len(filtered_chunks)} filtered chunks...")
            filtered_df = pd.concat(filtered_chunks, ignore_index=True)
        else:
            logger.warning(f"  No rows passed filter - returning empty DataFrame")
            # Read schema from file to create empty DataFrame with correct columns
            schema_df = pq.read_table(file_path).slice(0, 0).to_pandas()
            filtered_df = schema_df

        del filtered_chunks
        import gc
        gc.collect()

        return filtered_df

    def _load_filtered_matches_chunked_generator(self, file_path: str, filter_mask: np.ndarray):
        """
        Yield filtered row groups one at a time (memory-efficient chunk iterator).

        Instead of concatenating all filtered chunks, yields each chunk individually.
        Caller processes each chunk through step2_chunked and accumulates aggregated results.

        Args:
            file_path: Path to parquet file
            filter_mask: Boolean numpy array indicating which rows to load

        Yields:
            DataFrame chunks with only filtered rows, all columns included
        """
        import pyarrow.parquet as pq
        import pyarrow as pa

        logger.info(f"  Reading in row groups as chunk iterator (memory-efficient)...")

        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        num_row_groups = parquet_file.num_row_groups

        logger.info(f"  File has {num_row_groups} row groups, {total_rows:,} total rows")

        total_filtered = 0
        row_offset = 0

        # Yield filtered row groups one at a time
        for rg_idx in range(num_row_groups):
            # Read this row group as a table
            rg_table = parquet_file.read_row_group(rg_idx)
            rg_size = len(rg_table)

            # Extract filter mask for this row group
            rg_filter = filter_mask[row_offset:row_offset + rg_size]

            # Apply filter using PyArrow
            filtered_table = rg_table.filter(pa.array(rg_filter))

            if len(filtered_table) > 0:
                # Convert to pandas and yield
                chunk_filtered = filtered_table.to_pandas()
                total_filtered += len(chunk_filtered)

                # Yield this chunk to caller
                yield chunk_filtered

                del chunk_filtered

            # Update offset
            row_offset += rg_size

            del rg_table, filtered_table
            gc.collect()

            if (rg_idx + 1) % 10 == 0 or rg_idx == num_row_groups - 1:
                logger.info(f"  Processed {rg_idx + 1}/{num_row_groups} row groups ({row_offset:,} rows, {total_filtered:,} filtered)")

        logger.info(f"  Completed iteration over all {num_row_groups} row groups ({total_filtered:,} rows filtered)")

    def _precalculate_firm_app_mappings(self,
                                        matches_file_path: str,
                                        filter_mask: np.ndarray,
                                        expanded_job_app_mapping: pd.DataFrame) -> Tuple[Dict, pd.DataFrame]:
        """
        Pre-calculate firm-app-year mappings by scanning task_app_matches file in chunks.

        This is the first pass over the data that builds:
        1. firm_year_app_counts: {(firm, year): n_apps} for each firm-year
        2. firm_app_first_years: DataFrame with (firm, app, first_year) for time-variant mode

        Args:
            matches_file_path: Path to Stage 4 parquet file
            filter_mask: Boolean mask for current specification
            expanded_job_app_mapping: Pre-loaded expanded job mapping

        Returns:
            (firm_year_app_counts, firm_app_first_years)
        """
        from collections import defaultdict

        logger.info("\n" + "="*80)
        logger.info("PRE-CALCULATION PASS: Building firm-app-year mappings")
        logger.info("="*80)

        # Track firm-app relationships
        firm_app_years = defaultdict(dict)  # {firm: {app: first_year}}
        all_firms = set()
        all_years = set()

        chunk_count = 0
        total_rows_processed = 0

        # Iterate over filtered chunks
        for chunk in self._load_filtered_matches_chunked_generator(matches_file_path, filter_mask):
            chunk_count += 1
            total_rows_processed += len(chunk)

            # Explode job_uid arrays
            def extract_job_uid(x):
                if isinstance(x, (list, np.ndarray)):
                    return list(x) if len(x) > 0 else [np.nan]
                return [x]

            chunk['job_uid'] = chunk['job_uid'].apply(extract_job_uid)
            chunk_exploded = chunk.explode('job_uid').reset_index(drop=True)
            chunk_exploded['job_uid'] = chunk_exploded['job_uid'].astype(str)

            # Join with expanded job mapping to get firm info
            chunk_with_firms = chunk_exploded.merge(
                expanded_job_app_mapping[['app_text', 'job_uid', 'company_name', 'year']],
                on=['app_text', 'job_uid'],
                how='left',
                validate='many_to_one'
            )

            # Track first appearances
            for row in chunk_with_firms.itertuples():
                if pd.isna(row.company_name) or pd.isna(row.year):
                    continue

                all_firms.add(row.company_name)
                all_years.add(row.year)

                # Track first year for this (firm, app) combination
                if row.app_text not in firm_app_years[row.company_name]:
                    firm_app_years[row.company_name][row.app_text] = row.year
                else:
                    firm_app_years[row.company_name][row.app_text] = min(
                        firm_app_years[row.company_name][row.app_text],
                        row.year
                    )

            del chunk, chunk_exploded, chunk_with_firms
            gc.collect()

            if chunk_count % 10 == 0:
                logger.info(f"  Pre-calculation: processed {chunk_count} chunks ({total_rows_processed:,} rows)")

        logger.info(f"Pre-calculation complete: {chunk_count} chunks, {total_rows_processed:,} rows")
        logger.info(f"  Unique firms: {len(all_firms):,}")
        logger.info(f"  Unique years: {len(all_years)}")

        # Build firm_app_first_years DataFrame
        firm_app_records = []
        for firm, apps in firm_app_years.items():
            for app, first_year in apps.items():
                firm_app_records.append({
                    'company_name': firm,
                    'app_text': app,
                    'first_year': first_year
                })

        firm_app_first_years = pd.DataFrame(firm_app_records)
        logger.info(f"  Firm-app mappings: {len(firm_app_first_years):,} unique (firm, app) pairs")

        # Pre-calculate n_ai_apps_firm_year for all firm-years
        firm_year_app_counts = {}

        if self.time_invariant:
            # Time-invariant: All firms exposed to all their apps across all years
            for firm in all_firms:
                n_apps_ever = len(firm_app_years[firm])
                for year in all_years:
                    firm_year_app_counts[(firm, year)] = n_apps_ever
            logger.info(f"  Time-invariant mode: {len(firm_year_app_counts):,} firm-year combinations")
        else:
            # Time-variant: Firms exposed to apps by first-appearance year
            for firm in all_firms:
                for year in all_years:
                    # Count apps available by this year
                    available = [app for app, first in firm_app_years[firm].items() if first <= year]
                    firm_year_app_counts[(firm, year)] = len(available)
            logger.info(f"  Time-variant mode: {len(firm_year_app_counts):,} firm-year combinations")

        logger.info("✓ Pre-calculation complete\n")

        return firm_year_app_counts, firm_app_first_years

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
        task_df.columns = task_df.columns.str.strip()
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
        ratings_df.columns = ratings_df.columns.str.strip()
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

    def load_bls_soc_isco_crosswalk(self, bls_file_path: str) -> pd.DataFrame:
        """
        Load BLS 2010 SOC to ISCO-08 crosswalk.

        IMPORTANT: This crosswalk uses SOC 2010 taxonomy and is ONLY compatible
        with O*NET version 20. For O*NET 25+, use SOC 2018 crosswalk instead.

        Args:
            bls_file_path: Path to ISCO_SOC_Crosswalk.xls

        Returns:
            DataFrame with columns: soc_code, isco08_4d, isco08_title
        """
        logger.info(f"Loading BLS SOC-10 → ISCO-08 crosswalk from: {bls_file_path}")
        logger.info("NOTE: This crosswalk uses SOC 2010 taxonomy (compatible with O*NET v20 only)")

        # Load with header on row 6 (0-indexed)
        df = pd.read_excel(bls_file_path, header=6, dtype=str)
        logger.info(f"Loaded {len(df):,} raw rows from BLS crosswalk")

        # Clean column names
        df.columns = [c.strip().lower() for c in df.columns]
        logger.info(f"Column names: {list(df.columns)}")

        # Map to standard column names
        crosswalk = pd.DataFrame()
        crosswalk['soc_code'] = df['2010 soc code'].astype(str).str.strip()
        crosswalk['isco08_4d'] = df['isco-08 code'].astype(str).str.strip()
        crosswalk['isco08_title'] = df['isco-08 title en'].astype(str).str.strip()

        logger.info(f"Extracted columns: soc_code, isco08_4d, isco08_title")

        # Filter valid rows (non-null codes)
        rows_before = len(crosswalk)
        crosswalk = crosswalk.dropna(subset=['soc_code', 'isco08_4d'])
        logger.info(f"Removed {rows_before - len(crosswalk):,} rows with null SOC or ISCO codes")

        # Filter to valid 4-digit ISCO codes
        rows_before = len(crosswalk)
        crosswalk = crosswalk[crosswalk['isco08_4d'].str.len() == 4]
        logger.info(f"Removed {rows_before - len(crosswalk):,} rows with non-4-digit ISCO codes")

        # Deduplicate (handles 'part' column implicitly - treat all as full links)
        rows_before = len(crosswalk)
        crosswalk = crosswalk[['soc_code', 'isco08_4d', 'isco08_title']].drop_duplicates()
        logger.info(f"Removed {rows_before - len(crosswalk):,} duplicate SOC→ISCO mappings")

        logger.info(f"Final BLS crosswalk: {len(crosswalk):,} unique SOC-10 → ISCO-08 links")
        logger.info(f"  Unique SOC codes: {crosswalk['soc_code'].nunique():,}")
        logger.info(f"  Unique ISCO codes: {crosswalk['isco08_4d'].nunique():,}")
        logger.info(f"  Example mappings:")
        for idx, (_, row) in enumerate(crosswalk.head(3).iterrows()):
            logger.info(f"    {row['soc_code']} → {row['isco08_4d']} ({row['isco08_title']})")

        return crosswalk

    def load_employment_data(self, employment_file: str) -> pd.DataFrame:
        """
        Load BLS employment data for 6-digit SOC codes.

        Args:
            employment_file: Path to national_M2018_dl.xlsx (2018 employment data)

        Returns:
            DataFrame with columns: soc_6d, employment
        """
        logger.info(f"Loading BLS employment data from: {employment_file}")

        # Load employment file
        emp_df = pd.read_excel(employment_file, dtype={'OCC_CODE': str})
        logger.info(f"Loaded {len(emp_df):,} occupation employment records")

        # Clean column names
        emp_df.columns = [c.strip().upper() for c in emp_df.columns]

        # Extract relevant columns
        employment = pd.DataFrame()
        employment['soc_6d'] = emp_df['OCC_CODE'].astype(str).str.strip()

        # Handle employment (may have commas or be numeric)
        if emp_df['TOT_EMP'].dtype == 'object':
            # Remove commas and '#' symbols, convert to float
            employment['employment'] = emp_df['TOT_EMP'].str.replace(',', '').str.replace('#', '').astype(float)
        else:
            employment['employment'] = emp_df['TOT_EMP'].astype(float)

        # Remove summary rows (e.g., "00-0000" = All Occupations, "11-0000" = major groups)
        # Keep only detailed 6-digit occupations (XX-XXXX format)
        employment = employment[employment['soc_6d'].str.match(r'^\d{2}-\d{4}$')].copy()

        logger.info(f"Filtered to {len(employment):,} detailed 6-digit SOC occupations")
        logger.info(f"  Total employment in dataset: {employment['employment'].sum():,.0f}")
        logger.info(f"  Employment range: {employment['employment'].min():,.0f} - {employment['employment'].max():,.0f}")
        logger.info(f"  Median employment: {employment['employment'].median():,.0f}")

        return employment

    def load_webb_crosswalk_with_weights(self, webb_file: str, employment_data: pd.DataFrame, use_weights: bool) -> pd.DataFrame:
        """
        Load Webb crosswalk and add pre-normalized employment weights.

        Args:
            webb_file: Path to webb_crosswalk_clean.xls
            employment_data: BLS employment counts by 6-digit SOC
            use_weights: If True, add employment weights; if False, use equal weights

        Returns:
            DataFrame with columns: onet_8d, soc_6d, isco08_4d, isco08_title, weight
            Where weights sum to 1.0 within each isco08_4d group
        """
        logger.info(f"Loading Webb O*NET-SOC → ISCO-08 crosswalk from: {webb_file}")
        logger.info(f"  Weighting mode: {'Employment-weighted' if use_weights else 'Unweighted (equal weights)'}")

        # Load Webb crosswalk
        webb_df = pd.read_excel(webb_file, dtype=str)
        logger.info(f"Loaded {len(webb_df):,} raw crosswalk rows")

        # Clean column names
        webb_df.columns = [c.strip().lower() for c in webb_df.columns]

        # Extract relevant columns
        crosswalk = pd.DataFrame()
        crosswalk['onet_8d'] = webb_df['onetsoccode'].astype(str).str.strip()
        crosswalk['soc_6d'] = webb_df['onetsoccode_for_matching'].astype(str).str.strip()
        crosswalk['isco08_4d'] = webb_df['isco08'].astype(str).str.strip()
        crosswalk['isco08_title'] = webb_df['isco08_name'].astype(str).str.strip()

        logger.info(f"Extracted columns: onet_8d, soc_6d, isco08_4d, isco08_title")

        # Filter valid rows
        rows_before = len(crosswalk)
        crosswalk = crosswalk.dropna(subset=['onet_8d', 'soc_6d', 'isco08_4d'])
        logger.info(f"Removed {rows_before - len(crosswalk):,} rows with null codes")

        # Track ISCO code lengths (including non-4-digit codes)
        isco_lengths = crosswalk['isco08_4d'].str.len()
        non_4digit_rows = (isco_lengths != 4).sum()
        if non_4digit_rows > 0:
            logger.info(f"Found {non_4digit_rows} row(s) with non-4-digit ISCO codes")
            length_dist = isco_lengths.value_counts().sort_index()
            for length, count in length_dist.items():
                logger.info(f"  {length}-digit ISCO codes: {count} row(s)")

            # Show the non-4-digit codes
            non_4digit_codes = crosswalk[isco_lengths != 4][['onet_8d', 'soc_6d', 'isco08_4d']]
            logger.info(f"  Non-4-digit mappings:")
            for idx, row in non_4digit_codes.iterrows():
                logger.info(f"    {row['onet_8d']} → {row['soc_6d']} → {row['isco08_4d']} ({len(row['isco08_4d'])}-digit)")

        # Deduplicate
        rows_before = len(crosswalk)
        crosswalk = crosswalk.drop_duplicates(subset=['onet_8d', 'soc_6d', 'isco08_4d'])
        logger.info(f"Removed {rows_before - len(crosswalk):,} duplicate mappings")

        # Count 4-digit vs non-4-digit in final crosswalk
        final_isco_lengths = crosswalk['isco08_4d'].str.len()
        count_4digit = (final_isco_lengths == 4).sum()
        count_non4digit = (final_isco_lengths != 4).sum()

        logger.info(f"Final Webb crosswalk: {len(crosswalk):,} unique O*NET-8d → SOC-6d → ISCO links")
        logger.info(f"  Unique 8-digit O*NET codes: {crosswalk['onet_8d'].nunique():,}")
        logger.info(f"  Unique 6-digit SOC codes: {crosswalk['soc_6d'].nunique():,}")
        logger.info(f"  Unique ISCO codes: {crosswalk['isco08_4d'].nunique():,}")
        logger.info(f"  ISCO code lengths: {count_4digit} 4-digit + {count_non4digit} non-4-digit ({count_non4digit/len(crosswalk)*100:.1f}%)")

        # STEP: Add employment weights and pre-normalize
        if use_weights:
            logger.info("Adding employment weights to crosswalk...")

            # Merge with employment data (left join to keep all SOC codes)
            crosswalk = crosswalk.merge(employment_data, on='soc_6d', how='left')

            # Fill missing employment with median
            median_emp = employment_data['employment'].median()
            missing_emp = crosswalk['employment'].isna().sum()
            if missing_emp > 0:
                logger.warning(f"  {missing_emp} SOC-6d codes missing employment data - filling with median ({median_emp:,.0f})")
                crosswalk['employment'] = crosswalk['employment'].fillna(median_emp)

            logger.info(f"  Employment statistics: min={crosswalk['employment'].min():,.0f}, max={crosswalk['employment'].max():,.0f}, median={crosswalk['employment'].median():,.0f}")

            # PRE-NORMALIZE: Calculate weights that sum to 1.0 within each ISCO-4d group
            # Group by ISCO, sum employment, then divide each row's employment by group total
            crosswalk['total_employment_in_isco'] = crosswalk.groupby('isco08_4d')['employment'].transform('sum')
            crosswalk['weight'] = crosswalk['employment'] / crosswalk['total_employment_in_isco']

            # Drop temporary columns
            crosswalk.drop(columns=['employment', 'total_employment_in_isco'], inplace=True)

            logger.info("  Pre-normalized weights created (sum to 1.0 within each ISCO group)")

            # Validation: Check that weights sum to ~1.0 for each ISCO code
            weight_sums = crosswalk.groupby('isco08_4d')['weight'].sum()
            if not np.allclose(weight_sums, 1.0, rtol=1e-5):
                logger.warning(f"  WARNING: Some ISCO groups have weights that don't sum to 1.0! Range: {weight_sums.min():.6f} - {weight_sums.max():.6f}")
            else:
                logger.info(f"  ✓ Validation passed: All ISCO groups have weights summing to 1.0")

        else:
            # Unweighted mode: Use equal weights within each ISCO-4d group
            logger.info("Using equal weights (unweighted mode)...")

            # Count how many SOC-6d codes map to each ISCO-4d
            crosswalk['count_in_isco'] = crosswalk.groupby('isco08_4d')['soc_6d'].transform('nunique')
            crosswalk['weight'] = 1.0 / crosswalk['count_in_isco']

            crosswalk.drop(columns=['count_in_isco'], inplace=True)

            logger.info(f"  Equal weights assigned (sum to 1.0 within each ISCO group)")

        logger.info(f"  Example weighted mappings:")
        for idx, row in crosswalk.head(5).iterrows():
            logger.info(f"    {row['onet_8d']} → {row['soc_6d']} → {row['isco08_4d']} (weight={row['weight']:.4f})")

        return crosswalk

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
    
    def load_job_app_mapping(self, task_type: str = 'core', model: str = 'BGE') -> pd.DataFrame:
        """
        Load job-application mapping file to connect AI apps to specific jobs.

        Returns:
            DataFrame with app_text → job_uids mapping
        """
        logger.info("Loading job-application mapping...")

        # BGE percentile string (match stage_4 formatting: reverse-sorted)
        pct_list = getattr(self, 'bge_percentiles', None) or []
        if not pct_list:
            percentile_str = ""
        else:
            percentile_str = "_".join(
                str(int(p)) if isinstance(p, (int, float)) and p == int(p) else str(p)
                for p in sorted(pct_list, reverse=True)
            )

        # CE thresholds string (only include > 0.0 thresholds, format like 0p8)
        ce_list = getattr(self, 'ce_thresholds', None) or []
        ce_percentile_str = "_".join(f"{c:.1f}".replace('.', 'p') for c in sorted(ce_list, reverse=True) if c > 0.0)
        ce_part = f"_ce{ce_percentile_str}" if ce_percentile_str else ""

        # O*NET suffix
        onet_suffix = f"_onet{self.onet_version}" if getattr(self, 'onet_version', None) else ""

        # Task suffix
        task_suffix = f"_{task_type}"

        # Construct filename using same pattern as stage_4
        # e.g. task_exposure_matches_all_thresholds_openai_bge20_15_10_5_1_ce0p8_0p6_onet20_core.parquet
        bge_part = f"_bge{percentile_str}" if percentile_str else ""
        filename = f"job_app_mapping_{model}{bge_part}{ce_part}{onet_suffix}{task_suffix}.parquet"
        filepath = os.path.join(self.stage_4_dir, filename)

        if os.path.exists(filepath):
            logger.info(f"Using Stage 4 file: {filename}")
        else:
            logger.warning(f"  WARNING: Stage 4 file not found: {filename}")

        mapping_df = pd.read_parquet(self.stage_4_dir + filename)

        # Strictly require ai_app_id (no fallback hashing per user directive)
        if 'ai_app_id' not in mapping_df.columns:
            raise ValueError("ai_app_id missing from job_app_mapping parquet; regenerate Stage 4 outputs with ai_app_id included.")

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
            # Default to ai_development_deduplicated_custom.csv (Stage 2 output)
            # This is the primary deduplicated dataset used throughout the pipeline
            original_file_path = os.path.join(self.data_dir, "ai_development_deduplicated_custom.csv")
        
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
        original_df["tst_created"] = pd.to_datetime(original_df["tst_created"], errors="coerce", utc=True)
        
        # Rename uid to job_uid for consistency
        original_df.rename(columns={'uid': 'job_uid'}, inplace=True)
        
        logger.info(f"Loaded {len(original_df):,} original jobs from full dataset")
        logger.info(f"Unique companies: {original_df['company_name'].nunique():,}")
        logger.info(f"Year range: {original_df['year'].min()} - {original_df['year'].max()}")
        
        return original_df
    
    def load_stage2_deduplication_mapping(self) -> pd.DataFrame:
        """
        Load Stage 2 deduplication mapping to recover jobs removed by similarity deduplication.
        Only loads the UID columns needed for mapping.
        
        Args:
            stage2_mapping_file: Path to similar_duplicates_removed_* file
            
        Returns:
            DataFrame with kept_uid → removed_uid mapping for Stage 2 recovery
        """
        if self.stage_2_dir is None:
            # Auto-detect the most recent similar_duplicates_removed file
            stage2_files = list(Path(self.stage_2_dir).glob("similar_duplicates_removed_*.csv"))
            if not stage2_files:
                # Check for the standard filename
                standard_file = os.path.join(self.stage_2_dir, "similar_duplicates_removed.csv")
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
                                           stage2_mapping: pd.DataFrame = None) -> pd.DataFrame:
        """
        Expand the job→app mapping using only Stage 4 deduplication.
        Stage 2 deduplication reintegration has been removed.
        """
        logger.info("Expanding job-app mapping using Stage 4 deduplication only")

        # Stage 2 deduplication handling commented out - no longer needed
        # if stage2_mapping is not None and {'kept_uid','removed_uid'}.issubset(stage2_mapping.columns):
        #     rem2kept = stage2_mapping.set_index('removed_uid')['kept_uid'].to_dict()
        # else:
        #     rem2kept = {}

        # def to_kept(uid: str) -> str:
        #     return rem2kept.get(uid, uid)

        # Parse Stage 4 mapping - using original job UIDs directly
        jam = job_app_mapping.copy()
        jam['job_uids_list'] = jam['job_uids'].fillna('').astype(str).apply(
            lambda s: [u.strip() for u in s.split('|') if u.strip()]
        )

        # Keep full provenance (Stage 4 UIDs as they are)
        jam['job_uids_all'] = jam['job_uids_list']

        # Use original UIDs directly (no Stage 2 mapping needed)
        jam['job_uids_kept'] = jam['job_uids_list']

        # Explode UIDs for join; carry ai_app_id for downstream joins
        expanded = jam[['app_text', 'ai_app_id', 'job_uids_kept', 'first_occurrence_tst_created']].explode('job_uids_kept')
        expanded = expanded.rename(columns={'job_uids_kept': 'job_uid'})

        # Join to company data using original job UIDs
        merged = expanded.merge(
            original_jobs, how='left', on='job_uid', validate='many_to_one'
        )

        # Warn if any UIDs are missing from company data
        missing_uids = merged[merged['company_name'].isna()]['job_uid'].unique().tolist()
        if missing_uids:
            logger.warning(f"{len(missing_uids)} job UIDs not found in company data. First few: {missing_uids[:10]}")

        # Attach provenance as a pipe-joined string
        jam['provenance_job_uids'] = jam['job_uids_all'].apply(lambda L: '|'.join(L))
        prov = jam[['app_text', 'provenance_job_uids']].drop_duplicates()

        merged = merged.merge(prov, on='app_text', how='left')

        # Keep only rows that successfully joined
        out = merged.dropna(subset=['company_name']).copy()
        out = out[['app_text', 'ai_app_id', 'job_uid', 'company_name', 'x28_occupations', 'x28_industries',
                'year', 'tst_created', 'first_occurrence_tst_created', 'provenance_job_uids']]

        logger.info(f"Final expansion (Stage 4 deduplication only): {out['job_uid'].nunique():,} jobs, {len(out):,} app-job rows")
        return out

    def _add_spec_suffix_to_columns(self, df: pd.DataFrame, bge_col: str, ce_col: str) -> pd.DataFrame:
        """
        Add specification suffix to exposure columns before merging.

        Args:
            df: DataFrame with exposure columns
            bge_col: BGE percentile column name (e.g., 'pct_20')
            ce_col: CE threshold column name (e.g., 'ce_0.8')

        Returns:
            DataFrame with renamed exposure columns
        """
        # Extract spec identifiers: 'pct_20' -> 'p20', 'ce_0.8' -> 'ce08'
        pct_id = bge_col.replace('pct_', 'p')
        ce_id = ce_col.replace('ce_', 'ce').replace('.', '')
        suffix = f"_{pct_id}_{ce_id}"

        # Columns to rename (exposure metrics that vary by specification)
        exposure_columns = [
            'hampole_occupation_exposure',
            'binary_occupation_exposure',
            'log_ai_intensity',
            'n_ai_apps_firm_year',
            'hampole_ai_exposure_avg',
            'binary_ai_exposure_avg'
        ]

        # Build rename mapping
        rename_map = {col: f"{col}{suffix}" for col in exposure_columns if col in df.columns}

        # Apply renaming
        df_renamed = df.rename(columns=rename_map)

        logger.debug(f"  Renamed {len(rename_map)} columns with suffix {suffix}")
        return df_renamed

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
        logger.info(f"Step 2: Calculating task exposure at firm level (time_invariant={self.time_invariant}, exposure_table_format={self.exposure_table_format})...")
        logger.info(f"Using expanded dataset with Stage 4 deduplication only (Stage 2 deduplication temporarily disabled)...")
        
        # Stage 2 deduplication reintegration commented out - only using Stage 4 deduplication
        # stage2_mapping = self.load_stage2_deduplication_mapping(stage2_mapping_file)

        # Only expand job-app mapping using Stage 4 deduplication (already in job_app_mapping)
        # Stage 2 deduplication reintegration removed per user request
        expanded_job_app_mapping = self.expand_job_app_mapping_to_full_dataset(
            job_app_mapping, original_jobs, None
        )
        if self.save_debug_csv:
            expanded_job_app_mapping.to_csv(os.path.join(self.output_dir, 'expanded_jobs_matching.csv'), index=False)
            task_app_matches.to_csv(os.path.join(self.output_dir, 'task_app_matches.csv'), index=False)

        # Explode job_uid array to one row per job
        # Stage 4 stores job_uid as arrays since multiple jobs can use the same AI app text
        logger.info(f"Exploding task-app matches: converting job_uid arrays to individual rows")
        task_app_matches_exploded = task_app_matches.copy()

        # Check if job_uid is an array/list and explode it
        def extract_job_uid(x):
            if isinstance(x, (list, np.ndarray)):
                return list(x) if len(x) > 0 else [np.nan]
            return [x]

        task_app_matches_exploded['job_uid'] = task_app_matches_exploded['job_uid'].apply(extract_job_uid)
        task_app_matches_exploded = task_app_matches_exploded.explode('job_uid').reset_index(drop=True)
        task_app_matches_exploded['job_uid'] = task_app_matches_exploded['job_uid'].astype(str)

        logger.info(f"Exploded task-app matches from {len(task_app_matches):,} to {len(task_app_matches_exploded):,} rows")

        # Join task-app matches with expanded job mapping
        # Now both have job_uid as strings, so we can join on both keys
        task_job_matches = task_app_matches_exploded.merge(
            expanded_job_app_mapping[['app_text', 'job_uid', 'company_name', 'x28_occupations', 'x28_industries', 'year']],
            on=['app_text', 'job_uid'],
            how='left',
            validate='many_to_one'
        )

        # The expanded mapping already contains company and time info, so we use it directly
        task_firm_matches = task_job_matches.copy()
        
        logger.info(f"Mapped {len(task_firm_matches):,} task-app-firm relationships")

        # COMPREHENSIVE VALIDATION: Check for UID loss between expanded mapping and task-firm matches
        verbose_validation = os.environ.get("VERBOSE_VALIDATION", "").strip().lower() in {"1", "true", "yes", "y"}

        expanded_uids = set(expanded_job_app_mapping['job_uid'].dropna().unique())
        task_firm_uids = set(task_firm_matches['job_uid'].dropna().unique())
        missing_in_task_firm = expanded_uids - task_firm_uids

        uid_loss_verdict = "none"
        uid_loss_is_expected_filtering = False

        if missing_in_task_firm:
            # Get all apps from missing UIDs (from expanded mapping, not from task matches)
            missing_apps_df = expanded_job_app_mapping[expanded_job_app_mapping['job_uid'].isin(missing_in_task_firm)]
            missing_app_texts = missing_apps_df['app_text'].dropna().unique()

            # Reason coding
            original_uids = set(original_jobs['job_uid'].dropna().unique()) if 'job_uid' in original_jobs.columns else set()
            missing_from_original = missing_in_task_firm - original_uids

            task_matching_apps = set(task_app_matches['app_text'].dropna().unique())
            apps_that_should_match = set(missing_app_texts) & task_matching_apps

            n_missing = len(missing_in_task_firm)
            n_expanded = max(len(expanded_uids), 1)
            pct_missing = 100.0 * (n_missing / n_expanded)

            # Canonical, reason-coded summary (single source of truth)
            if missing_from_original:
                uid_loss_verdict = "error_data_source_mismatch"
                logger.error(
                    "VALIDATION_UID_LOSS: ERROR — %s/%s jobs (%.2f%%) are missing from final task-firm matches, "
                    "and %s of those job_uids do not exist in original_jobs (data source mismatch). "
                    "ACTION: verify the original_jobs extract used here matches Stage 4 inputs (same source + time window). "
                    "Examples: %s",
                    f"{n_missing:,}", f"{n_expanded:,}", pct_missing,
                    f"{len(missing_from_original):,}",
                    list(sorted(missing_from_original))[:10],
                )
            elif apps_that_should_match:
                uid_loss_verdict = "error_possible_join_issue"
                logger.error(
                    "VALIDATION_UID_LOSS: ERROR — %s/%s jobs (%.2f%%) are missing from final task-firm matches, "
                    "and %s associated app_text values *do* appear in task_app_matches (possible join/key issue). "
                    "ACTION: inspect join keys and normalization for ('app_text','job_uid') across exploded matches vs expanded mapping. "
                    "Examples: %s",
                    f"{n_missing:,}", f"{n_expanded:,}", pct_missing,
                    f"{len(apps_that_should_match):,}",
                    list(sorted(apps_that_should_match))[:5],
                )
            else:
                uid_loss_verdict = "expected_filtering_no_task_match"
                uid_loss_is_expected_filtering = True
                logger.info(
                    "VALIDATION_UID_LOSS: OK — %s/%s jobs (%.2f%%) are missing from final task-firm matches, "
                    "but this is expected filtering: those jobs' AI apps have zero O*NET task matches. "
                    "(%s unique AI apps across missing jobs). ACTION: none (expected).",
                    f"{n_missing:,}", f"{n_expanded:,}", pct_missing,
                    f"{len(missing_app_texts):,}",
                )

            # Optional detail for inspection (only when explicitly enabled, or when verdict indicates an error)
            if verbose_validation or not uid_loss_is_expected_filtering:
                logger.info(
                    "VALIDATION_UID_LOSS_DETAILS: showing examples (set VERBOSE_VALIDATION=1 for more routine detail)."
                )
                for uid in list(sorted(missing_in_task_firm))[:5]:
                    uid_apps = missing_apps_df[missing_apps_df['job_uid'] == uid]['app_text'].dropna().tolist()
                    logger.info("  UID %s: %s app_text values (examples below)", uid, f"{len(uid_apps):,}")
                    for app in uid_apps[:2]:
                        logger.info("    - %s...", str(app)[:120])
        else:
            logger.info("VALIDATION_UID_LOSS: OK — no jobs were lost between expanded mapping and final task-firm matches.")

        # Only log company info if company_name column exists
        if 'company_name' in task_firm_matches.columns:
            logger.info(f"Unique firms: {task_firm_matches['company_name'].nunique():,}")
        else:
            logger.info("Company name column not available yet")
        
        # CRITICAL VALIDATION: Ensure data consistency across the pipeline
        logger.info("="*50)
        logger.info("DATA CONSISTENCY VALIDATION")
        logger.info("="*50)

        # Stage 2 validation logic commented out since stage2_mapping is disabled
        # if stage2_mapping is not None and {'kept_uid','removed_uid'}.issubset(stage2_mapping.columns):
        #     rem2kept = stage2_mapping.set_index('removed_uid')['kept_uid'].to_dict()
        # else:
        #     rem2kept = {}
        # to_kept = lambda u: rem2kept.get(u, u)

        # Use Stage-4 UID list directly (no stage 2 normalization)
        stage4_jobs_kept_set = set()
        for row in job_app_mapping.itertuples():
            for u in str(row.job_uids).split('|'):
                u = u.strip()
                if not u:
                    continue
                stage4_jobs_kept_set.add(u)  # Use UID directly, no stage2 mapping

        stage4_jobs = len(stage4_jobs_kept_set)

        expanded_jobs = expanded_job_app_mapping['job_uid'].nunique()
        final_jobs_with_tasks = task_firm_matches['job_uid'].nunique()
        
        logger.info(f"Jobs with AI apps (Stage 4 mapping): {stage4_jobs:,}")
        logger.info(f"Jobs after deduplication recovery: {expanded_jobs:,}")
        logger.info(f"Jobs in final task-firm matches: {final_jobs_with_tasks:,}")
        
        # Validation checks
        if expanded_jobs < stage4_jobs:
            logger.warning(f"WARNING: Lost {stage4_jobs - expanded_jobs:,} jobs during expansion. "
                          f"This could indicate missing job UIDs in company data.")
        
        # Check for any jobs accidentally INCLUDED without AI-task provenance
        allowed_jobs = set(expanded_job_app_mapping['job_uid'].dropna().unique())
        final_jobs   = set(task_firm_matches['job_uid'].dropna().unique())

        if final_jobs_with_tasks != expanded_jobs:
            # Avoid repeating the canonical UID-loss warning above if it's the same issue.
            if missing_in_task_firm and missing_in_task_firm == (allowed_jobs - final_jobs) and uid_loss_is_expected_filtering:
                logger.info(
                    "DATA_CONSISTENCY: Job count mismatch (%s vs %s) is already explained above by expected filtering "
                    "(jobs with AI apps but zero O*NET task matches).",
                    f"{expanded_jobs:,}", f"{final_jobs_with_tasks:,}",
                )
            else:
                logger.warning(
                    "WARNING: Job count mismatch between expanded mapping (%s) and final task matches (%s). "
                    "This may indicate task-app matching loss beyond expected filtering.",
                    f"{expanded_jobs:,}", f"{final_jobs_with_tasks:,}",
                )

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
            # De-duplicate: if this is the same missing set already summarized above, don't re-warn.
            if missing_in_task_firm and missing_expected == missing_in_task_firm:
                if uid_loss_is_expected_filtering:
                    logger.info(
                        "DATA_CONSISTENCY: %s job(s) missing in final matches (same set as VALIDATION_UID_LOSS above; expected filtering).",
                        f"{len(missing_expected):,}",
                    )
                else:
                    logger.warning(
                        "DATA_CONSISTENCY: %s job(s) missing in final matches (same set as VALIDATION_UID_LOSS above; verdict=%s).",
                        f"{len(missing_expected):,}",
                        uid_loss_verdict,
                    )
            else:
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

            # VECTORIZED: Use groupby aggregations instead of nested loops
            firm_matched = []

            for firm in all_firms:
                # Get ALL apps this specific firm has ever used across all years
                firm_apps_ever = set(task_firm_matches[task_firm_matches['company_name'] == firm]['app_text'].unique())

                if len(firm_apps_ever) == 0:
                    continue  # Skip firms with no AI apps

                # Get all tasks that match any of this firm's apps (across all time)
                firm_app_task_matches = task_app_matches[task_app_matches['app_text'].isin(firm_apps_ever)]

                if len(firm_app_task_matches) == 0:
                    continue  # Skip firms with no task matches

                # VECTORIZED: Count unique apps per task in ONE groupby operation (not nested in loops!)
                task_counts = firm_app_task_matches.groupby('onet_task_id')['app_text'].nunique().reset_index()
                task_counts.columns = ['onet_task_id', 'n_task_matches_firm_year']

                n_firm_apps_ever = len(firm_apps_ever)
                task_counts['company_name'] = firm
                task_counts['n_ai_apps_firm_year'] = n_firm_apps_ever
                task_counts['hampole_task_exposure'] = task_counts['n_task_matches_firm_year'] / n_firm_apps_ever
                task_counts['binary_task_exposure'] = (task_counts['n_task_matches_firm_year'] >= 1).astype(int)

                firm_matched.append(task_counts)

            if firm_matched:
                # Combine all firm task counts
                firm_year_task_counts = pd.concat(firm_matched, ignore_index=True)

                # VECTORIZED: Broadcast across all years with cross join (no loop!)
                years_df = pd.DataFrame({'year': all_years})
                exposure_df = firm_year_task_counts.merge(
                    years_df,
                    how='cross'
                )[['company_name', 'year', 'onet_task_id', 'n_ai_apps_firm_year',
                   'n_task_matches_firm_year', 'hampole_task_exposure', 'binary_task_exposure']]

                logger.info(f"Vectorized time-invariant calculation: {len(exposure_df):,} firm-year-task combinations")
            else:
                exposure_df = pd.DataFrame(columns=[
                    'company_name', 'year', 'onet_task_id', 'n_ai_apps_firm_year',
                    'n_task_matches_firm_year', 'hampole_task_exposure', 'binary_task_exposure'
                ])
        else:
            logger.info(f"Time-variant mode: Apps available to firms from first appearance onwards")

            # VECTORIZED: Pre-compute first-year using groupby (not nested loops!)
            firm_app_first_years = (task_firm_matches
                .groupby(['company_name', 'app_text'])['year']
                .min()
                .reset_index()
                .rename(columns={'year': 'first_year'})
            )

            logger.info(f"Pre-computed first-year mapping for {len(firm_app_first_years):,} firm-app pairs")

            # Calculate firm-year-level AI app counts for logging
            firm_year_app_counts = task_firm_matches.groupby(['company_name', 'year'])['app_text'].nunique().reset_index()
            firm_year_app_counts.rename(columns={'app_text': 'n_ai_apps_firm_year'}, inplace=True)

            logger.info(f"Time-variant mode: AI apps per firm-year - Mean: {firm_year_app_counts['n_ai_apps_firm_year'].mean():.1f}, "
                       f"Max: {firm_year_app_counts['n_ai_apps_firm_year'].max():,}")
            logger.info(f"Unique firm-year combinations: {len(firm_year_app_counts):,}")

            # VECTORIZED: Only loop over firm-years, not firm-year-tasks
            firm_matched = []

            for firm in all_firms:
                # Get apps for this firm with their first years
                firm_app_years = firm_app_first_years[firm_app_first_years['company_name'] == firm]

                if len(firm_app_years) == 0:
                    continue  # Skip firms with no apps

                for year in all_years:
                    # Get apps available by this year for this firm (vectorized filter)
                    available_apps = firm_app_years[firm_app_years['first_year'] <= year]['app_text'].tolist()

                    if not available_apps:
                        continue  # Skip firm-years with no available apps

                    n_apps_firm_year = len(available_apps)

                    # Get all tasks that match available apps
                    available_task_matches = task_app_matches[task_app_matches['app_text'].isin(available_apps)]

                    if len(available_task_matches) == 0:
                        continue  # Skip if no task matches

                    # VECTORIZED: Count unique apps per task in ONE groupby operation (not per task!)
                    task_counts = available_task_matches.groupby('onet_task_id')['app_text'].nunique().reset_index()
                    task_counts.columns = ['onet_task_id', 'n_task_matches_firm_year']

                    task_counts['company_name'] = firm
                    task_counts['year'] = year
                    task_counts['n_ai_apps_firm_year'] = n_apps_firm_year
                    task_counts['hampole_task_exposure'] = task_counts['n_task_matches_firm_year'] / n_apps_firm_year
                    task_counts['binary_task_exposure'] = (task_counts['n_task_matches_firm_year'] >= 1).astype(int)

                    firm_matched.append(task_counts)

            if firm_matched:
                exposure_df = pd.concat(firm_matched, ignore_index=True)[
                    ['company_name', 'year', 'onet_task_id', 'n_ai_apps_firm_year',
                     'n_task_matches_firm_year', 'hampole_task_exposure', 'binary_task_exposure']
                ]

                logger.info(f"Vectorized time-variant calculation: {len(exposure_df):,} firm-year-task combinations")
            else:
                exposure_df = pd.DataFrame(columns=[
                    'company_name', 'year', 'onet_task_id', 'n_ai_apps_firm_year',
                    'n_task_matches_firm_year', 'hampole_task_exposure', 'binary_task_exposure'
                ])
        
        # Apply occupation-level exposure if specified
        if self.exposure_table_format != "exposed-firm-occ-year":
            logger.info(f"Applying exposure table format ({self.exposure_table_format}) - overriding firm-level logic...")
            exposure_df = self._apply_occupation_exposure(exposure_df, task_app_matches, task_firm_matches)
        
        logger.info(f"Calculated task-firm-year exposure: {len(exposure_df):,} task-firm-year combinations")
        if len(exposure_df) > 0:
            logger.info(f"  Hampole exposure range: {exposure_df['hampole_task_exposure'].min():.3f} - {exposure_df['hampole_task_exposure'].max():.3f}")
            logger.info(f"  Binary exposure - firm-years with exposed tasks: {(exposure_df['binary_task_exposure'] > 0).sum():,}")
            logger.info(f"  Unique firms: {exposure_df['company_name'].nunique():,}")
            logger.info(f"  Unique years: {sorted(exposure_df['year'].unique())}")

        return exposure_df

    def step2_calculate_firm_task_exposure_chunked(self,
                                                   task_app_matches_chunk: pd.DataFrame,
                                                   expanded_job_app_mapping: pd.DataFrame,
                                                   firm_year_app_counts: dict,
                                                   firm_app_first_years: pd.DataFrame = None) -> pd.DataFrame:
        """
        Chunked version of step2 that processes a single chunk of task_app_matches.

        Args:
            task_app_matches_chunk: Single chunk of task-application matches
            expanded_job_app_mapping: Pre-loaded expanded job mapping (shared across chunks)
            firm_year_app_counts: Pre-calculated {(firm, year): n_apps} dictionary
            firm_app_first_years: Pre-calculated first-year mapping (for time-variant mode)

        Returns:
            DataFrame with partial task-firm exposure (to be aggregated later)
        """
        # Explode job_uid arrays
        task_app_matches_exploded = task_app_matches_chunk.copy()

        def extract_job_uid(x):
            if isinstance(x, (list, np.ndarray)):
                return list(x) if len(x) > 0 else [np.nan]
            return [x]

        task_app_matches_exploded['job_uid'] = task_app_matches_exploded['job_uid'].apply(extract_job_uid)
        task_app_matches_exploded = task_app_matches_exploded.explode('job_uid').reset_index(drop=True)
        task_app_matches_exploded['job_uid'] = task_app_matches_exploded['job_uid'].astype(str)

        # Join with expanded job mapping
        task_job_matches = task_app_matches_exploded.merge(
            expanded_job_app_mapping[['app_text', 'job_uid', 'company_name', 'year']],
            on=['app_text', 'job_uid'],
            how='left',
            validate='many_to_one'
        )

        task_firm_matches = task_job_matches.copy()

        if self.time_invariant:
            # Time-invariant: Use all apps firm has ever used
            all_firms = task_firm_matches['company_name'].unique()
            firm_matched = []

            for firm in all_firms:
                firm_apps_ever = set(task_firm_matches[task_firm_matches['company_name'] == firm]['app_text'].unique())

                if len(firm_apps_ever) == 0:
                    continue

                # Get tasks matching this firm's apps FROM THIS CHUNK
                firm_app_task_matches = task_app_matches_chunk[task_app_matches_chunk['app_text'].isin(firm_apps_ever)]

                if len(firm_app_task_matches) == 0:
                    continue

                # Count unique apps per task
                task_counts = firm_app_task_matches.groupby('onet_task_id')['app_text'].nunique().reset_index()
                task_counts.columns = ['onet_task_id', 'n_task_matches_firm_year']

                task_counts['company_name'] = firm
                # Add year column for time-invariant mode (will be broadcast to all years later)
                # We'll add n_ai_apps_firm_year after aggregation since it's same for all years
                # Store raw counts; exposure will be calculated after aggregation
                firm_matched.append(task_counts)

            if firm_matched:
                chunk_exposure = pd.concat(firm_matched, ignore_index=True)
                # For time-invariant, n_ai_apps_firm_year will be added during aggregation
            else:
                chunk_exposure = pd.DataFrame(columns=['company_name', 'onet_task_id', 'n_task_matches_firm_year'])

        else:
            # Time-variant: Use apps available by that year
            firm_matched = []
            all_firms = task_firm_matches['company_name'].unique()
            all_years = sorted(task_firm_matches['year'].unique())

            for firm in all_firms:
                # Get apps for this firm with their first years (from pre-calculated mapping)
                firm_app_years = firm_app_first_years[firm_app_first_years['company_name'] == firm]

                if len(firm_app_years) == 0:
                    continue

                for year in all_years:
                    # Get apps available by this year
                    available_apps = firm_app_years[firm_app_years['first_year'] <= year]['app_text'].tolist()

                    if not available_apps:
                        continue

                    # Get tasks matching available apps FROM THIS CHUNK
                    available_task_matches = task_app_matches_chunk[task_app_matches_chunk['app_text'].isin(available_apps)]

                    if len(available_task_matches) == 0:
                        continue

                    # Count unique apps per task
                    task_counts = available_task_matches.groupby('onet_task_id')['app_text'].nunique().reset_index()
                    task_counts.columns = ['onet_task_id', 'n_task_matches_firm_year']

                    task_counts['company_name'] = firm
                    task_counts['year'] = year
                    # Add pre-calculated n_ai_apps_firm_year
                    task_counts['n_ai_apps_firm_year'] = firm_year_app_counts.get((firm, year), 0)
                    # Store raw counts; exposure will be calculated after aggregation
                    firm_matched.append(task_counts)

            if firm_matched:
                chunk_exposure = pd.concat(firm_matched, ignore_index=True)
            else:
                chunk_exposure = pd.DataFrame(columns=['company_name', 'year', 'onet_task_id', 'n_task_matches_firm_year', 'n_ai_apps_firm_year'])

        return chunk_exposure
    
    def _apply_occupation_exposure(self, firm_exposure_df: pd.DataFrame,
                                 task_app_matches: pd.DataFrame,
                                 task_firm_matches: pd.DataFrame) -> pd.DataFrame:
        """
        Apply occupation-level exposure logic with support for both firm-specific and pure occupation modes.

        This method supports two approaches:
        1. Firm-specific: Each firm maintains its own AI portfolio within occupation framework
        2. Pure occupation: All firms get identical exposure patterns for occupation-level analysis

        Args:
            firm_exposure_df: Firm-level exposure from previous calculation
            task_app_matches: Task-application matches from Step 1
            task_firm_matches: Task-app-firm relationships with years

        Returns:
            DataFrame with occupation-level exposure applied
        """
        logger.info(f"Exposure table format: {self.exposure_table_format}")

        # Get all firms and years from the data
        all_firms = task_firm_matches['company_name'].unique()
        all_years = sorted(task_firm_matches['year'].unique())
        all_apps = task_app_matches['app_text'].unique()
        all_tasks = task_app_matches['onet_task_id'].unique()

        # Handle the 3 exposure table format options: exposed-firm-occ-year, all-firm-occ-year, exposed-occ-year
        if self.exposure_table_format == "exposed-firm-occ-year":
            logger.info(f"Skipping exposure expansion (keeping firm-level only with exposure > 0)")
            return firm_exposure_df  # Return original firm-level data unchanged
        elif self.exposure_table_format == "exposed-occ-year":
            logger.info(f"Using occupation-year only (constant across firms, time_invariant={self.time_invariant})")
            return self._apply_pure_occupation_exposure(
                task_app_matches, task_firm_matches, all_firms, all_years, all_tasks, all_apps
            )
        elif self.exposure_table_format == "all-firm-occ-year":
            logger.info(f"Using all firm × occupation × year combinations (includes zeros, time_invariant={self.time_invariant})")
            return self._apply_firm_specific_occupation_exposure(
                task_app_matches, task_firm_matches, all_firms, all_years, all_tasks
            )
        else:
            # Backward compatibility - default to all-firm-occ-year for other values
            logger.warning(f"Unknown exposure_table_format value '{self.exposure_table_format}'. Defaulting to 'all-firm-occ-year'.")
            return self._apply_firm_specific_occupation_exposure(
                task_app_matches, task_firm_matches, all_firms, all_years, all_tasks
            )

    def _apply_firm_specific_occupation_exposure(self, task_app_matches: pd.DataFrame,
                                               task_firm_matches: pd.DataFrame,
                                               all_firms: list, all_years: list, all_tasks: list) -> pd.DataFrame:
        """Apply occupation exposure while maintaining firm-specific AI portfolios."""

        # Build firm-specific app portfolios with first appearance tracking
        firm_app_first_year = {}
        for firm in all_firms:
            firm_data = task_firm_matches[task_firm_matches['company_name'] == firm]
            firm_app_first_year[firm] = {}
            for app in firm_data['app_text'].unique():
                app_years = firm_data[firm_data['app_text'] == app]['year']
                if len(app_years) > 0:
                    firm_app_first_year[firm][app] = app_years.min()

        # Create firm-specific occupation-level exposure
        occupation_task_exposure = []

        for firm in all_firms:
            firm_apps_by_first_year = firm_app_first_year[firm]
            if not firm_apps_by_first_year:
                continue  # Skip firms with no AI apps

            for year in all_years:
                # Get firm-specific apps available by this year based on time_invariant parameter
                if self.time_invariant:
                    available_firm_apps = list(firm_apps_by_first_year.keys())
                else:  # time-variant
                    available_firm_apps = [app for app, first_year in firm_apps_by_first_year.items()
                                         if first_year <= year]

                if not available_firm_apps:
                    continue

                n_apps_firm_year = len(available_firm_apps)

                # For each task, check if any of this firm's available apps match it
                for task_id in all_tasks:
                    task_matches = task_app_matches[
                        (task_app_matches['onet_task_id'] == task_id) &
                        (task_app_matches['app_text'].isin(available_firm_apps))
                    ]
                    matching_firm_apps = task_matches['app_text'].unique()
                    n_matches = len(matching_firm_apps)

                    if n_matches > 0:
                        hampole_exposure = n_matches / n_apps_firm_year if n_apps_firm_year > 0 else 0
                        binary_exposure = 1 if n_matches >= 1 else 0

                        occupation_task_exposure.append({
                            'company_name': firm,
                            'year': year,
                            'onet_task_id': task_id,
                            'n_ai_apps_firm_year': n_apps_firm_year,
                            'n_task_matches_firm_year': n_matches,
                            'hampole_task_exposure': hampole_exposure,
                            'binary_task_exposure': binary_exposure
                        })

        return pd.DataFrame(occupation_task_exposure)

    def _apply_pure_occupation_exposure(self, task_app_matches: pd.DataFrame,
                                      task_firm_matches: pd.DataFrame,
                                      all_firms: list, all_years: list, all_tasks: list,
                                      all_apps: list) -> pd.DataFrame:
        """Apply pure occupation-level exposure where all firms get identical patterns."""

        # Find first appearance year of each AI application (for time-variant mode)
        app_first_year = {}
        if not self.time_invariant:  # time-variant
            for app in all_apps:
                app_years = task_firm_matches[task_firm_matches['app_text'] == app]['year'].dropna()
                if len(app_years) > 0:
                    app_first_year[app] = app_years.min()
                else:
                    app_first_year[app] = float('inf')

        logger.info(f"Pure occupation mode: All firms exposed to all {len(all_apps)} AI applications")

        # Create pure occupation-level exposure (all firms identical)
        occupation_task_exposure = []

        for firm in all_firms:
            for year in all_years:
                for task_id in all_tasks:
                    # Count how many apps (across ALL firms) match this task
                    task_matches_global = task_app_matches[task_app_matches['onet_task_id'] == task_id]
                    matching_apps = task_matches_global['app_text'].unique()

                    if self.time_invariant:
                        available_apps = matching_apps
                        contextual_app_count = len(all_apps)
                    else:  # time-variant
                        available_apps = [app for app in matching_apps if app_first_year.get(app, float('inf')) <= year]
                        available_apps_by_year = [app for app in all_apps if app_first_year.get(app, float('inf')) <= year]
                        contextual_app_count = len(available_apps_by_year)

                    n_matches = len(available_apps)

                    if n_matches > 0:
                        # Pure occupation mode: exposure based on all available apps
                        hampole_exposure = n_matches / contextual_app_count if contextual_app_count > 0 else 0
                        binary_exposure = 1 if n_matches >= 1 else 0

                        occupation_task_exposure.append({
                            'company_name': firm,
                            'year': year,
                            'onet_task_id': task_id,
                            'n_ai_apps_firm_year': contextual_app_count,  # Same for all firms
                            'n_task_matches_firm_year': n_matches,
                            'hampole_task_exposure': hampole_exposure,
                            'binary_task_exposure': binary_exposure
                        })

        new_exposure_df = pd.DataFrame(occupation_task_exposure)
        logger.info(f"Pure occupation-level exposure created {len(new_exposure_df):,} task-firm-year combinations")

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

        # CRITICAL FIX: Filter task_weights to only include Core tasks (consistent with task_statements)
        # This ensures total_tasks_occupation counts only Core tasks, matching manual filters
        core_task_ids = set(task_statements['Task ID'].unique())
        task_weights_core = task_weights[task_weights['onet_task_id'].isin(core_task_ids)].copy()
        logger.info(f"Filtered task weights from {len(task_weights):,} to {len(task_weights_core):,} Core tasks only")
        task_weights = task_weights_core
        
        # Join task exposure with occupation and weight information
        task_exposure_weighted = task_firm_exposure.merge(
            task_onet_mapping, on='onet_task_id', how='left'
        )

        # SAFETY: ensure onet_code exists after merge (handle suffixes or weird column names)
        if 'onet_code' not in task_exposure_weighted.columns:
            for alt in [
                'onet_code_x', 'onet_code_y',
                'O*NET-SOC Code', 'O*NET-SOC Code_x', 'O*NET-SOC Code_y'
            ]:
                if alt in task_exposure_weighted.columns:
                    task_exposure_weighted['onet_code'] = task_exposure_weighted[alt]
                    break
        if 'onet_code' not in task_exposure_weighted.columns:
            raise KeyError(f"onet_code missing after task mapping; cols={list(task_exposure_weighted.columns)}")

        task_exposure_weighted = task_exposure_weighted.merge(
            task_weights, on=['onet_code', 'onet_task_id'], how='left'
        )
        
        # Fill missing weights with 3.0
        task_exposure_weighted['importance_weight'] = task_exposure_weighted['importance_weight'].fillna(3.0)
        
        logger.info(f"Mapped {len(task_exposure_weighted):,} task-firm-year-occupation relationships")
        
        # Calculate weighted occupation-firm-year exposure scores
        def calc_weighted_occupation_exposure(group):
            # Get the occupation code for this group
            # NOTE: pandas 3.0+ may exclude group keys from columns in .apply()
            # so use group.name (tuple of keys) as the primary source.
            try:
                onet_code, company_name, year = group.name
            except Exception:
                onet_code = group['onet_code'].iloc[0]
                company_name = group['company_name'].iloc[0]
                year = group['year'].iloc[0]

            # Get ALL tasks for this occupation (not just AI-exposed ones)
            all_occ_tasks = task_weights[task_weights['onet_code'] == onet_code].copy()

            if len(all_occ_tasks) == 0:
                # Fallback if no tasks found - should not happen but defensive
                weights = group['importance_weight']
                total_weight = weights.sum()
                hampole_weighted = (group['hampole_task_exposure'] * weights).sum() / total_weight if total_weight > 0 else 0
                binary_weighted = (group['binary_task_exposure'] * weights).sum() / total_weight if total_weight > 0 else 0
                return pd.Series({
                    'hampole_occupation_exposure': hampole_weighted,
                    'binary_occupation_exposure': binary_weighted,
                    'total_tasks_occupation': len(group),
                    'total_importance_weight': total_weight,
                    'n_ai_apps_firm_year': group['n_ai_apps_firm_year'].iloc[0]
                })

            # Create full task exposure table for this occupation
            # Start with all tasks (exposure = 0 by default)
            all_occ_tasks['hampole_task_exposure'] = 0.0
            all_occ_tasks['binary_task_exposure'] = 0.0

            # Update exposure for AI-exposed tasks
            for _, exposed_task in group.iterrows():
                mask = all_occ_tasks['onet_task_id'] == exposed_task['onet_task_id']
                all_occ_tasks.loc[mask, 'hampole_task_exposure'] = exposed_task['hampole_task_exposure']
                all_occ_tasks.loc[mask, 'binary_task_exposure'] = exposed_task['binary_task_exposure']

            # Calculate proper weighted averages using ALL tasks
            weights = all_occ_tasks['importance_weight']
            total_weight = weights.sum()

            # Weighted average formula: Σ(task_exposure × importance_weight) / Σ(importance_weight)
            hampole_weighted = (all_occ_tasks['hampole_task_exposure'] * weights).sum() / total_weight if total_weight > 0 else 0
            binary_weighted = (all_occ_tasks['binary_task_exposure'] * weights).sum() / total_weight if total_weight > 0 else 0

            return pd.Series({
                'hampole_occupation_exposure': hampole_weighted,
                'binary_occupation_exposure': binary_weighted,
                'total_tasks_occupation': len(all_occ_tasks),  # Count ALL tasks, not just exposed
                'total_importance_weight': total_weight,  # Sum ALL importance weights
                'n_ai_apps_firm_year': group['n_ai_apps_firm_year'].iloc[0]
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
                              webb_crosswalk: pd.DataFrame,
                              isco_titles: pd.DataFrame,
                              company_mapping: pd.DataFrame = None,
                              use_weights: bool = False) -> pd.DataFrame:
        """
        Two-mode crosswalk: O*NET-SOC → ISCO-08.

        Unweighted mode: 8-digit O*NET → 4-digit ISCO (simple mean aggregation)
        Weighted mode: 8-digit O*NET → 6-digit SOC → 4-digit ISCO (employment-weighted)

        Args:
            onet_exposure: O*NET occupation-firm exposure scores
            webb_crosswalk: Webb crosswalk with pre-normalized weights (from load_webb_crosswalk_with_weights)
            isco_titles: ISCO-08 titles
            company_mapping: Optional company ID mapping
            use_weights: If True, use employment-weighted aggregation via 6-digit SOC

        Returns:
            DataFrame with ISCO-08 occupation-firm exposure scores
        """
        mode_str = "Employment-Weighted (8d→6d→4d)" if use_weights else "Unweighted (8d→4d)"
        logger.info(f"Crosswalking O*NET → ISCO-08 ({mode_str})...")

        # Add company names if not already present
        if company_mapping is not None and 'company_name' not in onet_exposure.columns:
            logger.info("Adding company names to O*NET exposure data...")
            onet_exposure = onet_exposure.merge(company_mapping, on='company_id', how='left')
            missing_companies = onet_exposure['company_name'].isna().sum()
            if missing_companies > 0:
                logger.warning(f"{missing_companies} company_ids could not be mapped to company names")

        working_df = onet_exposure.copy()

        logger.info(f"[CROSSWALK DEBUG] Input working_df: {len(working_df):,} rows")
        logger.info(f"[CROSSWALK DEBUG] Unique (company_name, year, onet_code): {working_df.groupby(['company_name', 'year', 'onet_code']).ngroups}")

        # Merge with Webb crosswalk to get mapping paths
        # Webb crosswalk has: onet_8d, soc_6d, isco08_4d, isco08_title, weight (pre-normalized)
        merged = working_df.merge(
            webb_crosswalk,
            left_on='onet_code',
            right_on='onet_8d',
            how='inner'
        )

        logger.info(f"[CROSSWALK DEBUG] After merge with Webb: {len(merged):,} rows")
        logger.info(f"[CROSSWALK DEBUG] Webb crosswalk caused {len(merged) - len(working_df):,} additional rows (1-to-many mappings)")

        # Log mapping coverage
        total_onet_codes = working_df['onet_code'].nunique()
        mapped_onet_codes = merged['onet_code'].nunique()
        unmapped_onet_codes = total_onet_codes - mapped_onet_codes

        logger.info(f"Crosswalk mapping results:")
        logger.info(f"  Total unique O*NET codes: {total_onet_codes:,}")
        logger.info(f"  Successfully mapped O*NET codes: {mapped_onet_codes:,} ({mapped_onet_codes/total_onet_codes*100:.1f}%)")
        logger.info(f"  Unmapped O*NET codes: {unmapped_onet_codes:,} ({unmapped_onet_codes/total_onet_codes*100:.1f}%)")
        logger.info(f"  Total input O*NET-firm-year rows: {len(working_df):,}")
        logger.info(f"  After 1-to-many crosswalk explosion: {len(merged):,} O*NET-firm-year → ISCO-firm-year rows")

        if unmapped_onet_codes > 0:
            unmapped_onets = set(working_df['onet_code']) - set(merged['onet_code'])
            logger.warning(f"  Unmapped O*NET codes (first 10): {list(sorted(unmapped_onets))[:10]}")

        # Track contribution from non-4-digit ISCO codes
        merged['isco_digit_length'] = merged['isco08_4d'].str.len()
        rows_from_4digit = (merged['isco_digit_length'] == 4).sum()
        rows_from_non4digit = (merged['isco_digit_length'] != 4).sum()
        pct_from_non4digit = rows_from_non4digit / len(merged) * 100 if len(merged) > 0 else 0

        logger.info(f"Non-4-digit ISCO contribution:")
        logger.info(f"  Rows from 4-digit ISCO codes: {rows_from_4digit:,}")
        logger.info(f"  Rows from non-4-digit ISCO codes: {rows_from_non4digit:,} ({pct_from_non4digit:.1f}%)")
        if rows_from_non4digit > 0:
            non4digit_sample = merged[merged['isco_digit_length'] != 4][['onet_8d', 'isco08_4d']].drop_duplicates()
            logger.info(f"  Non-4-digit mappings in merged data:")
            for idx, row in non4digit_sample.iterrows():
                logger.info(f"    {row['onet_8d']} → {row['isco08_4d']} ({len(row['isco08_4d'])}-digit)")

        # Drop the temporary column used for tracking
        merged.drop(columns=['isco_digit_length'], inplace=True)

        if use_weights:
            # WEIGHTED MODE: Two-stage aggregation (8d → 6d → 4d)
            logger.info("Using weighted aggregation (8-digit → 6-digit → 4-digit)...")

            # STAGE 1: Aggregate 8-digit O*NET to 6-digit SOC (averaging exposures)
            logger.info("  Stage 1: Averaging 8-digit O*NET codes within each 6-digit SOC...")

            # Group by SOC-6d + firm + year and average all exposure scores
            logger.info(f"[CROSSWALK DEBUG] BEFORE SOC-6d aggregation: {len(merged):,} rows")

            soc_6d_grouped = merged.groupby(['soc_6d', 'company_name', 'year'], as_index=False).agg({
                'hampole_ai_exposure_avg': 'mean',
                'binary_ai_exposure_avg': 'mean',
                'hampole_occupation_exposure': 'mean',
                'binary_occupation_exposure': 'mean',
                'log_ai_intensity': 'mean',
                'n_ai_apps_firm_year': 'mean',
                'total_tasks_occupation': 'mean',
                'total_importance_weight': 'mean',
                'onet_code': 'nunique'  # Count contributing O*NET codes
            })

            logger.info(f"  Stage 1 complete: {len(soc_6d_grouped):,} unique 6-digit SOC-firm-year combinations")
            logger.info(f"[CROSSWALK DEBUG] AFTER SOC-6d aggregation: {len(soc_6d_grouped):,} rows (reduced from {len(merged):,})")
            logger.info(f"    Unique 6-digit SOC codes: {soc_6d_grouped['soc_6d'].nunique():,}")

            # STAGE 2: Map 6-digit SOC to 4-digit ISCO using pre-normalized weights
            logger.info("  Stage 2: Applying employment weights to map 6-digit SOC → 4-digit ISCO...")

            # Get unique SOC-6d → ISCO-4d mappings with pre-normalized weights
            soc_to_isco_weights = webb_crosswalk[['soc_6d', 'isco08_4d', 'weight']].drop_duplicates()

            # Merge weights with averaged SOC-6d exposures
            weighted_merged = soc_6d_grouped.merge(soc_to_isco_weights, on='soc_6d', how='inner')

            logger.info(f"    Merged {len(weighted_merged):,} SOC-firm-year rows with ISCO weights")

            # Multiply exposures by pre-normalized weights (no division needed - weights already sum to 1.0)
            weighted_merged['weighted_hampole_ai'] = weighted_merged['hampole_ai_exposure_avg'] * weighted_merged['weight']
            weighted_merged['weighted_binary_ai'] = weighted_merged['binary_ai_exposure_avg'] * weighted_merged['weight']
            weighted_merged['weighted_hampole_occ'] = weighted_merged['hampole_occupation_exposure'] * weighted_merged['weight']
            weighted_merged['weighted_binary_occ'] = weighted_merged['binary_occupation_exposure'] * weighted_merged['weight']
            weighted_merged['weighted_total_tasks'] = weighted_merged['total_tasks_occupation'] * weighted_merged['weight']
            weighted_merged['weighted_importance'] = weighted_merged['total_importance_weight'] * weighted_merged['weight']

            # Aggregate to ISCO-firm-year level by summing weighted values
            # Because weights sum to 1.0, summing weighted values = weighted average
            logger.info(f"[CROSSWALK DEBUG] BEFORE ISCO aggregation: {len(weighted_merged):,} rows")

            isco_firm_exposure = weighted_merged.groupby(['isco08_4d', 'company_name', 'year'], as_index=False).agg({
                'weighted_hampole_ai': 'sum',
                'weighted_binary_ai': 'sum',
                'weighted_hampole_occ': 'sum',
                'weighted_binary_occ': 'sum',
                'weighted_total_tasks': 'sum',
                'weighted_importance': 'sum',
                'log_ai_intensity': 'mean',  # Unweighted metadata
                'n_ai_apps_firm_year': 'mean',
                'onet_code': 'sum'  # Sum of O*NET codes contributing through all SOC-6d paths
            })

            logger.info(f"[CROSSWALK DEBUG] AFTER ISCO aggregation: {len(isco_firm_exposure):,} rows (reduced from {len(weighted_merged):,})")

            # Rename weighted columns back to original names
            isco_firm_exposure.rename(columns={
                'weighted_hampole_ai': 'hampole_ai_exposure_avg',
                'weighted_binary_ai': 'binary_ai_exposure_avg',
                'weighted_hampole_occ': 'hampole_occupation_exposure',
                'weighted_binary_occ': 'binary_occupation_exposure',
                'weighted_total_tasks': 'total_tasks_occupation',
                'weighted_importance': 'total_importance_weight',
                'onet_code': 'n_onet_codes_contributing'
            }, inplace=True)

            logger.info(f"  Stage 2 complete: {len(isco_firm_exposure):,} ISCO-firm-year combinations")

        else:
            # UNWEIGHTED MODE: Direct aggregation (8d → 4d)
            logger.info("Using unweighted aggregation (8-digit → 4-digit direct)...")

            # Simple mean aggregation across all O*NET codes mapping to same ISCO
            isco_firm_exposure = merged.groupby(['isco08_4d', 'company_name', 'year'], as_index=False).agg({
                'hampole_ai_exposure_avg': 'mean',
                'binary_ai_exposure_avg': 'mean',
                'hampole_occupation_exposure': 'mean',
                'binary_occupation_exposure': 'mean',
                'log_ai_intensity': 'mean',
                'n_ai_apps_firm_year': 'mean',
                'total_tasks_occupation': 'mean',
                'total_importance_weight': 'mean',
                'onet_code': 'nunique'
            })

            # Rename O*NET code count column
            isco_firm_exposure.rename(columns={'onet_code': 'n_onet_codes_contributing'}, inplace=True)

            logger.info(f"  Unweighted aggregation complete: {len(isco_firm_exposure):,} ISCO-firm-year combinations")

        # Add ISCO titles
        isco_firm_exposure = isco_firm_exposure.merge(isco_titles, on='isco08_4d', how='left')

        # Add company_id for Stage 6 compatibility
        if company_mapping is not None:
            logger.info(f"[CROSSWALK DEBUG] BEFORE final company_id merge: {len(isco_firm_exposure):,} rows")
            logger.info(f"[CROSSWALK DEBUG] Unique (company_name, year, isco08_4d) BEFORE: {isco_firm_exposure.groupby(['company_name', 'year', 'isco08_4d']).ngroups}")

            isco_firm_exposure = isco_firm_exposure.merge(
                company_mapping[['company_name', 'company_id']],
                on='company_name',
                how='left'
            )

            logger.info(f"[CROSSWALK DEBUG] AFTER final company_id merge: {len(isco_firm_exposure):,} rows")
            logger.info(f"[CROSSWALK DEBUG] Cartesian product added {len(isco_firm_exposure) - len(isco_titles):,} rows")

            missing_companies = isco_firm_exposure['company_id'].isna().sum()
            if missing_companies > 0:
                logger.warning(f"{missing_companies} company names could not be mapped to company_ids")

        # Reorder columns (Stage 6 needs company_id first)
        base_cols = ['company_id', 'isco08_4d', 'isco08_title', 'company_name', 'year']
        other_cols = [col for col in isco_firm_exposure.columns if col not in base_cols]
        cols = [col for col in base_cols if col in isco_firm_exposure.columns] + other_cols
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
    
    def _generate_filename(self, occupation_system: str = "isco") -> str:
        """
        Generate output filename following pattern: {isco/onet}_{firm}_{occupation}_{year}_exposure_{task_type}.csv

        Args:
            occupation_system: Either "isco" or "onet"

        Returns:
            Complete filename based on configuration
        """
        components = [occupation_system]  # Start with isco or onet

        # Add firm if we have firm-level variation (not exposure_table_format="exposed-occ-year")
        if self.exposure_table_format != "exposed-occ-year":
            components.append("firm")

        # Add occupation if occupation processing is enabled
        if self.exposure_table_format in ["all-firm-occ-year", "exposed-occ-year"]:
            components.append("occupation")

        # Add year if time-variant
        if not self.time_invariant:
            components.append("year")

        # Add exposure and task type
        components.append(f"exposure_{self.task_type}_tasks.csv")
        return "_".join(components)

    def _find_stage4_file(self, task_type: str = 'core', model: str = 'BGE') -> str:
        """
        Auto-detect Stage 4 output file based on task type.

        Args:
            task_type: 'core' or 'all'

        Returns:
            Path to Stage 4 file
        """

        # BGE percentile string (match stage_4 formatting: reverse-sorted)
        pct_list = getattr(self, 'bge_percentiles', None) or []
        if not pct_list:
            percentile_str = ""
        else:
            percentile_str = "_".join(
                str(int(p)) if isinstance(p, (int, float)) and p == int(p) else str(p)
                for p in sorted(pct_list, reverse=True)
            )

        # CE thresholds string (only include > 0.0 thresholds, format like 0p8)
        ce_list = getattr(self, 'ce_thresholds', None) or []
        ce_percentile_str = "_".join(f"{c:.1f}".replace('.', 'p') for c in sorted(ce_list, reverse=True) if c > 0.0)
        ce_part = f"_ce{ce_percentile_str}" if ce_percentile_str else ""

        # O*NET suffix
        onet_suffix = f"_onet{self.onet_version}" if getattr(self, 'onet_version', None) else ""

        # Task suffix
        task_suffix = f"_{task_type}"

        # Construct filename using same pattern as stage_4
        # e.g. task_exposure_matches_all_thresholds_openai_bge20_15_10_5_1_ce0p8_0p6_onet20_core.parquet
        bge_part = f"_bge{percentile_str}" if percentile_str else ""
        filename = f"task_exposure_matches_all_thresholds_{model}{bge_part}{ce_part}{onet_suffix}{task_suffix}.parquet"
        filepath = os.path.join(self.stage_4_dir, filename)

        if os.path.exists(filepath):
            logger.info(f"Using Stage 4 file: {filename}")
            return filepath

        # Fallback: try without onet suffix (older files may omit onet version)
        alt_filename = f"task_exposure_matches_all_thresholds_{model}{bge_part}{ce_part}{task_suffix}.parquet"
        alt_path = os.path.join(self.stage_4_dir, alt_filename)
        if os.path.exists(alt_path):
            logger.info(f"Using Stage 4 file (no onet suffix): {alt_filename}")
            return alt_path

        # If not found, raise informative error listing attempted names
        raise FileNotFoundError(
            f"Stage 4 file not found in {self.stage_4_dir}\n"
            f"Tried: {filename}\n"
            f"Tried: {alt_filename}\n"
            f"You can inspect {self.stage_4_dir} for available files and adjust `bge_percentiles` / `ce_thresholds` / `model` accordingly."
        )

    def _generate_config_suffix(self) -> str:
        """
        Generate configuration suffix for auxiliary output files (summary, task-level).

        Returns:
            String suffix describing the configuration
        """
        components = []

        # Add occupation processing mode
        if self.exposure_table_format == "all-firm-occ-year":
            components.append("firm_occupation")
        elif self.exposure_table_format == "exposed-occ-year":
            components.append("occupation_only")
        else:  # "exposed-firm-occ-year"
            components.append("firm_only")

        # Add time variant info
        if not self.time_invariant:
            components.append("year")

        return "_" + "_".join(components) if components else ""

    def _ensure_output_directory(self, dir_path: str) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Output directory: {dir_path}")

    def run_full_pipeline(self,
                         top_matches_file: str = None,
                         company_file: str = None,
                         output_file: str = None,
                         save_onet_outputs: bool = False,
                         task_type: str = 'core',
                         model: str = 'BGE',
                         onet_version: int = 20) -> pd.DataFrame:
        """
        Run the complete 4-step Task → Occupation × Firm AI Exposure pipeline.
        
        Args:
            top_matches_file: Path to top_5_matches.csv file (auto-detect if None)
            company_file: Path to company data CSV (auto-detect if None)
            output_file: Output CSV path (auto-generate if None)
            stage2_mapping_file: Path to Stage 2 deduplication mapping (auto-detect if None)
            save_onet_outputs: If True, save intermediate O*NET exposure files

        Returns:
            DataFrame with final ISCO-08 occupation-firm exposure scores
        """
        logger.info("="*60)
        logger.info("STARTING 4-STEP TASK → OCCUPATION × FIRM AI EXPOSURE PIPELINE")
        logger.info("="*60)
        # Set top matches file (must be the unified multi-threshold file from Stage 4)
        # Auto-detect based on task type
        # Only auto-detect if top_matches_file is NOT provided
        if top_matches_file is None:
            logger.info(f"Auto-detecting Stage 4 file based on task_type: {task_type} and model: {model}")
            top_matches_file = self._find_stage4_file(task_type, model)
        else:
            # If provided, ensure the path is correct (handle relative paths to stage_4_dir)
            if not os.path.exists(top_matches_file):
                potential_path = os.path.join(self.stage_4_dir, top_matches_file)
                if os.path.exists(potential_path):
                    top_matches_file = potential_path
            logger.info(f"Using provided Stage 4 file: {top_matches_file}")
        logger.info(f"Auto-detected Stage 4 file: {top_matches_file}")

        # Validate file exists
        if not os.path.exists(top_matches_file):
            raise FileNotFoundError(f"Top matches file not found: {top_matches_file} in directory: {self.stage_4_dir}")
        
        # Auto-detect company file if not provided
        if company_file is None:
            company_file = os.path.join(self.data_dir, "ai_development_deduplicated_custom.csv")
            logger.info(f"Using company data file: {company_file}")
        
        if save_onet_outputs:
            logger.info(f"O*NET outputs will be saved to: {self.output_dir}")
        
        # Define other input files (using O*NET version)
        task_statements_file = os.path.join(self.data_dir, f"task_statements_{onet_version}.xlsx")
        task_ratings_file = os.path.join(self.data_dir, f"task_ratings_{onet_version}.xlsx")
        esco_onet_file = os.path.join(self.data_dir, "ESCO_to_ONET-SOC.xlsx")

        logger.info(f"Using O*NET version {onet_version} files:")
        logger.info(f"  Task statements: {task_statements_file}")
        logger.info(f"  Task ratings: {task_ratings_file}")
        
        # ================================================================
        # DATA LOADING
        # ================================================================
        logger.info("\n" + "="*50)
        logger.info("DATA LOADING")
        logger.info("="*50)
        
        # Step 1: Load exposed task-application match metadata (filter columns only)
        matches_file_path, filter_metadata = self.step1_load_task_application_matches(top_matches_file)

        # Auto-detect BGE percentiles from Stage 4 output columns
        pct_cols = [col for col in filter_metadata.columns if col.startswith('pct_')]
        logger.info(f"Found percentile columns in Stage 4 output: {sorted(pct_cols)}")

        detected_percentiles = []

        # Parse each percentile column to extract numeric value
        for col in pct_cols:
            # Parse pct_XX or pct_Xp1 format
            if 'p' in col:
                # Format: pct_Xp1 means X.1
                num_str = col.replace('pct_', '').replace('p', '.')
                detected_percentiles.append(float(num_str))
            else:
                # Format: pct_XX means integer
                num_str = col.replace('pct_', '')
                detected_percentiles.append(int(num_str))

        # Sort in descending order (largest percentiles first, like [20, 15, 10, 5, 1, 0.1])
        detected_percentiles.sort(reverse=True)
        logger.info(f"Detected percentiles from columns: {detected_percentiles}")
        logger.info(f"Current initialized percentiles: {self.bge_percentiles}")

        # Check if detected percentiles differ from current ones
        if detected_percentiles and detected_percentiles != self.bge_percentiles:
            logger.warning("="*80)
            logger.warning("BGE PERCENTILE AUTO-DETECTION")
            logger.warning("="*80)
            logger.warning(f"Stage 4 output percentiles: {detected_percentiles}")
            logger.warning(f"Stage 5 was initialized with: {self.bge_percentiles}")
            logger.warning(f"Updating Stage 5 to match Stage 4")
            logger.warning("="*80)
            self.bge_percentiles = detected_percentiles
            logger.info(f"✓ Updated BGE percentiles to: {self.bge_percentiles}")

        # Detect if cross-encoder scores are available (need to read from file)
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        import pyarrow as pa
        parquet_file = pq.ParquetFile(matches_file_path)
        ce_column_data = parquet_file.read(['cross_encoder_score']).column('cross_encoder_score')
        ce_scores_exist = pc.sum(pc.is_valid(ce_column_data)).as_py() > 0
        if not ce_scores_exist:
            logger.warning("="*80)
            logger.warning("NO CROSS-ENCODER SCORES DETECTED")
            logger.warning("="*80)
            logger.warning("Stage 4 was likely run with --skip-cross-encoder")
            logger.warning("Processing BGE-only specifications without CE filtering")
            logger.warning("="*80)

        # Validate O*NET version before loading crosswalk
        if self.onet_version >= 25:
            raise ValueError(
                f"Cannot use BLS SOC-10 crosswalk with O*NET v{self.onet_version}. "
                f"O*NET 25+ uses SOC 2018 taxonomy (incompatible with SOC 2010 crosswalk)."
            )

        # Load supporting data
        task_statements = self.load_task_statements(task_statements_file, core_only=True)
        task_ratings = self.load_task_ratings(task_ratings_file)
        job_app_mapping = self.load_job_app_mapping(task_type, model)
        original_jobs = self.load_original_full_dataset(company_file)  # Load full original dataset

        # Load BLS employment data (2018 snapshot, 6-digit SOC)
        employment_file = os.path.join(self.data_dir, "national_M2018_dl.xlsx")
        employment_data = self.load_employment_data(employment_file)

        # Load Webb O*NET-SOC → ISCO-08 crosswalk with pre-normalized weights
        webb_crosswalk_file = os.path.join(self.data_dir, "webb_crosswalk_clean.xls")
        webb_crosswalk = self.load_webb_crosswalk_with_weights(
            webb_crosswalk_file,
            employment_data,
            use_weights=self.use_employment_weights  # From CLI flag
        )

        # Extract ISCO titles from Webb crosswalk
        isco_titles = webb_crosswalk[['isco08_4d', 'isco08_title']].drop_duplicates()
        logger.info(f"Extracted {len(isco_titles):,} unique ISCO-08 titles from Webb crosswalk")
        
        # Load company mapping for Stage 6 compatibility
        company_mapping = self.load_company_mapping()
        
        # ================================================================
        # MULTI-SPECIFICATION 4-STEP HAMPOLE PIPELINE
        # ================================================================
        logger.info("\n" + "="*80)
        logger.info("PROCESSING MULTIPLE SPECIFICATIONS")
        logger.info("="*80)

        # Generate column names for BGE percentiles and CE thresholds
        bge_cols = []
        for p in self.bge_percentiles:
            # Format percentile name: handle both integers (20) and floats (0.1)
            if isinstance(p, int) or p == int(p):
                bge_cols.append(f'pct_{int(p):02d}')
            else:
                # For floats like 0.1, format as pct_0p1
                pct_str = f'{p:.1f}'.replace('.', 'p')
                bge_cols.append(f'pct_{pct_str}')

        # Conditionally set CE thresholds based on whether CE scores exist
        if ce_scores_exist:
            logger.info("✓ Cross-encoder scores available, processing all CE thresholds")
            ce_cols = [f'ce_{c:.1f}' for c in self.ce_thresholds]
        else:
            logger.info("⚠️ Using ce_0.0 (no CE filtering) as only threshold")
            ce_cols = ['ce_0.0']

        specifications = [(p, c) for p in bge_cols for c in ce_cols]

        logger.info(f"Total specifications to process: {len(specifications)}")
        logger.info(f"  BGE percentiles: {bge_cols}")
        logger.info(f"  CE thresholds: {ce_cols}")

        # Create checkpoint directory for per-spec checkpointing
        # Use fixed name to enable resume across runs
        checkpoint_dir = Path(self.output_dir) / "stage_5_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"\n✓ Using checkpoint directory: {checkpoint_dir}")
        logger.info(f"  (Will be cleaned up after successful merge)")

        # Process all specifications and collect results
        results_by_spec = {}
        isco_results_by_spec = {}
        for spec_idx, (bge_col, ce_col) in enumerate(specifications, 1):
            logger.info(f"\n[{spec_idx}/{len(specifications)}] {bge_col} × {ce_col}")

            # Check if checkpoint already exists (for resume capability)
            onet_checkpoint_path = checkpoint_dir / f"onet_spec_{bge_col}_{ce_col}.parquet"
            isco_checkpoint_path = checkpoint_dir / f"isco_spec_{bge_col}_{ce_col}.parquet"

            if onet_checkpoint_path.exists() and isco_checkpoint_path.exists():
                logger.info(f"  ✓ Using cached checkpoint for {bge_col} × {ce_col} (skipping processing)")
                continue  # Skip to next spec

            # Build filter mask using metadata (memory-efficient)
            # Always use percentile, optionally add minimum threshold and CE
            if ce_col == 'ce_0.0' and not ce_scores_exist:
                # CE is missing, so use BGE filter only (ce_0.0 is effectively always True)
                filter_mask = filter_metadata[bge_col]
            else:
                # Normal case: apply both BGE and CE filters
                filter_mask = filter_metadata[bge_col] & filter_metadata[ce_col]

            # Apply minimum threshold filter if enabled and column exists
            if self.apply_min_threshold and 'above_min_threshold' in filter_metadata.columns:
                filter_mask = filter_mask & filter_metadata['above_min_threshold']
                logger.info(f"  Applying minimum threshold filter (above_min_threshold column)")
            elif self.apply_min_threshold:
                logger.warning(f"  Minimum threshold filtering requested but 'above_min_threshold' column not found - skipping")

            match_count = filter_mask.sum()
            logger.info(f"  Matches: {match_count:,}")

            if match_count == 0:
                logger.warning(f"  ⚠️ No matches for this specification")
                results_by_spec[(bge_col, ce_col)] = None
            else:
                try:
                    # Determine processing strategy based on filtered dataset size
                    # Allow override via environment variable for testing
                    default_threshold = 50_000_000  # 50M rows
                    threshold_override = os.environ.get('FORCE_CHUNK_PROCESSING_THRESHOLD', '').strip()

                    if threshold_override:
                        try:
                            CHUNK_PROCESSING_THRESHOLD = int(threshold_override)
                            logger.info(f"  ⚠️ CHUNK_PROCESSING_THRESHOLD overridden via env var: {CHUNK_PROCESSING_THRESHOLD:,}")
                        except ValueError:
                            logger.warning(f"  Invalid FORCE_CHUNK_PROCESSING_THRESHOLD: {threshold_override}, using default")
                            CHUNK_PROCESSING_THRESHOLD = default_threshold
                    else:
                        CHUNK_PROCESSING_THRESHOLD = default_threshold

                    if getattr(self, "force_nonchunked", False):
                        logger.warning("  Force-nonchunked enabled: using STANDARD processing regardless of match count")
                        CHUNK_PROCESSING_THRESHOLD = float("inf")

                    if match_count >= CHUNK_PROCESSING_THRESHOLD:
                        logger.info(f"  Using CHUNK-BASED PROCESSING (filtered dataset ≥{CHUNK_PROCESSING_THRESHOLD:,} rows)")

                        # CHUNK-BASED PROCESSING PIPELINE
                        # Step 0: Pre-calculate firm-app-year mappings
                        firm_year_app_counts, firm_app_first_years = self._precalculate_firm_app_mappings(
                            matches_file_path,
                            filter_mask.values,
                            self.expand_job_app_mapping_to_full_dataset(job_app_mapping, original_jobs, None)
                        )

                        # Step 1: Process chunks through step2_chunked and accumulate results
                        logger.info(f"  Processing chunks through step2_chunked...")
                        expanded_job_app_mapping = self.expand_job_app_mapping_to_full_dataset(
                            job_app_mapping, original_jobs, None
                        )

                        task_firm_chunks = []
                        chunk_count = 0

                        for chunk in self._load_filtered_matches_chunked_generator(matches_file_path, filter_mask.values):
                            chunk_count += 1

                            # Process chunk through step2_chunked
                            chunk_exposure = self.step2_calculate_firm_task_exposure_chunked(
                                chunk,
                                expanded_job_app_mapping,
                                firm_year_app_counts,
                                firm_app_first_years
                            )

                            task_firm_chunks.append(chunk_exposure)

                            del chunk, chunk_exposure
                            gc.collect()

                            if chunk_count % 10 == 0:
                                logger.info(f"    Processed {chunk_count} chunks through step2")

                        logger.info(f"  Processed all {chunk_count} chunks through step2")

                        # Step 2: Merge and re-aggregate chunk results
                        logger.info(f"  Merging {len(task_firm_chunks)} chunk results...")
                        task_firm_exposure_combined = pd.concat(task_firm_chunks, ignore_index=True)

                        # DEBUG LOGGING
                        logger.info(f"  [DEBUG] Combined chunks: {len(task_firm_exposure_combined):,} rows")
                        logger.info(f"  [DEBUG] Unique companies: {task_firm_exposure_combined['company_name'].nunique()}")
                        logger.info(f"  [DEBUG] Unique tasks: {task_firm_exposure_combined['onet_task_id'].nunique()}")
                        if 'year' in task_firm_exposure_combined.columns:
                            logger.info(f"  [DEBUG] Unique years: {sorted(task_firm_exposure_combined['year'].unique())}")
                            logger.info(f"  [DEBUG] Unique (company, year, task) keys: {task_firm_exposure_combined.groupby(['company_name', 'year', 'onet_task_id']).ngroups}")
                        else:
                            logger.info(f"  [DEBUG] Unique (company, task) keys: {task_firm_exposure_combined.groupby(['company_name', 'onet_task_id']).ngroups}")

                        # Re-aggregate to handle overlaps between chunks
                        logger.info(f"  Re-aggregating {len(task_firm_exposure_combined):,} rows to handle chunk overlaps...")

                        if self.time_invariant:
                            # Time-invariant: No year column yet, aggregate by firm-task
                            task_firm_exposure = task_firm_exposure_combined.groupby(
                                ['company_name', 'onet_task_id'], as_index=False
                            ).agg({
                                'n_task_matches_firm_year': 'sum',  # Additive across chunks
                            })

                            # Get unique firms and years from pre-calculated data
                            all_firms = list(set([k[0] for k in firm_year_app_counts.keys()]))
                            all_years = sorted(set([k[1] for k in firm_year_app_counts.keys()]))

                            # Add n_ai_apps_firm_year (same for all years in time-invariant mode)
                            # Use firm_year_app_counts from first year for each firm
                            firm_app_counts_map = {}
                            for firm in all_firms:
                                # In time-invariant mode, all years have same count
                                firm_app_counts_map[firm] = firm_year_app_counts.get((firm, all_years[0]), 0)

                            task_firm_exposure['n_ai_apps_firm_year'] = task_firm_exposure['company_name'].map(firm_app_counts_map)

                            # Broadcast across all years
                            years_df = pd.DataFrame({'year': all_years})
                            task_firm_exposure = task_firm_exposure.merge(
                                years_df,
                                how='cross'
                            )[['company_name', 'year', 'onet_task_id', 'n_ai_apps_firm_year', 'n_task_matches_firm_year']]

                        else:
                            # Time-variant: Has year column, aggregate by firm-year-task
                            task_firm_exposure = task_firm_exposure_combined.groupby(
                                ['company_name', 'year', 'onet_task_id'], as_index=False
                            ).agg({
                                'n_ai_apps_firm_year': 'first',  # Same for all tasks in firm-year
                                'n_task_matches_firm_year': 'sum',  # Additive across chunks
                            })

                        # DEBUG LOGGING - After aggregation
                        logger.info(f"  [DEBUG] After aggregation: {len(task_firm_exposure):,} rows")
                        logger.info(f"  [DEBUG] Unique companies: {task_firm_exposure['company_name'].nunique()}")
                        logger.info(f"  [DEBUG] Unique tasks: {task_firm_exposure['onet_task_id'].nunique()}")
                        if 'year' in task_firm_exposure.columns:
                            logger.info(f"  [DEBUG] Unique (company, year, task) keys: {task_firm_exposure.groupby(['company_name', 'year', 'onet_task_id']).ngroups}")

                        # Recalculate exposure scores after aggregation (same for both modes)
                        task_firm_exposure['hampole_task_exposure'] = (
                            task_firm_exposure['n_task_matches_firm_year'] /
                            task_firm_exposure['n_ai_apps_firm_year']
                        )
                        task_firm_exposure['binary_task_exposure'] = (
                            (task_firm_exposure['n_task_matches_firm_year'] >= 1).astype(int)
                        )

                        logger.info(f"  ✓ Aggregated to {len(task_firm_exposure):,} firm-year-task exposures")

                        # DEBUG LOGGING - Sample data for verification
                        logger.info(f"  [DEBUG] Sample task_firm_exposure rows (first 3):")
                        sample_df = task_firm_exposure.head(3)[['company_name', 'year', 'onet_task_id', 'n_ai_apps_firm_year', 'n_task_matches_firm_year', 'hampole_task_exposure']]
                        for idx, row in sample_df.iterrows():
                            logger.info(f"    {row['company_name'][:30]} | {row['year']} | {row['onet_task_id']} | n_apps={row['n_ai_apps_firm_year']:.0f} | n_matches={row['n_task_matches_firm_year']:.0f} | exposure={row['hampole_task_exposure']:.6f}")

                        # Clean up intermediate data
                        del task_firm_chunks, task_firm_exposure_combined, expanded_job_app_mapping
                        del firm_year_app_counts, firm_app_first_years
                        gc.collect()

                    else:
                        logger.info(f"  Using STANDARD PROCESSING (filtered dataset <{CHUNK_PROCESSING_THRESHOLD:,} rows)")

                        # STANDARD PROCESSING (existing logic)
                        # Load ONLY filtered data for this spec (memory-efficient)
                        spec_matches = self._load_filtered_matches(matches_file_path, filter_mask.values)

                        # Run step2 (standard version)
                        task_firm_exposure = self.step2_calculate_firm_task_exposure(
                            spec_matches, job_app_mapping, original_jobs
                        )

                        # Clean up spec_matches
                        del spec_matches
                        gc.collect()

                    # Steps 3-4: Same for both processing modes (already aggregated data)
                    occupation_firm_exposure = self.step3_calculate_occupation_firm_exposure(
                        task_firm_exposure, task_statements, task_ratings
                    )

                    # DEBUG LOGGING - After Step 3
                    logger.info(f"  [DEBUG] After Step 3 (occupation aggregation): {len(occupation_firm_exposure):,} rows")
                    logger.info(f"  [DEBUG] Unique companies: {occupation_firm_exposure['company_name'].nunique()}")
                    logger.info(f"  [DEBUG] Unique occupations: {occupation_firm_exposure['onet_code'].nunique()}")
                    logger.info(f"  [DEBUG] Unique (company, year, occupation) keys: {occupation_firm_exposure.groupby(['company_name', 'year', 'onet_code']).ngroups}")

                    spec_result = self.step4_apply_ai_intensity_adjustment(occupation_firm_exposure)

                    # DEBUG LOGGING - After Step 4
                    logger.info(f"  [DEBUG] After Step 4 (AI intensity): {len(spec_result):,} rows")

                    # Add company_id by merging with company_mapping for merge compatibility
                    logger.info(f"  [DEBUG] BEFORE company_id merge: {len(spec_result):,} rows")
                    logger.info(f"  [DEBUG] Unique (company_name, year, onet_code) BEFORE merge: {spec_result.groupby(['company_name', 'year', 'onet_code']).ngroups}")

                    spec_result = spec_result.merge(
                        company_mapping[['company_name', 'company_id']],
                        on='company_name',
                        how='left'
                    )

                    logger.info(f"  [DEBUG] AFTER company_id merge: {len(spec_result):,} rows")
                    logger.info(f"  [DEBUG] Rows added by Cartesian product: {len(spec_result) - occupation_firm_exposure.shape[0]}")
                    logger.info(f"  [DEBUG] Unique (company_name, year, onet_code) AFTER merge: {spec_result.groupby(['company_name', 'year', 'onet_code']).ngroups}")
                    logger.info(f"  [DEBUG] Unique company_ids: {spec_result['company_id'].nunique()}")

                    # Check for companies with multiple IDs
                    company_id_counts = spec_result.groupby('company_name')['company_id'].nunique()
                    multi_id_companies = company_id_counts[company_id_counts > 1]
                    if len(multi_id_companies) > 0:
                        logger.info(f"  [DEBUG] Companies with multiple IDs: {len(multi_id_companies)}")
                        for company in list(multi_id_companies.index)[:5]:
                            ids = spec_result[spec_result['company_name'] == company]['company_id'].unique()
                            logger.info(f"    {company}: {ids}")

                    missing_companies = spec_result['company_id'].isna().sum()
                    if missing_companies > 0:
                        logger.warning(f"  {missing_companies} companies in spec result couldn't be mapped to company_ids")

                    # ================================================================
                    # SAVE INDIVIDUAL SPEC FILES (without suffixes)
                    # ================================================================
                    self._ensure_output_directory(self.output_dir)

                    # Generate filename for this spec
                    base_filename = self._generate_filename("onet").replace(".csv", "")
                    spec_suffix = f"_{bge_col}_{ce_col}.csv"

                    # Save individual O*NET file
                    onet_spec_file = os.path.join(self.output_dir, f"{base_filename}{spec_suffix}")
                    spec_result_with_soc = spec_result.copy()
                    spec_result_with_soc.rename(columns={'onet_code': 'onet_soc'}, inplace=True)

                    # Add O*NET titles
                    onet_titles = task_statements[['O*NET-SOC Code', 'Title']].drop_duplicates()
                    onet_titles.rename(columns={'O*NET-SOC Code': 'onet_soc', 'Title': 'onet_title'}, inplace=True)
                    spec_result_with_soc = spec_result_with_soc.merge(onet_titles, on='onet_soc', how='left')

                    spec_result_with_soc.to_csv(onet_spec_file, index=False, encoding='utf-8')
                    logger.info(f"  ✓ Saved O*NET spec file: {os.path.basename(onet_spec_file)}")

                    # Crosswalk individual spec to ISCO
                    logger.info(f"  [DEBUG] INPUT to crosswalk: {len(spec_result):,} O*NET rows")
                    logger.info(f"  [DEBUG] Unique (company_name, year, onet_code) going into crosswalk: {spec_result.groupby(['company_name', 'year', 'onet_code']).ngroups}")

                    isco_spec_result = self.crosswalk_onet_to_isco(
                        spec_result,
                        webb_crosswalk,
                        isco_titles,
                        company_mapping,
                        use_weights=self.use_employment_weights  # From CLI flag
                    )

                    logger.info(f"  [DEBUG] OUTPUT from crosswalk: {len(isco_spec_result):,} ISCO rows")
                    logger.info(f"  [DEBUG] Unique (company_name, year, isco08_4d): {isco_spec_result.groupby(['company_name', 'year', 'isco08_4d']).ngroups}")
                    logger.info(f"  [DEBUG] Unique companies: {isco_spec_result['company_name'].nunique()}")
                    logger.info(f"  [DEBUG] Unique ISCO codes: {isco_spec_result['isco08_4d'].nunique()}")

                    # ADD SPEC SUFFIXES TO ISCO EXPOSURE COLUMNS (for merged file)
                    isco_exposure_cols = [c for c in isco_spec_result.columns
                                          if any(x in c for x in ['exposure', 'intensity', 'n_ai_apps', 'total_tasks', 'total_importance', 'n_onet_codes'])]
                    isco_suffix = f"_{bge_col}_{ce_col}"
                    isco_rename_dict = {col: f"{col}{isco_suffix}" for col in isco_exposure_cols}
                    isco_spec_result_for_merge = isco_spec_result.rename(columns=isco_rename_dict)

                    # Save individual ISCO file (WITHOUT suffixes for standalone use)
                    isco_base_filename = self._generate_filename("isco").replace(".csv", "")
                    isco_spec_file = os.path.join(self.output_dir, f"{isco_base_filename}{spec_suffix}")
                    isco_spec_result.to_csv(isco_spec_file, index=False, encoding='utf-8')
                    logger.info(f"  ✓ Saved ISCO spec file: {os.path.basename(isco_spec_file)}")

                    # NOW add spec suffix for merged O*NET files
                    spec_result = self._add_spec_suffix_to_columns(spec_result, bge_col, ce_col)

                    # ================================================================
                    # SAVE CHECKPOINTS FOR MEMORY MANAGEMENT
                    # ================================================================
                    # Save ONET checkpoint (without suffixes for standalone, with suffixes for merge)
                    onet_checkpoint_path = checkpoint_dir / f"onet_spec_{bge_col}_{ce_col}.parquet"
                    spec_result.to_parquet(onet_checkpoint_path, compression='snappy', index=False)
                    logger.info(f"  ✓ Saved ONET checkpoint: {onet_checkpoint_path.name}")

                    # Save ISCO checkpoint (with suffixes for merge)
                    isco_checkpoint_path = checkpoint_dir / f"isco_spec_{bge_col}_{ce_col}.parquet"
                    isco_spec_result_for_merge.to_parquet(isco_checkpoint_path, compression='snappy', index=False)
                    logger.info(f"  ✓ Saved ISCO checkpoint: {isco_checkpoint_path.name}")

                    # Clear memory: delete large intermediate DataFrames
                    # Note: spec_matches only exists in standard processing mode, not chunk-based
                    del spec_result, isco_spec_result_for_merge, isco_spec_result
                    del task_firm_exposure, occupation_firm_exposure
                    del spec_result_with_soc  # Also delete the O*NET version with titles
                    gc.collect()
                    logger.info(f"  ✓ Cleared memory after spec {spec_idx}/{len(specifications)}")

                    logger.info(f"  ✓ Spec {spec_idx}/{len(specifications)} complete")
                except Exception as e:
                    logger.error(f"  ✗ Error processing spec {spec_idx}: {e}")
                    results_by_spec[(bge_col, ce_col)] = None
                    isco_results_by_spec[(bge_col, ce_col)] = None

        # Merge all specifications with outer join
        logger.info("\n" + "="*80)
        logger.info("MERGING SPECIFICATIONS FROM CHECKPOINTS")
        logger.info("="*80)

        # Load and merge O*NET checkpoints
        final_onet_exposure = None
        merge_keys = ['company_id', 'company_name', 'year', 'onet_code']
        valid_spec_count = 0

        for spec_idx, (bge_col, ce_col) in enumerate(specifications, 1):
            onet_checkpoint_path = checkpoint_dir / f"onet_spec_{bge_col}_{ce_col}.parquet"

            if not onet_checkpoint_path.exists():
                logger.warning(f"  [{spec_idx}/{len(specifications)}] Missing ONET checkpoint: {onet_checkpoint_path.name}, skipping")
                continue

            # Load spec result from checkpoint
            spec_result = pd.read_parquet(onet_checkpoint_path)
            valid_spec_count += 1
            logger.info(f"  [{spec_idx}/{len(specifications)}] Loaded ONET checkpoint {bge_col}×{ce_col}: {len(spec_result):,} rows")

            if final_onet_exposure is None:
                final_onet_exposure = spec_result
                logger.info(f"    Initialized with {bge_col}×{ce_col}: {len(spec_result):,} rows")
            else:
                # Merge with outer join
                final_onet_exposure = final_onet_exposure.merge(
                    spec_result,
                    on=merge_keys,
                    how='outer',
                    suffixes=('', '_dup')
                )
                # Drop duplicate metadata
                dup_cols = [c for c in final_onet_exposure.columns if c.endswith('_dup')]
                if dup_cols:
                    final_onet_exposure.drop(columns=dup_cols, inplace=True)

                logger.info(f"    Merged {bge_col}×{ce_col}: {len(final_onet_exposure):,} rows")

            # Clear memory after each merge
            del spec_result
            gc.collect()

        if valid_spec_count == 0:
            raise ValueError("All specifications produced empty results or checkpoints missing!")

        logger.info(f"\n✓ Valid specifications merged: {valid_spec_count}/{len(specifications)}")

        # Fill NaN exposures with 0
        exposure_cols = [c for c in final_onet_exposure.columns if 'exposure' in c.lower()]
        for col in exposure_cols:
            final_onet_exposure[col] = final_onet_exposure[col].fillna(0)

        logger.info(f"\n✓ Merge complete: {len(final_onet_exposure):,} rows, {len(final_onet_exposure.columns)} columns")
        
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
            
            # Rename onet_code to onet_soc for Stage 6 compatibility
            final_onet_with_titles.rename(columns={'onet_code': 'onet_soc'}, inplace=True)

            # Ensure company_id is included for Stage 6 compatibility
            if 'company_id' not in final_onet_with_titles.columns:
                final_onet_with_titles = final_onet_with_titles.merge(
                    company_mapping[['company_name', 'company_id']],
                    on='company_name',
                    how='left'
                )

            # Reorder columns for better readability (Stage 6 needs company_id and onet_soc)
            cols = ['company_id', 'onet_soc', 'onet_title', 'company_name', 'year'] + [col for col in final_onet_with_titles.columns
                                                                                       if col not in ['company_id', 'onet_soc', 'onet_title', 'company_name', 'year']]
            final_onet_with_titles = final_onet_with_titles[cols]

            # Save to output directory (merged all-specs file)
            self._ensure_output_directory(self.output_dir)
            base_filename = self._generate_filename("onet").replace(".csv", "")
            onet_firm_file = os.path.join(self.output_dir, f"{base_filename}_all_specs.csv")
            final_onet_with_titles.to_csv(onet_firm_file, index=False, encoding='utf-8')
            logger.info(f"✅ Saved merged O*NET firm-year exposure: {onet_firm_file}")

            # 2. Save O*NET Occupation-Level Exposure Summary
            # Load onet_task_id column from file for summary table
            import pyarrow.parquet as pq
            task_ids_table = pq.read_table(matches_file_path, columns=['onet_task_id'])
            task_ids_df = task_ids_table.to_pandas()
            task_exposure_table = self.create_task_exposure_table(task_ids_df, task_statements, task_ratings)
            onet_occupation_summary = self.aggregate_tasks_to_onet_occupations(task_exposure_table)
            
            # Add O*NET titles to the summary
            onet_titles = task_statements[['O*NET-SOC Code', 'Title']].drop_duplicates()
            onet_titles.rename(columns={'O*NET-SOC Code': 'onet_code', 'Title': 'onet_title'}, inplace=True)
            onet_occupation_summary = onet_occupation_summary.merge(onet_titles, on='onet_code', how='left')
            
            # Reorder columns to put title after code
            cols = ['onet_code', 'onet_title'] + [col for col in onet_occupation_summary.columns 
                                                  if col not in ['onet_code', 'onet_title']]
            onet_occupation_summary = onet_occupation_summary[cols]
            
            onet_summary_file = os.path.join(self.output_dir, f"onet_occupation_exposure_summary{config_suffix}.csv")
            onet_occupation_summary.to_csv(onet_summary_file, index=False, encoding='utf-8')
            logger.info(f"✅ Saved O*NET occupation summary: {onet_summary_file}")

            # 3. Save O*NET Task-Level Exposure Table
            onet_task_file = os.path.join(self.output_dir, f"onet_task_exposure{config_suffix}.csv")
            task_exposure_table.to_csv(onet_task_file, index=False, encoding='utf-8')
            logger.info(f"✅ Saved O*NET task exposure: {onet_task_file}")
            
            logger.info(f"📊 O*NET outputs summary:")
            logger.info(f"  - Firm-year combinations: {len(final_onet_with_titles):,}")
            logger.info(f"  - Unique O*NET occupations: {final_onet_with_titles['onet_soc'].nunique():,}")
            logger.info(f"  - Unique firms: {final_onet_with_titles['company_name'].nunique():,}")
            logger.info(f"  - Occupation-level summaries: {len(onet_occupation_summary):,}")
            logger.info(f"  - Task-level exposures: {len(task_exposure_table):,}")
        
        # ================================================================
        # MERGE INDIVIDUAL ISCO FILES (from checkpoints)
        # ================================================================
        logger.info("\n" + "="*50)
        logger.info("CREATING MERGED ISCO FILE FROM CHECKPOINTS")
        logger.info("="*50)

        # Load and merge ISCO checkpoints
        isco_firm_exposure = None
        merge_keys = ['company_id', 'company_name', 'year', 'isco08_4d']
        valid_isco_spec_count = 0

        for spec_idx, (bge_col, ce_col) in enumerate(specifications, 1):
            isco_checkpoint_path = checkpoint_dir / f"isco_spec_{bge_col}_{ce_col}.parquet"

            if not isco_checkpoint_path.exists():
                logger.warning(f"  [{spec_idx}/{len(specifications)}] Missing ISCO checkpoint: {isco_checkpoint_path.name}, skipping")
                continue

            # Load spec result from checkpoint
            isco_spec_result = pd.read_parquet(isco_checkpoint_path)
            valid_isco_spec_count += 1
            logger.info(f"  [{spec_idx}/{len(specifications)}] Loaded ISCO checkpoint {bge_col}×{ce_col}: {len(isco_spec_result):,} rows")

            if isco_firm_exposure is None:
                isco_firm_exposure = isco_spec_result
                logger.info(f"    Initialized ISCO merge with {bge_col}×{ce_col}: {len(isco_spec_result):,} rows")
            else:
                # Merge with outer join
                before_cols = len(isco_firm_exposure.columns)
                isco_firm_exposure = isco_firm_exposure.merge(
                    isco_spec_result,
                    on=merge_keys,
                    how='outer',
                    suffixes=('', '_dup')
                )
                # Drop duplicate metadata columns
                dup_cols = [c for c in isco_firm_exposure.columns if c.endswith('_dup')]
                if dup_cols:
                    isco_firm_exposure.drop(columns=dup_cols, inplace=True)

                after_cols = len(isco_firm_exposure.columns)
                added_cols = after_cols - before_cols
                logger.info(f"    Merged {bge_col}×{ce_col}: {len(isco_firm_exposure):,} rows, +{added_cols} columns (total: {after_cols})")

            # Clear memory after each merge
            del isco_spec_result
            gc.collect()

        if valid_isco_spec_count == 0:
            logger.error("No valid ISCO specifications to merge!")
            raise ValueError("All ISCO specifications produced empty results or checkpoints missing!")

        logger.info(f"\n✓ Valid ISCO specifications merged: {valid_isco_spec_count}/{len(specifications)}")

        # Fill NaN exposures with 0
        exposure_cols = [c for c in isco_firm_exposure.columns
                         if 'exposure' in c.lower() or 'intensity' in c.lower() or 'n_ai_apps' in c.lower()]
        for col in exposure_cols:
            isco_firm_exposure[col] = isco_firm_exposure[col].fillna(0)

        logger.info(f"\n✓ ISCO merge complete: {len(isco_firm_exposure):,} rows, {len(isco_firm_exposure.columns)} columns")

        # Verify we have spec-suffixed columns
        spec_cols = [c for c in isco_firm_exposure.columns if '_pct_' in c or '_ce_' in c]
        logger.info(f"  Spec-suffixed columns: {len(spec_cols)} (expected: ~{valid_isco_spec_count * 6})")
        if len(spec_cols) < valid_isco_spec_count * 4:
            logger.warning(f"  WARNING: Expected ~{valid_isco_spec_count * 6} spec columns, only got {len(spec_cols)}")

        # ================================================================
        # SAVE RESULTS
        # ================================================================
        logger.info("\n" + "="*50)
        logger.info("SAVING RESULTS")
        logger.info("="*50)

        if output_file is None:
            self._ensure_output_directory(self.output_dir)
            base_filename = self._generate_filename("isco").replace(".csv", "")
            output_file = os.path.join(self.output_dir, f"{base_filename}_all_specs.csv")

        isco_firm_exposure.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved merged ISCO-08 occupation-firm-year AI exposure to: {output_file}")
        
        # ================================================================
        # PIPELINE SUMMARY
        # ================================================================
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("="*60)
        # Get total row count from parquet metadata
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(matches_file_path)
        total_matches = parquet_file.metadata.num_rows
        logger.info(f"✅ Input: {total_matches:,} task-application matches from stage 4")
        logger.info(f"✅ Processed {valid_isco_spec_count}/{len(specifications)} specifications successfully")
        logger.info(f"✅ Step 2-4: Per-spec processing with checkpointing (memory-optimized)")
        logger.info(f"✅ Final: {len(isco_firm_exposure):,} ISCO-08 occupation-firm-year exposure combinations")
        logger.info(f"✅ Unique ISCO occupations: {isco_firm_exposure['isco08_4d'].nunique():,}")
        logger.info(f"✅ Unique firms: {isco_firm_exposure['company_name'].nunique():,}")
        logger.info(f"✅ Unique years: {sorted(isco_firm_exposure['year'].unique())}")
        logger.info(f"✅ Output file: {output_file}")
        
        # Show top exposed ISCO-firm-year combinations (Hampole variant)
        logger.info(f"\nTop 10 highest AI exposure combinations (Hampole variant):")
        # Find the hampole exposure column (may have spec suffix)
        hampole_cols = [c for c in isco_firm_exposure.columns if 'hampole_ai_exposure_avg' in c]
        if hampole_cols:
            # Use the first hampole column (or the one without suffix if available)
            hampole_col = 'hampole_ai_exposure_avg' if 'hampole_ai_exposure_avg' in isco_firm_exposure.columns else hampole_cols[0]

            # Extract spec suffix from hampole_col and build corresponding n_ai_apps column name
            spec_suffix = hampole_col.replace('hampole_ai_exposure_avg', '')
            n_apps_col = f'n_ai_apps_firm_year{spec_suffix}'

            top_exposed = isco_firm_exposure.nlargest(10, hampole_col)
            for _, row in top_exposed.iterrows():
                logger.info(f"  {row['isco08_4d']} @ {row['company_name']} ({row['year']:.0f}): "
                           f"AI Exposure = {row[hampole_col]:.3f} "
                           f"({row[n_apps_col]:.0f} apps)")
        else:
            logger.warning("No hampole_ai_exposure_avg column found in merged ISCO data")

        # ================================================================
        # CLEANUP CHECKPOINT DIRECTORY
        # ================================================================
        logger.info("\n" + "="*50)
        logger.info("CLEANING UP CHECKPOINTS")
        logger.info("="*50)

        try:
            shutil.rmtree(checkpoint_dir)
            logger.info(f"✓ Successfully deleted checkpoint directory: {checkpoint_dir}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete checkpoint directory {checkpoint_dir}: {e}")
            logger.warning(f"  You may want to manually delete it to free up disk space")

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
    parser.add_argument("--stage-4-dir", type=str, default="Data/stage_4/")
    parser.add_argument("--stage-2-dir", type=str, default="Data/stage_2/")
    parser.add_argument("--top-matches-file", type=str, default=None,
                       help="Path to Stage 4 output file (auto-detect from task type if not provided)")
    parser.add_argument("--aggregation", type=str, default="mean",
                       choices=["mean"],
                       help="Aggregation method for O*NET → ISCO (default: mean)")
    parser.add_argument("--data-dir", type=str, default="Data/",
                       help="Directory containing input data files (default: Data/)")
    parser.add_argument("--bge-percentiles", type=float, nargs='+', default=None,
                       help="BGE percentile cutoffs to process (auto-detected from Stage 4 if not provided)")
    parser.add_argument("--ce-thresholds", type=float, nargs='+', default=[0.8, 0.6, 0.4, 0.2, 0.0],
                       help="Cross-encoder thresholds to process (default: 0.8 0.6 0.4 0.2 0.0)")
    parser.add_argument("--time-invariant", action="store_true",
                       help="Use time-invariant exposure (firms exposed to all their AI apps across all years)")
    parser.add_argument("--ignore-min-threshold", action="store_true",
                       help="Ignore minimum similarity threshold (use percentile columns only, not above_min_threshold)")
    parser.add_argument("--exposure-table-format", type=str, default="exposed-firm-occ-year",
                       choices=["exposed-firm-occ-year", "all-firm-occ-year", "exposed-occ-year"],
                       help="Exposure table format: 'exposed-firm-occ-year' (only firm-occupation-year combinations with exposure > 0), 'all-firm-occ-year' (all firm-occupation-year combinations including zeros), 'exposed-occ-year' (occupation-year only, constant across firms)")
    parser.add_argument("--create-sample-report", action="store_true",
                       help="Create detailed ISCO task sample report")
    parser.add_argument("--n-sample", type=int, default=10,
                       help="Number of ISCO codes to sample for report (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Cross encoder threshold for exposed tasks (default: 0.6)")
    parser.add_argument("--jobs-file", type=str,
                       help="Path to original jobs CSV file (auto-detect if not provided)")
    parser.add_argument("--save-onet-outputs", action="store_true",
                       help="Save intermediate O*NET exposure files before ISCO crosswalk")
    parser.add_argument("--skip-debug-csv", action="store_true",
                       help="Skip saving large debug CSVs (expanded_jobs_matching.csv, task_app_matches.csv) to output-dir")
    parser.add_argument("--force-nonchunked", action="store_true",
                       help="Disable chunk-based processing paths (may be faster but uses more RAM)")
    parser.add_argument("--output-dir", type=str,
                       help="Directory for all output files (both O*NET and ISCO) (defaults to Data/firm_year_exposure/)")

    parser.add_argument("--task-type", type=str,
                       choices=['core', 'all'],
                       default='core',
                       help="Task type: 'core' (default) or 'all'")
    parser.add_argument("--model", type=str,
                       default='BGE',
                       help="Word Embedding model used in Stage 4 (default: BGE)")

    parser.add_argument("--onet-version", type=int, default=20,
                       help="O*NET version (default: 20). NOTE: v25+ incompatible with BLS SOC-10 crosswalk")

    parser.add_argument("--use-employment-weights", action="store_true",
                       help="Use employment-weighted aggregation for O*NET→ISCO crosswalk (via 6-digit SOC). "
                            "If False, uses simple mean aggregation (default).")

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
            exposure_table_format=args.exposure_table_format,
            data_dir=args.data_dir,
            stage_2_dir=args.stage_2_dir,
            stage_4_dir=args.stage_4_dir,
            output_dir=args.output_dir,
            bge_percentiles=args.bge_percentiles,
            ce_thresholds=args.ce_thresholds,
            task_type=args.task_type,
            onet_version=args.onet_version,
            use_employment_weights=args.use_employment_weights,
            apply_min_threshold=not args.ignore_min_threshold,
            save_debug_csv=not args.skip_debug_csv,
            force_nonchunked=args.force_nonchunked
        )

        # Auto-detect Stage 4 file based on task type if not provided
        top_matches_file = args.top_matches_file
        if top_matches_file is None:
            top_matches_file = pipeline._find_stage4_file(args.task_type, args.model)

        result_df = pipeline.run_full_pipeline(
            top_matches_file=top_matches_file,
            company_file=args.jobs_file,
            output_file=None,  # always use auto-generated isco_firm_year_exposure* filename
            save_onet_outputs=args.save_onet_outputs,
            task_type=args.task_type,
            model=args.model,
            onet_version=args.onet_version
        )
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Final dataset: {len(result_df)} ISCO-08 occupations with exposure scores")

if __name__ == "__main__":
    main()
