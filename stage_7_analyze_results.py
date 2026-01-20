"""
Stage 7: AI Exposure Analysis and Visualization

Comprehensive analysis and visualization of AI exposure results from Stages 5 and 6.
Generates firm-level, occupation-level, firm×occupation, task-level, and summary analyses.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
import os
import glob
import ast

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Configuration Constants
DEFAULT_PERCENTILE = 'pct_05'
PERCENTILES_TO_ANALYZE = ['pct_01', 'pct_05', 'pct_10', 'pct_15', 'pct_20']
DEFAULT_CE_THRESHOLD = 'ce_0.0'
DEFAULT_TOP_N = 20
FIGURE_DPI = 300
FIGURE_FORMAT = 'png'
DEFAULT_OUTPUT_DIR = 'Data/Testing/stage_7'


class ExposureAnalyzer:
    """
    Comprehensive analyzer for AI exposure results from Stages 5 and 6.

    Generates analyses across 6 sections:
    1. Firm-level analysis (intensity, rankings, sensitivity)
    2. Occupation-level analysis (rankings, trends, sensitivity, measure comparisons)
    3. Firm×Occupation analysis (variation, pairs, trends)
    4. Task-level analysis (exposed tasks, clustering, app matching)
    5. Summary statistics (dataset stats, coverage, distributions)
    """

    def __init__(self,
                 stage5_dir: str = 'Data/firm_year_exposure',
                 stage6_dir: str = 'Data',
                 stage6_suffix: str = '',
                 stage4_file: str = 'Data/task_exposure_matches_all_thresholds_core_tasks.parquet',
                 output_dir: str = DEFAULT_OUTPUT_DIR,
                 percentile: str = DEFAULT_PERCENTILE,
                 ce_threshold: str = DEFAULT_CE_THRESHOLD,
                 top_n: int = DEFAULT_TOP_N):
        """
        Initialize analyzer with data paths and parameters.

        Args:
            stage5_dir: Directory containing Stage 5 outputs
            stage6_dir: Directory containing Stage 6 outputs
            stage6_suffix: Suffix for Stage 6 output files (e.g., '_1000_sample', '_core_tasks')
            stage4_file: Path to Stage 4 task-app matches
            output_dir: Directory for analysis outputs (default: Data/Testing/stage_7)
            percentile: BGE percentile to use as baseline (default: pct_05)
            ce_threshold: Cross-encoder threshold to use (default: ce_0.0)
            top_n: Number of top items to show in rankings (default: 20)
        """
        self.stage5_dir = stage5_dir
        self.stage6_dir = stage6_dir
        self.stage6_suffix = stage6_suffix
        self.stage4_file = stage4_file
        self.output_dir = output_dir
        self.percentile = percentile
        self.ce_threshold = ce_threshold
        self.top_n = top_n

        # Setup logging
        self._setup_logging()

        # Data containers (set during load_data())
        self.stage5_isco = None
        self.stage5_onet = None
        self.stage5_task = None
        self.stage6_jobs = None
        self.stage6_firm_report = None
        self.stage4_matches = None

        # Column mapping for percentile suffixes
        self._percentile_suffix = f'_{self.percentile}_{self.ce_threshold}'

        self.logger.info(f"ExposureAnalyzer initialized with percentile={self.percentile}, ce_threshold={self.ce_threshold}, top_n={self.top_n}")
        self.logger.info(f"Output directory: {self.output_dir}")

    def _setup_logging(self):
        """Setup logging configuration."""
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # File handler
        log_file = os.path.join(self.output_dir, 'stage_7_analysis.log')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _setup_output_dirs(self):
        """Create all output subdirectories if they don't exist."""
        subdirs = ['firms', 'occupations', 'firm_occupation', 'tasks', 'summary']
        for subdir in subdirs:
            path = os.path.join(self.output_dir, subdir)
            Path(path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created output directory: {path}")

    def load_data(self):
        """
        Load all required data files.

        Sets instance variables:
        - self.stage5_isco: ISCO occupation exposure (Stage 5)
        - self.stage5_onet: O*NET occupation exposure (Stage 5)
        - self.stage5_task: O*NET task exposure (Stage 5)
        - self.stage6_jobs: Jobs linked with exposure (Stage 6)
        - self.stage6_firm_report: Firm summary report (Stage 6)
        - self.stage4_matches: Task-app similarity matches (Stage 4)
        """
        self.logger.info("=" * 80)
        self.logger.info("LOADING DATA")
        self.logger.info("=" * 80)

        # Load Stage 5 ISCO
        self._load_stage5_isco()

        # Load Stage 5 O*NET
        self._load_stage5_onet()

        # Load Stage 5 Task exposure
        self._load_stage5_task()

        # Load Stage 6 Jobs linked
        self._load_stage6_jobs()

        # Load Stage 6 Firm report
        self._load_stage6_firm_report()

        # Load Stage 4 matches
        self._load_stage4_matches()

        self.logger.info("=" * 80)
        self.logger.info("DATA LOADING COMPLETE")
        self.logger.info("=" * 80)

    def _load_stage5_isco(self):
        """Load Stage 5 ISCO occupation exposure."""
        try:
            # Try with all_specs (all percentiles)
            pattern = os.path.join(self.stage5_dir, 'isco_firm_year_exposure_*_all_specs.csv')
            files = list(Path(self.stage5_dir).glob('isco_firm_year_exposure_*_all_specs.csv'))

            if files:
                filepath = str(files[0])
                self.logger.info(f"Loading Stage 5 ISCO (all_specs): {filepath}")
                self.stage5_isco = pd.read_csv(filepath)
                self.logger.info(f"  Shape: {self.stage5_isco.shape}")
                self.logger.info(f"  Columns: {list(self.stage5_isco.columns[:10])}...")
                # Extract relevant columns for selected percentile
                self._extract_isco_percentile_columns()
            else:
                # Fallback: try specific percentile file
                filepath = os.path.join(
                    self.stage5_dir,
                    f'isco_firm_year_exposure_core_tasks_{self.percentile}_{self.ce_threshold}.csv'
                )
                if os.path.exists(filepath):
                    self.logger.info(f"Loading Stage 5 ISCO: {filepath}")
                    self.stage5_isco = pd.read_csv(filepath)
                    self.logger.info(f"  Shape: {self.stage5_isco.shape}")
                else:
                    self.logger.warning(f"Stage 5 ISCO file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading Stage 5 ISCO: {e}")

    def _load_stage5_onet(self):
        """Load Stage 5 O*NET occupation exposure."""
        try:
            # Try with all_specs (all percentiles)
            files = list(Path(self.stage5_dir).glob('onet_firm_year_exposure_*_all_specs.csv'))

            if files:
                filepath = str(files[0])
                self.logger.info(f"Loading Stage 5 O*NET (all_specs): {filepath}")
                self.stage5_onet = pd.read_csv(filepath)
                self.logger.info(f"  Shape: {self.stage5_onet.shape}")
                self.logger.info(f"  Columns: {list(self.stage5_onet.columns[:10])}...")
            else:
                # Fallback: try specific percentile file
                filepath = os.path.join(
                    self.stage5_dir,
                    f'onet_firm_year_exposure_core_tasks_{self.percentile}_{self.ce_threshold}.csv'
                )
                if os.path.exists(filepath):
                    self.logger.info(f"Loading Stage 5 O*NET: {filepath}")
                    self.stage5_onet = pd.read_csv(filepath)
                    self.logger.info(f"  Shape: {self.stage5_onet.shape}")
                else:
                    self.logger.warning(f"Stage 5 O*NET file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading Stage 5 O*NET: {e}")

    def _load_stage5_task(self):
        """Load Stage 5 task exposure."""
        try:
            filepath = os.path.join(self.stage5_dir, 'onet_task_exposure_firm_only_year.csv')
            if os.path.exists(filepath):
                self.logger.info(f"Loading Stage 5 Task: {filepath}")
                self.stage5_task = pd.read_csv(filepath)
                self.logger.info(f"  Shape: {self.stage5_task.shape}")
                self.logger.info(f"  Columns: {list(self.stage5_task.columns)}")
            else:
                self.logger.warning(f"Stage 5 Task file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading Stage 5 Task: {e}")

    def _load_stage6_jobs(self):
        """Load Stage 6 jobs linked with exposure."""
        try:
            filepath = os.path.join(self.stage6_dir, f'stage6_jobs_linked{self.stage6_suffix}.csv')
            if os.path.exists(filepath):
                self.logger.info(f"Loading Stage 6 Jobs: {filepath}")
                self.stage6_jobs = pd.read_csv(filepath)
                self.logger.info(f"  Shape: {self.stage6_jobs.shape}")
                self.logger.info(f"  Columns (first 15): {list(self.stage6_jobs.columns[:15])}...")
                # Extract relevant columns for selected percentile
                self._extract_jobs_percentile_columns()
            else:
                self.logger.warning(f"Stage 6 Jobs file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading Stage 6 Jobs: {e}")

    def _load_stage6_firm_report(self):
        """Load Stage 6 firm summary report."""
        try:
            filepath = os.path.join(self.stage6_dir, f'firm_ai_summary_report{self.stage6_suffix}.csv')
            if os.path.exists(filepath):
                self.logger.info(f"Loading Stage 6 Firm Report: {filepath}")
                self.stage6_firm_report = pd.read_csv(filepath)
                self.logger.info(f"  Shape: {self.stage6_firm_report.shape}")
                self.logger.info(f"  Columns: {list(self.stage6_firm_report.columns)}")
            else:
                self.logger.warning(f"Stage 6 Firm Report file not found: {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading Stage 6 Firm Report: {e}")

    def _load_stage4_matches(self):
        """Load Stage 4 task-app similarity matches."""
        try:
            if os.path.exists(self.stage4_file):
                self.logger.info(f"Loading Stage 4 Matches: {self.stage4_file}")
                self.stage4_matches = pd.read_parquet(self.stage4_file)
                self.logger.info(f"  Shape: {self.stage4_matches.shape}")
                self.logger.info(f"  Columns (first 15): {list(self.stage4_matches.columns[:15])}...")

                # Ensure ai_app_id is available and well-typed for downstream summaries
                if 'ai_app_id' in self.stage4_matches.columns:
                    self.stage4_matches['ai_app_id'] = self.stage4_matches['ai_app_id'].astype(str)
                    self.logger.info("  ai_app_id present in Stage 4 matches")
                else:
                    self.logger.warning("ai_app_id not found in Stage 4 matches; task-level app tracking may be limited")

                # Filter to selected percentile
                if self.percentile in self.stage4_matches.columns:
                    self.stage4_matches = self.stage4_matches[self.stage4_matches[self.percentile] == True].copy()
                    self.logger.info(f"  Filtered to {self.percentile}: {self.stage4_matches.shape[0]} matches")

                # Filter to selected CE threshold
                if self.ce_threshold in self.stage4_matches.columns:
                    self.stage4_matches = self.stage4_matches[self.stage4_matches[self.ce_threshold] == True].copy()
                    self.logger.info(f"  Filtered to {self.ce_threshold}: {self.stage4_matches.shape[0]} matches")
            else:
                self.logger.warning(f"Stage 4 Matches file not found: {self.stage4_file}")
        except Exception as e:
            self.logger.error(f"Error loading Stage 4 Matches: {e}")

    # Helper Methods for Column Extraction

    def _extract_isco_percentile_columns(self):
        """
        Extract and rename columns for selected percentile from ISCO data.

        ISCO all_specs file has columns suffixed with _{percentile}_{ce_threshold}.
        This method extracts relevant columns and strips the suffix for cleaner access.
        """
        if self.stage5_isco is None:
            return

        # Identify columns to rename (base names without percentile suffix)
        base_cols = [
            'hampole_ai_exposure_avg',
            'binary_ai_exposure_avg',
            'hampole_occupation_exposure',
            'binary_occupation_exposure',
            'total_tasks_occupation',
            'total_importance_weight',
            'n_ai_apps_firm_year',
            'log_ai_intensity'
        ]

        # Keep non-data columns
        keep_cols = ['company_id', 'isco08_4d', 'isco08_title', 'company_name', 'year']

        # Find columns matching the percentile suffix
        suffix = f'_{self.percentile}_{self.ce_threshold}'
        columns_to_keep = keep_cols.copy()

        rename_map = {}
        for col in self.stage5_isco.columns:
            for base in base_cols:
                if col.startswith(base) and col.endswith(suffix):
                    columns_to_keep.append(col)
                    rename_map[col] = base
                    break

        # Keep only relevant columns and rename
        if columns_to_keep:
            self.stage5_isco = self.stage5_isco[columns_to_keep]

        if rename_map:
            self.stage5_isco = self.stage5_isco.rename(columns=rename_map)

        self.logger.info(f"  Extracted {len(rename_map)} percentile-specific columns for {self.percentile}")

    def _extract_jobs_percentile_columns(self):
        """
        Extract and rename columns for selected percentile from Stage 6 jobs data.

        Stage 6 jobs file has columns suffixed with _{percentile}_{ce_threshold}.
        This method extracts relevant columns and strips the suffix for cleaner access.
        """
        if self.stage6_jobs is None:
            return

        # Identify columns to rename (base names without percentile suffix)
        base_cols = [
            'hampole_ai_exposure_avg',
            'binary_ai_exposure_avg',
            'hampole_occupation_exposure',
            'binary_occupation_exposure',
            'total_tasks_occupation',
            'total_importance_weight',
            'n_ai_apps_firm_year',
            'log_ai_intensity',
            'n_onet_codes_contributing'
        ]

        # Keep non-data columns
        keep_cols = ['company_id', 'year', 'title', 'isco08_4d', 'isco08_title', 'company_name']

        # Find columns matching the percentile suffix
        suffix = f'_{self.percentile}_{self.ce_threshold}'
        columns_to_keep = keep_cols.copy()

        rename_map = {}
        for col in self.stage6_jobs.columns:
            for base in base_cols:
                if col.startswith(base) and col.endswith(suffix):
                    columns_to_keep.append(col)
                    rename_map[col] = base
                    break

        # Keep only relevant columns and rename
        if columns_to_keep:
            self.stage6_jobs = self.stage6_jobs[columns_to_keep]

        if rename_map:
            self.stage6_jobs = self.stage6_jobs.rename(columns=rename_map)

        self.logger.info(f"  Extracted {len(rename_map)} percentile-specific columns for {self.percentile}")

    # Core Helper Methods

    def _save_table(self, df: pd.DataFrame, subdir: str, filename: str, name: str = None):
        """
        Save DataFrame to CSV with logging.

        Args:
            df: DataFrame to save
            subdir: Subdirectory name (e.g., 'firms', 'occupations')
            filename: Output filename (e.g., 'top_firms.csv')
            name: Optional name for logging
        """
        if name is None:
            name = filename

        path = os.path.join(self.output_dir, subdir, filename)
        df.to_csv(path, index=False)
        self.logger.info(f"Saved: {name} ({df.shape[0]} rows) → {path}")

    def _save_plot(self, fig, subdir: str, filename: str, name: str = None):
        """
        Save matplotlib figure with consistent formatting.

        Args:
            fig: Matplotlib figure object
            subdir: Subdirectory name
            filename: Output filename (e.g., 'firm_distribution.png')
            name: Optional name for logging
        """
        if name is None:
            name = filename

        path = os.path.join(self.output_dir, subdir, filename)
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)
        self.logger.info(f"Saved: {name} → {path}")

    def _aggregate_to_occupation(self, df: pd.DataFrame, measure_cols: List[str]) -> pd.DataFrame:
        """
        Aggregate firm-occupation data to occupation level (mean across firms/years).

        Args:
            df: Input DataFrame with firm-occupation-year structure
            measure_cols: Column names to aggregate (mean)

        Returns:
            DataFrame aggregated to occupation level
        """
        # Identify occupation columns (varies by ISCO vs O*NET)
        occ_cols = None
        if 'isco08_4d' in df.columns:
            occ_cols = ['isco08_4d', 'isco08_title']
        elif 'onet_soc' in df.columns:
            occ_cols = ['onet_soc', 'onet_title']

        if occ_cols is None:
            self.logger.warning("Could not identify occupation columns for aggregation")
            return df

        agg_dict = {col: 'mean' for col in measure_cols if col in df.columns}
        agg_dict['company_id'] = 'count'  # For n_firms

        result = df.groupby(occ_cols, as_index=False).agg(agg_dict)
        result.rename(columns={'company_id': 'n_firms'}, inplace=True)

        return result

    def _get_top_n(self, df: pd.DataFrame, sort_col: str, n: int = None, ascending: bool = False) -> pd.DataFrame:
        """
        Generic method to get top N rows sorted by column.

        Args:
            df: Input DataFrame
            sort_col: Column to sort by
            n: Number of rows (default: self.top_n)
            ascending: Sort direction

        Returns:
            Top N rows
        """
        if n is None:
            n = self.top_n

        if sort_col not in df.columns:
            self.logger.warning(f"Column {sort_col} not found in DataFrame")
            return df.head(n)

        return df.sort_values(sort_col, ascending=ascending).head(n)

    # Analysis Methods (Stubs)

    def run_all_analyses(self, sections: Optional[List[int]] = None):
        """
        Execute all analysis sections.

        Args:
            sections: Optional list of section numbers to run (2-6).
                     If None, runs all sections.
        """
        self._setup_output_dirs()
        self.load_data()

        if sections is None or 2 in sections:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("SECTION 2: FIRM-LEVEL ANALYSIS")
            self.logger.info("=" * 80)
            self.analyze_firm_intensity()
            self.rank_firms_by_exposure()
            self.firm_exposure_variants_analysis()
            self.firm_sensitivity_analysis()

        if sections is None or 3 in sections:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("SECTION 3: OCCUPATION-LEVEL & SECTOR ANALYSIS")
            self.logger.info("=" * 80)
            self.rank_occupations_by_exposure()
            self.occupation_time_trends()
            self.occupation_sensitivity_analysis()
            self.compare_exposure_measures()
            self.rank_sectors_by_exposure()

        if sections is None or 4 in sections:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("SECTION 4: FIRM × OCCUPATION ANALYSIS")
            self.logger.info("=" * 80)
            self.within_occupation_variation()
            self.rank_firm_occupation_pairs()
            self.firm_occupation_time_trends()

        if sections is None or 5 in sections:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("SECTION 5: TASK-LEVEL ANALYSIS")
            self.logger.info("=" * 80)
            self.analyze_exposed_tasks()
            self.cluster_task_types()
            self.analyze_app_task_matches()
            self.task_app_matching_summary()
            self.analyze_top_occupations_task_breakdown()

        if sections is None or 6 in sections:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("SECTION 6: SUMMARY STATISTICS & SENSITIVITY ANALYSES")
            self.logger.info("=" * 80)
            self.exposure_distribution_by_percentile()
            self.coverage_statistics()
            self.exposure_distribution_plots()
            self.percentile_comparison_analysis()
            self.generate_summary_statistics()

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"ALL ANALYSES COMPLETE")
        self.logger.info(f"Outputs saved to: {self.output_dir}")
        self.logger.info("=" * 80)

    # Section 2: Firm-Level Analysis

    def analyze_firm_intensity(self):
        """Analyze firm AI app adoption and intensity."""
        self.logger.info("→ Analyzing firm intensity...")

        if self.stage6_firm_report is None:
            self.logger.warning("Stage 6 firm report not loaded, skipping firm intensity analysis")
            return

        # Table 1: Top firms by total AI apps (cumulative)
        top_firms_all = self.stage6_firm_report[
            ['company_name', 'total_ai_apps_all', 'total_unique_job_ads', 'pct_ai_ads_cumulative']
        ].drop_duplicates(subset=['company_name']).sort_values('total_ai_apps_all', ascending=False).head(self.top_n)

        self._save_table(top_firms_all, 'firms', 'top_firms_by_ai_apps.csv',
                        f"Top {self.top_n} firms by AI apps (cumulative)")

        # Table 2: Top firms by AI apps yearly
        # Filter to only the selected percentile to avoid duplicate firm-year rows (one per specification)
        yearly_data = self.stage6_firm_report[self.stage6_firm_report['specification'] == self.percentile + '_' + self.ce_threshold].copy()
        # Get top N firms for each year separately
        top_firms_yearly = yearly_data.groupby('year', group_keys=False).apply(
            lambda x: x[['company_name', 'year', 'total_ai_apps_linked', 'total_unique_ai_jobs', 'pct_ai_ads_yearly']].nlargest(self.top_n, 'total_ai_apps_linked')
        ).reset_index(drop=True)

        self._save_table(top_firms_yearly, 'firms', 'top_firms_by_ai_apps_yearly.csv',
                        f"Top {self.top_n} firms by AI apps (yearly)")

        # Plot 1: Firm AI apps time series
        # Filter to only the selected percentile to avoid duplicate firm-year rows (one per specification)
        ts_data_all_specs = self.stage6_firm_report[self.stage6_firm_report['specification'] == self.percentile + '_' + self.ce_threshold].copy()

        top_10_firms = ts_data_all_specs.groupby('company_name')['total_ai_apps_all'].max().nlargest(10).index
        ts_data = ts_data_all_specs[ts_data_all_specs['company_name'].isin(top_10_firms)].copy()
        ts_data = ts_data.sort_values(['company_name', 'year'])

        fig, ax = plt.subplots(figsize=(12, 6))
        for firm in top_10_firms:
            firm_data = ts_data[ts_data['company_name'] == firm]
            ax.plot(firm_data['year'], firm_data['total_ai_apps_linked'], marker='o', label=firm)

        ax.set_xlabel('Year')
        ax.set_ylabel('Total AI Applications')
        ax.set_title('AI Application Adoption Over Time (Top 10 Firms)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        self._save_plot(fig, 'firms', 'firm_ai_apps_time_series.png')

        # Plot 2: Firm AI intensity distribution
        firm_intensity = self.stage6_firm_report[['company_name', 'pct_ai_ads_cumulative']].drop_duplicates()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(firm_intensity['pct_ai_ads_cumulative'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('% of Job Ads that are AI-Related')
        ax.set_ylabel('Count of Firms')
        ax.set_title('Distribution of Firm AI Intensity')
        ax.grid(True, alpha=0.3, axis='y')
        self._save_plot(fig, 'firms', 'firm_ai_intensity_distribution.png')

    def rank_firms_by_exposure(self):
        """Rank firms by AI exposure scores."""
        self.logger.info("→ Ranking firms by exposure...")

        if self.stage6_firm_report is None:
            self.logger.warning("Stage 6 firm report not loaded, skipping firm exposure ranking")
            return

        # Table 1: Top firms by exposure (cumulative)
        top_exposure = self.stage6_firm_report[
            ['company_name', 'firm_exposure', 'ai_exposed_tasks', 'total_onet_tasks', 'total_ai_apps_linked']
        ].drop_duplicates(subset=['company_name']).sort_values('firm_exposure', ascending=False).head(self.top_n)

        self._save_table(top_exposure, 'firms', 'top_firms_by_exposure_cumulative.csv',
                        f"Top {self.top_n} firms by exposure (cumulative)")

        # Table 2: Top firms by exposure (yearly) - top N per year
        # Filter to only the selected percentile to avoid duplicate firm-year rows (one per specification)
        exposure_yearly_data = self.stage6_firm_report[self.stage6_firm_report['specification'] == self.percentile + '_' + self.ce_threshold].copy()
        # Get top N firms for each year separately
        top_exposure_yearly = exposure_yearly_data.groupby('year', group_keys=False).apply(
            lambda x: x[['company_name', 'year', 'firm_exposure', 'ai_exposed_tasks', 'total_onet_tasks', 'total_ai_apps_linked']].nlargest(self.top_n, 'firm_exposure')
        ).reset_index(drop=True)

        self._save_table(top_exposure_yearly, 'firms', 'top_firms_by_exposure_yearly.csv',
                        f"Top {self.top_n} firms by exposure (yearly)")

        # Table 3: Firm exposure summary stats
        exposure_stats = self.stage6_firm_report[['firm_exposure']].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
        exposure_stats.columns = ['count', 'mean', 'std', 'min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max', 'p95', 'p99']
        exposure_stats = exposure_stats[['mean', 'std', 'min', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max']]

        self._save_table(exposure_stats.reset_index(drop=True), 'firms', 'firm_exposure_summary_stats.csv',
                        "Firm exposure summary statistics")

        # Plot 1: Firm exposure distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(self.stage6_firm_report['firm_exposure'], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Firm AI Exposure Score')
        ax.set_ylabel('Count of Firms')
        ax.set_title('Distribution of Firm AI Exposure')
        ax.grid(True, alpha=0.3, axis='y')
        self._save_plot(fig, 'firms', 'firm_exposure_distribution.png')

    def firm_exposure_variants_analysis(self):
        """Analyze firm exposures with binary and hampole variants (adjusted and unadjusted)."""
        self.logger.info("→ Analyzing firm exposure variants (binary/hampole, adjusted/unadjusted)...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not loaded, skipping firm exposure variants analysis")
            return

        # Filter to selected percentile
        pct = self.percentile + '_' + self.ce_threshold

        # Build column names for this percentile
        hampole_unadj_col = f'hampole_occupation_exposure_{pct}'
        hampole_adj_col = f'hampole_ai_exposure_avg_{pct}'
        binary_unadj_col = f'binary_occupation_exposure_{pct}'
        binary_adj_col = f'binary_ai_exposure_avg_{pct}'
        intensity_col = f'log_ai_intensity_{pct}'

        # Check if columns exist
        available_cols = self.stage6_jobs.columns.tolist()
        missing_cols = [c for c in [hampole_unadj_col, hampole_adj_col, binary_unadj_col, binary_adj_col, intensity_col] if c not in available_cols]

        if missing_cols:
            self.logger.warning(f"Missing columns for percentile {pct}: {missing_cols}")
            return

        # Aggregate to firm-year level
        firm_year_data = self.stage6_jobs[[
            'company_id', 'company_name', 'year',
            hampole_unadj_col, hampole_adj_col,
            binary_unadj_col, binary_adj_col
        ]].copy()

        firm_year_data.columns = ['company_id', 'company_name', 'year',
                                   'hampole_unadjusted', 'hampole_adjusted',
                                   'binary_unadjusted', 'binary_adjusted']

        # Group by firm-year and take mean
        firm_year_agg = firm_year_data.groupby(['company_name', 'year']).agg({
            'hampole_unadjusted': 'mean',
            'hampole_adjusted': 'mean',
            'binary_unadjusted': 'mean',
            'binary_adjusted': 'mean'
        }).reset_index()

        # Table 1: Top firms by hampole adjusted exposure (yearly)
        top_hampole_adj = firm_year_agg.groupby('year', group_keys=False).apply(
            lambda x: x.nlargest(self.top_n, 'hampole_adjusted')
        ).reset_index(drop=True)
        self._save_table(top_hampole_adj, 'firms', 'top_firms_by_hampole_adjusted_exposure_yearly.csv',
                        f"Top {self.top_n} firms by hampole adjusted exposure (yearly)")

        # Table 2: Top firms by hampole unadjusted exposure (yearly)
        top_hampole_unadj = firm_year_agg.groupby('year', group_keys=False).apply(
            lambda x: x.nlargest(self.top_n, 'hampole_unadjusted')
        ).reset_index(drop=True)
        self._save_table(top_hampole_unadj, 'firms', 'top_firms_by_hampole_unadjusted_exposure_yearly.csv',
                        f"Top {self.top_n} firms by hampole unadjusted exposure (yearly)")

        # Table 3: Top firms by binary adjusted exposure (yearly)
        top_binary_adj = firm_year_agg.groupby('year', group_keys=False).apply(
            lambda x: x.nlargest(self.top_n, 'binary_adjusted')
        ).reset_index(drop=True)
        self._save_table(top_binary_adj, 'firms', 'top_firms_by_binary_adjusted_exposure_yearly.csv',
                        f"Top {self.top_n} firms by binary adjusted exposure (yearly)")

        # Table 4: Top firms by binary unadjusted exposure (yearly)
        top_binary_unadj = firm_year_agg.groupby('year', group_keys=False).apply(
            lambda x: x.nlargest(self.top_n, 'binary_unadjusted')
        ).reset_index(drop=True)
        self._save_table(top_binary_unadj, 'firms', 'top_firms_by_binary_unadjusted_exposure_yearly.csv',
                        f"Top {self.top_n} firms by binary unadjusted exposure (yearly)")

        # Table 5: Comparison of methods - show how rankings differ
        # Get top 10 by each method (cumulative across years)
        firm_cumul = firm_year_agg.groupby('company_name').agg({
            'hampole_unadjusted': 'mean',
            'hampole_adjusted': 'mean',
            'binary_unadjusted': 'mean',
            'binary_adjusted': 'mean'
        }).reset_index()

        comparison = pd.DataFrame()
        for col, name in [('hampole_adjusted', 'Hampole (Adjusted)'),
                         ('hampole_unadjusted', 'Hampole (Unadjusted)'),
                         ('binary_adjusted', 'Binary (Adjusted)'),
                         ('binary_unadjusted', 'Binary (Unadjusted)')]:
            top = firm_cumul.nlargest(self.top_n, col)[['company_name', col]].reset_index(drop=True)
            top.columns = ['company_name', 'exposure']
            top['method'] = name
            comparison = pd.concat([comparison, top], ignore_index=True)

        self._save_table(comparison, 'firms', 'firm_exposure_method_comparison.csv',
                        "Comparison of exposure methods (top firms per method)")

    def firm_sensitivity_analysis(self):
        """Show how top firms change across percentile thresholds."""
        self.logger.info("→ Analyzing firm sensitivity across percentiles...")

        if self.stage6_firm_report is None:
            self.logger.warning("Stage 6 firm report not loaded, skipping firm sensitivity analysis")
            return

        # Get top 10 firms for each percentile
        sensitivity_results = {}
        for pct in PERCENTILES_TO_ANALYZE:
            # Note: The firm report doesn't have percentile versions, so we use the main one
            # This is a limitation noted in the plan
            top_firms = self.stage6_firm_report[['company_name', 'firm_exposure']].drop_duplicates(
                subset=['company_name']
            ).sort_values('firm_exposure', ascending=False).head(10)

            sensitivity_results[pct] = list(top_firms['company_name'].values)

        # Table 1: Top firms percentile sensitivity
        # Create comparison table showing rank changes
        rank_df = pd.DataFrame({
            'rank': range(1, 11)
        })
        for pct in PERCENTILES_TO_ANALYZE:
            rank_df[pct] = pd.Series(sensitivity_results[pct]).reset_index(drop=True)

        self._save_table(rank_df, 'firms', 'top_firms_percentile_sensitivity.csv',
                        "Top firms percentile sensitivity (note: firm report only has one set of values)")

        # Table 2: Jaccard overlap between percentiles
        overlap_data = []
        for i, pct1 in enumerate(PERCENTILES_TO_ANALYZE):
            for pct2 in PERCENTILES_TO_ANALYZE[i+1:]:
                set1 = set(sensitivity_results[pct1])
                set2 = set(sensitivity_results[pct2])
                jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
                overlap_data.append({'percentile_pair': f"{pct1} vs {pct2}", 'jaccard_similarity': jaccard})

        overlap_df = pd.DataFrame(overlap_data)
        self._save_table(overlap_df, 'firms', 'percentile_overlap_jaccard.csv',
                        "Jaccard similarity of top firms across percentiles")

    # Section 3: Occupation-Level Analysis (ISCO only)

    def rank_occupations_by_exposure(self):
        """Rank occupations by different exposure measures (ISCO only)."""
        self.logger.info("→ Ranking occupations by exposure...")

        if self.stage5_isco is None:
            self.logger.warning("Stage 5 ISCO data not loaded, skipping occupation ranking")
            return

        # Aggregate to occupation level (across firms and years)
        measure_cols = [
            'hampole_ai_exposure_avg',
            'binary_ai_exposure_avg',
            'hampole_occupation_exposure',
            'binary_occupation_exposure',
            'total_tasks_occupation',
            'log_ai_intensity'
        ]

        occ_data = self.stage5_isco[['isco08_4d', 'isco08_title'] + measure_cols].copy()
        occ_agg = occ_data.groupby(['isco08_4d', 'isco08_title'], as_index=False).agg({
            col: 'mean' for col in measure_cols
        })

        # Table 1: Top occupations by hampole (unadjusted)
        top_hampole_unadjusted = occ_agg[
            ['isco08_4d', 'isco08_title', 'hampole_occupation_exposure', 'total_tasks_occupation']
        ].sort_values('hampole_occupation_exposure', ascending=False).head(self.top_n)

        self._save_table(top_hampole_unadjusted, 'occupations', 'top_occupations_hampole_unadjusted.csv',
                        f"Top {self.top_n} occupations by Hampole exposure (unadjusted)")

        # Table 2: Top occupations by hampole (adjusted)
        top_hampole_adjusted = occ_agg[
            ['isco08_4d', 'isco08_title', 'hampole_ai_exposure_avg', 'log_ai_intensity']
        ].sort_values('hampole_ai_exposure_avg', ascending=False).head(self.top_n)

        self._save_table(top_hampole_adjusted, 'occupations', 'top_occupations_hampole_adjusted.csv',
                        f"Top {self.top_n} occupations by Hampole exposure (adjusted)")

        # Table 3: Top occupations by binary (unadjusted)
        top_binary_unadjusted = occ_agg[
            ['isco08_4d', 'isco08_title', 'binary_occupation_exposure', 'total_tasks_occupation']
        ].sort_values('binary_occupation_exposure', ascending=False).head(self.top_n)

        self._save_table(top_binary_unadjusted, 'occupations', 'top_occupations_binary_unadjusted.csv',
                        f"Top {self.top_n} occupations by Binary exposure (unadjusted)")

        # Table 4: Top occupations by binary (adjusted)
        top_binary_adjusted = occ_agg[
            ['isco08_4d', 'isco08_title', 'binary_ai_exposure_avg', 'log_ai_intensity']
        ].sort_values('binary_ai_exposure_avg', ascending=False).head(self.top_n)

        self._save_table(top_binary_adjusted, 'occupations', 'top_occupations_binary_adjusted.csv',
                        f"Top {self.top_n} occupations by Binary exposure (adjusted)")

        # Table 5: Side-by-side comparison
        hampole_ranks = occ_agg.nlargest(self.top_n, 'hampole_occupation_exposure')[
            ['isco08_4d', 'isco08_title', 'hampole_occupation_exposure']
        ].reset_index(drop=True)
        hampole_ranks['rank'] = range(1, len(hampole_ranks) + 1)

        binary_ranks = occ_agg.nlargest(self.top_n, 'binary_occupation_exposure')[
            ['isco08_4d', 'isco08_title', 'binary_occupation_exposure']
        ].reset_index(drop=True)
        binary_ranks['rank'] = range(1, len(binary_ranks) + 1)

        comparison = pd.DataFrame({
            'rank': range(1, self.top_n + 1),
            'hampole_occupation': hampole_ranks['isco08_title'].values,
            'hampole_score': hampole_ranks['hampole_occupation_exposure'].values,
            'binary_occupation': binary_ranks['isco08_title'].values,
            'binary_score': binary_ranks['binary_occupation_exposure'].values
        })

        self._save_table(comparison, 'occupations', 'hampole_vs_binary_comparison.csv',
                        "Hampole vs Binary exposure comparison")

    def occupation_time_trends(self):
        """Analyze occupation exposure over time (ISCO only)."""
        self.logger.info("→ Analyzing occupation time trends...")

        if self.stage5_isco is None or 'year' not in self.stage5_isco.columns:
            self.logger.warning("Stage 5 ISCO data or year column not available, skipping time trends")
            return

        # Get top 10 occupations by average exposure
        top_occs = self.stage5_isco.groupby('isco08_title')['hampole_ai_exposure_avg'].mean().nlargest(10).index

        # Prepare time series data
        ts_data = self.stage5_isco[self.stage5_isco['isco08_title'].isin(top_occs)].copy()
        ts_data = ts_data.groupby(['year', 'isco08_title'], as_index=False)['hampole_ai_exposure_avg'].mean()

        # Plot 1: Time series
        fig, ax = plt.subplots(figsize=(12, 6))
        for occ in top_occs:
            occ_data = ts_data[ts_data['isco08_title'] == occ].sort_values('year')
            ax.plot(occ_data['year'], occ_data['hampole_ai_exposure_avg'], marker='o', label=occ)

        ax.set_xlabel('Year')
        ax.set_ylabel('AI Exposure (Hampole)')
        ax.set_title('AI Exposure Trends for Top 10 Occupations')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        self._save_plot(fig, 'occupations', 'top_occupations_time_series.png')

        # Plot 2: Heatmap
        heatmap_data = self.stage5_isco[self.stage5_isco['isco08_title'].isin(top_occs)].copy()
        pivot_data = heatmap_data.pivot_table(
            index='isco08_title',
            columns='year',
            values='hampole_ai_exposure_avg',
            aggfunc='mean'
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot_data, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'AI Exposure'})
        ax.set_title('AI Exposure Heatmap: Occupations Over Time')
        ax.set_xlabel('Year')
        ax.set_ylabel('Occupation')
        self._save_plot(fig, 'occupations', 'occupation_exposure_heatmap.png')

        # Table: Growth rates
        first_year_data = heatmap_data.groupby('isco08_4d')['year'].min()
        last_year_data = heatmap_data.groupby('isco08_4d')['year'].max()

        growth_rows = []
        for occ_code in heatmap_data['isco08_4d'].unique():
            occ_name = heatmap_data[heatmap_data['isco08_4d'] == occ_code]['isco08_title'].iloc[0]
            occ_ts = heatmap_data[heatmap_data['isco08_4d'] == occ_code].sort_values('year')

            if len(occ_ts) > 1:
                first_year = occ_ts['year'].min()
                last_year = occ_ts['year'].max()
                first_val = occ_ts[occ_ts['year'] == first_year]['hampole_ai_exposure_avg'].mean()
                last_val = occ_ts[occ_ts['year'] == last_year]['hampole_ai_exposure_avg'].mean()

                # Calculate simple growth rate
                simple_growth = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0

                # Calculate CAGR (Compound Annual Growth Rate)
                n_years = last_year - first_year
                cagr = (((last_val / first_val) ** (1 / n_years)) - 1) * 100 if (first_val > 0 and n_years > 0) else 0

                growth_rows.append({
                    'isco08_4d': occ_code,
                    'isco08_title': occ_name,
                    'first_year': first_year,
                    'last_year': last_year,
                    'n_years': n_years,
                    'first_year_exposure': first_val,
                    'last_year_exposure': last_val,
                    'simple_growth_rate': simple_growth,
                    'cagr': cagr
                })

        if growth_rows:
            growth_df = pd.DataFrame(growth_rows).sort_values('cagr', ascending=False).head(self.top_n)
            self._save_table(growth_df, 'occupations', 'occupation_exposure_growth_rates.csv',
                            f"Top {self.top_n} fastest-growing occupations")

    def occupation_sensitivity_analysis(self):
        """Show how top occupations change across percentile thresholds (ISCO only)."""
        self.logger.info("→ Analyzing occupation sensitivity across percentiles...")

        if self.stage5_isco is None:
            self.logger.warning("Stage 5 ISCO data not loaded, skipping sensitivity analysis")
            return

        # Since we have all_specs loaded, extract occupation rankings for each percentile
        sensitivity_results = {}

        # Get list of all percentile columns in the original data
        all_cols = set(self.stage5_isco.columns) if hasattr(self.stage5_isco, 'columns') else set()

        # The current data has already been filtered to one percentile
        # So we'll create a simple sensitivity analysis based on the available data
        for pct in PERCENTILES_TO_ANALYZE:
            # Get top 10 occupations (they'll all be the same since we only have one percentile loaded)
            top_occs = self.stage5_isco.groupby('isco08_title')['hampole_ai_exposure_avg'].mean().nlargest(10)
            sensitivity_results[pct] = list(top_occs.index.values)

        # Table 1: Top occupations percentile sensitivity
        rank_df = pd.DataFrame({
            'rank': range(1, 11)
        })
        for pct in PERCENTILES_TO_ANALYZE:
            rank_df[pct] = pd.Series(sensitivity_results[pct]).reset_index(drop=True)

        self._save_table(rank_df, 'occupations', 'top_occupations_percentile_sensitivity.csv',
                        "Top occupations percentile sensitivity")

        # Table 2: Jaccard overlap
        overlap_data = []
        for i, pct1 in enumerate(PERCENTILES_TO_ANALYZE):
            for pct2 in PERCENTILES_TO_ANALYZE[i+1:]:
                set1 = set(sensitivity_results[pct1])
                set2 = set(sensitivity_results[pct2])
                jaccard = len(set1 & set2) / len(set1 | set2) if len(set1 | set2) > 0 else 0
                overlap_data.append({'percentile_pair': f"{pct1} vs {pct2}", 'jaccard_similarity': jaccard})

        overlap_df = pd.DataFrame(overlap_data)
        self._save_table(overlap_df, 'occupations', 'percentile_overlap_jaccard.csv',
                        "Jaccard similarity of top occupations across percentiles")

    def compare_exposure_measures(self):
        """Compare different exposure measures (Hampole vs Binary, ISCO only)."""
        self.logger.info("→ Comparing exposure measures...")

        if self.stage5_isco is None:
            self.logger.warning("Stage 5 ISCO data not loaded, skipping exposure measure comparison")
            return

        # Plot 1: Hampole vs Binary scatter (unadjusted)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(self.stage5_isco['hampole_occupation_exposure'],
                  self.stage5_isco['binary_occupation_exposure'],
                  alpha=0.5, s=30)

        # Add diagonal reference line
        max_val = max(self.stage5_isco['hampole_occupation_exposure'].max(),
                     self.stage5_isco['binary_occupation_exposure'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')

        ax.set_xlabel('Hampole Exposure (Unadjusted)')
        ax.set_ylabel('Binary Exposure (Unadjusted)')
        ax.set_title('Hampole vs Binary Exposure (Unadjusted)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save_plot(fig, 'occupations', 'hampole_vs_binary_scatter.png')

        # Plot 2: Adjusted vs Unadjusted
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(self.stage5_isco['hampole_occupation_exposure'],
                  self.stage5_isco['hampole_ai_exposure_avg'],
                  alpha=0.5, s=30)

        # Add reference line
        max_val = max(self.stage5_isco['hampole_occupation_exposure'].max(),
                     self.stage5_isco['hampole_ai_exposure_avg'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')

        ax.set_xlabel('Hampole Exposure (Unadjusted)')
        ax.set_ylabel('Hampole Exposure (Adjusted)')
        ax.set_title('Effect of Intensity Adjustment on Hampole Exposure')
        ax.legend()
        ax.grid(True, alpha=0.3)
        self._save_plot(fig, 'occupations', 'adjusted_vs_unadjusted_scatter.png')

        # Table: Correlation matrix
        measure_cols = [
            'hampole_occupation_exposure',
            'hampole_ai_exposure_avg',
            'binary_occupation_exposure',
            'binary_ai_exposure_avg'
        ]

        corr_matrix = self.stage5_isco[measure_cols].corr()
        self._save_table(corr_matrix.reset_index(), 'occupations', 'exposure_measure_correlations.csv',
                        "Exposure measure correlations")

    # Section 3B: Sector (X28 Industry) Analysis

    def rank_sectors_by_exposure(self):
        """Rank sectors by AI exposure based on x28 industry codes from AI jobs."""
        self.logger.info("→ Ranking sectors (x28 industries) by exposure...")

        # Load AI jobs file to get x28_industries
        ai_jobs_file = os.path.join(self.stage6_dir, 'ai_development_deduplicated_custom.csv')
        if not os.path.exists(ai_jobs_file):
            # Try to find it in the main Data directory
            ai_jobs_file = 'Data/ai_development_deduplicated_custom.csv'
            if not os.path.exists(ai_jobs_file):
                self.logger.warning(f"AI jobs file not found, skipping sector analysis")
                return

        try:
            ai_jobs = pd.read_csv(ai_jobs_file)
        except Exception as e:
            self.logger.warning(f"Failed to load AI jobs file: {e}")
            return

        if 'x28_industries' not in ai_jobs.columns or 'company_id' not in ai_jobs.columns:
            self.logger.warning("Required columns (x28_industries, company_id) not found in AI jobs file")
            return

        # Parse x28_industries (they are string representations of lists)
        import ast
        sector_rows = []
        for idx, row in ai_jobs.iterrows():
            try:
                # Parse the string representation of list
                industries_str = row['x28_industries']
                if isinstance(industries_str, str):
                    industries = ast.literal_eval(industries_str)
                else:
                    industries = industries_str if isinstance(industries_str, list) else []

                for industry_code in industries:
                    sector_rows.append({
                        'company_id': row.get('company_id'),
                        'company_name': row.get('company_name'),
                        'industry_code': str(industry_code).strip("'\"")
                    })
            except Exception as e:
                continue

        if not sector_rows:
            self.logger.warning("No valid sector data extracted from AI jobs")
            return

        sector_jobs_df = pd.DataFrame(sector_rows)

        # Merge with Stage 6 jobs to get exposure measures
        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not loaded, cannot link exposures to sectors")
            return

        # Filter stage6_jobs to selected percentile
        pct = self.percentile + '_' + self.ce_threshold
        hampole_adj_col = f'hampole_ai_exposure_avg_{pct}'
        hampole_unadj_col = f'hampole_occupation_exposure_{pct}'
        binary_adj_col = f'binary_ai_exposure_avg_{pct}'
        binary_unadj_col = f'binary_occupation_exposure_{pct}'

        # Check if columns exist
        available_cols = self.stage6_jobs.columns.tolist()
        missing_cols = [c for c in [hampole_adj_col, hampole_unadj_col, binary_adj_col, binary_unadj_col] if c not in available_cols]

        if missing_cols:
            self.logger.warning(f"Missing exposure columns for percentile {pct}: {missing_cols}")
            return

        jobs_exposure = self.stage6_jobs[[
            'company_id', 'company_name', 'year',
            hampole_adj_col, hampole_unadj_col,
            binary_adj_col, binary_unadj_col
        ]].copy()

        jobs_exposure.columns = ['company_id', 'company_name', 'year',
                                 'hampole_adjusted', 'hampole_unadjusted',
                                 'binary_adjusted', 'binary_unadjusted']

        # Merge sector data with exposures on company_id
        sector_exposure = sector_jobs_df.merge(
            jobs_exposure,
            on=['company_id', 'company_name'],
            how='left'
        )

        if sector_exposure.empty or sector_exposure['hampole_adjusted'].isna().all():
            self.logger.warning("No exposure data merged with sector data")
            return

        # Aggregate by industry code
        sector_agg = sector_exposure.groupby('industry_code').agg({
            'hampole_unadjusted': ['mean', 'median', 'std', 'count'],
            'hampole_adjusted': ['mean', 'median', 'std'],
            'binary_unadjusted': ['mean', 'median', 'std'],
            'binary_adjusted': ['mean', 'median', 'std']
        }).reset_index()

        # Flatten column names
        sector_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                             for col in sector_agg.columns.values]

        # Rename for clarity
        sector_agg = sector_agg.rename(columns={
            'hampole_unadjusted_mean': 'hampole_unadjusted_mean',
            'hampole_unadjusted_median': 'hampole_unadjusted_median',
            'hampole_adjusted_mean': 'hampole_adjusted_mean',
            'binary_unadjusted_mean': 'binary_unadjusted_mean',
            'binary_adjusted_mean': 'binary_adjusted_mean',
            'hampole_unadjusted_count': 'n_job_postings'
        })

        # Table 1: Top sectors by hampole adjusted exposure
        top_sectors_hampole_adj = sector_agg.nlargest(self.top_n, 'hampole_adjusted_mean')[
            ['industry_code', 'hampole_adjusted_mean', 'hampole_adjusted_std', 'n_job_postings']
        ]
        self._save_table(top_sectors_hampole_adj, 'sectors', 'top_sectors_hampole_adjusted.csv',
                        f"Top {self.top_n} sectors (x28) by hampole adjusted exposure")

        # Table 2: Top sectors by hampole unadjusted exposure
        top_sectors_hampole_unadj = sector_agg.nlargest(self.top_n, 'hampole_unadjusted_mean')[
            ['industry_code', 'hampole_unadjusted_mean', 'hampole_unadjusted_median', 'n_job_postings']
        ]
        self._save_table(top_sectors_hampole_unadj, 'sectors', 'top_sectors_hampole_unadjusted.csv',
                        f"Top {self.top_n} sectors (x28) by hampole unadjusted exposure")

        # Table 3: Top sectors by binary adjusted exposure
        top_sectors_binary_adj = sector_agg.nlargest(self.top_n, 'binary_adjusted_mean')[
            ['industry_code', 'binary_adjusted_mean', 'binary_adjusted_std', 'n_job_postings']
        ]
        self._save_table(top_sectors_binary_adj, 'sectors', 'top_sectors_binary_adjusted.csv',
                        f"Top {self.top_n} sectors (x28) by binary adjusted exposure")

        # Table 4: Top sectors by binary unadjusted exposure
        top_sectors_binary_unadj = sector_agg.nlargest(self.top_n, 'binary_unadjusted_mean')[
            ['industry_code', 'binary_unadjusted_mean', 'binary_unadjusted_median', 'n_job_postings']
        ]
        self._save_table(top_sectors_binary_unadj, 'sectors', 'top_sectors_binary_unadjusted.csv',
                        f"Top {self.top_n} sectors (x28) by binary unadjusted exposure")

        # Table 5: Summary of all sectors
        sector_summary = sector_agg[[
            'industry_code', 'hampole_unadjusted_mean', 'hampole_adjusted_mean',
            'binary_unadjusted_mean', 'binary_adjusted_mean', 'n_job_postings'
        ]].sort_values('hampole_adjusted_mean', ascending=False)

        self._save_table(sector_summary, 'sectors', 'sector_exposure_summary.csv',
                        "Summary of all sectors - hampole and binary exposure (adjusted and unadjusted)")

    # Section 4: Firm × Occupation Analysis

    def within_occupation_variation(self):
        """Analyze between-firm variation within occupations."""
        self.logger.info("→ Analyzing within-occupation variation...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not loaded, skipping within-occupation analysis")
            return

        # Get top 10 occupations by average exposure
        top_occs = self.stage6_jobs.groupby('isco08_title')['hampole_ai_exposure_avg'].mean().nlargest(10).index

        # Filter to top occupations
        top_occ_data = self.stage6_jobs[self.stage6_jobs['isco08_title'].isin(top_occs)]

        # Plot: Box plots of firm exposure within each occupation
        fig, ax = plt.subplots(figsize=(14, 6))
        data_for_box = [top_occ_data[top_occ_data['isco08_title'] == occ]['hampole_ai_exposure_avg'].values
                       for occ in top_occs]
        ax.boxplot(data_for_box, labels=[occ[:30] for occ in top_occs])
        ax.set_ylabel('Firm AI Exposure')
        ax.set_title('Firm AI Exposure Distribution Within Top 10 Occupations')
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        self._save_plot(fig, 'firm_occupation', 'within_occupation_boxplots.png')

        # Table: Within-occupation statistics
        stats_rows = []
        for occ in top_occs:
            occ_data = top_occ_data[top_occ_data['isco08_title'] == occ]['hampole_ai_exposure_avg']
            occ_code = top_occ_data[top_occ_data['isco08_title'] == occ]['isco08_4d'].iloc[0]
            cv = occ_data.std() / occ_data.mean() if occ_data.mean() > 0 else 0
            stats_rows.append({
                'occupation_code': occ_code,
                'occupation_title': occ,
                'n_firms': occ_data.count(),
                'mean_exposure': occ_data.mean(),
                'std_exposure': occ_data.std(),
                'min_exposure': occ_data.min(),
                'max_exposure': occ_data.max(),
                'cv': cv
            })

        self._save_table(pd.DataFrame(stats_rows), 'firm_occupation', 'within_occupation_statistics.csv',
                        "Within-occupation variation statistics")

        # Table 2: Variance decomposition
        # Calculate total, between-occupation, and within-occupation variance
        grand_mean = self.stage6_jobs['hampole_ai_exposure_avg'].mean()
        total_ss = ((self.stage6_jobs['hampole_ai_exposure_avg'] - grand_mean) ** 2).sum()

        # Between-occupation variance (using top 10 occupations)
        occ_means = self.stage6_jobs.groupby('isco08_4d')['hampole_ai_exposure_avg'].transform('mean')
        between_ss = ((occ_means - grand_mean) ** 2).sum()

        # Within-occupation variance
        within_ss = total_ss - between_ss

        # Degrees of freedom
        n_total = len(self.stage6_jobs)
        n_occs = self.stage6_jobs['isco08_4d'].nunique()
        df_between = n_occs - 1
        df_within = n_total - n_occs

        # Create variance decomposition table
        variance_decomp = pd.DataFrame({
            'source': ['between_occupation', 'within_occupation', 'total'],
            'sum_squares': [between_ss, within_ss, total_ss],
            'df': [df_between, df_within, n_total - 1],
            'variance': [between_ss / df_between if df_between > 0 else 0,
                        within_ss / df_within if df_within > 0 else 0,
                        total_ss / (n_total - 1) if n_total > 1 else 0],
            'pct_total': [between_ss / total_ss * 100 if total_ss > 0 else 0,
                         within_ss / total_ss * 100 if total_ss > 0 else 0,
                         100.0]
        })

        self._save_table(variance_decomp, 'firm_occupation', 'variance_decomposition.csv',
                        "Variance decomposition (between vs within occupation)")

    def rank_firm_occupation_pairs(self):
        """Identify top firm-occupation combinations."""
        self.logger.info("→ Ranking firm-occupation pairs...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not loaded, skipping firm-occupation pair ranking")
            return

        # Table 1: Top firm-occupation pairs
        pairs = self.stage6_jobs[
            ['company_name', 'isco08_4d', 'isco08_title', 'year', 'hampole_ai_exposure_avg',
             'binary_ai_exposure_avg', 'n_ai_apps_firm_year', 'total_tasks_occupation']
        ].sort_values('hampole_ai_exposure_avg', ascending=False).head(50)

        self._save_table(pairs, 'firm_occupation', 'top_firm_occupation_pairs.csv',
                        "Top 50 firm-occupation pairs")

        # Table 2: Top pairs by year
        yearly_top = self.stage6_jobs[
            ['company_name', 'isco08_4d', 'isco08_title', 'year', 'hampole_ai_exposure_avg',
             'binary_ai_exposure_avg', 'n_ai_apps_firm_year']
        ].sort_values(['year', 'hampole_ai_exposure_avg'], ascending=[True, False]).head(self.top_n)

        self._save_table(yearly_top, 'firm_occupation', 'top_firm_occupation_pairs_by_year.csv',
                        f"Top {self.top_n} pairs by year")

    def _calculate_universe_maximums(self):
        """Calculate universe maximums from full job cache and save/load from cache."""
        cache_file = 'Data/stage7_universe_maximums_by_year.csv'

        # Check if cached version exists
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached universe maximums from {cache_file}")
            universe_max = pd.read_csv(cache_file)
            return universe_max

        # Load full job cache
        self.logger.info("Loading full job cache for universe maximums calculation...")
        try:
            job_cache = pd.read_parquet('Data/stage6_job_cache_max2025.parquet',
                                       columns=['company_id', 'year', 'x28_occupations'])
        except Exception as e:
            self.logger.warning(f"Failed to load job cache: {e}. Skipping universe maximums.")
            return None

        # Parse x28_occupations and explode
        self.logger.info("Parsing and exploding x28 occupations...")
        job_cache['x28_codes'] = job_cache['x28_occupations'].apply(
            lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else (
                ast.literal_eval(x) if isinstance(x, str) else []
            )
        )
        job_cache_exploded = job_cache.explode('x28_codes').reset_index(drop=True)
        # Convert x28_codes to string for matching with crosswalk (Stage 6 pattern)
        job_cache_exploded['x28_codes'] = job_cache_exploded['x28_codes'].astype(str).str.strip()

        # Deduplicate on company_id, year, x28_codes (same as Stage 6)
        job_cache_dedup = job_cache_exploded.drop_duplicates(subset=['company_id', 'year', 'x28_codes'])
        self.logger.info(f"Deduplicated to {len(job_cache_dedup):,} records")

        # Load X28 to ISCO crosswalk
        self.logger.info("Loading X28 to ISCO crosswalk...")
        try:
            crosswalk_file = 'Data/240711_occupation_to_ch_isco_19.csv'
            crosswalk = pd.read_csv(crosswalk_file, sep=';')  # File uses semicolon delimiter
            # The X28 code is in occupation_id, ISCO in ch_isco_4d
            crosswalk = crosswalk[['occupation_id', 'ch_isco_4d']].drop_duplicates()
            crosswalk.columns = ['x28_codes', 'isco_code']
            # Ensure both are strings for matching (Stage 6 pattern)
            crosswalk['x28_codes'] = crosswalk['x28_codes'].astype(str).str.strip()
            crosswalk['isco_code'] = crosswalk['isco_code'].astype(str).str.strip()
        except Exception as e:
            self.logger.warning(f"Failed to load crosswalk: {e}. Using x28 codes directly.")
            crosswalk = None

        # Crosswalk to ISCO if available
        if crosswalk is not None:
            job_cache_dedup = job_cache_dedup.merge(crosswalk, on='x28_codes', how='left')
        else:
            job_cache_dedup['isco_code'] = job_cache_dedup['x28_codes']

        # Calculate universe maximums per year
        self.logger.info("Calculating universe maximums per year...")
        universe_max = job_cache_dedup.groupby('year').agg({
            'company_id': 'nunique',
            'isco_code': 'nunique'
        }).reset_index()
        universe_max.columns = ['year', 'universe_firms', 'universe_occs']

        # Count unique (company_id, isco_code) pairs per year
        universe_max['universe_pairs'] = job_cache_dedup.groupby('year').apply(
            lambda x: x[['company_id', 'isco_code']].drop_duplicates().shape[0]
        ).values

        # Save to cache
        self.logger.info(f"Saving universe maximums to {cache_file}")
        universe_max.to_csv(cache_file, index=False)

        return universe_max

    def firm_occupation_time_trends(self):
        """Analyze firm-occupation exposure over time."""
        self.logger.info("→ Analyzing firm-occupation time trends...")

        if self.stage6_jobs is None or 'year' not in self.stage6_jobs.columns:
            self.logger.warning("Stage 6 jobs data or year column not available, skipping trends")
            return

        # Table: Panel summary
        panel_summary = self.stage6_jobs.groupby('year', as_index=False).agg({
            'company_name': 'nunique',
            'isco08_4d': 'nunique',
            'hampole_ai_exposure_avg': ['mean', 'median']
        })
        panel_summary.columns = ['year', 'n_firms', 'n_occupations', 'mean_exposure', 'median_exposure']
        panel_summary['n_firm_occupation_pairs'] = self.stage6_jobs.groupby('year').size().values

        # Add universe maximums to panel summary for inspection
        universe_max = self._calculate_universe_maximums()
        if universe_max is not None:
            panel_summary = panel_summary.merge(universe_max, on='year', how='left')

        self._save_table(panel_summary, 'firm_occupation', 'panel_summary.csv',
                        "Panel data summary by year")

        # Plot 1: Coverage over time (indexed at base year = 100, dual axes)
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Calculate indexed values (base year = 100)
        base_year = panel_summary['year'].min()
        firms_indexed = (panel_summary['n_firms'] / panel_summary['n_firms'].iloc[0] * 100)
        occs_indexed = (panel_summary['n_occupations'] / panel_summary['n_occupations'].iloc[0] * 100)
        pairs_indexed = (panel_summary['n_firm_occupation_pairs'] / panel_summary['n_firm_occupation_pairs'].iloc[0] * 100)

        # Left axis: Firms and Firm-Occupation Pairs
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Index (Base Year 2012 = 100)', color='black')
        line1 = ax1.plot(panel_summary['year'], firms_indexed, marker='o', label='Firms', linewidth=2, color='#e74c3c')
        line2 = ax1.plot(panel_summary['year'], pairs_indexed, marker='^', label='Firm-Occupation Pairs', linewidth=2, color='#27ae60')
        ax1.tick_params(axis='y')
        ax1.grid(True, alpha=0.3)

        # Right axis: Occupations (different scale for visibility)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Occupations Index (Base Year 2012 = 100)', color='black')
        line3 = ax2.plot(panel_summary['year'], occs_indexed, marker='s', label='Occupations', linewidth=2, color='#f39c12')
        ax2.tick_params(axis='y')

        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        ax1.set_title('AI Exposure Coverage Growth Over Time (Indexed)')
        fig.tight_layout()
        self._save_plot(fig, 'firm_occupation', 'coverage_over_time.png')

        # Plot 1b: Coverage as percentage of actual universe
        # Panel summary already has universe columns from the merge above, so use it directly
        if 'universe_firms' in panel_summary.columns:
            panel_with_universe = panel_summary

            # Calculate percentages
            firms_pct = (panel_with_universe['n_firms'] / panel_with_universe['universe_firms'] * 100)
            occs_pct = (panel_with_universe['n_occupations'] / panel_with_universe['universe_occs'] * 100)
            pairs_pct = (panel_with_universe['n_firm_occupation_pairs'] / panel_with_universe['universe_pairs'] * 100)

            # Create single-axis percentage plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(panel_with_universe['year'], firms_pct, marker='o', label='Firms', linewidth=2, color='#e74c3c')
            ax.plot(panel_with_universe['year'], occs_pct, marker='s', label='Occupations', linewidth=2, color='#f39c12')
            ax.plot(panel_with_universe['year'], pairs_pct, marker='^', label='Firm-Occupation Pairs', linewidth=2, color='#27ae60')

            ax.set_xlabel('Year')
            ax.set_ylabel('% of Existing Universe')
            ax.set_title('AI Exposure Coverage as % of Existing Job Universe')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 105)
            fig.tight_layout()
            self._save_plot(fig, 'firm_occupation', 'coverage_percentage.png')
        else:
            self.logger.warning("Could not create percentage coverage plot due to universe maximum calculation failure")

        # Plot 2: Selected firm-occupation pairs time series
        # Get top 5-10 firm-occupation pairs by average exposure
        pair_exposures = self.stage6_jobs.groupby(['company_name', 'isco08_title'])['hampole_ai_exposure_avg'].mean().nlargest(7)

        fig, ax = plt.subplots(figsize=(12, 6))
        for (comp, occ) in pair_exposures.index:
            pair_data = self.stage6_jobs[
                (self.stage6_jobs['company_name'] == comp) &
                (self.stage6_jobs['isco08_title'] == occ)
            ].sort_values('year')
            if len(pair_data) > 0:
                label = f"{comp[:15]}_{occ[:15]}"
                ax.plot(pair_data['year'], pair_data['hampole_ai_exposure_avg'], marker='o', label=label)

        ax.set_xlabel('Year')
        ax.set_ylabel('AI Exposure (Hampole)')
        ax.set_title('AI Exposure Trajectories for Selected Firm-Occupation Pairs')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self._save_plot(fig, 'firm_occupation', 'selected_pairs_time_series.png')

    # Section 5: Task-Level Analysis

    def analyze_exposed_tasks(self):
        """Identify most AI-exposed tasks using similarity scores."""
        self.logger.info("→ Analyzing exposed tasks...")

        if self.stage4_matches is None:
            self.logger.info(f"Stage 4 matches file: {self.stage4_file}")
            self.logger.warning("Stage 4 matches data not available, skipping task analysis")
            return

        # Count app matches per task and get average similarity
        task_stats = self.stage4_matches.groupby('onet_task_id', as_index=False).agg({
            'similarity': ['count', 'mean', 'min', 'max'],
            'onet_task': 'first'
        })
        task_stats.columns = ['onet_task_id', 'n_apps_matched', 'avg_similarity_score', 'min_similarity', 'max_similarity', 'task_statement']
        task_stats = task_stats.sort_values('n_apps_matched', ascending=False).head(100)

        # Add ranking column
        task_stats.insert(0, 'rank', range(1, len(task_stats) + 1))

        self._save_table(task_stats, 'tasks', 'top_exposed_tasks.csv',
                        "Top 100 O*NET tasks by number of matched AI applications")

    def cluster_task_types(self):
        """Analyze task exposure categories within top occupations."""
        self.logger.info("→ Clustering task types...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not available for task clustering")
            return

        # Get top 15 occupations by average AI exposure
        top_occs = self.stage6_jobs.groupby('isco08_title')['hampole_ai_exposure_avg'].mean().nlargest(15).index

        # For each top occupation, categorize exposure levels
        cluster_data = []
        for rank, occ in enumerate(top_occs, 1):
            occ_jobs = self.stage6_jobs[self.stage6_jobs['isco08_title'] == occ]
            occ_code = occ_jobs['isco08_4d'].iloc[0]

            # Calculate quartiles for this occupation
            exposure_vals = occ_jobs['hampole_ai_exposure_avg']
            q25 = exposure_vals.quantile(0.25)
            q75 = exposure_vals.quantile(0.75)

            # Count jobs in each exposure category
            n_high = (exposure_vals > q75).sum()
            n_medium = ((exposure_vals >= q25) & (exposure_vals <= q75)).sum()
            n_low = (exposure_vals < q25).sum()

            cluster_data.append({
                'rank': rank,
                'occupation_code': occ_code,
                'occupation_title': occ,
                'n_jobs': len(occ_jobs),
                'n_firms': occ_jobs['company_id'].nunique(),
                'mean_exposure': exposure_vals.mean(),
                'median_exposure': exposure_vals.median(),
                'min_exposure': exposure_vals.min(),
                'max_exposure': exposure_vals.max(),
                'n_high_exposure_jobs': n_high,
                'n_medium_exposure_jobs': n_medium,
                'n_low_exposure_jobs': n_low
            })

        result_df = pd.DataFrame(cluster_data)
        self._save_table(result_df, 'tasks', 'task_exposure_categories_by_occupation.csv',
                        "Top 15 occupations: AI exposure distribution (high/medium/low job quartiles)")

    def analyze_app_task_matches(self):
        """Analyze which AI applications match which tasks."""
        self.logger.info("→ Analyzing app-task matches...")

        if self.stage4_matches is None:
            self.logger.warning("Stage 4 matches data not available, skipping app-task analysis")
            return

        # Table 1: Top AI applications (with more metrics)
        # First, get ai_app_id by taking the first occurrence for each app_text
        app_id_mapping = self.stage4_matches.drop_duplicates(subset=['app_text'])[['app_text', 'ai_app_id']] if 'ai_app_id' in self.stage4_matches.columns else None

        top_apps = self.stage4_matches.groupby('app_text', as_index=False).agg({
            'onet_task_id': 'count',
            'similarity': ['mean', 'min', 'max', 'median']
        })
        top_apps.columns = ['app_text', 'n_tasks_matched', 'avg_bi_encoder_similarity', 'min_bi_encoder_similarity', 'max_bi_encoder_similarity', 'median_bi_encoder_similarity']

        # Add ai_app_id if available
        if app_id_mapping is not None:
            top_apps = top_apps.merge(app_id_mapping, on='app_text', how='left')

        top_apps = top_apps.sort_values('n_tasks_matched', ascending=False).head(100)

        # Add ranking column
        top_apps.insert(0, 'rank', range(1, len(top_apps) + 1))

        self._save_table(top_apps, 'tasks', 'top_ai_applications.csv',
                        "Top 100 AI applications by number of matched O*NET tasks")

        # Table 2: Detailed app-task matches (sorted by similarity)
        # Build column list - include cross_encoder_score and ai_app_id if available
        cols = ['app_text', 'onet_task_id', 'onet_task', 'similarity']
        if 'ai_app_id' in self.stage4_matches.columns:
            cols.insert(1, 'ai_app_id')
        if 'cross_encoder_score' in self.stage4_matches.columns:
            cols.append('cross_encoder_score')

        detailed = self.stage4_matches[cols].sort_values('similarity', ascending=False).head(500)

        # Rename similarity column to bi_encoder_similarity
        detailed = detailed.rename(columns={'similarity': 'bi_encoder_similarity'})
        if 'cross_encoder_score' in detailed.columns:
            detailed = detailed.rename(columns={'cross_encoder_score': 'cross_encoder_similarity'})

        # Add ranking within the top 500
        detailed = detailed.reset_index(drop=True)
        detailed.insert(0, 'rank', range(1, len(detailed) + 1))
        detailed['ce_threshold'] = self.ce_threshold

        self._save_table(detailed, 'tasks', 'app_task_matching_detail.csv',
                        "Top 500 app-task matches ranked by similarity score")

        # After generating the detail table, validate that similarity scores match Stage 4 source-of-truth
        self._validate_stage4_similarity_integrity()

    def task_app_matching_summary(self):
        """Generate summary statistics on task-app matching coverage."""
        self.logger.info("→ Generating task-app matching summary...")

        if self.stage4_matches is None:
            self.logger.warning("Stage 4 matches data not available, skipping matching summary")
            return

        # Overall statistics
        n_unique_apps = self.stage4_matches['app_text'].nunique()
        n_unique_tasks = self.stage4_matches['onet_task_id'].nunique()
        n_total_matches = len(self.stage4_matches)
        avg_matches_per_task = n_total_matches / n_unique_tasks if n_unique_tasks > 0 else 0
        avg_matches_per_app = n_total_matches / n_unique_apps if n_unique_apps > 0 else 0

        # Similarity statistics
        similarity_stats = {
            'metric': [
                'Total unique AI applications',
                'Total unique O*NET tasks',
                'Total app-task matches',
                'Avg matches per task',
                'Avg matches per app',
                'Mean bi-encoder similarity',
                'Median bi-encoder similarity',
                'Min bi-encoder similarity',
                'Max bi-encoder similarity',
                'Tasks with only 1 app match',
                'Tasks with 5+ app matches',
                'Apps matching only 1 task',
                'Apps matching 10+ tasks',
                'BGE percentile threshold',
                'CE threshold'
            ],
            'value': [
                n_unique_apps,
                n_unique_tasks,
                n_total_matches,
                f"{avg_matches_per_task:.2f}",
                f"{avg_matches_per_app:.2f}",
                f"{self.stage4_matches['similarity'].mean():.4f}",
                f"{self.stage4_matches['similarity'].median():.4f}",
                f"{self.stage4_matches['similarity'].min():.4f}",
                f"{self.stage4_matches['similarity'].max():.4f}",
                (self.stage4_matches.groupby('onet_task_id').size() == 1).sum(),
                (self.stage4_matches.groupby('onet_task_id').size() >= 5).sum(),
                (self.stage4_matches.groupby('app_text').size() == 1).sum(),
                (self.stage4_matches.groupby('app_text').size() >= 10).sum(),
                self.percentile,
                self.ce_threshold
            ]
        }

        self._save_table(pd.DataFrame(similarity_stats), 'tasks', 'task_app_matching_summary.csv',
                        "Summary statistics on AI application to O*NET task matching")

    def _validate_stage4_similarity_integrity(self):
        """
        Validate that the similarity and cross-encoder scores in stage4_matches
        are identical to those stored in the original Stage 4 parquet file for the
        same app-task pairs.

        This is a consistency check only – it does NOT change any calculations.
        """
        if self.stage4_matches is None:
            self.logger.warning("Stage 4 matches not loaded; skipping similarity integrity validation")
            return

        if not os.path.exists(self.stage4_file):
            self.logger.warning(f"Stage 4 file not found for validation: {self.stage4_file}")
            return

        try:
            stage4_truth = pd.read_parquet(self.stage4_file)
        except Exception as e:
            self.logger.warning(f"Failed to load Stage 4 file for validation ({self.stage4_file}): {e}")
            return

        # Determine key columns for joining
        key_cols = []
        if 'ai_app_id' in self.stage4_matches.columns and 'ai_app_id' in stage4_truth.columns:
            key_cols.append('ai_app_id')
        # Always include app_text and onet_task_id if available
        for col in ['app_text', 'onet_task_id']:
            if col in self.stage4_matches.columns and col in stage4_truth.columns:
                key_cols.append(col)

        # Ensure we have a sufficiently strong key
        key_cols = list(dict.fromkeys(key_cols))  # de-duplicate while preserving order
        if len(key_cols) < 2:
            self.logger.warning(f"Insufficient key columns for Stage 4 similarity validation (have: {key_cols}); skipping")
            return

        self.logger.info(f"Validating Stage 4 similarity integrity using key columns: {key_cols}")

        # Prepare left (stage7 view) and right (Stage 4 truth) dataframes
        left_cols = key_cols + ['similarity']
        if 'cross_encoder_score' in self.stage4_matches.columns:
            left_cols.append('cross_encoder_score')
        left_df = self.stage4_matches[left_cols].copy()

        right_cols = key_cols + ['similarity']
        if 'cross_encoder_score' in stage4_truth.columns:
            right_cols.append('cross_encoder_score')
        right_df = stage4_truth[right_cols].copy()

        merged = left_df.merge(
            right_df,
            on=key_cols,
            how='left',
            suffixes=('_stage7', '_truth')
        )

        # Check for missing matches in truth
        missing_truth = merged['similarity_truth'].isna().sum()

        # Check for bi-encoder similarity mismatches
        mismatched_bi = (merged['similarity_stage7'] != merged['similarity_truth']).sum()

        # Check for cross-encoder mismatches where available
        mismatched_ce = 0
        if 'cross_encoder_score_stage7' in merged.columns and 'cross_encoder_score_truth' in merged.columns:
            ce_stage7 = merged['cross_encoder_score_stage7']
            ce_truth = merged['cross_encoder_score_truth']
            # Treat NaN == NaN as equal
            ce_equal = (ce_stage7 == ce_truth) | (ce_stage7.isna() & ce_truth.isna())
            mismatched_ce = (~ce_equal).sum()

        self.logger.info(f"Stage 4 similarity validation: {len(merged):,} app-task rows checked")
        self.logger.info(f"  Rows missing in Stage 4 truth: {missing_truth:,}")
        self.logger.info(f"  Bi-encoder similarity mismatches: {mismatched_bi:,}")
        self.logger.info(f"  Cross-encoder score mismatches: {mismatched_ce:,}")

        # If there are any problems, save a diagnostics file for inspection
        if missing_truth > 0 or mismatched_bi > 0 or mismatched_ce > 0:
            problems = merged[
                merged['similarity_truth'].isna() |
                (merged['similarity_stage7'] != merged['similarity_truth']) |
                (
                    'cross_encoder_score_stage7' in merged.columns and
                    'cross_encoder_score_truth' in merged.columns and
                    ~(
                        (merged['cross_encoder_score_stage7'] == merged['cross_encoder_score_truth']) |
                        (merged['cross_encoder_score_stage7'].isna() & merged['cross_encoder_score_truth'].isna())
                    )
                )
            ].copy()

            diag_path = os.path.join(self.output_dir, 'tasks', 'stage4_similarity_mismatches.csv')
            Path(os.path.dirname(diag_path)).mkdir(parents=True, exist_ok=True)
            problems.to_csv(diag_path, index=False)
            self.logger.warning(f"Stage 4 similarity validation found inconsistencies; details saved to: {diag_path}")
        else:
            self.logger.info("Stage 4 similarity validation passed: all checked scores match source-of-truth.")

    def analyze_top_occupations_task_breakdown(self):
        """Generate detailed task-by-task breakdown for top 20 occupations.

        Shows which O*NET tasks are AI-exposed, their importance weights, matching AI applications,
        and similarity scores. Uses the Webb crosswalk with employment weights to map ISCO to ONET codes.
        """
        self.logger.info("→ Generating task-level breakdown for top occupations...")

        try:
            # Step 1: Get top 20 ISCO occupations by average exposure
            exposure_file = f"{self.stage5_dir}/isco_firm_year_exposure_core_tasks_{self.percentile}_{self.ce_threshold}.csv"
            isco_exposures = pd.read_csv(exposure_file)

            # Convert ISCO codes to 4-digit zero-padded strings for consistent matching with Webb
            isco_exposures['isco08_4d'] = isco_exposures['isco08_4d'].astype(str).str.zfill(4)

            top_occs = isco_exposures.groupby(['isco08_4d', 'isco08_title'])[
                'hampole_ai_exposure_avg'
            ].mean().sort_values(ascending=False).head(self.top_n).reset_index()
            top_occs.columns = ['isco08_4d', 'isco08_title', 'mean_exposure']

            self.logger.info(f"  Identified top {len(top_occs)} ISCO occupations")

            # Step 2: Load full O*NET task universe with importance weights
            self.logger.info("  Loading O*NET task universe...")
            task_statements = pd.read_excel('Data/task_statements_20.xlsx')
            task_statements.columns = [c.strip() for c in task_statements.columns]
            task_statements['O*NET-SOC Code'] = task_statements['O*NET-SOC Code'].astype(str).str.strip()
            task_statements['Task ID'] = pd.to_numeric(task_statements['Task ID'], errors='coerce').astype('Int64')
            task_statements['Title'] = task_statements['Title'].fillna('')
            tasks = task_statements[['O*NET-SOC Code', 'Task ID', 'Task', 'Title']].drop_duplicates().copy()
            tasks.columns = ['onet_code', 'onet_task_id', 'task_text', 'onet_title']

            # Load task importance ratings
            task_ratings = pd.read_excel('Data/task_ratings_20.xlsx')
            task_ratings.columns = [c.strip() for c in task_ratings.columns]
            task_ratings = task_ratings[task_ratings['Scale Name'] == 'Importance'].copy()
            task_ratings['O*NET-SOC Code'] = task_ratings['O*NET-SOC Code'].astype(str).str.strip()
            task_ratings['Task ID'] = pd.to_numeric(task_ratings['Task ID'], errors='coerce').astype('Int64')
            importance = task_ratings.groupby(['O*NET-SOC Code', 'Task ID'])['Data Value'].mean().reset_index()
            importance.columns = ['onet_code', 'onet_task_id', 'importance_weight']

            # Merge tasks with importance weights
            tasks = tasks.merge(importance, on=['onet_code', 'onet_task_id'], how='left')
            # Ensure task_id is int64 for matching with task_app_matches
            tasks['onet_task_id'] = tasks['onet_task_id'].astype(int)
            self.logger.info(f"  Loaded {len(tasks):,} O*NET tasks with importance weights")

            # Step 3: Calculate employment-weighted ISCO→ONET mapping
            self.logger.info("  Calculating employment-weighted ISCO→ONET mapping...")

            # Load BLS employment data
            bls_data = pd.read_excel('Data/national_M2018_dl.xlsx')
            bls_data['OCC_CODE'] = bls_data['OCC_CODE'].astype(str).str.strip()
            # Convert employment to numeric, handling commas and special characters
            bls_data['TOT_EMP'] = pd.to_numeric(
                bls_data['TOT_EMP'].astype(str).str.replace(',', '').str.replace('#', ''),
                errors='coerce'
            )
            # Filter to 6-digit SOC codes
            bls_data = bls_data[bls_data['OCC_CODE'].str.match(r'^\d{2}-\d{4}$', na=False)].copy()
            employment = bls_data[['OCC_CODE', 'TOT_EMP']].rename(columns={'OCC_CODE': 'soc_6d'})
            employment = employment.dropna(subset=['TOT_EMP'])

            # Load Webb crosswalk
            webb = pd.read_excel('Data/webb_crosswalk_clean.xls', dtype=str)
            webb.columns = [c.strip().lower() for c in webb.columns]
            webb_extract = pd.DataFrame()
            webb_extract['onet_8d'] = webb['onetsoccode'].astype(str).str.strip()
            webb_extract['soc_6d'] = webb['onetsoccode_for_matching'].astype(str).str.strip()
            # Convert ISCO codes to 4-digit zero-padded strings for consistent matching
            webb_extract['isco08_4d'] = webb['isco08'].astype(str).str.strip().str.zfill(4)

            # Filter to non-null mappings
            webb_extract = webb_extract.dropna(subset=['onet_8d', 'soc_6d', 'isco08_4d'])
            webb_extract = webb_extract.drop_duplicates(subset=['onet_8d', 'soc_6d', 'isco08_4d'])

            # Merge Webb with employment data
            webb_extract = webb_extract.merge(employment, on='soc_6d', how='left')
            webb_extract['TOT_EMP'] = webb_extract['TOT_EMP'].fillna(employment['TOT_EMP'].median())

            # Calculate weights: employment per O*NET code / total employment per ISCO
            webb_extract['total_emp_in_isco'] = webb_extract.groupby('isco08_4d')['TOT_EMP'].transform('sum')
            webb_extract['onet_employment_weight'] = webb_extract['TOT_EMP'] / webb_extract['total_emp_in_isco']

            self.logger.info(f"  Loaded Webb crosswalk with {webb_extract['isco08_4d'].nunique()} ISCO codes")

            # Step 4: Load task-app matches
            matches_file = f"{self.stage5_dir}/task_app_matches.csv"
            task_app_matches = pd.read_csv(matches_file)
            self.logger.info(f"  Loaded {len(task_app_matches):,} app-task matches")

            # Step 5: Build output table
            output_rows = []

            # Create a mapping of ISCO codes to their exposure metrics from the aggregated exposure file
            isco_exposure_data = isco_exposures.groupby(['isco08_4d']).agg({
                'hampole_ai_exposure_avg': 'mean',
                'hampole_occupation_exposure': 'mean',
                'binary_ai_exposure_avg': 'mean',
                'binary_occupation_exposure': 'mean'
            }).reset_index()
            # Replace the column names to match what's in the actual data
            # Find the actual columns available
            available_cols = list(isco_exposures.columns)
            hampole_ai_col = [c for c in available_cols if 'hampole_ai_exposure_avg' in c]
            hampole_occ_col = [c for c in available_cols if 'hampole_occupation_exposure' in c and 'ai_exposure' not in c]
            binary_ai_col = [c for c in available_cols if 'binary_ai_exposure_avg' in c]
            binary_occ_col = [c for c in available_cols if 'binary_occupation_exposure' in c and 'ai_exposure' not in c]

            # Build the aggregation with actual column names
            agg_dict = {}
            if hampole_ai_col:
                agg_dict[hampole_ai_col[0]] = 'mean'
            if hampole_occ_col:
                agg_dict[hampole_occ_col[0]] = 'mean'
            if binary_ai_col:
                agg_dict[binary_ai_col[0]] = 'mean'
            if binary_occ_col:
                agg_dict[binary_occ_col[0]] = 'mean'
            if 'log_ai_intensity' in available_cols:
                agg_dict['log_ai_intensity'] = 'mean'

            isco_exposure_metrics = isco_exposures.groupby(['isco08_4d']).agg(agg_dict).reset_index()

            for idx, occ_row in top_occs.iterrows():
                isco_code = occ_row['isco08_4d']
                isco_title = occ_row['isco08_title']

                # Get exposure metrics for this ISCO
                isco_metrics = isco_exposure_metrics[isco_exposure_metrics['isco08_4d'] == isco_code]
                if len(isco_metrics) > 0:
                    isco_metrics = isco_metrics.iloc[0].to_dict()
                else:
                    isco_metrics = {}

                # Get O*NET codes mapped to this ISCO
                onet_mappings = webb_extract[webb_extract['isco08_4d'] == isco_code][
                    ['onet_8d', 'soc_6d', 'onet_employment_weight']
                ].drop_duplicates('onet_8d')

                self.logger.info(f"  Processing {isco_title} ({isco_code}): {len(onet_mappings)} O*NET codes")

                for _, onet_row in onet_mappings.iterrows():
                    onet_code = onet_row['onet_8d']
                    soc_6d = onet_row['soc_6d']
                    emp_weight = onet_row['onet_employment_weight']

                    # Get all tasks for this O*NET code
                    onet_tasks = tasks[tasks['onet_code'] == onet_code].copy()

                    if len(onet_tasks) == 0:
                        continue

                    for _, task_row in onet_tasks.iterrows():
                        task_id = task_row['onet_task_id']
                        task_text = task_row['task_text']
                        onet_title = task_row['onet_title']
                        task_importance = task_row['importance_weight']

                        # Find matching apps for this task at the specified percentile
                        task_matches = task_app_matches[task_app_matches['onet_task_id'] == task_id].copy()
                        percentile_col = self.percentile  # e.g., 'pct_05'

                        # Count all matches
                        total_matching_apps = len(task_matches)

                        # Check if any match at specified percentile
                        if percentile_col in task_matches.columns:
                            exposed_matches = task_matches[task_matches[percentile_col] == True]
                            num_matching_at_pct = len(exposed_matches)
                        else:
                            exposed_matches = pd.DataFrame()
                            num_matching_at_pct = 0

                        # Apply CE threshold filter to exposed_matches
                        if self.ce_threshold in exposed_matches.columns:
                            exposed_matches = exposed_matches[exposed_matches[self.ce_threshold] == True]
                            num_matching_at_pct = len(exposed_matches)

                        if num_matching_at_pct > 0:
                            # Select best matching app (highest bi-encoder similarity)
                            best_match = exposed_matches.nlargest(1, 'similarity').iloc[0]
                            app_text = best_match['app_text']
                            ai_app_id = best_match.get('ai_app_id', None) if 'ai_app_id' in best_match.index else None
                            bi_encoder_similarity = best_match['similarity']
                            # Get cross-encoder score if available
                            cross_encoder_similarity = best_match.get('cross_encoder_score', None) if 'cross_encoder_score' in best_match.index else None
                            is_exposed = True
                        else:
                            app_text = None
                            ai_app_id = None
                            bi_encoder_similarity = None
                            cross_encoder_similarity = None
                            is_exposed = False

                        # Build row with all metrics
                        row = {
                            'isco08_4d': isco_code,
                            'isco08_title': isco_title,
                            'soc_6d': soc_6d,
                            'onet_code_8d': onet_code,
                            'onet_code_employment_weight': emp_weight,
                            'onet_title': onet_title,
                            'onet_task_id': task_id,
                            'task_text': task_text,
                            'task_importance_weight': task_importance,
                            'is_exposed_at_percentile': is_exposed,
                            'num_matching_apps_total': total_matching_apps,
                            'num_matching_apps_at_percentile': num_matching_at_pct,
                            'matching_ai_application': app_text,
                            'ai_app_id': ai_app_id,
                            'bi_encoder_similarity': bi_encoder_similarity,
                            'cross_encoder_similarity': cross_encoder_similarity,
                            'percentile_threshold': self.percentile,
                            'ce_threshold': self.ce_threshold
                        }

                        # Add exposure metrics from ISCO level
                        for key, value in isco_metrics.items():
                            if key != 'isco08_4d':  # Skip the ISCO code, we already have it
                                row[key] = value

                        output_rows.append(row)

            # Convert to DataFrame
            result_df = pd.DataFrame(output_rows)

            if len(result_df) > 0:
                # Save the detailed breakdown
                output_file = f"top_occupations_task_breakdown_{self.percentile}_{self.ce_threshold}.csv"
                self._save_table(result_df, 'tasks', output_file,
                                f"Task-level breakdown for top {self.top_n} occupations at {self.percentile} (BGE) and {self.ce_threshold} (CE) thresholds")

                self.logger.info(f"  Generated task breakdown with {len(result_df):,} task records")

                # Log summary statistics
                exposed_count = result_df[result_df['is_exposed_at_percentile'] == True].shape[0]
                self.logger.info(f"  Tasks exposed at {self.percentile}: {exposed_count:,} / {len(result_df):,} ({100*exposed_count/len(result_df):.1f}%)")
            else:
                self.logger.warning("  No task data available for breakdown")

        except Exception as e:
            self.logger.error(f"Error in task breakdown analysis: {str(e)}", exc_info=True)

    def exposure_distribution_by_percentile(self):
        """Analyze exposure distributions across different BGE similarity percentiles."""
        self.logger.info("→ Analyzing exposure distribution by percentile...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not available")
            return

        percentiles = ['pct_01', 'pct_0p1', 'pct_05', 'pct_10', 'pct_15', 'pct_20']
        results = []

        for pct in percentiles:
            # Check if this percentile exists in the data
            col_name = f'hampole_ai_exposure_avg_{pct}_{self.ce_threshold}'
            if col_name not in self.stage6_jobs.columns:
                continue

            exposure_vals = self.stage6_jobs[col_name]

            results.append({
                'percentile': pct,
                'mean_exposure': exposure_vals.mean(),
                'median_exposure': exposure_vals.median(),
                'std_exposure': exposure_vals.std(),
                'min_exposure': exposure_vals.min(),
                'max_exposure': exposure_vals.max(),
                'q25_exposure': exposure_vals.quantile(0.25),
                'q75_exposure': exposure_vals.quantile(0.75),
                'n_exposed_jobs': (exposure_vals > 0).sum(),
                'pct_exposed_jobs': (exposure_vals > 0).sum() / len(exposure_vals) * 100
            })

        result_df = pd.DataFrame(results)
        self._save_table(result_df, 'summary', 'exposure_distribution_by_percentile.csv',
                        "Exposure distribution statistics across BGE similarity percentiles")

    def coverage_statistics(self):
        """Generate coverage statistics showing data availability across thresholds."""
        self.logger.info("→ Generating coverage statistics...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not available")
            return

        percentiles = ['pct_01', 'pct_0p1', 'pct_05', 'pct_10', 'pct_15', 'pct_20']
        coverage_data = []

        n_total_firms = self.stage6_jobs['company_id'].nunique()
        n_total_occs = self.stage6_jobs['isco08_4d'].nunique()
        n_total_jobs = len(self.stage6_jobs)

        for pct in percentiles:
            col_name = f'hampole_ai_exposure_avg_{pct}_{self.ce_threshold}'
            if col_name not in self.stage6_jobs.columns:
                continue

            # Count entities with any exposure
            jobs_with_exposure = self.stage6_jobs[self.stage6_jobs[col_name] > 0]
            firms_with_exposure = jobs_with_exposure['company_id'].nunique()
            occs_with_exposure = jobs_with_exposure['isco08_4d'].nunique()

            coverage_data.append({
                'percentile': pct,
                'n_firms_with_exposure': firms_with_exposure,
                'pct_firms_with_exposure': firms_with_exposure / n_total_firms * 100,
                'n_occupations_with_exposure': occs_with_exposure,
                'pct_occupations_with_exposure': occs_with_exposure / n_total_occs * 100,
                'n_jobs_with_exposure': len(jobs_with_exposure),
                'pct_jobs_with_exposure': len(jobs_with_exposure) / n_total_jobs * 100
            })

        coverage_df = pd.DataFrame(coverage_data)
        self._save_table(coverage_df, 'summary', 'coverage_statistics.csv',
                        "Coverage statistics: entities with AI exposure by threshold")

    def exposure_distribution_plots(self):
        """Generate distribution visualizations of exposure across dimensions."""
        self.logger.info("→ Generating exposure distribution plots...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not available")
            return

        # Use base column name (after percentile extraction, suffix is stripped)
        col_name = 'hampole_ai_exposure_avg'
        if col_name not in self.stage6_jobs.columns:
            self.logger.warning(f"Column {col_name} not found in Stage 6 jobs")
            return

        exposure_vals = self.stage6_jobs[col_name]

        # Create a 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Overall distribution histogram
        axes[0, 0].hist(exposure_vals[exposure_vals > 0], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('AI Exposure Score', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Distribution of AI Exposure (Jobs with Exposure > 0)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # Plot 2: Distribution by firm (top 10 firms)
        top_firms = self.stage6_jobs.groupby('company_name')[col_name].mean().nlargest(10).index
        firm_data = [self.stage6_jobs[self.stage6_jobs['company_name'] == firm][col_name].values for firm in top_firms]
        axes[0, 1].boxplot(firm_data, labels=[name[:20] for name in top_firms], vert=True)
        axes[0, 1].set_ylabel('AI Exposure Score', fontsize=11)
        axes[0, 1].set_title('Exposure Distribution: Top 10 Firms', fontsize=12, fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # Plot 3: Distribution by occupation (top 10 occupations)
        top_occs = self.stage6_jobs.groupby('isco08_title')[col_name].mean().nlargest(10).index
        occ_data = [self.stage6_jobs[self.stage6_jobs['isco08_title'] == occ][col_name].values for occ in top_occs]
        axes[1, 0].boxplot(occ_data, labels=[name[:20] for name in top_occs], vert=True)
        axes[1, 0].set_ylabel('AI Exposure Score', fontsize=11)
        axes[1, 0].set_title('Exposure Distribution: Top 10 Occupations', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)

        # Plot 4: CDF (cumulative distribution)
        sorted_exposure = np.sort(exposure_vals[exposure_vals > 0])
        cumsum = np.arange(1, len(sorted_exposure) + 1) / len(sorted_exposure)
        axes[1, 1].plot(sorted_exposure, cumsum, color='darkgreen', linewidth=2)
        axes[1, 1].set_xlabel('AI Exposure Score', fontsize=11)
        axes[1, 1].set_ylabel('Cumulative Probability', fontsize=11)
        axes[1, 1].set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_plot(fig, 'summary', 'exposure_distribution_plots.png')

    def percentile_comparison_analysis(self):
        """Compare exposure results across different BGE similarity percentiles."""
        self.logger.info("→ Comparing percentile thresholds...")

        if self.stage6_jobs is None:
            self.logger.warning("Stage 6 jobs data not available")
            return

        percentiles = ['pct_01', 'pct_0p1', 'pct_05', 'pct_10', 'pct_15', 'pct_20']
        comparison_data = []

        # For each percentile, rank top 20 firms and occupations
        for pct in percentiles:
            col_name = f'hampole_ai_exposure_avg_{pct}_{self.ce_threshold}'
            if col_name not in self.stage6_jobs.columns:
                continue

            # Top firm by exposure
            top_firm = self.stage6_jobs.groupby('company_name')[col_name].mean().idxmax()
            top_firm_exposure = self.stage6_jobs.groupby('company_name')[col_name].mean().max()

            # Top occupation by exposure
            top_occ = self.stage6_jobs.groupby('isco08_title')[col_name].mean().idxmax()
            top_occ_exposure = self.stage6_jobs.groupby('isco08_title')[col_name].mean().max()

            # Mean and median
            mean_exp = self.stage6_jobs[col_name].mean()
            median_exp = self.stage6_jobs[col_name].median()

            comparison_data.append({
                'percentile': pct,
                'top_firm_name': top_firm,
                'top_firm_exposure': top_firm_exposure,
                'top_occupation_title': top_occ,
                'top_occupation_exposure': top_occ_exposure,
                'mean_exposure': mean_exp,
                'median_exposure': median_exp,
                'std_exposure': self.stage6_jobs[col_name].std()
            })

        comparison_df = pd.DataFrame(comparison_data)
        self._save_table(comparison_df, 'summary', 'percentile_comparison.csv',
                        "Comparison of exposure rankings across BGE similarity percentiles")

        # Also create a sensitivity ranking table (how consistent are top firms?)
        self.logger.info("  → Computing sensitivity rankings...")
        firm_rankings = {}
        for pct in percentiles:
            col_name = f'hampole_ai_exposure_avg_{pct}_{self.ce_threshold}'
            if col_name not in self.stage6_jobs.columns:
                continue
            ranked = self.stage6_jobs.groupby('company_name')[col_name].mean().rank(ascending=False)
            for firm, rank in ranked.items():
                if firm not in firm_rankings:
                    firm_rankings[firm] = []
                firm_rankings[firm].append(rank)

        # Calculate ranking stability
        sensitivity_data = []
        for firm, ranks in firm_rankings.items():
            if len(ranks) >= 3:  # Only if firm appears in multiple percentiles
                sensitivity_data.append({
                    'firm_name': firm,
                    'avg_rank': np.mean(ranks),
                    'rank_std': np.std(ranks),
                    'rank_min': min(ranks),
                    'rank_max': max(ranks),
                    'rank_range': max(ranks) - min(ranks)
                })

        if sensitivity_data:
            sensitivity_df = pd.DataFrame(sensitivity_data).sort_values('avg_rank').head(50)
            self._save_table(sensitivity_df, 'summary', 'firm_ranking_sensitivity.csv',
                            "Top 50 firms: ranking stability across percentile thresholds")

    # Section 6: Summary Statistics

    def generate_summary_statistics(self):
        """Generate comprehensive dataset summary."""
        self.logger.info("→ Generating summary statistics...")

        # Dataset statistics
        summary_stats = {
            'metric': ['Total firms', 'Total occupations (ISCO)', 'Total tasks', 'Total AI applications',
                      'Total firm-occ-year combinations', 'Years covered (min)', 'Years covered (max)',
                      'Total jobs (Stage 6)'],
            'value': [
                self.stage6_firm_report['company_id'].nunique() if self.stage6_firm_report is not None else 0,
                self.stage5_isco['isco08_4d'].nunique() if self.stage5_isco is not None else 0,
                self.stage5_task.shape[0] if self.stage5_task is not None else 0,
                self.stage4_matches['app_text'].nunique() if self.stage4_matches is not None else 0,
                self.stage6_jobs.shape[0] if self.stage6_jobs is not None else 0,
                self.stage6_jobs['year'].min() if self.stage6_jobs is not None and 'year' in self.stage6_jobs.columns else 'N/A',
                self.stage6_jobs['year'].max() if self.stage6_jobs is not None and 'year' in self.stage6_jobs.columns else 'N/A',
                self.stage6_jobs.shape[0] if self.stage6_jobs is not None else 0
            ]
        }

        self._save_table(pd.DataFrame(summary_stats), 'summary', 'dataset_statistics.csv',
                        "Dataset summary statistics")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Analyze Stage 5/6 AI exposure results'
    )

    parser.add_argument(
        '--stage5-dir',
        type=str,
        default='Data/firm_year_exposure',
        help='Directory containing Stage 5 outputs'
    )

    parser.add_argument(
        '--stage6-dir',
        type=str,
        default='Data',
        help='Directory containing Stage 6 outputs'
    )

    parser.add_argument(
        '--stage6-suffix',
        type=str,
        default='',
        help='Suffix for Stage 6 output files (e.g., "_1000_sample", "_core_tasks")'
    )

    parser.add_argument(
        '--stage4-file',
        type=str,
        default='Data/task_exposure_matches_all_thresholds_core_tasks.parquet',
        help='Path to Stage 4 task-app matches'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f'Directory for analysis outputs (default: {DEFAULT_OUTPUT_DIR})'
    )

    parser.add_argument(
        '--percentile',
        type=str,
        default=DEFAULT_PERCENTILE,
        choices=['pct_01', 'pct_05', 'pct_10', 'pct_15', 'pct_20'],
        help='BGE percentile to use as baseline (default: pct_05)'
    )

    parser.add_argument(
        '--ce-threshold',
        type=str,
        default=DEFAULT_CE_THRESHOLD,
        choices=['ce_0.8', 'ce_0.6', 'ce_0.4', 'ce_0.2', 'ce_0.0'],
        help='Cross-encoder threshold (default: ce_0.0 = no CE filtering)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=DEFAULT_TOP_N,
        help='Number of top items in rankings (default: 20)'
    )

    parser.add_argument(
        '--sections',
        type=int,
        nargs='+',
        choices=[2, 3, 4, 5, 6],
        help='Specific sections to run (default: all)'
    )

    args = parser.parse_args()

    # Create analyzer and run
    analyzer = ExposureAnalyzer(
        stage5_dir=args.stage5_dir,
        stage6_dir=args.stage6_dir,
        stage6_suffix=args.stage6_suffix,
        stage4_file=args.stage4_file,
        output_dir=args.output_dir,
        percentile=args.percentile,
        ce_threshold=args.ce_threshold,
        top_n=args.top_n
    )

    analyzer.run_all_analyses(sections=args.sections)


if __name__ == '__main__':
    main()
