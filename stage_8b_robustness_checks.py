#!/usr/bin/env python3
"""
Stage 8b: Robustness Checks Framework for SHP Panel Analysis (SIMPLIFIED)

Purpose:
  Run econometric specifications using de-meaning approach (Frisch-Waugh-Lovell)
  which is more stable than fitting many FEs simultaneously

Key insight: For firm-year FE, instead of creating 30k dummy variables,
de-mean all variables by firm-year, then estimate person FE on residuals.

Input:
  Data/shp_panel_prepared.csv (from stage_8a_prepare_panel_data.py)

Output:
  Data/shp_econometric_results/
    ├── all_specifications_results.csv
    ├── coefficient_comparison.csv
    ├── endogeneity_evidence/
    └── robustness_summary.txt

Author: Claude Code
Date: 2026-04-10
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

# Setup
project_root = Path(__file__).parent
data_dir = project_root / "Data"
results_dir = data_dir / "shp_econometric_results"
results_dir.mkdir(exist_ok=True)
endogeneity_dir = results_dir / "endogeneity_evidence"
endogeneity_dir.mkdir(exist_ok=True)

log_file = results_dir / "robustness_summary.txt"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SimplifiedSpecificationRunner:
    """Efficient specification runner using de-meaning approach."""

    def __init__(self, df):
        self.df = df.copy()
        self.outcome_vars = [col for col in df.columns if col.startswith('outcome_')]
        logger.info(f"Initialized SpecificationRunner with {len(self.outcome_vars)} outcomes")

    def demean_by_group(self, df, cols_to_demean, group_col):
        """De-mean columns by a grouping variable."""
        df_out = df.copy()
        for col in cols_to_demean:
            if col in df.columns:
                df_out[col + '_dm'] = (df[col] -
                                       df.groupby(group_col)[col].transform('mean'))
        return df_out

    def fit_ols_demeaned(self, y, X, cluster_var=None):
        """Fit OLS on de-meaned data with optional clustering."""
        try:
            X_const = sm.add_constant(X)
            if cluster_var is not None:
                model = sm.OLS(y, X_const).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': cluster_var}
                )
            else:
                model = sm.OLS(y, X_const).fit()
            return model
        except Exception as e:
            logger.warning(f"OLS fit failed: {e}")
            return None

    def run_specification_1(self, outcome_var):
        """
        SPEC 1: No FE (baseline)
        Y_ift = β * Exposure + controls + ε_ift
        """
        try:
            data = self.df[[
                'idpers', outcome_var,
                'hampole_ai_exposure_avg_foy', 'age_centered', 'female', 'employed'
            ]].dropna(subset=[outcome_var]).copy()

            if len(data) < 100:
                return {'status': 'failed', 'error': f'N={len(data)} < 100'}

            y = data[outcome_var]
            X = data[['hampole_ai_exposure_avg_foy', 'age_centered', 'female', 'employed']]

            model = self.fit_ols_demeaned(y, X, cluster_var=data['idpers'])
            if model is None:
                return {'status': 'failed', 'error': 'Model fit failed'}

            coef = model.params['hampole_ai_exposure_avg_foy']
            se = model.bse['hampole_ai_exposure_avg_foy']
            pval = model.pvalues['hampole_ai_exposure_avg_foy']

            return {
                'spec': 'Spec1_NoFE',
                'outcome': outcome_var,
                'n_obs': len(data),
                'coef': coef,
                'se': se,
                'pval': pval,
                'ci_lower': coef - 1.96 * se,
                'ci_upper': coef + 1.96 * se,
                'r_squared': model.rsquared,
                'status': 'success'
            }
        except Exception as e:
            return {'outcome': outcome_var, 'status': 'failed', 'error': str(e)[:100]}

    def run_specification_2(self, outcome_var):
        """
        SPEC 2: Firm-Year FE + Person FE (PRIMARY)
        Y_ift = α_i + α_ft + β * Exposure + controls + ε_ift

        Implementation: De-mean by firm-year, then fit person FE
        """
        try:
            data = self.df[[
                'idpers', 'firm_year', outcome_var,
                'hampole_ai_exposure_avg_foy', 'age_centered', 'female', 'employed'
            ]].dropna(subset=[outcome_var]).copy()

            if len(data) < 100:
                return {'status': 'failed', 'error': f'N={len(data)} < 100'}

            # Step 1: De-mean by firm_year to absorb firm-year FE
            cols_to_dm = ['hampole_ai_exposure_avg_foy', outcome_var, 'age_centered', 'female', 'employed']
            data = self.demean_by_group(data, cols_to_dm, 'firm_year')

            # Step 2: De-mean again by person to absorb person FE
            dm_cols_2 = ['hampole_ai_exposure_avg_foy_dm', outcome_var + '_dm',
                        'age_centered_dm', 'female_dm', 'employed_dm']
            data = self.demean_by_group(data, dm_cols_2, 'idpers')

            # Step 3: Fit OLS on doubly de-meaned data (no intercept needed)
            y = data[outcome_var + '_dm_dm']
            X = data[['hampole_ai_exposure_avg_foy_dm_dm', 'age_centered_dm_dm',
                     'female_dm_dm', 'employed_dm_dm']]

            # Remove any rows with NaNs or infs
            good_idx = ~(y.isna() | y.isin([np.inf, -np.inf])) & \
                      ~(X.isna().any(axis=1) | X.isin([np.inf, -np.inf]).any(axis=1))
            y = y[good_idx]
            X = X[good_idx]

            if len(y) < 50:
                return {'status': 'failed', 'error': f'After demean: N={len(y)} < 50'}

            model = self.fit_ols_demeaned(y, X, cluster_var=data.loc[good_idx, 'idpers'])
            if model is None:
                return {'status': 'failed', 'error': 'Model fit failed'}

            coef = model.params['hampole_ai_exposure_avg_foy_dm_dm']
            se = model.bse['hampole_ai_exposure_avg_foy_dm_dm']
            pval = model.pvalues['hampole_ai_exposure_avg_foy_dm_dm']

            return {
                'spec': 'Spec2_FY_PersonFE',
                'outcome': outcome_var,
                'n_obs': len(y),
                'coef': coef,
                'se': se,
                'pval': pval,
                'ci_lower': coef - 1.96 * se,
                'ci_upper': coef + 1.96 * se,
                'r_squared': model.rsquared,
                'status': 'success',
                'note': 'De-meaned by firm-year, then person'
            }
        except Exception as e:
            return {'outcome': outcome_var, 'status': 'failed', 'error': str(e)[:100]}

    def run_specification_3(self, outcome_var):
        """
        SPEC 3: Year FE + Person FE (Robustness)
        Y_ift = α_i + α_t + β * Exposure + controls + ε_ift
        """
        try:
            data = self.df[[
                'idpers', 'year', outcome_var,
                'hampole_ai_exposure_avg_foy', 'age_centered', 'female', 'employed'
            ]].dropna(subset=[outcome_var]).copy()

            if len(data) < 100:
                return {'status': 'failed', 'error': f'N={len(data)} < 100'}

            # De-mean by year
            cols_to_dm = ['hampole_ai_exposure_avg_foy', outcome_var, 'age_centered', 'female', 'employed']
            data = self.demean_by_group(data, cols_to_dm, 'year')

            # De-mean again by person
            dm_cols_2 = ['hampole_ai_exposure_avg_foy_dm', outcome_var + '_dm',
                        'age_centered_dm', 'female_dm', 'employed_dm']
            data = self.demean_by_group(data, dm_cols_2, 'idpers')

            # Fit on doubly de-meaned data
            y = data[outcome_var + '_dm_dm']
            X = data[['hampole_ai_exposure_avg_foy_dm_dm', 'age_centered_dm_dm',
                     'female_dm_dm', 'employed_dm_dm']]

            # Clean
            good_idx = ~(y.isna() | y.isin([np.inf, -np.inf])) & \
                      ~(X.isna().any(axis=1) | X.isin([np.inf, -np.inf]).any(axis=1))
            y = y[good_idx]
            X = X[good_idx]

            if len(y) < 50:
                return {'status': 'failed', 'error': f'After demean: N={len(y)} < 50'}

            model = self.fit_ols_demeaned(y, X, cluster_var=data.loc[good_idx, 'idpers'])
            if model is None:
                return {'status': 'failed', 'error': 'Model fit failed'}

            coef = model.params['hampole_ai_exposure_avg_foy_dm_dm']
            se = model.bse['hampole_ai_exposure_avg_foy_dm_dm']
            pval = model.pvalues['hampole_ai_exposure_avg_foy_dm_dm']

            return {
                'spec': 'Spec3_YearPersonFE',
                'outcome': outcome_var,
                'n_obs': len(y),
                'coef': coef,
                'se': se,
                'pval': pval,
                'ci_lower': coef - 1.96 * se,
                'ci_upper': coef + 1.96 * se,
                'r_squared': model.rsquared,
                'status': 'success',
                'note': 'De-meaned by year, then person'
            }
        except Exception as e:
            return {'outcome': outcome_var, 'status': 'failed', 'error': str(e)[:100]}

    def run_all_specs(self):
        """Run all three specs for all outcomes."""
        logger.info(f"\nRunning {len(self.outcome_vars)} outcomes × 3 specs...\n")

        all_results = []
        for i, outcome_var in enumerate(self.outcome_vars, 1):
            logger.info(f"[{i}/{len(self.outcome_vars)}] {outcome_var}")

            result1 = self.run_specification_1(outcome_var)
            all_results.append(result1)
            if result1.get('status') == 'success':
                logger.info(f"  ✓ Spec 1: β={result1['coef']:.4f} (p={result1['pval']:.3f})")
            else:
                logger.info(f"  ✗ Spec 1: {result1.get('error', 'Unknown error')}")

            result2 = self.run_specification_2(outcome_var)
            all_results.append(result2)
            if result2.get('status') == 'success':
                logger.info(f"  ✓ Spec 2: β={result2['coef']:.4f} (p={result2['pval']:.3f}) [n={result2['n_obs']}]")
            else:
                logger.info(f"  ✗ Spec 2: {result2.get('error', 'Unknown error')}")

            result3 = self.run_specification_3(outcome_var)
            all_results.append(result3)
            if result3.get('status') == 'success':
                logger.info(f"  ✓ Spec 3: β={result3['coef']:.4f} (p={result3['pval']:.3f}) [n={result3['n_obs']}]")
            else:
                logger.info(f"  ✗ Spec 3: {result3.get('error', 'Unknown error')}")

        return pd.DataFrame(all_results)


class EndogeneityAnalyzer:
    """Analyze endogeneity: task-fit vs. restructuring narratives."""

    def __init__(self, df):
        self.df = df
        logger.info("Initialized EndogeneityAnalyzer")

    def task_fit_analysis(self):
        """Does AI exposure correlate with task-replaceability?"""
        logger.info("\nPerforming task-fit analysis...")

        try:
            occ_analysis = self.df.groupby('isco08_4d').agg({
                'hampole_ai_exposure_avg_o': 'mean',
                'total_tasks_occupation': 'mean'
            }).dropna()

            if len(occ_analysis) < 2:
                return None

            correlation = occ_analysis['hampole_ai_exposure_avg_o'].corr(
                occ_analysis['total_tasks_occupation']
            )

            result = {
                'analysis': 'Task-Fit Correlation',
                'n_occupations': len(occ_analysis),
                'correlation': correlation,
                'mean_exposure': occ_analysis['hampole_ai_exposure_avg_o'].mean(),
                'interpretation': 'Strong positive → benign (task-driven); Near zero → concerning'
            }

            logger.info(f"  Correlation (AI exposure vs. task count): {correlation:.3f}")
            return result

        except Exception as e:
            logger.error(f"  Task-fit analysis failed: {e}")
            return None

    def pre_treatment_balance(self):
        """Does pre-2016 occupation health predict AI exposure?"""
        logger.info("\nPerforming pre-treatment balance analysis...")

        try:
            pre_period = self.df[self.df['year'] <= 2015].copy()

            if len(pre_period) < 100:
                logger.warning(f"  Insufficient pre-treatment data: {len(pre_period)}")
                return None

            pre_data = pre_period[[
                'isco08_4d', 'hampole_ai_exposure_avg_o', 'outcome_log_income'
            ]].dropna()

            if len(pre_data) < 50:
                logger.warning(f"  Insufficient pre-outcome data: {len(pre_data)}")
                return None

            model = sm.OLS(
                pre_data['outcome_log_income'],
                sm.add_constant(pre_data['hampole_ai_exposure_avg_o'])
            ).fit()

            result = {
                'analysis': 'Pre-Treatment Balance',
                'n_obs': len(pre_data),
                'coef': model.params['hampole_ai_exposure_avg_o'],
                'pval': model.pvalues['hampole_ai_exposure_avg_o'],
                'interpretation': 'Non-sig → good (no targeting); Sig → concerning (targeting troubled occupations)'
            }

            logger.info(f"  Pre-treatment coef: {result['coef']:.4f} (p={result['pval']:.3f})")
            return result

        except Exception as e:
            logger.error(f"  Pre-treatment analysis failed: {e}")
            return None

    def generate_narrative(self, task_fit_result, pre_treatment_result):
        """Generate endogeneity narrative."""
        narrative = [
            "=" * 80,
            "ENDOGENEITY EVIDENCE: BENIGN VS. MALIGN NARRATIVE",
            "=" * 80,
            ""
        ]

        if task_fit_result:
            narrative.append("1. TASK-FIT EVIDENCE (Benign Narrative):")
            narrative.append(f"   Correlation: {task_fit_result['correlation']:.3f}")
            if abs(task_fit_result['correlation']) > 0.2:
                narrative.append("   ✓ Moderate-to-strong: Task-driven deployment likely")
            else:
                narrative.append("   ⚠ Weak: Deployment may not follow task structure")
        else:
            narrative.append("1. TASK-FIT EVIDENCE: Insufficient data")

        narrative.append("")

        if pre_treatment_result:
            narrative.append("2. PRE-TREATMENT BALANCE (Addressing Malign Narrative):")
            narrative.append(f"   Coef: {pre_treatment_result['coef']:.4f} (p={pre_treatment_result['pval']:.3f})")
            if pre_treatment_result['pval'] > 0.1:
                narrative.append("   ✓ Non-significant: No evidence of targeting troubled occupations")
            else:
                narrative.append("   ⚠ Significant: Possible targeting of low-income occupations")
        else:
            narrative.append("2. PRE-TREATMENT BALANCE: Insufficient data")

        narrative.append("")
        narrative.append("See specification-shp-panel-econometrics.md for full interpretation.")
        narrative.append("=" * 80)

        return "\n".join(narrative)


def main():
    logger.info("=" * 80)
    logger.info("STAGE 8B: ROBUSTNESS CHECKS & ENDOGENEITY ANALYSIS (SIMPLIFIED)")
    logger.info("=" * 80)

    try:
        # Load data
        logger.info("\nStep 1: Loading prepared panel data...")
        prepared_file = data_dir / "shp_panel_prepared.csv"
        if not prepared_file.exists():
            raise FileNotFoundError(f"Prepared data not found: {prepared_file}")

        df = pd.read_csv(prepared_file, low_memory=False)
        logger.info(f"✓ Loaded {len(df):,} person-year observations")

        # Run specifications
        logger.info("\nStep 2: Running all econometric specifications...")
        runner = SimplifiedSpecificationRunner(df)
        results_df = runner.run_all_specs()

        # Save results
        results_df.to_csv(results_dir / "all_specifications_results.csv", index=False)
        logger.info(f"\n✓ Saved results for {len(results_df)} models")

        # Run endogeneity analysis
        logger.info("\nStep 3: Analyzing endogeneity...")
        analyzer = EndogeneityAnalyzer(df)
        task_fit = analyzer.task_fit_analysis()
        pre_treatment = analyzer.pre_treatment_balance()

        # Save endogeneity evidence
        if task_fit:
            pd.DataFrame([task_fit]).to_csv(
                endogeneity_dir / "task_fit_correlation.csv", index=False
            )
        if pre_treatment:
            pd.DataFrame([pre_treatment]).to_csv(
                endogeneity_dir / "pre_treatment_balance.csv", index=False
            )

        # Generate narrative
        narrative = analyzer.generate_narrative(task_fit, pre_treatment)
        with open(endogeneity_dir / "endogeneity_narrative.txt", 'w') as f:
            f.write(narrative)
        logger.info("✓ Generated endogeneity narrative")

        logger.info("\n" + "=" * 80)
        logger.info("✓ STAGE 8B COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nResults saved to: {results_dir}/")

        return 0

    except Exception as e:
        logger.error(f"\n✗ ERROR: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
