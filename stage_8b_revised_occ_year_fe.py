#!/usr/bin/env python3
"""
Stage 8b REVISED: Econometric Specifications with OCCUPATION-YEAR FE

Purpose:
  Run specifications using OCCUPATION-YEAR FE (instead of firm-year FE)
  This is more appropriate for political outcomes analysis because:
  - Occupation-level labor trends drive political shifts
  - Nationwide occupation trends (not firm-specific shocks) matter for politics

Specification:
  Y_ift = α_i + α_ot + β*Exposure_iot + controls + ε_ift

  Person FE (α_i): Time-invariant individual heterogeneity
  Occupation-Year FE (α_ot): Nationwide occupation trends in year t

  Identifying variation: Within occupation-year, across firms
  Logic: In occupation o in year t, do workers in higher-exposure firms
         differ politically from workers in lower-exposure firms?

Input:
  Data/shp_panel_prepared.csv

Output:
  Data/shp_econometric_results/occupation_year_fe_results.csv
  Data/shp_econometric_results/fe_comparison.csv (firm-year vs. occ-year)

Author: Claude Code
Date: 2026-04-10
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys
import logging

# Setup
project_root = Path(__file__).parent
data_dir = project_root / "Data"
results_dir = data_dir / "shp_econometric_results"
results_dir.mkdir(exist_ok=True)

logger = logging.getLogger(__name__)

class OccupationYearFESpecification:
    """Run specs with occupation-year FE (better for political outcomes)."""

    def __init__(self, df):
        self.df = df.copy()
        self.outcome_vars = [col for col in df.columns if col.startswith('outcome_')]
        print(f"Initialized with {len(self.outcome_vars)} outcomes")
        print(f"Sample size: {len(self.df):,} person-years")

    def demean_by_group(self, df, cols_to_demean, group_col):
        """De-mean columns by a grouping variable."""
        df_out = df.copy()
        for col in cols_to_demean:
            if col in df.columns:
                df_out[col + '_dm'] = (df[col] -
                                       df.groupby(group_col)[col].transform('mean'))
        return df_out

    def run_occ_year_fe_spec(self, outcome_var):
        """
        Run: Person FE + Occupation-Year FE
        Y_ift = α_i + α_ot + β*Exposure + controls + ε_ift
        """
        try:
            data = self.df[[
                'idpers', 'occ_year', outcome_var,
                'hampole_ai_exposure_avg_foy', 'age_centered', 'female', 'employed'
            ]].dropna(subset=[outcome_var]).copy()

            if len(data) < 100:
                return {'status': 'failed', 'error': f'N={len(data)} < 100'}

            # Step 1: De-mean by occupation-year to absorb occ-year FE
            cols_to_dm = ['hampole_ai_exposure_avg_foy', outcome_var, 'age_centered', 'female', 'employed']
            data = self.demean_by_group(data, cols_to_dm, 'occ_year')

            # Step 2: De-mean again by person to absorb person FE
            dm_cols_2 = ['hampole_ai_exposure_avg_foy_dm', outcome_var + '_dm',
                        'age_centered_dm', 'female_dm', 'employed_dm']
            data = self.demean_by_group(data, dm_cols_2, 'idpers')

            # Step 3: Fit OLS on doubly de-meaned data
            y = data[outcome_var + '_dm_dm']
            X = data[['hampole_ai_exposure_avg_foy_dm_dm', 'age_centered_dm_dm',
                     'female_dm_dm', 'employed_dm_dm']]

            # Clean NaNs and infs
            good_idx = ~(y.isna() | y.isin([np.inf, -np.inf])) & \
                      ~(X.isna().any(axis=1) | X.isin([np.inf, -np.inf]).any(axis=1))
            y = y[good_idx]
            X = X[good_idx]

            if len(y) < 50:
                return {'status': 'failed', 'error': f'After demean: N={len(y)} < 50'}

            # Fit with clustering by person
            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit(
                cov_type='cluster',
                cov_kwds={'groups': data.loc[good_idx, 'idpers']}
            )

            coef = model.params['hampole_ai_exposure_avg_foy_dm_dm']
            se = model.bse['hampole_ai_exposure_avg_foy_dm_dm']
            pval = model.pvalues['hampole_ai_exposure_avg_foy_dm_dm']

            return {
                'spec': 'OccYearPersonFE',
                'outcome': outcome_var,
                'n_obs': len(y),
                'coef': coef,
                'se': se,
                'pval': pval,
                'ci_lower': coef - 1.96 * se,
                'ci_upper': coef + 1.96 * se,
                'r_squared': model.rsquared,
                'status': 'success',
                'note': 'De-meaned by occ-year, then person'
            }

        except Exception as e:
            return {
                'outcome': outcome_var,
                'status': 'failed',
                'error': str(e)[:100]
            }

    def run_all(self):
        """Run occ-year FE spec for all outcomes."""
        print(f"\nRunning Occupation-Year FE specification for {len(self.outcome_vars)} outcomes...\n")

        all_results = []

        for i, outcome_var in enumerate(self.outcome_vars, 1):
            print(f"[{i}/{len(self.outcome_vars)}] {outcome_var}...")

            result = self.run_occ_year_fe_spec(outcome_var)
            all_results.append(result)

            if result.get('status') == 'success':
                sig = "**" if result['pval'] < 0.05 else "*" if result['pval'] < 0.10 else ""
                print(f"  ✓ β={result['coef']:7.4f} (p={result['pval']:.3f}){sig} [n={result['n_obs']}]")
            else:
                print(f"  ✗ {result.get('error', 'Unknown error')}")

        return pd.DataFrame(all_results)

def compare_fe_specifications(firm_year_results, occ_year_results):
    """Create comparison table: Firm-Year FE vs. Occ-Year FE."""

    # Clean results
    fy_clean = firm_year_results[firm_year_results['status'] == 'success'].copy()
    oy_clean = occ_year_results[occ_year_results['status'] == 'success'].copy()

    # Get common outcomes
    outcomes = sorted(set(fy_clean['outcome'].unique()) & set(oy_clean['outcome'].unique()))

    comparison = []
    for outcome in outcomes:
        row = {'Outcome': outcome.replace('outcome_', '')}

        fy = fy_clean[fy_clean['outcome'] == outcome]
        if len(fy) > 0:
            fy = fy.iloc[0]
            row['FY_Coef'] = fy['coef']
            row['FY_SE'] = fy['se']
            row['FY_Pval'] = fy['pval']
            row['FY_Sig'] = "***" if fy['pval'] < 0.001 else "**" if fy['pval'] < 0.01 else \
                            "*" if fy['pval'] < 0.05 else "†" if fy['pval'] < 0.10 else ""

        oy = oy_clean[oy_clean['outcome'] == outcome]
        if len(oy) > 0:
            oy = oy.iloc[0]
            row['OY_Coef'] = oy['coef']
            row['OY_SE'] = oy['se']
            row['OY_Pval'] = oy['pval']
            row['OY_Sig'] = "***" if oy['pval'] < 0.001 else "**" if oy['pval'] < 0.01 else \
                            "*" if oy['pval'] < 0.05 else "†" if oy['pval'] < 0.10 else ""
            row['Coef_Diff'] = oy['coef'] - fy['coef'] if 'FY_Coef' in row else np.nan

        comparison.append(row)

    return pd.DataFrame(comparison)

def main():
    print("="*100)
    print("STAGE 8B REVISED: OCCUPATION-YEAR FE SPECIFICATION")
    print("="*100)

    # Load prepared data
    prepared_file = data_dir / "shp_panel_prepared.csv"
    if not prepared_file.exists():
        print(f"ERROR: Prepared data not found: {prepared_file}")
        return 1

    print(f"\nLoading prepared panel data...")
    df = pd.read_csv(prepared_file, low_memory=False)
    print(f"✓ Loaded {len(df):,} person-year observations")

    # Run occupation-year FE spec
    print(f"\n" + "="*100)
    print("RUNNING OCCUPATION-YEAR FE SPECIFICATION")
    print("="*100)

    runner = OccupationYearFESpecification(df)
    oy_results = runner.run_all()

    # Save occupation-year results
    oy_results.to_csv(results_dir / "occupation_year_fe_results.csv", index=False)
    print(f"\n✓ Saved: {results_dir / 'occupation_year_fe_results.csv'}")

    # Load firm-year results for comparison
    fy_file = results_dir / "all_specifications_results.csv"
    if fy_file.exists():
        fy_results = pd.read_csv(fy_file)
        fy_spec2 = fy_results[fy_results['spec'] == 'Spec2_FY_PersonFE'].copy()

        # Create comparison table
        comparison = compare_fe_specifications(fy_spec2, oy_results)

        # Save comparison
        comparison.to_csv(results_dir / "fe_comparison_firm_vs_occ_year.csv", index=False)
        print(f"✓ Saved: {results_dir / 'fe_comparison_firm_vs_occ_year.csv'}")

        # Print comparison
        print(f"\n" + "="*100)
        print("COMPARISON: FIRM-YEAR FE vs. OCCUPATION-YEAR FE")
        print("="*100)
        print("\nNotes:")
        print("  FY = Firm-Year FE (controls firm-level shocks)")
        print("  OY = Occupation-Year FE (controls nationwide occupation trends)")
        print("  *** p<0.001, ** p<0.01, * p<0.05, † p<0.10\n")
        print(comparison.to_string(index=False))

        # Highlight differences
        print(f"\n" + "="*100)
        print("KEY DIFFERENCES (Occ-Year vs. Firm-Year)")
        print("="*100)

        for _, row in comparison.iterrows():
            if pd.notna(row.get('OY_Pval')) and pd.notna(row.get('FY_Pval')):
                oy_sig = row['OY_Pval'] < 0.10
                fy_sig = row['FY_Pval'] < 0.10

                if oy_sig != fy_sig:  # Significance changed
                    print(f"\n{row['Outcome']}:")
                    if oy_sig and not fy_sig:
                        print(f"  ✓ BECOMES SIGNIFICANT with Occ-Year FE")
                        print(f"    Firm-Year: β={row['FY_Coef']:.4f}, p={row['FY_Pval']:.3f}")
                        print(f"    Occ-Year:  β={row['OY_Coef']:.4f}, p={row['OY_Pval']:.3f}")
                    else:
                        print(f"  ✗ LOSES SIGNIFICANCE with Occ-Year FE")
                        print(f"    Firm-Year: β={row['FY_Coef']:.4f}, p={row['FY_Pval']:.3f}")
                        print(f"    Occ-Year:  β={row['OY_Coef']:.4f}, p={row['OY_Pval']:.3f}")

    print(f"\n" + "="*100)
    print("✓ COMPLETE")
    print("="*100)

    return 0

if __name__ == '__main__':
    sys.exit(main())
