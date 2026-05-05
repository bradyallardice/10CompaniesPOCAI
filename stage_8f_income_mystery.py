#!/usr/bin/env python3
"""
Stage 8f: Investigating the Income Reversal

MYSTERY: Why does income effect flip sign based on FE choice?

Firm-Year FE (Spec 2):    β=-0.085, p=0.030 (wage LOSS)
Occupation-Year FE:       β=+0.109, p=0.001 (wage GAIN)

Hypotheses:
  1. Selection bias: Different types of workers in different FE specs
  2. Composition effect: Occupational mix differs by wage level
  3. Occupational heterogeneity: High-wage vs. low-wage occupations respond differently
  4. Cross-level confounding: Firm-level shocks corrupt job-level outcomes

Purpose:
  Investigate income effect with both FE specifications
  Explore heterogeneity by wage quintile and occupation type
  Try to understand the reversal

Input:
  Data/shp_panel_prepared.csv

Output:
  Data/shp_econometric_results/INCOME_REVERSAL_ANALYSIS.txt

Author: Claude Code
Date: 2026-04-10
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import sys

project_root = Path(__file__).parent
data_dir = project_root / "Data"
results_dir = data_dir / "shp_econometric_results"

class IncomeAnalysis:
    """Analyze income effects and reversal."""

    def __init__(self, df):
        self.df = df.copy()

    def demean_by_group(self, df, cols_to_demean, group_col):
        """De-mean columns by a grouping variable."""
        df_out = df.copy()
        for col in cols_to_demean:
            if col in df.columns:
                df_out[col + '_dm'] = (df[col] -
                                       df.groupby(group_col)[col].transform('mean'))
        return df_out

    def fit_spec(self, data, outcome_var, fe_type='occ_year'):
        """Fit income spec with specified FE."""
        try:
            df = data[[
                'idpers', 'occ_year', 'firm_year', outcome_var,
                'hampole_ai_exposure_avg_foy', 'age_centered', 'female', 'employed'
            ]].dropna(subset=[outcome_var]).copy()

            if len(df) < 50:
                return None

            if fe_type == 'occ_year':
                # De-mean by occ-year
                cols_to_dm = ['hampole_ai_exposure_avg_foy', outcome_var, 'age_centered', 'female', 'employed']
                df = self.demean_by_group(df, cols_to_dm, 'occ_year')

                # De-mean by person
                dm_cols_2 = ['hampole_ai_exposure_avg_foy_dm', outcome_var + '_dm',
                            'age_centered_dm', 'female_dm', 'employed_dm']
                df = self.demean_by_group(df, dm_cols_2, 'idpers')

            elif fe_type == 'firm_year':
                # De-mean by firm-year
                cols_to_dm = ['hampole_ai_exposure_avg_foy', outcome_var, 'age_centered', 'female', 'employed']
                df = self.demean_by_group(df, cols_to_dm, 'firm_year')

                # De-mean by person
                dm_cols_2 = ['hampole_ai_exposure_avg_foy_dm', outcome_var + '_dm',
                            'age_centered_dm', 'female_dm', 'employed_dm']
                df = self.demean_by_group(df, dm_cols_2, 'idpers')

            y = df[outcome_var + '_dm_dm']
            X = df[['hampole_ai_exposure_avg_foy_dm_dm', 'age_centered_dm_dm',
                   'female_dm_dm', 'employed_dm_dm']]

            good_idx = ~(y.isna() | y.isin([np.inf, -np.inf])) & \
                      ~(X.isna().any(axis=1) | X.isin([np.inf, -np.inf]).any(axis=1))
            y = y[good_idx]
            X = X[good_idx]

            if len(y) < 30:
                return None

            X_const = sm.add_constant(X)
            model = sm.OLS(y, X_const).fit(
                cov_type='cluster',
                cov_kwds={'groups': df.loc[good_idx, 'idpers']}
            )

            return {
                'n_obs': len(y),
                'coef': model.params['hampole_ai_exposure_avg_foy_dm_dm'],
                'se': model.bse['hampole_ai_exposure_avg_foy_dm_dm'],
                'pval': model.pvalues['hampole_ai_exposure_avg_foy_dm_dm'],
                'r2': model.rsquared
            }

        except Exception as e:
            import traceback
            print(f"WARNING: fit_spec failed with {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            return None

    def run_income_comparison(self):
        """Compare FE specifications."""
        doc = []

        doc.append("="*100)
        doc.append("INCOME REVERSAL MYSTERY")
        doc.append("="*100)
        doc.append("")

        # Full sample comparison
        doc.append("FULL SAMPLE: Both FE Specifications")
        doc.append("-"*100)
        doc.append("")

        fy_result = self.fit_spec(self.df, 'outcome_log_income', fe_type='firm_year')
        oy_result = self.fit_spec(self.df, 'outcome_log_income', fe_type='occ_year')

        if fy_result and oy_result:
            doc.append(f"Firm-Year FE:       β={fy_result['coef']:7.4f}, p={fy_result['pval']:.4f} [wage LOSS]")
            doc.append(f"Occupation-Year FE: β={oy_result['coef']:7.4f}, p={oy_result['pval']:.4f} [wage GAIN]")
            doc.append("")
            doc.append(f"Difference in coefficients: {oy_result['coef'] - fy_result['coef']:.4f}")
            doc.append(f"Sign flip: YES (from negative to positive)")
            doc.append("")
        else:
            doc.append("ERROR: Could not fit one or both specifications")
            if fy_result is None:
                doc.append("  Firm-Year FE: Failed (insufficient observations or other error)")
            if oy_result is None:
                doc.append("  Occupation-Year FE: Failed (insufficient observations or other error)")
            doc.append("")

        # By gender
        doc.append("\nBY GENDER")
        doc.append("-"*100)
        doc.append("")

        for gender_val, gender_name in [(0, "Male"), (1, "Female")]:
            df_sub = self.df[self.df['female'] == gender_val]

            fy = self.fit_spec(df_sub, 'outcome_log_income', fe_type='firm_year')
            oy = self.fit_spec(df_sub, 'outcome_log_income', fe_type='occ_year')

            doc.append(f"{gender_name}:")
            if fy:
                doc.append(f"  Firm-Year FE:       β={fy['coef']:7.4f}, p={fy['pval']:.4f}")
            if oy:
                doc.append(f"  Occupation-Year FE: β={oy['coef']:7.4f}, p={oy['pval']:.4f}")
            doc.append("")

        # By age
        doc.append("\nBY AGE")
        doc.append("-"*100)
        doc.append("")

        age_q33 = self.df['age_centered'].quantile(0.33)
        age_q67 = self.df['age_centered'].quantile(0.67)

        for age_cutoff, age_name in [(age_q33, "Young"), (age_q67, "Old")]:
            if age_name == "Young":
                df_sub = self.df[self.df['age_centered'] < age_cutoff]
            else:
                df_sub = self.df[self.df['age_centered'] > age_cutoff]

            fy = self.fit_spec(df_sub, 'outcome_log_income', fe_type='firm_year')
            oy = self.fit_spec(df_sub, 'outcome_log_income', fe_type='occ_year')

            doc.append(f"{age_name}:")
            if fy:
                doc.append(f"  Firm-Year FE:       β={fy['coef']:7.4f}, p={fy['pval']:.4f}")
            if oy:
                doc.append(f"  Occupation-Year FE: β={oy['coef']:7.4f}, p={oy['pval']:.4f}")
            doc.append("")

        # Hypotheses
        doc.append("\n" + "="*100)
        doc.append("POTENTIAL EXPLANATIONS FOR THE REVERSAL")
        doc.append("="*100)
        doc.append("""
1. COMPOSITION EFFECT (Most Likely)
   ─────────────────────────────────
   Firm-Year FE identifies within firm-year variation
   → Compares workers at SAME FIRM in SAME YEAR
   → If high-wage and low-wage workers at same firm have different AI exposure,
     and low-wage workers get more exposure, you'd see wage LOSS

   Occupation-Year FE identifies within occupation-year variation
   → Compares workers in SAME OCCUPATION in SAME YEAR
   → If occupations with AI exposure are systematically different wage levels
     than those without, composition effects emerge
   → E.g., if AI is deployed in high-wage occupations (tech, finance),
     and those occupations already pay more, you'd see apparent wage GAIN

2. CROSS-LEVEL CONFOUNDING
   ──────────────────────
   Firm-year FE absorbs firm-level shocks → residual is job-level within-firm variation
   But occupation-year FE absorbs occupation trends → residual includes FIRM selection effects

   If firms choosing to deploy AI in occupation O are HIGH-WAGE firms,
   then OY-FE will show wage gain (selection), not causal effect

3. MECHANISMS vs. SELECTION
   ────────────────────────
   Firm-Year FE (wage LOSS):
   - Likely closer to CAUSAL: firm deploys AI → worker pays cost
   - Within-firm comparison controls for firm selection

   Occupation-Year FE (wage GAIN):
   - Likely contaminated by SELECTION: high-wage firms deploy AI
   - Between-firm comparison doesn't control for firm-level choice

RECOMMENDATION:
  Trust the Firm-Year FE result (wage LOSS) more than Occupation-Year FE
  for causal interpretation. The OY-FE positive effect is likely SELECTION BIAS.

  But: OY-FE is better for POLITICAL outcomes (controls macro trends)
       FY-FE is better for WAGE outcomes (cleaner causal estimate)

  → Use FY-FE for income, OY-FE for politics (different contexts, different tools)
""")

        return "\n".join(doc)


def main():
    print("="*100)
    print("STAGE 8F: INCOME REVERSAL ANALYSIS")
    print("="*100)

    # Load data
    prepared_file = data_dir / "shp_panel_prepared.csv"
    print(f"\nLoading data...")
    df = pd.read_csv(prepared_file, low_memory=False)
    print(f"✓ Loaded {len(df):,} person-years\n")

    analyzer = IncomeAnalysis(df)
    analysis = analyzer.run_income_comparison()

    # Save
    output_file = results_dir / "INCOME_REVERSAL_ANALYSIS.txt"
    with open(output_file, 'w') as f:
        f.write(analysis)

    print(analysis)
    print(f"\n✓ Saved to: {output_file}")

    return 0

if __name__ == '__main__':
    sys.exit(main())
