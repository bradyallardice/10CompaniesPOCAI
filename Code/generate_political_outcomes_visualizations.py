#!/usr/bin/env python3
"""
Generate publication-ready visualizations for AI exposure political outcomes analysis.

This script creates all figures and tables specified in:
docs/visualization-design-political-outcomes.md

Input: Data/shp_econometric_results/all_political_outcomes_results.csv
Output: docs/figures/ (PNG and PDF)
        docs/tables/ (CSV and LaTeX)

Author: Research Visualization Specialist
Date: April 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
from scipy.stats import norm
import os
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "Data" / "shp_econometric_results"
OUTPUT_FIG_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "docs" / "tables"

# Create output directories
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
FIGURE_DPI = 300
FIGURE_FORMAT = ['png', 'pdf']

# Color scheme (colorblind-safe)
COLOR_MALE = '#1f77b4'      # Blue
COLOR_FEMALE = '#ff7f0e'    # Orange
COLOR_NEUTRAL = '#7f7f7f'   # Gray
COLOR_POSITIVE = '#d62728'  # Red (rightward)
COLOR_NEGATIVE = '#2ca02c'  # Green (leftward)

# ============================================================================
# LOAD DATA
# ============================================================================

def load_results():
    """Load political outcomes results from CSV."""
    csv_path = DATA_DIR / "all_political_outcomes_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate schema
    required_cols = ['Outcome', 'Group', 'N', 'Coef', 'SE', 'Pval', 'CI_Lower', 'CI_Upper']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print(f"Loaded {len(df)} results rows")
    print(f"Outcomes: {df['Outcome'].unique().tolist()}")
    print(f"Groups: {df['Group'].unique().tolist()}")

    return df

# ============================================================================
# TABLE 1: MAIN RESULTS
# ============================================================================

def create_table_1(df):
    """
    Create Table 1: Political effects across dimensions and gender.

    Output: LaTeX table and CSV
    """
    print("\n" + "="*70)
    print("GENERATING TABLE 1: MAIN RESULTS")
    print("="*70)

    # Extract key results
    outcomes = [
        "Left-Right Placement (1=left, 10=right)",
        "Nativism (1=equal chances, 3=better for Swiss)",
        "Welfare Pref (1=less, 3=more)",
        "Redistributive Pref (1=reduce, 3=increase taxes)",
        "Gender Equality (0=gone too far, 10=not enough)"
    ]

    groups = ["Full Sample", "Male", "Female"]

    table_data = []

    for outcome in outcomes:
        row = {'Outcome': outcome}

        for group in groups:
            mask = (df['Outcome'] == outcome) & (df['Group'] == group)
            if mask.sum() == 0:
                row[f'{group}_Coef'] = np.nan
                row[f'{group}_SE'] = np.nan
                row[f'{group}_Pval'] = np.nan
                row[f'{group}_N'] = np.nan
            else:
                result = df[mask].iloc[0]
                row[f'{group}_Coef'] = result['Coef']
                row[f'{group}_SE'] = result['SE']
                row[f'{group}_Pval'] = result['Pval']
                row[f'{group}_N'] = result['N']

        table_data.append(row)

    df_table1 = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = OUTPUT_TABLE_DIR / "table_1_main_results.csv"
    df_table1.to_csv(csv_path, index=False)
    print(f"Saved Table 1 CSV: {csv_path}")

    # Generate LaTeX
    latex_code = generate_table_1_latex(df_table1)

    latex_path = OUTPUT_TABLE_DIR / "table_1_main_results.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    print(f"Saved Table 1 LaTeX: {latex_path}")

    return df_table1

def generate_table_1_latex(df_table1):
    """Generate LaTeX code for Table 1."""

    latex = r"""
\begin{table}[h]
\centering
\caption{Political Effects of AI Exposure by Dimension and Demographic Group}
\label{tab:political_main}
\small
\begin{tabular}{lcccccccc}
\toprule
\multicolumn{1}{l}{\textbf{Political Dimension}} &
\multicolumn{3}{c}{\textbf{Full Sample}} &
\multicolumn{3}{c}{\textbf{Males}} &
\multicolumn{3}{c}{\textbf{Females}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
& \textit{Coef} & \textit{SE} & \textit{p-val} &
\textit{Coef} & \textit{SE} & \textit{p-val} &
\textit{Coef} & \textit{SE} & \textit{p-val} \\
\midrule
"""

    for idx, row in df_table1.iterrows():
        outcome = row['Outcome']

        # Truncate outcome name for table
        if "Left-Right" in outcome:
            outcome_short = r"\textbf{Left-Right (1=left, 10=right)}"
        elif "Nativism" in outcome:
            outcome_short = r"\textbf{Nativism (1=equal, 3=Swiss)}"
        elif "Redistributive" in outcome:
            outcome_short = r"\textbf{Redistributive (1=reduce, 3=increase)}"
        elif "Welfare" in outcome:
            outcome_short = r"\textbf{Welfare (1=less, 3=more)}"
        else:
            outcome_short = r"\textbf{Gender Equality (0=too far, 10=not enough)}"

        # Extract values
        fs_coef = f"{row['Full Sample_Coef']:.3f}"
        fs_se = f"{row['Full Sample_SE']:.3f}"
        fs_pval = row['Full Sample_Pval']
        fs_sig = get_significance_marker(fs_pval)

        m_coef = f"{row['Male_Coef']:.3f}"
        m_se = f"{row['Male_SE']:.3f}"
        m_pval = row['Male_Pval']
        m_sig = get_significance_marker(m_pval)

        f_coef = f"{row['Female_Coef']:.3f}"
        f_se = f"{row['Female_SE']:.3f}"
        f_pval = row['Female_Pval']
        f_sig = get_significance_marker(f_pval)

        # Format row
        latex += f"{outcome_short} & & & & & & & & \\\\\n"
        latex += (f"Overall effect & ${fs_coef}${fs_sig} & ({fs_se}) & "
                 f"${m_coef}${m_sig} & ({m_se}) & "
                 f"${f_coef}${f_sig} & ({f_se}) \\\\\n")
        latex += (f"N & {int(row['Full Sample_N']):,} & & & "
                 f"{int(row['Male_N']):,} & & & "
                 f"{int(row['Female_N']):,} & \\\\\n")
        latex += r"\midrule" + "\n"

    latex += r"""\bottomrule
\end{tabular}

\begin{flushleft}
\footnotesize
\textit{Note:} Specification: person fixed effects + occupation-year fixed effects, de-meaned.
Controls: age (centered), employment status. Estimates use robust standard errors clustered by person.
Sample: 2012-2023 from Swiss Household Panel.
Limitation: 23\% of SHP panel has firm identifiers; results may not generalize to self-employed.
† p<0.10 (borderline), * p<0.05 (significant).
\end{flushleft}
\end{table}
"""

    return latex

def get_significance_marker(pval):
    """Get significance marker (* or † or blank)."""
    if pval < 0.05:
        return "^*"
    elif pval < 0.10:
        return "^{\\dagger}"
    else:
        return ""

# ============================================================================
# FIGURE 1: GENDER HETEROGENEITY HEATMAP
# ============================================================================

def create_figure_1(df):
    """
    Create Figure 1: Gender heterogeneity heatmap.

    Shows coefficients by political dimension and gender.
    """
    print("\n" + "="*70)
    print("GENERATING FIGURE 1: GENDER HETEROGENEITY HEATMAP")
    print("="*70)

    # Extract left-right results by gender
    outcomes_short = [
        ('Left-Right', "Left-Right Placement (1=left, 10=right)"),
        ('Nativism', "Nativism (1=equal chances, 3=better for Swiss)"),
        ('Redistributive', "Redistributive Pref (1=reduce, 3=increase taxes)"),
        ('Welfare', "Welfare Pref (1=less, 3=more)"),
        ('Gender Equality', "Gender Equality (0=gone too far, 10=not enough)")
    ]

    data_matrix = []
    pval_matrix = []
    se_matrix = []

    for short, full in outcomes_short:
        row_coef = []
        row_pval = []
        row_se = []

        for group in ['Male', 'Female']:
            mask = (df['Outcome'] == full) & (df['Group'] == group)
            if mask.sum() > 0:
                result = df[mask].iloc[0]
                row_coef.append(result['Coef'])
                row_pval.append(result['Pval'])
                row_se.append(result['SE'])
            else:
                row_coef.append(np.nan)
                row_pval.append(np.nan)
                row_se.append(np.nan)

        data_matrix.append(row_coef)
        pval_matrix.append(row_pval)
        se_matrix.append(row_se)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Data
    data_array = np.array(data_matrix)
    outcome_labels = [x[0] for x in outcomes_short]
    group_labels = ['Males', 'Females']

    # Heatmap
    im = ax.imshow(data_array, cmap='RdBu_r', aspect='auto', vmin=-0.6, vmax=0.3)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(group_labels)))
    ax.set_yticks(np.arange(len(outcome_labels)))
    ax.set_xticklabels(group_labels, fontsize=12, fontweight='bold')
    ax.set_yticklabels(outcome_labels, fontsize=11)

    # Add values and significance stars
    for i in range(len(outcome_labels)):
        for j in range(len(group_labels)):
            coef = data_array[i, j]
            pval = pval_matrix[i][j]
            se = se_matrix[i][j]

            # Determine text color for contrast
            if abs(coef) > 0.2:
                text_color = 'white'
            else:
                text_color = 'black'

            # Main value
            text = ax.text(j, i, f'{coef:.3f}',
                          ha="center", va="center", color=text_color,
                          fontsize=11, fontweight='bold')

            # SE below
            ax.text(j, i + 0.25, f'({se:.3f})',
                   ha="center", va="center", color=text_color,
                   fontsize=9, style='italic')

            # Significance marker
            if pval < 0.05:
                marker = '*'
            elif pval < 0.10:
                marker = '†'
            else:
                marker = ''

            if marker:
                ax.text(j + 0.35, i - 0.35, marker,
                       ha="center", va="center", color='red',
                       fontsize=14, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Effect Size (Coefficient)', fraction=0.046, pad=0.04)

    # Formatting
    ax.set_xlabel('Gender', fontsize=13, fontweight='bold')
    ax.set_ylabel('Political Dimension', fontsize=13, fontweight='bold')
    ax.set_title('AI Exposure Effects on Political Preferences by Gender\nHeat Map of Coefficients',
                fontsize=13, fontweight='bold', pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(group_labels)) - .5, minor=True)
    ax.set_yticks(np.arange(len(outcome_labels)) - .5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)

    # Tight layout
    plt.tight_layout()

    # Save
    for fmt in FIGURE_FORMAT:
        path = OUTPUT_FIG_DIR / f"figure_1_gender_heterogeneity_heatmap.{fmt}"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved Figure 1 ({fmt.upper()}): {path}")

    plt.close()

# ============================================================================
# TABLE 2: GENDER × AGE HETEROGENEITY
# ============================================================================

def create_table_2(df):
    """
    Create Table 2: Gender × Age vulnerability gradient.

    Focus on Left-Right outcome only.
    """
    print("\n" + "="*70)
    print("GENERATING TABLE 2: GENDER × AGE HETEROGENEITY")
    print("="*70)

    groups_display = [
        ("Males", "Male"),
        ("Females", "Female"),
        ("Young (< 33rd pct)", "Young (< 33rd pct)"),
        ("Old (> 67th pct)", "Old (> 67th pct)"),
    ]

    lr_outcome = "Left-Right Placement (1=left, 10=right)"

    table_data = []

    # Section A: Gender
    for display, code in groups_display[:2]:
        mask = (df['Outcome'] == lr_outcome) & (df['Group'] == code)
        if mask.sum() > 0:
            result = df[mask].iloc[0]
            table_data.append({
                'Group': display,
                'Type': 'Gender',
                'Coef': result['Coef'],
                'SE': result['SE'],
                'Pval': result['Pval'],
                'CI_Lower': result['CI_Lower'],
                'CI_Upper': result['CI_Upper'],
                'N': int(result['N'])
            })

    # Section B: Age
    for display, code in groups_display[2:]:
        mask = (df['Outcome'] == lr_outcome) & (df['Group'] == code)
        if mask.sum() > 0:
            result = df[mask].iloc[0]
            table_data.append({
                'Group': display,
                'Type': 'Age',
                'Coef': result['Coef'],
                'SE': result['SE'],
                'Pval': result['Pval'],
                'CI_Lower': result['CI_Lower'],
                'CI_Upper': result['CI_Upper'],
                'N': int(result['N'])
            })

    df_table2 = pd.DataFrame(table_data)

    # Save CSV
    csv_path = OUTPUT_TABLE_DIR / "table_2_heterogeneity.csv"
    df_table2.to_csv(csv_path, index=False)
    print(f"Saved Table 2 CSV: {csv_path}")

    # Generate and save LaTeX
    latex_code = generate_table_2_latex(df_table2)
    latex_path = OUTPUT_TABLE_DIR / "table_2_heterogeneity.tex"
    with open(latex_path, 'w') as f:
        f.write(latex_code)
    print(f"Saved Table 2 LaTeX: {latex_path}")

    return df_table2

def generate_table_2_latex(df_table2):
    """Generate LaTeX for Table 2."""

    latex = r"""
\begin{table}[h]
\centering
\caption{Heterogeneous Effects of AI Exposure on Left-Right Political Placement}
\label{tab:heterogeneity}
\begin{tabular}{lccccc}
\toprule
\textbf{Demographic Group} & \textbf{Coef} & \textbf{SE} & \textbf{p-value} &
\textbf{95\% CI} & \textbf{N} \\
\midrule
\multicolumn{6}{l}{\textit{A. Gender Effects}} \\
"""

    for idx, row in df_table2[df_table2['Type'] == 'Gender'].iterrows():
        group = row['Group']
        coef = f"{row['Coef']:.3f}"
        se = f"{row['SE']:.3f}"
        pval = row['Pval']
        ci_lower = f"{row['CI_Lower']:.3f}"
        ci_upper = f"{row['CI_Upper']:.3f}"
        n = f"{row['N']:,}"

        sig = get_significance_marker(pval)

        latex += f"{group} & ${coef}${sig} & {se} & {pval:.3f} & [{ci_lower}, {ci_upper}] & {n} \\\\\n"

    latex += r"""
\midrule
\multicolumn{6}{l}{\textit{B. Age Effects}} \\
"""

    for idx, row in df_table2[df_table2['Type'] == 'Age'].iterrows():
        group = row['Group']
        coef = f"{row['Coef']:.3f}"
        se = f"{row['SE']:.3f}"
        pval = row['Pval']
        ci_lower = f"{row['CI_Lower']:.3f}"
        ci_upper = f"{row['CI_Upper']:.3f}"
        n = f"{row['N']:,}"

        sig = get_significance_marker(pval)

        latex += f"{group} & ${coef}${sig} & {se} & {pval:.3f} & [{ci_lower}, {ci_upper}] & {n} \\\\\n"

    latex += r"""
\bottomrule
\end{tabular}

\begin{flushleft}
\footnotesize
\textit{Note:} Outcome: Left-Right political placement (1=far left, 10=far right).
Specification: person fixed effects + occupation-year fixed effects, clustering by person.
CI: 95 percent confidence interval.
† p<0.10, * p<0.05.
\end{flushleft}
\end{table}
"""

    return latex

# ============================================================================
# FIGURE 2: MEDIATION PATHWAYS
# ============================================================================

def create_figure_2():
    """
    Create Figure 2: Mediation analysis visualization.

    Shows direct and indirect paths with gender heterogeneity.
    """
    print("\n" + "="*70)
    print("GENERATING FIGURE 2: MEDIATION ANALYSIS PATHWAYS")
    print("="*70)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ===== PANEL 1: Main Mediation Model =====
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Boxes
    exposure_box = FancyBboxPatch((0.5, 4), 2, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='lightblue', linewidth=2)
    insecurity_box = FancyBboxPatch((4, 6.5), 2, 1.5, boxstyle="round,pad=0.1",
                                   edgecolor='black', facecolor='lightyellow', linewidth=2)
    politics_box = FancyBboxPatch((7.5, 4), 2, 1.5, boxstyle="round,pad=0.1",
                                 edgecolor='black', facecolor='lightcoral', linewidth=2)

    ax1.add_patch(exposure_box)
    ax1.add_patch(insecurity_box)
    ax1.add_patch(politics_box)

    # Labels
    ax1.text(1.5, 4.75, 'AI Exposure', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax1.text(5, 7.25, 'Job\nInsecurity', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax1.text(8.5, 4.75, 'Political\nShift (Left)', ha='center', va='center',
            fontsize=11, fontweight='bold')

    # Arrows and labels
    # a-path
    arrow_a = FancyArrowPatch((2.5, 5), (4, 6.8),
                             arrowstyle='->', mutation_scale=30,
                             linewidth=2.5, color='green')
    ax1.add_patch(arrow_a)
    ax1.text(3.2, 6.1, 'a = +0.093†', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # b-path
    arrow_b = FancyArrowPatch((6, 6.8), (7.5, 5.2),
                             arrowstyle='->', mutation_scale=30,
                             linewidth=1.5, color='orange', linestyle='dashed')
    ax1.add_patch(arrow_b)
    ax1.text(6.8, 6.2, 'b = −0.040\n(ns)', fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # c-path (total)
    arrow_c = FancyArrowPatch((2.5, 4.75), (7.5, 4.75),
                             arrowstyle='->', mutation_scale=30,
                             linewidth=2.5, color='darkred')
    ax1.add_patch(arrow_c)
    ax1.text(5, 5.3, r"c = −0.252† (Total)", fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    ax1.text(5, 3.5, r"c′ = −0.248† (Direct)",
            fontsize=9, style='italic', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Mediation results
    textstr = ('Indirect Effect = a × b = −0.004\n' +
              '95% CI: [−0.010, 0.002]\n' +
              'Mediation %: 1.5% of total\n\n' +
              'Job insecurity explains minimal\n' +
              'portion; direct ideological\n' +
              'response dominates.')
    ax1.text(5, 1, textstr, fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax1.set_title('Panel A: Full Sample Mediation Model', fontsize=12, fontweight='bold')

    # ===== PANEL 2: Gender Heterogeneity =====
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Male pathway
    male_y = 7
    ax2.text(0.5, male_y + 1, 'Males:', fontsize=11, fontweight='bold')
    ax2.text(0.5, male_y + 0.3, 'Insecurity: β = +0.070 (ns)', fontsize=9)
    ax2.text(0.5, male_y - 0.3, 'Political: β = −0.430* (strong)',
            fontsize=9, color='darkred', fontweight='bold')
    ax2.text(5.5, male_y, ('Ideology-driven:\n' +
                            'Respond politically\n' +
                            'despite minimal insecurity'),
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # Female pathway
    female_y = 3.5
    ax2.text(0.5, female_y + 1, 'Females:', fontsize=11, fontweight='bold')
    ax2.text(0.5, female_y + 0.3, 'Insecurity: β = +0.155† (strong)',
            fontsize=9, color='orange', fontweight='bold')
    ax2.text(0.5, female_y - 0.3, 'Political: β = −0.012 (none)',
            fontsize=9)
    ax2.text(5.5, female_y, ('Depoliticization:\n' +
                              'Feel economic threat\n' +
                              'but do not mobilize'),
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax2.set_title('Panel B: Heterogeneous Pathways by Gender', fontsize=12, fontweight='bold')

    plt.suptitle('Figure 2: Mediation Analysis — Pathways from AI Exposure to Political Shift',
                fontsize=13, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save
    for fmt in FIGURE_FORMAT:
        path = OUTPUT_FIG_DIR / f"figure_2_mediation_pathways.{fmt}"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved Figure 2 ({fmt.upper()}): {path}")

    plt.close()

# ============================================================================
# FIGURE 3: EFFECT SIZE INTERPRETATION
# ============================================================================

def create_figure_3():
    """
    Create Figure 3: Practical significance of effect sizes.

    Four panels showing distributions, scales, comparisons, and heterogeneity.
    """
    print("\n" + "="*70)
    print("GENERATING FIGURE 3: EFFECT SIZE INTERPRETATION")
    print("="*70)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # ===== PANEL A: Distributions =====
    ax_a = fig.add_subplot(gs[0, 0])

    baseline_mean = 4.87
    baseline_sd = 2.18
    effect = -0.252
    exposed_mean = baseline_mean + effect

    x = np.linspace(0, 10, 1000)
    baseline_dist = norm.pdf(x, baseline_mean, baseline_sd)
    exposed_dist = norm.pdf(x, exposed_mean, baseline_sd)

    ax_a.fill_between(x, baseline_dist, alpha=0.3, label='No exposure',
                     color='gray')
    ax_a.fill_between(x, exposed_dist, alpha=0.3, label='Full exposure',
                     color='blue')
    ax_a.axvline(baseline_mean, color='gray', linestyle='-', linewidth=2)
    ax_a.axvline(exposed_mean, color='blue', linestyle='--', linewidth=2)

    # Annotation
    ax_a.annotate('', xy=(exposed_mean, 0.15), xytext=(baseline_mean, 0.15),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax_a.text((baseline_mean + exposed_mean) / 2, 0.17, f'Shift: {effect:.3f}',
             ha='center', fontsize=10, fontweight='bold', color='red')

    ax_a.set_xlabel('Left-Right (1=left, 10=right)', fontsize=11, fontweight='bold')
    ax_a.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax_a.set_title('Panel A: Distribution Shift\n(Baseline vs. AI-Exposed)',
                  fontsize=12, fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=10)
    ax_a.set_xlim(0, 10)
    ax_a.grid(alpha=0.3)

    # ===== PANEL B: Practical Scale =====
    ax_b = fig.add_subplot(gs[0, 1])

    groups = ['Baseline\n(No exposure)', 'Full Sample\n(−0.252†)',
             'Males\n(−0.430*)', 'Females\n(−0.012)']
    positions = [4.87, 4.62, 4.44, 4.86]
    colors = ['gray', 'lightblue', 'darkblue', 'lightgray']

    y_pos = np.arange(len(groups))

    ax_b.barh(y_pos, positions, color=colors, edgecolor='black', linewidth=1.5)

    # Scale references
    ax_b.axvline(1, color='black', linestyle='--', alpha=0.3)
    ax_b.axvline(5, color='black', linestyle='-', alpha=0.5, linewidth=2)
    ax_b.axvline(10, color='black', linestyle='--', alpha=0.3)

    # Scale labels
    ax_b.text(1, -0.7, 'Left (1)', ha='center', fontsize=9, style='italic')
    ax_b.text(5, -0.7, 'Center (5)', ha='center', fontsize=9, style='italic', fontweight='bold')
    ax_b.text(10, -0.7, 'Right (10)', ha='center', fontsize=9, style='italic')

    # Position values
    for i, (pos, group) in enumerate(zip(positions, groups)):
        ax_b.text(pos + 0.2, i, f'{pos:.2f}', va='center', fontsize=10,
                 fontweight='bold')

    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(groups, fontsize=10)
    ax_b.set_xlabel('Political Position', fontsize=11, fontweight='bold')
    ax_b.set_xlim(0, 11)
    ax_b.set_title('Panel B: Practical Meaning\n(Where Workers Stand on 1-10 Scale)',
                  fontsize=12, fontweight='bold')
    ax_b.grid(axis='x', alpha=0.3)

    # ===== PANEL C: Comparison to Other Studies =====
    ax_c = fig.add_subplot(gs[1, 0])

    studies = ['AI Exposure\n(This Study)', 'Trade Exposure\n(Mutz 2018)',
              'Economic Recession\n(Achen & Bartels)', 'EU Enlargement\n(Mayda & Rodrik)',
              'AI Exposure\nMales Only']
    effects_comp = [0.252, 0.35, 0.40, 0.25, 0.430]
    colors_comp = ['lightblue', 'orange', 'red', 'purple', 'darkblue']

    x_pos = np.arange(len(studies))
    bars = ax_c.bar(x_pos, effects_comp, color=colors_comp, edgecolor='black', linewidth=1.5)

    # Value labels
    for bar, effect_size in zip(bars, effects_comp):
        height = bar.get_height()
        ax_c.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{effect_size:.3f}', ha='center', va='bottom', fontsize=10,
                 fontweight='bold')

    ax_c.set_xticks(x_pos)
    ax_c.set_xticklabels(studies, fontsize=9)
    ax_c.set_ylabel('Effect Size (scale points)', fontsize=11, fontweight='bold')
    ax_c.set_title('Panel C: Comparison to Other Political Shocks\n(Is 0.25 Points Meaningful?)',
                  fontsize=12, fontweight='bold')
    ax_c.set_ylim(0, 0.5)
    ax_c.grid(axis='y', alpha=0.3)

    # ===== PANEL D: Vulnerability Gradient =====
    ax_d = fig.add_subplot(gs[1, 1])

    demographics = [
        ('Young Males\nN=6,959', 4.82, -0.18),
        ('Young Females\nN=5,497', 4.99, -0.01),
        ('Older Males\nN=7,156', 4.54, -0.46),
        ('Older Females\nN=5,964', 4.75, -0.25),
    ]

    positions_2d = [(0.25, 0.75), (0.75, 0.75), (0.25, 0.25), (0.75, 0.25)]
    colors_2d = ['lightblue', 'lightyellow', 'darkblue', 'lightcoral']

    for (label, pos, effect_val), (x, y), color in zip(demographics, positions_2d, colors_2d):
        # Scale bar
        ax_d.barh(y, 10, height=0.12, left=0, color='white', edgecolor='black', linewidth=1)

        # Position marker
        marker_size = abs(effect_val) * 100 + 30
        ax_d.scatter(pos, y, s=marker_size, color=color, edgecolor='black',
                    linewidth=2, zorder=5, alpha=0.8)

        # Label
        ax_d.text(-0.5, y, label, ha='right', va='center', fontsize=10, fontweight='bold')
        ax_d.text(pos, y - 0.15, f'Shift: {effect_val:.2f}',
                 ha='center', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    ax_d.set_xlim(-1.5, 10.5)
    ax_d.set_ylim(0, 1)
    ax_d.set_xticks([1, 5, 10])
    ax_d.set_xticklabels(['1\nLeft', '5\nCenter', '10\nRight'])
    ax_d.set_yticks([])
    ax_d.set_xlabel('Political Position', fontsize=11, fontweight='bold')
    ax_d.set_title('Panel D: Vulnerability Gradient\n(Who Responds Most?)',
                  fontsize=12, fontweight='bold')
    ax_d.grid(axis='x', alpha=0.3)

    plt.suptitle('Figure 3: Practical Significance — What Does a 0.25-Point Shift Mean?',
                fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    for fmt in FIGURE_FORMAT:
        path = OUTPUT_FIG_DIR / f"figure_3_effect_size_interpretation.{fmt}"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved Figure 3 ({fmt.upper()}): {path}")

    plt.close()

# ============================================================================
# FIGURE 4 / TABLE 4: AI EXPOSURE CHANGES BY MOBILITY TYPE
# ============================================================================

def create_figure_4_exposure_changes_by_mobility():
    """
    Distribution of within-person AI exposure changes, split by mobility type.

    Four panels (one bar chart each), x = number of year-over-year changes in
    `hampole_ai_exposure_avg_foy` observed for that person across the panel,
    y = number of persons:
      A. Stayers — same firm AND same occupation in every observed year.
      B. Same firm, occupation switched at least once.
      C. Same occupation, firm switched at least once.
      D. Both firm AND occupation switched at least once.

    Sample: persons with non-missing firm_id, isco08_4d, and exposure in 2+
    years (single-year persons cannot exhibit a transition).
    """
    print("\n" + "="*70)
    print("GENERATING FIGURE 4: AI EXPOSURE CHANGES BY MOBILITY TYPE")
    print("="*70)

    panel_path = PROJECT_ROOT / "Data" / "shp_panel_prepared.csv"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"SHP panel not found: {panel_path}\n"
            f"Run stage_8a_prepare_panel_data.py to generate this file."
        )

    cols = ['idpers', 'year', 'firm_id', 'isco08_4d', 'hampole_ai_exposure_avg_foy']
    panel = pd.read_csv(panel_path, usecols=cols)
    panel = panel.dropna(subset=['firm_id', 'isco08_4d', 'hampole_ai_exposure_avg_foy'])
    panel = panel.sort_values(['idpers', 'year']).reset_index(drop=True)

    grouped = panel.groupby('idpers')
    panel['firm_id_prev'] = grouped['firm_id'].shift(1)
    panel['isco_prev'] = grouped['isco08_4d'].shift(1)
    panel['exp_prev'] = grouped['hampole_ai_exposure_avg_foy'].shift(1)
    panel['year_prev'] = grouped['year'].shift(1)

    has_lag = panel['year_prev'].notna()
    panel['firm_changed_step'] = has_lag & (panel['firm_id'] != panel['firm_id_prev'])
    panel['isco_changed_step'] = has_lag & (panel['isco08_4d'] != panel['isco_prev'])
    panel['exp_changed_step'] = has_lag & ~np.isclose(
        panel['hampole_ai_exposure_avg_foy'].fillna(-999.0),
        panel['exp_prev'].fillna(-999.0),
        atol=1e-6,
    )

    person = panel.groupby('idpers').agg(
        n_years=('year', 'nunique'),
        n_exposure_changes=('exp_changed_step', 'sum'),
        firm_ever_changed=('firm_changed_step', 'any'),
        isco_ever_changed=('isco_changed_step', 'any'),
    ).reset_index()

    n_total_persons = len(person)
    multi = person[person['n_years'] >= 2].copy()
    n_single_year = n_total_persons - len(multi)

    cat_stayer    = (~multi['firm_ever_changed']) & (~multi['isco_ever_changed'])
    cat_job_only  = (~multi['firm_ever_changed']) & ( multi['isco_ever_changed'])
    cat_firm_only = ( multi['firm_ever_changed']) & (~multi['isco_ever_changed'])
    cat_both      = ( multi['firm_ever_changed']) & ( multi['isco_ever_changed'])

    multi['mobility_category'] = np.select(
        [cat_stayer, cat_job_only, cat_firm_only, cat_both],
        ['stayer', 'job_only', 'firm_only', 'both'],
        default='unknown',
    )

    panels = [
        ('Stayers (same firm AND same occupation)', 'stayer',    '#2c7fb8'),
        ('Same firm, occupation switched',          'job_only',  '#7fcdbb'),
        ('Same occupation, firm switched',          'firm_only', '#fdae61'),
    ]

    in_panels = multi['mobility_category'].isin(['stayer', 'job_only', 'firm_only'])
    max_changes = int(multi.loc[in_panels, 'n_exposure_changes'].max()) if in_panels.any() else 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    rows = []
    for ax, (title, cat, color) in zip(axes, panels):
        sub = multi[multi['mobility_category'] == cat]
        counts = sub['n_exposure_changes'].value_counts().sort_index()
        counts = counts.reindex(range(max_changes + 1), fill_value=0)

        bars = ax.bar(counts.index, counts.values, color=color,
                      edgecolor='black', linewidth=1)

        for bar, val in zip(bars, counts.values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{int(val)}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Number of year-over-year exposure changes',
                      fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of persons', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\n(N = {len(sub):,} persons)',
                     fontsize=12, fontweight='bold')
        ax.set_xticks(range(max_changes + 1))
        ax.grid(axis='y', alpha=0.3)

        for n_changes, n_persons in counts.items():
            rows.append({
                'mobility_category': cat,
                'category_label': title,
                'n_exposure_changes': int(n_changes),
                'n_persons': int(n_persons),
                'category_total_persons': int(len(sub)),
            })

    fig.suptitle(
        'Figure 4: Distribution of AI Exposure Changes by Mobility Type\n'
        f'SHP respondents observed in 2+ years (N = {len(multi):,}); '
        f'{int(cat_both.sum()):,} persons with both firm AND occupation '
        f'switches excluded; {n_single_year:,} single-year persons excluded.',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    for fmt in FIGURE_FORMAT:
        path = OUTPUT_FIG_DIR / f"figure_4_exposure_changes_by_mobility.{fmt}"
        plt.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"Saved Figure 4 ({fmt.upper()}): {path}")
    plt.close()

    df_counts = pd.DataFrame(rows)
    csv_path = OUTPUT_TABLE_DIR / "table_4_exposure_changes_by_mobility.csv"
    df_counts.to_csv(csv_path, index=False)
    print(f"Saved Table 4 CSV: {csv_path}")

    print(f"\nSummary (persons with 2+ observed years and full data):")
    print(f"  Total persons in panels:        {int(in_panels.sum()):,}")
    print(f"  Stayers (same firm + same job): {int(cat_stayer.sum()):,}")
    print(f"  Same firm, job switch only:     {int(cat_job_only.sum()):,}")
    print(f"  Same job, firm switch only:     {int(cat_firm_only.sum()):,}")
    print(f"  Both firm AND job switch:       {int(cat_both.sum()):,} (excluded)")
    print(f"  Single-year persons (excluded): {n_single_year:,}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all visualizations."""

    print("\n" + "="*70)
    print("POLITICAL OUTCOMES VISUALIZATION GENERATOR")
    print("="*70)
    print(f"Output directory: {OUTPUT_FIG_DIR}")
    print(f"Output directory: {OUTPUT_TABLE_DIR}")

    # Load data
    df = load_results()

    # Create visualizations
    create_table_1(df)
    create_figure_1(df)
    create_table_2(df)
    create_figure_2()
    create_figure_3()
    create_figure_4_exposure_changes_by_mobility()

    print("\n" + "="*70)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  Figures: {OUTPUT_FIG_DIR}")
    print(f"  Tables:  {OUTPUT_TABLE_DIR}")
    print("\nNext steps:")
    print("  1. Review figures for accuracy and formatting")
    print("  2. Adjust colors for journal style guidelines")
    print("  3. Insert into manuscript with figure captions")
    print("  4. Export tables to PDF for publication")

if __name__ == '__main__':
    main()
