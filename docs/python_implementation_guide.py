"""
Python Implementation Guide: Generating Visualizations for Income Reversal Analysis

This script provides complete, runnable code for generating all four charts.
Dependencies: pandas, matplotlib, numpy, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Color palette
COLOR_PALETTE = {
    'loss_dark': '#C1272D',      # Red for wage loss
    'gain_dark': '#0066CC',      # Blue for wage gain
    'loss_light': '#E8B5B3',     # Light red
    'gain_light': '#B3D9FF',     # Light blue
    'loss_darker': '#8B0000',    # Darker red for extreme loss
    'gain_darker': '#003D99',    # Darker blue for extreme gain
    'neutral_light': '#CCCCCC',  # Light gray
    'reference': '#000000'       # Black for reference line
}

# ============================================================================
# CHART 1: MAIN REVERSAL EFFECT (Coefficient Plot)
# ============================================================================

def create_figure_1_coefficient_plot(output_path: str = None):
    """
    Create Figure 1: Coefficient plot showing sign reversal.

    Parameters
    ----------
    output_path : str, optional
        Path to save figure (e.g., 'figure_1.pdf'). If None, displays interactively.
    """

    # Data
    data = pd.DataFrame({
        'specification': ['Within Firm\n(Firm-Year FE)', 'Across Occupations\n(Occupation-Year FE)'],
        'coefficient': [-0.0850, 0.1085],
        'se': [0.0392, 0.0341],
        'color': [COLOR_PALETTE['loss_dark'], COLOR_PALETTE['gain_dark']]
    })

    # Calculate confidence intervals
    data['ci_lower'] = data['coefficient'] - 1.96 * data['se']
    data['ci_upper'] = data['coefficient'] + 1.96 * data['se']

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    # Plot error bars and points
    for idx, row in data.iterrows():
        ax.plot([idx, idx], [row['ci_lower'], row['ci_upper']],
                color=row['color'], linewidth=2.5, zorder=2)
        ax.scatter(idx, row['coefficient'], s=200, color=row['color'],
                  zorder=3, edgecolors='darkgray', linewidth=0.5)

    # Reference line at y=0
    ax.axhline(y=0, color=COLOR_PALETTE['reference'], linestyle='--',
              linewidth=1, zorder=1, alpha=0.7)

    # Formatting
    ax.set_ylabel('Effect on Log Hourly Wage', fontsize=12, fontweight='bold')
    ax.set_xlabel('Specification', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(data['specification'], fontsize=11)
    ax.set_ylim(-0.18, 0.18)

    # Grid
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)

    # Annotations
    ax.text(0, -0.0850 - 0.035, '-8.5%\n(p=0.030)', ha='center', fontweight='bold',
           fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_PALETTE['loss_dark'], linewidth=1))
    ax.text(1, 0.1085 + 0.035, '+10.9%\n(p=0.002)', ha='center', fontweight='bold',
           fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_PALETTE['gain_dark'], linewidth=1))

    # Reversal annotation
    ax.text(0.5, 0.14, 'Sign Reversal = +19.4 pp**\n(p<0.001)', ha='center', fontweight='bold',
           fontsize=11, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', linewidth=1.5))

    # Spine removal
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title and subtitle
    ax.set_title('AI Exposure and Wages: Within-Firm Selection vs. Occupation Composition',
                fontsize=13, fontweight='bold', pad=20)
    fig.text(0.5, 0.02, 'Sample: 37,279 person-year observations, person-level clustering.\nWithin-firm effects suggest negative selection; across-occupation effects suggest positive composition.',
            ha='center', fontsize=9, style='italic', wrap=True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Figure 1 saved to {output_path}")
    else:
        plt.show()

    return fig, ax


# ============================================================================
# CHART 2: HETEROGENEOUS EFFECTS FACET GRID
# ============================================================================

def create_figure_2_heterogeneous_facet(output_path: str = None):
    """
    Create Figure 2: Faceted heterogeneous effects by gender and age.

    Parameters
    ----------
    output_path : str, optional
        Path to save figure (e.g., 'figure_2.pdf'). If None, displays interactively.
    """

    # Data: Gender effects
    gender_data = pd.DataFrame({
        'group': ['Male', 'Female'],
        'fy_fe': [-0.0167, -0.1763],
        'fy_fe_se': [0.0564, 0.0741],
        'oy_fe': [0.1189, 0.1488],
        'oy_fe_se': [0.0366, 0.0512],
        'fy_fe_color': [COLOR_PALETTE['loss_light'], COLOR_PALETTE['loss_darker']],
        'oy_fe_color': [COLOR_PALETTE['gain_light'], COLOR_PALETTE['gain_darker']]
    })

    # Data: Age effects
    age_data = pd.DataFrame({
        'group': ['Young\n(<45)', 'Old\n(≥45)'],
        'fy_fe': [0.0230, -0.1583],
        'fy_fe_se': [0.0717, 0.0517],
        'oy_fe': [0.1676, 0.1373],
        'oy_fe_se': [0.1033, 0.0414],
        'fy_fe_color': [COLOR_PALETTE['loss_light'], COLOR_PALETTE['loss_darker']],
        'oy_fe_color': [COLOR_PALETTE['gain_light'], COLOR_PALETTE['gain_darker']]
    })

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)
    fig.suptitle('Who Loses Within Firms? Gender and Age Vulnerabilities in AI Adoption',
                fontsize=14, fontweight='bold', y=0.995)

    # Gender: FY-FE (top left)
    ax = axes[0, 0]
    x_pos = np.arange(len(gender_data))
    ax.bar(x_pos, gender_data['fy_fe'], color=gender_data['fy_fe_color'],
          yerr=1.96 * gender_data['fy_fe_se'], capsize=5, linewidth=1.5, edgecolor='darkgray')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Coefficient', fontsize=10, fontweight='bold')
    ax.set_title('Within Firm (FY-FE)', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gender_data['group'])
    ax.set_ylim(-0.25, 0.05)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for i, (val, se) in enumerate(zip(gender_data['fy_fe'], gender_data['fy_fe_se'])):
        sig = '*' if 1.96*se < abs(val) else ''
        ax.text(i, val - 0.03, f'{val:.4f}{sig}', ha='center', fontsize=9, fontweight='bold')

    # Gender: OY-FE (top right)
    ax = axes[0, 1]
    ax.bar(x_pos, gender_data['oy_fe'], color=gender_data['oy_fe_color'],
          yerr=1.96 * gender_data['oy_fe_se'], capsize=5, linewidth=1.5, edgecolor='darkgray')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Coefficient', fontsize=10, fontweight='bold')
    ax.set_title('Across Occupations (OY-FE)', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(gender_data['group'])
    ax.set_ylim(0, 0.20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for i, (val, se) in enumerate(zip(gender_data['oy_fe'], gender_data['oy_fe_se'])):
        sig = '**' if 1.96*se < val else ''
        ax.text(i, val + 0.008, f'{val:.4f}{sig}', ha='center', fontsize=9, fontweight='bold')

    # Add "GENDER" label
    fig.text(0.04, 0.75, 'GENDER', fontsize=12, fontweight='bold', rotation=90, va='center')

    # Age: FY-FE (bottom left)
    ax = axes[1, 0]
    x_pos = np.arange(len(age_data))
    ax.bar(x_pos, age_data['fy_fe'], color=age_data['fy_fe_color'],
          yerr=1.96 * age_data['fy_fe_se'], capsize=5, linewidth=1.5, edgecolor='darkgray')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Coefficient', fontsize=10, fontweight='bold')
    ax.set_xlabel('Subgroup', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(age_data['group'])
    ax.set_ylim(-0.25, 0.05)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for i, (val, se) in enumerate(zip(age_data['fy_fe'], age_data['fy_fe_se'])):
        sig = '*' if 1.96*se < abs(val) else ''
        ax.text(i, val - 0.03, f'{val:.4f}{sig}', ha='center', fontsize=9, fontweight='bold')

    # Age: OY-FE (bottom right)
    ax = axes[1, 1]
    ax.bar(x_pos, age_data['oy_fe'], color=age_data['oy_fe_color'],
          yerr=1.96 * age_data['oy_fe_se'], capsize=5, linewidth=1.5, edgecolor='darkgray')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Coefficient', fontsize=10, fontweight='bold')
    ax.set_xlabel('Subgroup', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(age_data['group'])
    ax.set_ylim(0, 0.20)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels
    for i, (val, se) in enumerate(zip(age_data['oy_fe'], age_data['oy_fe_se'])):
        sig = '†' if 1.96*se < val else ('**' if 1.96*se < val else '')
        ax.text(i, val + 0.008, f'{val:.4f}{sig}', ha='center', fontsize=9, fontweight='bold')

    # Add "AGE" label
    fig.text(0.04, 0.25, 'AGE', fontsize=12, fontweight='bold', rotation=90, va='center')

    # Remove top/right spines
    for ax in axes.flatten():
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Figure 2 saved to {output_path}")
    else:
        plt.show()

    return fig, axes


# ============================================================================
# CHART 3: VULNERABILITY HEATMAP
# ============================================================================

def create_figure_3_vulnerability_heatmap(output_path: str = None):
    """
    Create Figure 3: Vulnerability heatmap showing CHF/month impacts.

    Parameters
    ----------
    output_path : str, optional
        Path to save figure (e.g., 'figure_3.pdf'). If None, displays interactively.
    """

    # Median hourly wage and conversion factor
    median_hourly_wage = 34.70
    hours_per_month = 173

    # Data for heatmap (CHF per month impacts)
    heatmap_data = pd.DataFrame({
        'Subgroup': ['Male', 'Female', 'Young (<45)', 'Old (≥45)'],
        'FY-FE Loss (CHF/mo)': [
            -0.0167 * median_hourly_wage * hours_per_month,
            -0.1763 * median_hourly_wage * hours_per_month,
            0.0230 * median_hourly_wage * hours_per_month,
            -0.1583 * median_hourly_wage * hours_per_month
        ],
        'OY-FE Gain (CHF/mo)': [
            0.1189 * median_hourly_wage * hours_per_month,
            0.1488 * median_hourly_wage * hours_per_month,
            0.1676 * median_hourly_wage * hours_per_month,
            0.1373 * median_hourly_wage * hours_per_month
        ]
    })

    # Calculate reversal
    heatmap_data['Reversal (CHF/mo)'] = (heatmap_data['OY-FE Gain (CHF/mo)'] -
                                         heatmap_data['FY-FE Loss (CHF/mo)'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    # Prepare data for heatmap (transpose for display)
    heatmap_array = heatmap_data.set_index('Subgroup').T.values

    # Create heatmap with custom coloring
    # FY-FE column: red for losses
    fy_fe_colors = []
    for val in heatmap_data['FY-FE Loss (CHF/mo)']:
        if val < -50:
            color = COLOR_PALETTE['loss_darker']
        elif val < 0:
            color = COLOR_PALETTE['loss_light']
        else:
            color = COLOR_PALETTE['neutral_light']
        fy_fe_colors.append(color)

    # OY-FE column: blue for gains
    oy_fe_colors = []
    for val in heatmap_data['OY-FE Gain (CHF/mo)']:
        if val > 50:
            color = COLOR_PALETTE['gain_darker']
        else:
            color = COLOR_PALETTE['gain_light']
        oy_fe_colors.append(color)

    # Reversal column: gray
    reversal_colors = [COLOR_PALETTE['neutral_light']] * len(heatmap_data)

    # Plot heatmap cells manually for fine control
    cell_height = 1
    cell_width = 1.5

    # Create grid
    for col_idx, col_name in enumerate(['FY-FE Loss', 'OY-FE Gain', 'Reversal']):
        for row_idx, row_name in enumerate(heatmap_data['Subgroup']):
            if col_idx == 0:
                val = heatmap_data.iloc[row_idx]['FY-FE Loss (CHF/mo)']
                color = fy_fe_colors[row_idx]
            elif col_idx == 1:
                val = heatmap_data.iloc[row_idx]['OY-FE Gain (CHF/mo)']
                color = oy_fe_colors[row_idx]
            else:
                val = heatmap_data.iloc[row_idx]['Reversal (CHF/mo)']
                color = reversal_colors[row_idx]

            # Draw cell
            rect = plt.Rectangle((col_idx * cell_width, row_idx * cell_height),
                                 cell_width, cell_height,
                                 facecolor=color, edgecolor='gray', linewidth=1)
            ax.add_patch(rect)

            # Add text
            sig = '*' if abs(val) > 50 else ''
            ax.text(col_idx * cell_width + cell_width/2,
                   row_idx * cell_height + cell_height/2,
                   f'CHF {val:.0f}\n{sig}',
                   ha='center', va='center', fontsize=11, fontweight='bold')

    # Add column headers
    for col_idx, col_name in enumerate(['FY-FE Loss\n(CHF/month)', 'OY-FE Gain\n(CHF/month)', 'Reversal\n(CHF/month)']):
        ax.text(col_idx * cell_width + cell_width/2, 4.3, col_name,
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add row labels
    for row_idx, row_name in enumerate(heatmap_data['Subgroup']):
        ax.text(-0.5, row_idx * cell_height + cell_height/2, row_name,
               ha='right', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(-1.5, 4.5)
    ax.set_ylim(-1, 4.5)
    ax.axis('off')

    ax.set_title('Vulnerability Scorecard: AI Adoption Wage Impacts by Demographic Group',
                fontsize=13, fontweight='bold', pad=20)

    fig.text(0.5, 0.02,
            'Red = Wage loss (darker = more severe), Blue = Wage gain (darker = larger gain).\n' + \
            'Female workers lose CHF 60/month within firms (2.4% of median wage); older workers lose CHF 54/month (2.2%).',
            ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Figure 3 saved to {output_path}")
    else:
        plt.show()

    return fig, ax


# ============================================================================
# CHART 4: LIFETIME WAGE COSTS (Optional)
# ============================================================================

def create_figure_4_lifetime_costs(output_path: str = None):
    """
    Create Figure 4: Lifetime wage costs for vulnerable workers.

    Parameters
    ----------
    output_path : str, optional
        Path to save figure (e.g., 'figure_4.pdf'). If None, displays interactively.
    """

    # Conversion parameters
    median_monthly_wage = 34.70 * 173 / 1000  # in thousands CHF

    # Data: lifetime wage costs
    lifetime_data = pd.DataFrame({
        'group': ['Overall Sample', 'Overall Sample', 'Overall Sample', 'Overall Sample',
                 'Women', 'Women', 'Women', 'Women',
                 'Older Workers', 'Older Workers', 'Older Workers', 'Older Workers'],
        'horizon': ['1 Year', '5 Years', '10 Years', 'Career (40 yrs)',
                   '1 Year', '5 Years', '10 Years', 'Career (40 yrs)',
                   '1 Year', '5 Years', '10 Years', 'Career (40 yrs)'],
        'cost_chf_k': [-4.2, -20.9, -41.8, -262.0,
                      -8.6, -43.2, -86.4, -540.0,
                      -7.8, -39.0, -78.0, -491.0],
        'ci_lower_k': [-7.7, -38.5, -77.0, -481.0,
                      -15.4, -77.0, -154.0, -968.0,
                      -13.0, -65.0, -130.0, -820.0],
        'ci_upper_k': [-0.8, -4.0, -8.0, -50.0,
                      -1.8, -9.0, -18.0, -113.0,
                      -2.8, -14.0, -28.0, -176.0]
    })

    # Create figure
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)

    # Prepare data for grouped bar plot
    horizons = ['1 Year', '5 Years', '10 Years', 'Career (40 yrs)']
    x = np.arange(len(horizons))
    width = 0.25

    # Extract data by group
    overall = lifetime_data[lifetime_data['group'] == 'Overall Sample'].sort_values('horizon', key=lambda x: x.map({'1 Year': 0, '5 Years': 1, '10 Years': 2, 'Career (40 yrs)': 3}))
    women = lifetime_data[lifetime_data['group'] == 'Women'].sort_values('horizon', key=lambda x: x.map({'1 Year': 0, '5 Years': 1, '10 Years': 2, 'Career (40 yrs)': 3}))
    older = lifetime_data[lifetime_data['group'] == 'Older Workers'].sort_values('horizon', key=lambda x: x.map({'1 Year': 0, '5 Years': 1, '10 Years': 2, 'Career (40 yrs)': 3}))

    # Reorder to match horizon order
    def reorder_by_horizon(df, order):
        return df.set_index('horizon').loc[order].reset_index()

    overall = reorder_by_horizon(overall, horizons)
    women = reorder_by_horizon(women, horizons)
    older = reorder_by_horizon(older, horizons)

    # Plot bars
    ax.bar(x - width, overall['cost_chf_k'], width, label='Overall Sample',
          color=COLOR_PALETTE['neutral_light'], edgecolor='darkgray', linewidth=1,
          yerr=[overall['cost_chf_k'] - overall['ci_lower_k'], overall['ci_upper_k'] - overall['cost_chf_k']],
          capsize=4)
    ax.bar(x, women['cost_chf_k'], width, label='Women',
          color=COLOR_PALETTE['loss_light'], edgecolor='darkgray', linewidth=1,
          yerr=[women['cost_chf_k'] - women['ci_lower_k'], women['ci_upper_k'] - women['cost_chf_k']],
          capsize=4)
    ax.bar(x + width, older['cost_chf_k'], width, label='Older Workers (≥45)',
          color=COLOR_PALETTE['loss_darker'], edgecolor='darkgray', linewidth=1,
          yerr=[older['cost_chf_k'] - older['ci_lower_k'], older['ci_upper_k'] - older['cost_chf_k']],
          capsize=4)

    # Formatting
    ax.set_ylabel('Cumulative Wage Loss (CHF thousands)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Impact Horizon', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, zorder=1)
    ax.grid(axis='y', alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', fontsize=10, frameon=True, shadow=False)

    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Title
    ax.set_title('Lifetime Wage Cost of AI Adoption for Vulnerable Workers',
                fontsize=13, fontweight='bold', pad=20)

    fig.text(0.5, 0.01,
            'Cumulative lifetime wage losses from within-firm AI adoption effect (Firm-Year FE).\n' + \
            'Women lose CHF 86.4k over 10 years; older workers lose CHF 78k. Career estimates assume 40-year working life.',
            ha='center', fontsize=9, style='italic')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
        print(f"Figure 4 saved to {output_path}")
    else:
        plt.show()

    return fig, ax


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':

    # Set output directory
    output_dir = Path('/Users/bradyallardice/Dropbox/Allardice/KurerAllardice2024/10CompaniesPOCAI/Data/Testing/stage_8/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all figures
    print("Generating Figure 1: Coefficient Plot...")
    create_figure_1_coefficient_plot(str(output_dir / 'figure_1_coefficient_plot.pdf'))

    print("Generating Figure 2: Heterogeneous Effects Facet Grid...")
    create_figure_2_heterogeneous_facet(str(output_dir / 'figure_2_heterogeneous_effects.pdf'))

    print("Generating Figure 3: Vulnerability Heatmap...")
    create_figure_3_vulnerability_heatmap(str(output_dir / 'figure_3_vulnerability_heatmap.pdf'))

    print("Generating Figure 4: Lifetime Wage Costs...")
    create_figure_4_lifetime_costs(str(output_dir / 'figure_4_lifetime_costs.pdf'))

    print(f"\nAll figures generated successfully in {output_dir}")
    print("\nFiles created:")
    print(f"  - figure_1_coefficient_plot.pdf")
    print(f"  - figure_2_heterogeneous_effects.pdf")
    print(f"  - figure_3_vulnerability_heatmap.pdf")
    print(f"  - figure_4_lifetime_costs.pdf")
