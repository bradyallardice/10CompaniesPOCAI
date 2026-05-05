# Visualization Specifications for Income Reversal Analysis

## Overview
This document provides detailed specifications for visualizing the AI exposure income reversal phenomenon. The core story: AI exposure **reduces wages within firms** (negative selection effect) but **increases wages across occupations** (positive composition effect), creating a sign reversal that reveals fundamental heterogeneity in how workers are sorted into AI-exposed roles.

---

## Chart 1: Main Reversal Effect (Coefficient Plot)

### Purpose
Visualize the central finding: the sign reversal between FY-FE and OY-FE specifications.

### Type
Coefficient plot with error bars (95% CI)

### Layout

```
TITLE: "AI Exposure and Wages: Within-Firm Selection vs. Occupation Composition"

Y-AXIS LABEL: "Effect on Log Hourly Wage"
Y-AXIS SCALE: -0.15 to +0.15 (log points)
Y-AXIS GRID: Light gray, major ticks every 0.05

X-AXIS: Two groups
  - LEFT: "Within Firm (FY-FE)"
  - RIGHT: "Across Occupations (OY-FE)"

DATA POINTS (with 95% CI error bars):
  - LEFT (FY-FE): β = -0.0850, SE = 0.0392
    * Point: Positioned at -0.0850
    * Error bar: From (-0.162) to (-0.008)
    * Color: Dark red (#C1272D)
    * Marker: Circle, size = 120pt
    * Line width: 2.5pt
    
  - RIGHT (OY-FE): β = +0.1085, SE = 0.0341
    * Point: Positioned at +0.1085
    * Error bar: From (+0.041) to (+0.176)
    * Color: Dark blue (#0066CC)
    * Marker: Circle, size = 120pt
    * Line width: 2.5pt

REFERENCE LINE: Horizontal line at y=0 (black, dashed, width=1pt)

ANNOTATIONS:
  - Above FY-FE point: "-8.5%"
  - Above OY-FE point: "+10.9%"
  - Below x-axis at center: "Reversal = +19.4 pp*** (p<0.01)"
  - Font size: 11pt, bold for reversal annotation

LEGEND (optional):
  - "Firm-Year FE: Controls for composition within firms"
  - "Occupation-Year FE: Captures shifts across occupations"

SUBTITLE (smaller font, ~9pt):
"Sample: 37,279 person-year observations, person-level clustering.
Within-firm effects suggest negative selection (high-AI firms pay lower wages).
Across-occupation effects suggest positive composition (workers in AI-exposed occupations earn more)."
```

### Data Structure for Generation

```
Group,Coefficient,SE,Lower_CI,Upper_CI,Color
"Within Firm (FY-FE)",-0.0850,0.0392,-0.162,-0.008,"#C1272D"
"Across Occupations (OY-FE)",0.1085,0.0341,0.041,0.176,"#0066CC"
```

### Visual Encoding
- **Color**: Red (loss) vs. Blue (gain) — intuitive for wage impacts
- **Marker size**: Large circles (120pt) to emphasize statistical estimates
- **Error bar width**: 2.5pt, proportional to CI size
- **Spacing**: Equal horizontal spacing between specifications

### Use Case
- **Academic papers**: Main results section (Figure 1 or 2)
- **Presentations**: Opening slide to establish the reversal phenomenon
- **Policy briefs**: Front-and-center to show distributional complexity

---

## Chart 2: Heterogeneous Effects Facet Grid (Gender & Age)

### Purpose
Show how the reversal pattern differs for vulnerable populations (women, older workers).

### Type
Faceted bar chart (2×2 grid): 2 panels (Gender, Age) × 2 sub-panels each (FY-FE, OY-FE)

### Layout

```
TITLE: "Who Loses Within Firms? Gender and Age Vulnerabilities in AI Adoption"

GRID LAYOUT:
┌─────────────────────────────────────────────────────────┐
│                    PANEL 1: GENDER                      │
├────────────────────────────┬────────────────────────────┤
│  FY-FE (Within Firm)       │  OY-FE (Across Occupations)│
│  Left: Male                │  Left: Male                │
│  Right: Female             │  Right: Female             │
├────────────────────────────┴────────────────────────────┤
│                     PANEL 2: AGE                        │
├────────────────────────────┬────────────────────────────┤
│  FY-FE (Within Firm)       │  OY-FE (Across Occupations)│
│  Left: Young (<45)         │  Left: Young (<45)         │
│  Right: Old (≥45)          │  Right: Old (≥45)          │
└────────────────────────────┴────────────────────────────┘

EACH PANEL (FY-FE):
  Y-AXIS: Log points from -0.25 to +0.05
  X-AXIS: Subgroup (e.g., Male | Female)
  Bars: Grouped by subgroup, with error bars
  Bar colors:
    - Male/Young: Light red (#E8B5B3)
    - Female/Old: Dark red (#8B0000) — darker = more vulnerable
  
EACH PANEL (OY-FE):
  Y-AXIS: Log points from 0.00 to +0.20
  X-AXIS: Subgroup
  Bars: Grouped by subgroup, with error bars
  Bar colors:
    - Male/Young: Light blue (#B3D9FF)
    - Female/Old: Dark blue (#003D99) — darker = larger positive effect

SHARED Y-AXES: No (FY-FE and OY-FE have different scales for visibility)

ANNOTATIONS (above/below bars):
  - FY-FE (Male): "-0.0167 (ns)"
  - FY-FE (Female): "-0.1763*"
  - FY-FE (Young): "+0.0230 (ns)"
  - FY-FE (Old): "-0.1583*"
  - OY-FE (Male): "+0.1189**"
  - OY-FE (Female): "+0.1488**"
  - OY-FE (Young): "+0.1676†"
  - OY-FE (Old): "+0.1373**"
  (ns = not significant, † = marginal p<0.10, * p<0.05, ** p<0.01)

GRID LABELS:
  - Top row: "Within Firm (FY-FE)" | "Across Occupations (OY-FE)"
  - Left column: "GENDER" | "AGE"
  - Font: Bold, 12pt

REFERENCE LINES: y=0 (dashed black line in each panel)

LEGEND (bottom center):
  "Red = Wage loss, Blue = Wage gain, Darker shade = Larger effect size"
```

### Data Structure for Generation

```
Panel,Specification,Subgroup,Coefficient,SE,Lower_CI,Upper_CI,Color,Significance
"Gender","FY-FE","Male",-0.0167,0.0564,-0.1272,0.0938,"#E8B5B3","ns"
"Gender","FY-FE","Female",-0.1763,0.0741,-0.3215,-0.0311,"#8B0000","*"
"Gender","OY-FE","Male",0.1189,0.0366,0.0472,0.1906,"#B3D9FF","**"
"Gender","OY-FE","Female",0.1488,0.0512,0.0484,0.2492,"#003D99","**"
"Age","FY-FE","Young (<45)",0.0230,0.0717,-0.1175,0.1635,"#E8B5B3","ns"
"Age","FY-FE","Old (≥45)",-0.1583,0.0517,-0.2596,-0.0570,"#8B0000","*"
"Age","OY-FE","Young (<45)",0.1676,0.1033,-0.0348,0.3700,"#B3D9FF","†"
"Age","OY-FE","Old (≥45)",0.1373,0.0414,0.0562,0.2184,"#003D99","**"
```

### Visual Encoding
- **Color intensity**: Darker = larger effect (vulnerability for negative, robustness for positive)
- **Bar width**: 0.35 units within each subgroup pair
- **Error bar**: Extends from Lower_CI to Upper_CI
- **Spacing**: 0.1 units between subgroup pairs, 0.5 units between FY-FE and OY-FE panels

### Use Case
- **Academic papers**: Heterogeneous effects section (Figure 2 or Table 4)
- **Presentations**: Slide emphasizing vulnerability (especially women, older workers)
- **Policy briefs**: Central figure showing distributional impacts

---

## Chart 3: Vulnerability Heatmap (Optional: for Presentations)

### Purpose
Provide a quick visual reference for which groups face the largest wage losses (within-firm FE).

### Type
2×2 heatmap grid (Rows: Gender/Age, Columns: FY-FE loss magnitude, OY-FE gain magnitude, Reversal)

### Layout

```
TITLE: "Vulnerability to AI Adoption: Who Loses Within Firms?"

HEATMAP STRUCTURE (Gender Variant):
┌──────────┬──────────────┬──────────────┬──────────────┐
│ Subgroup │ FY-FE Loss   │ OY-FE Gain   │ Reversal     │
│          │ (CHF/month)  │ (CHF/month)  │ Magnitude    │
├──────────┼──────────────┼──────────────┼──────────────┤
│ Male     │     -CHF 5   │    +CHF 41   │    +CHF 46   │
│          │   (pale red) │  (pale blue) │   (white)    │
├──────────┼──────────────┼──────────────┼──────────────┤
│ Female   │    -CHF 60   │    +CHF 51   │   +CHF 111   │
│          │  (dark red)  │  (dark blue) │  (light gray)│
└──────────┴──────────────┴──────────────┴──────────────┘

HEATMAP STRUCTURE (Age Variant):
┌──────────┬──────────────┬──────────────┬──────────────┐
│ Subgroup │ FY-FE Loss   │ OY-FE Gain   │ Reversal     │
│          │ (CHF/month)  │ (CHF/month)  │ Magnitude    │
├──────────┼──────────────┼──────────────┼──────────────┤
│ Young    │    +CHF 8    │    +CHF 57   │    +CHF 49   │
│          │  (pale red)  │  (pale blue) │   (white)    │
├──────────┼──────────────┼──────────────┼──────────────┤
│ Old      │    -CHF 54   │    +CHF 47   │   +CHF 101   │
│          │  (dark red)  │  (dark blue) │  (light gray)│
└──────────┴──────────────┴──────────────┴──────────────┘

COLOR SCHEME (Diverging Red-Blue):
  FY-FE Loss Magnitude:
    - Pale red (CHF -5): RGB(255, 200, 200)
    - Dark red (CHF -60): RGB(178, 34, 52)
  
  OY-FE Gain Magnitude:
    - Pale blue (CHF +41): RGB(179, 217, 255)
    - Dark blue (CHF +57): RGB(0, 51, 153)
  
  Reversal Magnitude:
    - White (CHF +46): RGB(255, 255, 255)
    - Light gray (CHF +111): RGB(200, 200, 200)

CELL ANNOTATIONS: Bold text, 10pt font, contrasting color
  - Example: "Male" row shows "-CHF 5" with notation "(ns)"
  - Example: "Female" row shows "-CHF 60" with notation "*"

BORDERS: Light gray, 1pt

LEGEND (below heatmap):
  "Red intensity = Wage loss severity within firms
   Blue intensity = Wage gain size across occupations
   Darker shades indicate larger effects (greater vulnerability or protection)"

SUBTITLE (9pt):
  "Economic magnitudes: CHF/month for 1 SD increase in AI exposure.
   Female workers lose 12× more within firms (-CHF 60 vs -CHF 5).
   Older workers lose 6.75× more within firms (-CHF 54 vs +CHF 8)."
```

### Data Structure for Generation

```
Subgroup_Type,Subgroup,FY_FE_CHF,FY_FE_Significance,OY_FE_CHF,OY_FE_Significance,Reversal_CHF
"Gender","Male",-5,"ns",41,"**",46
"Gender","Female",-60,"*",51,"**",111
"Age","Young",-8,"ns",57,"†",49
"Age","Old",-54,"*",47,"**",101
```

### Visual Encoding
- **Cell color saturation**: Proportional to absolute effect magnitude
- **Red vs. Blue**: Intuitively represents loss vs. gain
- **Saturation**: Darker = larger effect
- **Grid layout**: Aligned for easy comparison across rows and columns

### Use Case
- **Presentations**: Single slide summarizing distributional vulnerabilities
- **Policy briefs**: Quick reference for vulnerable groups
- **Social media / infographics**: Simple visual story

---

## Chart 4: Economic Magnitudes Timeline (Optional: for Policy Briefs)

### Purpose
Show annual and lifetime wage impacts for vulnerable groups.

### Type
Grouped bar chart (Subgroups × Annual impact, 5-year cumulative, 10-year cumulative)

### Layout

```
TITLE: "Lifetime Wage Cost of AI Adoption for Vulnerable Workers"

X-AXIS: Impact horizon
  - "1 Year" (annual)
  - "5 Years" (cumulative)
  - "10 Years" (cumulative)
  - "Career Avg." (assuming 40-year career)

Y-AXIS: Cumulative wage loss (CHF thousands)
  Scale: 0 to -60 (in thousands)

GROUPED BARS (three groups per x position):
  - Overall sample (light gray): -CHF 4.2k | -CHF 20.9k | -CHF 41.8k
  - Women (red): -CHF 8.6k | -CHF 43.2k | -CHF 86.4k
  - Older workers (dark red): -CHF 7.8k | -CHF 39.0k | -CHF 78.0k

BAR COLORS:
  - Overall: #CCCCCC (light gray)
  - Women: #E8B5B3 (light red)
  - Older: #8B0000 (dark red)

ERROR BARS: Extend from lower CI to upper CI for each bar

ANNOTATIONS (on top of bars):
  - "−CHF 4.2k"
  - "−CHF 8.6k*"
  - "−CHF 7.8k*"
  (with asterisk for significance)

LEGEND (top right):
  "Overall sample | Women | Older workers (≥45)"

SUBTITLE (9pt):
  "Cumulative lifetime wage losses from 1 SD increase in firm AI exposure (FY-FE effect).
   Women and older workers face 2× larger annual losses, accumulating to CHF 86k--78k over 10 years.
   Career average assumes 40-year working life from age 25 (or 20) to 65."
```

### Data Structure for Generation

```
Subgroup,Horizon,Cumulative_Loss_CHF,Lower_CI_CHF,Upper_CI_CHF
"Overall Sample","1 Year",-4200,-7700,-800
"Overall Sample","5 Years",-20900,-38500,-4000
"Overall Sample","10 Years",-41800,-77000,-8000
"Overall Sample","Career Avg.",-262000,-481000,-50000
"Women","1 Year",-8600,-15400,-1800
"Women","5 Years",-43200,-77000,-9000
"Women","10 Years",-86400,-154000,-18000
"Women","Career Avg.",-540000,-968000,-113000
"Older Workers","1 Year",-7800,-13000,-2800
"Older Workers","5 Years",-39000,-65000,-14000
"Older Workers","10 Years",-78000,-130000,-28000
"Older Workers","Career Avg.",-491000,-820000,-176000
```

### Use Case
- **Policy briefs**: Emphasize long-term economic impact
- **Labor union presentations**: Show cost to workers
- **Academic papers**: Appendix or supplementary materials

---

## Synthesis: Which Charts for Which Audience?

### For Academic Papers
**Must-have:**
1. **Chart 1 (Coefficient Plot)**: Main text, opening of results section
2. **Table 1 (Main Results)**: Accompanies Chart 1
3. **Table 2 (Heterogeneous Effects)**: After Chart 2
4. **Chart 2 (Faceted Heatmap)**: Heterogeneous effects section

**Optional:**
5. **Chart 3 (Vulnerability Heatmap)**: Appendix for quick reference
6. **Table 3 (Economic Magnitudes)**: Appendix for policy relevance

### For Presentations (Slides)
**Opening (3 slides):**
1. Slide 1: Chart 1 (coefficient plot) — establish the reversal
2. Slide 2: Chart 2 (faceted heterogeneity) — show vulnerability
3. Slide 3: Chart 3 (vulnerability heatmap) — visual summary

**Mechanism Discussion (2 slides):**
4. Slide 4: Table 1 (main results) with interpretation
5. Slide 5: Economic magnitudes (Table 3 excerpt or Chart 4)

### For Policy Briefs / Executive Summaries
**Essential:**
1. Chart 1 (coefficient plot) — the headline finding
2. Chart 3 (vulnerability heatmap) — who's affected
3. Chart 4 (lifetime wages) — policy impact

### For Media / General Public
**Simplified version:**
1. Chart 3 (vulnerability heatmap) — key insight
2. Simplified Chart 4 — economic impact (annual only, CHF millions for aggregate)

---

## Technical Notes for Implementation

### Color Palette
```
Red (loss):     #C1272D (main), #E8B5B3 (light), #8B0000 (dark)
Blue (gain):    #0066CC (main), #B3D9FF (light), #003D99 (dark)
Gray/neutral:   #CCCCCC (light), #666666 (medium), #000000 (dark)
Accent:         #FFB000 (orange, for significance stars if needed)
```

### Typography
- **Title**: 14pt, bold, sans-serif (e.g., Helvetica, Arial)
- **Axis labels**: 11pt, sans-serif
- **Annotations**: 10pt, sans-serif
- **Legend**: 9pt, sans-serif
- **Notes/Footnotes**: 8pt, sans-serif, italic

### Dimensions (for papers)
- **Single column**: 3.5 inches wide × 2.8 inches tall
- **Double column**: 7 inches wide × 3.5 inches tall
- **Resolution**: 300 dpi for print, 150 dpi for screen

### Dimensions (for presentations)
- **Slide aspect ratio**: 16:9 (1920 × 1080)
- **Chart area**: 900 × 600 pixels (leave margins)
- **Resolution**: 150 dpi (on-screen viewing)

