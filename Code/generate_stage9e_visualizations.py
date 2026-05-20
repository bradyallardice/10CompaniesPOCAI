#!/usr/bin/env python3
"""
Generate modern academic visualizations for Stage 9e results.

Inputs (under Data/):
  - stage9e_results_long.csv  (Modules A/B/C/D/F/G — person-level estimates)
  - stage9e_firm_hiring.csv   (Module E — firm-level future hiring)
  - shp_panel_prepared.csv    (matched-sample exposure distribution for Fig 3)

Outputs (under docs/figures/stage9e/, PNG @ 300 dpi):
  - fig1_module_a_coefplot.png         Headline coefficient plot (Module A)
  - fig2_module_d_heterogeneity.png    Subgroup small multiples (Module D)
  - fig3_svp_nonlinearity.png          Quadratic fit + exposure distribution
  - fig4_module_e_firm_hiring.png      Firm hiring level vs growth
  - fig5_summary_composite.png         Slide-ready summary of headline findings
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Data"
FIG_DIR = PROJECT_ROOT / "docs" / "figures" / "stage9e"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = DATA_DIR / "stage9e_results_long.csv"
FIRM_FILE = DATA_DIR / "stage9e_firm_hiring.csv"
PANEL_FILE = DATA_DIR / "shp_panel_prepared.csv"

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

# Modern academic palette — sign-based with three saturation levels
# (p<.05 strong, p<.10 mid, ns soft tint instead of grey)
COL_POS_STRONG = "#1F6FB4"   # deep blue
COL_POS_MID    = "#5A9BD4"
COL_POS_SOFT   = "#A8C7E5"
COL_NEG_STRONG = "#C73E3A"   # deep red
COL_NEG_MID    = "#D6776F"
COL_NEG_SOFT   = "#E8B5B0"
COL_NULL       = "#B0B0B0"   # only when sign is undefined

COL_GREY = "#999999"
COL_BG = "#F5F5F5"

# Family / panel accent colors (used for band tints and title underlines)
FAMILY_COLORS = {
    "Vote choice":         "#FFE9B3",  # warm yellow
    "Ideology":            "#CFE8D1",  # green
    "Policy attitudes":    "#E0CBE8",  # lavender
    "Institutional trust": "#BFE2E5",  # teal
    "Economic perceptions": "#FFD9B8", # peach
}
PANEL_ACCENTS = {
    "job_insecurity":  "#E67E22",  # orange
    "vote_svp":        "#7D3C98",  # purple (politically neutral hue, not party-color)
    "vote_sp":         "#2C7BB6",  # blue
    "redistributive":  "#1B9E77",  # green
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def sig_color(p, beta):
    if pd.isna(p) or pd.isna(beta):
        return COL_NULL
    pos = beta > 0
    if p < 0.05:
        return COL_POS_STRONG if pos else COL_NEG_STRONG
    if p < 0.10:
        return COL_POS_MID if pos else COL_NEG_MID
    return COL_POS_SOFT if pos else COL_NEG_SOFT


def sig_alpha(p):
    # With tinted null colors we no longer need to fade out — keep all bars solid
    return 1.0


def sig_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def save_fig(fig, name):
    out = FIG_DIR / f"{name}.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")


# ── Pretty labels ──────────────────────────────────────────────────────────────
OUTCOME_LABELS = {
    "vote_svp": "Vote SVP (right-populist)",
    "vote_sp": "Vote SP (social-democratic)",
    "vote_fdp": "Vote FDP (center-right)",
    "vote_cvp": "Vote CVP (center)",
    "vote_glp": "Vote GLP (green-liberal)",
    "vote_bdp": "Vote BDP (conservative-democratic)",
    "vote_no_party": "Vote: no party",
    "vote_sp_gps": "Vote SP/GPS (left)",
    "leftright": "Left–right self-placement (0–10)",
    "social_trust": "Social trust (0–10)",
    "redistributive": "Redistribution support (1–3)",
    "welfare": "Welfare/social spending (1–3)",
    "nativism": "Nativism / opposition to foreigners (1–3)",
    "gender_equality": "Gender equality (0–10)",
    "trust_govt": "Trust in government (0–10)",
    "democracy_sat": "Democracy satisfaction (0–10)",
    "political_efficacy": "Political efficacy (0–10)",
    "eu_opinion": "EU opinion (1–3)",
    "job_insecurity": "Job insecurity",
}

OUTCOME_FAMILY = {
    "vote_svp": "Vote choice", "vote_sp": "Vote choice", "vote_fdp": "Vote choice",
    "vote_cvp": "Vote choice", "vote_glp": "Vote choice", "vote_bdp": "Vote choice",
    "vote_no_party": "Vote choice",
    "leftright": "Ideology", "social_trust": "Ideology",
    "redistributive": "Policy attitudes", "welfare": "Policy attitudes",
    "nativism": "Policy attitudes", "gender_equality": "Policy attitudes",
    "eu_opinion": "Policy attitudes",
    "trust_govt": "Institutional trust", "democracy_sat": "Institutional trust",
    "political_efficacy": "Institutional trust",
    "job_insecurity": "Economic perceptions",
}

FAMILY_ORDER = ["Vote choice", "Ideology", "Policy attitudes",
                "Institutional trust", "Economic perceptions"]

SUBGROUP_LABELS = {
    "subgroup_full": "Full sample",
    "subgroup_male": "Men",
    "subgroup_female": "Women",
    "subgroup_edu_high": "High education",
    "subgroup_edu_low": "Low education",
    "subgroup_pre2018": "Pre-2018",
    "subgroup_post2018": "Post-2018 (LLM era)",
}
SUBGROUP_ORDER = ["subgroup_full", "subgroup_male", "subgroup_female",
                  "subgroup_edu_high", "subgroup_edu_low",
                  "subgroup_pre2018", "subgroup_post2018"]

# ── Spec descriptions for figure footers ───────────────────────────────────────
PERSON_SPEC = (
    "Spec: Person + ISCO-3d×Year + Industry×Year (NOGA2M 17-sector) FE; "
    "SE clustered on idpers; matched sample (exposure > 0).\n"
    "Controls: age, age², permanent contract (pw36), part-time (pw39), "
    "computer use at work (pw607), firm size (pw85), public sector (pw32), "
    "lag log(job ads)."
)
FIRM_SPEC = (
    "Spec: Firm + Year FE; SE clustered on company_id; "
    "all firm-years 2010–2024. No additional controls — fixed effects absorb "
    "time-invariant firm characteristics and common annual shocks."
)
SIG_LEGEND = "Significance: *** p<.01, ** p<.05, * p<.10."


# ── Data loading ───────────────────────────────────────────────────────────────
def load_results():
    df = pd.read_csv(RESULTS_FILE)
    needed = {"module", "outcome", "spec", "term", "estimate",
              "std_error", "p_value", "ci_lower", "ci_upper", "n_obs"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {RESULTS_FILE.name}: {missing}")
    return df


def load_firm():
    df = pd.read_csv(FIRM_FILE)
    df["ci_lower"] = df["estimate"] - 1.96 * df["std_error"]
    df["ci_upper"] = df["estimate"] + 1.96 * df["std_error"]
    return df


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Module A coefficient plot
# ──────────────────────────────────────────────────────────────────────────────
def figure1_module_a(results):
    rows = results[
        (results["module"] == "A")
        & (results["term"] == "hampole_ai_exposure_avg_foy")
    ].copy()
    # Add job insecurity from Module D full subgroup (not in Module A)
    ji = results[
        (results["module"] == "D")
        & (results["outcome"] == "job_insecurity")
        & (results["spec"] == "subgroup_full")
    ]
    rows = pd.concat([rows, ji], ignore_index=True)

    rows["family"] = rows["outcome"].map(OUTCOME_FAMILY).fillna("Other")
    rows["label"] = rows["outcome"].map(OUTCOME_LABELS).fillna(rows["outcome"])
    rows["family_rank"] = rows["family"].map(
        {f: i for i, f in enumerate(FAMILY_ORDER)}
    )
    rows = rows[rows["outcome"] != "vote_sp_gps"]  # drop legacy grouped vote
    rows = rows.sort_values(["family_rank", "estimate"], ascending=[True, True])
    rows = rows.reset_index(drop=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.6, 0.42 * n + 1.6))

    ys = np.arange(n)[::-1]
    for y, (_, r) in zip(ys, rows.iterrows()):
        c = sig_color(r["p_value"], r["estimate"])
        a = sig_alpha(r["p_value"])
        ax.hlines(y, r["ci_lower"], r["ci_upper"],
                  color=c, alpha=a, linewidth=2.0)
        ax.plot(r["estimate"], y, "o", color=c, alpha=a,
                markersize=6, markeredgecolor="white", markeredgewidth=0.6)
        stars = sig_stars(r["p_value"])
        if stars:
            ax.text(r["ci_upper"], y, " " + stars, va="center",
                    fontsize=9, color=c, fontweight="bold")

    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels(rows["label"].tolist())
    ax.set_xlabel("Coefficient on AI exposure (β)   ·   95% CI")
    ax.set_ylim(-0.7, n - 0.3)

    # Family bands — colored shading per family + horizontal label inside band
    family_y = {}
    for fam in FAMILY_ORDER:
        idx = rows.index[rows["family"] == fam].tolist()
        if not idx:
            continue
        ys_fam = [ys[i] for i in idx]
        family_y[fam] = (min(ys_fam) - 0.45, max(ys_fam) + 0.45)

    # Add right-side padding so band labels and stars don't crowd the edge
    xlim = list(ax.get_xlim())
    span = xlim[1] - xlim[0]
    xlim[1] = xlim[1] + 0.18 * span
    ax.set_xlim(xlim)

    for fam, (lo, hi) in family_y.items():
        ax.axhspan(lo, hi, color=FAMILY_COLORS.get(fam, COL_BG),
                   alpha=0.45, zorder=0)
        ax.text(xlim[1], (lo + hi) / 2, " " + fam,
                va="center", ha="right",
                fontsize=9, color="#333333", fontweight="bold",
                style="italic")
    ax.grid(axis="x", alpha=0.4)
    ax.grid(axis="y", alpha=0.0)

    ax.set_title(
        "Effects of AI exposure on political outcomes",
        loc="left", pad=10,
    )
    fig.text(0.5, -0.04,
             f"{PERSON_SPEC}\n{SIG_LEGEND}  "
             "Bar color: blue = positive, red = negative; saturation by p-value.",
             ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout()
    save_fig(fig, "fig1_module_a_coefplot")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Module D heterogeneity small multiples
# ──────────────────────────────────────────────────────────────────────────────
def figure2_heterogeneity(results):
    outcomes = ["job_insecurity", "vote_svp", "vote_sp", "redistributive"]
    titles = {
        "job_insecurity": "Job insecurity",
        "vote_svp": "Vote SVP (right-populist)",
        "vote_sp": "Vote SP (social-democratic)",
        "redistributive": "Redistribution support",
    }

    d = results[
        (results["module"] == "D")
        & (results["outcome"].isin(outcomes))
        & (results["term"] == "hampole_ai_exposure_avg_foy")
    ].copy()
    d["sg_rank"] = d["spec"].map({s: i for i, s in enumerate(SUBGROUP_ORDER)})
    d["sg_label"] = d["spec"].map(SUBGROUP_LABELS)
    d = d.sort_values("sg_rank")

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.2), sharey=True)

    for ax, oc in zip(axes, outcomes):
        sub = d[d["outcome"] == oc].copy()
        ys = np.arange(len(sub))[::-1]
        for y, (_, r) in zip(ys, sub.iterrows()):
            c = sig_color(r["p_value"], r["estimate"])
            a = sig_alpha(r["p_value"])
            ax.hlines(y, r["ci_lower"], r["ci_upper"],
                      color=c, alpha=a, linewidth=2.0)
            ax.plot(r["estimate"], y, "o", color=c, alpha=a,
                    markersize=5.5, markeredgecolor="white",
                    markeredgewidth=0.6)
            stars = sig_stars(r["p_value"])
            if stars:
                ax.text(r["ci_upper"], y, " " + stars, va="center",
                        fontsize=8, color=c, fontweight="bold")
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["sg_label"].tolist())
        ax.set_title(titles[oc], loc="left")
        ax.set_xlabel("β (95% CI)")
        ax.grid(axis="x", alpha=0.4)
        ax.grid(axis="y", alpha=0.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

    fig.suptitle("Heterogeneous effects of AI exposure across subgroups",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
             "Each panel: Module D subgroup estimates with 95% CIs. "
             "Same controls as headline spec. *** p<.01, ** p<.05, * p<.10",
             ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout()
    save_fig(fig, "fig2_module_d_heterogeneity")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — SVP non-linearity
# ──────────────────────────────────────────────────────────────────────────────
def figure3_svp_nonlinearity(results):
    quad = results[
        (results["module"] == "B")
        & (results["outcome"] == "vote_svp")
        & (results["spec"] == "quadratic")
    ]
    if quad.empty:
        print("  WARN: no quadratic spec found for vote_svp — skipping Fig 3")
        return
    beta_lin = float(quad.loc[quad["term"] == "hampole_ai_exposure_avg_foy",
                              "estimate"].iloc[0])
    se_lin = float(quad.loc[quad["term"] == "hampole_ai_exposure_avg_foy",
                            "std_error"].iloc[0])
    beta_sq = float(quad.loc[quad["term"] == "exp2", "estimate"].iloc[0])
    se_sq = float(quad.loc[quad["term"] == "exp2", "std_error"].iloc[0])
    p_lin = float(quad.loc[quad["term"] == "hampole_ai_exposure_avg_foy",
                           "p_value"].iloc[0])
    p_sq = float(quad.loc[quad["term"] == "exp2", "p_value"].iloc[0])

    # Tercile point estimates (T1 reference, so T1 = 0)
    terc = results[
        (results["module"] == "B")
        & (results["outcome"] == "vote_svp")
        & (results["spec"] == "tercile_dummies")
    ]
    terc_beta = {r["term"]: r for _, r in terc.iterrows()}

    # Exposure distribution from matched sample
    try:
        exp_df = pd.read_csv(PANEL_FILE, usecols=["hampole_ai_exposure_avg_foy"])
        exp_vals = exp_df["hampole_ai_exposure_avg_foy"].dropna()
        exp_vals = exp_vals[exp_vals > 0]   # matched sample
    except Exception as e:
        print(f"  WARN: could not load exposure values: {e}")
        exp_vals = pd.Series(np.linspace(0.01, 1, 500))

    x_max = float(np.quantile(exp_vals, 0.99))
    xs = np.linspace(0, x_max, 200)

    # Center linear interpretation: the quadratic spec uses raw exposure
    # and exp2 = exposure**2 (per stage_9e). Predict centered at sample mean.
    x_mean = float(exp_vals.mean())
    yhat = beta_lin * (xs - x_mean) + beta_sq * (xs**2 - x_mean**2)

    # Approximate point-wise SE via delta method, treating cov(β_lin,β_sq)=0
    # (conservative; full vcov not in CSV).
    se_yhat = np.sqrt((xs - x_mean) ** 2 * se_lin ** 2
                      + (xs ** 2 - x_mean ** 2) ** 2 * se_sq ** 2)
    yhat_lo = yhat - 1.96 * se_yhat
    yhat_hi = yhat + 1.96 * se_yhat

    fig = plt.figure(figsize=(8.8, 5.4))
    gs = GridSpec(2, 1, height_ratios=[4.2, 1.0], hspace=0.06)
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1], sharex=ax)

    ax.fill_between(xs, yhat_lo, yhat_hi, color=COL_ACCENT, alpha=0.18,
                    label="95% CI (delta method)")
    ax.plot(xs, yhat, color=COL_ACCENT, linewidth=2.2,
            label="Predicted Δ Vote SVP (vs. sample mean)")
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7)

    # Tercile cut points
    t1, t2 = np.quantile(exp_vals, [1/3, 2/3])
    for cx, lbl in [(t1, "T1│T2"), (t2, "T2│T3")]:
        ax.axvline(cx, color=COL_GREY, linewidth=0.7, linestyle=":")
        ax.text(cx, ax.get_ylim()[1], f" {lbl}", color=COL_GREY,
                fontsize=8, va="top", ha="left")

    # Overlay tercile β at midpoints
    for tname, x_pos in [("exp_t2", (t1 + t2) / 2),
                         ("exp_t3", (t2 + x_max) / 2)]:
        if tname in terc_beta:
            r = terc_beta[tname]
            ax.errorbar(x_pos, r["estimate"],
                        yerr=[[r["estimate"] - r["ci_lower"]],
                              [r["ci_upper"] - r["estimate"]]],
                        fmt="s", color=COL_NEG, markersize=6,
                        markeredgecolor="white", markeredgewidth=0.7,
                        capsize=3, label="Tercile β (vs T1)"
                        if tname == "exp_t2" else None)

    ax.set_ylabel("Change in Pr(Vote SVP)")
    ax.set_title(
        f"Non-linear effect of AI exposure on SVP voting   "
        f"·   β_lin={beta_lin:+.3f} (p={p_lin:.3f}),  "
        f"β_exp²={beta_sq:+.3f} (p={p_sq:.3f})",
        loc="left", pad=8,
    )
    ax.legend(loc="upper right")
    plt.setp(ax.get_xticklabels(), visible=False)

    # Density / rug below
    axh.hist(exp_vals[exp_vals <= x_max], bins=60,
             color=COL_GREY, alpha=0.55, edgecolor="white", linewidth=0.4)
    axh.set_xlabel("Hampole AI exposure (matched sample)")
    axh.set_ylabel("Density", fontsize=8.5)
    axh.set_yticks([])
    axh.grid(False)
    axh.spines["left"].set_visible(False)

    fig.text(0.5, -0.02,
             "Quadratic fit from Module B vote_svp spec; curve centered at "
             "the matched-sample mean. Tercile point estimates plotted at "
             "the midpoint of each tercile range.",
             ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout()
    save_fig(fig, "fig3_svp_nonlinearity")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Module E firm hiring
# ──────────────────────────────────────────────────────────────────────────────
def figure4_firm_hiring(firm):
    # Baseline specs only (drop joint-with-expertise variants)
    keep_specs = {
        "Log job ads at t+1": "Log job ads (t+1)  ·  level",
        "Δlog job ads": "Δlog job ads  ·  growth",
    }
    f = firm[firm["spec"].isin(keep_specs.keys())].copy()
    # Drop expertise treatment for the headline panel
    treat_order = ["pct_ai_ads_cumulative", "firm_ai_exposure", "log_ai_apps"]
    treat_labels = {
        "pct_ai_ads_cumulative": "% AI job ads (cumulative)",
        "firm_ai_exposure": "Firm AI exposure (Hampole)",
        "log_ai_apps": "log(AI applications)",
    }
    f = f[f["term"].isin(treat_order)].copy()
    f["spec_label"] = f["spec"].map(keep_specs)
    f["treat_label"] = f["term"].map(treat_labels)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.7), sharey=True)

    for ax, spec_label in zip(axes, keep_specs.values()):
        sub = f[f["spec_label"] == spec_label].copy()
        sub["t_rank"] = sub["term"].map({t: i for i, t in enumerate(treat_order)})
        sub = sub.sort_values("t_rank")
        ys = np.arange(len(sub))[::-1]
        for y, (_, r) in zip(ys, sub.iterrows()):
            c = sig_color(r["p_value"], r["estimate"])
            a = sig_alpha(r["p_value"])
            ax.hlines(y, r["ci_lower"], r["ci_upper"],
                      color=c, alpha=a, linewidth=2.4)
            ax.plot(r["estimate"], y, "o", color=c, alpha=a,
                    markersize=7, markeredgecolor="white", markeredgewidth=0.7)
            stars = sig_stars(r["p_value"])
            label_text = f"  β={r['estimate']:+.3f}{(' ' + stars) if stars else ''}"
            ax.text(r["ci_upper"], y, label_text, va="center",
                    fontsize=9, color=c)
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels(sub["treat_label"].tolist())
        ax.set_xlabel("β  (Firm + Year FE, cluster company_id)")
        ax.set_title(spec_label, loc="left")
        ax.grid(axis="x", alpha=0.4)
        ax.grid(axis="y", alpha=0.0)
        # Pad right side so β labels don't touch the edge
        xlim = list(ax.get_xlim())
        span = xlim[1] - xlim[0]
        xlim[1] = xlim[1] + 0.22 * span
        ax.set_xlim(xlim)

    fig.suptitle("AI adoption and future firm hiring",
                 x=0.01, ha="left", fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, -0.04,
             "N=72,720 firm-years; 7,197 firms. "
             "AI-exposed firms hire more in levels but not in growth. "
             "*** p<.01, ** p<.05, * p<.10",
             ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout()
    save_fig(fig, "fig4_module_e_firm_hiring")


# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Composite summary
# ──────────────────────────────────────────────────────────────────────────────
def figure5_summary(results, firm):
    # (a) Job insecurity pre/post-2018
    ji = results[(results["module"] == "D") & (results["outcome"] == "job_insecurity")
                 & (results["spec"].isin(["subgroup_pre2018", "subgroup_post2018",
                                          "subgroup_full"]))]
    ji = ji.set_index("spec").reindex(
        ["subgroup_pre2018", "subgroup_post2018", "subgroup_full"])

    # (b) Vote SP gender reversal
    sp = results[(results["module"] == "D") & (results["outcome"] == "vote_sp")
                 & (results["spec"].isin(["subgroup_male", "subgroup_female",
                                          "subgroup_full"]))]
    sp = sp.set_index("spec").reindex(
        ["subgroup_male", "subgroup_female", "subgroup_full"])

    # (c) Redistribution high vs low education
    rd = results[(results["module"] == "D") & (results["outcome"] == "redistributive")
                 & (results["spec"].isin(["subgroup_edu_high", "subgroup_edu_low",
                                          "subgroup_full"]))]
    rd = rd.set_index("spec").reindex(
        ["subgroup_edu_high", "subgroup_edu_low", "subgroup_full"])

    # (d) Firm hiring level — three treatments
    f = firm[(firm["spec"] == "Log job ads at t+1")
             & (firm["term"].isin(["pct_ai_ads_cumulative", "firm_ai_exposure",
                                   "log_ai_apps"]))].copy()
    f = f.set_index("term").reindex(
        ["pct_ai_ads_cumulative", "firm_ai_exposure", "log_ai_apps"])

    panels = [
        ("Job insecurity rises after 2018",
         ji,
         {"subgroup_pre2018": "Pre-2018", "subgroup_post2018": "Post-2018",
          "subgroup_full": "Full sample"},
         "β on AI exposure"),
        ("Vote SP: gender reversal",
         sp,
         {"subgroup_male": "Men", "subgroup_female": "Women",
          "subgroup_full": "Full sample"},
         "β on AI exposure"),
        ("Redistribution: high-education driven",
         rd,
         {"subgroup_edu_high": "High edu", "subgroup_edu_low": "Low edu",
          "subgroup_full": "Full sample"},
         "β on AI exposure"),
        ("Firm hiring level (t+1)",
         f,
         {"pct_ai_ads_cumulative": "% AI ads",
          "firm_ai_exposure": "Firm AI exposure",
          "log_ai_apps": "log(AI apps)"},
         "β on treatment"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    axes = axes.flatten()

    for ax, (title, df, labels, xlab) in zip(axes, panels):
        df = df.copy()
        df["ci_lo"] = df["ci_lower"] if "ci_lower" in df.columns else (
            df["estimate"] - 1.96 * df["std_error"])
        df["ci_hi"] = df["ci_upper"] if "ci_upper" in df.columns else (
            df["estimate"] + 1.96 * df["std_error"])
        # firm df has ci_lower / ci_upper from load_firm()
        if "ci_lower" not in df.columns:
            df["ci_lo"] = df["estimate"] - 1.96 * df["std_error"]
            df["ci_hi"] = df["estimate"] + 1.96 * df["std_error"]
        else:
            df["ci_lo"] = df["ci_lower"]
            df["ci_hi"] = df["ci_upper"]

        order_keys = list(labels.keys())
        ys = np.arange(len(order_keys))[::-1]
        for y, key in zip(ys, order_keys):
            r = df.loc[key]
            c = sig_color(r["p_value"], r["estimate"])
            a = sig_alpha(r["p_value"])
            ax.hlines(y, r["ci_lo"], r["ci_hi"],
                      color=c, alpha=a, linewidth=2.4)
            ax.plot(r["estimate"], y, "o", color=c, alpha=a,
                    markersize=7, markeredgecolor="white", markeredgewidth=0.7)
            stars = sig_stars(r["p_value"])
            ax.text(r["ci_hi"], y,
                    f"  β={r['estimate']:+.3f}{(' ' + stars) if stars else ''}",
                    va="center", fontsize=9, color=c)
        ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_yticks(ys)
        ax.set_yticklabels([labels[k] for k in order_keys])
        ax.set_title(title, loc="left")
        ax.set_xlabel(xlab)
        ax.grid(axis="x", alpha=0.4)
        ax.grid(axis="y", alpha=0.0)
        # Pad right side so β value labels don't touch the panel edge
        xlim = list(ax.get_xlim())
        span = xlim[1] - xlim[0]
        xlim[1] = xlim[1] + 0.22 * span
        ax.set_xlim(xlim)

    fig.suptitle("AI exposure: headline findings",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.01)
    fig.text(0.5, -0.02,
             "Panels (a)–(c): person-level FE with Industry×Year and ISCO-3d×Year "
             "(matched sample, cluster idpers). Panel (d): firm-level with Firm + "
             "Year FE (cluster company_id). *** p<.01, ** p<.05, * p<.10",
             ha="center", fontsize=8.5, color="#555555")
    plt.tight_layout()
    save_fig(fig, "fig5_summary_composite")


# ──────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {RESULTS_FILE.name} ...")
    results = load_results()
    print(f"  {len(results)} rows")
    print(f"Loading {FIRM_FILE.name} ...")
    firm = load_firm()
    print(f"  {len(firm)} rows")

    print(f"\nWriting figures to {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    figure1_module_a(results)
    figure2_heterogeneity(results)
    figure3_svp_nonlinearity(results)
    figure4_firm_hiring(firm)
    figure5_summary(results, firm)
    print("\nDone.")


if __name__ == "__main__":
    main()
