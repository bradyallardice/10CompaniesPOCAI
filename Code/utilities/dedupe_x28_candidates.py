#!/usr/bin/env python3
"""
Deduplicate the size-filtered X28 update candidates list into a clean
"firms to send to X28" file.

Three collapse rules:
  1. Drop firm_name == "no answer" placeholder rows.
  2. Collapse known-parent variants (e.g. all CHUV / EPFL / Universität Luzern
     sub-units) into one row per parent. The parent name is extracted from
     llm_rejected_match (which has the form "<parent> not in X28" for rows
     where x28_status == "known_major_parent_missing").
  3. Collapse same-name, different-location rows (e.g. Gruner AG in Renens and
     Oberwil) into one row per firm; locations are preserved in a list column.

Inputs:
  --in   Size-filtered X28 candidate list (after Thomas's pw85 filter).
         Schema: firm_name, firm_location, first_seen_year, n_observations,
                 years_present, noga08_sample, mj_mis_noga, x28_status,
                 llm_rejected_match
  --out  Deduplicated output CSV.

Output schema:
  firm_name_canonical, firm_name_variants, n_rows_collapsed, firm_locations,
  first_seen_year, n_observations_total, years_present_union, noga08_sample,
  mj_mis_noga, x28_status, llm_rejected_match
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IN = (
    PROJECT_ROOT
    / "Data"
    / "shp_matching"
    / "w24_w27_run"
    / "x28_update_candidates_for_thomas_checked_only_larger_firms.csv"
)
DEFAULT_OUT = PROJECT_ROOT / "Data" / "x28_update_candidates_deduplicated.csv"

STATUS_PRIORITY = {
    "known_major_parent_missing": 0,
    "llm_rejected_candidate": 1,
    "no_candidate_found": 2,
}
PARENT_SUFFIX = " not in X28"


def extract_parent(row: pd.Series) -> str:
    """Return the canonical cluster key for a candidate row.

    For known_major_parent_missing rows whose llm_rejected_match ends in
    " not in X28" (e.g. "CHUV not in X28"), the cluster key is the parent
    name. Otherwise it is the firm_name itself.
    """
    if row["x28_status"] == "known_major_parent_missing":
        rej = row["llm_rejected_match"]
        if isinstance(rej, str) and rej.endswith(PARENT_SUFFIX):
            return rej[: -len(PARENT_SUFFIX)].strip()
    return row["firm_name"]


def join_unique(series: pd.Series, sep: str = "; ") -> str:
    seen: set = set()
    out: list = []
    for v in series.dropna():
        s = str(v).strip()
        if s in ("", "nan", "NA"):
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return sep.join(out)


def mode_or_first(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode()
    return m.iloc[0] if not m.empty else s.iloc[0]


def best_status(series: pd.Series):
    """Pick the most informative x28_status (highest-priority value wins)."""
    s = list(series.dropna().unique())
    if not s:
        return np.nan
    return sorted(s, key=lambda x: STATUS_PRIORITY.get(x, 99))[0]


def dedupe(in_path: Path, out_path: Path) -> pd.DataFrame:
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    required = {
        "firm_name", "firm_location", "first_seen_year", "n_observations",
        "years_present", "noga08_sample", "mj_mis_noga", "x28_status",
        "llm_rejected_match",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}\n"
            f"Got: {list(df.columns)}"
        )

    n_in = len(df)
    print(f"Loaded {n_in} rows from {in_path}")

    # Step 1: drop "no answer" placeholders
    mask_no_answer = df["firm_name"].astype(str).str.strip().str.lower() == "no answer"
    n_no_answer = int(mask_no_answer.sum())
    df = df.loc[~mask_no_answer].copy()
    print(f"Step 1: dropped {n_no_answer} 'no answer' rows -> {len(df)} remain")

    # Step 2: assign cluster_key
    df["cluster_key"] = df.apply(extract_parent, axis=1)
    if df["cluster_key"].isna().any():
        bad = df[df["cluster_key"].isna()]
        raise ValueError(f"cluster_key NaN in {len(bad)} rows:\n{bad}")
    print(f"Step 2: {df['cluster_key'].nunique()} distinct cluster keys")

    # Step 3: aggregate
    df["n_observations"] = pd.to_numeric(df["n_observations"], errors="coerce")
    df["first_seen_year"] = pd.to_numeric(df["first_seen_year"], errors="coerce")

    agg = df.groupby("cluster_key", dropna=False).agg(
        firm_name_canonical=("cluster_key", "first"),
        firm_name_variants=("firm_name", join_unique),
        n_rows_collapsed=("firm_name", "size"),
        firm_locations=("firm_location", join_unique),
        first_seen_year=("first_seen_year", "min"),
        n_observations_total=("n_observations", "sum"),
        years_present_union=("years_present", join_unique),
        noga08_sample=("noga08_sample", mode_or_first),
        mj_mis_noga=("mj_mis_noga", mode_or_first),
        x28_status=("x28_status", best_status),
        llm_rejected_match=("llm_rejected_match", mode_or_first),
    ).reset_index(drop=True)

    agg = agg.sort_values(
        ["n_rows_collapsed", "firm_name_canonical"], ascending=[False, True]
    ).reset_index(drop=True)

    if not agg["firm_name_canonical"].notna().all():
        raise ValueError("NaN in firm_name_canonical")
    if len(agg) >= n_in:
        raise ValueError(
            f"Output rows ({len(agg)}) >= input rows ({n_in}); dedup did not reduce."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_path, index=False)

    print(f"\nWrote {len(agg)} deduplicated rows -> {out_path}")
    print("\n=== Summary ===")
    print(f"  Input rows:           {n_in}")
    print(f"  Dropped 'no answer':  {n_no_answer}")
    print(f"  After drop:           {n_in - n_no_answer}")
    print(f"  Output unique firms:  {len(agg)}")
    print(f"  Total rows collapsed: {(n_in - n_no_answer) - len(agg)}")
    print("\n=== x28_status distribution (output) ===")
    print(agg["x28_status"].value_counts(dropna=False).to_string())
    print("\n=== Top 10 most-collapsed clusters ===")
    print(
        agg[["firm_name_canonical", "n_rows_collapsed", "x28_status", "firm_locations"]]
        .head(10)
        .to_string(max_colwidth=80)
    )
    print("\n=== NaN check on output ===")
    print(agg.isna().sum().to_string())

    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN,
                   help=f"Input CSV (default: {DEFAULT_IN.relative_to(PROJECT_ROOT)})")
    p.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT,
                   help=f"Output CSV (default: {DEFAULT_OUT.relative_to(PROJECT_ROOT)})")
    args = p.parse_args()
    dedupe(args.in_path.resolve(), args.out_path.resolve())


if __name__ == "__main__":
    sys.exit(main())
