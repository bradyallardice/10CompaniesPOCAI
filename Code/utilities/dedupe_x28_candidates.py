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
import re
import sys
import unicodedata
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
DEFAULT_X28_REF = PROJECT_ROOT / "Data" / "company_mapping.csv"
DEFAULT_COLLISIONS = (
    PROJECT_ROOT / "Data" / "x28_update_candidates_existing_collisions.csv"
)

# Tokens stripped during aggressive normalization for cross-checking against
# the existing X28 firm list. These are legal-form suffixes, country/region
# qualifiers, and common holding/group qualifiers that we want to treat as
# equivalent. Intentionally aggressive — false positives here are surfaced
# for review, not auto-removed.
_NORMALIZE_DROP = re.compile(
    r"\b("
    r"ag|sa|sarl|sa\.r\.l|gmbh|llc|inc|ltd|spa|s\.p\.a|s\.a|kg|ohg|eg|ev|"
    r"gesellschaft|stiftung|fondation|verein|association|cooperative|"
    r"kooperative|partnership|holding|group|co|services|"
    r"schweiz|suisse|svizzera|swiss|switzerland"
    r")\b"
)


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _norm(s):
    """Aggressive normalization for collision detection. Strips accents,
    lowercases, removes legal-form and country tokens, removes punctuation."""
    if not isinstance(s, str):
        return ""
    s = _strip_accents(s).lower()
    s = _NORMALIZE_DROP.sub(" ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def check_existing_x28(
    dedup: pd.DataFrame, x28_path: Path, out_path: Path
) -> pd.DataFrame:
    """Cross-check deduplicated candidates against the existing X28 firm list.

    For each candidate, normalize the canonical name AND each name variant,
    then look them up in company_mapping.csv. Any hits represent firms that
    are already in X28 and should NOT be on the list sent to Thomas.

    Behaviour: prints collisions, writes them to out_path, and returns them.
    Does NOT raise — the caller decides whether to act on the result. Some
    collisions are legitimate (e.g. holding vs operating company are
    different legal entities that share a normalized name).
    """
    if not x28_path.exists():
        raise FileNotFoundError(
            f"X28 reference file not found: {x28_path}\n"
            f"Cannot run collision check."
        )

    ref = pd.read_csv(x28_path)
    if "company_name" not in ref.columns:
        raise ValueError(
            f"X28 reference at {x28_path} is missing 'company_name' column. "
            f"Got: {list(ref.columns)}"
        )

    ref["_norm"] = ref["company_name"].apply(_norm)
    # name → list of (company_id, company_name) — preserves duplicate-id firms
    norm_to_refs: dict = {}
    for _, r in ref.iterrows():
        n = r["_norm"]
        if not n:
            continue
        norm_to_refs.setdefault(n, []).append((r.get("company_id"), r["company_name"]))

    def collisions_for_row(row: pd.Series):
        """Return list of (variant, matched_company_id, matched_company_name)."""
        names_to_check = [row["firm_name_canonical"]]
        if isinstance(row.get("firm_name_variants"), str):
            names_to_check += [v.strip() for v in row["firm_name_variants"].split(";")]
        hits = []
        seen = set()
        for v in names_to_check:
            nn = _norm(v)
            if not nn or nn not in norm_to_refs:
                continue
            for cid, cname in norm_to_refs[nn]:
                key = (v.strip(), cname)
                if key in seen:
                    continue
                seen.add(key)
                hits.append({
                    "candidate_variant": v.strip(),
                    "x28_company_id": cid,
                    "x28_company_name": cname,
                })
        return hits

    rows_out = []
    for _, r in dedup.iterrows():
        hits = collisions_for_row(r)
        if not hits:
            continue
        for h in hits:
            rows_out.append({
                "candidate_canonical": r["firm_name_canonical"],
                "candidate_variant":   h["candidate_variant"],
                "x28_company_id":      h["x28_company_id"],
                "x28_company_name":    h["x28_company_name"],
                "candidate_x28_status": r.get("x28_status"),
                "n_rows_collapsed":     r.get("n_rows_collapsed"),
                "firm_locations":       r.get("firm_locations"),
            })

    collisions = pd.DataFrame(rows_out)

    print()
    print("=" * 72)
    print("=== Cross-check against existing X28 firm list ===")
    print(f"  Reference:        {x28_path}  ({len(ref)} firms)")
    print(f"  Candidates:       {len(dedup)} firms")
    print(f"  Collision rows:   {len(collisions)} "
          f"(unique candidates: {collisions['candidate_canonical'].nunique() if len(collisions) else 0})")

    if len(collisions) == 0:
        print()
        print("  No collisions found. Safe to send to Thomas.")
        return collisions

    print()
    print("  WARNING: candidates below may already be in X28. Some are likely")
    print("  legitimate distinct entities (e.g. holding vs operating company)")
    print("  — review each before stripping them from the deliverable.")
    print()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    collisions.to_csv(out_path, index=False)
    print(f"  Wrote {len(collisions)} collision rows -> {out_path}")
    print()
    print(collisions[
        ["candidate_canonical", "candidate_variant", "x28_company_name", "x28_company_id"]
    ].to_string(index=False, max_colwidth=60))
    return collisions

STATUS_PRIORITY = {
    "known_major_parent_missing": 0,
    "llm_rejected_candidate": 1,
    "no_candidate_found": 2,
}
PARENT_SUFFIX = " not in X28"


def extract_parent(row: pd.Series) -> str:
    """Return the canonical cluster key for a candidate row.

    For known_major_parent_missing rows, the cluster key is the parent name
    (e.g. "CHUV", "EPFL"). The parent name is taken from llm_rejected_match
    or reason if they end in " not in X28". Otherwise the cluster key is
    firm_name itself.

    Schema note: older pipeline put "<parent> not in X28" in
    llm_rejected_match; newer pipeline (post-May 2026 finalize subcommand)
    puts it in `reason`. We check both.
    """
    if row.get("x28_status") == "known_major_parent_missing":
        for col in ("llm_rejected_match", "reason"):
            val = row.get(col)
            if isinstance(val, str) and val.endswith(PARENT_SUFFIX):
                return val[: -len(PARENT_SUFFIX)].strip()
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


def dedupe(
    in_path: Path,
    out_path: Path,
    x28_ref_path: Path = DEFAULT_X28_REF,
    collisions_path: Path = DEFAULT_COLLISIONS,
    skip_x28_check: bool = False,
) -> pd.DataFrame:
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    df = pd.read_csv(in_path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    required = {"firm_name", "firm_location", "x28_status", "llm_rejected_match"}
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

    # Step 3: aggregate. Only operate on columns present in input — schema can vary
    # between pipeline runs (e.g. new finalize output lacks the SHP-derived metadata
    # fields the old pipeline added).
    if "n_observations" in df.columns:
        df["n_observations"] = pd.to_numeric(df["n_observations"], errors="coerce")
    if "first_seen_year" in df.columns:
        df["first_seen_year"] = pd.to_numeric(df["first_seen_year"], errors="coerce")

    agg_spec = {
        "firm_name_canonical":  ("cluster_key", "first"),
        "firm_name_variants":   ("firm_name", join_unique),
        "n_rows_collapsed":     ("firm_name", "size"),
        "firm_locations":       ("firm_location", join_unique),
        "x28_status":           ("x28_status", best_status),
        "llm_rejected_match":   ("llm_rejected_match", mode_or_first),
    }
    optional_aggs = [
        ("first_seen_year",      "first_seen_year", "min"),
        ("n_observations_total", "n_observations", "sum"),
        ("years_present_union",  "years_present", join_unique),
        ("noga08_sample",        "noga08_sample", mode_or_first),
        ("mj_mis_noga",          "mj_mis_noga", mode_or_first),
        ("reason",               "reason", mode_or_first),
        ("company_id",           "company_id", mode_or_first),
        ("company_name",         "company_name", mode_or_first),
        ("match_method",         "match_method", mode_or_first),
        ("match_score",          "match_score", "max"),
        ("llm_verdict",          "llm_verdict", mode_or_first),
    ]
    for out_col, in_col, fn in optional_aggs:
        if in_col in df.columns:
            agg_spec[out_col] = (in_col, fn)

    agg = df.groupby("cluster_key", dropna=False).agg(**agg_spec).reset_index(drop=True)

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
