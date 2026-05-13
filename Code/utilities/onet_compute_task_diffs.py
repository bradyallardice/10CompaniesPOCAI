#!/usr/bin/env python3
"""
Compute task additions across O*NET database versions.

For each unique Task ID, determine the earliest O*NET version that contains it
(in Task Statements). Output two tables:

  - Data/onet_task_first_appearance.csv
      One row per unique Task ID with the version where it first appears,
      the SOC code, occupation title, task text, and Task Type.

  - Data/onet_added_tasks_per_version.csv
      Summary: how many new tasks each version added (excluding the baseline).

Also separately tracks "Emerging Tasks" (BLS's own flag for tasks they've
identified as emerging but not yet integrated into the main task list).

Inputs:
  Data/onet_versions/{version}/Task Statements.xlsx
  Data/onet_versions/{version}/Emerging Tasks.xlsx

Usage:
  python3 Code/utilities/onet_compute_task_diffs.py
  python3 Code/utilities/onet_compute_task_diffs.py --baseline-version 20.0
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
ONET_VERSIONS_DIR = DATA_DIR / "onet_versions"

# Legacy v20.0 baseline file (from before we downloaded versioned dirs)
LEGACY_V20_PATH = DATA_DIR / "task_statements_20.xlsx"
LEGACY_V20_LABEL = "20.0"

OUT_FIRST_APPEARANCE = DATA_DIR / "onet_task_first_appearance.csv"
OUT_PER_VERSION = DATA_DIR / "onet_added_tasks_per_version.csv"
OUT_EMERGING = DATA_DIR / "onet_emerging_tasks_aggregated.csv"

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def version_key(v: str) -> tuple[int, int]:
    """Sort O*NET version strings numerically: '5.0' < '20.1' < '30.1'."""
    maj, minor = v.split(".")
    return (int(maj), int(minor))


def discover_versions() -> list[str]:
    """List versions present in Data/onet_versions/, sorted oldest → newest."""
    if not ONET_VERSIONS_DIR.exists():
        raise FileNotFoundError(f"O*NET versions dir not found: {ONET_VERSIONS_DIR}")
    versions = []
    for d in ONET_VERSIONS_DIR.iterdir():
        if d.is_dir() and (d / "Task Statements.xlsx").exists():
            versions.append(d.name)
    versions.sort(key=version_key)
    return versions


def load_task_statements(version: str) -> pd.DataFrame:
    """Load Task Statements for one version. Returns a tagged DataFrame."""
    if version == LEGACY_V20_LABEL and LEGACY_V20_PATH.exists():
        path = LEGACY_V20_PATH
    else:
        path = ONET_VERSIONS_DIR / version / "Task Statements.xlsx"
    df = pd.read_excel(path)
    # Required columns
    required = ["O*NET-SOC Code", "Task ID", "Task", "Title", "Task Type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Version {version} missing cols: {missing}")
    df = df[required].copy()
    df["version"] = version
    return df


def compute_first_appearance(versions: list[str]) -> pd.DataFrame:
    """For each unique Task ID across all versions, find the earliest version
    in which it appears in Task Statements.
    """
    seen_first = {}  # task_id -> (version, soc, title, text, task_type)
    for v in versions:
        df = load_task_statements(v)
        logger.info(f"  v{v}: {len(df):,} task rows, {df['Task ID'].nunique():,} unique Task IDs")
        for _, row in df[~df["Task ID"].isin(seen_first)].iterrows():
            tid = row["Task ID"]
            seen_first[tid] = {
                "Task ID": tid,
                "first_appeared_version": v,
                "onet_soc_code": row["O*NET-SOC Code"],
                "onet_title": row["Title"],
                "task_text": row["Task"],
                "task_type": row["Task Type"],
            }
    result = pd.DataFrame.from_records(list(seen_first.values()))
    result = result.sort_values(["first_appeared_version", "onet_soc_code", "Task ID"]).reset_index(drop=True)
    return result


def summarize_per_version(first_appearance: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Per-version summary of how many *new* tasks each version added.

    Baseline version's tasks are not "new" — they're the starting set.
    """
    df = first_appearance[first_appearance["first_appeared_version"] != baseline].copy()
    summary = (
        df.groupby("first_appeared_version")
          .agg(
              new_tasks=("Task ID", "nunique"),
              n_occupations_touched=("onet_soc_code", "nunique"),
              n_core=("task_type", lambda s: (s == "Core").sum()),
              n_supplemental=("task_type", lambda s: (s == "Supplemental").sum()),
          )
          .reset_index()
          .rename(columns={"first_appeared_version": "version"})
    )
    summary = summary.sort_values("version", key=lambda s: s.map(version_key)).reset_index(drop=True)
    return summary


def aggregate_emerging_tasks(versions: list[str]) -> pd.DataFrame:
    """Track BLS-flagged Emerging Tasks across versions. Each row in an
    Emerging Tasks file is keyed by (O*NET-SOC, Task, Category) — these are
    candidate new tasks that BLS has surfaced but not yet integrated.

    We keep the first version a given (SOC, Task, Category) combo appeared.
    """
    seen = {}
    for v in versions:
        path = ONET_VERSIONS_DIR / v / "Emerging Tasks.xlsx"
        if not path.exists():
            continue
        df = pd.read_excel(path)
        if not {"O*NET-SOC Code", "Task", "Category"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            key = (row["O*NET-SOC Code"], str(row["Task"]).strip(), row.get("Category", ""))
            if key not in seen:
                seen[key] = {
                    "onet_soc_code": key[0],
                    "task_text": key[1],
                    "category": key[2],
                    "first_appeared_in_emerging_version": v,
                    "title": row.get("Title", ""),
                    "original_task_id": row.get("Original Task ID", ""),
                }
    result = pd.DataFrame.from_records(list(seen.values()))
    if len(result) > 0:
        result = result.sort_values(
            ["first_appeared_in_emerging_version", "onet_soc_code"],
            key=lambda s: s.map(version_key) if s.name == "first_appeared_in_emerging_version" else s,
        ).reset_index(drop=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-version", default="20.0",
                        help="Version to treat as the baseline; tasks present "
                             "here are not counted as 'new'. Default: 20.0 (uses "
                             "the legacy task_statements_20.xlsx file).")
    parser.add_argument("--show-samples", type=int, default=10,
                        help="Number of sample new tasks to print per recent version (default: 10)")
    args = parser.parse_args()

    logger.info(f"Discovering O*NET versions in {ONET_VERSIONS_DIR}")
    versions = discover_versions()
    logger.info(f"Found {len(versions)} versions: {versions[0]} → {versions[-1]}")

    # Insert baseline (v20.0 from legacy file) at the front if not already present
    if args.baseline_version not in versions and LEGACY_V20_PATH.exists():
        versions = [args.baseline_version] + versions
        logger.info(f"Prepended baseline v{args.baseline_version} from legacy file: {LEGACY_V20_PATH}")
    elif args.baseline_version not in versions:
        logger.warning(f"Baseline v{args.baseline_version} not available; using {versions[0]} as baseline")
        args.baseline_version = versions[0]

    logger.info("=" * 70)
    logger.info(f"Computing first-appearance table across {len(versions)} versions...")
    logger.info("=" * 70)
    first_appearance = compute_first_appearance(versions)
    logger.info(f"Total unique Task IDs ever seen: {len(first_appearance):,}")

    # Save first-appearance table
    first_appearance.to_csv(OUT_FIRST_APPEARANCE, index=False)
    logger.info(f"✓ Saved: {OUT_FIRST_APPEARANCE} ({len(first_appearance):,} rows)")

    # Per-version summary
    summary = summarize_per_version(first_appearance, args.baseline_version)
    summary.to_csv(OUT_PER_VERSION, index=False)
    logger.info(f"✓ Saved: {OUT_PER_VERSION} ({len(summary):,} rows)")

    logger.info("\n" + "=" * 70)
    logger.info(f"PER-VERSION NEW TASK COUNTS (baseline = v{args.baseline_version})")
    logger.info("=" * 70)
    logger.info(f"{'Version':<10}{'New tasks':>12}{'Occs touched':>16}{'Core':>10}{'Supplemental':>15}")
    logger.info(f"{'-'*10}{'-'*12}{'-'*16}{'-'*10}{'-'*15}")
    for _, row in summary.iterrows():
        logger.info(
            f"{row['version']:<10}{row['new_tasks']:>12,}"
            f"{row['n_occupations_touched']:>16,}"
            f"{row['n_core']:>10,}{row['n_supplemental']:>15,}"
        )
    logger.info(f"{'-'*10}{'-'*12}{'-'*16}{'-'*10}{'-'*15}")
    logger.info(f"{'TOTAL':<10}{summary['new_tasks'].sum():>12,}{'':>16}"
                f"{summary['n_core'].sum():>10,}{summary['n_supplemental'].sum():>15,}")

    # Top occupations by new-task count
    new_only = first_appearance[first_appearance["first_appeared_version"] != args.baseline_version]
    top_occs = (new_only.groupby(["onet_soc_code", "onet_title"])
                .size().reset_index(name="n_new_tasks")
                .sort_values("n_new_tasks", ascending=False).head(15))
    logger.info("\n" + "=" * 70)
    logger.info(f"TOP 15 OCCUPATIONS BY NEW-TASK COUNT (since v{args.baseline_version})")
    logger.info("=" * 70)
    for _, row in top_occs.iterrows():
        logger.info(f"  {row['onet_soc_code']}  {str(row['onet_title'])[:55]:<55}  {row['n_new_tasks']:>4}")

    # Sample new tasks from recent versions
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE NEW TASKS FROM RECENT VERSIONS")
    logger.info("=" * 70)
    recent_versions = sorted(set(new_only["first_appeared_version"]),
                             key=version_key, reverse=True)[:5]
    for v in recent_versions:
        sub = new_only[new_only["first_appeared_version"] == v]
        logger.info(f"\nv{v} — {len(sub):,} new tasks. Sample:")
        for _, row in sub.head(args.show_samples).iterrows():
            text = str(row["task_text"])
            text_preview = text[:100] + ("..." if len(text) > 100 else "")
            logger.info(f"  [{row['onet_soc_code']}] {text_preview}")

    # Aggregate Emerging Tasks
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATING BLS EMERGING TASKS")
    logger.info("=" * 70)
    emerging = aggregate_emerging_tasks(versions)
    if len(emerging) > 0:
        emerging.to_csv(OUT_EMERGING, index=False)
        logger.info(f"✓ Saved: {OUT_EMERGING} ({len(emerging):,} unique emerging tasks)")
        if "category" in emerging.columns:
            cat_counts = emerging["category"].value_counts().to_dict()
            logger.info(f"  Categories: {cat_counts}")
        # Top occupations by emerging-task count
        top_em = (emerging.groupby(["onet_soc_code", "title"])
                  .size().reset_index(name="n").sort_values("n", ascending=False).head(10))
        logger.info(f"\n  Top 10 occupations by emerging-task count:")
        for _, row in top_em.iterrows():
            logger.info(f"    {row['onet_soc_code']}  {str(row['title'])[:55]:<55}  {row['n']:>4}")
    else:
        logger.info("No Emerging Tasks files found.")

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
