#!/usr/bin/env python3
"""
Download O*NET task-related Excel files across all O*NET database versions.

For each version listed at https://www.onetcenter.org/db_releases.html, this
script downloads four files:
  - Task Statements.xlsx
  - Task Ratings.xlsx
  - Task Categories.xlsx
  - Emerging Tasks.xlsx

Files are saved to Data/onet_versions/{version}/ where {version} is the
release string (e.g. "30.1", "20.0").

Older versions may not have all four files (e.g. Emerging Tasks was introduced
in a later release). 404s are logged and skipped.

URL pattern used:
  https://www.onetcenter.org/dl_files/database/db_{V}_excel/{Filename}.xlsx
  where {V} is the version with '.' replaced by '_'.

Usage:
  # Download every version (resumes from cache for files already present)
  python3 Code/utilities/download_onet_versions.py

  # Download only specific versions
  python3 Code/utilities/download_onet_versions.py --version 30.1 29.0 20.0

  # Just list what would be downloaded
  python3 Code/utilities/download_onet_versions.py --dry-run

  # Force re-download even if files already exist
  python3 Code/utilities/download_onet_versions.py --force
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Data" / "onet_versions"

RELEASE_PAGE_URL = "https://www.onetcenter.org/db_releases.html"
FILE_URL_TEMPLATE = (
    "https://www.onetcenter.org/dl_files/database/db_{v_underscore}_excel/{filename}"
)

# The four task-related files we want for every release.
TASK_FILES = [
    "Task Statements.xlsx",
    "Task Ratings.xlsx",
    "Task Categories.xlsx",
    "Emerging Tasks.xlsx",
]

logging.basicConfig(
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def discover_versions(timeout: int = 30) -> list[str]:
    """Scrape the O*NET release page and return the list of version strings.

    Versions look like '30.1', '29.3', '20.0', '5.0', etc. We pull them out of
    any links on the page that match the download URL pattern, falling back to
    a hardcoded list (current at time of writing) if the page structure shifts.
    """
    fallback = [
        "30.1", "30.0", "29.3", "29.2", "29.1", "29.0",
        "28.3", "28.2", "28.1", "28.0",
        "27.3", "27.2", "27.1", "27.0",
        "26.3", "26.2", "26.1", "26.0",
        "25.3", "25.2", "25.1", "25.0",
        "24.3", "24.2", "24.1", "24.0",
        "23.3", "23.2", "23.1", "23.0",
        "22.3", "22.2", "22.1", "22.0",
        "21.3", "21.2", "21.1", "21.0",
        "20.3", "20.2", "20.1", "20.0",
        "19.0", "18.1", "18.0", "17.0", "16.0",
        "15.1", "15.0", "14.0", "13.0", "12.0", "11.0", "10.0",
        "9.0", "8.0", "7.0", "6.0", "5.1", "5.0",
    ]

    try:
        r = requests.get(RELEASE_PAGE_URL, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Could not fetch release page ({e}); using fallback list")
        return fallback

    # Match db_{V}_excel.zip-style links — pull out the version part.
    found = set(re.findall(r"db_(\d+)_(\d+)_excel", r.text))
    versions = [f"{maj}.{minor}" for maj, minor in found]
    if not versions:
        logger.warning("No versions parsed from release page; using fallback list")
        return fallback

    # Sort descending: by major then minor (numeric).
    versions.sort(key=lambda v: tuple(int(x) for x in v.split(".")), reverse=True)
    return versions


def file_url(version: str, filename: str) -> str:
    """Build the file URL for a given version and filename.

    O*NET URL-encodes the space in the directory portion, so we use quote
    to handle the filename too. The version's dot is replaced with underscore.
    """
    v_underscore = version.replace(".", "_")
    return FILE_URL_TEMPLATE.format(
        v_underscore=v_underscore,
        filename=quote(filename),
    )


def download_file(url: str, dest: Path, *, force: bool = False,
                  timeout: int = 120) -> str:
    """Download a single file. Returns 'cached', 'downloaded', or 'missing'."""
    if dest.exists() and not force:
        size_kb = dest.stat().st_size / 1024
        logger.info(f"    cached: {dest.name} ({size_kb:.1f} KB)")
        return "cached"

    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, timeout=timeout, stream=True) as r:
            if r.status_code == 404:
                logger.info(f"    404:    {dest.name}")
                return "missing"
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=32 * 1024):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        logger.error(f"    ERROR:  {dest.name} — {e}")
        # Clean up a partial write.
        if dest.exists():
            dest.unlink()
        return "missing"

    size_kb = dest.stat().st_size / 1024
    logger.info(f"    got:    {dest.name} ({size_kb:.1f} KB)")
    return "downloaded"


def download_version(version: str, output_dir: Path, *, force: bool,
                     sleep_seconds: float) -> dict[str, int]:
    """Download all task files for one O*NET version."""
    logger.info(f"O*NET v{version}:")
    version_dir = output_dir / version
    counts = {"downloaded": 0, "cached": 0, "missing": 0}
    for filename in TASK_FILES:
        url = file_url(version, filename)
        dest = version_dir / filename
        status = download_file(url, dest, force=force)
        counts[status] += 1
        if status == "downloaded" and sleep_seconds > 0:
            time.sleep(sleep_seconds)
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--version", nargs="+",
        help="Specific versions to download (e.g., 30.1 29.0). Default: all "
             "discovered versions.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download files that already exist locally.",
    )
    parser.add_argument(
        "--sleep", type=float, default=0.3,
        help="Seconds to pause between downloads (default: 0.3)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the versions that would be processed and exit.",
    )
    args = parser.parse_args()

    if args.version:
        versions = list(args.version)
        logger.info(f"Using {len(versions)} CLI-specified versions: {versions}")
    else:
        logger.info(f"Discovering versions from {RELEASE_PAGE_URL}")
        versions = discover_versions()
        logger.info(f"Found {len(versions)} versions")

    if args.dry_run:
        logger.info("DRY RUN — would process the following versions:")
        for v in versions:
            logger.info(f"  {v}  →  {args.output_dir / v}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")

    grand = {"downloaded": 0, "cached": 0, "missing": 0}
    failed = []
    for v in versions:
        try:
            counts = download_version(
                v, args.output_dir, force=args.force, sleep_seconds=args.sleep
            )
            for k in grand:
                grand[k] += counts[k]
        except Exception as e:  # noqa: BLE001
            logger.error(f"Unexpected error on v{v}: {e}")
            failed.append(v)

    logger.info("=" * 60)
    logger.info(
        f"Done. Versions processed: {len(versions)} | "
        f"downloaded: {grand['downloaded']} | "
        f"cached: {grand['cached']} | "
        f"missing (404/error): {grand['missing']}"
    )
    if failed:
        logger.warning(f"Versions that errored: {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
