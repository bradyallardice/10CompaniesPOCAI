#!/usr/bin/env bash
# Run stage_5 three times to produce the _fo / _oy / _o exposure level files
# that stage_6_shp_exposure.py consumes via --exposure-file-firm-occ /
# --exposure-file-occ-year / --exposure-file-occ.
#
# The existing pct_05 / ce_0.0 / BGE bge5 _foy file lives at:
#   Data/Testing/stage_5_expertise/full_sample/isco_firm_year_exposure_core_tasks_pct_05_ce_0.0.csv
# (the Feb 7 ~/Dropbox/.../_BGE.csv copy is from older stage_4 data —
# avoid mixing provenance, use the May 11 file paired with the Apr 11
# stage_4 parquet below for all 4 levels.)
#
# Each variant runs sequentially (not parallel) to avoid 3x memory pressure.

set -euo pipefail

PROJECT="/Users/bradyallardice/Dropbox/Allardice/KurerAllardice2024/10CompaniesPOCAI"
cd "$PROJECT"

STAGE4_DIR="Data/Testing/stage_4_expertise/full_sample/skip_ce"
STAGE4_FILE="task_exposure_matches_all_thresholds_bge_bge20_15_10_5_1_ce0p8_0p6_0p4_0p2_onet20_core.parquet"

# Shared args. --bge-percentiles 5 and --ce-thresholds 0.0 pin the spec to
# what the existing _foy file uses. --model bge matches the parquet name.
COMMON_ARGS=(
  --stage-4-dir "$STAGE4_DIR"
  --top-matches-file "$STAGE4_FILE"
  --model bge
  --bge-percentiles 5
  --ce-thresholds 0.0
  --task-type core
  --onet-version 20
)

run_one () {
  local label="$1"; shift
  local outdir="$1"; shift
  mkdir -p "$outdir"
  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running stage_5: $label -> $outdir"
  echo "============================================================"
  python3 stage_5_onet_to_isco_exposure.py "${COMMON_ARGS[@]}" --output-dir "$outdir" "$@"
}

# _fo: firm × occupation, time-invariant
run_one "fo (firm-occ, time-invariant)" \
  "Data/stage_5_levels/fo" \
  --exposure-table-format exposed-firm-occ-year \
  --time-invariant

# _oy: occupation × year (no firm dimension)
run_one "oy (occ-year)" \
  "Data/stage_5_levels/oy" \
  --exposure-table-format exposed-occ-year

# _o: occupation only, time-invariant
run_one "o (occ, time-invariant)" \
  "Data/stage_5_levels/o" \
  --exposure-table-format exposed-occ-year \
  --time-invariant

echo "============================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 3 stage_5 runs complete"
echo "============================================================"
