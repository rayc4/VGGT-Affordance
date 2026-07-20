#!/usr/bin/env bash
#
# Parallel SceneFun3D downloader.
#
# Splits the benchmark file list into chunks and runs one downloader process per
# chunk concurrently, so scenes download in parallel instead of one-at-a-time.
# Pairs with the aria2c multi-connection support in
# data_downloader/download_utils/download_data.py: each process uses aria2c
# per file when it is installed.
#
# Re-running is safe: already-downloaded files are skipped, partial files resume.
#
# Usage:
#   scripts/download_parallel.sh
#
# Override defaults via env vars, e.g.:
#   N_JOBS=12 TOTAL_CONN=48 DOWNLOAD_DIR=train_val_set scripts/download_parallel.sh
#
set -euo pipefail

# ---- config (override via env) -------------------------------------------
# Directory that contains the `data_downloader` package and `benchmark_file_lists`.
SCENEFUN3D_DIR="${SCENEFUN3D_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scenefun3d}"
SPLIT="${SPLIT:-train_val_set}"                 # train_val_set | test_set
DOWNLOAD_DIR="${DOWNLOAD_DIR:-train_val_set}"
N_JOBS="${N_JOBS:-8}"                            # number of concurrent processes
TOTAL_CONN="${TOTAL_CONN:-32}"                   # target total aria2c connections to the server
ASSETS="${ASSETS:-hires_wide hires_wide_intrinsics hires_poses hires_depth}"
# --------------------------------------------------------------------------

cd "$SCENEFUN3D_DIR"

CSV="benchmark_file_lists/${SPLIT}_only_one_video.csv"
[ -f "$CSV" ] || { echo "ERROR: CSV not found: $CSV (in $SCENEFUN3D_DIR)" >&2; exit 1; }

# Keep per-IP connection count reasonable: split TOTAL_CONN across the jobs.
CONN_PER_FILE=$(( TOTAL_CONN / N_JOBS ))
[ "$CONN_PER_FILE" -lt 1 ] && CONN_PER_FILE=1
export SF3D_ARIA2_CONNECTIONS="$CONN_PER_FILE"

# Scenes = CSV rows minus header. chunk size = ceil(rows / N_JOBS).
ROWS=$(( $(wc -l < "$CSV") - 1 ))
CHUNK_SIZE=$(( (ROWS + N_JOBS - 1) / N_JOBS ))

command -v aria2c >/dev/null 2>&1 \
  && echo "aria2c: yes (SF3D_ARIA2_CONNECTIONS=$CONN_PER_FILE per file)" \
  || echo "aria2c: no (falling back to curl) -- install with: conda install -c conda-forge aria2"
echo "scenes=$ROWS  jobs=$N_JOBS  chunk_size=$CHUNK_SIZE  download_dir=$DOWNLOAD_DIR"
echo "assets: $ASSETS"

# Clean any stale chunk files from a previous run (does not match the source CSV).
CHUNK_GLOB="benchmark_file_lists/${SPLIT}_only_one_video_*.csv"
rm -f $CHUNK_GLOB

conda activate scenefun

python -m data_downloader.download_utils.split_csv_into_chunks \
  --csv_file "$CSV" --number_of_chunks "$CHUNK_SIZE"

pids=()
for f in $CHUNK_GLOB; do
  echo "[*] launching chunk: $f"
  python -m data_downloader.data_asset_download \
    --split custom --video_id_csv "$f" \
    --download_dir "$DOWNLOAD_DIR" \
    --dataset_assets $ASSETS &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

rm -f $CHUNK_GLOB

if [ "$fail" -eq 0 ]; then
  echo "All downloads finished."
else
  echo "Some chunks failed -- just re-run this script to resume (existing files are skipped)." >&2
  exit 1
fi
