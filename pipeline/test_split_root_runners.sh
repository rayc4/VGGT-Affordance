#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_ROOT="$(mktemp -d /tmp/tasa-split-runners.XXXXXX)"
trap 'rm -rf "$TEST_ROOT"' EXIT

mkdir -p \
  "$TEST_ROOT/bin" \
  "$TEST_ROOT/conda/etc/profile.d" \
  "$TEST_ROOT/step4 root/val" \
  "$TEST_ROOT/step6 root/val" \
  "$TEST_ROOT/data"

cat > "$TEST_ROOT/bin/conda" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = "info" ] && [ "${2:-}" = "--base" ]; then
  printf '%s\n' "$FAKE_CONDA_BASE"
  exit 0
fi
exit 1
EOF
cat > "$TEST_ROOT/conda/etc/profile.d/conda.sh" <<'EOF'
conda() {
  [ "${1:-}" = "activate" ]
}
EOF
cat > "$TEST_ROOT/bin/python" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "$STEP5_CAPTURE"
EOF
chmod +x "$TEST_ROOT/bin/conda" "$TEST_ROOT/bin/python"

CAPTURE="$TEST_ROOT/step5-argv"
FAKE_CONDA_BASE="$TEST_ROOT/conda" \
STEP5_CAPTURE="$CAPTURE" \
PATH="$TEST_ROOT/bin:/usr/bin:/bin" \
REPO_ROOT="$REPO_ROOT" \
ENV_NAME=test \
GPUS=7 \
LOG_DIR="$TEST_ROOT/step5-logs" \
  "$REPO_ROOT/pipeline/step5_molmo_sam/run_step5_parallel.sh" val \
    --input-root "$TEST_ROOT/step4 root" \
    --output-root "$TEST_ROOT/step5 root"

mapfile -t ARGV < "$CAPTURE"
value_after() {
  local wanted="$1"
  local i
  for i in "${!ARGV[@]}"; do
    if [ "${ARGV[$i]}" = "$wanted" ]; then
      printf '%s\n' "${ARGV[$((i + 1))]}"
      return 0
    fi
  done
  echo "ERROR: missing argument $wanted" >&2
  return 1
}
[ "$(value_after --input_root)" = "$TEST_ROOT/step4 root" ]
[ "$(value_after --output_root)" = "$TEST_ROOT/step5 root" ]
[ -d "$TEST_ROOT/step5 root/val" ]

DRY_OUTPUT="$TEST_ROOT/step6b-dry-run"
FAKE_CONDA_BASE="$TEST_ROOT/conda" \
PATH="$TEST_ROOT/bin:/usr/bin:/bin" \
REPO_ROOT="$REPO_ROOT" \
ENV_NAME=test \
GPUS=7 \
DATA_ROOT="$TEST_ROOT/data" \
LOG_DIR="$TEST_ROOT/step6b-logs" \
  "$REPO_ROOT/pipeline/step6b_vggt_track/run_step6b_parallel.sh" val \
    --input-root "$TEST_ROOT/step6 root" \
    --output-root "$TEST_ROOT/step6b root" \
    --dry-run > "$DRY_OUTPUT"

grep -F -- "--input_root $TEST_ROOT/step6\\ root" "$DRY_OUTPUT"
grep -F -- "--output_root $TEST_ROOT/step6b\\ root" "$DRY_OUTPUT"
[ -d "$TEST_ROOT/step6b root/val" ]

echo "Step 5 and Step 6b split-root runner tests passed."
