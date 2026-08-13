#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$REPO_ROOT/pipeline/step3_point_prediction/run_step3_parallel.sh"
TEST_ROOT="$(mktemp -d /tmp/tasa-step3-runner.XXXXXX)"
trap 'rm -rf "$TEST_ROOT"' EXIT

mkdir -p \
  "$TEST_ROOT/bin" \
  "$TEST_ROOT/conda/etc/profile.d" \
  "$TEST_ROOT/input root/val" \
  "$TEST_ROOT/logs"

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
printf '%s\n' "$@" > "$STEP3_CAPTURE"
EOF
chmod +x "$TEST_ROOT/bin/conda" "$TEST_ROOT/bin/python"

CAPTURE="$TEST_ROOT/argv"
FAKE_CONDA_BASE="$TEST_ROOT/conda" \
STEP3_CAPTURE="$CAPTURE" \
PATH="$TEST_ROOT/bin:/usr/bin:/bin" \
REPO_ROOT="$REPO_ROOT" \
ENV_NAME=test \
GPUS=7 \
INPUT_ROOT=/unused/input \
OUTPUT_ROOT=/unused/output \
LOG_DIR="$TEST_ROOT/logs" \
  "$RUNNER" val \
    --input-root "$TEST_ROOT/input root" \
    --output-root "$TEST_ROOT/output root"

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
  echo "ERROR: runner did not forward $wanted" >&2
  return 1
}

[ "$(value_after --split)" = "val" ]
[ "$(value_after --input_root)" = "$TEST_ROOT/input root" ]
[ "$(value_after --output_root)" = "$TEST_ROOT/output root" ]
[ "$(value_after --num_shards)" = "1" ]
[ "$(value_after --shard)" = "0" ]
[ -d "$TEST_ROOT/output root/val" ]

echo "Step 3 runner input/output forwarding test passed."
