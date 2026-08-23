#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-dry-run}"
PACKAGE_DIR="${EXTERNAL_REF_PACKAGE:-/home/chuye/g1_general_tracking_external_ref_package_20260822}"
POLICY="$PACKAGE_DIR/policy/flip360_pure_tracking_decoder.onnx"
REFERENCE="$PACKAGE_DIR/reference/ep100_pick100_goldprefix_fixed62_21_relativeyaw_motionlib_ref.onnx"
CONTROL="$PACKAGE_DIR/policy/control_params_from_onnx_metadata.json"
INIT_STATE="$PACKAGE_DIR/reference/ep100_common_initial_state.npz"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${EXTERNAL_REF_LOG_DIR:-$REPO_DIR/logs/external_ref_ep100}"
mkdir -p "$LOG_DIR"

for required in "$POLICY" "$REFERENCE" "$CONTROL" "$INIT_STATE"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required ep100 package file: $required" >&2
    exit 1
  fi
done

cd "$REPO_DIR"

case "$MODE" in
  dry-run)
    cmake -S . -B build_external_ref -DCMAKE_BUILD_TYPE=Release
    cmake --build build_external_ref --target external_ref_dry_run -j"$(nproc)"
    exec target/release/external_ref_dry_run \
      "$POLICY" "$REFERENCE" "$CONTROL" "$INIT_STATE" 941 \
      "${EXTERNAL_REF_DRY_FRAMES:-5}" "$LOG_DIR/dry_run.csv"
    ;;
  sim|real)
    ;;
  *)
    echo "Usage: $0 {dry-run|sim|real}" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "sim" ]]; then
  NETWORK_INTERFACE="lo"
  CRC_FLAG="--disable-crc-check"
  GAIN_SCALE="${EXTERNAL_REF_GAIN_SCALE:-1.0}"
  MAX_TICKS="${EXTERNAL_REF_MAX_TICKS:-941}"
else
  NETWORK_INTERFACE="${EXTERNAL_REF_REAL_INTERFACE:-}"
  if [[ -z "$NETWORK_INTERFACE" ]]; then
    NETWORK_INTERFACE="$(ip -o -4 addr show | awk '$4 ~ /^192\.168\.123\./ {print $2; exit}')"
  fi
  if [[ -z "$NETWORK_INTERFACE" ]]; then
    echo "No 192.168.123.x robot interface found; set EXTERNAL_REF_REAL_INTERFACE." >&2
    exit 1
  fi
  CRC_FLAG=""
  GAIN_SCALE="${EXTERNAL_REF_GAIN_SCALE:-1.0}"
  MAX_TICKS="${EXTERNAL_REF_MAX_TICKS:-941}"
  echo "REAL ROBOT external-reference replay: interface=$NETWORK_INTERFACE gain=$GAIN_SCALE max_ticks=$MAX_TICKS"
fi

cmake -S . -B build_external_ref -DCMAKE_BUILD_TYPE=Release
cmake --build build_external_ref --target g1_deploy_onnx_ref -j"$(nproc)"

arguments=(
  "$NETWORK_INTERFACE" "$POLICY" "$PACKAGE_DIR/reference"
  --policy-type external_ref_tracking
  --external-reference "$REFERENCE"
  --external-control "$CONTROL"
  --external-init-state "$INIT_STATE"
  --external-reference-frames 941
  --external-max-ticks "$MAX_TICKS"
  --external-gain-scale "$GAIN_SCALE"
  --external-max-first-error "${EXTERNAL_REF_MAX_FIRST_ERROR:-0}"
  --external-max-target-step "${EXTERNAL_REF_MAX_TARGET_STEP:-0}"
  --external-log "$LOG_DIR/${MODE}_control.csv"
  --input-type keyboard
  --output-type zmq
)
if [[ -n "$CRC_FLAG" ]]; then arguments+=("$CRC_FLAG"); fi
exec target/release/g1_deploy_onnx_ref "${arguments[@]}"
