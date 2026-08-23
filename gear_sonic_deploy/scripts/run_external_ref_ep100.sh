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

# TensorRT loads its builder resource with dlopen() the first time an ONNX
# policy is converted.  Unlike the linked libraries, that resource is not
# found through the executable's RUNPATH, so its directory must be present in
# LD_LIBRARY_PATH at runtime.
TENSORRT_ROOT="${TensorRT_ROOT:-/home/chuye/TensorRT}"
TENSORRT_LIB_DIR=""
for candidate in \
  "$TENSORRT_ROOT/lib" \
  "$TENSORRT_ROOT/lib64" \
  "$TENSORRT_ROOT/targets/$(uname -m)-linux-gnu/lib"; do
  if [[ -f "$candidate/libnvinfer_builder_resource.so.10.13.0" ]]; then
    TENSORRT_LIB_DIR="$candidate"
    break
  fi
done
if [[ -z "$TENSORRT_LIB_DIR" ]]; then
  echo "Missing TensorRT builder resource under: $TENSORRT_ROOT" >&2
  exit 1
fi
export TensorRT_ROOT
export LD_LIBRARY_PATH="$TENSORRT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

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
  MAX_TICKS="${EXTERNAL_REF_MAX_TICKS:-0}"
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
  MAX_TICKS="${EXTERNAL_REF_MAX_TICKS:-0}"
  echo "REAL ROBOT external-reference K/I/P replay: interface=$NETWORK_INTERFACE gain=$GAIN_SCALE"
fi

cmake -S . -B build_external_ref -DCMAKE_BUILD_TYPE=Release
cmake --build build_external_ref --target g1_deploy_onnx_ref -j"$(nproc)"

PLANNER="${EXTERNAL_REF_PLANNER:-/home/chuye/GR00T-WholeBodyControl/gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx}"
if [[ ! -f "$PLANNER" ]]; then
  echo "Missing SONIC planner required by the K/I/P positioning flow: $PLANNER" >&2
  exit 1
fi

arguments=(
  "$NETWORK_INTERFACE" "$POLICY" "$REPO_DIR/reference/example"
  --policy-type external_ref_tracking
  --external-reference-source stream
  --external-control "$CONTROL"
  --external-init-state "$INIT_STATE"
  --external-max-ticks "$MAX_TICKS"
  --external-gain-scale "$GAIN_SCALE"
  --external-max-first-error "${EXTERNAL_REF_MAX_FIRST_ERROR:-0}"
  --external-max-target-step "${EXTERNAL_REF_MAX_TARGET_STEP:-0}"
  --external-log "$LOG_DIR/${MODE}_control.csv"
  --planner-file "$PLANNER"
  --input-type zmq_manager
  --output-type zmq
  --zmq-host localhost
)
if [[ -n "$CRC_FLAG" ]]; then arguments+=("$CRC_FLAG"); fi
if [[ "${EXTERNAL_REF_PRINT_COMMAND:-0}" == "1" ]]; then
  printf '%q ' target/release/g1_deploy_onnx_ref "${arguments[@]}"
  printf '\n'
  exit 0
fi
exec target/release/g1_deploy_onnx_ref "${arguments[@]}"
