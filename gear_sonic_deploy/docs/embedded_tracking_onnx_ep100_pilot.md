# Embedded tracking ONNX ep100 pilot

This feature is opt-in. `./deploy.sh real` continues to select the legacy
SONIC decoder/encoder/policy path. The embedded policy is selected only with
`--policy-type embedded_tracking_onnx` and never passes through SONIC's
encoder or decoder.

## Package dry-run

```bash
./gear_sonic_deploy/deploy.sh sim \
  --policy-type embedded_tracking_onnx \
  --tracking-onnx /path/to/package/policy/flip360_general_tracking_ep100_pick100_fixed62_21_relativeyaw.onnx \
  --tracking-control /path/to/package/policy/flip360_general_tracking_ep100_pick100_fixed62_21_relativeyaw.control.json \
  --motion-data /path/to/package/reference \
  --init-state /path/to/package/init/common_initial_state.npz \
  --embedded-tracking-dry-run
```

The dry-run performs no DDS setup and sends no motor command. It checks the
control/init/reference schemas, TensorRT inputs and output, the 994-element
observation, finite anchor, direct hardware-order target mapping and final
time-step clamp.

## Real pilot

Use a short, explicit pilot limit first:

```bash
./gear_sonic_deploy/deploy.sh real \
  --policy-type embedded_tracking_onnx \
  --tracking-onnx /path/to/package/policy/flip360_general_tracking_ep100_pick100_fixed62_21_relativeyaw.onnx \
  --tracking-control /path/to/package/policy/flip360_general_tracking_ep100_pick100_fixed62_21_relativeyaw.control.json \
  --motion-data /path/to/package/reference \
  --init-state /path/to/package/init/common_initial_state.npz \
  --tracking-max-ticks 25 \
  --tracking-max-first-target-error-rad 0.5 \
  --input-type keyboard
```

The controller ramps to `common_initial_state.npz`, starts `time_step` at 0,
runs at 50 Hz, and holds the final command without looping. Actions, scales,
Kp and Kd remain in direct G1 hardware/MuJoCo order. `dq_target` and `tau_ff`
are zero.

## Current package safety finding

On the supplied 2026-08-22 package, frame-0 TensorRT inference produces a
maximum `|q_target - common_init_qpos|` of approximately **1.4451 rad at
hardware joint 18**. The default 0.5-rad guard therefore rejects the first
command. Do not disable or raise this guard on the real robot until the
initial-pose/reference mismatch has been reviewed and an intentional
transition strategy has been approved.

Passing `--tracking-max-first-target-error-rad 0` disables the guard. This is
provided for offline inference contract testing only; it is not a safe real
robot default.
