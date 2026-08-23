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

## MuJoCo closed-loop gate

MuJoCo replaces only the physical robot. The normal C++ deployment binary
still builds the 994-element observation, runs TensorRT at 50 Hz, advances the
reference cursor, and publishes Unitree `rt/lowcmd`. The existing simulator
publishes MuJoCo joint/IMU/root state on `rt/lowstate`/`rt/odostate` and steps
physics at 200 Hz with:

```text
tau = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
```

Install the isolated simulator environment once from this worktree:

```bash
bash install_scripts/install_mujoco_sim.sh
```

Terminal A starts the instrumented robot replacement from the package common
initial state. It validates all 29 joint/qpos/qvel/actuator mappings before
publishing any state, holds the elastic support through C++ initialization,
waits until it detects the first policy command, then releases the support and
writes a JSON report plus an annotated MuJoCo MP4:

```bash
source .venv_sim/bin/activate

PACKAGE_DIR=/home/chuye/g1_general_tracking_onnx_ep100_deploy_package_20260822
REPORT_DIR=/tmp/embedded_tracking_mujoco_ep100_250ticks

python gear_sonic/scripts/run_embedded_tracking_mujoco_validation.py \
  --init-state "$PACKAGE_DIR/init/common_initial_state.npz" \
  --tracking-control "$PACKAGE_DIR/policy/flip360_general_tracking_ep100_pick100_fixed62_21_relativeyaw.control.json" \
  --output "$REPORT_DIR/report.json" \
  --video "$REPORT_DIR/fall.mp4" \
  --command-timeout-s 60 \
  --policy-start-timeout-s 45 \
  --release-after-command-s 4 \
  --duration-after-release-s 8
```

Terminal B runs the same C++ policy-to-LowCmd path. The zero first-target
threshold below is permitted only for this deliberately adverse simulator
test; keep the default guard enabled on hardware:

```bash
PACKAGE_DIR=/home/chuye/g1_general_tracking_onnx_ep100_deploy_package_20260822
REPORT_DIR=/tmp/embedded_tracking_mujoco_ep100_250ticks

./gear_sonic_deploy/deploy.sh \
  --policy-type embedded_tracking_onnx \
  --tracking-onnx "$PACKAGE_DIR/policy/flip360_general_tracking_ep100_pick100_fixed62_21_relativeyaw.onnx" \
  --tracking-control "$PACKAGE_DIR/policy/flip360_general_tracking_ep100_pick100_fixed62_21_relativeyaw.control.json" \
  --motion-data "$PACKAGE_DIR/reference" \
  --init-state "$PACKAGE_DIR/init/common_initial_state.npz" \
  --tracking-max-ticks 250 \
  --tracking-max-first-target-error-rad 0 \
  --input-type keyboard \
  --enable-csv-logs \
  --logs-dir "$REPORT_DIR/cpp" \
  sim
```

Confirm that the launcher says `Environment: sim` and `Network Interface: lo`.
Press `]` after `Init Done`. The validator does not release support merely
because a LowCmd exists: it also requires a command step above 0.1 rad, which
distinguishes the embedded policy from the initialization ramp. It fails on a
fall, non-finite state, excessive root tilt/height loss/joint speed, a command
step above 0.5 rad, excessive torque saturation, or a Kp/Kd/dq/tau contract
mismatch.

The deploy launcher now rejects unknown `--options` instead of treating them
as a network interface. This prevents a misspelled simulator flag from silently
selecting the real-robot path.

## Real pilot

Only use this section after the dry-run and MuJoCo gate both pass. Start with a
short, explicit pilot limit:

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

The 250-tick closed-loop MuJoCo run also **fails the hardware gate**. The DDS
and command contract itself is correct: direct 29-DoF mapping matched, Kp/Kd
matched the control JSON within `1e-6`, and both `dq_target` and `tau_ff` stayed
zero. However, the first policy command changed the left-elbow target by
approximately **1.4452 rad**. After support release, MuJoCo triggered its fall
detector after approximately **4.65 s**; the run reached 98.57 degrees root
tilt, 0.2079 m minimum pre-reset root height, and 12.81 rad/s maximum body-joint
speed. Peak commanded body torque was 58.39 Nm and torque clipping occurred in
about 1.4% of released simulation samples.

MuJoCo cannot prove real-hardware stability, but this failed gate plus the
first-target discontinuity is enough to reject the current package for a real
pilot. Keep the real-robot guard enabled and do not deploy this checkpoint until
the initialization/reference transition or checkpoint has been corrected and
the closed-loop gate passes.
