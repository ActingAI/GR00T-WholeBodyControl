"""Closed-loop MuJoCo safety check for the embedded tracking C++ runtime.

This process replaces only the physical G1.  The production C++ deployment
binary still owns observation construction, TensorRT inference, time-step
advancement, and LowCmd publication over Unitree DDS.  This process publishes
MuJoCo state on the normal LowState/OdoState topics and applies LowCmd with the
same PD equation used by ``run_sim_loop.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the normal Unitree DDS MuJoCo bridge from the package common "
            "initial state, release the elastic support, and write safety metrics."
        )
    )
    parser.add_argument("--init-state", type=Path, required=True)
    parser.add_argument("--tracking-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interface", default="sim")
    parser.add_argument("--onscreen", action="store_true")
    parser.add_argument("--command-timeout-s", type=float, default=30.0)
    parser.add_argument("--policy-start-timeout-s", type=float, default=45.0)
    parser.add_argument(
        "--policy-start-command-step-rad",
        type=float,
        default=0.10,
        help="q-command change used to detect that C++ left its init/WAIT state.",
    )
    parser.add_argument(
        "--release-after-command-s",
        type=float,
        default=4.0,
        help="Support-band hold after the first C++ LowCmd (the C++ init ramp is 3 s).",
    )
    parser.add_argument("--duration-after-release-s", type=float, default=8.0)
    parser.add_argument("--min-root-height-m", type=float, default=0.35)
    parser.add_argument("--max-root-tilt-deg", type=float, default=60.0)
    parser.add_argument("--max-body-joint-speed-rad-s", type=float, default=37.0)
    parser.add_argument("--max-command-step-rad", type=float, default=0.5)
    parser.add_argument("--max-torque-saturation-fraction", type=float, default=0.10)
    return parser.parse_args()


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value}")


def _load_inputs(
    init_state_path: Path, control_path: Path
) -> tuple[dict[str, np.ndarray | float], dict[str, Any]]:
    if not init_state_path.is_file():
        raise FileNotFoundError(f"init state not found: {init_state_path}")
    if not control_path.is_file():
        raise FileNotFoundError(f"tracking control not found: {control_path}")

    with np.load(init_state_path, allow_pickle=False) as data:
        required = {
            "body_qpos_mujoco",
            "body_qvel_mujoco",
            "root_quat_wxyz",
            "root_height_m",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"init state is missing fields: {missing}")
        init_state: dict[str, np.ndarray | float] = {
            "body_qpos_mujoco": np.asarray(data["body_qpos_mujoco"], dtype=np.float64),
            "body_qvel_mujoco": np.asarray(data["body_qvel_mujoco"], dtype=np.float64),
            "root_quat_wxyz": np.asarray(data["root_quat_wxyz"], dtype=np.float64),
            "root_height_m": float(np.asarray(data["root_height_m"]).item()),
        }

    with control_path.open(encoding="utf-8") as stream:
        control = json.load(stream)

    qpos = init_state["body_qpos_mujoco"]
    qvel = init_state["body_qvel_mujoco"]
    quat = init_state["root_quat_wxyz"]
    if not isinstance(qpos, np.ndarray) or qpos.shape != (29,):
        raise ValueError(f"body_qpos_mujoco must have shape (29,), got {qpos.shape}")
    if not isinstance(qvel, np.ndarray) or qvel.shape != (29,):
        raise ValueError(f"body_qvel_mujoco must have shape (29,), got {qvel.shape}")
    if not isinstance(quat, np.ndarray) or quat.shape != (4,):
        raise ValueError(f"root_quat_wxyz must have shape (4,), got {quat.shape}")
    if not all(np.all(np.isfinite(value)) for value in (qpos, qvel, quat)):
        raise ValueError("init state contains a non-finite value")
    _finite_nonnegative("root_height_m", float(init_state["root_height_m"]))
    quat_norm = float(np.linalg.norm(quat))
    if abs(quat_norm - 1.0) > 1e-3:
        raise ValueError(f"root_quat_wxyz is not unit length: norm={quat_norm}")

    expected_names = control.get("joint_names")
    if control.get("joint_order") != "mujoco_g1_29dof_hardware_order":
        raise ValueError("control joint_order is not direct MuJoCo/G1 hardware order")
    if control.get("action_order") != "mujoco_g1_29dof_hardware_order":
        raise ValueError("control action_order is not direct MuJoCo/G1 hardware order")
    if not isinstance(expected_names, list) or len(expected_names) != 29:
        raise ValueError("control joint_names must contain exactly 29 entries")
    for field in (
        "default_joint_pos",
        "action_scale",
        "joint_stiffness",
        "joint_damping",
    ):
        values = np.asarray(control.get(field), dtype=np.float64)
        if values.shape != (29,) or not np.all(np.isfinite(values)):
            raise ValueError(f"control {field} must contain 29 finite values")
    return init_state, control


def _initialize_and_validate_model(
    sim: Any, init_state: dict[str, np.ndarray | float], control: dict[str, Any]
) -> dict[str, Any]:
    env = sim.sim_env
    model = env.mj_model
    data = env.mj_data
    body_joint_ids = np.asarray(env.body_joint_index, dtype=np.int32)
    expected_names = list(control["joint_names"])
    actual_names = [model.joint(int(joint_id)).name for joint_id in body_joint_ids]
    if actual_names != expected_names:
        raise ValueError(
            "MuJoCo body-joint order does not match control JSON:\n"
            f"expected={expected_names}\nactual={actual_names}"
        )

    qpos_addresses = np.asarray(model.jnt_qposadr[body_joint_ids], dtype=np.int32)
    qvel_addresses = np.asarray(model.jnt_dofadr[body_joint_ids], dtype=np.int32)
    bridge_qpos_addresses = body_joint_ids + env.qpos_offset - 1
    bridge_qvel_addresses = body_joint_ids + env.qvel_offset - 1
    if not np.array_equal(qpos_addresses, bridge_qpos_addresses):
        raise ValueError("MuJoCo qpos addressing differs from the existing DDS bridge mapping")
    if not np.array_equal(qvel_addresses, bridge_qvel_addresses):
        raise ValueError("MuJoCo qvel addressing differs from the existing DDS bridge mapping")

    body_actuator_ids = body_joint_ids - 1
    actuator_joint_ids = np.asarray(model.actuator_trnid[body_actuator_ids, 0], dtype=np.int32)
    if not np.array_equal(actuator_joint_ids, body_joint_ids):
        raise ValueError("body actuator order does not map one-to-one to the 29 body joints")

    mujoco.mj_resetData(model, data)
    data.qpos[:3] = np.array([0.0, 0.0, float(init_state["root_height_m"])])
    data.qpos[3:7] = init_state["root_quat_wxyz"]
    data.qpos[qpos_addresses] = init_state["body_qpos_mujoco"]
    data.qvel[:] = 0.0
    data.qvel[qvel_addresses] = init_state["body_qvel_mujoco"]
    data.qacc[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    if not getattr(env, "elastic_band", None):
        raise ValueError("validation requires ENABLE_ELASTIC_BAND during the init hold")
    env.elastic_band.enable = True
    env.elastic_band.length = 0.0
    env.elastic_band.point = data.xpos[env.band_attached_link].copy()

    return {
        "joint_names": actual_names,
        "qpos_addresses": qpos_addresses.tolist(),
        "qvel_addresses": qvel_addresses.tolist(),
        "body_actuator_ids": body_actuator_ids.tolist(),
        "direct_hardware_order": True,
        "pd_equation": "tau_ff + kp*(q_des-q) + kd*(dq_des-dq)",
    }


def _low_command_snapshot(bridge: Any, count: int) -> dict[str, np.ndarray] | None:
    with bridge.low_cmd_lock:
        if not bridge.low_cmd_received:
            return None
        motor_cmd = bridge.low_cmd.motor_cmd
        return {
            "q": np.array([motor_cmd[i].q for i in range(count)], dtype=np.float64),
            "dq": np.array([motor_cmd[i].dq for i in range(count)], dtype=np.float64),
            "kp": np.array([motor_cmd[i].kp for i in range(count)], dtype=np.float64),
            "kd": np.array([motor_cmd[i].kd for i in range(count)], dtype=np.float64),
            "tau_ff": np.array([motor_cmd[i].tau for i in range(count)], dtype=np.float64),
        }


def _root_tilt_deg(quat_wxyz: np.ndarray) -> float:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat /= np.linalg.norm(quat)
    _, qx, qy, _ = quat
    local_up_world_z = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.acos(float(np.clip(local_up_world_z, -1.0, 1.0))))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    for name in (
        "command_timeout_s",
        "policy_start_timeout_s",
        "policy_start_command_step_rad",
        "release_after_command_s",
        "duration_after_release_s",
        "min_root_height_m",
        "max_root_tilt_deg",
        "max_body_joint_speed_rad_s",
        "max_command_step_rad",
        "max_torque_saturation_fraction",
    ):
        _finite_nonnegative(name, float(getattr(args, name)))
    if args.duration_after_release_s == 0.0:
        raise ValueError("duration_after_release_s must be greater than zero")
    if args.max_torque_saturation_fraction > 1.0:
        raise ValueError("max_torque_saturation_fraction must be in [0, 1]")

    init_state, control = _load_inputs(args.init_state, args.tracking_control)
    config = SimLoopConfig(
        interface=args.interface,
        enable_onscreen=args.onscreen,
        enable_offscreen=False,
        enable_image_publish=False,
        verbose=False,
    )
    wbc_config = config.load_wbc_yaml()
    wbc_config["ENV_NAME"] = config.env_name
    sim = SimulatorFactory.create_simulator(
        config=wbc_config,
        env_name=config.env_name,
        onscreen=args.onscreen,
        offscreen=False,
        enable_image_publish=False,
    )

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAILED",
        "failure_reasons": [],
        "inputs": {
            "init_state": str(args.init_state.resolve()),
            "tracking_control": str(args.tracking_control.resolve()),
        },
        "thresholds": {
            "min_root_height_m": args.min_root_height_m,
            "max_root_tilt_deg": args.max_root_tilt_deg,
            "max_body_joint_speed_rad_s": args.max_body_joint_speed_rad_s,
            "max_command_step_rad": args.max_command_step_rad,
            "max_torque_saturation_fraction": args.max_torque_saturation_fraction,
            "policy_start_command_step_rad": args.policy_start_command_step_rad,
        },
    }

    try:
        mapping = _initialize_and_validate_model(sim, init_state, control)
        report["mapping"] = mapping
        env = sim.sim_env
        model = env.mj_model
        data = env.mj_data
        bridge = sim.unitree_bridge
        body_joint_ids = np.asarray(env.body_joint_index, dtype=np.int32)
        qpos_addresses = np.asarray(model.jnt_qposadr[body_joint_ids], dtype=np.int32)
        qvel_addresses = np.asarray(model.jnt_dofadr[body_joint_ids], dtype=np.int32)
        body_actuator_ids = body_joint_ids - 1
        torque_limits = np.asarray(env.torque_limit[body_actuator_ids], dtype=np.float64)
        expected_kp = np.asarray(control["joint_stiffness"], dtype=np.float64)
        expected_kd = np.asarray(control["joint_damping"], dtype=np.float64)

        first_command_sim_time: float | None = None
        policy_start_sim_time: float | None = None
        release_sim_time: float | None = None
        previous_q_command: np.ndarray | None = None
        max_command_step = 0.0
        max_command_step_joint = -1
        max_target_error = 0.0
        max_target_error_joint = -1
        max_kp_error = 0.0
        max_kd_error = 0.0
        max_dq_command = 0.0
        max_tau_ff_command = 0.0
        samples = 0
        saturation_samples = 0
        fall_count = 0
        finite = True
        min_root_height = math.inf
        max_root_tilt = 0.0
        max_body_speed = 0.0
        max_root_linear_speed = 0.0
        max_root_angular_speed = 0.0
        max_commanded_torque = 0.0
        max_actuator_force = 0.0
        max_torque_limit_ratio = 0.0
        command_deadline = time.monotonic() + args.command_timeout_s

        print("[MuJoCoValidation] Waiting for the C++ runtime on Unitree DDS rt/lowcmd ...")
        while True:
            step_wall_start = time.monotonic()
            pre_step_height = float(data.qpos[2])
            env.sim_step()
            q_command = _low_command_snapshot(bridge, len(body_joint_ids))

            if q_command is not None:
                if not all(np.all(np.isfinite(value)) for value in q_command.values()):
                    finite = False
                    report["failure_reasons"].append("LowCmd contains a non-finite value")
                    break
                if first_command_sim_time is None:
                    first_command_sim_time = float(data.time)
                    print(
                        "[MuJoCoValidation] First LowCmd received; holding package init "
                        f"for {args.release_after_command_s:.2f} s before support release."
                    )
                if previous_q_command is not None:
                    command_steps = np.abs(q_command["q"] - previous_q_command)
                    joint = int(np.argmax(command_steps))
                    if (
                        policy_start_sim_time is None
                        and float(command_steps[joint])
                        >= args.policy_start_command_step_rad
                    ):
                        policy_start_sim_time = float(data.time)
                        print(
                            "[MuJoCoValidation] C++ policy command detected at joint "
                            f"{joint}: step={float(command_steps[joint]):.4f} rad."
                        )
                    if float(command_steps[joint]) > max_command_step:
                        max_command_step = float(command_steps[joint])
                        max_command_step_joint = joint
                previous_q_command = q_command["q"].copy()
                max_kp_error = max(
                    max_kp_error, float(np.max(np.abs(q_command["kp"] - expected_kp)))
                )
                max_kd_error = max(
                    max_kd_error, float(np.max(np.abs(q_command["kd"] - expected_kd)))
                )
                max_dq_command = max(max_dq_command, float(np.max(np.abs(q_command["dq"]))))
                max_tau_ff_command = max(
                    max_tau_ff_command, float(np.max(np.abs(q_command["tau_ff"])))
                )

                target_errors = np.abs(q_command["q"] - data.qpos[qpos_addresses])
                joint = int(np.argmax(target_errors))
                if float(target_errors[joint]) > max_target_error:
                    max_target_error = float(target_errors[joint])
                    max_target_error_joint = joint

            if (
                first_command_sim_time is not None
                and policy_start_sim_time is not None
                and release_sim_time is None
                and float(data.time) - first_command_sim_time >= args.release_after_command_s
            ):
                env.elastic_band.enable = False
                release_sim_time = float(data.time)
                print("[MuJoCoValidation] Elastic support released; collecting free-base metrics.")

            if release_sim_time is not None:
                samples += 1
                state_arrays = (data.qpos, data.qvel, data.qacc, data.ctrl, data.actuator_force)
                if not all(np.all(np.isfinite(value)) for value in state_arrays):
                    finite = False
                    report["failure_reasons"].append("MuJoCo state became non-finite")
                    break

                min_root_height = min(min_root_height, pre_step_height, float(data.qpos[2]))
                max_root_tilt = max(max_root_tilt, _root_tilt_deg(data.qpos[3:7]))
                max_body_speed = max(
                    max_body_speed, float(np.max(np.abs(data.qvel[qvel_addresses])))
                )
                max_root_linear_speed = max(
                    max_root_linear_speed, float(np.linalg.norm(data.qvel[:3]))
                )
                max_root_angular_speed = max(
                    max_root_angular_speed, float(np.linalg.norm(data.qvel[3:6]))
                )
                body_torques = np.asarray(env.torques[body_actuator_ids], dtype=np.float64)
                actuator_forces = np.asarray(
                    data.actuator_force[body_actuator_ids], dtype=np.float64
                )
                torque_ratios = np.abs(body_torques) / torque_limits
                max_commanded_torque = max(
                    max_commanded_torque, float(np.max(np.abs(body_torques)))
                )
                max_actuator_force = max(
                    max_actuator_force, float(np.max(np.abs(actuator_forces)))
                )
                max_torque_limit_ratio = max(
                    max_torque_limit_ratio, float(np.max(torque_ratios))
                )
                if np.any(torque_ratios >= 0.99):
                    saturation_samples += 1

                if env.fall:
                    fall_count += 1
                    report["failure_reasons"].append(
                        "MuJoCo fall detector triggered (root height below 0.2 m)"
                    )
                    break
                if float(data.time) - release_sim_time >= args.duration_after_release_s:
                    break
            elif first_command_sim_time is None and time.monotonic() >= command_deadline:
                report["failure_reasons"].append(
                    f"no LowCmd received within {args.command_timeout_s:.1f} s"
                )
                break
            elif (
                first_command_sim_time is not None
                and policy_start_sim_time is None
                and float(data.time) - first_command_sim_time >= args.policy_start_timeout_s
            ):
                report["failure_reasons"].append(
                    "C++ policy start was not detected within "
                    f"{args.policy_start_timeout_s:.1f} s after the first LowCmd"
                )
                break

            elapsed = time.monotonic() - step_wall_start
            if elapsed < env.sim_dt:
                time.sleep(env.sim_dt - elapsed)
    except KeyboardInterrupt:
        report["failure_reasons"].append("validation interrupted by user")
    finally:
        sim.close()

    if "mapping" not in report:
        return 2, report

    saturation_fraction = saturation_samples / samples if samples else 0.0
    joint_names = report["mapping"]["joint_names"]
    report["timing"] = {
        "first_command_sim_time_s": first_command_sim_time,
        "policy_start_sim_time_s": policy_start_sim_time,
        "support_release_sim_time_s": release_sim_time,
        "release_after_command_s": args.release_after_command_s,
        "requested_duration_after_release_s": args.duration_after_release_s,
        "observed_duration_after_release_s": samples * float(wbc_config["SIMULATE_DT"]),
        "released_samples": samples,
        "sim_dt_s": float(wbc_config["SIMULATE_DT"]),
    }
    report["metrics"] = {
        "finite": finite,
        "fall_count": fall_count,
        "min_root_height_m": None if samples == 0 else min_root_height,
        "max_root_tilt_deg": max_root_tilt,
        "max_body_joint_speed_rad_s": max_body_speed,
        "max_root_linear_speed_m_s": max_root_linear_speed,
        "max_root_angular_speed_rad_s": max_root_angular_speed,
        "max_commanded_body_torque_nm": max_commanded_torque,
        "max_body_actuator_force_nm": max_actuator_force,
        "max_torque_limit_ratio": max_torque_limit_ratio,
        "torque_saturation_sample_fraction": saturation_fraction,
        "max_q_command_step_rad": max_command_step,
        "max_q_command_step_joint_index": max_command_step_joint,
        "max_q_command_step_joint_name": (
            joint_names[max_command_step_joint] if max_command_step_joint >= 0 else None
        ),
        "max_q_target_error_rad": max_target_error,
        "max_q_target_error_joint_index": max_target_error_joint,
        "max_q_target_error_joint_name": (
            joint_names[max_target_error_joint] if max_target_error_joint >= 0 else None
        ),
        "max_kp_error_vs_control_json": max_kp_error,
        "max_kd_error_vs_control_json": max_kd_error,
        "max_abs_dq_command_rad_s": max_dq_command,
        "max_abs_tau_ff_command_nm": max_tau_ff_command,
    }

    if release_sim_time is None:
        report["failure_reasons"].append("elastic support was never released")
    if samples == 0:
        report["failure_reasons"].append("no free-base samples were collected")
    elif min_root_height < args.min_root_height_m:
        report["failure_reasons"].append(
            f"minimum root height {min_root_height:.4f} m is below "
            f"{args.min_root_height_m:.4f} m"
        )
    if max_root_tilt > args.max_root_tilt_deg:
        report["failure_reasons"].append(
            f"maximum root tilt {max_root_tilt:.2f} deg exceeds "
            f"{args.max_root_tilt_deg:.2f} deg"
        )
    if max_body_speed > args.max_body_joint_speed_rad_s:
        report["failure_reasons"].append(
            f"maximum body joint speed {max_body_speed:.3f} rad/s exceeds "
            f"{args.max_body_joint_speed_rad_s:.3f} rad/s"
        )
    if max_command_step > args.max_command_step_rad:
        report["failure_reasons"].append(
            f"maximum q command step {max_command_step:.4f} rad exceeds "
            f"{args.max_command_step_rad:.4f} rad"
        )
    command_contract_tolerance = 1e-4
    if max_kp_error > command_contract_tolerance:
        report["failure_reasons"].append(
            f"LowCmd kp differs from control JSON by up to {max_kp_error:.6g}"
        )
    if max_kd_error > command_contract_tolerance:
        report["failure_reasons"].append(
            f"LowCmd kd differs from control JSON by up to {max_kd_error:.6g}"
        )
    if max_dq_command > command_contract_tolerance:
        report["failure_reasons"].append(
            f"LowCmd dq target is nonzero: max {max_dq_command:.6g} rad/s"
        )
    if max_tau_ff_command > command_contract_tolerance:
        report["failure_reasons"].append(
            f"LowCmd feed-forward torque is nonzero: max {max_tau_ff_command:.6g} Nm"
        )
    if saturation_fraction > args.max_torque_saturation_fraction:
        report["failure_reasons"].append(
            f"torque saturation sample fraction {saturation_fraction:.4f} exceeds "
            f"{args.max_torque_saturation_fraction:.4f}"
        )

    report["failure_reasons"] = list(dict.fromkeys(report["failure_reasons"]))
    report["status"] = "PASSED" if not report["failure_reasons"] else "FAILED"
    return (0 if report["status"] == "PASSED" else 2), report


def main() -> int:
    args = _parse_args()
    try:
        code, report = _run(args)
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "ERROR",
            "failure_reasons": [f"{type(error).__name__}: {error}"],
        }
        code = 1
    _write_report(args.output, report)
    print(f"[MuJoCoValidation] {report['status']}; report: {args.output.resolve()}")
    for reason in report.get("failure_reasons", []):
        print(f"[MuJoCoValidation] - {reason}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
