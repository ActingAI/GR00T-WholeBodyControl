"""
Dataset configuration for the Sonic VLA pipeline.

Provides feature/modality config dicts and a convenience function to
instantiate the G1 RobotModel needed for FK and joint configuration
assembly during data collection.

Joint names, counts, and group indices are derived at runtime from the
``RobotModel`` (via its ``supplemental_info``).
"""

from __future__ import annotations

from typing import Literal

from gear_sonic.data.robot_model import RobotModel

EGO_VIEW_HEIGHT: int = 480
EGO_VIEW_WIDTH: int = 640
WRIST_VIEW_HEIGHT: int = 480
WRIST_VIEW_WIDTH: int = 640
FPS: int = 50


_JOINT_GROUPS_FOR_STATE: list[str] = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]


def _get_joint_group_slices(robot_model: RobotModel) -> dict[str, dict[str, int]]:
    """Derive ``{group_name: {"start": ..., "end": ...}}`` from the robot model."""
    slices: dict[str, dict[str, int]] = {}
    for group in _JOINT_GROUPS_FOR_STATE:
        indices = sorted(robot_model.get_joint_group_indices(group))
        slices[group] = {"start": indices[0], "end": indices[-1] + 1}
    return slices


def get_modality_config_sonic_vla(robot_model: RobotModel) -> dict:
    """Return the modality config for the Sonic VLA dataset.

    Produces the exact content of meta/modality.json.
    """
    group_slices = _get_joint_group_slices(robot_model)

    return {
        "state": {
            **group_slices,
            "left_wrist_pos": {
                "start": 0,
                "end": 3,
                "original_key": "observation.eef_state",
            },
            "left_wrist_abs_quat": {
                "start": 3,
                "end": 7,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
            "right_wrist_pos": {
                "start": 7,
                "end": 10,
                "original_key": "observation.eef_state",
            },
            "right_wrist_abs_quat": {
                "start": 10,
                "end": 14,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
            "root_orientation": {
                "start": 0,
                "end": 4,
                "original_key": "observation.root_orientation",
                "rotation_type": "quaternion",
            },
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
            "cpp_rotation_offset": {
                "start": 0,
                "end": 4,
                "original_key": "observation.cpp_rotation_offset",
                "rotation_type": "quaternion",
            },
            "init_base_quat": {
                "start": 0,
                "end": 4,
                "original_key": "observation.init_base_quat",
                "rotation_type": "quaternion",
            },
            "body_qpos": {
                "start": 0,
                "end": 29,
                "original_key": "observation.body_qpos",
            },
            "body_qvel": {
                "start": 0,
                "end": 29,
                "original_key": "observation.body_qvel",
            },
            "base_angular_velocity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.base_angular_velocity",
            },
            "base_linear_acceleration": {
                "start": 0,
                "end": 3,
                "original_key": "observation.base_linear_acceleration",
            },
            "torso_orientation": {
                "start": 0,
                "end": 4,
                "original_key": "observation.torso_orientation",
                "rotation_type": "quaternion",
            },
            "torso_angular_velocity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.torso_angular_velocity",
            },
            "sonic_motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "observation.sonic_motion_token",
            },
            "sonic_status": {
                "start": 0,
                "end": 4,
                "original_key": "observation.sonic_status",
            },
            "robot_state_index": {
                "start": 0,
                "end": 2,
                "original_key": "observation.robot_state_index",
            },
            "sync_timestamps": {
                "start": 0,
                "end": 17,
                "original_key": "observation.sync_timestamps",
            },
            "sync_frame_changed": {
                "start": 0,
                "end": 6,
                "original_key": "observation.sync_frame_changed",
            },
            "sync_receive_age_s": {
                "start": 0,
                "end": 5,
                "original_key": "observation.sync_receive_age_s",
            },
        },
        "action": {
            "delta_heading": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.delta_heading",
            },
            "motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "action.motion_token",
            },
            "smpl_joints": {
                "start": 0,
                "end": 72,
                "original_key": "teleop.smpl_joints",
            },
            "smpl_pose": {
                "start": 0,
                "end": 63,
                "original_key": "teleop.smpl_pose",
            },
            "body_quat_w": {
                "start": 0,
                "end": 4,
                "original_key": "teleop.body_quat_w",
                "rotation_type": "quaternion",
            },
            "target_body_orientation": {
                "start": 0,
                "end": 6,
                "original_key": "teleop.target_body_orientation",
                "rotation_type": "rotation_6d",
            },
            "left_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": 7,
                "original_key": "teleop.right_hand_joints",
            },
            "left_wrist_joints": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.left_wrist_joints",
            },
            "right_wrist_joints": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.right_wrist_joints",
            },
            "stream_mode": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.stream_mode",
            },
            "planner_mode": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_mode",
            },
            "planner_movement": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.planner_movement",
            },
            "planner_facing": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.planner_facing",
            },
            "planner_speed": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_speed",
            },
            "planner_height": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_height",
            },
            "vr_3pt_position": {
                "start": 0,
                "end": 9,
                "original_key": "teleop.vr_3pt_position",
            },
            "vr_3pt_orientation": {
                "start": 0,
                "end": 18,
                "original_key": "teleop.vr_3pt_orientation",
                "rotation_type": "rotation_6d",
            },
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def get_features_sonic_vla(robot_model: RobotModel) -> dict:
    """Return the dataset features for the Sonic VLA dataset.

    The returned dict populates the "features" key of meta/info.json.
    """
    joint_names = robot_model.joint_names
    num_joints = robot_model.num_joints

    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float64",
            "shape": (num_joints,),
            "names": joint_names,
        },
        "observation.eef_state": {
            "dtype": "float64",
            "shape": (14,),
            "names": [
                "left_wrist_pos",
                "left_wrist_abs_quat",
                "right_wrist_pos",
                "right_wrist_abs_quat",
            ],
        },
        "action.wbc": {
            "dtype": "float64",
            "shape": (num_joints,),
            "names": joint_names,
        },
        "observation.root_orientation": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
        },
        "observation.projected_gravity": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "observation.cpp_rotation_offset": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["rot_offset_qw", "rot_offset_qx", "rot_offset_qy", "rot_offset_qz"],
        },
        "observation.init_base_quat": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["init_base_qw", "init_base_qx", "init_base_qy", "init_base_qz"],
        },
        "observation.body_qpos": {
            "dtype": "float64",
            "shape": (29,),
            "names": "body_qpos",
        },
        "observation.body_qvel": {
            "dtype": "float64",
            "shape": (29,),
            "names": "body_qvel",
        },
        "observation.base_angular_velocity": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["base_wx", "base_wy", "base_wz"],
        },
        "observation.base_linear_acceleration": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["base_ax", "base_ay", "base_az"],
        },
        "observation.torso_orientation": {
            "dtype": "float64",
            "shape": (4,),
            "names": ["torso_qw", "torso_qx", "torso_qy", "torso_qz"],
        },
        "observation.torso_angular_velocity": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["torso_wx", "torso_wy", "torso_wz"],
        },
        "observation.sonic_motion_token": {
            "dtype": "float64",
            "shape": (64,),
            "names": "sonic_motion_token",
        },
        "observation.sonic_status": {
            "dtype": "int32",
            "shape": (4,),
            "names": [
                "stream_mode",
                "encoder_mode",
                "motion_playing",
                "robot_state_skipped",
            ],
        },
        "observation.robot_state_index": {
            "dtype": "int64",
            "shape": (2,),
            "names": ["state_index", "delta_index"],
        },
        "observation.sync_timestamps": {
            "dtype": "float64",
            "shape": (17,),
            "names": [
                "exporter_realtime_s",
                "exporter_monotonic_s",
                "robot_ros_s",
                "robot_receive_s",
                "motion_token_s",
                "ego_view_camera_s",
                "ego_view_depth_camera_s",
                "left_wrist_camera_s",
                "right_wrist_camera_s",
                "primary_camera_receive_s",
                "wrist_camera_receive_s",
                "sonic_pose_realtime_s",
                "sonic_pose_monotonic_s",
                "sonic_pose_receive_s",
                "sonic_planner_receive_s",
                "genrobot_state_publish_s",
                "genrobot_encoder_receive_s",
            ],
        },
        "observation.sync_frame_changed": {
            "dtype": "int32",
            "shape": (6,),
            "names": [
                "robot_state_changed",
                "motion_token_changed",
                "ego_view_changed",
                "ego_view_depth_changed",
                "left_wrist_changed",
                "right_wrist_changed",
            ],
        },
        "observation.sync_receive_age_s": {
            "dtype": "float64",
            "shape": (5,),
            "names": [
                "robot_receive_age_s",
                "primary_camera_receive_age_s",
                "wrist_camera_receive_age_s",
                "sonic_pose_receive_age_s",
                "genrobot_receive_age_s",
            ],
        },
        "teleop.delta_heading": {
            "dtype": "float64",
            "shape": (1,),
            "names": ["delta_heading"],
        },
        "action.motion_token": {
            "dtype": "float64",
            "shape": (64,),
            "names": "motion_token",
        },
        "teleop.smpl_joints": {
            "dtype": "float32",
            "shape": (72,),
            "names": "smpl_joints",
        },
        "teleop.smpl_pose": {
            "dtype": "float32",
            "shape": (63,),
            "names": "smpl_pose",
        },
        "teleop.body_quat_w": {
            "dtype": "float32",
            "shape": (4,),
            "names": "body_quat_w",
        },
        "teleop.target_body_orientation": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "target_body_r00",
                "target_body_r10",
                "target_body_r01",
                "target_body_r11",
                "target_body_r02",
                "target_body_r12",
            ],
        },
        "teleop.left_hand_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": "left_hand_joints",
        },
        "teleop.right_hand_joints": {
            "dtype": "float32",
            "shape": (7,),
            "names": "right_hand_joints",
        },
        "teleop.smpl_frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["smpl_frame_index"],
        },
        "teleop.left_wrist_joints": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
        },
        "teleop.right_wrist_joints": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
        },
        "teleop.stream_mode": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["stream_mode"],
        },
        "teleop.planner_mode": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["locomotion_mode"],
        },
        "teleop.planner_movement": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["movement_x", "movement_y", "movement_z"],
        },
        "teleop.planner_facing": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["facing_x", "facing_y", "facing_z"],
        },
        "teleop.planner_speed": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["speed"],
        },
        "teleop.planner_height": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["height"],
        },
        "teleop.vr_3pt_position": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "lwrist_x", "lwrist_y", "lwrist_z",
                "rwrist_x", "rwrist_y", "rwrist_z",
                "neck_x", "neck_y", "neck_z",
            ],
        },
        "teleop.vr_3pt_orientation": {
            "dtype": "float32",
            "shape": (18,),
            "names": [
                "lwrist_r00", "lwrist_r10", "lwrist_r01", "lwrist_r11", "lwrist_r02", "lwrist_r12",
                "rwrist_r00", "rwrist_r10", "rwrist_r01", "rwrist_r11", "rwrist_r02", "rwrist_r12",
                "neck_r00", "neck_r10", "neck_r01", "neck_r11", "neck_r02", "neck_r12",
            ],
        },
    }


def get_wrist_camera_features() -> dict:
    """Features for optional wrist cameras (added when ``record_wrist_cameras`` is enabled)."""
    return {
        "observation.images.left_wrist": {
            "dtype": "video",
            "shape": [WRIST_VIEW_HEIGHT, WRIST_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.images.right_wrist": {
            "dtype": "video",
            "shape": [WRIST_VIEW_HEIGHT, WRIST_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
    }


def get_wrist_camera_modality_config() -> dict:
    """Modality config entries for optional wrist cameras."""
    return {
        "video": {
            "left_wrist": {"original_key": "observation.images.left_wrist"},
            "right_wrist": {"original_key": "observation.images.right_wrist"},
        },
    }


def get_genrobot_gripper_features() -> dict:
    """Features for the GENROBOT DAS gripper opening distance."""
    return {
        "observation.genrobot_gripper_width": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_gripper_width_m", "right_gripper_width_m"],
        },
        "action.genrobot_gripper_target": {
            "dtype": "float32",
            "shape": (2,),
            "names": ["left_gripper_target_m", "right_gripper_target_m"],
        },
        "observation.genrobot_gripper_timestamps": {
            "dtype": "float64",
            "shape": (5,),
            "names": [
                "state_publish_time_s",
                "left_encoder_ros_time_s",
                "right_encoder_ros_time_s",
                "encoder_callback_time_s",
                "exporter_receive_time_s",
            ],
        },
    }


def get_genrobot_gripper_modality_config() -> dict:
    """Modality config entries for GENROBOT actual and commanded opening."""
    return {
        "state": {
            "genrobot_gripper_width": {
                "start": 0,
                "end": 2,
                "original_key": "observation.genrobot_gripper_width",
            },
            "genrobot_gripper_timestamps": {
                "start": 0,
                "end": 5,
                "original_key": "observation.genrobot_gripper_timestamps",
            },
        },
        "action": {
            "genrobot_gripper_target": {
                "start": 0,
                "end": 2,
                "original_key": "action.genrobot_gripper_target",
            },
        },
    }


def get_g1_robot_model(
    waist_location: Literal[
        "lower_body", "upper_body", "lower_and_upper_body"
    ] = "lower_and_upper_body",
    high_elbow_pose: bool = False,
):
    """Instantiate the G1 + ThreeFinger RobotModel for Sonic VLA."""
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

    return instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=high_elbow_pose,
    )
