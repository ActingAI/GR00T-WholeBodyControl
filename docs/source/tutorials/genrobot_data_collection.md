# GENROBOT Gripper Data Collection

This fork adds a GENROBOT DAS gripper path on top of the SONIC G1 teleoperation
and data collection stack. The goal is to keep the normal SONIC workflow
unchanged while replacing the Unitree Dex3-1 hand behavior with GENROBOT
open/close commands and recording the gripper wrist cameras.

## What This Adds

- PICO trigger control for the left and right GENROBOT grippers.
- A G1-side UDP-to-ROS bridge that publishes GENROBOT target distances.
- A G1-side ROS-image-to-ZMQ bridge for the two central wrist cameras.
- Data exporter support for three camera streams:
  - `observation.images.ego_view`
  - `observation.images.left_wrist`
  - `observation.images.right_wrist`
- Data exporter support for GENROBOT gripper state:
  - `observation.genrobot_gripper_width`
  - `action.genrobot_gripper_target`

The added code is intentionally optional. If you do not pass the GENROBOT flags,
the upstream SONIC workflow behaves as before.

## Network and Ports

The tested G1 onboard computer address is `192.168.123.164`.

| Port | Direction | Producer | Consumer | Payload |
| --- | --- | --- | --- | --- |
| `5555` | G1 -> workstation | existing camera server | data exporter/viewer | ego camera JPEG frames |
| `5556` | workstation local | PICO manager | C++ deploy / data exporter | SMPL pose, planner, manager state |
| `5557` | workstation local | C++ deploy | data exporter | robot debug state |
| `5559` | G1 -> workstation | `genrobot_wrist_camera_zmq_bridge.py` | data exporter | left/right wrist JPEG frames |
| `5568` | workstation -> G1 | PICO manager | `genrobot_gripper_ros_bridge.py` | target gripper distances |
| `5569` | G1 -> workstation | `genrobot_gripper_ros_bridge.py` | data exporter | actual/target gripper state |

## G1-Side Processes

The GENROBOT SDK runs on the G1 onboard computer. The expected ROS topics are:

```bash
/left_gripper/target_distance
/right_gripper/target_distance
/left_gripper/encoder
/right_gripper/encoder
/left_gripper/camera/color/image_raw
/right_gripper/camera/color/image_raw
```

The two bridge scripts in this repository are:

```bash
gear_sonic/scripts/genrobot_gripper_ros_bridge.py
gear_sonic/scripts/genrobot_wrist_camera_zmq_bridge.py
```

For manual startup on the G1:

```bash
cd ~/gen_controller_sdk_release
source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch robot_driver dual_gripper_start.launch \
  show_preview:=false \
  camera_resolutions:=640x480 \
  camera_count:=1
```

Then, in separate shells on the G1:

```bash
source /opt/ros/noetic/setup.bash
source ~/gen_controller_sdk_release/devel/setup.bash
python ~/GR00T-WholeBodyControl/gear_sonic/scripts/genrobot_gripper_ros_bridge.py
```

```bash
source /opt/ros/noetic/setup.bash
source ~/gen_controller_sdk_release/devel/setup.bash
python ~/GR00T-WholeBodyControl/gear_sonic/scripts/genrobot_wrist_camera_zmq_bridge.py
```

In the lab setup these are normally installed as systemd services:

```bash
genrobot-roscore.service
genrobot-gripper-driver.service
genrobot-gripper-bridge.service
genrobot-wrist-camera-bridge.service
```

Check them after a G1 reboot:

```bash
for s in \
  genrobot-roscore.service \
  genrobot-gripper-driver.service \
  genrobot-gripper-bridge.service \
  genrobot-wrist-camera-bridge.service
do
  systemctl is-active "$s"
done

ss -ltnup | egrep '11311|5555|5559|5568|5569'
```

Expected camera and encoder rates are about 30 Hz when only each gripper's
central wrist camera is active:

```bash
source /opt/ros/noetic/setup.bash
timeout 8 rostopic hz /left_gripper/camera/color/image_raw
timeout 8 rostopic hz /right_gripper/camera/color/image_raw
timeout 5 rostopic hz /left_gripper/encoder
timeout 5 rostopic hz /right_gripper/encoder
```

## Workstation Data Collection Command

Use the tmux launcher from the repository root:

```bash
cd ~/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_data_collection.py \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --record-wrist-cameras \
  --wrist-camera-host 192.168.123.164 \
  --wrist-camera-port 5559 \
  --record-genrobot-gripper \
  --genrobot-gripper-state-host 192.168.123.164 \
  --genrobot-gripper-state-port 5569 \
  --pico-genrobot-gripper-host 192.168.123.164 \
  --pico-genrobot-gripper-port 5568 \
  --task-prompt "blanket_yellow_test"
```

This launches:

- C++ SONIC deploy.
- PICO manager.
- Data exporter.
- Camera viewer for the primary camera stream.

The built-in camera viewer connects to `--camera-host/--camera-port`, so it
shows the primary ego camera server. The wrist cameras are still recorded by
the exporter through `--wrist-camera-host/--wrist-camera-port`.

## Gripper Controls

By default, GENROBOT gripper control uses the PICO triggers:

- left trigger controls the left gripper
- right trigger controls the right gripper
- trigger released means open, about `0.103 m`
- trigger fully pressed means closed, `0.0 m`

The side-grip buttons are left for data collection controls. This avoids
conflicts with the normal recording gestures:

- left grip + `A`: start or save an episode
- left grip + `B`: discard the current episode while it is still recording

If you deliberately want side grip to also close the gripper, add:

```bash
--pico-genrobot-gripper-use-grip
```

That changes the gripper command to `max(trigger, grip)`.

## Dataset Fields

When `--record-wrist-cameras` is enabled, the dataset contains:

```text
observation.images.ego_view
observation.images.left_wrist
observation.images.right_wrist
```

When `--record-genrobot-gripper` is enabled, the dataset also contains:

```text
observation.genrobot_gripper_width  # shape [2], left/right actual opening in meters
action.genrobot_gripper_target      # shape [2], left/right commanded opening in meters
```

The gripper features are registered in:

```bash
gear_sonic/data/features_sonic_vla.py
```

The exporter records them in:

```bash
gear_sonic/scripts/run_data_exporter.py
```

## Quick Troubleshooting

If the gripper does not move, check the G1 first:

```bash
ssh unitree@192.168.123.164
source /opt/ros/noetic/setup.bash
rostopic info /left_gripper/target_distance
timeout 3 rostopic echo /left_gripper/encoder
ss -lunp | grep 5568
```

If wrist cameras are missing from saved data:

```bash
ssh unitree@192.168.123.164
ss -ltnp | grep 5559
timeout 8 rostopic hz /left_gripper/camera/color/image_raw
timeout 8 rostopic hz /right_gripper/camera/color/image_raw
```

If the data exporter keeps waiting, its status line names the missing part:

```text
missing images ['left_wrist', 'right_wrist']
genrobot False
```

That usually means the wrist bridge or gripper state bridge is not running on
the G1, or the workstation is pointed at the wrong host/port.
