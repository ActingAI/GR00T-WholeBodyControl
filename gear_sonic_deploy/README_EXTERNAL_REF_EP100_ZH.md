# External-reference tracking ep100 部署

该功能只在分支 `try/flip360_external_ref_policy` 上，通过显式 backend 开启。默认 SONIC 启动路径、encoder、observation registry、控制参数和 action reorder 均未替换。

## 数据流

```text
Terminal 2 gold-sync collector --K/I/P + qpos/qvel/root--> ZMQ 5556
                                   |
                                   v
SONIC motion/planner/cursor transport --46+ frame streamed reference
机器人 LowState --硬件顺序--> 10 帧 robot history
reference cursor --[0,5,...,45]--> 10 帧 future reference + root anchor
                                   |
                                   v
                         obs_dict float32[1,1570]
                                   |
                 flip360_pure_tracking_decoder.onnx
                                   |
                         raw_action float32[1,29]
                                   |
      q_target = JSON default_joint_pos + raw_action * JSON action_scale
```

真机 replay 不从 reference ONNX 内部播放 motion。Terminal 2 使用既有 ep100 golden NPZ，通过 protocol-v1 连续窗口送到 Terminal 1；C++ 沿用原 SONIC planner、stream merger、cursor 和 5557 回传。tracking policy 本身只有 `obs_dict` 一个输入，不存在 embedded motion policy。reference ONNX 只用于无 DDS 的 dry-run/MuJoCo 对照验证。

## 强制 dry-run

```bash
cd /home/chuye/GR00T-WholeBodyControl-external-ref/gear_sonic_deploy
./scripts/run_external_ref_ep100.sh dry-run
```

它不会初始化 DDS，也不会发机器人命令。检查内容包括：

- policy 输入输出严格为 `obs_dict[1,1570] -> action[1,29]`
- policy/reference/control 的 29 个 joint names 与顺序逐项一致
- future frame offsets、history 和 anchor layout
- 前几帧 obs/action/q_target 范围
- `SONIC_OUTPUT_REORDER_APPLIED=false`

默认 CSV 位于 `logs/external_ref_ep100/dry_run.csv`。

## MuJoCo

指定包已包含验证视频：

```text
/home/chuye/g1_general_tracking_external_ref_package_20260822/
verification/ep100_standard_941_video/mujoco_replay.mp4
```

视频为 640x480、50 fps、941 帧、18.82 秒。包内 summary 的基准指标为：joint mean absolute error `0.0589140`、p95 `0.1587665`、root XY final error `0.2376618`、最低 root z `0.6373838`。

若要走与真机相同的两终端 C++ observation/policy/control 路径连接 Unitree MuJoCo DDS：先启动现有 Unitree MuJoCo lowstate/lowcmd 仿真，然后按下面真机相同方式启动 Terminal 1 和 Terminal 2（Terminal 1 使用 `sim`）。

```bash
./scripts/run_external_ref_ep100.sh sim  # Terminal 1
```

Terminal 2 仍执行 K→I→P。JSON 原始 PD，reference cursor 在最终帧保持，与旧 SONIC replay 管线一致。

## 真机完整 replay 入口

真机前必须先通过 dry-run 和 MuJoCo。准备好急停后开启两个终端。

Terminal 1：

```bash
EXTERNAL_REF_REAL_INTERFACE=enp5s0 \
./scripts/run_external_ref_ep100.sh real
```

Terminal 2：

```bash
cd /home/chuye/Dynamic-DP-publish-external-ref
./scripts/run_g1_pick147_external_ref_policy_ep100.sh
```

Terminal 2 操作保持原流程：`K` 启动/进入 planner，移动机器人完成摆位；`I` 执行 common initial state 到 ep100 frame 0 的平滑初始化；`P` 开始正式 replay 和同步采集。

真机使用 JSON 中原始 kp/kd（gain scale 1.0），首条 target 和逐 tick target guard 均关闭。reference cursor 到第 940 帧后保持，由 Terminal 2 完成 post-roll/日志打包；非有限值、LowState/IMU 丢失、关节速度异常和急停仍会停止控制。

本次代码验证不包含实际真机启动。

## 回退

```bash
cd /home/chuye/GR00T-WholeBodyControl
git switch <原SONIC分支>
```

原 SONIC 和 Dynamic-DP 工作树没有被修改。两个 repo 均使用独立的 `try/flip360_external_ref_policy` worktree/branch；不传 `--policy-type external_ref_tracking` 仍走原 SONIC backend。
