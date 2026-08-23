# External-reference tracking ep100 部署

该功能只在分支 `try/flip360_external_ref_policy` 上，通过显式 backend 开启。默认 SONIC 启动路径、encoder、observation registry、控制参数和 action reorder 均未替换。

## 数据流

```text
ep100 reference ONNX --启动时预加载 qpos/qvel/root quat--> RAM reference buffer
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

reference ONNX 的 `time_step` 仅由 reference provider 在启动预加载时用于读取外部 qpos/qvel/root quat；tracking policy 本身只有 `obs_dict` 一个输入。运行时 policy cursor 由部署端维护，不存在 embedded motion policy。

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

若要走与真机相同的 C++ observation/policy/control 路径连接 Unitree MuJoCo DDS：先启动现有 Unitree MuJoCo lowstate/lowcmd 仿真，再运行：

```bash
./scripts/run_external_ref_ep100.sh sim
```

sim 默认完整 941 policy tick、JSON 原始 PD；可通过 `EXTERNAL_REF_GAIN_SCALE` 和 `EXTERNAL_REF_MAX_TICKS` 覆盖。

## 真机保护入口

真机前必须先通过 dry-run 和 MuJoCo。准备好急停后：

```bash
EXTERNAL_REF_REAL_INTERFACE=enp5s0 \
./scripts/run_external_ref_ep100.sh real
```

脚本要求手工输入 `RUN_EP100_EXTERNAL_REF`。默认只运行 250 tick，PD 为 JSON 参数的 0.25 倍，首条 target 相对实测关节的最大误差限制为 0.5 rad，逐 tick target 最大变化限制为 0.5 rad；非有限值、关节速度超限或 target guard 触发会切 damping 并停止。真机模式拒绝把 gain scale 调到 0.25 以上。

这只是准备好的低风险入口；本次代码验证不包含实际真机启动。

## 回退

```bash
cd /home/chuye/GR00T-WholeBodyControl
git switch <原SONIC分支>
```

原仓库和 Dynamic-DP 仓库没有被该工作树修改。即使留在本分支，不传 `--policy-type external_ref_tracking` 也仍走原 SONIC backend。
