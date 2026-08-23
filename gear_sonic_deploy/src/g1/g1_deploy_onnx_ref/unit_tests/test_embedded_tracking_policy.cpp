#include <gtest/gtest.h>

#include "embedded_tracking_policy.hpp"

namespace {

EmbeddedTrackingControlConfig MakeControl() {
  EmbeddedTrackingControlConfig control;
  control.default_joint_pos.fill(0.25);
  control.action_scale.fill(0.5);
  return control;
}

TEST(EmbeddedTrackingPolicy, BuildsExactHistoryLayoutAndAnchor) {
  const auto control = MakeControl();
  std::vector<EmbeddedTrackingObservationState> history(
      EmbeddedTrackingControlConfig::kHistoryLength);
  for (size_t frame = 0; frame < history.size(); ++frame) {
    history[frame].base_quat_wxyz = {1.0, 0.0, 0.0, 0.0};
    history[frame].base_ang_vel = {
        static_cast<double>(frame), static_cast<double>(frame + 1),
        static_cast<double>(frame + 2)};
    for (int joint = 0; joint < G1_NUM_MOTOR; ++joint) {
      history[frame].joint_pos_mujoco[joint] =
          0.25 + 100.0 * static_cast<double>(frame) + joint;
      history[frame].joint_vel_mujoco[joint] =
          200.0 * static_cast<double>(frame) + joint;
      history[frame].previous_raw_actions_mujoco[joint] =
          300.0 * static_cast<double>(frame) + joint;
    }
  }

  const auto obs = EmbeddedTrackingObservationBuilder::Build(
      history, control, {1.0, 0.0, 0.0, 0.0});
  EXPECT_FLOAT_EQ(obs[0], 0.0f);
  EXPECT_FLOAT_EQ(obs[1], 0.0f);
  EXPECT_FLOAT_EQ(obs[2], -1.0f);
  EXPECT_FLOAT_EQ(obs[30 + 9 * 3 + 2], 11.0f);
  EXPECT_FLOAT_EQ(obs[60 + 7 * G1_NUM_MOTOR + 4], 704.0f);
  EXPECT_FLOAT_EQ(obs[350 + 6 * G1_NUM_MOTOR + 3], 1203.0f);
  EXPECT_FLOAT_EQ(obs[640 + 5 * G1_NUM_MOTOR + 2], 1502.0f);
  for (size_t i = 930; i < 988; ++i) { EXPECT_FLOAT_EQ(obs[i], 0.0f); }
  const std::array<float, 6> identity_anchor = {1.0f, 0.0f, 0.0f,
                                                1.0f, 0.0f, 0.0f};
  for (size_t i = 0; i < identity_anchor.size(); ++i) {
    EXPECT_NEAR(obs[988 + i], identity_anchor[i], 1e-6f);
  }
}

TEST(EmbeddedTrackingPolicy, JointTargetUsesDirectHardwareIndex) {
  auto control = MakeControl();
  for (int i = 0; i < G1_NUM_MOTOR; ++i) {
    control.default_joint_pos[i] = static_cast<double>(i);
    control.action_scale[i] = 0.01 * static_cast<double>(i + 1);
  }
  EXPECT_DOUBLE_EQ(control.JointTarget(18, 2.0), 18.38);
  EXPECT_DOUBLE_EQ(control.JointTarget(25, -1.0), 24.74);
  EXPECT_THROW(control.JointTarget(G1_NUM_MOTOR, 0.0), std::runtime_error);
}

TEST(EmbeddedTrackingPolicy, RejectsWrongHistoryLength) {
  const auto control = MakeControl();
  std::vector<EmbeddedTrackingObservationState> short_history(9);
  EXPECT_THROW(EmbeddedTrackingObservationBuilder::Build(
                   short_history, control, {1.0, 0.0, 0.0, 0.0}),
               std::runtime_error);
}

}  // namespace
