#include <gtest/gtest.h>

#include "external_reference_tracking.hpp"

namespace ext = external_ref_tracking;

namespace {

ext::ControlConfig MakeControl() {
  ext::ControlConfig control;
  for (std::size_t i = 0; i < ext::kJointCount; ++i) {
    control.joint_names.emplace_back(ext::kHardwareJointNames[i]);
    control.default_joint_pos[i] = static_cast<float>(i) * 0.01f;
    control.action_scale[i] = 0.1f + static_cast<float>(i) * 0.001f;
    control.joint_stiffness[i] = 10.0f + static_cast<float>(i);
    control.joint_damping[i] = 1.0f + static_cast<float>(i) * 0.1f;
  }
  return control;
}

ext::ReferenceProvider MakeReference(std::size_t count) {
  std::vector<ext::ReferenceFrame> frames(count);
  for (std::size_t frame = 0; frame < count; ++frame) {
    frames[frame].root_quat_wxyz = {1.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t joint = 0; joint < ext::kJointCount; ++joint) {
      frames[frame].joint_pos[joint] = 1000.0f + 100.0f * frame + joint;
      frames[frame].joint_vel[joint] = 2000.0f + 100.0f * frame + joint;
    }
  }
  return ext::ReferenceProvider(std::move(frames));
}

}  // namespace

TEST(ExternalReferenceTracking, IdentityAnchorHasRequiredColumnMajorSemanticLayout) {
  const auto anchor = ext::RelativeRootAnchor(
      {1.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f});
  const ext::AnchorArray expected{1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
  EXPECT_EQ(anchor, expected);
}

TEST(ExternalReferenceTracking, ActionToTargetIsDirectHardwareOrder) {
  const auto control = MakeControl();
  for (std::size_t joint = 0; joint < ext::kJointCount; ++joint) {
    const float action = static_cast<float>(joint) + 0.25f;
    EXPECT_FLOAT_EQ(control.JointTarget(joint, action),
                    control.default_joint_pos[joint] +
                        action * control.action_scale[joint]);
  }
  // A reorder would map hardware index 0 to a different action. Pin that fact.
  EXPECT_NE(control.JointTarget(0, 0.25f), control.JointTarget(0, 6.25f));
}

TEST(ExternalReferenceTracking, ObservationUsesExactHistoryAndFutureOffsets) {
  const auto control = MakeControl();
  auto reference = MakeReference(60);
  ext::ObservationBuilder builder(control);
  ext::RobotState state;
  state.base_quat_wxyz = {1.0f, 0.0f, 0.0f, 0.0f};
  state.base_ang_vel = {0.1f, 0.2f, 0.3f};
  for (std::size_t joint = 0; joint < ext::kJointCount; ++joint) {
    state.joint_pos[joint] = control.default_joint_pos[joint] + 0.5f + joint;
    state.joint_vel[joint] = 10.0f + joint;
  }
  builder.Reset(state);
  ext::JointArray action{};
  for (std::size_t joint = 0; joint < ext::kJointCount; ++joint) {
    action[joint] = 20.0f + joint;
  }
  builder.PushRawAction(action);

  const auto obs = builder.Build(reference, 2, state.base_quat_wxyz);
  EXPECT_FLOAT_EQ(obs[0], 0.0f);
  EXPECT_FLOAT_EQ(obs[2], -1.0f);
  EXPECT_FLOAT_EQ(obs[30], 0.1f);
  EXPECT_FLOAT_EQ(obs[60], 0.5f);
  EXPECT_FLOAT_EQ(obs[350], 10.0f);
  EXPECT_FLOAT_EQ(obs[640 + 9 * ext::kJointCount], 20.0f);

  for (std::size_t future = 0; future < ext::kFutureFrames; ++future) {
    const std::size_t frame = 2 + ext::kFutureOffsets[future];
    const std::size_t base = 930 + future * 2 * ext::kJointCount;
    EXPECT_FLOAT_EQ(obs[base], 1000.0f + 100.0f * frame);
    EXPECT_FLOAT_EQ(obs[base + ext::kJointCount], 2000.0f + 100.0f * frame);
    EXPECT_FLOAT_EQ(obs[1510 + future * 6], 1.0f);
    EXPECT_FLOAT_EQ(obs[1510 + future * 6 + 3], 1.0f);
  }
}

TEST(ExternalReferenceTracking, FutureWindowClampsAtFinalFrame) {
  const auto control = MakeControl();
  auto reference = MakeReference(3);
  ext::ObservationBuilder builder(control);
  ext::RobotState state;
  builder.Reset(state);
  const auto obs = builder.Build(reference, 2, state.base_quat_wxyz);
  for (std::size_t future = 0; future < ext::kFutureFrames; ++future) {
    const std::size_t base = 930 + future * 2 * ext::kJointCount;
    EXPECT_FLOAT_EQ(obs[base], 1200.0f);
    EXPECT_FLOAT_EQ(obs[base + ext::kJointCount], 2200.0f);
  }
}
