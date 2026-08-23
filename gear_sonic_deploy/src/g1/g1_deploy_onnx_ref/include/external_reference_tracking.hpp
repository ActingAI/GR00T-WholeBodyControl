/**
 * @file external_reference_tracking.hpp
 * @brief External-reference tracking contract, reference loader, and obs builder.
 *
 * This backend is intentionally independent of the SONIC observation registry,
 * encoder, planner, action reorder, default angles, action scale, and PD gains.
 * The policy contract is exactly obs_dict[1,1570] -> action[1,29].
 */

#ifndef EXTERNAL_REFERENCE_TRACKING_HPP
#define EXTERNAL_REFERENCE_TRACKING_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

#include "cnpy.h"
#include "math_utils.hpp"
#include "robot_parameters.hpp"

namespace external_ref_tracking {

constexpr std::size_t kJointCount = G1_NUM_MOTOR;
constexpr std::size_t kHistoryFrames = 10;
constexpr std::size_t kFutureFrames = 10;
constexpr std::size_t kRobotHistoryDim = 930;
constexpr std::size_t kFutureJointDim = 580;
constexpr std::size_t kFutureAnchorDim = 60;
constexpr std::size_t kObservationDim = 1570;
constexpr std::array<int, kFutureFrames> kFutureOffsets = {
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45};

using JointArray = std::array<float, kJointCount>;
using QuatArray = std::array<float, 4>;
using Vec3Array = std::array<float, 3>;
using AnchorArray = std::array<float, 6>;
using Observation = std::array<float, kObservationDim>;

inline constexpr std::array<std::string_view, kJointCount> kHardwareJointNames = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"};

inline std::vector<std::string> SplitCsv(const std::string& value) {
  std::vector<std::string> result;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(',', begin);
    result.emplace_back(value.substr(begin, end == std::string::npos
                                               ? std::string::npos
                                               : end - begin));
    if (end == std::string::npos) { break; }
    begin = end + 1;
  }
  return result;
}

inline void ValidateJointNames(const std::vector<std::string>& names,
                               const std::string& source) {
  if (names.size() != kJointCount) {
    throw std::runtime_error(source + " must contain exactly 29 joint names");
  }
  for (std::size_t i = 0; i < kJointCount; ++i) {
    if (names[i] != kHardwareJointNames[i]) {
      throw std::runtime_error(
          source + " joint order mismatch at index " + std::to_string(i) +
          ": expected " + std::string(kHardwareJointNames[i]) + ", got " + names[i]);
    }
  }
}

inline std::vector<std::string> ReadOnnxJointNames(const std::string& model_path) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "external_ref_metadata");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(1);
  Ort::Session session(env, model_path.c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;
  auto metadata = session.GetModelMetadata();
  auto value = metadata.LookupCustomMetadataMapAllocated("joint_names", allocator);
  if (!value || std::string(value.get()).empty()) {
    throw std::runtime_error("ONNX has no joint_names metadata: " + model_path);
  }
  auto names = SplitCsv(value.get());
  ValidateJointNames(names, model_path + " metadata");
  return names;
}

struct ControlConfig {
  std::vector<std::string> joint_names;
  JointArray default_joint_pos{};
  JointArray action_scale{};
  JointArray joint_stiffness{};
  JointArray joint_damping{};
  std::string joint_order;

  static ControlConfig Load(const std::string& path) {
    std::ifstream stream(path);
    if (!stream.good()) {
      throw std::runtime_error("Cannot open external tracking control JSON: " + path);
    }
    nlohmann::json json;
    stream >> json;

    ControlConfig config;
    config.joint_order = json.at("joint_order").get<std::string>();
    if (config.joint_order.find("mujoco_hardware_order_from_metadata") != 0 ||
        config.joint_order.find("do not apply SONIC") == std::string::npos) {
      throw std::runtime_error(
          "control JSON does not explicitly prohibit the SONIC output reorder");
    }
    config.joint_names = json.at("joint_names").get<std::vector<std::string>>();
    ValidateJointNames(config.joint_names, path);

    const auto load_array = [&json, &path](const char* name, JointArray& output) {
      const auto values = json.at(name).get<std::vector<float>>();
      if (values.size() != kJointCount) {
        throw std::runtime_error(path + " field " + name + " must have 29 values");
      }
      for (std::size_t i = 0; i < kJointCount; ++i) {
        if (!std::isfinite(values[i])) {
          throw std::runtime_error(path + " field " + name + " contains non-finite data");
        }
        output[i] = values[i];
      }
    };
    load_array("default_joint_pos", config.default_joint_pos);
    load_array("action_scale", config.action_scale);
    load_array("joint_stiffness", config.joint_stiffness);
    load_array("joint_damping", config.joint_damping);
    for (std::size_t i = 0; i < kJointCount; ++i) {
      if (config.action_scale[i] <= 0.0f || config.joint_stiffness[i] <= 0.0f ||
          config.joint_damping[i] <= 0.0f) {
        throw std::runtime_error("action_scale/kp/kd must be positive at joint " +
                                 std::to_string(i));
      }
    }
    return config;
  }

  float JointTarget(std::size_t hardware_index, float raw_action) const {
    if (hardware_index >= kJointCount) { throw std::out_of_range("hardware joint index"); }
    return default_joint_pos[hardware_index] +
           raw_action * action_scale[hardware_index];
  }
};

template <std::size_t N>
inline std::array<float, N> LoadNpzFloatArray(const cnpy::npz_t& values,
                                              const std::string& name) {
  const auto iterator = values.find(name);
  if (iterator == values.end()) { throw std::runtime_error("NPZ missing field: " + name); }
  const auto& array = iterator->second;
  if (array.num_vals != N) {
    throw std::runtime_error("NPZ field " + name + " has wrong element count");
  }
  std::array<float, N> result{};
  if (array.word_size == sizeof(float)) {
    std::copy_n(array.data<float>(), N, result.begin());
  } else if (array.word_size == sizeof(double)) {
    const double* data = array.data<double>();
    std::transform(data, data + N, result.begin(),
                   [](double value) { return static_cast<float>(value); });
  } else {
    throw std::runtime_error("NPZ field " + name + " must be float32 or float64");
  }
  if (!std::all_of(result.begin(), result.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("NPZ field " + name + " contains non-finite data");
  }
  return result;
}

struct InitialState {
  JointArray body_qpos{};
  JointArray body_qvel{};
  QuatArray root_quat_wxyz{};
  float root_height_m = 0.0f;

  static InitialState Load(const std::string& path) {
    const auto values = cnpy::npz_load(path);
    InitialState state;
    state.body_qpos = LoadNpzFloatArray<kJointCount>(values, "body_qpos_mujoco");
    state.body_qvel = LoadNpzFloatArray<kJointCount>(values, "body_qvel_mujoco");
    state.root_quat_wxyz = LoadNpzFloatArray<4>(values, "root_quat_wxyz");
    state.root_height_m = LoadNpzFloatArray<1>(values, "root_height_m")[0];
    const float norm = std::sqrt(std::inner_product(
        state.root_quat_wxyz.begin(), state.root_quat_wxyz.end(),
        state.root_quat_wxyz.begin(), 0.0f));
    if (std::abs(norm - 1.0f) > 1e-3f || state.root_height_m <= 0.0f) {
      throw std::runtime_error("Invalid root quaternion/height in init NPZ");
    }
    return state;
  }
};

struct ReferenceFrame {
  JointArray joint_pos{};
  JointArray joint_vel{};
  QuatArray root_quat_wxyz{};
};

class ReferenceProvider {
 public:
  ReferenceProvider(const std::string& model_path, std::size_t frame_count)
      : model_path_(model_path), frame_count_(frame_count) {
    if (frame_count_ < 2) { throw std::runtime_error("reference frame count must be >= 2"); }
    Load();
  }

  // In-memory constructor used by deterministic unit tests and log replays.
  // Production deployment uses the ONNX constructor above.
  explicit ReferenceProvider(std::vector<ReferenceFrame> frames)
      : frame_count_(frames.size()), frames_(std::move(frames)) {
    if (frames_.empty()) { throw std::runtime_error("external reference is empty"); }
    joint_names_.reserve(kJointCount);
    for (const auto name : kHardwareJointNames) { joint_names_.emplace_back(name); }
  }

  std::size_t Size() const { return frames_.size(); }
  const std::vector<std::string>& JointNames() const { return joint_names_; }

  const ReferenceFrame& Frame(std::size_t index) const {
    if (frames_.empty()) { throw std::runtime_error("external reference is empty"); }
    return frames_[std::min(index, frames_.size() - 1)];
  }

 private:
  static void RequireShape(const Ort::Session& session, bool input, std::size_t index,
                           const std::vector<int64_t>& expected,
                           ONNXTensorElementDataType expected_type,
                           const std::string& label) {
    const auto info = input ? session.GetInputTypeInfo(index) : session.GetOutputTypeInfo(index);
    const auto tensor = info.GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != expected_type || tensor.GetShape() != expected) {
      throw std::runtime_error("Invalid tensor contract for " + label);
    }
  }

  void Load() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "external_reference_provider");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path_.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    if (session.GetInputCount() != 2 || session.GetOutputCount() < 4) {
      throw std::runtime_error("External reference ONNX has unexpected input/output count");
    }
    std::vector<std::string> input_names;
    for (std::size_t i = 0; i < session.GetInputCount(); ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      input_names.emplace_back(name.get());
    }
    const auto find_input = [&input_names](const std::string& name) {
      const auto iterator = std::find(input_names.begin(), input_names.end(), name);
      if (iterator == input_names.end()) {
        throw std::runtime_error("External reference ONNX missing input: " + name);
      }
      return static_cast<std::size_t>(std::distance(input_names.begin(), iterator));
    };
    RequireShape(session, true, find_input("obs"), {1, 994},
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "obs");
    RequireShape(session, true, find_input("time_step"), {1, 1},
                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, "time_step");

    auto metadata = session.GetModelMetadata();
    auto metadata_names = metadata.LookupCustomMetadataMapAllocated("joint_names", allocator);
    if (!metadata_names) {
      throw std::runtime_error("External reference ONNX has no joint_names metadata");
    }
    joint_names_ = SplitCsv(metadata_names.get());
    ValidateJointNames(joint_names_, model_path_ + " metadata");

    std::array<float, 994> zero_obs{};
    std::array<int64_t, 2> obs_shape{1, 994};
    std::array<int64_t, 2> time_shape{1, 1};
    const char* input_node_names[] = {"obs", "time_step"};
    const char* output_node_names[] = {"joint_pos", "joint_vel", "body_quat_w"};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    frames_.reserve(frame_count_);

    for (std::size_t frame_index = 0; frame_index < frame_count_; ++frame_index) {
      float time_step = static_cast<float>(frame_index);
      std::array<Ort::Value, 2> inputs = {
          Ort::Value::CreateTensor<float>(memory, zero_obs.data(), zero_obs.size(),
                                          obs_shape.data(), obs_shape.size()),
          Ort::Value::CreateTensor<float>(memory, &time_step, 1,
                                          time_shape.data(), time_shape.size())};
      auto outputs = session.Run(Ort::RunOptions{nullptr}, input_node_names, inputs.data(),
                                 inputs.size(), output_node_names, 3);
      if (outputs.size() != 3) {
        throw std::runtime_error("External reference ONNX returned wrong output count");
      }
      const auto q_info = outputs[0].GetTensorTypeAndShapeInfo();
      const auto dq_info = outputs[1].GetTensorTypeAndShapeInfo();
      const auto quat_info = outputs[2].GetTensorTypeAndShapeInfo();
      if (q_info.GetShape() != std::vector<int64_t>({1, 29}) ||
          dq_info.GetShape() != std::vector<int64_t>({1, 29}) ||
          quat_info.GetShape() != std::vector<int64_t>({1, 14, 4})) {
        throw std::runtime_error("External reference output shape mismatch");
      }
      ReferenceFrame frame;
      std::copy_n(outputs[0].GetTensorData<float>(), kJointCount,
                  frame.joint_pos.begin());
      std::copy_n(outputs[1].GetTensorData<float>(), kJointCount,
                  frame.joint_vel.begin());
      std::copy_n(outputs[2].GetTensorData<float>(), 4,
                  frame.root_quat_wxyz.begin());
      const bool finite =
          std::all_of(frame.joint_pos.begin(), frame.joint_pos.end(),
                      [](float value) { return std::isfinite(value); }) &&
          std::all_of(frame.joint_vel.begin(), frame.joint_vel.end(),
                      [](float value) { return std::isfinite(value); }) &&
          std::all_of(frame.root_quat_wxyz.begin(), frame.root_quat_wxyz.end(),
                      [](float value) { return std::isfinite(value); });
      if (!finite) { throw std::runtime_error("External reference contains non-finite data"); }
      frames_.push_back(frame);
    }

    // The declared count must end exactly at the graph's clamped terminal frame.
    float terminal_step = static_cast<float>(frame_count_);
    std::array<Ort::Value, 2> terminal_inputs = {
        Ort::Value::CreateTensor<float>(memory, zero_obs.data(), zero_obs.size(),
                                        obs_shape.data(), obs_shape.size()),
        Ort::Value::CreateTensor<float>(memory, &terminal_step, 1,
                                        time_shape.data(), time_shape.size())};
    auto terminal_outputs = session.Run(Ort::RunOptions{nullptr}, input_node_names,
                                        terminal_inputs.data(), terminal_inputs.size(),
                                        output_node_names, 3);
    const float* terminal_q = terminal_outputs[0].GetTensorData<float>();
    float terminal_error = 0.0f;
    for (std::size_t i = 0; i < kJointCount; ++i) {
      terminal_error = std::max(
          terminal_error, std::abs(terminal_q[i] - frames_.back().joint_pos[i]));
    }
    if (terminal_error > 1e-6f) {
      throw std::runtime_error(
          "reference frame count does not end at the ONNX clamp boundary");
    }
  }

  std::string model_path_;
  std::size_t frame_count_;
  std::vector<std::string> joint_names_;
  std::vector<ReferenceFrame> frames_;
};

struct RobotState {
  QuatArray base_quat_wxyz{1.0f, 0.0f, 0.0f, 0.0f};
  Vec3Array base_ang_vel{};
  JointArray joint_pos{};
  JointArray joint_vel{};
};

inline Vec3Array ProjectedGravity(const QuatArray& base_quat_wxyz) {
  const auto unit_quat = quat_unit(base_quat_wxyz);
  const auto inverse = quat_conjugate(unit_quat);
  return quat_rotate(inverse, Vec3Array{0.0f, 0.0f, -1.0f});
}

inline AnchorArray RelativeRootAnchor(const QuatArray& robot_quat_wxyz,
                                      const QuatArray& reference_quat_wxyz) {
  const auto robot = quat_unit(robot_quat_wxyz);
  const auto reference = quat_unit(reference_quat_wxyz);
  const auto relative = quat_unit(quat_mul(quat_conjugate(robot), reference));
  const float w = relative[0];
  const float x = relative[1];
  const float y = relative[2];
  const float z = relative[3];
  const float r00 = 1.0f - 2.0f * (y * y + z * z);
  const float r01 = 2.0f * (x * y - z * w);
  const float r10 = 2.0f * (x * y + z * w);
  const float r11 = 1.0f - 2.0f * (x * x + z * z);
  const float r20 = 2.0f * (x * z - y * w);
  const float r21 = 2.0f * (y * z + x * w);
  return {r00, r01, r10, r11, r20, r21};
}

class ObservationBuilder {
 public:
  explicit ObservationBuilder(const ControlConfig& control) : control_(control) {}

  void Reset(const RobotState& state) {
    gravity_history_.clear();
    angular_velocity_history_.clear();
    joint_position_history_.clear();
    joint_velocity_history_.clear();
    action_history_.clear();
    const auto gravity = ProjectedGravity(state.base_quat_wxyz);
    const auto relative_q = RelativeJointPosition(state.joint_pos);
    const JointArray zero_action{};
    for (std::size_t i = 0; i < kHistoryFrames; ++i) {
      gravity_history_.push_back(gravity);
      angular_velocity_history_.push_back(state.base_ang_vel);
      joint_position_history_.push_back(relative_q);
      joint_velocity_history_.push_back(state.joint_vel);
      action_history_.push_back(zero_action);
    }
  }

  void PushRobotState(const RobotState& state) {
    RequireInitialized();
    Push(gravity_history_, ProjectedGravity(state.base_quat_wxyz));
    Push(angular_velocity_history_, state.base_ang_vel);
    Push(joint_position_history_, RelativeJointPosition(state.joint_pos));
    Push(joint_velocity_history_, state.joint_vel);
  }

  void PushRawAction(const JointArray& action) {
    RequireInitialized();
    Push(action_history_, action);
  }

  Observation Build(const ReferenceProvider& reference, std::size_t frame_index,
                    const QuatArray& robot_quat_wxyz) const {
    RequireInitialized();
    Observation observation{};
    CopyFrames(gravity_history_, observation, 0);
    CopyFrames(angular_velocity_history_, observation, 30);
    CopyFrames(joint_position_history_, observation, 60);
    CopyFrames(joint_velocity_history_, observation, 350);
    CopyFrames(action_history_, observation, 640);

    std::size_t future_joint_offset = 930;
    std::size_t future_anchor_offset = 1510;
    for (std::size_t future = 0; future < kFutureFrames; ++future) {
      const std::size_t selected = std::min(
          frame_index + static_cast<std::size_t>(kFutureOffsets[future]),
          reference.Size() - 1);
      const auto& frame = reference.Frame(selected);
      std::copy(frame.joint_pos.begin(), frame.joint_pos.end(),
                observation.begin() + future_joint_offset);
      future_joint_offset += kJointCount;
      std::copy(frame.joint_vel.begin(), frame.joint_vel.end(),
                observation.begin() + future_joint_offset);
      future_joint_offset += kJointCount;
      const auto anchor = RelativeRootAnchor(robot_quat_wxyz, frame.root_quat_wxyz);
      std::copy(anchor.begin(), anchor.end(),
                observation.begin() + future_anchor_offset);
      future_anchor_offset += anchor.size();
    }
    return observation;
  }

 private:
  template <typename T>
  static void Push(std::deque<T>& history, const T& value) {
    history.pop_front();
    history.push_back(value);
  }

  template <typename T>
  static void CopyFrames(const std::deque<T>& history, Observation& output,
                         std::size_t offset) {
    for (const auto& frame : history) {
      std::copy(frame.begin(), frame.end(), output.begin() + offset);
      offset += frame.size();
    }
  }

  JointArray RelativeJointPosition(const JointArray& absolute) const {
    JointArray relative{};
    for (std::size_t i = 0; i < kJointCount; ++i) {
      relative[i] = absolute[i] - control_.default_joint_pos[i];
    }
    return relative;
  }

  void RequireInitialized() const {
    if (gravity_history_.size() != kHistoryFrames ||
        angular_velocity_history_.size() != kHistoryFrames ||
        joint_position_history_.size() != kHistoryFrames ||
        joint_velocity_history_.size() != kHistoryFrames ||
        action_history_.size() != kHistoryFrames) {
      throw std::runtime_error("External-reference observation history is not initialized");
    }
  }

  const ControlConfig& control_;
  std::deque<Vec3Array> gravity_history_;
  std::deque<Vec3Array> angular_velocity_history_;
  std::deque<JointArray> joint_position_history_;
  std::deque<JointArray> joint_velocity_history_;
  std::deque<JointArray> action_history_;
};

}  // namespace external_ref_tracking

#endif  // EXTERNAL_REFERENCE_TRACKING_HPP
