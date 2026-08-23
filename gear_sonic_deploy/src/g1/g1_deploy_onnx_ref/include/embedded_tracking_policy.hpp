/**
 * @file embedded_tracking_policy.hpp
 * @brief Isolated TensorRT runtime and deployment contract for embedded tracking ONNX policies.
 *
 * This path is deliberately separate from SONIC's PolicyEngine/EncoderEngine.
 * Embedded tracking policies own their reference motion and expose two inputs
 * (`obs`, `time_step`) plus an `actions` output in MuJoCo/G1 hardware order.
 */

#ifndef EMBEDDED_TRACKING_POLICY_HPP
#define EMBEDDED_TRACKING_POLICY_HPP

#include <TRTInference/InferenceEngine.h>
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cnpy.h"
#include "math_utils.hpp"
#include "robot_parameters.hpp"

enum class DeployPolicyType {
  SONIC,
  EMBEDDED_TRACKING_ONNX,
};

inline DeployPolicyType ParseDeployPolicyType(const std::string& value) {
  if (value == "sonic") { return DeployPolicyType::SONIC; }
  if (value == "embedded_tracking_onnx") {
    return DeployPolicyType::EMBEDDED_TRACKING_ONNX;
  }
  throw std::runtime_error(
      "Unsupported policy type '" + value +
      "' (expected 'sonic' or 'embedded_tracking_onnx')");
}

inline const char* DeployPolicyTypeName(DeployPolicyType type) {
  return type == DeployPolicyType::EMBEDDED_TRACKING_ONNX
             ? "embedded_tracking_onnx"
             : "sonic";
}

struct EmbeddedTrackingControlConfig {
  static constexpr size_t kNumJoints = G1_NUM_MOTOR;
  static constexpr size_t kObservationDim = 994;
  static constexpr size_t kHistoryLength = 10;

  int schema_version = 0;
  int frame_count = 0;
  int fps = 0;
  double control_dt = 0.0;
  std::string policy_name;
  std::string joint_order;
  std::string action_order;
  std::vector<std::string> joint_names;
  std::array<double, kNumJoints> default_joint_pos{};
  std::array<double, kNumJoints> action_scale{};
  std::array<double, kNumJoints> joint_stiffness{};
  std::array<double, kNumJoints> joint_damping{};

  static const std::array<std::string_view, kNumJoints>& ExpectedJointNames() {
    static const std::array<std::string_view, kNumJoints> names = {
        "left_hip_pitch_joint",     "left_hip_roll_joint",
        "left_hip_yaw_joint",       "left_knee_joint",
        "left_ankle_pitch_joint",   "left_ankle_roll_joint",
        "right_hip_pitch_joint",    "right_hip_roll_joint",
        "right_hip_yaw_joint",      "right_knee_joint",
        "right_ankle_pitch_joint",  "right_ankle_roll_joint",
        "waist_yaw_joint",          "waist_roll_joint",
        "waist_pitch_joint",        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint",         "left_wrist_roll_joint",
        "left_wrist_pitch_joint",   "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint",   "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    };
    return names;
  }

  static EmbeddedTrackingControlConfig Load(const std::string& path) {
    if (path.empty()) {
      throw std::runtime_error("Embedded tracking control JSON path is empty");
    }
    std::ifstream stream(path);
    if (!stream.good()) {
      throw std::runtime_error("Cannot open embedded tracking control JSON: " + path);
    }

    nlohmann::json json;
    stream >> json;

    EmbeddedTrackingControlConfig config;
    config.schema_version = json.at("schema_version").get<int>();
    config.policy_name = json.at("policy_name").get<std::string>();
    config.frame_count = json.at("frame_count").get<int>();
    config.fps = json.at("fps").get<int>();
    config.control_dt = json.at("control_dt").get<double>();
    config.joint_order = json.at("joint_order").get<std::string>();
    config.action_order = json.at("action_order").get<std::string>();
    config.joint_names = json.at("joint_names").get<std::vector<std::string>>();
    config.default_joint_pos = ReadArray(json, "default_joint_pos");
    config.action_scale = ReadArray(json, "action_scale");
    config.joint_stiffness = ReadArray(json, "joint_stiffness");
    config.joint_damping = ReadArray(json, "joint_damping");
    config.Validate();
    return config;
  }

  void Validate() const {
    if (schema_version != 1) {
      throw std::runtime_error("Unsupported embedded tracking control schema version: " +
                               std::to_string(schema_version));
    }
    if (frame_count <= 0 || fps != 50 || std::abs(control_dt - 0.02) > 1e-9) {
      throw std::runtime_error(
          "Embedded tracking policy must declare a positive frame_count at 50 Hz / 0.02 s");
    }
    constexpr std::string_view expected_order = "mujoco_g1_29dof_hardware_order";
    if (joint_order != expected_order || action_order != expected_order) {
      throw std::runtime_error(
          "Embedded tracking joint/action order must be mujoco_g1_29dof_hardware_order");
    }
    if (joint_names.size() != kNumJoints) {
      throw std::runtime_error("Embedded tracking control JSON must contain exactly 29 joint names");
    }
    const auto& expected_names = ExpectedJointNames();
    for (size_t i = 0; i < kNumJoints; ++i) {
      if (joint_names[i] != expected_names[i]) {
        throw std::runtime_error("Embedded tracking joint name/order mismatch at index " +
                                 std::to_string(i) + ": " + joint_names[i]);
      }
      if (!std::isfinite(default_joint_pos[i]) || !std::isfinite(action_scale[i]) ||
          !std::isfinite(joint_stiffness[i]) || !std::isfinite(joint_damping[i])) {
        throw std::runtime_error("Non-finite embedded tracking control value at joint " +
                                 std::to_string(i));
      }
      if (action_scale[i] <= 0.0 || joint_stiffness[i] <= 0.0 ||
          joint_damping[i] <= 0.0) {
        throw std::runtime_error("Embedded tracking scale/Kp/Kd must be positive at joint " +
                                 std::to_string(i));
      }
    }
  }

  double JointTarget(size_t hardware_joint_index, double raw_action) const {
    if (hardware_joint_index >= kNumJoints || !std::isfinite(raw_action)) {
      throw std::runtime_error("Invalid embedded tracking joint/action target request");
    }
    return default_joint_pos[hardware_joint_index] +
           raw_action * action_scale[hardware_joint_index];
  }

 private:
  static std::array<double, kNumJoints> ReadArray(const nlohmann::json& json,
                                                   const char* key) {
    const auto values = json.at(key).get<std::vector<double>>();
    if (values.size() != kNumJoints) {
      throw std::runtime_error(std::string("Embedded tracking field '") + key +
                               "' must contain exactly 29 values");
    }
    std::array<double, kNumJoints> result{};
    std::copy(values.begin(), values.end(), result.begin());
    return result;
  }
};

struct EmbeddedTrackingInitState {
  std::array<double, G1_NUM_MOTOR> body_qpos_mujoco{};
  std::array<double, G1_NUM_MOTOR> body_qvel_mujoco{};
  std::array<double, 4> root_quat_wxyz{1.0, 0.0, 0.0, 0.0};
  double root_height_m = 0.0;

  static EmbeddedTrackingInitState Load(const std::string& path) {
    if (path.empty()) {
      throw std::runtime_error("Embedded tracking init-state NPZ path is empty");
    }
    if (!std::filesystem::is_regular_file(path)) {
      throw std::runtime_error("Cannot open embedded tracking init-state NPZ: " + path);
    }

    const cnpy::npz_t arrays = cnpy::npz_load(path);
    EmbeddedTrackingInitState state;
    state.body_qpos_mujoco = ReadFloatArray<G1_NUM_MOTOR>(arrays, "body_qpos_mujoco");
    state.body_qvel_mujoco = ReadFloatArray<G1_NUM_MOTOR>(arrays, "body_qvel_mujoco");
    state.root_quat_wxyz = ReadFloatArray<4>(arrays, "root_quat_wxyz");
    state.root_height_m = ReadFloatScalar(arrays, "root_height_m");
    state.Validate();
    return state;
  }

  void Validate() const {
    for (double value : body_qpos_mujoco) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite body_qpos_mujoco value in init state");
      }
    }
    for (double value : body_qvel_mujoco) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite body_qvel_mujoco value in init state");
      }
    }
    double quat_norm_sq = 0.0;
    for (double value : root_quat_wxyz) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite root_quat_wxyz value in init state");
      }
      quat_norm_sq += value * value;
    }
    if (std::abs(std::sqrt(quat_norm_sq) - 1.0) > 1e-3) {
      throw std::runtime_error("init-state root_quat_wxyz is not normalized");
    }
    if (!std::isfinite(root_height_m) || root_height_m <= 0.0) {
      throw std::runtime_error("init-state root_height_m must be finite and positive");
    }
  }

 private:
  template <size_t N>
  static std::array<double, N> ReadFloatArray(const cnpy::npz_t& arrays,
                                               const std::string& key) {
    const auto it = arrays.find(key);
    if (it == arrays.end()) {
      throw std::runtime_error("Missing init-state NPZ field: " + key);
    }
    const cnpy::NpyArray& array = it->second;
    if (array.word_size != sizeof(float) || array.num_vals != N || array.fortran_order) {
      throw std::runtime_error("Invalid float32 shape/layout for init-state NPZ field: " + key);
    }
    const float* data = array.data<float>();
    std::array<double, N> result{};
    for (size_t i = 0; i < N; ++i) { result[i] = static_cast<double>(data[i]); }
    return result;
  }

  static double ReadFloatScalar(const cnpy::npz_t& arrays, const std::string& key) {
    const auto it = arrays.find(key);
    if (it == arrays.end()) {
      throw std::runtime_error("Missing init-state NPZ field: " + key);
    }
    const cnpy::NpyArray& array = it->second;
    if (array.word_size != sizeof(float) || array.num_vals != 1 || array.fortran_order) {
      throw std::runtime_error("Invalid float32 scalar for init-state NPZ field: " + key);
    }
    return static_cast<double>(array.data<float>()[0]);
  }
};

struct EmbeddedTrackingObservationState {
  std::array<double, 4> base_quat_wxyz{1.0, 0.0, 0.0, 0.0};
  std::array<double, 3> base_ang_vel{};
  std::array<double, G1_NUM_MOTOR> joint_pos_mujoco{};
  std::array<double, G1_NUM_MOTOR> joint_vel_mujoco{};
  std::array<double, G1_NUM_MOTOR> previous_raw_actions_mujoco{};
};

class EmbeddedTrackingObservationBuilder {
 public:
  using Observation = std::array<float, EmbeddedTrackingControlConfig::kObservationDim>;

  static Observation Build(
      const std::vector<EmbeddedTrackingObservationState>& history_oldest_first,
      const EmbeddedTrackingControlConfig& control,
      const std::array<double, 4>& reference_root_quat_wxyz) {
    if (history_oldest_first.size() != EmbeddedTrackingControlConfig::kHistoryLength) {
      throw std::runtime_error("Embedded tracking observation history must contain exactly 10 frames");
    }

    Observation observation{};
    for (size_t frame = 0; frame < history_oldest_first.size(); ++frame) {
      const auto& state = history_oldest_first[frame];
      const auto gravity = ProjectedGravity(state.base_quat_wxyz);
      for (size_t axis = 0; axis < 3; ++axis) {
        observation[frame * 3 + axis] = static_cast<float>(gravity[axis]);
        observation[30 + frame * 3 + axis] = static_cast<float>(state.base_ang_vel[axis]);
      }
      for (size_t joint = 0; joint < G1_NUM_MOTOR; ++joint) {
        observation[60 + frame * G1_NUM_MOTOR + joint] = static_cast<float>(
            state.joint_pos_mujoco[joint] - control.default_joint_pos[joint]);
        observation[350 + frame * G1_NUM_MOTOR + joint] =
            static_cast<float>(state.joint_vel_mujoco[joint]);
        observation[640 + frame * G1_NUM_MOTOR + joint] =
            static_cast<float>(state.previous_raw_actions_mujoco[joint]);
      }
    }

    // 930:988 is a reserved command vector and must remain zero.
    const auto anchor = Anchor6D(history_oldest_first.back().base_quat_wxyz,
                                 reference_root_quat_wxyz);
    std::copy(anchor.begin(), anchor.end(), observation.begin() + 988);

    for (float value : observation) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Embedded tracking observation contains a non-finite value");
      }
    }
    return observation;
  }

  static std::array<float, 6> Anchor6D(
      const std::array<double, 4>& robot_base_quat_wxyz,
      const std::array<double, 4>& reference_root_quat_wxyz) {
    const auto robot_quat = NormalizeQuaternion(robot_base_quat_wxyz, "robot base quaternion");
    const auto reference_quat = NormalizeQuaternion(reference_root_quat_wxyz,
                                                     "reference root quaternion");
    const auto base_to_reference =
        quat_mul_d(quat_conjugate_d(robot_quat), reference_quat);
    const auto rotation = quat_to_rotation_matrix_d(base_to_reference);
    return {
        static_cast<float>(rotation[0][0]), static_cast<float>(rotation[0][1]),
        static_cast<float>(rotation[1][0]), static_cast<float>(rotation[1][1]),
        static_cast<float>(rotation[2][0]), static_cast<float>(rotation[2][1]),
    };
  }

  static std::array<double, 3> ProjectedGravity(
      const std::array<double, 4>& base_quat_wxyz) {
    const auto base_quat = NormalizeQuaternion(base_quat_wxyz, "base quaternion");
    return quat_rotate_d(quat_conjugate_d(base_quat), {0.0, 0.0, -1.0});
  }

 private:
  static std::array<double, 4> NormalizeQuaternion(
      const std::array<double, 4>& quat,
      const char* description) {
    double norm_sq = 0.0;
    for (double value : quat) {
      if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(description) + " contains a non-finite value");
      }
      norm_sq += value * value;
    }
    const double norm = std::sqrt(norm_sq);
    if (norm < 1e-9) {
      throw std::runtime_error(std::string(description) + " has near-zero norm");
    }
    return {quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm};
  }
};

class EmbeddedTrackingPolicyEngine {
 public:
  EmbeddedTrackingPolicyEngine() = default;
  ~EmbeddedTrackingPolicyEngine() { Destroy(); }

  EmbeddedTrackingPolicyEngine(const EmbeddedTrackingPolicyEngine&) = delete;
  EmbeddedTrackingPolicyEngine& operator=(const EmbeddedTrackingPolicyEngine&) = delete;

  bool Initialize(const std::string& model_path, bool use_fp16 = false) {
    if (model_path.empty()) {
      std::cerr << "EmbeddedTrackingPolicyEngine: empty model path" << std::endl;
      return false;
    }
    try {
      inference_engine_ = std::make_unique<TRTInferenceEngine>();
      Options options;
      options.deviceID = 0;
      std::string prefix = "embedded_tracking_";
      if (use_fp16) {
        options.precision = Precision::FP16;
        prefix += "fp16_";
      }

      std::string cached_trt_file;
      if (!ConvertONNXToTRT(options, model_path, cached_trt_file, prefix, false)) {
        std::cerr << "Failed to convert embedded tracking ONNX to TensorRT: "
                  << model_path << std::endl;
        return false;
      }
      if (!inference_engine_->Initialize(cached_trt_file, options.deviceID,
                                         options.dynamic_axes_names) ||
          !inference_engine_->InitInputs({})) {
        std::cerr << "Failed to initialize embedded tracking TensorRT engine: "
                  << cached_trt_file << std::endl;
        return false;
      }
      if (!ValidateTensorContract()) { return false; }

      obs_buffer_.resize(EmbeddedTrackingControlConfig::kObservationDim, 0.0f);
      time_step_buffer_.resize(1, 0.0f);
      action_buffer_.resize(G1_NUM_MOTOR, 0.0f);
      inference_engine_->SetInputData("obs", obs_buffer_);
      inference_engine_->SetInputData("time_step", time_step_buffer_);

      const cudaError_t status = cudaStreamCreate(&cuda_stream_);
      if (status != cudaSuccess) {
        std::cerr << "Failed to create embedded tracking CUDA stream: "
                  << cudaGetErrorString(status) << std::endl;
        return false;
      }

      initialized_ = true;
      std::cout << "Embedded tracking policy initialized" << std::endl;
      std::cout << "  ONNX: " << model_path << std::endl;
      std::cout << "  TensorRT cache: " << cached_trt_file << std::endl;
      std::cout << "  Inputs: obs[1,994] float32, time_step[1,1] float32" << std::endl;
      std::cout << "  Output: actions[1,29] float32" << std::endl;
      return true;
    } catch (const std::exception& error) {
      std::cerr << "EmbeddedTrackingPolicyEngine initialization failed: "
                << error.what() << std::endl;
      Destroy();
      return false;
    }
  }

  bool CaptureGraph() {
    if (!initialized_ || !inference_engine_) { return false; }
    if (graph_captured_) { return true; }
    cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeRelaxed);
    if (!inference_engine_->Enqueue(cuda_stream_)) {
      cudaStreamEndCapture(cuda_stream_, &cuda_graph_);
      return false;
    }
    if (cudaStreamEndCapture(cuda_stream_, &cuda_graph_) != cudaSuccess ||
        cuda_graph_ == nullptr) {
      return false;
    }
    if (cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0) !=
        cudaSuccess) {
      return false;
    }
    graph_captured_ = true;
    return true;
  }

  void SetObservation(const EmbeddedTrackingObservationBuilder::Observation& observation) {
    std::copy(observation.begin(), observation.end(), obs_buffer_.begin());
  }

  void SetTimeStep(int time_step) {
    time_step_buffer_[0] = static_cast<float>(time_step);
  }

  bool Infer() {
    if (!initialized_ || !inference_engine_) { return false; }
    inference_engine_->SetInputData("obs", obs_buffer_);
    inference_engine_->SetInputData("time_step", time_step_buffer_);

    if (graph_captured_) {
      if (cudaGraphLaunch(cuda_graph_exec_, cuda_stream_) != cudaSuccess) { return false; }
    } else if (!inference_engine_->Enqueue(cuda_stream_)) {
      return false;
    }
    inference_engine_->GetOutputDataAsync("actions", action_buffer_, cuda_stream_);
    return cudaStreamSynchronize(cuda_stream_) == cudaSuccess;
  }

  const TPinnedVector<float>& GetActionBuffer() const { return action_buffer_; }
  bool IsInitialized() const { return initialized_; }

  void Destroy() {
    if (cuda_graph_exec_ != nullptr) {
      cudaGraphExecDestroy(cuda_graph_exec_);
      cuda_graph_exec_ = nullptr;
    }
    if (cuda_graph_ != nullptr) {
      cudaGraphDestroy(cuda_graph_);
      cuda_graph_ = nullptr;
    }
    if (cuda_stream_ != nullptr) {
      cudaStreamDestroy(cuda_stream_);
      cuda_stream_ = nullptr;
    }
    if (inference_engine_) {
      inference_engine_->Destroy();
      inference_engine_.reset();
    }
    graph_captured_ = false;
    initialized_ = false;
  }

 private:
  bool ValidateTensorContract() const {
    const auto inputs = inference_engine_->GetInputTensorNames();
    if (inputs.size() != 2 ||
        std::find(inputs.begin(), inputs.end(), "obs") == inputs.end() ||
        std::find(inputs.begin(), inputs.end(), "time_step") == inputs.end()) {
      std::cerr << "Embedded tracking ONNX must have exactly inputs 'obs' and 'time_step'"
                << std::endl;
      return false;
    }
    const auto outputs = inference_engine_->GetOutputTensorNames();
    if (std::find(outputs.begin(), outputs.end(), "actions") == outputs.end()) {
      std::cerr << "Embedded tracking ONNX is missing output 'actions'" << std::endl;
      return false;
    }
    if (inference_engine_->GetTensorDataType("obs") != DataType::FLOAT ||
        inference_engine_->GetTensorDataType("time_step") != DataType::FLOAT ||
        inference_engine_->GetTensorDataType("actions") != DataType::FLOAT) {
      std::cerr << "Embedded tracking obs/time_step/actions tensors must be float32"
                << std::endl;
      return false;
    }
    return ValidateShape(
               "obs", {1, EmbeddedTrackingControlConfig::kObservationDim}) &&
           ValidateShape("time_step", {1, 1}) &&
           ValidateShape("actions", {1, G1_NUM_MOTOR});
  }

  bool ValidateShape(const std::string& tensor,
                     const std::vector<size_t>& expected_shape) const {
    std::vector<int64_t> shape;
    if (!inference_engine_->GetTensorShape(tensor, shape)) {
      std::cerr << "Cannot read TensorRT tensor shape: " << tensor << std::endl;
      return false;
    }
    if (shape.size() != expected_shape.size()) {
      std::cerr << "Embedded tracking tensor '" << tensor << "' has rank "
                << shape.size() << ", expected " << expected_shape.size() << std::endl;
      return false;
    }
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] <= 0 || static_cast<size_t>(shape[i]) != expected_shape[i]) {
        std::cerr << "Embedded tracking tensor '" << tensor
                  << "' shape mismatch at dimension " << i << ": " << shape[i]
                  << " vs " << expected_shape[i] << std::endl;
        return false;
      }
    }
    return true;
  }

  std::unique_ptr<TRTInferenceEngine> inference_engine_;
  TPinnedVector<float> obs_buffer_;
  TPinnedVector<float> time_step_buffer_;
  TPinnedVector<float> action_buffer_;
  cudaStream_t cuda_stream_ = nullptr;
  cudaGraph_t cuda_graph_ = nullptr;
  cudaGraphExec_t cuda_graph_exec_ = nullptr;
  bool initialized_ = false;
  bool graph_captured_ = false;
};

#endif  // EMBEDDED_TRACKING_POLICY_HPP
