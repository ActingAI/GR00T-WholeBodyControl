#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "external_reference_tracking.hpp"

namespace ext = external_ref_tracking;

namespace {

template <typename Range>
std::pair<float, float> MinMax(const Range& values) {
  const auto result = std::minmax_element(values.begin(), values.end());
  return {*result.first, *result.second};
}

void PrintRange(const std::string& label, const auto& values) {
  const auto [minimum, maximum] = MinMax(values);
  std::cout << label << "=[" << minimum << ", " << maximum << "]";
}

void RequirePolicyContract(Ort::Session& session) {
  Ort::AllocatorWithDefaultOptions allocator;
  if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
    throw std::runtime_error("policy must have exactly one input and one output");
  }
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto output_name = session.GetOutputNameAllocated(0, allocator);
  const auto input_type_info = session.GetInputTypeInfo(0);
  const auto output_type_info = session.GetOutputTypeInfo(0);
  const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
  const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
  const auto input_shape = input_info.GetShape();
  const auto output_shape = output_info.GetShape();
  if (std::string(input_name.get()) != "obs_dict" ||
      std::string(output_name.get()) != "action" ||
      input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      input_shape != std::vector<int64_t>({1, 1570}) ||
      output_shape != std::vector<int64_t>({1, 29})) {
    throw std::runtime_error(
        "policy contract mismatch: input=" + std::string(input_name.get()) + " type=" +
        std::to_string(input_info.GetElementType()) + " shape=[" +
        std::to_string(input_shape.at(0)) + "," + std::to_string(input_shape.at(1)) +
        "] output=" + std::string(output_name.get()) + " type=" +
        std::to_string(output_info.GetElementType()) + " shape=[" +
        std::to_string(output_shape.at(0)) + "," + std::to_string(output_shape.at(1)) + "]");
  }
}

void WriteCsvHeader(std::ofstream& csv) {
  csv << "frame,obs_min,obs_max,action_min,action_max,q_target_min,q_target_max";
  for (const auto name : ext::kHardwareJointNames) csv << ",raw_" << name;
  for (const auto name : ext::kHardwareJointNames) csv << ",q_target_" << name;
  csv << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 5 || argc > 8) {
      std::cerr << "Usage: " << argv[0]
                << " POLICY.onnx REFERENCE.onnx CONTROL.json INIT.npz"
                   " [REFERENCE_FRAMES=941] [REPLAY_FRAMES=5] [LOG.csv]\n";
      return 2;
    }
    const std::string policy_path = argv[1];
    const std::string reference_path = argv[2];
    const std::string control_path = argv[3];
    const std::string init_path = argv[4];
    const std::size_t reference_frames = argc >= 6 ? std::stoul(argv[5]) : 941;
    const std::size_t replay_frames = argc >= 7 ? std::stoul(argv[6]) : 5;
    const std::string log_path = argc >= 8 ? argv[7] : "";
    if (replay_frames == 0 || replay_frames > reference_frames) {
      throw std::runtime_error("REPLAY_FRAMES must be in [1, REFERENCE_FRAMES]");
    }

    const auto control = ext::ControlConfig::Load(control_path);
    const auto init = ext::InitialState::Load(init_path);
    ext::ReferenceProvider reference(reference_path, reference_frames);
    const auto policy_names = ext::ReadOnnxJointNames(policy_path);
    ext::ValidateJointNames(reference.JointNames(), "reference ONNX");
    if (policy_names != control.joint_names || policy_names != reference.JointNames()) {
      throw std::runtime_error("policy/reference/control joint names differ");
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "external_ref_dry_run");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session policy(env, policy_path.c_str(), options);
    RequirePolicyContract(policy);
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 2> obs_shape{1, 1570};
    const char* input_names[] = {"obs_dict"};
    const char* output_names[] = {"action"};

    std::ofstream csv;
    if (!log_path.empty()) {
      csv.open(log_path, std::ios::trunc);
      if (!csv.good()) throw std::runtime_error("cannot open log CSV: " + log_path);
      csv << std::setprecision(9);
      WriteCsvHeader(csv);
    }

    ext::RobotState state;
    // Match the package's verified --init-from-ref replay: reference frame 0
    // supplies root orientation and joints; the NPZ supplies the root height
    // (not represented in this orientation/joint-only observation).
    state.base_quat_wxyz = reference.Frame(0).root_quat_wxyz;
    state.joint_pos = reference.Frame(0).joint_pos;
    state.joint_vel.fill(0.0f);
    ext::ObservationBuilder builder(control);
    builder.Reset(state);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "EXTERNAL_REF_DRY_RUN contract=PASS\n"
              << "policy_input=obs_dict[1,1570] policy_output=action[1,29]\n"
              << "reference_frames=" << reference.Size()
              << " future_offsets=[0,5,10,15,20,25,30,35,40,45]\n"
              << "joint_order=metadata_mujoco_hardware count=29 EXACT_MATCH=true\n"
              << "SONIC_OUTPUT_REORDER_APPLIED=false\n";
    for (std::size_t joint = 0; joint < ext::kJointCount; ++joint) {
      std::cout << "joint[" << joint << "]=" << ext::kHardwareJointNames[joint]
                << " action_index=" << joint << " (direct)\n";
    }

    for (std::size_t frame_index = 0; frame_index < replay_frames; ++frame_index) {
      if (frame_index > 0) {
        const auto& ref = reference.Frame(frame_index);
        state.base_quat_wxyz = ref.root_quat_wxyz;
        state.joint_pos = ref.joint_pos;
        state.joint_vel = ref.joint_vel;
      }
      builder.PushRobotState(state);
      auto obs = builder.Build(reference, frame_index, state.base_quat_wxyz);
      auto input = Ort::Value::CreateTensor<float>(
          memory, obs.data(), obs.size(), obs_shape.data(), obs_shape.size());
      auto outputs = policy.Run(Ort::RunOptions{nullptr}, input_names, &input, 1,
                                output_names, 1);
      const float* action_data = outputs[0].GetTensorData<float>();
      ext::JointArray action{};
      ext::JointArray q_target{};
      std::copy_n(action_data, ext::kJointCount, action.begin());
      for (std::size_t joint = 0; joint < ext::kJointCount; ++joint) {
        if (!std::isfinite(action[joint])) {
          throw std::runtime_error("policy produced non-finite action");
        }
        q_target[joint] = control.JointTarget(joint, action[joint]);
      }
      builder.PushRawAction(action);

      const auto [obs_min, obs_max] = MinMax(obs);
      const auto [action_min, action_max] = MinMax(action);
      const auto [target_min, target_max] = MinMax(q_target);
      std::cout << "frame=" << frame_index << " obs_shape=[1,1570] ";
      PrintRange("obs", obs);
      std::cout << ' ';
      PrintRange("robot_history", std::vector<float>(obs.begin(), obs.begin() + 930));
      std::cout << ' ';
      PrintRange("future_joint", std::vector<float>(obs.begin() + 930, obs.begin() + 1510));
      std::cout << ' ';
      PrintRange("anchor", std::vector<float>(obs.begin() + 1510, obs.end()));
      std::cout << ' ';
      PrintRange("action", action);
      std::cout << ' ';
      PrintRange("q_target", q_target);
      std::cout << '\n';

      if (csv.good()) {
        csv << frame_index << ',' << obs_min << ',' << obs_max << ','
            << action_min << ',' << action_max << ',' << target_min << ',' << target_max;
        for (float value : action) csv << ',' << value;
        for (float value : q_target) csv << ',' << value;
        csv << '\n';
      }
    }
    std::cout << "EXTERNAL_REF_DRY_RUN result=PASS";
    if (!log_path.empty()) std::cout << " log=" << log_path;
    std::cout << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "EXTERNAL_REF_DRY_RUN result=FAIL error=" << error.what() << '\n';
    return 1;
  }
}
