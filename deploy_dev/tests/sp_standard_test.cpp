#include "pipeline/sp_standard_config.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    const auto path = std::filesystem::temp_directory_path() / "sp_standard_test.yaml";
    {
        std::ofstream output(path);
        output << "detector:\n"
                  "  backend: awakening\n"
                  "  model_path: model.onnx\n"
                  "  conf_threshold: 0.2\n"
                  "  performance_mode: throughput\n"
                  "camera:\n"
                  "  type: hik\n"
                  "  serial_number: test-serial\n"
                  "  exposure: 2500\n"
                  "  gain: 3.5\n"
                  "  frame_rate: 120\n"
                  "serial:\n"
                  "  device: /dev/test\n"
                  "feedback_pose:\n"
                  "  enabled: true\n"
                  "  yaw_sign: -1\n"
                  "  pitch_sign: 1\n"
                  "  roll_sign: -1\n"
                  "  order: ZYX\n"
                  "  time_offset_ms: -3\n"
                  "  max_age_ms: 40\n"
                  "  expected_address: 2\n"
                  "  reject_echo: true\n"
                  "  echo_window_ms: 15\n"
                  "imu_quaternion_wxyz: [2, 0, 0, 0]\n";
    }
    const auto config = hnu25::standard::loadConfig(path);
    std::filesystem::remove(path);
    require(!config.command_enabled, "commands default disabled");
    require(!config.fire_gate.enabled, "fire gate defaults disabled");
    require(!config.planner.enabled, "planner defaults disabled");
    require(!config.serial.enabled, "serial defaults disabled when omitted");
    require(!config.web.enabled, "web defaults disabled");
    require(config.detector.backend == "awakening", "detector backend parsed");
    require(config.detector.conf_threshold == 0.2F &&
                config.detector.performance_mode == "throughput",
            "detector options parsed");
    require(config.camera.type == "hik", "camera type parsed");
    require(config.camera.serial_number == "test-serial", "camera serial parsed");
    require(config.camera.exposure == 2500.0 && config.camera.gain == 3.5 &&
                config.camera.frame_rate == 120.0,
            "camera controls parsed");
    require(config.serial.device == "/dev/test", "serial device parsed");
    require(config.imu_quaternion_wxyz[0] == 1.0, "fallback quaternion normalized");
    require(config.feedback_pose.enabled && config.feedback_pose.expected_address == 2,
            "feedback pose and address parsed");
    require(config.feedback_pose.time_offset_ms == -3 && config.feedback_pose.max_age_ms == 40,
            "feedback pose timing parsed");

    hnu25::sp::GimbalFeedback feedback;
    feedback.yaw_deg = 90.0;
    feedback.pitch_deg = 30.0;
    feedback.roll_deg = 20.0;
    const Eigen::Quaterniond actual = hnu25::standard::feedbackQuaternion(feedback, config.feedback_pose);
    constexpr double degrees_to_radians = 0.01745329251994329577;
    const Eigen::Quaterniond expected =
        Eigen::AngleAxisd(-90.0 * degrees_to_radians, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(30.0 * degrees_to_radians, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(-20.0 * degrees_to_radians, Eigen::Vector3d::UnitX());
    require(std::abs(actual.dot(expected)) > 1.0 - 1e-12,
            "angle signs and ZYX quaternion composition");

    sp_core::AimResult aim_result;
    aim_result.valid = true;
    aim_result.yaw = 0.25;
    aim_result.pitch = -0.1;
    aim::FireDecision fire;
    fire.allowed = true;
    require(!hnu25::standard::mapCommand(aim_result, false, fire),
            "disabled command produces no mapping");
    fire.allowed = false;
    auto command = hnu25::standard::mapCommand(aim_result, true, fire);
    require(command && command->control && !command->fire, "command follows denied fire decision");
    fire.allowed = true;
    command = hnu25::standard::mapCommand(aim_result, true, fire);
    require(command && command->fire, "command follows allowed fire decision");
    aim_result.pitch = std::numeric_limits<double>::quiet_NaN();
    require(!hnu25::standard::mapCommand(aim_result, true, fire),
            "non-finite angle produces no command");
    return 0;
}
