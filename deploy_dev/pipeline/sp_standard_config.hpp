#pragma once

#include "communication/sp_protocol.hpp"
#include "aim/fire_gate.hpp"
#include "debug_web/debug_web_publisher.hpp"
#include "detection/detector.hpp"
#include "planner/mpc_planner.hpp"
#include "sp_core/sp_core.hpp"
#include "telemetry/telemetry.hpp"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <Eigen/Geometry>

namespace hnu25::standard {

struct SerialConfig {
    bool enabled = false;
    std::string device = sp::DEFAULT_DEVICE;
    int baud = 115200;
};

struct FeedbackPoseConfig {
    bool enabled = false;
    double yaw_sign = 1.0;
    double pitch_sign = 1.0;
    double roll_sign = 1.0;
    std::string order = "ZYX";
    int time_offset_ms = 0;
    int max_age_ms = 50;
    std::optional<std::uint8_t> expected_address;
    bool reject_echo = true;
    int echo_window_ms = 20;
};

struct CameraConfig {
    std::string type = "opencv";
    int camera_id = 0;
    int width = 1280;
    int height = 1024;
    std::string serial_number;
    double exposure = 3000.0;
    double gain = 0.0;
    double frame_rate = 100.0;
};

struct Config {
    CameraConfig camera;
    DetectorConfig detector;
    double bullet_speed = 23.0;
    sp_core::AimerConfig aimer;
    int binary_threshold = 120;
    bool command_enabled = false;
    std::array<double, 4> imu_quaternion_wxyz{1.0, 0.0, 0.0, 0.0};
    bool configured_imu_fallback = false;
    SerialConfig serial;
    FeedbackPoseConfig feedback_pose;
    aim::FireGateConfig fire_gate;
    planner::Config planner;
    telemetry::Config telemetry;
    debug_web::Config web;
};

Config loadConfig(const std::filesystem::path& path);
Eigen::Quaterniond feedbackQuaternion(const sp::GimbalFeedback& feedback,
                                      const FeedbackPoseConfig& config);
std::optional<sp::GimbalCommand> mapCommand(const sp_core::AimResult& aim,
                                             bool command_enabled,
                                             const aim::FireDecision& fire);

}  // namespace hnu25::standard
