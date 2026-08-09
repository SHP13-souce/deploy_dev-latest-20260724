#include "sp_standard_config.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>

namespace hnu25::standard {
namespace {

std::filesystem::path resolvePath(const std::filesystem::path& config_path,
                                  const std::string& value) {
    const std::filesystem::path path(value);
    return path.is_absolute() ? path : std::filesystem::absolute(config_path.parent_path() / path);
}

}  // namespace

Config loadConfig(const std::filesystem::path& path) {
    const YAML::Node yaml = YAML::LoadFile(path.string());
    Config config;
    config.camera.camera_id = yaml["camera_id"].as<int>(config.camera.camera_id);
    config.camera.width = yaml["width"].as<int>(config.camera.width);
    config.camera.height = yaml["height"].as<int>(config.camera.height);
    if (const YAML::Node camera = yaml["camera"]) {
        config.camera.type = camera["type"].as<std::string>(config.camera.type);
        config.camera.camera_id = camera["camera_id"].as<int>(config.camera.camera_id);
        config.camera.width = camera["width"].as<int>(config.camera.width);
        config.camera.height = camera["height"].as<int>(config.camera.height);
        config.camera.serial_number = camera["serial_number"].as<std::string>(config.camera.serial_number);
        config.camera.exposure = camera["exposure"].as<double>(config.camera.exposure);
        config.camera.gain = camera["gain"].as<double>(config.camera.gain);
        config.camera.frame_rate = camera["frame_rate"].as<double>(config.camera.frame_rate);
    }
    if (config.camera.type != "opencv" && config.camera.type != "hik")
        throw std::runtime_error("camera.type must be opencv or hik");
    if (config.camera.width <= 0 || config.camera.height <= 0)
        throw std::runtime_error("camera dimensions must be positive");
    if (config.camera.exposure < 0.0 || config.camera.gain < 0.0 || config.camera.frame_rate < 0.0)
        throw std::runtime_error("camera exposure, gain, and frame_rate must be non-negative");
    const YAML::Node detector = yaml["detector"];
    if (detector) {
        config.detector.backend = detector["backend"].as<std::string>(config.detector.backend);
        if (!detector["model_path"])
            throw std::runtime_error("detector.model_path is required");
        config.detector.model_path = resolvePath(
            path, detector["model_path"].as<std::string>()).string();
        if (detector["classifier_path"])
            config.detector.classifier_path = resolvePath(
                path, detector["classifier_path"].as<std::string>()).string();
        config.detector.conf_threshold = detector["conf_threshold"].as<float>(
            config.detector.backend == "awakening" ? 0.2F : config.detector.conf_threshold);
        config.detector.performance_mode = detector["performance_mode"].as<std::string>(
            config.detector.performance_mode);
    } else {
        if (!yaml["model_path"])
            throw std::runtime_error("detector.model_path is required");
        config.detector.model_path = resolvePath(path, yaml["model_path"].as<std::string>()).string();
        config.detector.conf_threshold = yaml["conf_threshold"].as<float>(
            config.detector.conf_threshold);
    }
    if (config.detector.backend != "hebei" && config.detector.backend != "awakening")
        throw std::runtime_error("detector.backend must be hebei or awakening");
    if (config.detector.performance_mode != "latency" &&
        config.detector.performance_mode != "throughput")
        throw std::runtime_error("detector.performance_mode must be latency or throughput");
    if (!(config.detector.conf_threshold >= 0.0F && config.detector.conf_threshold <= 1.0F))
        throw std::runtime_error("detector.conf_threshold must be in [0,1]");
    config.bullet_speed = yaml["bullet_speed"].as<double>(config.bullet_speed);
    config.aimer.yaw_offset = yaml["yaw_offset"].as<double>(config.aimer.yaw_offset);
    config.aimer.pitch_offset = yaml["pitch_offset"].as<double>(config.aimer.pitch_offset);
    config.aimer.coming_angle = yaml["comming_angle"].as<double>(config.aimer.coming_angle);
    config.aimer.leaving_angle = yaml["leaving_angle"].as<double>(config.aimer.leaving_angle);
    config.aimer.decision_speed = yaml["decision_speed"].as<double>(config.aimer.decision_speed);
    config.aimer.high_speed_delay = yaml["high_speed_delay_time"].as<double>(config.aimer.high_speed_delay);
    config.aimer.low_speed_delay = yaml["low_speed_delay_time"].as<double>(config.aimer.low_speed_delay);
    config.command_enabled = yaml["command_enabled"].as<bool>(false);
    if (const YAML::Node planner = yaml["planner"]) {
        config.planner.enabled = planner["enabled"].as<bool>(config.planner.enabled);
        config.planner.dt_s = planner["dt_s"].as<double>(config.planner.dt_s);
        config.planner.horizon_steps = planner["horizon_steps"].as<int>(config.planner.horizon_steps);
        config.planner.q_angle = planner["q_angle"].as<double>(config.planner.q_angle);
        config.planner.q_velocity = planner["q_velocity"].as<double>(config.planner.q_velocity);
        config.planner.r_acceleration = planner["r_acceleration"].as<double>(config.planner.r_acceleration);
        config.planner.max_yaw_acc_rad_s2 = planner["max_yaw_acc_rad_s2"].as<double>(config.planner.max_yaw_acc_rad_s2);
        config.planner.max_pitch_acc_rad_s2 = planner["max_pitch_acc_rad_s2"].as<double>(config.planner.max_pitch_acc_rad_s2);
        config.planner.feedback_timeout_s = planner["feedback_timeout_s"].as<double>(config.planner.feedback_timeout_s);
        config.planner.use_feedback_velocity = planner["use_feedback_velocity"].as<bool>(
            config.planner.use_feedback_velocity);
    }
    if (const YAML::Node gate = yaml["fire_gate"]) {
        config.fire_gate.enabled = gate["enabled"].as<bool>(config.fire_gate.enabled);
        config.fire_gate.max_measurement_age = gate["max_measurement_age"].as<double>(config.fire_gate.max_measurement_age);
        config.fire_gate.max_reprojection = gate["max_reprojection"].as<double>(config.fire_gate.max_reprojection);
        config.fire_gate.max_nis = gate["max_nis"].as<double>(config.fire_gate.max_nis);
        config.fire_gate.min_consecutive = gate["min_consecutive"].as<int>(config.fire_gate.min_consecutive);
        constexpr double degrees_to_radians = 0.01745329251994329577;
        config.fire_gate.min_yaw_tolerance = gate["min_yaw_tolerance_deg"].as<double>(
            config.fire_gate.min_yaw_tolerance / degrees_to_radians) * degrees_to_radians;
        config.fire_gate.min_pitch_tolerance = gate["min_pitch_tolerance_deg"].as<double>(
            config.fire_gate.min_pitch_tolerance / degrees_to_radians) * degrees_to_radians;
        config.fire_gate.small_armor_width = gate["small_armor_width"].as<double>(config.fire_gate.small_armor_width);
        config.fire_gate.large_armor_width = gate["large_armor_width"].as<double>(config.fire_gate.large_armor_width);
        config.fire_gate.armor_height = gate["armor_height"].as<double>(config.fire_gate.armor_height);
        config.fire_gate.armor_tolerance_ratio = gate["armor_tolerance_ratio"].as<double>(config.fire_gate.armor_tolerance_ratio);
    }

    if (const YAML::Node serial = yaml["serial"]) {
        config.serial.enabled = serial["enabled"].as<bool>(false);
        config.serial.device = serial["device"].as<std::string>(config.serial.device);
        config.serial.baud = serial["baud"].as<int>(config.serial.baud);
    }
    if (const YAML::Node pose = yaml["feedback_pose"]) {
        config.feedback_pose.enabled = pose["enabled"].as<bool>(config.feedback_pose.enabled);
        config.feedback_pose.yaw_sign = pose["yaw_sign"].as<double>(config.feedback_pose.yaw_sign);
        config.feedback_pose.pitch_sign = pose["pitch_sign"].as<double>(config.feedback_pose.pitch_sign);
        config.feedback_pose.roll_sign = pose["roll_sign"].as<double>(config.feedback_pose.roll_sign);
        config.feedback_pose.order = pose["order"].as<std::string>(config.feedback_pose.order);
        config.feedback_pose.time_offset_ms = pose["time_offset_ms"].as<int>(config.feedback_pose.time_offset_ms);
        config.feedback_pose.max_age_ms = pose["max_age_ms"].as<int>(config.feedback_pose.max_age_ms);
        config.feedback_pose.reject_echo = pose["reject_echo"].as<bool>(config.feedback_pose.reject_echo);
        config.feedback_pose.echo_window_ms = pose["echo_window_ms"].as<int>(config.feedback_pose.echo_window_ms);
        if (pose["expected_address"])
            config.feedback_pose.expected_address = pose["expected_address"].as<std::uint8_t>();
    }
    const auto& pose = config.feedback_pose;
    if (pose.order != "ZYX") throw std::runtime_error("feedback_pose.order only supports ZYX");
    if (!std::isfinite(pose.yaw_sign) || !std::isfinite(pose.pitch_sign) ||
        !std::isfinite(pose.roll_sign) || pose.max_age_ms < 0 || pose.echo_window_ms < 0)
        throw std::runtime_error("invalid feedback_pose configuration");
    if (const YAML::Node telemetry = yaml["telemetry"]) {
        config.telemetry.enabled = telemetry["enabled"].as<bool>(config.telemetry.enabled);
        config.telemetry.address = telemetry["address"].as<std::string>(config.telemetry.address);
        config.telemetry.port = telemetry["port"].as<std::uint16_t>(config.telemetry.port);
        config.telemetry.publish_hz = telemetry["publish_hz"].as<double>(config.telemetry.publish_hz);
    }
    if (const YAML::Node web = yaml["web"]) {
        config.web.enabled = web["enabled"].as<bool>(config.web.enabled);
        config.web.shm_name = web["shm_name"].as<std::string>(config.web.shm_name);
        config.web.quality = web["quality"].as<int>(config.web.quality);
        config.web.publish_hz = web["publish_hz"].as<double>(config.web.publish_hz);
        config.binary_threshold = web["binary_threshold"].as<int>(config.binary_threshold);
    }
    if (config.binary_threshold < 0 || config.binary_threshold > 255)
        throw std::runtime_error("web.binary_threshold must be in [0,255]");
    if (const YAML::Node quaternion = yaml["imu_quaternion_wxyz"]) {
        if (!quaternion.IsSequence() || quaternion.size() != 4)
            throw std::runtime_error("imu_quaternion_wxyz must contain [w, x, y, z]");
        double norm_squared = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            config.imu_quaternion_wxyz[i] = quaternion[i].as<double>();
            norm_squared += config.imu_quaternion_wxyz[i] * config.imu_quaternion_wxyz[i];
        }
        if (!std::isfinite(norm_squared) || norm_squared <= 0.0)
            throw std::runtime_error("imu_quaternion_wxyz must be finite and non-zero");
        const double norm = std::sqrt(norm_squared);
        for (double& value : config.imu_quaternion_wxyz) value /= norm;
        config.configured_imu_fallback = true;
    }
    return config;
}

Eigen::Quaterniond feedbackQuaternion(const sp::GimbalFeedback& feedback,
                                      const FeedbackPoseConfig& config) {
    if (config.order != "ZYX") throw std::invalid_argument("feedback pose order must be ZYX");
    constexpr double degrees_to_radians = 0.01745329251994329577;
    const double yaw = feedback.yaw_deg * config.yaw_sign * degrees_to_radians;
    const double pitch = feedback.pitch_deg * config.pitch_sign * degrees_to_radians;
    const double roll = feedback.roll_deg * config.roll_sign * degrees_to_radians;
    if (!std::isfinite(yaw) || !std::isfinite(pitch) || !std::isfinite(roll))
        throw std::invalid_argument("feedback angles must be finite");
    return (Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())).normalized();
}

std::optional<sp::GimbalCommand> mapCommand(const sp_core::AimResult& aim,
                                             bool command_enabled,
                                             const aim::FireDecision& fire) {
    if (!command_enabled || !std::isfinite(aim.yaw) || !std::isfinite(aim.pitch)) return std::nullopt;
    sp::GimbalCommand command;
    command.control = aim.valid;
    command.fire = fire.allowed;
    command.yaw_rad = aim.yaw;
    command.pitch_rad = aim.pitch;
    return command;
}

}  // namespace hnu25::standard
