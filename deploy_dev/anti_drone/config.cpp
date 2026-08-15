#include "anti_drone/config.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace hnu25::anti_drone {

namespace {

void fail(const std::string& message) {
    throw std::runtime_error(message);
}

// Validates every field after loading. Invalid user input throws instead of
// being silently clamped: the detector's own defensive clamps are a separate
// concern for programmatically built Config objects.
void validateTraditionalDetectorConfig(
    const TraditionalDetectorConfig& config) {
    // ── White mask HSV ────────────────────────────────────────────────────
    if (config.white_saturation_max < 0 || config.white_saturation_max > 255) {
        fail("white_saturation_max must be within [0, 255]");
    }
    if (config.white_value_min < 0 || config.white_value_min > 255) {
        fail("white_value_min must be within [0, 255]");
    }

    // ── Red hue ranges ────────────────────────────────────────────────────
    const bool hue_ok =
        config.red_hue_low_1 >= 0 && config.red_hue_low_1 <= 179 &&
        config.red_hue_high_1 >= 0 && config.red_hue_high_1 <= 179 &&
        config.red_hue_low_2 >= 0 && config.red_hue_low_2 <= 179 &&
        config.red_hue_high_2 >= 0 && config.red_hue_high_2 <= 179;
    if (!hue_ok) {
        fail("red hue parameters must be within [0, 179]");
    }
    if (config.red_hue_low_1 > config.red_hue_high_1) {
        fail("red_hue_low_1 must be <= red_hue_high_1");
    }
    if (config.red_hue_low_2 > config.red_hue_high_2) {
        fail("red_hue_low_2 must be <= red_hue_high_2");
    }
    if (config.red_saturation_min < 0 || config.red_saturation_min > 255) {
        fail("red_saturation_min must be within [0, 255]");
    }
    if (config.red_value_min < 0 || config.red_value_min > 255) {
        fail("red_value_min must be within [0, 255]");
    }

    // ── Area ratios ───────────────────────────────────────────────────────
    if (!std::isfinite(config.min_candidate_area_ratio) ||
        !std::isfinite(config.max_candidate_area_ratio)) {
        fail("candidate area ratios must be finite");
    }
    if (config.min_candidate_area_ratio < 0.0) {
        fail("min_candidate_area_ratio must be >= 0");
    }
    if (config.min_candidate_area_ratio > config.max_candidate_area_ratio) {
        fail("min_candidate_area_ratio must be <= max_candidate_area_ratio");
    }
    if (config.max_candidate_area_ratio > 1.0) {
        fail("max_candidate_area_ratio must be <= 1");
    }

    if (!std::isfinite(config.min_red_area_ratio)) {
        fail("min_red_area_ratio must be finite");
    }
    if (config.min_red_area_ratio < 0.0 || config.min_red_area_ratio > 1.0) {
        fail("min_red_area_ratio must be within [0, 1]");
    }

    // ── Geometry ──────────────────────────────────────────────────────────
    if (!std::isfinite(config.min_aspect_ratio) ||
        !std::isfinite(config.max_aspect_ratio)) {
        fail("aspect ratios must be finite");
    }
    if (config.min_aspect_ratio <= 0.0) {
        fail("min_aspect_ratio must be > 0");
    }
    if (config.max_aspect_ratio <= 0.0) {
        fail("max_aspect_ratio must be > 0");
    }
    if (config.min_aspect_ratio > config.max_aspect_ratio) {
        fail("min_aspect_ratio must be <= max_aspect_ratio");
    }

    if (!std::isfinite(config.min_rectangularity)) {
        fail("min_rectangularity must be finite");
    }
    if (config.min_rectangularity < 0.0 || config.min_rectangularity > 1.0) {
        fail("min_rectangularity must be within [0, 1]");
    }

    if (!std::isfinite(config.polygon_epsilon_ratio)) {
        fail("polygon_epsilon_ratio must be finite");
    }
    if (config.polygon_epsilon_ratio <= 0.0 ||
        config.polygon_epsilon_ratio > 1.0) {
        fail("polygon_epsilon_ratio must be within (0, 1]");
    }

    // ── Bullseye offset ───────────────────────────────────────────────────
    if (!std::isfinite(config.max_bullseye_offset_ratio)) {
        fail("max_bullseye_offset_ratio must be finite");
    }
    if (config.max_bullseye_offset_ratio < 0.0 ||
        config.max_bullseye_offset_ratio > 1.0) {
        fail("max_bullseye_offset_ratio must be within [0, 1]");
    }

    // ── Morphology ────────────────────────────────────────────────────────
    if (config.morphology_kernel_size < 0) {
        fail("morphology_kernel_size must be >= 0");
    }
    if (config.morphology_open_iterations < 0) {
        fail("morphology_open_iterations must be >= 0");
    }
    if (config.morphology_close_iterations < 0) {
        fail("morphology_close_iterations must be >= 0");
    }

    // ── Weights ───────────────────────────────────────────────────────────
    if (!std::isfinite(config.geometry_weight) ||
        config.geometry_weight < 0.0F) {
        fail("geometry_weight must be finite and >= 0");
    }
    if (!std::isfinite(config.color_weight) ||
        config.color_weight < 0.0F) {
        fail("color_weight must be finite and >= 0");
    }

    // ── Score / NMS ───────────────────────────────────────────────────────
    if (!std::isfinite(config.min_cv_score) ||
        config.min_cv_score < 0.0F || config.min_cv_score > 1.0F) {
        fail("min_cv_score must be finite and within [0, 1]");
    }
    if (!std::isfinite(config.nms_iou_threshold) ||
        config.nms_iou_threshold < 0.0F || config.nms_iou_threshold > 1.0F) {
        fail("nms_iou_threshold must be finite and within [0, 1]");
    }
}

// Compensation values only need to be finite at this stage: on-site mechanical
// mounting and axis conventions are not yet measured, so no magnitude bounds
// are enforced yet.
void validateVisionCompensationConfig(
    const VisionCompensationConfig& config) {
    if (!std::isfinite(config.yaw_offset_deg)) {
        fail("yaw_offset_deg must be finite");
    }
    if (!std::isfinite(config.pitch_offset_deg)) {
        fail("pitch_offset_deg must be finite");
    }
    if (!std::isfinite(config.x_offset_m)) {
        fail("x_offset_m must be finite");
    }
    if (!std::isfinite(config.y_offset_m)) {
        fail("y_offset_m must be finite");
    }
    if (!std::isfinite(config.z_offset_m)) {
        fail("z_offset_m must be finite");
    }
}

// Tracker tuning must satisfy the same constraints TargetTracker3D enforces in
// its constructor. Loader-side validation throws instead of silently clamping:
// a YAML typo should fail fast at startup rather than run with a wrong value.
void validateTrackerConfig(const TrackerConfig& config) {
    if (config.min_detect_count < 1) {
        fail("min_detect_count must be >= 1");
    }
    if (config.max_missed_count < 0) {
        fail("max_missed_count must be >= 0");
    }
    if (!std::isfinite(config.max_dt_s) || !(config.max_dt_s > 0.0)) {
        fail("max_dt_s must be finite and > 0");
    }
    if (!std::isfinite(config.max_association_distance_m) ||
        !(config.max_association_distance_m > 0.0)) {
        fail("max_association_distance_m must be finite and > 0");
    }
    if (!std::isfinite(config.position_gain) ||
        !(config.position_gain > 0.0) || config.position_gain > 1.0) {
        fail("position_gain must be finite and within (0, 1]");
    }
    if (!std::isfinite(config.velocity_gain) || config.velocity_gain < 0.0 ||
        config.velocity_gain > 1.0) {
        fail("velocity_gain must be finite and within [0, 1]");
    }
}

// Prediction horizon has no meaningful upper bound yet: real system latency has
// not been measured, so no magic cap is enforced. Only finiteness and >= 0.
void validatePredictionConfig(const PredictionConfig& config) {
    if (!std::isfinite(config.horizon_s) || config.horizon_s < 0.0) {
        fail("horizon_s must be finite and >= 0");
    }
}

// Camera runtime values are plain, conservative checks with no magic upper
// bounds until real hardware behaviour is measured on site.
void validateCameraConfig(const RuntimeCameraConfig& config) {
    if (!std::isfinite(config.exposure) || !(config.exposure > 0.0)) {
        fail("camera.exposure must be finite and > 0");
    }
    if (!std::isfinite(config.gain) || config.gain < 0.0) {
        fail("camera.gain must be finite and >= 0");
    }
    if (!std::isfinite(config.frame_rate) || !(config.frame_rate > 0.0)) {
        fail("camera.frame_rate must be finite and > 0");
    }
    if (config.frame_timeout_ms <= 0) {
        fail("camera.frame_timeout_ms must be > 0");
    }
    if (config.max_consecutive_timeouts < 1) {
        fail("camera.max_consecutive_timeouts must be >= 1");
    }
}

void validateRuntimeConfig(const AntiDroneRuntimeConfig& config) {
    if (config.log_every_n_frames < 1) {
        fail("runtime.log_every_n_frames must be >= 1");
    }
}

void validateGimbalConfig(const GimbalConfig& config) {
    if (!std::isfinite(config.speed_filter_alpha) ||
        config.speed_filter_alpha < 0.0 || config.speed_filter_alpha > 1.0) {
        fail("gimbal.speed_filter_alpha must be finite and within [0, 1]");
    }
}

// The serial transport delegates baud-rate handling to hnu25::SerialPort,
// which supports exactly these five rates. Rejecting anything else here fails
// fast at startup instead of at first open.
void validateTelemetryConfig(const TelemetryTransportConfig& config) {
    switch (config.baud_rate) {
        case 9600:
        case 19200:
        case 38400:
        case 57600:
        case 115200:
            break;
        default:
            fail("telemetry.baud_rate must be one of "
                 "9600, 19200, 38400, 57600, 115200");
    }
    if (config.max_consecutive_failures < 1) {
        fail("telemetry.max_consecutive_failures must be >= 1");
    }
    if (telemetryTransportModeIsSerial(config.mode) && config.device.empty()) {
        fail("telemetry.device must be non-empty for a serial transport mode");
    }
}

TelemetryTransportMode telemetryTransportModeFromString(
    const std::string& value) {
    if (value == "loopback") {
        return TelemetryTransportMode::LOOPBACK;
    }
    if (value == "serial_diagnostic") {
        return TelemetryTransportMode::SERIAL_DIAGNOSTIC;
    }
    if (value == "serial") {
        return TelemetryTransportMode::SERIAL;
    }
    throw std::runtime_error(
        "telemetry.mode must be \"loopback\", \"serial_diagnostic\", or "
        "\"serial\"");
}

// Reads a root-level 3x3 matrix expressed as a row-major YAML sequence of
// exactly 9 finite doubles. This mirrors the original project's Eigen::RowMajor
// semantics: values[0..2] are row 0, values[3..5] row 1, values[6..8] row 2.
// No auto-correction is performed; malformed input throws std::runtime_error.
cv::Matx33d matrix3x3FromYaml(const YAML::Node& root, const char* key) {
    const std::vector<double> values =
        root[key].as<std::vector<double>>();
    if (values.size() != 9) {
        throw std::runtime_error(std::string(key) +
                                 " must have exactly 9 elements");
    }
    for (const double v : values) {
        if (!std::isfinite(v)) {
            throw std::runtime_error(std::string(key) + " must be finite");
        }
    }
    cv::Matx33d m;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            m(r, c) = values[r * 3 + c];
        }
    }
    return m;
}

// Loads the optional root-level calibration section. Uses the original
// project's root-level keys (camera_matrix, distort_coeffs, R_camera2gimbal,
// t_camera2gimbal, R_gimbal2imubody, max_reprojection_error_px) with an
// all-or-nothing rule: if none of the six keys appear, calibration is nullopt
// and loading succeeds; if any appear, the five core keys must all be present.
std::optional<CalibrationConfig> loadCalibrationConfig(
    const YAML::Node& root) {
    const bool has_camera_matrix = static_cast<bool>(root["camera_matrix"]);
    const bool has_distort_coeffs = static_cast<bool>(root["distort_coeffs"]);
    const bool has_R_camera2gimbal =
        static_cast<bool>(root["R_camera2gimbal"]);
    const bool has_t_camera2gimbal =
        static_cast<bool>(root["t_camera2gimbal"]);
    const bool has_R_gimbal2imubody =
        static_cast<bool>(root["R_gimbal2imubody"]);
    const bool has_max_reprojection =
        static_cast<bool>(root["max_reprojection_error_px"]);

    const bool any = has_camera_matrix || has_distort_coeffs ||
                     has_R_camera2gimbal || has_t_camera2gimbal ||
                     has_R_gimbal2imubody || has_max_reprojection;
    if (!any) {
        return std::nullopt;
    }

    // Once the user begins providing calibration, the full core set is
    // required: partial calibration must not silently enter the system.
    if (!(has_camera_matrix && has_distort_coeffs && has_R_camera2gimbal &&
          has_t_camera2gimbal && has_R_gimbal2imubody)) {
        throw std::runtime_error(
            "incomplete calibration: camera_matrix, distort_coeffs, "
            "R_camera2gimbal, t_camera2gimbal and R_gimbal2imubody are all "
            "required");
    }

    CalibrationConfig calib;

    calib.pnp.camera_matrix = matrix3x3FromYaml(root, "camera_matrix");
    if (!(calib.pnp.camera_matrix(0, 0) > 0.0)) {
        throw std::runtime_error("camera_matrix fx must be > 0");
    }
    if (!(calib.pnp.camera_matrix(1, 1) > 0.0)) {
        throw std::runtime_error("camera_matrix fy must be > 0");
    }

    std::vector<double> distort =
        root["distort_coeffs"].as<std::vector<double>>();
    if (distort.empty()) {
        throw std::runtime_error("distort_coeffs must not be empty");
    }
    for (const double d : distort) {
        if (!std::isfinite(d)) {
            throw std::runtime_error("distort_coeffs must be finite");
        }
    }
    calib.pnp.distort_coeffs = std::move(distort);

    calib.pnp.R_camera2gimbal = matrix3x3FromYaml(root, "R_camera2gimbal");

    const std::vector<double> t =
        root["t_camera2gimbal"].as<std::vector<double>>();
    if (t.size() != 3) {
        throw std::runtime_error(
            "t_camera2gimbal must have exactly 3 elements");
    }
    for (const double v : t) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("t_camera2gimbal must be finite");
        }
    }
    calib.pnp.t_camera2gimbal = cv::Vec3d(t[0], t[1], t[2]);

    calib.R_gimbal2imubody = matrix3x3FromYaml(root, "R_gimbal2imubody");

    // max_reprojection_error_px is optional and defaults to the struct's 5.0.
    if (has_max_reprojection) {
        const double max_err =
            root["max_reprojection_error_px"].as<double>();
        if (!std::isfinite(max_err) || !(max_err > 0.0)) {
            throw std::runtime_error(
                "max_reprojection_error_px must be finite and > 0");
        }
        calib.pnp.max_reprojection_error_px = max_err;
    }

    return calib;
}

}  // namespace

const char* telemetryTransportModeName(TelemetryTransportMode mode) noexcept {
    switch (mode) {
        case TelemetryTransportMode::LOOPBACK:
            return "LOOPBACK";
        case TelemetryTransportMode::SERIAL_DIAGNOSTIC:
            return "SERIAL_DIAGNOSTIC";
        case TelemetryTransportMode::SERIAL:
            return "SERIAL";
    }
    return "UNKNOWN";
}

bool telemetryTransportModeIsSerial(TelemetryTransportMode mode) noexcept {
    return mode == TelemetryTransportMode::SERIAL_DIAGNOSTIC ||
           mode == TelemetryTransportMode::SERIAL;
}

AntiDroneConfig loadAntiDroneConfig(
    const std::filesystem::path& path) {
    const YAML::Node root = YAML::LoadFile(path.string());

    AntiDroneConfig config;

    // ── traditional_detector (required) ──────────────────────────────────
    const YAML::Node node = root["traditional_detector"];
    if (!node || !node.IsMap()) {
        throw std::runtime_error(
            "traditional_detector configuration section is required");
    }

    TraditionalDetectorConfig& td = config.traditional_detector;

    // ── White target board ───────────────────────────────────────────────
    td.white_saturation_max =
        node["white_saturation_max"].as<int>(td.white_saturation_max);
    td.white_value_min =
        node["white_value_min"].as<int>(td.white_value_min);

    // ── Red bullseye ─────────────────────────────────────────────────────
    td.red_hue_low_1 = node["red_hue_low_1"].as<int>(td.red_hue_low_1);
    td.red_hue_high_1 = node["red_hue_high_1"].as<int>(td.red_hue_high_1);
    td.red_hue_low_2 = node["red_hue_low_2"].as<int>(td.red_hue_low_2);
    td.red_hue_high_2 = node["red_hue_high_2"].as<int>(td.red_hue_high_2);
    td.red_saturation_min =
        node["red_saturation_min"].as<int>(td.red_saturation_min);
    td.red_value_min = node["red_value_min"].as<int>(td.red_value_min);

    // ── Candidate geometry ───────────────────────────────────────────────
    td.min_candidate_area_ratio =
        node["min_candidate_area_ratio"].as<double>(
            td.min_candidate_area_ratio);
    td.max_candidate_area_ratio =
        node["max_candidate_area_ratio"].as<double>(
            td.max_candidate_area_ratio);
    td.min_aspect_ratio =
        node["min_aspect_ratio"].as<double>(td.min_aspect_ratio);
    td.max_aspect_ratio =
        node["max_aspect_ratio"].as<double>(td.max_aspect_ratio);
    td.min_rectangularity =
        node["min_rectangularity"].as<double>(td.min_rectangularity);
    td.polygon_epsilon_ratio =
        node["polygon_epsilon_ratio"].as<double>(td.polygon_epsilon_ratio);

    // ── Bullseye validation ──────────────────────────────────────────────
    td.min_red_area_ratio =
        node["min_red_area_ratio"].as<double>(td.min_red_area_ratio);
    td.max_bullseye_offset_ratio =
        node["max_bullseye_offset_ratio"].as<double>(
            td.max_bullseye_offset_ratio);

    // ── Morphology ───────────────────────────────────────────────────────
    td.morphology_kernel_size =
        node["morphology_kernel_size"].as<int>(td.morphology_kernel_size);
    td.morphology_open_iterations =
        node["morphology_open_iterations"].as<int>(
            td.morphology_open_iterations);
    td.morphology_close_iterations =
        node["morphology_close_iterations"].as<int>(
            td.morphology_close_iterations);

    // ── Scoring ──────────────────────────────────────────────────────────
    td.geometry_weight = node["geometry_weight"].as<float>(td.geometry_weight);
    td.color_weight = node["color_weight"].as<float>(td.color_weight);
    td.min_cv_score = node["min_cv_score"].as<float>(td.min_cv_score);

    // ── Candidate deduplication ──────────────────────────────────────────
    td.nms_iou_threshold =
        node["nms_iou_threshold"].as<float>(td.nms_iou_threshold);

    validateTraditionalDetectorConfig(td);

    // ── tracker (optional) ────────────────────────────────────────────────
    if (const YAML::Node trk = root["tracker"]) {
        if (!trk.IsMap()) {
            throw std::runtime_error("tracker must be a mapping");
        }
        TrackerConfig& tracker = config.tracker;
        tracker.min_detect_count =
            trk["min_detect_count"].as<int>(tracker.min_detect_count);
        tracker.max_missed_count =
            trk["max_missed_count"].as<int>(tracker.max_missed_count);
        tracker.max_dt_s = trk["max_dt_s"].as<double>(tracker.max_dt_s);
        tracker.max_association_distance_m =
            trk["max_association_distance_m"].as<double>(
                tracker.max_association_distance_m);
        tracker.position_gain =
            trk["position_gain"].as<double>(tracker.position_gain);
        tracker.velocity_gain =
            trk["velocity_gain"].as<double>(tracker.velocity_gain);
    }

    validateTrackerConfig(config.tracker);

    // ── prediction (optional) ─────────────────────────────────────────────
    if (const YAML::Node pred = root["prediction"]) {
        if (!pred.IsMap()) {
            throw std::runtime_error("prediction must be a mapping");
        }
        PredictionConfig& prediction = config.prediction;
        prediction.horizon_s =
            pred["horizon_s"].as<double>(prediction.horizon_s);
    }

    validatePredictionConfig(config.prediction);

    // ── calibration (optional) ───────────────────────────────────────────
    config.calibration = loadCalibrationConfig(root);

    // ── vision_compensation (optional) ───────────────────────────────────
    if (const YAML::Node comp = root["vision_compensation"]) {
        if (!comp.IsMap()) {
            throw std::runtime_error("vision_compensation must be a mapping");
        }
        VisionCompensationConfig& vc = config.vision_compensation;
        vc.yaw_offset_deg =
            comp["yaw_offset_deg"].as<double>(vc.yaw_offset_deg);
        vc.pitch_offset_deg =
            comp["pitch_offset_deg"].as<double>(vc.pitch_offset_deg);
        vc.x_offset_m = comp["x_offset_m"].as<double>(vc.x_offset_m);
        vc.y_offset_m = comp["y_offset_m"].as<double>(vc.y_offset_m);
        vc.z_offset_m = comp["z_offset_m"].as<double>(vc.z_offset_m);
    }

    validateVisionCompensationConfig(config.vision_compensation);

    // ── camera (optional) ─────────────────────────────────────────────────
    if (const YAML::Node cam = root["camera"]) {
        if (!cam.IsMap()) {
            throw std::runtime_error("camera must be a mapping");
        }
        RuntimeCameraConfig& camera = config.camera;
        camera.serial_number =
            cam["serial_number"].as<std::string>(camera.serial_number);
        camera.exposure = cam["exposure"].as<double>(camera.exposure);
        camera.gain = cam["gain"].as<double>(camera.gain);
        camera.frame_rate = cam["frame_rate"].as<double>(camera.frame_rate);
        camera.frame_timeout_ms =
            cam["frame_timeout_ms"].as<int>(camera.frame_timeout_ms);
        camera.max_consecutive_timeouts =
            cam["max_consecutive_timeouts"].as<int>(
                camera.max_consecutive_timeouts);
    }

    validateCameraConfig(config.camera);

    // ── runtime (optional) ────────────────────────────────────────────────
    if (const YAML::Node rt = root["runtime"]) {
        if (!rt.IsMap()) {
            throw std::runtime_error("runtime must be a mapping");
        }
        config.runtime.log_every_n_frames =
            rt["log_every_n_frames"].as<int>(
                config.runtime.log_every_n_frames);
    }

    validateRuntimeConfig(config.runtime);

    // ── telemetry (optional) ──────────────────────────────────────────────
    if (const YAML::Node tel = root["telemetry"]) {
        if (!tel.IsMap()) {
            throw std::runtime_error("telemetry must be a mapping");
        }
        TelemetryTransportConfig& telemetry = config.telemetry;
        if (const YAML::Node mode = tel["mode"]) {
            telemetry.mode =
                telemetryTransportModeFromString(mode.as<std::string>());
        }
        telemetry.device =
            tel["device"].as<std::string>(telemetry.device);
        telemetry.baud_rate =
            tel["baud_rate"].as<int>(telemetry.baud_rate);
        telemetry.max_consecutive_failures =
            tel["max_consecutive_failures"].as<int>(
                telemetry.max_consecutive_failures);
        telemetry.flush_after_write =
            tel["flush_after_write"].as<bool>(telemetry.flush_after_write);
    }

    validateTelemetryConfig(config.telemetry);

    // ── gimbal (optional) ─────────────────────────────────────────────────
    if (const YAML::Node gim = root["gimbal"]) {
        if (!gim.IsMap()) {
            throw std::runtime_error("gimbal must be a mapping");
        }
        GimbalConfig& gimbal = config.gimbal;
        gimbal.enable = gim["enable"].as<bool>(gimbal.enable);
        gimbal.send_speed = gim["send_speed"].as<bool>(gimbal.send_speed);
        gimbal.speed_filter_alpha =
            gim["speed_filter_alpha"].as<double>(gimbal.speed_filter_alpha);
    }

    validateGimbalConfig(config.gimbal);

    return config;
}

TraditionalDetectorConfig loadTraditionalDetectorConfig(
    const std::filesystem::path& path) {
    return loadAntiDroneConfig(path).traditional_detector;
}

}  // namespace hnu25::anti_drone
