#pragma once

#include "anti_drone/pnp_solver.hpp"
#include "anti_drone/predictor.hpp"
#include "anti_drone/tracker.hpp"
#include "anti_drone/traditional_detector.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace hnu25::anti_drone {

// Residual compensation for the downstream PnP / gimbal-direction stage.
// These are NOT geometric calibration parameters; they are small manual
// adjustments applied on-site after real calibration.
//
// yaw_offset_deg / pitch_offset_deg:
//   Manual residual correction for the PnP / gimbal pointing direction,
//   in degrees.
//
// x/y/z_offset_m:
//   After the camera -> gimbal coordinate transform, a residual translation
//   correction applied to the gimbal-frame position, in meters.
//
// This stage only loads and stores these values; it does not apply them.
struct VisionCompensationConfig {
    double yaw_offset_deg = 0.0;
    double pitch_offset_deg = 0.0;

    double x_offset_m = 0.0;
    double y_offset_m = 0.0;
    double z_offset_m = 0.0;
};

// Camera / PnP calibration. The PnP solver only consumes the fields stored in
// PnpSolverConfig (camera_matrix, distort_coeffs, R_camera2gimbal,
// t_camera2gimbal, max_reprojection_error_px). R_gimbal2imubody is part of the
// original project's real extrinsic chain and is kept here for later stages,
// even though the PnP solver does not use it yet.
struct CalibrationConfig {
    PnpSolverConfig pnp;
    cv::Matx33d R_gimbal2imubody = cv::Matx33d::eye();
};

// Camera runtime settings for the production application. This is a plain
// value store only: the anti_drone library must NOT include
// camera/hik_frame_source.hpp. app_main.cpp maps this onto a
// hnu25::camera::HikConfig.
struct RuntimeCameraConfig {
    // Empty means "use the first camera the MVS SDK discovers".
    std::string serial_number;

    double exposure = 3000.0;
    double gain = 0.0;
    double frame_rate = 100.0;

    int frame_timeout_ms = 1000;
    int max_consecutive_timeouts = 5;
};

// Application behaviour for the production entry point.
struct AntiDroneRuntimeConfig {
    // Status lines are printed every N valid frames (and always on state
    // changes). Must be >= 1.
    int log_every_n_frames = 10;
};

// Aggregates all anti-drone configuration into a single loadable unit.
struct AntiDroneConfig {
    TraditionalDetectorConfig traditional_detector;

    // Runtime algorithm tuning for the 3D tracker and prediction stages. These
    // are independent of camera calibration and vision compensation.
    TrackerConfig tracker;
    PredictionConfig prediction;

    // Optional: real camera calibration may not be available yet. When the
    // YAML contains none of the calibration keys this stays nullopt, so the
    // existing detector / image_preview workflow keeps working unchanged.
    std::optional<CalibrationConfig> calibration;

    VisionCompensationConfig vision_compensation;

    // Production runtime camera + application behaviour. Loaded from the
    // optional "camera:" / "runtime:" YAML sections; the defaults keep older
    // config files compatible.
    RuntimeCameraConfig camera;
    AntiDroneRuntimeConfig runtime;
};

// Loads the full anti-drone configuration from a YAML file. The file must
// contain a top-level "traditional_detector:" mapping; the optional
// "vision_compensation:" mapping defaults to all-zero when absent. Invalid
// values throw std::runtime_error.
AntiDroneConfig loadAntiDroneConfig(const std::filesystem::path& path);

// Convenience wrapper that returns only the traditional detector portion.
// Kept for backward compatibility (e.g. image_preview).
TraditionalDetectorConfig loadTraditionalDetectorConfig(
    const std::filesystem::path& path);

}  // namespace hnu25::anti_drone
