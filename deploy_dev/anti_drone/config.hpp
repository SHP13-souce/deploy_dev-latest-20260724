#pragma once

#include "anti_drone/traditional_detector.hpp"

#include <filesystem>

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

// Aggregates all anti-drone configuration into a single loadable unit.
struct AntiDroneConfig {
    TraditionalDetectorConfig traditional_detector;
    VisionCompensationConfig vision_compensation;
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
