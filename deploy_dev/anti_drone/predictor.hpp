#pragma once

#include "anti_drone/tracker.hpp"

#include <opencv2/core.hpp>

namespace hnu25::anti_drone {

// Tuning for the 3D constant-velocity prediction diagnostics.
struct PredictionConfig {
    // How many seconds to extrapolate forward from the current tracker state.
    // This is purely a diagnostic horizon, not a ballistic flight time or a
    // fire delay.
    double horizon_s = 0.0;
};

// The predicted 3D position of the target, in raw gimbal-relative coordinates.
struct Prediction3D {
    bool valid = false;

    cv::Vec3d current_position_gimbal_m{0.0, 0.0, 0.0};
    cv::Vec3d velocity_gimbal_m_s{0.0, 0.0, 0.0};
    cv::Vec3d predicted_position_gimbal_m{0.0, 0.0, 0.0};

    double horizon_s = 0.0;
};

// Pure function: extrapolates a confirmed track forward by config.horizon_s
// using a constant-velocity model. Only TRACKING and TEMP_LOST states are
// allowed to produce a valid prediction; DETECTING and LOST return invalid.
// Never mutates the input track and never applies vision compensation.
Prediction3D predictTrack(
    const TrackEstimate3D& track,
    const PredictionConfig& config);

}  // namespace hnu25::anti_drone
