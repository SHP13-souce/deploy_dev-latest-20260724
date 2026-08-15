#include "anti_drone/predictor.hpp"

#include <cmath>
#include <stdexcept>

namespace hnu25::anti_drone {

namespace {

bool finite(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

}  // namespace

Prediction3D predictTrack(
    const TrackEstimate3D& track,
    const PredictionConfig& config) {
    Prediction3D result;

    if (!std::isfinite(config.horizon_s) || config.horizon_s < 0.0) {
        throw std::invalid_argument("horizon_s must be finite and >= 0");
    }

    // Unconfirmed (DETECTING) and LOST tracks must not produce a prediction.
    if (track.state != TrackState::TRACKING &&
        track.state != TrackState::TEMP_LOST) {
        return result;
    }

    if (!finite(track.position_gimbal_m) ||
        !finite(track.velocity_gimbal_m_s)) {
        return result;
    }

    // Constant-velocity extrapolation, one axis at a time:
    //   p' = p + v * t
    const cv::Vec3d predicted =
        track.position_gimbal_m + track.velocity_gimbal_m_s * config.horizon_s;
    if (!finite(predicted)) {
        return result;
    }

    result.valid = true;
    result.current_position_gimbal_m = track.position_gimbal_m;
    result.velocity_gimbal_m_s = track.velocity_gimbal_m_s;
    result.predicted_position_gimbal_m = predicted;
    result.horizon_s = config.horizon_s;
    return result;
}

}  // namespace hnu25::anti_drone
