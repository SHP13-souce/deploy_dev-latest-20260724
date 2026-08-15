#include "anti_drone/diagnostic_pipeline.hpp"

#include <cmath>
#include <stdexcept>

namespace hnu25::anti_drone {

DiagnosticPipelineConfig makeDiagnosticPipelineConfig(
    const AntiDroneConfig& config) {
    DiagnosticPipelineConfig result;
    result.tracker = config.tracker;
    result.prediction = config.prediction;
    result.compensation = config.vision_compensation;
    return result;
}

DiagnosticPipeline::DiagnosticPipeline(DiagnosticPipelineConfig config)
    : config_(config), tracker_(config.tracker) {
    // The tracker validates its own knobs in its constructor. Validate the
    // layers this pipeline owns (or passes straight through) here.
    if (!std::isfinite(config_.prediction.horizon_s) ||
        config_.prediction.horizon_s < 0.0) {
        throw std::invalid_argument(
            "prediction.horizon_s must be finite and >= 0");
    }
    if (!std::isfinite(config_.compensation.yaw_offset_deg) ||
        !std::isfinite(config_.compensation.pitch_offset_deg) ||
        !std::isfinite(config_.compensation.x_offset_m) ||
        !std::isfinite(config_.compensation.y_offset_m) ||
        !std::isfinite(config_.compensation.z_offset_m)) {
        throw std::invalid_argument("compensation values must be finite");
    }
}

DiagnosticFrameResult DiagnosticPipeline::update(
    const std::vector<PnpResult>& measurements,
    std::chrono::steady_clock::time_point timestamp) {
    DiagnosticFrameResult result;

    const std::optional<TrackEstimate3D> track =
        tracker_.update(measurements, timestamp);
    if (!track.has_value()) {
        // No track (LOST, or a DETECTING frame lost on prediction alone).
        result.track_state = tracker_.state();
        return result;
    }

    result.track_available = true;
    result.track_state = track->state;
    result.measurement_updated = track->measurement_updated;
    result.consecutive_detects = track->consecutive_detects;
    result.missed_count = track->missed_count;
    result.tracked_position_gimbal_m = track->position_gimbal_m;
    result.velocity_gimbal_m_s = track->velocity_gimbal_m_s;

    const Prediction3D prediction = predictTrack(*track, config_.prediction);
    if (!prediction.valid) {
        // DETECTING tracks are intentionally not extrapolated.
        return result;
    }

    result.prediction_valid = true;
    result.predicted_position_gimbal_m = prediction.predicted_position_gimbal_m;
    result.prediction_horizon_s = prediction.horizon_s;

    const VisionSolution solution = makeVisionSolution(
        prediction.predicted_position_gimbal_m, config_.compensation);
    if (!solution.valid) {
        return result;
    }

    result.solution_valid = true;
    result.compensated_position_gimbal_m = solution.xyz_gimbal_compensated;
    result.predicted_yaw_rad = solution.yaw_compensated_rad;
    result.predicted_pitch_rad = solution.pitch_compensated_rad;

    return result;
}

void DiagnosticPipeline::reset() noexcept {
    tracker_.reset();
}

}  // namespace hnu25::anti_drone
