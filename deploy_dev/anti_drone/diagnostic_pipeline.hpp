#pragma once

#include "anti_drone/config.hpp"
#include "anti_drone/pnp_solver.hpp"
#include "anti_drone/predictor.hpp"
#include "anti_drone/tracker.hpp"
#include "anti_drone/vision_solution.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <vector>

namespace hnu25::anti_drone {

// Configuration for the synthetic end-to-end diagnostic pipeline. It composes
// the already-validated building blocks; no serial/fire/control/ballistics
// parameters live here.
struct DiagnosticPipelineConfig {
    TrackerConfig tracker;
    PredictionConfig prediction;
    VisionCompensationConfig compensation;
};

// Converts the already-loaded global anti-drone configuration into the subset
// consumed by DiagnosticPipeline. It performs no re-validation and does not use
// detector or calibration configuration.
DiagnosticPipelineConfig makeDiagnosticPipelineConfig(
    const AntiDroneConfig& config);

// Per-frame diagnostic output. The raw and predicted positions stay untouched
// by compensation; only compensated_position and the predicted pointing
// direction come from VisionSolution.
struct DiagnosticFrameResult {
    bool track_available = false;
    bool prediction_valid = false;
    bool solution_valid = false;

    TrackState track_state = TrackState::LOST;

    bool measurement_updated = false;

    int consecutive_detects = 0;
    int missed_count = 0;

    cv::Vec3d tracked_position_gimbal_m{0.0, 0.0, 0.0};
    cv::Vec3d velocity_gimbal_m_s{0.0, 0.0, 0.0};
    cv::Vec3d predicted_position_gimbal_m{0.0, 0.0, 0.0};
    cv::Vec3d compensated_position_gimbal_m{0.0, 0.0, 0.0};

    double prediction_horizon_s = 0.0;

    // Diagnostic pointing direction from the compensated predicted position.
    // These are not command / aim / fire outputs.
    double predicted_yaw_rad = 0.0;
    double predicted_pitch_rad = 0.0;
};

// Orchestration / composition layer over the existing PnP -> tracker ->
// prediction -> solution chain. It only wires the modules together; it does
// not reimplement any of their math.
class DiagnosticPipeline {
public:
    explicit DiagnosticPipeline(DiagnosticPipelineConfig config = {});

    DiagnosticFrameResult update(
        const std::vector<PnpResult>& measurements,
        std::chrono::steady_clock::time_point timestamp);

    void reset() noexcept;

private:
    DiagnosticPipelineConfig config_;
    TargetTracker3D tracker_;
};

}  // namespace hnu25::anti_drone
