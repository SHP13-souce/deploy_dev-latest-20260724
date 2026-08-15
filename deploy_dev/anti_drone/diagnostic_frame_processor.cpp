#include "anti_drone/diagnostic_frame_processor.hpp"

#include <stdexcept>

namespace hnu25::anti_drone {

DiagnosticFrameProcessor::DiagnosticFrameProcessor(
    const AntiDroneConfig& config)
    : detector_(config.traditional_detector) {
    if (config.calibration.has_value()) {
        pnp_solver_.emplace(config.calibration->pnp);
        diagnostic_pipeline_.emplace(
            makeDiagnosticPipelineConfig(config));
    }
}

DiagnosticFrameProcessorResult DiagnosticFrameProcessor::process(
    const cv::Mat& bgr_image,
    std::chrono::steady_clock::time_point timestamp) {
    if (bgr_image.empty()) {
        throw std::invalid_argument(
            "DiagnosticFrameProcessor received empty image");
    }

    DiagnosticFrameProcessorResult result;
    result.calibration_available = pnp_solver_.has_value();

    result.observations = detector_.detect(bgr_image);

    if (!result.calibration_available) {
        // Detector-only mode: no PnP / tracker / prediction / compensation.
        result.diagnostic_enabled = false;
        return result;
    }

    // Calibration mode: solve PnP for every valid-corner observation, keeping
    // only valid results as tracker measurements.
    for (const auto& observation : result.observations) {
        if (!observation.corners_valid) {
            continue;
        }
        ++result.pnp_attempt_count;
        const PnpResult pnp = pnp_solver_->solve(observation);
        if (pnp.valid) {
            result.measurements.push_back(pnp);
        }
    }

    // Always update the pipeline, even with empty measurements, so the tracker
    // advances its state machine (TEMP_LOST / LOST / missed_count).
    result.diagnostic =
        diagnostic_pipeline_->update(result.measurements, timestamp);
    result.diagnostic_enabled = true;

    return result;
}

void DiagnosticFrameProcessor::reset() noexcept {
    if (diagnostic_pipeline_.has_value()) {
        diagnostic_pipeline_->reset();
    }
}

}  // namespace hnu25::anti_drone
