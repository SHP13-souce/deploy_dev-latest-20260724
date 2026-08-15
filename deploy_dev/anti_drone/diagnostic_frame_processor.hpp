#pragma once

#include "anti_drone/config.hpp"
#include "anti_drone/diagnostic_pipeline.hpp"
#include "anti_drone/pnp_solver.hpp"
#include "anti_drone/traditional_detector.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <optional>
#include <vector>

namespace hnu25::anti_drone {

// Structured result of processing a single frame through the reusable core.
struct DiagnosticFrameProcessorResult {
    // True when a real calibration (and therefore the diagnostic chain) is
    // available; false means detector-only.
    bool calibration_available = false;

    // True when DiagnosticPipeline::update() was actually called this frame.
    bool diagnostic_enabled = false;

    // Raw detector output.
    std::vector<TargetObservation> observations;

    // How many corners_valid observations were actually sent to PnpSolver.
    std::size_t pnp_attempt_count = 0;

    // Only valid PnP results, ready to feed the tracker as measurements.
    std::vector<PnpResult> measurements;

    // Structured diagnostic output (only meaningful when diagnostic_enabled).
    DiagnosticFrameResult diagnostic;
};

// Reusable per-frame core shared by the sequence preview, the future Hik
// runtime, and other frame sources. It only composes the existing modules; it
// never reimplements detection, PnP, tracking, prediction, or compensation.
class DiagnosticFrameProcessor {
public:
    explicit DiagnosticFrameProcessor(const AntiDroneConfig& config);

    DiagnosticFrameProcessorResult process(
        const cv::Mat& bgr_image,
        std::chrono::steady_clock::time_point timestamp);

    void reset() noexcept;

private:
    TraditionalTargetDetector detector_;

    std::optional<PnpSolver> pnp_solver_;
    std::optional<DiagnosticPipeline> diagnostic_pipeline_;
};

}  // namespace hnu25::anti_drone
