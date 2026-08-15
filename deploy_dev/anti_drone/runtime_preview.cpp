#include "anti_drone/config.hpp"
#include "anti_drone/diagnostic_pipeline.hpp"
#include "anti_drone/pnp_solver.hpp"
#include "anti_drone/traditional_detector.hpp"

#include <exception>
#include <iostream>
#include <string>

// Minimal startup skeleton for the future NUC runtime binary. It verifies only
// the construction chain (config -> detector -> pipeline config -> pipeline,
// plus optional PnP when calibration exists) and prints a startup summary. It
// deliberately does NOT open a camera, read images, enter a frame loop, connect
// communication, or produce control commands.
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: anti_drone_runtime_preview <config_yaml>\n";
        return 1;
    }

    const std::string config_path = argv[1];

    try {
        // ── 1. Load config through the production API ────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config:\n"
                      << config_path << '\n'
                      << error.what() << '\n';
            return 2;
        }

        // ── 2. Detector construction ──────────────────────────────────────
        // No detect() call: there is no frame input in this preview.
        hnu25::anti_drone::TraditionalTargetDetector detector(
            config.traditional_detector);

        // ── 3. Pipeline config via the adapter (no field-by-field copy) ──
        const auto pipeline_config =
            hnu25::anti_drone::makeDiagnosticPipelineConfig(config);

        // ── 4. Diagnostic pipeline construction ───────────────────────────
        hnu25::anti_drone::DiagnosticPipeline pipeline(pipeline_config);

        // ── 5. Optional PnP construction ──────────────────────────────────
        // Calibration absence is NOT an error: the runtime skeleton itself is
        // ready, only the real PnP data chain is not yet available. The solver
        // is validated for constructability inside this block and is not kept
        // beyond it, because the frame loop does not exist yet.
        if (config.calibration.has_value()) {
            hnu25::anti_drone::PnpSolver solver(config.calibration->pnp);
        }

        // ── 6. Startup summary ────────────────────────────────────────────
        std::cout << "=== Anti-Drone Runtime Preview ===\n\n";

        std::cout << "Config:\n  " << config_path << "\n\n";

        std::cout << "Detector:\n  READY\n\n";

        std::cout << "Tracker:\n";
        std::cout << "  min_detect_count: "
                  << config.tracker.min_detect_count << '\n';
        std::cout << "  max_missed_count: "
                  << config.tracker.max_missed_count << '\n';
        std::cout << "  max_dt_s: " << config.tracker.max_dt_s << '\n';
        std::cout << "  max_association_distance_m: "
                  << config.tracker.max_association_distance_m << '\n';
        std::cout << "  position_gain: "
                  << config.tracker.position_gain << '\n';
        std::cout << "  velocity_gain: "
                  << config.tracker.velocity_gain << '\n';
        std::cout << '\n';

        std::cout << "Prediction:\n";
        std::cout << "  horizon_s: " << config.prediction.horizon_s << '\n';
        std::cout << '\n';

        std::cout << "Vision compensation:\n";
        std::cout << "  yaw_offset_deg: "
                  << config.vision_compensation.yaw_offset_deg << '\n';
        std::cout << "  pitch_offset_deg: "
                  << config.vision_compensation.pitch_offset_deg << '\n';
        std::cout << "  x_offset_m: "
                  << config.vision_compensation.x_offset_m << '\n';
        std::cout << "  y_offset_m: "
                  << config.vision_compensation.y_offset_m << '\n';
        std::cout << "  z_offset_m: "
                  << config.vision_compensation.z_offset_m << '\n';
        std::cout << '\n';

        std::cout << "Calibration:\n";
        if (config.calibration.has_value()) {
            std::cout << "  configured: YES\n";
            std::cout << "  fx: "
                      << config.calibration->pnp.camera_matrix(0, 0) << '\n';
            std::cout << "  fy: "
                      << config.calibration->pnp.camera_matrix(1, 1) << '\n';
            std::cout << "  max_reprojection_error_px: "
                      << config.calibration->pnp.max_reprojection_error_px
                      << '\n';
        } else {
            std::cout << "  configured: NO\n";
        }
        std::cout << '\n';

        std::cout << "PnP:\n";
        if (config.calibration.has_value()) {
            std::cout << "  READY\n";
        } else {
            std::cout << "  NOT READY\n";
            std::cout << "  reason: camera calibration is not configured\n";
        }
        std::cout << '\n';

        std::cout << "Diagnostic pipeline:\n  READY\n\n";

        std::cout << "Runtime:\n";
        std::cout << "  camera: DISABLED\n";
        std::cout << "  frame loop: DISABLED\n";
        std::cout << "  communication: DISABLED\n\n";

        std::cout << "Startup preview completed successfully.\n";

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime preview startup failed: " << error.what()
                  << '\n';
        return 3;
    }
}
