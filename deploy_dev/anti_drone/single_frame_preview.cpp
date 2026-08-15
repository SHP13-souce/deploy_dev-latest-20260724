#include "anti_drone/config.hpp"
#include "anti_drone/pnp_solver.hpp"
#include "anti_drone/traditional_detector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void printPoint2f(const char* name, const cv::Point2f& p) {
    std::cout << "  " << name << ": (" << std::fixed << std::setprecision(2)
              << p.x << ", " << p.y << ")\n";
}

void printVec3d(const char* name, const cv::Vec3d& v) {
    std::cout << "    " << name << ": (" << std::fixed << std::setprecision(3)
              << v[0] << ", " << v[1] << ", " << v[2] << ")\n";
}

}  // namespace

// Offline single-frame visual diagnostics entry point: image -> detection ->
// optional PnP -> console output. No camera, no communication, no control
// commands, no automatic execution, no tracking/prediction/CSV.
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: anti_drone_single_frame_preview "
                     "<image_path> <config_yaml>\n";
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string config_path = argv[2];

    try {
        // ── Load config through the production API ────────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config: " << config_path
                      << ": " << error.what() << '\n';
            return 2;
        }

        // ── Read image ────────────────────────────────────────────────────
        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "Failed to read input image: " << image_path << '\n';
            return 3;
        }
        // imread with IMREAD_COLOR guarantees a non-empty result is CV_8UC3.

        // ── Detect ────────────────────────────────────────────────────────
        hnu25::anti_drone::TraditionalTargetDetector detector(
            config.traditional_detector);
        const auto observations = detector.detect(image);

        // ── Image / detection summary ─────────────────────────────────────
        std::cout << "Image:\n";
        std::cout << "  path: " << image_path << '\n';
        std::cout << "  width: " << image.cols << '\n';
        std::cout << "  height: " << image.rows << '\n';
        std::cout << "Detections:\n";
        std::cout << "  count: " << observations.size() << '\n';

        // ── Per-detection console output ──────────────────────────────────
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const auto& obs = observations[i];
            std::cout << "Target " << i << '\n';

            std::cout << "  cv_score: " << std::fixed << std::setprecision(2)
                      << obs.cv_score << '\n';

            printPoint2f("center", obs.center);

            std::cout << "  corners_valid: " << std::boolalpha
                      << obs.corners_valid << '\n';
            std::cout << "  bullseye_valid: " << obs.bullseye_valid << '\n';

            if (obs.corners_valid) {
                printPoint2f("TL", obs.corners[0]);
                printPoint2f("TR", obs.corners[1]);
                printPoint2f("BR", obs.corners[2]);
                printPoint2f("BL", obs.corners[3]);
            }
        }

        // ── Optional PnP ──────────────────────────────────────────────────
        // Calibration absence is NOT an error. Never fabricate calibration
        // with a default PnpSolverConfig.
        if (!config.calibration.has_value()) {
            std::cout << "PnP:\n";
            std::cout << "  SKIPPED\n";
            std::cout << "  reason: camera calibration is not configured\n";
        } else {
            hnu25::anti_drone::PnpSolver solver(config.calibration->pnp);
            std::cout << "PnP:\n";
            for (std::size_t i = 0; i < observations.size(); ++i) {
                const auto& obs = observations[i];
                if (!obs.corners_valid) {
                    continue;
                }
                const auto result = solver.solve(obs);
                std::cout << "  Target " << i << ":\n";
                std::cout << "    valid: " << std::boolalpha << result.valid
                          << '\n';
                printVec3d("xyz_camera", result.xyz_camera);
                printVec3d("xyz_gimbal", result.xyz_gimbal);
                std::cout << "    reprojection_error_px: " << std::fixed
                          << std::setprecision(3)
                          << result.reprojection_error_px << '\n';
            }
        }

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Single-frame preview failed: " << error.what() << '\n';
        return 4;
    }
}
