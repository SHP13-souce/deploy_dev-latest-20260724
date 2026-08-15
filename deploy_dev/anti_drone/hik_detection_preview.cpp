#include "anti_drone/config.hpp"
#include "anti_drone/traditional_detector.hpp"

#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/hik_frame_source.hpp"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>

// Read-only real-time Hik camera visual diagnostics entry point:
// HikFrameSource -> BGR frame -> TraditionalTargetDetector -> console output.
// No PnP, tracker, predictor, vision solution, communication, control, or
// serial commands.
int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cout << "Usage: anti_drone_hik_detection_preview "
                     "<config_yaml> [serial_number] [frame_count]\n";
        return 1;
    }

    const std::string config_path = argv[1];
    const std::string serial_number = (argc >= 3) ? argv[2] : "";

    int frame_count = 100;
    if (argc >= 4) {
        try {
            frame_count = std::stoi(argv[3]);
        } catch (const std::exception&) {
            std::cout << "Usage: anti_drone_hik_detection_preview "
                         "<config_yaml> [serial_number] [frame_count]\n";
            return 1;
        }
        if (frame_count <= 0) {
            std::cout << "Usage: anti_drone_hik_detection_preview "
                         "<config_yaml> [serial_number] [frame_count]\n";
            return 1;
        }
    }

#if HNU25_HAS_MVS
    try {
        // ── Load config through the production API ────────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config: " << config_path
                      << ": " << error.what() << '\n';
            return 3;
        }

        hnu25::anti_drone::TraditionalTargetDetector detector(
            config.traditional_detector);

        // ── Hik camera source (reuse the existing camera module) ─────────
        // Keep the default exposure/gain/frame_rate; real exposure values are
        // decided on-site later and do NOT belong in anti_drone.yaml yet.
        hnu25::camera::HikConfig camera_config;
        camera_config.serial_number = serial_number;

        hnu25::camera::HikFrameSource source(camera_config);
        try {
            source.start();
        } catch (const std::exception& error) {
            std::cerr << "Failed to start Hik camera:\n" << error.what()
                      << '\n';
            return 4;
        }

        // ── Frame acquisition loop ────────────────────────────────────────
        std::size_t received_frames = 0;
        std::size_t timeout_count = 0;
        std::size_t frames_with_detection = 0;
        int consecutive_timeouts = 0;

        while (received_frames < static_cast<std::size_t>(frame_count)) {
            hnu25::camera::Frame frame;
            const bool received = source.waitForFrame(
                frame, std::chrono::milliseconds(1000));
            if (!received) {
                std::cout << "Frame timeout\n";
                ++timeout_count;
                if (++consecutive_timeouts >= 5) {
                    std::cerr << "5 consecutive frame timeouts; "
                                 "stopping acquisition.\n";
                    break;
                }
                continue;
            }
            consecutive_timeouts = 0;

            if (frame.image.empty()) {
                std::cerr << "Frame empty (warning); skipping.\n";
                continue;
            }

            ++received_frames;
            const auto observations = detector.detect(frame.image);
            if (!observations.empty()) {
                ++frames_with_detection;
            }

            // Print the first frame, every 10th frame, or any frame with a
            // detection, to avoid flooding the terminal at 100 FPS.
            const bool should_print =
                received_frames == 1 || received_frames % 10 == 0 ||
                !observations.empty();
            if (!should_print) {
                continue;
            }

            std::cout << "Frame:\n";
            std::cout << "  index: " << received_frames << '\n';
            std::cout << "  camera_frame_number: " << frame.frame_number
                      << '\n';
            std::cout << "  width: " << frame.image.cols << '\n';
            std::cout << "  height: " << frame.image.rows << '\n';
            std::cout << "  detections: " << observations.size() << '\n';

            if (!observations.empty()) {
                // Highest cv_score observation, used only for a console
                // diagnostic summary — not an aim/selected/fire target.
                const hnu25::anti_drone::TargetObservation* best_observation =
                    &observations[0];
                for (const auto& obs : observations) {
                    if (obs.cv_score > best_observation->cv_score) {
                        best_observation = &obs;
                    }
                }
                std::cout << "  best_cv_score: " << std::fixed
                          << std::setprecision(2)
                          << best_observation->cv_score << '\n';

                for (std::size_t i = 0; i < observations.size(); ++i) {
                    const auto& obs = observations[i];
                    std::cout << "  target " << i << ":\n";
                    std::cout << "    cv_score: " << std::fixed
                              << std::setprecision(2) << obs.cv_score << '\n';
                    std::cout << "    center: (" << std::fixed
                              << std::setprecision(2) << obs.center.x << ", "
                              << obs.center.y << ")\n";
                    std::cout << "    corners_valid: " << std::boolalpha
                              << obs.corners_valid << '\n';
                    std::cout << "    bullseye_valid: "
                              << obs.bullseye_valid << '\n';
                }
            }
        }

        source.stop();

        // ── Final summary ─────────────────────────────────────────────────
        std::cout << "=== Hik Detection Preview Summary ===\n";
        std::cout << "requested_frames: " << frame_count << '\n';
        std::cout << "received_frames: " << received_frames << '\n';
        std::cout << "frames_with_detection: " << frames_with_detection
                  << '\n';
        std::cout << "timeouts: " << timeout_count << '\n';
        std::cout << '\n';
        std::cout << "Camera:\n  read-only acquisition completed\n";
        std::cout << "Detector:\n  completed\n";
        std::cout << "PnP:\n  DISABLED\n";
        std::cout << "Tracker:\n  DISABLED\n";
        std::cout << "Prediction:\n  DISABLED\n";
        std::cout << "Communication:\n  DISABLED\n";

        if (received_frames == 0) {
            return 5;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Hik detection preview failed: " << error.what() << '\n';
        return 6;
    }
#else
    std::cerr << "Hik MVS support is not available in this build.\n";
    return 2;
#endif
}
