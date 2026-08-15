#include "anti_drone/config.hpp"
#include "anti_drone/diagnostic_csv.hpp"
#include "anti_drone/diagnostic_frame_processor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Offline multi-frame visual replay entry point: image sequence ->
// DiagnosticFrameProcessor (detector -> optional PnP -> optional
// DiagnosticPipeline) -> optional diagnostic CSV. No camera, no communication,
// no control commands, no automatic execution.
int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cout << "Usage: anti_drone_sequence_preview "
                     "<image_dir> <config_yaml> <output_csv> [fps]\n";
        return 1;
    }

    const std::filesystem::path image_dir = argv[1];
    const std::string config_path = argv[2];
    const std::string output_csv = argv[3];

    double fps = 50.0;
    if (argc >= 5) {
        try {
            fps = std::stod(argv[4]);
        } catch (const std::exception&) {
            std::cout << "Usage: anti_drone_sequence_preview "
                         "<image_dir> <config_yaml> <output_csv> [fps]\n";
            return 1;
        }
        if (!std::isfinite(fps) || fps <= 0.0) {
            std::cout << "Usage: anti_drone_sequence_preview "
                         "<image_dir> <config_yaml> <output_csv> [fps]\n";
            return 1;
        }
    }

    try {
        // ── Enumerate images (one level, no recursion) ────────────────────
        if (!std::filesystem::is_directory(image_dir)) {
            std::cerr << "Image directory does not exist: " << image_dir
                      << '\n';
            return 2;
        }

        std::vector<std::filesystem::path> image_files;
        for (const auto& entry :
             std::filesystem::directory_iterator(image_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                image_files.push_back(entry.path());
            }
        }

        if (image_files.empty()) {
            std::cerr << "No image files found in directory: " << image_dir
                      << '\n';
            return 2;
        }

        // Lexicographical order by filename (frame_0001, frame_0002, ...).
        std::sort(image_files.begin(), image_files.end());

        // ── Load config through the production API ────────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config: " << config_path
                      << ": " << error.what() << '\n';
            return 3;
        }

        // ── Reusable per-frame core: detector -> optional PnP -> optional
        //    DiagnosticPipeline ────────────────────────────────────────────
        hnu25::anti_drone::DiagnosticFrameProcessor processor(config);

        const bool has_calibration = config.calibration.has_value();
        std::ofstream csv;

        if (has_calibration) {
            csv.open(output_csv);
            if (!csv.is_open()) {
                std::cerr << "Failed to open output CSV: " << output_csv
                          << '\n';
                return 5;
            }
            hnu25::anti_drone::writeDiagnosticCsvHeader(csv);
        } else {
            std::cout << "Diagnostic CSV:\n";
            std::cout << "  DISABLED\n";
            std::cout << "  reason: camera calibration is not configured\n";
        }

        // ── Replay loop ───────────────────────────────────────────────────
        // Offline replay timeline, NOT real capture time and NOT a real-latency
        // estimate.
        const auto base = std::chrono::steady_clock::time_point{};

        std::size_t processed_frames = 0;
        std::size_t failed_images = 0;
        std::size_t frames_with_detection = 0;
        std::size_t total_detections = 0;
        std::size_t pnp_attempt_count = 0;
        std::size_t pnp_valid_count = 0;
        std::uint64_t frame_index = 0;

        for (const auto& path : image_files) {
            const cv::Mat image = cv::imread(path.string(), cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Failed to read image (skipping): " << path
                          << '\n';
                ++failed_images;
                continue;
            }

            ++processed_frames;
            const std::uint64_t index = frame_index;
            ++frame_index;

            const double seconds = static_cast<double>(index) / fps;
            const auto timestamp =
                base + std::chrono::duration_cast<
                           std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(seconds));

            const auto frame_result = processor.process(image, timestamp);
            const auto& observations = frame_result.observations;

            total_detections += observations.size();
            if (!observations.empty()) {
                ++frames_with_detection;
            }

            if (has_calibration) {
                pnp_attempt_count += frame_result.pnp_attempt_count;
                pnp_valid_count += frame_result.measurements.size();

                try {
                    hnu25::anti_drone::writeDiagnosticCsvRow(
                        csv, index, seconds, frame_result.diagnostic);
                } catch (const std::exception& error) {
                    std::cerr << "Failed to write diagnostic CSV: "
                              << error.what() << '\n';
                    return 5;
                }
            }

            // Print first frame, every 10th frame, or any frame with a
            // detection, to avoid flooding the terminal.
            const bool should_print =
                index == 0 || index % 10 == 0 || !observations.empty();
            if (!should_print) {
                continue;
            }

            std::cout << "Frame:\n";
            std::cout << "  index: " << index << '\n';
            std::cout << "  file: " << path << '\n';
            std::cout << "  width: " << image.cols << '\n';
            std::cout << "  height: " << image.rows << '\n';
            std::cout << "  detections: " << observations.size() << '\n';

            if (has_calibration) {
                std::cout << "  pnp_attempts: "
                          << frame_result.pnp_attempt_count << '\n';
                std::cout << "  pnp_valid: "
                          << frame_result.measurements.size() << '\n';
                std::cout << "  track_state: "
                          << hnu25::anti_drone::trackStateName(
                                 frame_result.diagnostic.track_state)
                          << '\n';
                std::cout << "  track_available: " << std::boolalpha
                          << frame_result.diagnostic.track_available << '\n';
                std::cout << "  prediction_valid: "
                          << frame_result.diagnostic.prediction_valid << '\n';
            }
        }

        // ── Final summary ─────────────────────────────────────────────────
        std::cout << "=== Anti-Drone Sequence Preview Summary ===\n";
        std::cout << "input_directory: " << image_dir << '\n';
        std::cout << "configured_fps: " << fps << '\n';
        std::cout << "image_files: " << image_files.size() << '\n';
        std::cout << "processed_frames: " << processed_frames << '\n';
        std::cout << "failed_images: " << failed_images << '\n';
        std::cout << "frames_with_detection: " << frames_with_detection
                  << '\n';
        std::cout << "total_detections: " << total_detections << '\n';
        std::cout << '\n';

        if (has_calibration) {
            std::cout << "Calibration:\n  READY\n\n";
            std::cout << "PnP:\n  ENABLED\n";
            std::cout << "  attempts: " << pnp_attempt_count << '\n';
            std::cout << "  valid: " << pnp_valid_count << '\n';
            std::cout << '\n';
            std::cout << "Diagnostic pipeline:\n  ENABLED\n\n";
            std::cout << "Diagnostic CSV:\n  ENABLED\n  path: " << output_csv
                      << '\n';
        } else {
            std::cout << "Calibration:\n  NOT CONFIGURED\n\n";
            std::cout << "PnP:\n  DISABLED\n\n";
            std::cout << "Diagnostic pipeline:\n  DISABLED\n\n";
            std::cout << "Diagnostic CSV:\n  DISABLED\n";
        }
        std::cout << '\n';
        std::cout << "Camera:\n  DISABLED\n\n";
        std::cout << "Communication:\n  DISABLED\n";

        if (processed_frames == 0) {
            return 4;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Sequence preview failed: " << error.what() << '\n';
        return 6;
    }
}
