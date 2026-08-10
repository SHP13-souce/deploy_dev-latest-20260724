#include "vest/detector.hpp"
#include "vest/tracker.hpp"
#include "vest/target_selector.hpp"
#include "vest/vision_target_core.hpp"

#if HNU25_HAS_MVS
#include "camera/hik_frame_source.hpp"
#endif

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

const char* stateName(hnu25::vest::VisionTrackingState state) {
    switch (state) {
        case hnu25::vest::VisionTrackingState::NoTarget:
            return "NoTarget";
        case hnu25::vest::VisionTrackingState::Tracking:
            return "Tracking";
        case hnu25::vest::VisionTrackingState::TemporarilyLost:
            return "TemporarilyLost";
    }
    return "Unknown";
}

bool parsePositiveMaxFrames(const char* arg, int& out_value) {
    std::size_t parsed_chars = 0;
    int value = 0;

    try {
        value = std::stoi(arg, &parsed_chars);
    } catch (const std::exception&) {
        return false;
    }

    if (parsed_chars != std::string(arg).size() || value <= 0) {
        return false;
    }

    out_value = value;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // ── CLI parsing ───────────────────────────────────────────────────
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: vest_hik_smoke <model.xml> [max_frames]\n";
        return 2;
    }

    const std::string model_path = argv[1];

    int max_frames = 300;
    if (argc == 3) {
        if (!parsePositiveMaxFrames(argv[2], max_frames)) {
            std::cerr << "[Error] invalid max_frames: " << argv[2] << '\n';
            return 2;
        }
    }

    // ── MVS guard ─────────────────────────────────────────────────────
#if !HNU25_HAS_MVS
    (void)model_path;
    (void)max_frames;
    std::cerr << "Hikrobot MVS support is not enabled in this build\n";
    return 2;
#else

    try {
        // ── Model initialization ────────────────────────────────────
        hnu25::vest::VestDetectorConfig detector_config;
        detector_config.model_path = model_path;

        hnu25::vest::VestDetector detector(detector_config);
        hnu25::vest::VestTracker tracker;
        hnu25::vest::TargetSelector selector;
        hnu25::vest::VisionTargetCore core;

        // ── Camera ──────────────────────────────────────────────────
        hnu25::camera::HikConfig hik_config;

        hnu25::camera::HikFrameSource camera(hik_config);
        camera.start();

        // ── Per-frame loop ──────────────────────────────────────────
        std::uint64_t processed_frames = 0;
        int consecutive_timeouts = 0;

        std::uint64_t frames_with_detections = 0;
        std::uint64_t frames_with_target = 0;
        std::uint64_t frames_with_valid_target = 0;

        bool first_frame_logged = false;
        std::uint64_t prev_camera_frame_number = 0;
        bool have_prev_camera_frame = false;

        const auto start_time = std::chrono::steady_clock::now();

        while (processed_frames < static_cast<std::uint64_t>(max_frames)) {
            hnu25::camera::Frame frame;

            if (!camera.waitForFrame(frame, std::chrono::milliseconds(1000))) {
                std::cerr << "camera frame timeout\n";
                ++consecutive_timeouts;
                if (consecutive_timeouts >= 5) {
                    throw std::runtime_error(
                        "camera timed out 5 consecutive times");
                }
                continue;
            }

            consecutive_timeouts = 0;

            if (frame.image.empty()) {
                std::cerr << "[Warning] empty camera frame\n";
                continue;
            }

            // ── First-frame log ────────────────────────────────────
            if (!first_frame_logged) {
                std::cout << "camera opened\n"
                          << "resolution=" << frame.image.cols << "x"
                          << frame.image.rows << '\n'
                          << "first_camera_frame=" << frame.frame_number
                          << '\n';
                first_frame_logged = true;
            }

            // ── Frame-number gap detection ─────────────────────────
            if (have_prev_camera_frame) {
                if (frame.frame_number != prev_camera_frame_number + 1) {
                    std::cerr << "[Warning] camera frame_number jump: "
                              << prev_camera_frame_number << " -> "
                              << frame.frame_number << '\n';
                }
            }
            prev_camera_frame_number = frame.frame_number;
            have_prev_camera_frame = true;

            // ── Detector ───────────────────────────────────────────
            auto detections = detector.detect(frame.image);

            // Override timestamps from camera frame when available.
            if (frame.captured_at !=
                std::chrono::steady_clock::time_point{}) {
                for (auto& d : detections) {
                    d.timestamp = frame.captured_at;
                }
            } else if (frame.received_at !=
                       std::chrono::steady_clock::time_point{}) {
                for (auto& d : detections) {
                    d.timestamp = frame.received_at;
                }
            }

            if (!detections.empty()) {
                ++frames_with_detections;
            }

            // ── Tracker ────────────────────────────────────────────
            auto tracks = tracker.update(detections);

            // ── Selector ───────────────────────────────────────────
            auto selected = selector.select(tracks, frame.image.size());

            // ── Core ───────────────────────────────────────────────
            auto observation = core.build(selected, frame.image.size(),
                                          processed_frames);

            if (observation.has_target) {
                ++frames_with_target;
            }

            if (observation.target_valid) {
                ++frames_with_valid_target;
            }

            // ── Per-frame log ──────────────────────────────────────
            std::cout << std::fixed << std::setprecision(4);

            std::cout << "vision_frame=" << observation.frame_id
                      << " camera_frame=" << frame.frame_number
                      << " detections=" << detections.size()
                      << " tracks=" << tracks.size()
                      << " has_target=" << (observation.has_target ? 1 : 0)
                      << " valid=" << (observation.target_valid ? 1 : 0)
                      << " track_id=" << observation.track_id
                      << " state=" << stateName(observation.tracking_state);

            if (observation.has_target) {
                std::cout << " center=(" << observation.center_x << ","
                          << observation.center_y << ")"
                          << " error=(" << observation.error_x << ","
                          << observation.error_y << ")"
                          << " velocity=(" << observation.velocity_x << ","
                          << observation.velocity_y << ")"
                          << " predicted=(" << observation.predicted_x << ","
                          << observation.predicted_y << ")"
                          << " bbox=(" << observation.bbox_w << ","
                          << observation.bbox_h << ")"
                          << " conf=" << observation.confidence
                          << " timestamp_us="
                          << observation.measurement_timestamp_us;
            }

            std::cout << '\n';

            ++processed_frames;
        }

        // ── Shutdown ───────────────────────────────────────────────
        camera.stop();

        const auto end_time = std::chrono::steady_clock::now();
        const double elapsed_sec =
            std::chrono::duration<double>(end_time - start_time).count();
        const double processing_fps =
            (elapsed_sec > 0.0)
                ? static_cast<double>(processed_frames) / elapsed_sec
                : 0.0;

        std::cout << '\n'
                  << "hik vest smoke complete\n"
                  << "frames=" << processed_frames << '\n'
                  << "frames_with_detections=" << frames_with_detections
                  << '\n'
                  << "frames_with_target=" << frames_with_target << '\n'
                  << "frames_with_valid_target=" << frames_with_valid_target
                  << '\n'
                  << "elapsed_sec=" << elapsed_sec << '\n'
                  << "processing_fps=" << processing_fps << '\n';

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "vest_hik_smoke error: " << error.what() << '\n';
        return 1;
    }
#endif  // HNU25_HAS_MVS
}
