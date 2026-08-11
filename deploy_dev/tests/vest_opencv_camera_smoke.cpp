#include "vest/detector.hpp"
#include "vest/tracker.hpp"
#include "vest/target_selector.hpp"
#include "vest/vision_target_core.hpp"

#include <opencv2/videoio.hpp>

#include <chrono>
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

bool parseInt(const char* arg, int& out_value) {
    std::size_t parsed_chars = 0;
    int value = 0;

    try {
        value = std::stoi(arg, &parsed_chars);
    } catch (const std::exception&) {
        return false;
    }

    if (parsed_chars != std::string(arg).size()) {
        return false;
    }

    out_value = value;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    // ── CLI parsing ───────────────────────────────────────────────────
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: vest_opencv_camera_smoke <model.xml>"
                  << " [camera_id] [max_frames]\n";
        return 2;
    }

    const std::string model_path = argv[1];

    int camera_id = 0;
    if (argc >= 3) {
        if (!parseInt(argv[2], camera_id) || camera_id < 0) {
            std::cerr << "[Error] invalid camera_id: " << argv[2] << '\n';
            return 2;
        }
    }

    int max_frames = 300;
    if (argc == 4) {
        if (!parseInt(argv[3], max_frames) || max_frames <= 0) {
            std::cerr << "[Error] invalid max_frames: " << argv[3] << '\n';
            return 2;
        }
    }

    try {
        // ── OpenCV VideoCapture ─────────────────────────────────────
        cv::VideoCapture cap(camera_id);

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1440);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1080);

        if (!cap.isOpened()) {
            std::cerr << "cannot open camera id=" << camera_id << '\n';
            return 1;
        }

        const double actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        const double actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

        std::cout << "camera opened\n"
                  << "camera_id=" << camera_id << '\n'
                  << "requested_resolution=1440x1080\n"
                  << "actual_resolution=" << static_cast<int>(actual_width)
                  << "x" << static_cast<int>(actual_height) << '\n';

        // ── Model initialization ────────────────────────────────────
        hnu25::vest::VestDetectorConfig detector_config;
        detector_config.model_path = model_path;

        hnu25::vest::VestDetector detector(detector_config);
        hnu25::vest::VestTracker tracker;
        hnu25::vest::TargetSelector selector;
        hnu25::vest::VisionTargetCore core;

        // ── Per-frame loop ──────────────────────────────────────────
        std::uint64_t processed_frames = 0;

        std::uint64_t frames_with_detections = 0;
        std::uint64_t frames_with_target = 0;
        std::uint64_t frames_with_valid_target = 0;

        bool timing_started = false;
        std::chrono::steady_clock::time_point start_time;

        while (processed_frames < static_cast<std::uint64_t>(max_frames)) {
            cv::Mat frame;

            if (!cap.read(frame)) {
                throw std::runtime_error("camera read failed");
            }

            if (frame.empty()) {
                throw std::runtime_error("camera returned empty frame");
            }

            const auto frame_timestamp = std::chrono::steady_clock::now();

            if (!timing_started) {
                start_time = std::chrono::steady_clock::now();
                timing_started = true;
            }

            // ── Detector ───────────────────────────────────────────
            auto detections = detector.detect(frame);

            for (auto& d : detections) {
                d.timestamp = frame_timestamp;
            }

            if (!detections.empty()) {
                ++frames_with_detections;
            }

            // ── Tracker ────────────────────────────────────────────
            auto tracks = tracker.update(detections);

            // ── Selector ───────────────────────────────────────────
            auto selected = selector.select(tracks, frame.size());

            // ── Core ───────────────────────────────────────────────
            auto observation = core.build(selected, frame.size(),
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
        const auto end_time = std::chrono::steady_clock::now();

        cap.release();

        const double elapsed_sec =
            std::chrono::duration<double>(end_time - start_time).count();
        const double processing_fps =
            (elapsed_sec > 0.0)
                ? static_cast<double>(processed_frames) / elapsed_sec
                : 0.0;

        std::cout << '\n'
                  << "opencv camera vest smoke complete\n"
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
        std::cerr << "vest_opencv_camera_smoke error: " << error.what()
                  << '\n';
        return 1;
    }
}
