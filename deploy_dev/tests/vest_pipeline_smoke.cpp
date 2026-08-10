#include "vest/detector.hpp"
#include "vest/tracker.hpp"
#include "vest/target_selector.hpp"
#include "vest/vision_target_core.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void printUsage(const char* program) {
    std::cerr << "Usage:\n"
              << "  " << program << " image <model.xml> <image_path>\n"
              << "  " << program << " video <model.xml> <video_path> [max_frames]\n";
}

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

void printObservation(std::uint64_t frame_id,
                      std::size_t detection_count,
                      std::size_t track_count,
                      const hnu25::vest::VisionTargetObservation& obs) {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "frame=" << frame_id
              << " det=" << detection_count
              << " tracks=" << track_count
              << " has_target=" << (obs.has_target ? 1 : 0)
              << " valid=" << (obs.target_valid ? 1 : 0)
              << " id=" << obs.track_id
              << " state=" << stateName(obs.tracking_state);

    if (obs.has_target) {
        std::cout << " center=(" << obs.center_x << "," << obs.center_y << ")"
                  << " error=(" << obs.error_x << "," << obs.error_y << ")"
                  << " velocity=(" << obs.velocity_x << "," << obs.velocity_y << ")"
                  << " predicted=(" << obs.predicted_x << "," << obs.predicted_y << ")"
                  << " bbox=(" << obs.bbox_w << "," << obs.bbox_h << ")"
                  << " conf=" << obs.confidence
                  << " timestamp_us=" << obs.measurement_timestamp_us;
    }

    std::cout << '\n';
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

int runImageMode(const std::string& model_path, const std::string& image_path) {
    std::cout << "mode=image\nmodel=" << model_path << "\ninput=" << image_path << '\n';

    // ── Load image ────────────────────────────────────────────────────
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[Fatal] cannot read image: " << image_path << '\n';
        return 1;
    }

    std::cout << "image=" << image.cols << "x" << image.rows << '\n';

    // ── Detector ──────────────────────────────────────────────────────
    hnu25::vest::VestDetectorConfig detector_config;
    detector_config.model_path = model_path;

    hnu25::vest::VestDetector detector(detector_config);

    auto detections = detector.detect(image);

    std::cout << "detections=" << detections.size() << '\n';

    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto& d = detections[i];
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "det[" << i << "]"
                  << " conf=" << d.confidence
                  << " center=(" << d.center.x << "," << d.center.y << ")"
                  << " bbox=(x=" << d.box.x
                  << ",y=" << d.box.y
                  << ",w=" << d.box.width
                  << ",h=" << d.box.height << ")\n";
    }

    // ── Tracker (confirm_hits=1 for single-image smoke) ───────────────
    hnu25::vest::VestTrackerConfig tracker_config;
    tracker_config.confirm_hits = 1;

    hnu25::vest::VestTracker tracker(tracker_config);

    auto tracks = tracker.update(detections);

    // ── Selector ──────────────────────────────────────────────────────
    hnu25::vest::TargetSelector selector;
    auto selected = selector.select(tracks, image.size());

    // ── Core ──────────────────────────────────────────────────────────
    hnu25::vest::VisionTargetCore core;
    auto observation = core.build(selected, image.size(), 0);

    printObservation(0, detections.size(), tracks.size(), observation);

    std::cout << "\nimage smoke complete\n";
    return 0;
}

int runVideoMode(const std::string& model_path,
                 const std::string& video_path,
                 int max_frames) {
    std::cout << "mode=video\nmodel=" << model_path << "\ninput=" << video_path
              << "\nmax_frames=" << max_frames << '\n';

    // ── Open video ────────────────────────────────────────────────────
    cv::VideoCapture capture(video_path);
    if (!capture.isOpened()) {
        std::cerr << "[Fatal] cannot open video: " << video_path << '\n';
        return 1;
    }

    const double video_width = capture.get(cv::CAP_PROP_FRAME_WIDTH);
    const double video_height = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
    const double video_fps = capture.get(cv::CAP_PROP_FPS);

    std::cout << "video=" << static_cast<int>(video_width) << "x"
              << static_cast<int>(video_height)
              << " fps=" << video_fps << '\n';

    // ── Detector ──────────────────────────────────────────────────────
    hnu25::vest::VestDetectorConfig detector_config;
    detector_config.model_path = model_path;

    hnu25::vest::VestDetector detector(detector_config);

    // ── Tracker (default config) ──────────────────────────────────────
    hnu25::vest::VestTracker tracker;

    // ── Selector ──────────────────────────────────────────────────────
    hnu25::vest::TargetSelector selector;

    // ── Core ──────────────────────────────────────────────────────────
    hnu25::vest::VisionTargetCore core;

    // ── Per-frame loop ────────────────────────────────────────────────
    int processed_frames = 0;
    int frames_with_detections = 0;
    int frames_with_target = 0;
    int frames_with_valid_target = 0;

    std::uint64_t frame_id = 0;

    while (processed_frames < max_frames) {
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) {
            break;
        }

        ++processed_frames;

        auto detections = detector.detect(frame);

        if (!detections.empty()) {
            ++frames_with_detections;
        }

        auto tracks = tracker.update(detections);

        auto selected = selector.select(tracks, frame.size());

        auto observation = core.build(selected, frame.size(), frame_id);

        if (observation.has_target) {
            ++frames_with_target;
        }

        if (observation.target_valid) {
            ++frames_with_valid_target;
        }

        printObservation(frame_id, detections.size(), tracks.size(), observation);

        ++frame_id;
    }

    if (processed_frames == 0) {
        std::cerr << "[Fatal] video contained no readable frames\n";
        return 1;
    }

    std::cout << "\nvideo smoke complete\n"
              << "frames=" << processed_frames << '\n'
              << "frames_with_detections=" << frames_with_detections << '\n'
              << "frames_with_target=" << frames_with_target << '\n'
              << "frames_with_valid_target=" << frames_with_valid_target << '\n';

    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 2;
    }

    const std::string mode = argv[1];

    if (mode != "image" && mode != "video") {
        std::cerr << "[Error] unknown mode: " << mode
                  << " (expected 'image' or 'video')\n\n";
        printUsage(argv[0]);
        return 2;
    }

    if (mode == "image") {
        if (argc != 4) {
            printUsage(argv[0]);
            return 2;
        }

        const std::string model_path = argv[2];
        const std::string image_path = argv[3];

        try {
            return runImageMode(model_path, image_path);
        } catch (const std::exception& e) {
            std::cerr << "[Fatal] " << e.what() << '\n';
            return 1;
        }
    }

    // mode == "video"
    if (argc != 4 && argc != 5) {
        printUsage(argv[0]);
        return 2;
    }

    const std::string model_path = argv[2];
    const std::string video_path = argv[3];

    int max_frames = 300;

    if (argc == 5) {
        if (!parsePositiveMaxFrames(argv[4], max_frames)) {
            std::cerr << "[Error] invalid max_frames: " << argv[4] << '\n';
            return 2;
        }
    }

    try {
        return runVideoMode(model_path, video_path, max_frames);
    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << '\n';
        return 1;
    }
}
