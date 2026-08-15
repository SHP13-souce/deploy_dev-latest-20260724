#include "anti_drone/diagnostic_frame_processor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <iostream>
#include <string>

namespace {

using Clock = std::chrono::steady_clock;

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

// ── Synthetic images (no external asset files) ─────────────────────────────

// 640x480 green background, 160 px white square board centered at (320, 240),
// with a red bullseye. This is the same geometry the detector test family
// relies on and must not be read from disk.
cv::Mat makeTargetImage() {
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 255, 0));  // green BGR
    cv::rectangle(image, cv::Point(240, 160), cv::Point(400, 320),
                  cv::Scalar(255, 255, 255), cv::FILLED);  // white board
    cv::circle(image, cv::Point(320, 240), 28,
               cv::Scalar(0, 0, 255), cv::FILLED);  // red bullseye
    return image;
}

// A valid CV_8UC3 frame with no target.
cv::Mat makeBlankImage() {
    return cv::Mat(480, 640, CV_8UC3, cv::Scalar(0, 255, 0));
}

// ── Synthetic calibration (unit test only; never production YAML) ──────────

hnu25::anti_drone::CalibrationConfig makeTestCalibration() {
    hnu25::anti_drone::CalibrationConfig calib;
    calib.pnp.camera_matrix = cv::Matx33d(
        800.0, 0.0, 320.0,
        0.0, 800.0, 240.0,
        0.0, 0.0, 1.0);
    calib.pnp.distort_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    calib.pnp.R_camera2gimbal = cv::Matx33d::eye();
    calib.pnp.t_camera2gimbal = cv::Vec3d(0.0, 0.0, 0.0);
    calib.pnp.target_width_m = 0.50;
    calib.pnp.target_height_m = 0.50;
    calib.pnp.max_reprojection_error_px = 5.0;
    return calib;
}

hnu25::anti_drone::AntiDroneConfig makeCalibratedConfig() {
    hnu25::anti_drone::AntiDroneConfig config;
    config.calibration = makeTestCalibration();
    return config;
}

// ── Tests ───────────────────────────────────────────────────────────────────

// Test 1: no calibration -> detector-only; no 3D data is fabricated.
void testNoCalibration() {
    hnu25::anti_drone::AntiDroneConfig config;
    config.calibration = std::nullopt;

    hnu25::anti_drone::DiagnosticFrameProcessor processor(config);

    const auto result = processor.process(makeTargetImage(), Clock::now());

    check(!result.calibration_available, "calibration_available == false");
    check(!result.diagnostic_enabled, "diagnostic_enabled == false");
    check(result.observations.size() == 1, "observations.size() == 1");
    check(result.pnp_attempt_count == 0, "pnp_attempt_count == 0");
    check(result.measurements.empty(), "measurements empty");
}

// Test 2: calibration + synthetic target -> the full chain is connected.
void testCalibrationSingleFrame() {
    hnu25::anti_drone::DiagnosticFrameProcessor processor(
        makeCalibratedConfig());

    const auto result = processor.process(makeTargetImage(), Clock::now());

    check(result.calibration_available, "calibration_available == true");
    check(result.diagnostic_enabled, "diagnostic_enabled == true");
    check(result.observations.size() == 1, "observations.size() == 1");
    check(result.pnp_attempt_count == 1, "pnp_attempt_count == 1");
    check(result.measurements.size() == 1, "measurements.size() == 1");
    if (!result.measurements.empty()) {
        check(result.measurements[0].valid, "measurement valid");
    }
}

// Test 3: two consecutive frames confirm TRACKING and keep tracker state
// across calls (the processor does not rebuild the pipeline each frame).
void testTwoConsecutiveFrames() {
    hnu25::anti_drone::DiagnosticFrameProcessor processor(
        makeCalibratedConfig());
    const auto t0 = Clock::now();

    processor.process(makeTargetImage(), t0);
    const auto result = processor.process(
        makeTargetImage(), t0 + std::chrono::milliseconds(20));

    check(result.diagnostic_enabled, "diagnostic_enabled == true");
    check(result.diagnostic.track_state ==
              hnu25::anti_drone::TrackState::TRACKING,
          "track_state == TRACKING");
    check(result.diagnostic.measurement_updated, "measurement_updated == true");
}

// Test 4: an empty measurement frame still calls pipeline.update() and the
// tracker advances to TEMP_LOST with missed_count > 0.
void testEmptyMeasurementStillUpdates() {
    hnu25::anti_drone::DiagnosticFrameProcessor processor(
        makeCalibratedConfig());
    const auto t0 = Clock::now();

    processor.process(makeTargetImage(), t0);
    processor.process(makeTargetImage(), t0 + std::chrono::milliseconds(20));

    const auto result = processor.process(
        makeBlankImage(), t0 + std::chrono::milliseconds(40));

    check(result.observations.empty(), "observations empty");
    check(result.measurements.empty(), "measurements empty");
    check(result.diagnostic_enabled, "diagnostic_enabled == true");
    check(result.diagnostic.track_state ==
              hnu25::anti_drone::TrackState::TEMP_LOST,
          "track_state == TEMP_LOST");
    check(result.diagnostic.missed_count > 0, "missed_count > 0");
}

// Test 5: reset() restarts confirmation, dropping back to DETECTING on the
// next target frame.
void testReset() {
    hnu25::anti_drone::DiagnosticFrameProcessor processor(
        makeCalibratedConfig());
    const auto t0 = Clock::now();

    processor.process(makeTargetImage(), t0);
    processor.process(makeTargetImage(), t0 + std::chrono::milliseconds(20));

    processor.reset();

    const auto result = processor.process(
        makeTargetImage(), t0 + std::chrono::milliseconds(40));

    check(result.diagnostic_enabled, "diagnostic_enabled == true");
    check(result.diagnostic.track_state ==
              hnu25::anti_drone::TrackState::DETECTING,
          "track_state == DETECTING after reset");
}

// Test 6: an empty image is rejected with std::invalid_argument.
void testEmptyImageThrows() {
    hnu25::anti_drone::DiagnosticFrameProcessor processor(
        makeCalibratedConfig());

    bool threw = false;
    try {
        processor.process(cv::Mat{}, Clock::now());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "empty image throws invalid_argument");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"no calibration detector-only", testNoCalibration},
        {"calibration single frame", testCalibrationSingleFrame},
        {"two consecutive frames", testTwoConsecutiveFrames},
        {"empty measurement still updates", testEmptyMeasurementStillUpdates},
        {"reset restarts confirmation", testReset},
        {"empty image throws", testEmptyImageThrows},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone diagnostic frame processor tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
