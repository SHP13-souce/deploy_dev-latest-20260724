#include "vest_runtime/vest_runtime.hpp"

#if HNU25_HAS_MVS
#include "camera/hik_frame_source.hpp"
#endif
#include "camera/opencv_frame_source.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

const char* trackingStateName(
    hnu25::vest::VisionTrackingState state) {
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

std::unique_ptr<hnu25::camera::FrameSource> makeFrameSource(
    const hnu25::vest_runtime::RuntimeConfig& config) {
    switch (config.runtime.camera_backend) {
        case hnu25::vest_runtime::CameraBackend::OpenCv:
            return std::make_unique<hnu25::camera::OpenCvFrameSource>(
                config.opencv_camera);

        case hnu25::vest_runtime::CameraBackend::Hik:
#if HNU25_HAS_MVS
            return std::make_unique<hnu25::camera::HikFrameSource>(
                config.hik_camera);
#else
            throw std::runtime_error(
                "Hik camera backend requested but Hikrobot MVS support "
                "is not enabled in this build");
#endif
    }

    throw std::runtime_error("unknown camera backend");
}

std::chrono::steady_clock::time_point frameMeasurementTimestamp(
    const hnu25::camera::Frame& frame) {
    if (frame.captured_at != std::chrono::steady_clock::time_point{}) {
        return frame.captured_at;
    }
    if (frame.received_at != std::chrono::steady_clock::time_point{}) {
        return frame.received_at;
    }
    return std::chrono::steady_clock::now();
}

}  // namespace

namespace hnu25::vest_runtime {

VestRuntime::VestRuntime(RuntimeConfig config)
    : config_(std::move(config))
    , camera_(makeFrameSource(config_))
    , detector_(config_.detector)
    , tracker_(config_.tracker) {}

void VestRuntime::run(std::atomic_bool& stop_requested,
                       ObservationSink& sink) {
    // ── Startup log ──────────────────────────────────────────────────
    {
        const char* backend_name = "unknown";
        switch (config_.runtime.camera_backend) {
            case CameraBackend::Hik:
                backend_name = "hik";
                break;
            case CameraBackend::OpenCv:
                backend_name = "opencv";
                break;
        }
        std::cerr << "[VestRuntime] starting camera backend="
                  << backend_name << '\n';
    }

    camera_->start();

    try {
        std::uint64_t vision_frame_id = 0;
        std::uint64_t processed_frames = 0;
        int consecutive_timeouts = 0;

        while (!stop_requested.load(std::memory_order_relaxed)) {
            hnu25::camera::Frame frame;

            const auto timeout = std::chrono::milliseconds(
                config_.runtime.frame_timeout_ms);

            if (!camera_->waitForFrame(frame, timeout)) {
                if (stop_requested.load(std::memory_order_relaxed)) {
                    break;
                }

                ++consecutive_timeouts;
                std::cerr << "[VestRuntime] camera frame timeout"
                          << " current=" << consecutive_timeouts
                          << " limit="
                          << config_.runtime.max_consecutive_timeouts
                          << '\n';

                if (consecutive_timeouts >=
                    config_.runtime.max_consecutive_timeouts) {
                    throw std::runtime_error(
                        "camera frame timeout limit exceeded");
                }
                continue;
            }

            consecutive_timeouts = 0;

            // ── Frame validity ─────────────────────────────────────
            if (frame.image.empty()) {
                throw std::runtime_error(
                    "camera returned an empty frame");
            }

            // ── Timestamp ──────────────────────────────────────────
            const auto measurement_timestamp =
                frameMeasurementTimestamp(frame);

            // ── Detector ───────────────────────────────────────────
            auto detections = detector_.detect(frame.image);

            for (auto& detection : detections) {
                detection.timestamp = measurement_timestamp;
            }

            // ── Tracker ────────────────────────────────────────────
            auto tracks = tracker_.update(detections);

            // ── Selector ───────────────────────────────────────────
            const auto selected =
                selector_.select(tracks, frame.image.size());

            // ── Core ───────────────────────────────────────────────
            const auto observation =
                core_.build(selected, frame.image.size(),
                            vision_frame_id);

            ++vision_frame_id;
            ++processed_frames;

            // ── Sink ───────────────────────────────────────────────
            sink.onObservation(observation);

            // ── Interval log ───────────────────────────────────────
            if (processed_frames %
                    static_cast<std::uint64_t>(
                        config_.runtime.log_interval_frames) ==
                0) {
                std::cerr << "[VestRuntime]"
                          << " vision_frame=" << observation.frame_id
                          << " camera_frame=" << frame.frame_number
                          << " detections=" << detections.size()
                          << " tracks=" << tracks.size()
                          << " has_target="
                          << (observation.has_target ? 1 : 0)
                          << " target_valid="
                          << (observation.target_valid ? 1 : 0)
                          << " track_id=" << observation.track_id
                          << " state="
                          << trackingStateName(
                                 observation.tracking_state)
                          << '\n';
            }
        }

        camera_->stop();

        std::cerr << "[VestRuntime] stopped"
                  << " processed_frames=" << processed_frames << '\n';

    } catch (...) {
        camera_->stop();
        throw;
    }
}

}  // namespace hnu25::vest_runtime
