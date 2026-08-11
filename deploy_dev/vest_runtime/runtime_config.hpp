#pragma once

#include "camera/hik_frame_source.hpp"
#include "camera/opencv_frame_source.hpp"
#include "vest/detector.hpp"
#include "vest/tracker.hpp"

#include <string>

namespace hnu25::vest_runtime {

enum class CameraBackend {
    Hik,
    OpenCv
};

struct RuntimeSettings {
    CameraBackend camera_backend = CameraBackend::Hik;

    int frame_timeout_ms = 1000;
    int max_consecutive_timeouts = 5;
    int log_interval_frames = 30;
};

struct RuntimeConfig {
    RuntimeSettings runtime;

    hnu25::vest::VestDetectorConfig detector;
    hnu25::vest::VestTrackerConfig tracker;

    hnu25::camera::HikConfig hik_camera;
    hnu25::camera::OpenCvConfig opencv_camera;

    static RuntimeConfig loadFromFile(const std::string& path);
};

}  // namespace hnu25::vest_runtime
