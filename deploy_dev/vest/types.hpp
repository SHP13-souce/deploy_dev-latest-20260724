#pragma once

#include <chrono>
#include <opencv2/core/types.hpp>

namespace hnu25::vest {

struct DetectedVest {
    cv::Rect2f box;
    cv::Point2f center;

    float confidence = 0.0F;
    int class_id = 0;

    std::chrono::steady_clock::time_point timestamp;
};

enum class VestTrackState {
    Tentative,
    Tracking,
    TemporarilyLost,
    Lost
};

struct TrackedVest {
    int track_id = -1;

    cv::Rect2f box;
    cv::Point2f center;

    cv::Point2f velocity;
    cv::Point2f predicted_center;

    float confidence = 0.0F;

    VestTrackState state = VestTrackState::Tentative;

    std::chrono::steady_clock::time_point timestamp;
};

}  // namespace hnu25::vest
