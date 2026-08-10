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

}  // namespace hnu25::vest
