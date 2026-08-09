#pragma once

#include <opencv2/core/mat.hpp>

#include <chrono>
#include <cstdint>

namespace hnu25::camera {

enum class TimestampQuality {
    Received,
    ApproximateExposureCenter,
    ReliableHostTimestamp,
};

struct Frame {
    cv::Mat image;
    std::chrono::steady_clock::time_point captured_at{};
    std::chrono::steady_clock::time_point received_at{};
    std::uint64_t frame_number = 0;
    TimestampQuality timestamp_quality = TimestampQuality::Received;
};

}  // namespace hnu25::camera
