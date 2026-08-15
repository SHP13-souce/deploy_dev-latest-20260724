#pragma once

#include <opencv2/core.hpp>

#include <array>

namespace hnu25::anti_drone {

struct TargetObservation {
    // Target bounding box in original image coordinates.
    cv::Rect box;

    // Physical target-board outer corners in image coordinates.
    // Order is strictly:
    // [0] top-left
    // [1] top-right
    // [2] bottom-right
    // [3] bottom-left
    std::array<cv::Point2f, 4> corners{};

    // Geometric center of the detected target board.
    cv::Point2f center{};

    // Center estimated from the red bullseye region.
    cv::Point2f bullseye_center{};

    // Traditional-vision confidence.
    float cv_score = 0.0F;

    // Reserved for the future YOLO detector.
    float yolo_score = 0.0F;

    // Confidence after future multi-source fusion.
    // For the traditional-only stage this may equal cv_score.
    float fused_score = 0.0F;

    // Whether corners contain a usable ordered quadrilateral.
    bool corners_valid = false;

    // Whether a valid bullseye center was found.
    bool bullseye_valid = false;

    // Detection source flags. They may both become true after future fusion.
    bool from_cv = false;
    bool from_yolo = false;
};

}  // namespace hnu25::anti_drone
