#include "anti_drone/traditional_detector.hpp"

#include <utility>

namespace hnu25::anti_drone {

TraditionalTargetDetector::TraditionalTargetDetector(
    TraditionalDetectorConfig config)
    : config_(std::move(config)) {}

std::vector<TargetObservation>
TraditionalTargetDetector::detect(const cv::Mat& bgr_image) {
    if (bgr_image.empty()) {
        return {};
    }

    return {};
}

}  // namespace hnu25::anti_drone
