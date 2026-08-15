#pragma once

#include "anti_drone/types.hpp"

#include <opencv2/core/mat.hpp>

#include <vector>

namespace hnu25::anti_drone {

class TargetDetector {
public:
    virtual ~TargetDetector() = default;

    virtual std::vector<TargetObservation> detect(
        const cv::Mat& bgr_image) = 0;
};

}  // namespace hnu25::anti_drone
