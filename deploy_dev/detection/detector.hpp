#pragma once

#include "shared/types.hpp"

#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hnu25 {

struct DetectorConfig {
    std::string backend = "hebei";
    std::string model_path;
    std::string classifier_path;
    float conf_threshold = 0.25F;
    std::string performance_mode = "latency";
};

class Detector {
public:
    virtual ~Detector() = default;
    virtual std::vector<DetectedArmor> detect(const cv::Mat& bgr_img) = 0;
    virtual void setConfidenceThreshold(float value) = 0;
};

std::unique_ptr<Detector> makeDetector(const DetectorConfig& config);

}  // namespace hnu25
