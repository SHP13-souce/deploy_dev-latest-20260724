#include "detector.hpp"

#include "awakening_detector.hpp"
#include "yolo26_detector.hpp"

#include <stdexcept>

namespace hnu25 {

std::unique_ptr<Detector> makeDetector(const DetectorConfig& config) {
    if (config.backend == "hebei") {
        return std::make_unique<Yolo26Detector>(
            config.model_path, config.conf_threshold, config.performance_mode);
    }
    if (config.backend == "awakening") {
        return std::make_unique<AwakeningDetector>(
            config.model_path, config.classifier_path,
            config.conf_threshold, config.performance_mode);
    }
    throw std::invalid_argument("detector.backend must be hebei or awakening");
}

}  // namespace hnu25
