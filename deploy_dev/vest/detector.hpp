#pragma once

#include "vest/types.hpp"

#include <openvino/openvino.hpp>
#include <opencv2/core/mat.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hnu25::vest {

struct VestDetectorConfig {
    std::string model_path;

    float conf_threshold = 0.25F;
    float nms_threshold = 0.45F;

    std::string performance_mode = "latency";
};

class VestDetector {
public:
    explicit VestDetector(const VestDetectorConfig& config);

    std::vector<DetectedVest> detect(const cv::Mat& bgr_img);

    void setConfidenceThreshold(float value);

private:
    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int OUTPUT_CHANNELS = 5;
    static constexpr int OUTPUT_CANDIDATES = 8400;

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;

    float conf_threshold_ = 0.25F;
    float nms_threshold_ = 0.45F;

    cv::Mat letterbox_canvas_;
    cv::Mat resized_;

    float last_scale_ = 1.0F;
    int last_pad_x_ = 0;
    int last_pad_y_ = 0;
};

}  // namespace hnu25::vest
