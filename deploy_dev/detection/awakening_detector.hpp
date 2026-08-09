#pragma once

#include "detector.hpp"

#include <openvino/openvino.hpp>
#include <opencv2/core/mat.hpp>

#include <array>
#include <string>
#include <vector>

namespace hnu25 {

// Minimal adapter for Awakening's MIT-licensed opt-1208 TUP detector.
class AwakeningDetector final : public Detector {
public:
    explicit AwakeningDetector(const std::string& model_path,
                               const std::string& classifier_path = {},
                               float conf_threshold = 0.2F,
                               const std::string& performance_mode = "latency");

    std::vector<DetectedArmor> detect(const cv::Mat& bgr_img) override;
    void setConfidenceThreshold(float value) override;

private:
    struct Candidate {
        std::array<cv::Point2f, 4> points{};  // TL, BL, BR, TR
        float confidence = 0.0F;
        int color = 2;
        int label = 8;
        int merged = 1;
    };

    void preprocess(const cv::Mat& image, float& scale, int& pad_x, int& pad_y);
    std::vector<DetectedArmor> postprocess(const ov::Tensor& output,
                                           const cv::Mat& network_image,
                                           const cv::Size& image_size,
                                           float scale, int pad_x, int pad_y) const;
    int classify(const cv::Mat& network_image, const Candidate& candidate) const;

    static constexpr int INPUT_SIZE = 416;
    static constexpr int OUTPUT_COLS = 21;

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    ov::CompiledModel classifier_model_;
    mutable ov::InferRequest classifier_request_;
    cv::Mat letterbox_;
    cv::Mat resized_;
    float conf_threshold_;
};

}  // namespace hnu25
