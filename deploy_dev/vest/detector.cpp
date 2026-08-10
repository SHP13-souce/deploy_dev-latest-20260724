#include "vest/detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace hnu25::vest {

namespace {

ov::hint::PerformanceMode performanceMode(const std::string& mode) {
    if (mode == "latency") return ov::hint::PerformanceMode::LATENCY;
    if (mode == "throughput") return ov::hint::PerformanceMode::THROUGHPUT;
    throw std::invalid_argument("vest.performance_mode must be latency or throughput");
}

}  // namespace

VestDetector::VestDetector(const VestDetectorConfig& config) {
    if (config.model_path.empty()) {
        throw std::invalid_argument("vest.model_path must not be empty");
    }

    setConfidenceThreshold(config.conf_threshold);

    if (!std::isfinite(config.nms_threshold) ||
        config.nms_threshold < 0.0F ||
        config.nms_threshold > 1.0F) {
        throw std::invalid_argument("vest NMS threshold must be in [0,1]");
    }
    nms_threshold_ = config.nms_threshold;

    auto model = core_.read_model(config.model_path);

    if (model->inputs().size() != 1) {
        throw std::runtime_error("target_v3 model must have exactly one input");
    }
    if (model->outputs().size() != 1) {
        throw std::runtime_error("target_v3 model must have exactly one output");
    }

    if (model->input().get_element_type() != ov::element::f32) {
        throw std::runtime_error("target_v3 model input must be FP32");
    }

    const auto input_shape = model->input().get_shape();
    if (input_shape != ov::Shape{1, 3, INPUT_H, INPUT_W}) {
        throw std::runtime_error("target_v3 model input must be [1,3,640,640]");
    }

    if (model->output().get_element_type() != ov::element::f32) {
        throw std::runtime_error("target_v3 model output must be FP32");
    }

    const auto output_shape = model->output().get_shape();
    if (output_shape != ov::Shape{1, OUTPUT_CHANNELS, OUTPUT_CANDIDATES}) {
        throw std::runtime_error("target_v3 model output must be [1,5,8400]");
    }

    compiled_model_ = core_.compile_model(
        model, "CPU",
        ov::hint::performance_mode(performanceMode(config.performance_mode)));

    infer_request_ = compiled_model_.create_infer_request();

    input_tensor_ = ov::Tensor(ov::element::f32, input_shape);
    infer_request_.set_input_tensor(input_tensor_);

    letterbox_canvas_.create(INPUT_H, INPUT_W, CV_8UC3);
}

void VestDetector::setConfidenceThreshold(float value) {
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
        throw std::invalid_argument("vest confidence threshold must be in [0,1]");
    }
    conf_threshold_ = value;
}

void VestDetector::preprocess(const cv::Mat& bgr_img) {
    if (bgr_img.empty()) {
        throw std::invalid_argument("vest input image must not be empty");
    }
    if (bgr_img.type() != CV_8UC3) {
        throw std::invalid_argument("vest input image must be CV_8UC3");
    }

    const float scale = std::min(
        INPUT_W / static_cast<float>(bgr_img.cols),
        INPUT_H / static_cast<float>(bgr_img.rows));

    const int resized_w = std::max(1, static_cast<int>(std::round(bgr_img.cols * scale)));
    const int resized_h = std::max(1, static_cast<int>(std::round(bgr_img.rows * scale)));

    const int pad_x = (INPUT_W - resized_w) / 2;
    const int pad_y = (INPUT_H - resized_h) / 2;

    last_scale_ = scale;
    last_pad_x_ = pad_x;
    last_pad_y_ = pad_y;

    cv::resize(bgr_img, resized_, cv::Size(resized_w, resized_h), 0.0, 0.0, cv::INTER_LINEAR);

    letterbox_canvas_.setTo(cv::Scalar(114, 114, 114));
    resized_.copyTo(letterbox_canvas_(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

    float* blob = input_tensor_.data<float>();
    constexpr int plane = INPUT_H * INPUT_W;
    constexpr float norm = 1.0F / 255.0F;

    for (int y = 0; y < INPUT_H; ++y) {
        const auto* src = letterbox_canvas_.ptr<cv::Vec3b>(y);
        const int row = y * INPUT_W;
        for (int x = 0; x < INPUT_W; ++x) {
            const auto& pixel = src[x];
            const int index = row + x;
            blob[index] = pixel[2] * norm;               // R
            blob[plane + index] = pixel[1] * norm;       // G
            blob[2 * plane + index] = pixel[0] * norm;   // B
        }
    }
}

}  // namespace hnu25::vest
