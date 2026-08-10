#include "vest/detector.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

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
}

void VestDetector::setConfidenceThreshold(float value) {
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
        throw std::invalid_argument("vest confidence threshold must be in [0,1]");
    }
    conf_threshold_ = value;
}

}  // namespace hnu25::vest
