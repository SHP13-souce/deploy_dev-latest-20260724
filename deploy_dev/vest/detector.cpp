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

ov::Tensor VestDetector::runInference(const cv::Mat& bgr_img) {
    preprocess(bgr_img);

    infer_request_.infer();

    ov::Tensor output = infer_request_.get_output_tensor();

    if (output.get_element_type() != ov::element::f32) {
        throw std::runtime_error("target_v3 inference output must be FP32");
    }

    const auto output_shape = output.get_shape();
    if (output_shape != ov::Shape{1, OUTPUT_CHANNELS, OUTPUT_CANDIDATES}) {
        throw std::runtime_error("target_v3 inference output must be [1,5,8400]");
    }

    return output;
}

std::vector<VestDetector::RawCandidate> VestDetector::parseOutput(const ov::Tensor& output) const {
    if (output.get_element_type() != ov::element::f32) {
        throw std::runtime_error("target_v3 output parser requires FP32 tensor");
    }

    if (output.get_shape() != ov::Shape{1, OUTPUT_CHANNELS, OUTPUT_CANDIDATES}) {
        throw std::runtime_error("target_v3 output parser requires [1,5,8400]");
    }

    const float* data = output.data<const float>();
    constexpr int stride = OUTPUT_CANDIDATES;

    std::vector<RawCandidate> candidates;
    candidates.reserve(OUTPUT_CANDIDATES);

    for (int i = 0; i < OUTPUT_CANDIDATES; ++i) {
        const float cx = data[0 * stride + i];
        const float cy = data[1 * stride + i];
        const float w  = data[2 * stride + i];
        const float h  = data[3 * stride + i];
        const float confidence = data[4 * stride + i];

        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(w) || !std::isfinite(h) ||
            !std::isfinite(confidence)) {
            continue;
        }

        if (confidence < conf_threshold_) {
            continue;
        }

        if (w <= 0.0F || h <= 0.0F) {
            continue;
        }

        RawCandidate candidate;
        candidate.x1 = cx - 0.5F * w;
        candidate.y1 = cy - 0.5F * h;
        candidate.x2 = cx + 0.5F * w;
        candidate.y2 = cy + 0.5F * h;
        candidate.confidence = confidence;
        candidates.push_back(candidate);
    }

    return candidates;
}

float VestDetector::intersectionOverUnion(const RawCandidate& a, const RawCandidate& b) {
    const float inter_x1 = std::max(a.x1, b.x1);
    const float inter_y1 = std::max(a.y1, b.y1);
    const float inter_x2 = std::min(a.x2, b.x2);
    const float inter_y2 = std::min(a.y2, b.y2);

    const float inter_w = std::max(0.0F, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0F, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float area_b = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);

    const float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0F) {
        return 0.0F;
    }

    const float iou = inter_area / union_area;

    if (!std::isfinite(iou)) {
        return 0.0F;
    }

    return iou;
}

std::vector<VestDetector::RawCandidate> VestDetector::applyNms(
    std::vector<RawCandidate> candidates) const {
    if (candidates.empty()) {
        return {};
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const RawCandidate& lhs, const RawCandidate& rhs) {
                  return lhs.confidence > rhs.confidence;
              });

    std::vector<RawCandidate> kept;
    kept.reserve(candidates.size());

    for (const auto& candidate : candidates) {
        bool suppressed = false;

        for (const auto& selected : kept) {
            if (intersectionOverUnion(candidate, selected) > nms_threshold_) {
                suppressed = true;
                break;
            }
        }

        if (!suppressed) {
            kept.push_back(candidate);
        }
    }

    return kept;
}

}  // namespace hnu25::vest
