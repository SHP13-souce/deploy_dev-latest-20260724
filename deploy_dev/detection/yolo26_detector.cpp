#include "yolo26_detector.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hnu25 {

namespace {

float clampf(float value, float low, float high) {
    return std::max(low, std::min(value, high));
}

Color decodeColor(int color_id) {
    switch (color_id) {
        case 0: return Color::BLUE;
        case 1: return Color::RED;
        case 2: return Color::EXTINGUISH;
        case 3: return Color::PURPLE;
        default: return Color::EXTINGUISH;
    }
}

ov::hint::PerformanceMode performanceMode(const std::string& mode) {
    if (mode == "latency") return ov::hint::PerformanceMode::LATENCY;
    if (mode == "throughput") return ov::hint::PerformanceMode::THROUGHPUT;
    throw std::invalid_argument("detector.performance_mode must be latency or throughput");
}

ArmorLabel decodeLabel(int class_id) {
    // 官方类别: s0_o0,o2,o3,o4,o5,o6,s1_o0,o1,o3,o4,o5,o7。
    static constexpr std::array<ArmorLabel, 12> labels{
        ArmorLabel::SENTRY, ArmorLabel::TWO,   ArmorLabel::THREE,
        ArmorLabel::FOUR,  ArmorLabel::FIVE,  ArmorLabel::OUTPOST,
        ArmorLabel::SENTRY, ArmorLabel::ONE,  ArmorLabel::THREE,
        ArmorLabel::FOUR,  ArmorLabel::FIVE,  ArmorLabel::BASE,
    };
    return class_id >= 0 && class_id < static_cast<int>(labels.size())
               ? labels[class_id]
               : ArmorLabel::UNKNOWN;
}

}  // namespace

Yolo26Detector::Yolo26Detector(const std::string& model_path,
                               float conf_threshold,
                               const std::string& performance_mode)
    : conf_threshold_(conf_threshold) {
    auto model = core_.read_model(model_path);
    const auto input_shape = model->input().get_shape();
    if (input_shape != ov::Shape{1, 3, INPUT_H, INPUT_W}) {
        throw std::runtime_error("YOLO26 model input must be [1,3,640,640]");
    }

    compiled_model_ = core_.compile_model(
        model, "CPU", ov::hint::performance_mode(performanceMode(performance_mode)));
    infer_request_ = compiled_model_.create_infer_request();
    input_tensor_ = ov::Tensor(ov::element::f32, input_shape);
    infer_request_.set_input_tensor(input_tensor_);
    letterbox_canvas_.create(INPUT_H, INPUT_W, CV_8UC3);
}

std::vector<DetectedArmor> Yolo26Detector::detect(const cv::Mat& bgr_img) {
    if (bgr_img.empty() || bgr_img.type() != CV_8UC3) return {};

    float scale = 1.0f;
    int pad_x = 0;
    int pad_y = 0;
    preprocess(bgr_img, scale, pad_x, pad_y);
    infer_request_.infer();
    return postprocess(infer_request_.get_output_tensor(), bgr_img.size(),
                       scale, pad_x, pad_y);
}

void Yolo26Detector::setConfidenceThreshold(float value) {
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
        throw std::invalid_argument("confidence threshold must be in [0,1]");
    conf_threshold_ = value;
}

void Yolo26Detector::preprocess(const cv::Mat& bgr, float& scale,
                                int& pad_x, int& pad_y) {
    scale = std::min(INPUT_W / static_cast<float>(bgr.cols),
                     INPUT_H / static_cast<float>(bgr.rows));
    const int resized_w = static_cast<int>(bgr.cols * scale);
    const int resized_h = static_cast<int>(bgr.rows * scale);
    pad_x = (INPUT_W - resized_w) / 2;
    pad_y = (INPUT_H - resized_h) / 2;

    cv::resize(bgr, resized_, cv::Size(resized_w, resized_h), 0.0, 0.0,
               cv::INTER_LINEAR);
    letterbox_canvas_.setTo(cv::Scalar(114, 114, 114));
    resized_.copyTo(letterbox_canvas_(cv::Rect(pad_x, pad_y, resized_w, resized_h)));

    float* blob = input_tensor_.data<float>();
    constexpr int plane = INPUT_H * INPUT_W;
    constexpr float norm = 1.0f / 255.0f;
    for (int y = 0; y < INPUT_H; ++y) {
        const auto* src = letterbox_canvas_.ptr<cv::Vec3b>(y);
        const int row = y * INPUT_W;
        for (int x = 0; x < INPUT_W; ++x) {
            const auto& pixel = src[x];
            const int index = row + x;
            blob[index] = pixel[2] * norm;
            blob[plane + index] = pixel[1] * norm;
            blob[2 * plane + index] = pixel[0] * norm;
        }
    }
}

std::vector<DetectedArmor> Yolo26Detector::postprocess(
    const ov::Tensor& output, const cv::Size& orig_size,
    float scale, int pad_x, int pad_y) const {
    const auto shape = output.get_shape();
    if (shape != ov::Shape{1, MAX_DETECTIONS, OUTPUT_COLS}) {
        throw std::runtime_error("YOLO26 model output must be [1,30,18]");
    }

    const float* data = output.data<const float>();
    std::vector<DetectedArmor> results;
    results.reserve(MAX_DETECTIONS);
    const float max_x = static_cast<float>(orig_size.width - 1);
    const float max_y = static_cast<float>(orig_size.height - 1);

    for (int i = 0; i < MAX_DETECTIONS; ++i) {
        const float* row = data + i * OUTPUT_COLS;
        const float confidence = row[4];
        if (confidence < conf_threshold_) continue;

        InternalDet det;
        det.confidence = confidence;
        det.class_id = static_cast<int>(std::lround(row[5]));
        det.color_id = static_cast<int>(
            std::max_element(row + 6, row + 10) - (row + 6));

        const float x1 = clampf((row[0] - pad_x) / scale, 0.0f, max_x);
        const float y1 = clampf((row[1] - pad_y) / scale, 0.0f, max_y);
        const float x2 = clampf((row[2] - pad_x) / scale, 0.0f, max_x);
        const float y2 = clampf((row[3] - pad_y) / scale, 0.0f, max_y);
        const int left = static_cast<int>(x1);
        const int top = static_cast<int>(y1);
        const int right = static_cast<int>(x2);
        const int bottom = static_cast<int>(y2);
        if (right - left <= 1 || bottom - top <= 1) continue;
        det.box = cv::Rect(left, top, right - left, bottom - top);

        for (int k = 0; k < 4; ++k) {
            det.keypoints[k] = {
                clampf((row[10 + k * 2] - pad_x) / scale, 0.0f, max_x),
                clampf((row[11 + k * 2] - pad_y) / scale, 0.0f, max_y),
            };
        }
        results.push_back(toResult(det));
    }
    return results;
}

DetectedArmor Yolo26Detector::toResult(const InternalDet& det) {
    DetectedArmor armor;
    armor.class_id = det.class_id;
    armor.confidence = det.confidence;
    armor.box = det.box;
    armor.timestamp = std::chrono::steady_clock::now();
    armor.color = decodeColor(det.color_id);
    armor.label = decodeLabel(det.class_id);
    armor.kind = det.class_id >= 6 ? ArmorKind::LARGE : ArmorKind::SMALL;

    // 官方模型顺时针输出 TL, BL, BR, TR；下游 PnP 使用 TL, TR, BR, BL。
    armor.keypoints = {
        det.keypoints[0], det.keypoints[3], det.keypoints[2], det.keypoints[1],
    };
    return armor;
}

}  // namespace hnu25
