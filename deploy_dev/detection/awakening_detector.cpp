#include "awakening_detector.hpp"

#include <openvino/core/layout.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hnu25 {
namespace {

ov::hint::PerformanceMode performanceMode(const std::string& mode) {
    if (mode == "latency") return ov::hint::PerformanceMode::LATENCY;
    if (mode == "throughput") return ov::hint::PerformanceMode::THROUGHPUT;
    throw std::invalid_argument("detector.performance_mode must be latency or throughput");
}

float intersectionOverUnion(const std::array<cv::Point2f, 4>& lhs,
                            const std::array<cv::Point2f, 4>& rhs) {
    const cv::Rect2f a = cv::boundingRect(std::vector<cv::Point2f>(lhs.begin(), lhs.end()));
    const cv::Rect2f b = cv::boundingRect(std::vector<cv::Point2f>(rhs.begin(), rhs.end()));
    const float intersection = (a & b).area();
    const float union_area = a.area() + b.area() - intersection;
    return union_area > 0.0F ? intersection / union_area : 0.0F;
}

Color decodeColor(int value) {
    static constexpr std::array<Color, 4> colors{
        Color::BLUE, Color::RED, Color::PURPLE, Color::EXTINGUISH};
    return value >= 0 && value < static_cast<int>(colors.size())
               ? colors[value]
               : Color::EXTINGUISH;
}

ArmorLabel decodeLabel(int value) {
    static constexpr std::array<ArmorLabel, 9> labels{
        ArmorLabel::SENTRY, ArmorLabel::ONE, ArmorLabel::TWO,
        ArmorLabel::THREE, ArmorLabel::FOUR, ArmorLabel::FIVE,
        ArmorLabel::OUTPOST, ArmorLabel::BASE, ArmorLabel::UNKNOWN};
    return value >= 0 && value < static_cast<int>(labels.size())
               ? labels[value]
               : ArmorLabel::UNKNOWN;
}

}  // namespace

AwakeningDetector::AwakeningDetector(const std::string& model_path,
                                     const std::string& classifier_path,
                                     float conf_threshold,
                                     const std::string& performance_mode)
    : conf_threshold_(conf_threshold) {
    auto model = core_.read_model(model_path);
    const auto shape = model->input().get_shape();
    if (shape != ov::Shape{1, 3, INPUT_SIZE, INPUT_SIZE})
        throw std::runtime_error("Awakening opt-1208 input must be [1,3,416,416]");

    ov::preprocess::PrePostProcessor preprocess(model);
    preprocess.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC");
    preprocess.input().model().set_layout("NCHW");
    preprocess.input().preprocess().convert_element_type(ov::element::f32);
    preprocess.output().tensor().set_element_type(ov::element::f32);
    model = preprocess.build();

    compiled_model_ = core_.compile_model(
        model, "CPU", ov::hint::performance_mode(performanceMode(performance_mode)));
    infer_request_ = compiled_model_.create_infer_request();
    if (!classifier_path.empty()) {
        auto classifier = core_.read_model(classifier_path);
        classifier_model_ = core_.compile_model(classifier, "CPU");
        classifier_request_ = classifier_model_.create_infer_request();
    }
    letterbox_.create(INPUT_SIZE, INPUT_SIZE, CV_8UC3);
}

std::vector<DetectedArmor> AwakeningDetector::detect(const cv::Mat& bgr_img) {
    if (bgr_img.empty() || bgr_img.type() != CV_8UC3) return {};
    float scale = 1.0F;
    int pad_x = 0;
    int pad_y = 0;
    preprocess(bgr_img, scale, pad_x, pad_y);
    ov::Tensor tensor(ov::element::u8, {1, INPUT_SIZE, INPUT_SIZE, 3}, letterbox_.data);
    infer_request_.set_input_tensor(tensor);
    infer_request_.infer();
    return postprocess(infer_request_.get_output_tensor(), letterbox_,
                       bgr_img.size(), scale, pad_x, pad_y);
}

void AwakeningDetector::setConfidenceThreshold(float value) {
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
        throw std::invalid_argument("confidence threshold must be in [0,1]");
    conf_threshold_ = value;
}

void AwakeningDetector::preprocess(const cv::Mat& image, float& scale,
                                   int& pad_x, int& pad_y) {
    scale = std::min(INPUT_SIZE / static_cast<float>(image.cols),
                     INPUT_SIZE / static_cast<float>(image.rows));
    const int width = static_cast<int>(image.cols * scale + 0.5F);
    const int height = static_cast<int>(image.rows * scale + 0.5F);
    pad_x = (INPUT_SIZE - width) / 2;
    pad_y = (INPUT_SIZE - height) / 2;
    cv::resize(image, resized_, {width, height}, 0.0, 0.0, cv::INTER_LINEAR);
    letterbox_.setTo(cv::Scalar(114, 114, 114));
    resized_.copyTo(letterbox_(cv::Rect(pad_x, pad_y, width, height)));
}

std::vector<DetectedArmor> AwakeningDetector::postprocess(
    const ov::Tensor& output, const cv::Mat& network_image,
    const cv::Size& image_size,
    float scale, int pad_x, int pad_y) const {
    const auto shape = output.get_shape();
    if (shape.size() != 3 || shape[0] != 1 || shape[2] != OUTPUT_COLS)
        throw std::runtime_error("Awakening opt-1208 output must be [1,N,21]");

    static const std::vector<std::array<int, 3>> grids = [] {
        std::vector<std::array<int, 3>> result;
        for (const int stride : {8, 16, 32}) {
            for (int y = 0; y < INPUT_SIZE / stride; ++y)
                for (int x = 0; x < INPUT_SIZE / stride; ++x)
                    result.push_back({x, y, stride});
        }
        return result;
    }();
    const std::size_t rows = std::min<std::size_t>(shape[1], grids.size());
    const float* data = output.data<const float>();
    std::vector<Candidate> candidates;
    candidates.reserve(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const float* row = data + i * OUTPUT_COLS;
        if (!std::isfinite(row[8]) || row[8] < conf_threshold_) continue;
        Candidate candidate;
        candidate.confidence = row[8];
        candidate.color = static_cast<int>(
            std::max_element(row + 9, row + 13) - (row + 9));
        candidate.label = static_cast<int>(
            std::max_element(row + 13, row + 21) - (row + 13));
        const auto& grid = grids[i];
        for (int point = 0; point < 4; ++point) {
            candidate.points[point] = {
                (row[point * 2] + grid[0]) * grid[2],
                (row[point * 2 + 1] + grid[1]) * grid[2]};
        }
        candidates.push_back(candidate);
    }
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        return a.confidence > b.confidence;
    });
    if (candidates.size() > 128) candidates.resize(128);

    std::vector<Candidate> selected;
    for (const auto& candidate : candidates) {
        auto duplicate = std::find_if(selected.begin(), selected.end(), [&](const Candidate& kept) {
            return intersectionOverUnion(candidate.points, kept.points) > 0.35F;
        });
        if (duplicate == selected.end()) {
            selected.push_back(candidate);
        } else if (candidate.label == duplicate->label && candidate.color == duplicate->color &&
                   intersectionOverUnion(candidate.points, duplicate->points) > 0.9F &&
                   std::abs(candidate.confidence - duplicate->confidence) < 0.95F) {
            for (std::size_t i = 0; i < duplicate->points.size(); ++i) {
                duplicate->points[i] =
                    (duplicate->points[i] * static_cast<float>(duplicate->merged) +
                     candidate.points[i]) /
                    static_cast<float>(duplicate->merged + 1);
            }
            ++duplicate->merged;
        }
    }

    const float max_x = static_cast<float>(image_size.width - 1);
    const float max_y = static_cast<float>(image_size.height - 1);
    std::vector<DetectedArmor> result;
    result.reserve(selected.size());
    for (const auto& candidate : selected) {
        DetectedArmor armor;
        armor.confidence = candidate.confidence;
        const int classified_label = classify(network_image, candidate);
        armor.class_id = classified_label >= 0 ? classified_label : candidate.label;
        armor.color = decodeColor(candidate.color);
        armor.label = decodeLabel(armor.class_id);
        armor.kind = armor.label == ArmorLabel::ONE ? ArmorKind::LARGE : ArmorKind::SMALL;
        armor.timestamp = std::chrono::steady_clock::now();
        armor.keypoints.reserve(4);
        // TUP emits TL, BL, BR, TR; the shared solver consumes TL, TR, BR, BL.
        for (const int index : {0, 3, 2, 1}) {
            armor.keypoints.emplace_back(
                std::clamp((candidate.points[index].x - pad_x) / scale, 0.0F, max_x),
                std::clamp((candidate.points[index].y - pad_y) / scale, 0.0F, max_y));
        }
        armor.box = cv::boundingRect(armor.keypoints);
        if (armor.box.width > 1 && armor.box.height > 1) result.push_back(std::move(armor));
    }
    return result;
}

int AwakeningDetector::classify(const cv::Mat& network_image,
                                const Candidate& candidate) const {
    if (!classifier_model_) return -1;
    constexpr int light_length = 12;
    constexpr int warp_height = 28;
    constexpr int small_width = 32;
    constexpr int large_width = 54;
    constexpr int roi_width = 20;
    const float left_length = cv::norm(candidate.points[0] - candidate.points[1]);
    const float right_length = cv::norm(candidate.points[3] - candidate.points[2]);
    const float average_length = 0.5F * (left_length + right_length);
    if (!(average_length > 1e-3F)) return -1;
    const cv::Point2f left_center = 0.5F * (candidate.points[0] + candidate.points[1]);
    const cv::Point2f right_center = 0.5F * (candidate.points[2] + candidate.points[3]);
    const bool large = cv::norm(left_center - right_center) / average_length > 3.5F;
    const int warp_width = large ? large_width : small_width;
    const int top = (warp_height - light_length) / 2 - 1;
    const std::array<cv::Point2f, 4> source{
        candidate.points[1], candidate.points[0], candidate.points[3], candidate.points[2]};
    const std::array<cv::Point2f, 4> target{
        cv::Point2f(0.0F, static_cast<float>(top + light_length)),
        cv::Point2f(0.0F, static_cast<float>(top)),
        cv::Point2f(static_cast<float>(warp_width - 1), static_cast<float>(top)),
        cv::Point2f(static_cast<float>(warp_width - 1), static_cast<float>(top + light_length))};
    cv::Mat warped;
    cv::warpPerspective(network_image, warped, cv::getPerspectiveTransform(source.data(), target.data()),
                        {warp_width, warp_height});
    cv::Mat gray;
    cv::cvtColor(warped(cv::Rect((warp_width - roi_width) / 2, 0, roi_width, warp_height)),
                 gray, cv::COLOR_BGR2GRAY);
    cv::threshold(gray, gray, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::Mat input_float;
    gray.convertTo(input_float, CV_32F, 1.0 / 255.0);
    const ov::Shape classifier_shape = classifier_model_.input().get_shape();
    if (classifier_shape != ov::Shape{1, roi_width, warp_height, 1}) return -1;
    ov::Tensor input(ov::element::f32, classifier_shape, input_float.data);
    classifier_request_.set_input_tensor(input);
    classifier_request_.infer();
    const ov::Tensor output = classifier_request_.get_output_tensor();
    if (output.get_size() < 8) return -1;
    const float* logits = output.data<const float>();
    const int index = static_cast<int>(std::max_element(logits, logits + 8) - logits);
    // MLP labels: one, two, three, four, five, outpost, sentry, base.
    static constexpr std::array<int, 8> labels{1, 2, 3, 4, 5, 6, 0, 7};
    return labels[index];
}

}  // namespace hnu25
