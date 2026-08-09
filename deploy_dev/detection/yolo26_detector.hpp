#pragma once

#include "detector.hpp"

#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include <array>
#include <string>
#include <vector>

namespace hnu25 {

// 河北科技大学 Actor&Thinker YOLO26n-Pose OpenVINO 检测器。
// 模型输入为 RGB NCHW float32，输出固定为 [1, 30, 18]。
class Yolo26Detector final : public Detector {
public:
    explicit Yolo26Detector(const std::string& model_path,
                            float conf_threshold = 0.25f,
                            const std::string& performance_mode = "latency");

    std::vector<DetectedArmor> detect(const cv::Mat& bgr_img) override;
    void setConfidenceThreshold(float value) override;

private:
    struct InternalDet {
        int class_id = 0;
        int color_id = 0;
        float confidence = 0.0f;
        cv::Rect box;
        std::array<cv::Point2f, 4> keypoints{};
    };

    void preprocess(const cv::Mat& bgr, float& scale, int& pad_x, int& pad_y);
    std::vector<DetectedArmor> postprocess(const ov::Tensor& output,
                                           const cv::Size& orig_size,
                                           float scale, int pad_x, int pad_y) const;
    static DetectedArmor toResult(const InternalDet& det);

    static constexpr int INPUT_W = 640;
    static constexpr int INPUT_H = 640;
    static constexpr int MAX_DETECTIONS = 30;
    static constexpr int OUTPUT_COLS = 18;

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;

    cv::Mat letterbox_canvas_;
    cv::Mat resized_;
    float conf_threshold_;
};

}  // namespace hnu25
