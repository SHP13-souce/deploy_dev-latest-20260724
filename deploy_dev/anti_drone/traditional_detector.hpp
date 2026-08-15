#pragma once

#include "anti_drone/detector.hpp"

namespace hnu25::anti_drone {

struct TraditionalDetectorConfig {
    // -------------------- White target board --------------------

    // OpenCV HSV saturation upper bound for white regions.
    // H is intentionally not restricted for white.
    int white_saturation_max = 80;

    // OpenCV HSV value lower bound for white regions.
    int white_value_min = 150;


    // -------------------- Red bullseye --------------------

    // OpenCV hue range is [0, 179].
    // Red wraps around hue=0, therefore use two ranges.

    int red_hue_low_1 = 0;
    int red_hue_high_1 = 15;

    int red_hue_low_2 = 165;
    int red_hue_high_2 = 179;

    int red_saturation_min = 80;
    int red_value_min = 60;


    // -------------------- Candidate geometry --------------------

    // Candidate contour area divided by complete image area.
    //
    // Using a ratio instead of only a fixed pixel area avoids tying
    // the detector to one camera resolution.

    double min_candidate_area_ratio = 0.0001;
    double max_candidate_area_ratio = 0.85;

    // Width / height limits.
    //
    // The physical board is square, but perspective projection can
    // make its apparent aspect ratio deviate significantly from 1.

    double min_aspect_ratio = 0.45;
    double max_aspect_ratio = 2.20;

    // contourArea / minAreaRect.area
    double min_rectangularity = 0.55;

    // approxPolyDP epsilon = perimeter * polygon_epsilon_ratio
    double polygon_epsilon_ratio = 0.03;


    // -------------------- Bullseye validation --------------------

    // Red pixels inside candidate board mask / candidate board mask area.
    double min_red_area_ratio = 0.003;

    // Distance between detected bullseye center and board center,
    // normalized by the candidate bounding-box diagonal.
    //
    // This is deliberately loose for the initial implementation.
    double max_bullseye_offset_ratio = 0.25;


    // -------------------- Morphology --------------------

    int morphology_kernel_size = 3;
    int morphology_open_iterations = 1;
    int morphology_close_iterations = 2;


    // -------------------- Scoring --------------------

    // These two weights will later combine the geometry and color
    // evidence into TargetObservation::cv_score.
    //
    // Defaults intentionally use equal weights until real data exists.

    float geometry_weight = 0.50F;
    float color_weight = 0.50F;

    // Minimum CV confidence required for a candidate to survive.
    float min_cv_score = 0.55F;


    // -------------------- Candidate deduplication --------------------

    // IoU threshold used by future NMS.
    float nms_iou_threshold = 0.40F;
};

class TraditionalTargetDetector final : public TargetDetector {
public:
    explicit TraditionalTargetDetector(
        TraditionalDetectorConfig config = {});

    std::vector<TargetObservation> detect(
        const cv::Mat& bgr_image) override;

private:
    TraditionalDetectorConfig config_;
};

}  // namespace hnu25::anti_drone
