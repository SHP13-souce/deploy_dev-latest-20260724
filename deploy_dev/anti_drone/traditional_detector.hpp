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
    //
    // The real 50 cm x 50 cm target carries substantial red area from its
    // concentric rings, so this is far above the original 0.003. It is a
    // loose bullseye gate only: the concentric-ring verifier below is what
    // actually rejects false positives.
    double min_red_area_ratio = 0.02;

    // Distance between detected bullseye center and board center,
    // normalized by the candidate bounding-box diagonal.
    double max_bullseye_offset_ratio = 0.15;


    // -------------------- Morphology --------------------

    int morphology_kernel_size = 3;
    int morphology_open_iterations = 1;
    int morphology_close_iterations = 2;


    // -------------------- Production hard gates --------------------

    // When true, a candidate whose bullseye is invalid is rejected outright.
    // Disable for geometry-only evaluation of the white-board stage.
    bool require_bullseye = true;

    // When true, a candidate without a usable ordered quadrilateral
    // (corners_valid == false) is rejected outright. Disable for
    // geometry-only evaluation. The concentric-ring verifier needs real
    // corners to perspective-normalize the board.
    bool require_valid_corners = true;


    // -------------------- Concentric-ring verification --------------------

    // Master switch for the Stage 2 ring verifier. When false, the detector
    // behaves as a geometry + bullseye-only detector (no warp / radial model).
    bool ring_pattern_enabled = true;

    // Side length of the canonical square the candidate board is warped to
    // before the radial profile is measured. Must be >= 64.
    int ring_warp_size = 192;

    // Number of radial annulus bins for the red-fraction profile. Must be
    // >= 8.
    int ring_radial_bins = 48;

    // Hysteresis thresholds for classifying a smoothed red fraction as RED or
    // NON_RED. Fractions in (low, high) are UNKNOWN and do not create a
    // transition. Must satisfy 0 <= ring_red_low < ring_red_high <= 1.
    double ring_red_low = 0.25;
    double ring_red_high = 0.55;

    // Center red evidence is measured over radius <=
    // ring_center_radius_ratio * N. The center must be mostly red.
    double ring_center_radius_ratio = 0.08;
    double ring_center_red_min = 0.50;

    // Minimum number of RED / NON_RED transitions in the compressed radial
    // profile. The real target is R W R W R from the center outward, i.e.
    // at least 4 transitions. This is the core anti-false-positive check.
    int ring_min_transitions = 4;

    // Non-red evidence over 0.38*N <= r <= 0.45*N. A red fraction above this
    // lowers the ring score (soft penalty, not a hard gate).
    double ring_outer_red_max = 0.40;

    // Minimum ring score for a candidate to survive Stage 2.
    double min_ring_score = 0.60;


    // -------------------- Scoring --------------------

    // Weights combining geometry, color, and concentric-ring evidence into
    // TargetObservation::cv_score. The sum is normalized at runtime, so the
    // weights need not add to 1, but at least one must be > 0 (enforced by
    // config validation). The ring weight dominates production confidence
    // because the ring structure is the strongest target discriminator.

    float geometry_weight = 0.20F;
    float color_weight = 0.25F;
    float ring_weight = 0.55F;

    // Minimum CV confidence required for a candidate to survive.
    float min_cv_score = 0.65F;


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
