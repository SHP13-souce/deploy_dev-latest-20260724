#include "anti_drone/traditional_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace hnu25::anti_drone {

namespace {

// Order four corners into the fixed convention:
//   [0] TL   [1] TR   [2] BR   [3] BL
// Sort by y first (top pair / bottom pair), then by x inside each pair.
// Stable under rotation and perspective; never throws.
std::array<cv::Point2f, 4> orderCorners(
    const std::array<cv::Point2f, 4>& pts) {
    std::array<cv::Point2f, 4> ordered = pts;

    std::sort(ordered.begin(), ordered.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) {
                  return a.y < b.y;
              });

    // Two smaller y -> top pair, two larger y -> bottom pair.
    const cv::Point2f top0 = ordered[0];
    const cv::Point2f top1 = ordered[1];
    const cv::Point2f bottom0 = ordered[2];
    const cv::Point2f bottom1 = ordered[3];

    // Top pair by x: small = TL, large = TR.
    const cv::Point2f tl = (top0.x <= top1.x) ? top0 : top1;
    const cv::Point2f tr = (top0.x <= top1.x) ? top1 : top0;

    // Bottom pair by x: small = BL, large = BR.
    const cv::Point2f bl = (bottom0.x <= bottom1.x) ? bottom0 : bottom1;
    const cv::Point2f br = (bottom0.x <= bottom1.x) ? bottom1 : bottom0;

    return {tl, tr, br, bl};
}

// Bbox intersection-over-union, computed entirely in double to avoid
// integer overflow on large images. Returns 0.0 for non-overlapping or
// degenerate rectangles, and clamps the ratio to [0, 1].
double intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect intersection = a & b;
    if (intersection.width <= 0 || intersection.height <= 0) {
        return 0.0;
    }

    const double intersection_area =
        static_cast<double>(intersection.width) *
        static_cast<double>(intersection.height);
    const double a_area =
        static_cast<double>(a.width) * static_cast<double>(a.height);
    const double b_area =
        static_cast<double>(b.width) * static_cast<double>(b.height);
    const double union_area = a_area + b_area - intersection_area;

    if (union_area <= 0.0) {
        return 0.0;
    }

    return std::clamp(intersection_area / union_area, 0.0, 1.0);
}

}  // namespace

TraditionalTargetDetector::TraditionalTargetDetector(
    TraditionalDetectorConfig config)
    : config_(std::move(config)) {}

std::vector<TargetObservation>
TraditionalTargetDetector::detect(const cv::Mat& bgr_image) {
    // ── Input safety ──────────────────────────────────────────────────────
    if (bgr_image.empty()) {
        return {};
    }
    if (bgr_image.type() != CV_8UC3) {
        return {};
    }

    // ── HSV + white mask ──────────────────────────────────────────────────
    cv::Mat hsv;
    cv::cvtColor(bgr_image, hsv, cv::COLOR_BGR2HSV);

    cv::Mat white_mask;
    cv::inRange(
        hsv,
        cv::Scalar(0, 0, config_.white_value_min),
        cv::Scalar(179, config_.white_saturation_max, 255),
        white_mask);

    // ── Morphology ────────────────────────────────────────────────────────
    const int ksize = config_.morphology_kernel_size;
    if (ksize > 0) {
        const cv::Mat kernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(ksize, ksize));

        if (config_.morphology_open_iterations > 0) {
            cv::morphologyEx(white_mask, white_mask, cv::MORPH_OPEN, kernel,
                             cv::Point(-1, -1),
                             config_.morphology_open_iterations);
        }
        if (config_.morphology_close_iterations > 0) {
            cv::morphologyEx(white_mask, white_mask, cv::MORPH_CLOSE, kernel,
                             cv::Point(-1, -1),
                             config_.morphology_close_iterations);
        }
    }

    // ── Contours ──────────────────────────────────────────────────────────
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(white_mask, contours, cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    // ── Image area ────────────────────────────────────────────────────────
    const double image_area =
        static_cast<double>(bgr_image.cols) * static_cast<double>(bgr_image.rows);
    if (image_area <= 0.0) {
        return {};
    }

    std::vector<TargetObservation> observations;

    for (const auto& contour : contours) {
        // ── Area ratio ───────────────────────────────────────────────────
        const double contour_area = std::abs(cv::contourArea(contour));
        const double area_ratio = contour_area / image_area;
        if (area_ratio < config_.min_candidate_area_ratio ||
            area_ratio > config_.max_candidate_area_ratio) {
            continue;
        }

        // ── minAreaRect ──────────────────────────────────────────────────
        const cv::RotatedRect rect = cv::minAreaRect(contour);
        const double width = rect.size.width;
        const double height = rect.size.height;
        if (width <= 1e-6 || height <= 1e-6) {
            continue;
        }
        const double rect_area = width * height;
        if (rect_area <= 1e-6) {
            continue;
        }

        // ── Aspect ratio ─────────────────────────────────────────────────
        const double aspect_ratio = width / height;
        if (aspect_ratio < config_.min_aspect_ratio ||
            aspect_ratio > config_.max_aspect_ratio) {
            continue;
        }

        // ── Rectangularity ───────────────────────────────────────────────
        const double rectangularity = contour_area / rect_area;
        if (rectangularity < config_.min_rectangularity) {
            continue;
        }
        const double rectangularity_quality =
            std::min(1.0, std::max(0.0, rectangularity));

        // ── approxPolyDP ─────────────────────────────────────────────────
        const double perimeter = cv::arcLength(contour, true);
        if (perimeter <= 1e-6) {
            continue;
        }
        const double epsilon = perimeter * config_.polygon_epsilon_ratio;

        std::vector<cv::Point> polygon;
        cv::approxPolyDP(contour, polygon, epsilon, true);

        const bool corners_valid =
            (polygon.size() == 4) && cv::isContourConvex(polygon);

        // ── Corners ──────────────────────────────────────────────────────
        std::array<cv::Point2f, 4> corners{};
        if (corners_valid) {
            corners[0] = cv::Point2f(static_cast<float>(polygon[0].x),
                                     static_cast<float>(polygon[0].y));
            corners[1] = cv::Point2f(static_cast<float>(polygon[1].x),
                                     static_cast<float>(polygon[1].y));
            corners[2] = cv::Point2f(static_cast<float>(polygon[2].x),
                                     static_cast<float>(polygon[2].y));
            corners[3] = cv::Point2f(static_cast<float>(polygon[3].x),
                                     static_cast<float>(polygon[3].y));
        } else {
            cv::Point2f raw[4];
            rect.points(raw);
            corners[0] = raw[0];
            corners[1] = raw[1];
            corners[2] = raw[2];
            corners[3] = raw[3];
        }

        TargetObservation observation;
        observation.corners = orderCorners(corners);
        observation.corners_valid = corners_valid;

        // ── Center ───────────────────────────────────────────────────────
        if (corners_valid) {
            cv::Point2f sum(0.0F, 0.0F);
            for (const auto& c : observation.corners) {
                sum += c;
            }
            observation.center = cv::Point2f(sum.x / 4.0F, sum.y / 4.0F);
        } else {
            observation.center = rect.center;
        }

        // ── Box ──────────────────────────────────────────────────────────
        cv::Rect box = cv::boundingRect(contour);
        box &= cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
        if (box.width <= 0 || box.height <= 0) {
            continue;
        }
        observation.box = box;

        // ── Geometry score ───────────────────────────────────────────────
        const double square_quality =
            std::min(width, height) / std::max(width, height);
        const double geometry_score =
            std::sqrt(square_quality * rectangularity_quality);

        // ── Red bullseye analysis (inside candidate board mask) ──────────
        const cv::Mat hsv_roi = hsv(box);

        // Board mask: the filled candidate contour translated into ROI
        // coordinates. It excludes background inside the bounding box.
        cv::Mat board_mask(box.height, box.width, CV_8UC1, cv::Scalar(0));
        {
            std::vector<cv::Point> local_contour;
            local_contour.reserve(contour.size());
            for (const cv::Point& p : contour) {
                local_contour.emplace_back(p.x - box.x, p.y - box.y);
            }
            std::vector<std::vector<cv::Point>> local_contours{
                std::move(local_contour)};
            cv::drawContours(board_mask, local_contours, 0, cv::Scalar(255),
                             cv::FILLED);
        }

        // Red mask: two hue ranges (red wraps around hue=0), then restricted
        // to the board mask so background red outside the board is ignored.
        cv::Mat red_mask_1;
        cv::Mat red_mask_2;
        cv::inRange(hsv_roi,
                    cv::Scalar(config_.red_hue_low_1,
                               config_.red_saturation_min,
                               config_.red_value_min),
                    cv::Scalar(config_.red_hue_high_1, 255, 255),
                    red_mask_1);
        cv::inRange(hsv_roi,
                    cv::Scalar(config_.red_hue_low_2,
                               config_.red_saturation_min,
                               config_.red_value_min),
                    cv::Scalar(config_.red_hue_high_2, 255, 255),
                    red_mask_2);

        cv::Mat red_mask;
        cv::bitwise_or(red_mask_1, red_mask_2, red_mask);
        cv::bitwise_and(red_mask, board_mask, red_mask);

        // Red area ratio (red pixels over board pixels).
        const int board_pixels = cv::countNonZero(board_mask);
        const int red_pixels = cv::countNonZero(red_mask);
        const double red_area_ratio =
            (board_pixels <= 0)
                ? 0.0
                : static_cast<double>(red_pixels) /
                      static_cast<double>(board_pixels);

        // Red centroid -> bullseye center in original image coordinates.
        cv::Point2f bullseye_center{};
        bool have_bullseye_center = false;
        if (red_pixels > 0) {
            const cv::Moments m = cv::moments(red_mask, true);
            if (m.m00 > 1e-6) {
                const double local_x = m.m10 / m.m00;
                const double local_y = m.m01 / m.m00;
                bullseye_center.x =
                    static_cast<float>(box.x + local_x);
                bullseye_center.y =
                    static_cast<float>(box.y + local_y);
                have_bullseye_center = true;
            }
        }

        // Bullseye offset relative to board center, normalized by box diag.
        double offset_ratio = 0.0;
        bool offset_valid = false;
        if (have_bullseye_center) {
            const double dx =
                static_cast<double>(bullseye_center.x) -
                static_cast<double>(observation.center.x);
            const double dy =
                static_cast<double>(bullseye_center.y) -
                static_cast<double>(observation.center.y);
            const double distance = std::sqrt(dx * dx + dy * dy);

            const double diag = std::sqrt(
                static_cast<double>(box.width) * box.width +
                static_cast<double>(box.height) * box.height);
            if (diag > 1e-6) {
                offset_ratio = distance / diag;
                offset_valid = true;
            }
        }

        // bullseye_valid: red evidence present, enough red, center close.
        const bool bullseye_valid =
            have_bullseye_center &&
            (red_pixels > 0) &&
            (red_area_ratio >= config_.min_red_area_ratio) &&
            offset_valid &&
            (offset_ratio <= config_.max_bullseye_offset_ratio);

        observation.bullseye_center =
            have_bullseye_center ? bullseye_center : cv::Point2f{};
        observation.bullseye_valid = bullseye_valid;

        // ── Color score ──────────────────────────────────────────────────
        double red_area_quality = 0.0;
        if (config_.min_red_area_ratio <= 1e-9) {
            red_area_quality = (red_pixels > 0) ? 1.0 : 0.0;
        } else {
            red_area_quality = std::clamp(
                red_area_ratio / config_.min_red_area_ratio, 0.0, 1.0);
        }

        double center_quality = 0.0;
        if (have_bullseye_center &&
            config_.max_bullseye_offset_ratio > 1e-9) {
            center_quality = std::clamp(
                1.0 - offset_ratio / config_.max_bullseye_offset_ratio,
                0.0, 1.0);
        }

        const double color_score =
            std::sqrt(red_area_quality * center_quality);

        // ── Geometry + color fusion ──────────────────────────────────────
        const double geometry_weight =
            std::max(0.0, static_cast<double>(config_.geometry_weight));
        const double color_weight =
            std::max(0.0, static_cast<double>(config_.color_weight));
        const double weight_sum = geometry_weight + color_weight;

        double cv_score = 0.0;
        if (weight_sum > 1e-6) {
            cv_score = (geometry_weight * geometry_score +
                        color_weight * color_score) /
                       weight_sum;
        } else {
            cv_score = geometry_score;
        }
        cv_score = std::clamp(cv_score, 0.0, 1.0);

        // ── Final min_cv_score gate ──────────────────────────────────────
        if (cv_score < config_.min_cv_score) {
            continue;
        }

        observation.cv_score = static_cast<float>(cv_score);
        observation.yolo_score = 0.0F;
        observation.fused_score = observation.cv_score;

        observation.from_cv = true;
        observation.from_yolo = false;

        observations.push_back(std::move(observation));
    }

    // ── Sort by cv_score descending ───────────────────────────────────────
    std::sort(observations.begin(), observations.end(),
              [](const TargetObservation& a, const TargetObservation& b) {
                  return a.cv_score > b.cv_score;
              });

    // ── Non-maximum suppression (bbox IoU) ────────────────────────────────
    // Greedy suppression over the already score-sorted observations: a
    // candidate is dropped only when its IoU with a higher-scoring kept
    // observation strictly exceeds the (clamped) threshold.
    const double nms_threshold = std::clamp(
        static_cast<double>(config_.nms_iou_threshold), 0.0, 1.0);

    std::vector<TargetObservation> kept;
    kept.reserve(observations.size());

    for (auto& candidate : observations) {
        bool suppressed = false;
        for (const auto& existing : kept) {
            if (intersectionOverUnion(candidate.box, existing.box) >
                nms_threshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) {
            kept.push_back(std::move(candidate));
        }
    }

    return kept;
}

}  // namespace hnu25::anti_drone
