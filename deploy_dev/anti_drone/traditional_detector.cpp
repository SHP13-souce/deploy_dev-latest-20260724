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
        observation.cv_score = static_cast<float>(geometry_score);

        // ── Bullseye (not implemented in this stage) ─────────────────────
        observation.bullseye_center = cv::Point2f{};
        observation.bullseye_valid = false;

        // ── Scores / sources ─────────────────────────────────────────────
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

    return observations;
}

}  // namespace hnu25::anti_drone
