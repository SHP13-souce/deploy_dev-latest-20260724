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

// Stage 2 concentric-ring verification. Perspective-normalizes the candidate
// board quadrilateral into a canonical square, then measures the radial
// red/non-red profile from the center. The real target is a 50 cm x 50 cm
// white board with a RED / WHITE / RED / WHITE / RED concentric bullseye, so
// from the center outward the profile must alternate at least
// ring_min_transitions times and start with RED.
//
// Returns a score in [0, 1]; 0.0 means the concentric structure is
// definitively absent (center not red, too few alternations, or leading
// non-red), so the caller rejects the candidate.
double computeRingScore(
    const cv::Mat& bgr_image,
    const std::array<cv::Point2f, 4>& corners,
    const TraditionalDetectorConfig& config) {
    const int N = std::max(64, config.ring_warp_size);
    const int bins = std::max(8, config.ring_radial_bins);

    // ── Perspective normalization (candidate quadrilateral only) ──────────
    // Warp only the bounding box around the four corners, never the whole
    // frame, so the cost stays proportional to candidate size.
    const cv::Rect corners_rect = cv::boundingRect(
        std::vector<cv::Point2f>(corners.begin(), corners.end())) &
        cv::Rect(0, 0, bgr_image.cols, bgr_image.rows);
    if (corners_rect.width < 4 || corners_rect.height < 4) {
        return 0.0;
    }

    const cv::Mat roi = bgr_image(corners_rect);

    std::vector<cv::Point2f> src(4);
    const float ox = static_cast<float>(corners_rect.x);
    const float oy = static_cast<float>(corners_rect.y);
    for (int i = 0; i < 4; ++i) {
        src[i] = cv::Point2f(corners[i].x - ox, corners[i].y - oy);
    }

    std::vector<cv::Point2f> dst(4);
    dst[0] = cv::Point2f(0.0F, 0.0F);
    dst[1] = cv::Point2f(static_cast<float>(N - 1), 0.0F);
    dst[2] = cv::Point2f(static_cast<float>(N - 1), static_cast<float>(N - 1));
    dst[3] = cv::Point2f(0.0F, static_cast<float>(N - 1));

    const cv::Mat H = cv::getPerspectiveTransform(src, dst);
    if (H.empty()) {
        return 0.0;
    }

    cv::Mat warped;
    cv::warpPerspective(roi, warped, H, cv::Size(N, N), cv::INTER_LINEAR);
    if (warped.empty()) {
        return 0.0;
    }

    // ── Canonical red mask (same HSV thresholds as Stage 1) ───────────────
    cv::Mat hsv;
    cv::cvtColor(warped, hsv, cv::COLOR_BGR2HSV);

    cv::Mat red1;
    cv::Mat red2;
    cv::inRange(hsv,
                cv::Scalar(config.red_hue_low_1, config.red_saturation_min,
                           config.red_value_min),
                cv::Scalar(config.red_hue_high_1, 255, 255), red1);
    cv::inRange(hsv,
                cv::Scalar(config.red_hue_low_2, config.red_saturation_min,
                           config.red_value_min),
                cv::Scalar(config.red_hue_high_2, 255, 255), red2);

    cv::Mat red_mask;
    cv::bitwise_or(red1, red2, red_mask);

    // ── Radial annulus profile ────────────────────────────────────────────
    const double center = (N - 1) * 0.5;
    const double max_radius = 0.45 * N;
    const double center_radius = config.ring_center_radius_ratio * N;
    const double outer_inner = 0.38 * N;

    std::vector<long long> red_count(bins, 0);
    std::vector<long long> total_count(bins, 0);
    long long center_red = 0;
    long long center_total = 0;
    long long outer_red = 0;
    long long outer_total = 0;

    for (int y = 0; y < N; ++y) {
        const double dy = static_cast<double>(y) - center;
        const uchar* red_row = red_mask.ptr<uchar>(y);
        for (int x = 0; x < N; ++x) {
            const double dx = static_cast<double>(x) - center;
            const double r = std::sqrt(dx * dx + dy * dy);
            if (r > max_radius) {
                continue;
            }

            const bool is_red = red_row[x] != 0;
            const int bin = std::min(
                bins - 1, static_cast<int>(r / max_radius * bins));
            ++total_count[bin];
            if (is_red) {
                ++red_count[bin];
            }

            if (r <= center_radius) {
                ++center_total;
                if (is_red) {
                    ++center_red;
                }
            }
            if (r >= outer_inner) {
                ++outer_total;
                if (is_red) {
                    ++outer_red;
                }
            }
        }
    }

    // ── Red fraction per bin, then 3-bin moving average ───────────────────
    std::vector<double> fraction(bins, 0.0);
    for (int i = 0; i < bins; ++i) {
        if (total_count[i] > 0) {
            fraction[i] = static_cast<double>(red_count[i]) /
                          static_cast<double>(total_count[i]);
        }
    }

    std::vector<double> smoothed(bins, 0.0);
    for (int i = 0; i < bins; ++i) {
        double sum = 0.0;
        int count = 0;
        for (int j = i - 1; j <= i + 1; ++j) {
            if (j < 0 || j >= bins) {
                continue;
            }
            sum += fraction[j];
            ++count;
        }
        smoothed[i] = (count > 0) ? sum / count : 0.0;
    }

    // ── Center red evidence (hard requirement) ────────────────────────────
    const double center_red_fraction =
        center_total > 0
            ? static_cast<double>(center_red) /
                  static_cast<double>(center_total)
            : 0.0;
    if (center_red_fraction < config.ring_center_red_min) {
        return 0.0;  // center is not red → not a concentric bullseye
    }

    // ── Hysteresis RED / NON_RED / UNKNOWN → compressed segments ──────────
    enum class State { kUnknown, kRed, kNonRed };

    std::vector<State> segments;
    State prev = State::kUnknown;
    for (int i = 0; i < bins; ++i) {
        State s = State::kUnknown;
        if (smoothed[i] >= config.ring_red_high) {
            s = State::kRed;
        } else if (smoothed[i] <= config.ring_red_low) {
            s = State::kNonRed;
        }
        if (s == State::kUnknown) {
            continue;  // UNKNOWN does not create a transition
        }
        if (s != prev) {
            segments.push_back(s);
            prev = s;
        }
    }

    const int transitions = static_cast<int>(segments.size()) - 1;
    const bool first_is_red =
        !segments.empty() && segments.front() == State::kRed;

    if (transitions < config.ring_min_transitions) {
        return 0.0;  // not enough red/non-red alternation
    }
    if (!first_is_red) {
        return 0.0;  // the center band must be RED
    }

    // ── Outer non-red evidence (soft penalty) ─────────────────────────────
    const double outer_red_fraction =
        outer_total > 0
            ? static_cast<double>(outer_red) /
                  static_cast<double>(outer_total)
            : 0.0;

    // ── ring_score (soft combination of the four evidence terms) ──────────
    const double center_q = std::clamp(center_red_fraction, 0.0, 1.0);
    const double transition_q = std::clamp(
        static_cast<double>(transitions) /
            static_cast<double>(config.ring_min_transitions),
        0.0, 1.0);
    const double alternate_q = first_is_red ? 1.0 : 0.0;
    const double outer_q = std::clamp(
        1.0 - outer_red_fraction / std::max(config.ring_outer_red_max, 1e-6),
        0.0, 1.0);

    const double ring_score =
        0.25 * center_q + 0.35 * transition_q +
        0.20 * alternate_q + 0.20 * outer_q;

    return std::clamp(ring_score, 0.0, 1.0);
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

        // ── Production hard gates ─────────────────────────────────────────
        if (config_.require_bullseye && !bullseye_valid) {
            continue;
        }
        if (config_.require_valid_corners && !corners_valid) {
            continue;
        }

        // ── Stage 2: concentric-ring verification ────────────────────────
        double ring_score = 0.0;
        if (config_.ring_pattern_enabled) {
            if (!corners_valid) {
                // A real quadrilateral is required to perspective-normalize
                // the board; minAreaRect fallback corners are not valid here.
                continue;
            }
            ring_score =
                computeRingScore(bgr_image, observation.corners, config_);
            if (ring_score < config_.min_ring_score) {
                continue;
            }
        }

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

        // ── Geometry + color + ring fusion ───────────────────────────────
        const double geometry_weight =
            std::max(0.0, static_cast<double>(config_.geometry_weight));
        const double color_weight =
            std::max(0.0, static_cast<double>(config_.color_weight));
        // The ring weight only participates when the ring verifier is
        // enabled; otherwise ring evidence is undefined and must not dilute
        // the normalized score.
        const double ring_weight = config_.ring_pattern_enabled
                                       ? std::max(0.0, static_cast<double>(
                                                           config_.ring_weight))
                                       : 0.0;
        const double weight_sum =
            geometry_weight + color_weight + ring_weight;

        double cv_score = 0.0;
        if (weight_sum > 1e-6) {
            cv_score = (geometry_weight * geometry_score +
                        color_weight * color_score +
                        ring_weight * ring_score) /
                       weight_sum;
        } else {
            // Defensive fallback; config validation requires a positive
            // weight, so this branch is only reachable via a hand-built
            // config object.
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
