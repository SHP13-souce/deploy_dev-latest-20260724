#include "vest/tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hnu25::vest {

VestTracker::VestTracker(const VestTrackerConfig& config)
    : config_(config) {
    if (config_.confirm_hits <= 0) {
        throw std::invalid_argument("vest tracker confirm_hits must be positive");
    }

    if (config_.max_lost_frames < 0) {
        throw std::invalid_argument("vest tracker max_lost_frames must be non-negative");
    }

    if (!std::isfinite(config_.min_iou) ||
        config_.min_iou < 0.0F ||
        config_.min_iou > 1.0F) {
        throw std::invalid_argument("vest tracker min_iou must be finite and within [0, 1]");
    }

    if (!std::isfinite(config_.max_center_distance_ratio) ||
        config_.max_center_distance_ratio < 0.0F) {
        throw std::invalid_argument(
            "vest tracker max_center_distance_ratio must be finite and non-negative");
    }
}

float VestTracker::intersectionOverUnion(const cv::Rect2f& a, const cv::Rect2f& b) {
    if (!std::isfinite(a.x) || !std::isfinite(a.y) ||
        !std::isfinite(a.width) || !std::isfinite(a.height) ||
        !std::isfinite(b.x) || !std::isfinite(b.y) ||
        !std::isfinite(b.width) || !std::isfinite(b.height)) {
        return 0.0F;
    }

    if (a.width <= 0.0F || a.height <= 0.0F ||
        b.width <= 0.0F || b.height <= 0.0F) {
        return 0.0F;
    }

    const float a_x2 = a.x + a.width;
    const float a_y2 = a.y + a.height;
    const float b_x2 = b.x + b.width;
    const float b_y2 = b.y + b.height;

    const float inter_left = std::max(a.x, b.x);
    const float inter_top = std::max(a.y, b.y);
    const float inter_right = std::min(a_x2, b_x2);
    const float inter_bottom = std::min(a_y2, b_y2);

    const float inter_w = std::max(0.0F, inter_right - inter_left);
    const float inter_h = std::max(0.0F, inter_bottom - inter_top);
    const float inter_area = inter_w * inter_h;

    const float area_a = a.width * a.height;
    const float area_b = b.width * b.height;
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

float VestTracker::normalizedCenterDistance(const InternalTrack& track,
                                            const DetectedVest& detection) {
    if (!std::isfinite(track.output.predicted_center.x) ||
        !std::isfinite(track.output.predicted_center.y) ||
        !std::isfinite(detection.center.x) ||
        !std::isfinite(detection.center.y) ||
        !std::isfinite(track.output.box.width) ||
        !std::isfinite(track.output.box.height) ||
        !std::isfinite(detection.box.width) ||
        !std::isfinite(detection.box.height)) {
        return std::numeric_limits<float>::infinity();
    }

    if (track.output.box.width <= 0.0F || track.output.box.height <= 0.0F ||
        detection.box.width <= 0.0F || detection.box.height <= 0.0F) {
        return std::numeric_limits<float>::infinity();
    }

    const float dx = detection.center.x - track.output.predicted_center.x;
    const float dy = detection.center.y - track.output.predicted_center.y;
    const float center_distance = std::hypot(dx, dy);

    const float track_diag = std::hypot(track.output.box.width, track.output.box.height);
    const float detection_diag = std::hypot(detection.box.width, detection.box.height);

    const float box_scale = std::max(track_diag, detection_diag);

    if (box_scale <= 0.0F || !std::isfinite(box_scale)) {
        return std::numeric_limits<float>::infinity();
    }

    const float ratio = center_distance / box_scale;

    if (!std::isfinite(ratio)) {
        return std::numeric_limits<float>::infinity();
    }

    return ratio;
}

VestTracker::AssociationMetrics VestTracker::evaluateAssociation(
    const InternalTrack& track,
    const DetectedVest& detection) const {
    AssociationMetrics metrics;

    metrics.iou = intersectionOverUnion(track.output.box, detection.box);
    metrics.center_distance_ratio = normalizedCenterDistance(track, detection);

    metrics.passes_gate = (metrics.iou >= config_.min_iou) ||
                          (metrics.center_distance_ratio <= config_.max_center_distance_ratio);

    return metrics;
}

VestTracker::InternalTrack VestTracker::createTrack(const DetectedVest& detection) {
    InternalTrack track;

    track.output.track_id = next_track_id_;
    ++next_track_id_;

    track.output.box = detection.box;
    track.output.center = detection.center;
    track.output.confidence = detection.confidence;
    track.output.timestamp = detection.timestamp;

    track.output.velocity = cv::Point2f(0.0F, 0.0F);

    track.output.predicted_center = detection.center;

    track.output.state = VestTrackState::Tentative;

    track.hits = 1;
    track.lost_frames = 0;

    return track;
}

}  // namespace hnu25::vest
