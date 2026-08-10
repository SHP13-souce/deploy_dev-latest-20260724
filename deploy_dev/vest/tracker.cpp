#include "vest/tracker.hpp"

#include <algorithm>
#include <chrono>
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

    const bool passes_iou_gate =
        config_.min_iou > 0.0F && metrics.iou >= config_.min_iou;
    const bool passes_distance_gate =
        metrics.center_distance_ratio <= config_.max_center_distance_ratio;

    metrics.passes_gate = passes_iou_gate || passes_distance_gate;

    return metrics;
}

int VestTracker::statePriority(VestTrackState state) {
    switch (state) {
        case VestTrackState::Tracking:        return 3;
        case VestTrackState::TemporarilyLost: return 2;
        case VestTrackState::Tentative:       return 1;
        case VestTrackState::Lost:            return 0;
    }
    return 0;
}

std::vector<VestTracker::AssociationMatch> VestTracker::associate(
    const std::vector<DetectedVest>& detections) const {
    std::vector<AssociationMatch> candidates;

    for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
        const auto& track = tracks_[track_index];

        if (track.output.state == VestTrackState::Lost) {
            continue;
        }

        for (std::size_t detection_index = 0; detection_index < detections.size();
             ++detection_index) {
            const auto metrics = evaluateAssociation(track, detections[detection_index]);

            if (!metrics.passes_gate) {
                continue;
            }

            AssociationMatch match;
            match.track_index = track_index;
            match.detection_index = detection_index;
            match.metrics = metrics;
            candidates.push_back(match);
        }
    }

    auto isIouGated = [this](const AssociationMatch& m) {
        return config_.min_iou > 0.0F && m.metrics.iou >= config_.min_iou;
    };

    std::sort(candidates.begin(), candidates.end(),
              [this, &isIouGated](const AssociationMatch& lhs, const AssociationMatch& rhs) {
                  const bool lhs_iou = isIouGated(lhs);
                  const bool rhs_iou = isIouGated(rhs);

                  if (lhs_iou != rhs_iou) {
                      return lhs_iou;
                  }

                  if (lhs_iou) {
                      if (lhs.metrics.iou != rhs.metrics.iou) {
                          return lhs.metrics.iou > rhs.metrics.iou;
                      }
                      if (lhs.metrics.center_distance_ratio !=
                          rhs.metrics.center_distance_ratio) {
                          return lhs.metrics.center_distance_ratio <
                                 rhs.metrics.center_distance_ratio;
                      }
                  } else {
                      if (lhs.metrics.center_distance_ratio !=
                          rhs.metrics.center_distance_ratio) {
                          return lhs.metrics.center_distance_ratio <
                                 rhs.metrics.center_distance_ratio;
                      }
                      if (lhs.metrics.iou != rhs.metrics.iou) {
                          return lhs.metrics.iou > rhs.metrics.iou;
                      }
                  }

                  const int lhs_prio =
                      statePriority(tracks_[lhs.track_index].output.state);
                  const int rhs_prio =
                      statePriority(tracks_[rhs.track_index].output.state);
                  if (lhs_prio != rhs_prio) {
                      return lhs_prio > rhs_prio;
                  }

                  if (tracks_[lhs.track_index].output.track_id !=
                      tracks_[rhs.track_index].output.track_id) {
                      return tracks_[lhs.track_index].output.track_id <
                             tracks_[rhs.track_index].output.track_id;
                  }

                  return lhs.detection_index < rhs.detection_index;
              });

    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> detection_used(detections.size(), false);

    std::vector<AssociationMatch> accepted_matches;
    accepted_matches.reserve(std::min(tracks_.size(), detections.size()));

    for (const auto& candidate : candidates) {
        if (track_used[candidate.track_index]) {
            continue;
        }
        if (detection_used[candidate.detection_index]) {
            continue;
        }

        accepted_matches.push_back(candidate);
        track_used[candidate.track_index] = true;
        detection_used[candidate.detection_index] = true;
    }

    return accepted_matches;
}

void VestTracker::updateMotionEstimate(InternalTrack& track,
                                       const DetectedVest& detection) {
    // Handle invalid new center: cannot use detection.center for prediction
    if (!std::isfinite(detection.center.x) || !std::isfinite(detection.center.y)) {
        track.output.velocity = cv::Point2f(0.0F, 0.0F);

        if (std::isfinite(track.output.center.x) && std::isfinite(track.output.center.y)) {
            track.output.predicted_center = track.output.center;
        } else {
            track.output.predicted_center = cv::Point2f(0.0F, 0.0F);
        }
        return;
    }

    // Fallback: zero velocity, prediction at current detection position
    track.output.velocity = cv::Point2f(0.0F, 0.0F);
    track.output.predicted_center = detection.center;

    // Check old center is finite
    if (!std::isfinite(track.output.center.x) || !std::isfinite(track.output.center.y)) {
        return;
    }

    // Compute time delta in seconds
    const double dt_sec = std::chrono::duration<double>(
        detection.timestamp - track.output.timestamp).count();

    // Validate dt
    if (!std::isfinite(dt_sec) || dt_sec < MIN_MOTION_DT_SEC || dt_sec > MAX_MOTION_DT_SEC) {
        return;
    }

    // Compute velocity in pixels/second (use double for safety)
    const double dx = static_cast<double>(detection.center.x) -
                      static_cast<double>(track.output.center.x);
    const double dy = static_cast<double>(detection.center.y) -
                      static_cast<double>(track.output.center.y);

    const double velocity_x = dx / dt_sec;
    const double velocity_y = dy / dt_sec;

    // Validate velocity
    if (!std::isfinite(velocity_x) || !std::isfinite(velocity_y)) {
        return;
    }

    // Predict next center using current dt as horizon
    const double predicted_x = static_cast<double>(detection.center.x) + velocity_x * dt_sec;
    const double predicted_y = static_cast<double>(detection.center.y) + velocity_y * dt_sec;

    // Validate prediction
    if (!std::isfinite(predicted_x) || !std::isfinite(predicted_y)) {
        return;
    }

    // Validate float range
    const double max_float = static_cast<double>(std::numeric_limits<float>::max());
    if (std::abs(velocity_x) > max_float ||
        std::abs(velocity_y) > max_float ||
        std::abs(predicted_x) > max_float ||
        std::abs(predicted_y) > max_float) {
        return;
    }

    // Success: write back real estimates
    track.output.velocity = cv::Point2f(
        static_cast<float>(velocity_x),
        static_cast<float>(velocity_y));
    track.output.predicted_center = cv::Point2f(
        static_cast<float>(predicted_x),
        static_cast<float>(predicted_y));
}

void VestTracker::applyMatchedDetection(InternalTrack& track,
                                        const DetectedVest& detection) {
    updateMotionEstimate(track, detection);

    track.output.box = detection.box;
    track.output.center = detection.center;
    track.output.confidence = detection.confidence;
    track.output.timestamp = detection.timestamp;

    ++track.hits;
    track.lost_frames = 0;

    switch (track.output.state) {
        case VestTrackState::Tentative:
            if (track.hits >= config_.confirm_hits) {
                track.output.state = VestTrackState::Tracking;
            }
            break;

        case VestTrackState::Tracking:
            break;

        case VestTrackState::TemporarilyLost:
            track.output.state = VestTrackState::Tracking;
            break;

        case VestTrackState::Lost:
            break;
    }
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
