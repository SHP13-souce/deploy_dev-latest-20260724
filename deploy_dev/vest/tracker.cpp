#include "vest/tracker.hpp"

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
