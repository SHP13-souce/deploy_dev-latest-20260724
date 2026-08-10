#include "vest/target_selector.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hnu25::vest {

SelectedTarget TargetSelector::select(const std::vector<TrackedVest>& tracks,
                                       const cv::Size& image_size) {
    if (image_size.width <= 0 || image_size.height <= 0) {
        throw std::invalid_argument("target selector image size must be positive");
    }

    if (selected_track_id_ >= 0) {
        for (const auto& track : tracks) {
            if (track.track_id != selected_track_id_) {
                continue;
            }

            if (track.state == VestTrackState::Tracking) {
                SelectedTarget result;
                result.has_target = true;
                result.measurement_valid = true;
                result.track = track;
                return result;
            }

            if (track.state == VestTrackState::TemporarilyLost) {
                SelectedTarget result;
                result.has_target = true;
                result.measurement_valid = false;
                result.track = track;
                return result;
            }

            break;
        }

        selected_track_id_ = -1;
    }

    const double image_center_x = static_cast<double>(image_size.width) * 0.5;
    const double image_center_y = static_cast<double>(image_size.height) * 0.5;

    const TrackedVest* best_track = nullptr;
    double best_distance_squared = std::numeric_limits<double>::infinity();

    for (const auto& track : tracks) {
        if (track.state != VestTrackState::Tracking) {
            continue;
        }

        const double dx = static_cast<double>(track.center.x) - image_center_x;
        const double dy = static_cast<double>(track.center.y) - image_center_y;
        const double distance_squared = dx * dx + dy * dy;

        if (!std::isfinite(distance_squared)) {
            continue;
        }

        if (best_track == nullptr || distance_squared < best_distance_squared ||
            (distance_squared == best_distance_squared &&
             track.track_id < best_track->track_id)) {
            best_track = &track;
            best_distance_squared = distance_squared;
        }
    }

    if (best_track == nullptr) {
        selected_track_id_ = -1;
        return SelectedTarget{};
    }

    selected_track_id_ = best_track->track_id;

    SelectedTarget result;
    result.has_target = true;
    result.measurement_valid = true;
    result.track = *best_track;

    return result;
}

}  // namespace hnu25::vest
