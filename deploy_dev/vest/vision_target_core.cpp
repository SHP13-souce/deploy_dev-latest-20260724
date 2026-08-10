#include "vest/vision_target_core.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace hnu25::vest {

VisionTargetObservation VisionTargetCore::build(
    const SelectedTarget& selected_target,
    const cv::Size& image_size,
    std::uint64_t frame_id) const {

    if (image_size.width <= 0 || image_size.height <= 0) {
        throw std::invalid_argument("vision target core image size must be positive");
    }

    VisionTargetObservation result;
    result.frame_id = frame_id;

    if (!selected_target.has_target) {
        return result;
    }

    const auto& track = selected_target.track;

    if (track.state != VestTrackState::Tracking &&
        track.state != VestTrackState::TemporarilyLost) {
        return result;
    }

    result.has_target = true;
    result.track_id = static_cast<std::int32_t>(track.track_id);

    const auto timestamp_count = std::chrono::duration_cast<std::chrono::microseconds>(
        track.timestamp.time_since_epoch()).count();

    if (timestamp_count > 0) {
        result.measurement_timestamp_us = static_cast<std::uint64_t>(timestamp_count);
    }

    if (track.state == VestTrackState::TemporarilyLost) {
        result.tracking_state = VisionTrackingState::TemporarilyLost;
        return result;
    }

    result.tracking_state = VisionTrackingState::Tracking;

    if (!selected_target.measurement_valid) {
        return result;
    }

    const double width = static_cast<double>(image_size.width);
    const double height = static_cast<double>(image_size.height);

    const double center_x = static_cast<double>(track.center.x) / width;
    const double center_y = static_cast<double>(track.center.y) / height;

    const double error_x = center_x - 0.5;
    const double error_y = center_y - 0.5;

    const double velocity_x = static_cast<double>(track.velocity.x) / width;
    const double velocity_y = static_cast<double>(track.velocity.y) / height;

    const double predicted_x = static_cast<double>(track.predicted_center.x) / width;
    const double predicted_y = static_cast<double>(track.predicted_center.y) / height;

    const double bbox_w = static_cast<double>(track.box.width) / width;
    const double bbox_h = static_cast<double>(track.box.height) / height;

    const double confidence = static_cast<double>(track.confidence);

    const bool finite = std::isfinite(center_x) && std::isfinite(center_y) &&
                        std::isfinite(error_x) && std::isfinite(error_y) &&
                        std::isfinite(velocity_x) && std::isfinite(velocity_y) &&
                        std::isfinite(predicted_x) && std::isfinite(predicted_y) &&
                        std::isfinite(bbox_w) && std::isfinite(bbox_h) &&
                        std::isfinite(confidence);

    if (!finite || track.box.width < 0.0F || track.box.height < 0.0F) {
        return result;
    }

    result.target_valid = true;

    result.center_x = static_cast<float>(center_x);
    result.center_y = static_cast<float>(center_y);

    result.error_x = static_cast<float>(error_x);
    result.error_y = static_cast<float>(error_y);

    result.velocity_x = static_cast<float>(velocity_x);
    result.velocity_y = static_cast<float>(velocity_y);

    result.predicted_x = static_cast<float>(predicted_x);
    result.predicted_y = static_cast<float>(predicted_y);

    result.bbox_w = static_cast<float>(bbox_w);
    result.bbox_h = static_cast<float>(bbox_h);

    result.confidence = static_cast<float>(confidence);

    return result;
}

}  // namespace hnu25::vest
