#pragma once

#include "vest/target_selector.hpp"

#include <opencv2/core/types.hpp>

#include <cstdint>

namespace hnu25::vest {

enum class VisionTrackingState : std::uint8_t {
    NoTarget = 0,
    Tracking = 1,
    TemporarilyLost = 2
};

struct VisionTargetObservation {
    std::uint64_t frame_id = 0;

    bool has_target = false;
    bool target_valid = false;

    std::int32_t track_id = -1;

    VisionTrackingState tracking_state = VisionTrackingState::NoTarget;

    float center_x = 0.0F;
    float center_y = 0.0F;

    float error_x = 0.0F;
    float error_y = 0.0F;

    float velocity_x = 0.0F;
    float velocity_y = 0.0F;

    float predicted_x = 0.0F;
    float predicted_y = 0.0F;

    float bbox_w = 0.0F;
    float bbox_h = 0.0F;

    float confidence = 0.0F;

    std::uint64_t measurement_timestamp_us = 0;
};

class VisionTargetCore {
public:
    VisionTargetObservation build(const SelectedTarget& selected_target,
                                  const cv::Size& image_size,
                                  std::uint64_t frame_id) const;
};

}  // namespace hnu25::vest
