#pragma once

#include "vest/types.hpp"

#include <opencv2/core/types.hpp>

#include <vector>

namespace hnu25::vest {

struct SelectedTarget {
    bool has_target = false;
    bool measurement_valid = false;

    TrackedVest track;
};

class TargetSelector {
public:
    SelectedTarget select(const std::vector<TrackedVest>& tracks,
                          const cv::Size& image_size);

private:
    int selected_track_id_ = -1;
};

}  // namespace hnu25::vest
