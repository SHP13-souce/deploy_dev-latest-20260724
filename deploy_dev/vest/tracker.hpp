#pragma once

#include "vest/types.hpp"

#include <vector>

namespace hnu25::vest {

struct VestTrackerConfig {
    int confirm_hits = 2;
    int max_lost_frames = 5;
};

class VestTracker {
public:
    explicit VestTracker(const VestTrackerConfig& config = {});

    std::vector<TrackedVest> update(const std::vector<DetectedVest>& detections);

private:
    VestTrackerConfig config_;
};

}  // namespace hnu25::vest
