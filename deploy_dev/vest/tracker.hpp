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
    struct InternalTrack {
        TrackedVest output;

        int hits = 0;
        int lost_frames = 0;
    };

    InternalTrack createTrack(const DetectedVest& detection);

    VestTrackerConfig config_;

    std::vector<InternalTrack> tracks_;

    int next_track_id_ = 0;
};

}  // namespace hnu25::vest
