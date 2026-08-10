#pragma once

#include "vest/types.hpp"

#include <cstddef>
#include <vector>

namespace hnu25::vest {

struct VestTrackerConfig {
    int confirm_hits = 2;
    int max_lost_frames = 5;

    float min_iou = 0.10F;
    float max_center_distance_ratio = 1.5F;
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

    struct AssociationMetrics {
        float iou = 0.0F;
        float center_distance_ratio = 0.0F;
        bool passes_gate = false;
    };

    static float intersectionOverUnion(const cv::Rect2f& a, const cv::Rect2f& b);

    static float normalizedCenterDistance(const InternalTrack& track,
                                          const DetectedVest& detection);

    struct AssociationMatch {
        std::size_t track_index = 0;
        std::size_t detection_index = 0;

        AssociationMetrics metrics;
    };

    static int statePriority(VestTrackState state);

    AssociationMetrics evaluateAssociation(const InternalTrack& track,
                                           const DetectedVest& detection) const;

    std::vector<AssociationMatch> associate(
        const std::vector<DetectedVest>& detections) const;

    void applyMatchedDetection(InternalTrack& track,
                               const DetectedVest& detection);

    void updateMotionEstimate(InternalTrack& track,
                              const DetectedVest& detection);

    InternalTrack createTrack(const DetectedVest& detection);

    static constexpr double MIN_MOTION_DT_SEC = 0.001;
    static constexpr double MAX_MOTION_DT_SEC = 0.5;

    VestTrackerConfig config_;

    std::vector<InternalTrack> tracks_;

    int next_track_id_ = 0;
};

}  // namespace hnu25::vest
