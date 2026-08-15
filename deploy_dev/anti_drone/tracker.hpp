#pragma once

#include "anti_drone/pnp_solver.hpp"

#include <opencv2/core.hpp>

#include <chrono>
#include <optional>
#include <vector>

namespace hnu25::anti_drone {

// Tracking state.
enum class TrackState {
    LOST,
    DETECTING,
    TRACKING,
    TEMP_LOST,
};

// Tuning knobs for the first-generation 3D constant-velocity tracker.
struct TrackerConfig {
    int min_detect_count = 2;
    int max_missed_count = 5;

    // Maximum inter-frame gap, in seconds, before the track is treated as
    // temporally discontinuous and reset.
    double max_dt_s = 0.25;

    // Measurement association gate. This is NOT the target's maximum speed;
    // it only bounds how far a new measurement may drift from the prediction
    // before it is considered a different (or spurious) target.
    double max_association_distance_m = 1.0;

    double position_gain = 1.0;
    double velocity_gain = 1.0;
};

// The tracker's 3D state estimate. Position/velocity are in the gimbal frame
// (see TargetTracker3D below).
struct TrackEstimate3D {
    TrackState state = TrackState::LOST;

    cv::Vec3d position_gimbal_m{0.0, 0.0, 0.0};

    // Apparent relative velocity of the target with respect to the current
    // gimbal frame. It may currently contain a mixture of target motion,
    // vehicle motion, and gimbal motion — it is NOT a world-frame target
    // velocity.
    cv::Vec3d velocity_gimbal_m_s{0.0, 0.0, 0.0};

    bool measurement_updated = false;

    int consecutive_detects = 0;
    int missed_count = 0;
};

// First-generation 3D constant-velocity tracker over raw PnP results.
//
// The tracked coordinate frame is the gimbal-relative frame produced directly
// by PnpSolver::solve() (xyz_gimbal). The tracker consumes raw xyz_gimbal
// BEFORE any vision compensation is applied; the manual residual offsets are a
// downstream diagnostic layer and must never enter the motion state.
//
// Future extensions (NOT implemented here):
//  - Rail-direction constraint: once a stable rail direction is known in a
//    consistent frame, the velocity vector can be projected onto it. The axis
//    is deliberately not guessed now.
//  - Vehicle / gimbal motion compensation: this tracker holds gimbal-relative
//    state. If IMU, gimbal feedback, or chassis odometry become available,
//    they should be handled in a separate stabilized/world tracker before or
//    above this one, without changing this API.
class TargetTracker3D {
public:
    explicit TargetTracker3D(TrackerConfig config = {});

    // Feeds one frame of PnP measurements. Returns the current estimate when a
    // track exists (or was just acquired), otherwise std::nullopt. Invalid /
    // non-finite measurements are ignored.
    std::optional<TrackEstimate3D> update(
        const std::vector<PnpResult>& measurements,
        std::chrono::steady_clock::time_point timestamp);

    void reset() noexcept;

    TrackState state() const noexcept;

private:
    TrackEstimate3D makeEstimate(bool measurement_updated) const;

    std::optional<TrackEstimate3D> acquire(
        const std::vector<const PnpResult*>& valid,
        std::chrono::steady_clock::time_point timestamp);

    TrackerConfig config_;

    TrackState state_ = TrackState::LOST;

    cv::Vec3d position_gimbal_m_{0.0, 0.0, 0.0};
    cv::Vec3d velocity_gimbal_m_s_{0.0, 0.0, 0.0};

    int consecutive_detects_ = 0;
    int missed_count_ = 0;

    std::optional<std::chrono::steady_clock::time_point> last_timestamp_;
};

}  // namespace hnu25::anti_drone
