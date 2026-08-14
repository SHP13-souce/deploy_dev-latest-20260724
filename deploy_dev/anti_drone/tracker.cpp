#include "anti_drone/tracker.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace hnu25::anti_drone {

namespace {

bool finite(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

}  // namespace

TargetTracker3D::TargetTracker3D(TrackerConfig config) : config_(config) {
    if (config_.min_detect_count < 1) {
        throw std::invalid_argument("min_detect_count must be >= 1");
    }
    if (config_.max_missed_count < 0) {
        throw std::invalid_argument("max_missed_count must be >= 0");
    }
    if (!std::isfinite(config_.max_dt_s) || !(config_.max_dt_s > 0.0)) {
        throw std::invalid_argument("max_dt_s must be finite and > 0");
    }
    if (!std::isfinite(config_.max_association_distance_m) ||
        !(config_.max_association_distance_m > 0.0)) {
        throw std::invalid_argument(
            "max_association_distance_m must be finite and > 0");
    }
    if (!std::isfinite(config_.position_gain) ||
        !(config_.position_gain > 0.0) || config_.position_gain > 1.0) {
        throw std::invalid_argument(
            "position_gain must be finite and within (0, 1]");
    }
    if (!std::isfinite(config_.velocity_gain) || config_.velocity_gain < 0.0 ||
        config_.velocity_gain > 1.0) {
        throw std::invalid_argument(
            "velocity_gain must be finite and within [0, 1]");
    }
}

std::optional<TrackEstimate3D> TargetTracker3D::update(
    const std::vector<PnpResult>& measurements,
    std::chrono::steady_clock::time_point timestamp) {
    // Collect valid measurements. Only raw gimbal-frame position is consumed;
    // xyz_camera and VisionSolution are deliberately unused here.
    std::vector<const PnpResult*> valid;
    valid.reserve(measurements.size());
    for (const PnpResult& m : measurements) {
        if (m.valid && finite(m.xyz_gimbal) &&
            std::isfinite(m.reprojection_error_px) &&
            m.reprojection_error_px >= 0.0) {
            valid.push_back(&m);
        }
    }

    if (state_ == TrackState::LOST) {
        return acquire(valid, timestamp);
    }

    // ── Temporal continuity ──────────────────────────────────────────────
    const double dt = std::chrono::duration<double>(timestamp - *last_timestamp_)
                          .count();
    if (!(dt > 0.0) || dt > config_.max_dt_s || !std::isfinite(dt)) {
        reset();
        return acquire(valid, timestamp);
    }

    // ── Constant-velocity prediction ─────────────────────────────────────
    const cv::Vec3d predicted =
        position_gimbal_m_ + velocity_gimbal_m_s_ * dt;
    if (!finite(predicted)) {
        reset();
        return acquire(valid, timestamp);
    }

    // ── 3D association: nearest in space, not best single-frame reprojection.
    const PnpResult* matched = nullptr;
    double best_distance = std::numeric_limits<double>::infinity();
    for (const PnpResult* m : valid) {
        const cv::Vec3d diff = m->xyz_gimbal - predicted;
        const double distance = std::sqrt(diff[0] * diff[0] +
                                          diff[1] * diff[1] +
                                          diff[2] * diff[2]);
        if (!std::isfinite(distance)) {
            continue;
        }
        if (distance <= config_.max_association_distance_m &&
            distance < best_distance) {
            best_distance = distance;
            matched = m;
        }
    }

    if (matched != nullptr) {
        const cv::Vec3d residual = matched->xyz_gimbal - predicted;
        const cv::Vec3d position_new =
            predicted + config_.position_gain * residual;
        const cv::Vec3d velocity_new =
            velocity_gimbal_m_s_ +
            config_.velocity_gain * (residual / dt);

        if (!finite(position_new) || !finite(velocity_new)) {
            reset();
            return acquire(valid, timestamp);
        }

        position_gimbal_m_ = position_new;
        velocity_gimbal_m_s_ = velocity_new;
        last_timestamp_ = timestamp;

        if (state_ == TrackState::DETECTING) {
            ++consecutive_detects_;
            missed_count_ = 0;
            if (consecutive_detects_ >= config_.min_detect_count) {
                state_ = TrackState::TRACKING;
            }
        } else {
            // TRACKING or TEMP_LOST re-match -> confirmed tracking.
            state_ = TrackState::TRACKING;
            missed_count_ = 0;
        }

        return makeEstimate(true);
    }

    // ── No match: coast on the prediction ────────────────────────────────
    position_gimbal_m_ = predicted;
    last_timestamp_ = timestamp;

    if (state_ == TrackState::DETECTING) {
        // An unconfirmed track is not allowed to survive on prediction alone.
        reset();
        return std::nullopt;
    }

    if (state_ == TrackState::TRACKING) {
        missed_count_ = 1;
        state_ = TrackState::TEMP_LOST;
        return makeEstimate(false);
    }

    // TEMP_LOST.
    ++missed_count_;
    if (missed_count_ > config_.max_missed_count) {
        reset();
        return std::nullopt;
    }
    return makeEstimate(false);
}

void TargetTracker3D::reset() noexcept {
    state_ = TrackState::LOST;
    position_gimbal_m_ = cv::Vec3d(0.0, 0.0, 0.0);
    velocity_gimbal_m_s_ = cv::Vec3d(0.0, 0.0, 0.0);
    consecutive_detects_ = 0;
    missed_count_ = 0;
    last_timestamp_.reset();
}

TrackState TargetTracker3D::state() const noexcept {
    return state_;
}

TrackEstimate3D TargetTracker3D::makeEstimate(bool measurement_updated) const {
    TrackEstimate3D estimate;
    estimate.state = state_;
    estimate.position_gimbal_m = position_gimbal_m_;
    estimate.velocity_gimbal_m_s = velocity_gimbal_m_s_;
    estimate.measurement_updated = measurement_updated;
    estimate.consecutive_detects = consecutive_detects_;
    estimate.missed_count = missed_count_;
    return estimate;
}

std::optional<TrackEstimate3D> TargetTracker3D::acquire(
    const std::vector<const PnpResult*>& valid,
    std::chrono::steady_clock::time_point timestamp) {
    if (valid.empty()) {
        return std::nullopt;
    }

    // In LOST, pick the single best measurement by reprojection error; ties go
    // to the earliest element in the input vector.
    const PnpResult* best = valid.front();
    for (const PnpResult* m : valid) {
        if (m->reprojection_error_px < best->reprojection_error_px) {
            best = m;
        }
    }

    position_gimbal_m_ = best->xyz_gimbal;
    velocity_gimbal_m_s_ = cv::Vec3d(0.0, 0.0, 0.0);
    consecutive_detects_ = 1;
    missed_count_ = 0;
    last_timestamp_ = timestamp;

    state_ = (config_.min_detect_count == 1) ? TrackState::TRACKING
                                             : TrackState::DETECTING;

    return makeEstimate(true);
}

}  // namespace hnu25::anti_drone
