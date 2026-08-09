#include "aim/fire_gate.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aim {
namespace {
constexpr double kTwoPi = 6.28318530717958647692;
}

const char* toString(FireReason reason) {
    switch (reason) {
        case FireReason::Allowed: return "allowed";
        case FireReason::Disabled: return "disabled";
        case FireReason::NotTracking: return "not_tracking";
        case FireReason::InvalidTarget: return "invalid_target";
        case FireReason::StalePose: return "stale_pose";
        case FireReason::NoFeedback: return "no_feedback";
        case FireReason::BadPnp: return "bad_pnp";
        case FireReason::BadNis: return "bad_nis";
        case FireReason::NoBallistic: return "no_ballistic";
        case FireReason::OutOfWindow: return "out_of_window";
        case FireReason::NotStable: return "not_stable";
    }
    return "unknown";
}

FireGate::FireGate(FireGateConfig config) : config_(config) {
    if (config_.max_measurement_age < 0.0 || config_.max_reprojection < 0.0 ||
        config_.max_nis < 0.0 || config_.min_consecutive <= 0 ||
        config_.min_yaw_tolerance < 0.0 || config_.min_pitch_tolerance < 0.0 ||
        config_.small_armor_width <= 0.0 || config_.large_armor_width <= 0.0 ||
        config_.armor_height <= 0.0 ||
        config_.armor_tolerance_ratio < 0.0) {
        throw std::invalid_argument("invalid fire gate configuration");
    }
}

FireDecision FireGate::reject(FireDecision decision, FireReason reason) {
    consecutive_ = 0;
    decision.reason = reason;
    return decision;
}

FireDecision FireGate::evaluate(const FireGateInput& input) {
    FireDecision decision;
    const double distance = input.aim.point.norm();
    if (std::isfinite(distance) && distance > 0.0) {
        const double width = input.armor_type == sp_core::ArmorType::LARGE
                                 ? config_.large_armor_width : config_.small_armor_width;
        const double yaw_angular = std::atan2(width * config_.armor_tolerance_ratio, distance);
        const double pitch_angular = std::atan2(
            config_.armor_height * config_.armor_tolerance_ratio, distance);
        decision.yaw_tolerance = std::max(config_.min_yaw_tolerance, yaw_angular);
        decision.pitch_tolerance = std::max(config_.min_pitch_tolerance, pitch_angular);
    } else {
        decision.yaw_tolerance = config_.min_yaw_tolerance;
        decision.pitch_tolerance = config_.min_pitch_tolerance;
    }
    decision.yaw_error = std::remainder(input.aim.yaw - input.feedback_yaw, kTwoPi);
    decision.pitch_error = input.aim.pitch - input.feedback_pitch;

    if (!config_.enabled) return reject(decision, FireReason::Disabled);
    if (input.track_state != sp_core::TrackState::TRACKING) return reject(decision, FireReason::NotTracking);
    if (!input.target_valid) return reject(decision, FireReason::InvalidTarget);
    if (!input.pose_valid || !std::isfinite(input.measurement_age) ||
        input.measurement_age < 0.0 || input.measurement_age > config_.max_measurement_age)
        return reject(decision, FireReason::StalePose);
    if (!input.feedback_valid || !std::isfinite(input.feedback_yaw) ||
        !std::isfinite(input.feedback_pitch)) return reject(decision, FireReason::NoFeedback);
    if (!std::isfinite(input.reprojection_error) || input.reprojection_error < 0.0 ||
        input.reprojection_error > config_.max_reprojection) return reject(decision, FireReason::BadPnp);
    if (!std::isfinite(input.nis) || input.nis < 0.0 || input.nis > config_.max_nis)
        return reject(decision, FireReason::BadNis);
    if (!input.aim.valid || !std::isfinite(input.aim.yaw) || !std::isfinite(input.aim.pitch) ||
        !std::isfinite(input.aim.fly_time) || input.aim.fly_time <= 0.0 ||
        !std::isfinite(distance) || distance <= 0.0) return reject(decision, FireReason::NoBallistic);
    if (!std::isfinite(decision.yaw_error) || !std::isfinite(decision.pitch_error) ||
        std::abs(decision.yaw_error) > decision.yaw_tolerance ||
        std::abs(decision.pitch_error) > decision.pitch_tolerance)
        return reject(decision, FireReason::OutOfWindow);

    ++consecutive_;
    if (consecutive_ < config_.min_consecutive) {
        decision.reason = FireReason::NotStable;
        return decision;
    }
    decision.allowed = true;
    decision.reason = FireReason::Allowed;
    return decision;
}

}  // namespace aim
