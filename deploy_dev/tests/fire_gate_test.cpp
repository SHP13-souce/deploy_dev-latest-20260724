#include "aim/fire_gate.hpp"

#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAILED: " << message << '\n'; std::exit(1); }
}

aim::FireGateInput validInput() {
    aim::FireGateInput input;
    input.aim.valid = true;
    input.aim.yaw = 0.1;
    input.aim.pitch = -0.05;
    input.aim.fly_time = 0.1;
    input.aim.point = Eigen::Vector3d(3.0, 0.0, 0.0);
    input.track_state = sp_core::TrackState::TRACKING;
    input.target_valid = input.pose_valid = input.feedback_valid = true;
    input.feedback_yaw = input.aim.yaw;
    input.feedback_pitch = input.aim.pitch;
    input.reprojection_error = 1.0;
    input.nis = 1.0;
    return input;
}
}

int main() {
    aim::FireGateConfig config;
    config.enabled = true;
    config.min_consecutive = 2;
    aim::FireGate gate(config);
    auto input = validInput();
    auto reason = [&](const aim::FireGateInput& value) { return gate.evaluate(value).reason; };
    auto changed = input; changed.track_state = sp_core::TrackState::DETECTING;
    require(reason(changed) == aim::FireReason::NotTracking, "not tracking");
    changed = input; changed.target_valid = false;
    require(reason(changed) == aim::FireReason::InvalidTarget, "invalid target");
    changed = input; changed.pose_valid = false;
    require(reason(changed) == aim::FireReason::StalePose, "stale pose");
    changed = input; changed.measurement_age = 1.0;
    require(reason(changed) == aim::FireReason::StalePose, "old measurement");
    changed = input; changed.feedback_valid = false;
    require(reason(changed) == aim::FireReason::NoFeedback, "no feedback");
    changed = input; changed.reprojection_error = 100.0;
    require(reason(changed) == aim::FireReason::BadPnp, "bad pnp");
    changed = input; changed.nis = 100.0;
    require(reason(changed) == aim::FireReason::BadNis, "bad nis");
    changed = input; changed.aim.valid = false;
    require(reason(changed) == aim::FireReason::NoBallistic, "no ballistic");
    changed = input; changed.feedback_yaw += 1.0;
    require(reason(changed) == aim::FireReason::OutOfWindow, "outside window");
    require(reason(input) == aim::FireReason::NotStable, "first stable frame held");
    const auto allowed = gate.evaluate(input);
    require(allowed.allowed && allowed.reason == aim::FireReason::Allowed, "consecutive gate opens");
    aim::FireGate tolerance_gate(config);
    auto far_small = input;
    far_small.aim.point = Eigen::Vector3d(6.0, 0.0, 0.0);
    const auto far_decision = tolerance_gate.evaluate(far_small);
    auto near_large = input;
    near_large.armor_type = sp_core::ArmorType::LARGE;
    near_large.aim.point = Eigen::Vector3d(2.0, 0.0, 0.0);
    const auto near_decision = tolerance_gate.evaluate(near_large);
    require(near_decision.yaw_tolerance > far_decision.yaw_tolerance,
            "armor size and distance widen dynamic yaw tolerance");
    aim::FireGate disabled;
    require(disabled.evaluate(input).reason == aim::FireReason::Disabled, "disabled");
    require(std::string(aim::toString(aim::FireReason::NoFeedback)) == "no_feedback", "reason string");
    return 0;
}
