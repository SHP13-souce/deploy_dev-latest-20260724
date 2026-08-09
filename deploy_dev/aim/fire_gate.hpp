#pragma once

#include "sp_core/sp_core.hpp"

namespace aim {

enum class FireReason {
    Allowed,
    Disabled,
    NotTracking,
    InvalidTarget,
    StalePose,
    NoFeedback,
    BadPnp,
    BadNis,
    NoBallistic,
    OutOfWindow,
    NotStable
};

const char* toString(FireReason reason);

struct FireGateConfig {
    bool enabled = false;
    double max_measurement_age = 0.05;
    double max_reprojection = 5.0;
    double max_nis = 13.277;
    int min_consecutive = 3;
    double min_yaw_tolerance = 0.003;
    double min_pitch_tolerance = 0.003;
    double small_armor_width = 0.135;
    double large_armor_width = 0.230;
    double armor_height = 0.056;
    double armor_tolerance_ratio = 0.25;
};

struct FireGateInput {
    sp_core::AimResult aim;
    sp_core::TrackState track_state = sp_core::TrackState::LOST;
    bool target_valid = false;
    bool pose_valid = false;
    bool feedback_valid = false;
    double measurement_age = 0.0;
    double reprojection_error = 0.0;
    double nis = 0.0;
    double feedback_yaw = 0.0;
    double feedback_pitch = 0.0;
    sp_core::ArmorType armor_type = sp_core::ArmorType::SMALL;
};

struct FireDecision {
    bool allowed = false;
    FireReason reason = FireReason::Disabled;
    double yaw_error = 0.0;
    double pitch_error = 0.0;
    double yaw_tolerance = 0.0;
    double pitch_tolerance = 0.0;
};

class FireGate {
public:
    explicit FireGate(FireGateConfig config = {});
    FireDecision evaluate(const FireGateInput& input);
    void reset() { consecutive_ = 0; }

private:
    FireDecision reject(FireDecision decision, FireReason reason);
    FireGateConfig config_;
    int consecutive_ = 0;
};

}  // namespace aim
