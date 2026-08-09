#pragma once

#include "sp_core/sp_core.hpp"

#include <optional>

namespace planner {

enum class SolverStatus {
    Disabled = 0,
    Solved = 1,
    InvalidInput = -1,
    NotConverged = -2,
    NonFinite = -3,
};

struct Config {
    bool enabled = false;
    double dt_s = 0.01;
    int horizon_steps = 30;
    double q_angle = 100.0;
    double q_velocity = 1.0;
    double r_acceleration = 0.1;
    double max_yaw_acc_rad_s2 = 50.0;
    double max_pitch_acc_rad_s2 = 100.0;
    double feedback_timeout_s = 0.05;
    bool use_feedback_velocity = false;
};

struct FeedbackState {
    double yaw_rad = 0.0;
    double pitch_rad = 0.0;
    std::optional<double> yaw_velocity_rad_s;
    std::optional<double> pitch_velocity_rad_s;
};

struct PlannedCommand {
    bool valid = false;
    double reference_yaw = 0.0;
    double reference_pitch = 0.0;
    double yaw = 0.0;
    double pitch = 0.0;
    double yaw_vel = 0.0;
    double pitch_vel = 0.0;
    double yaw_acc = 0.0;
    double pitch_acc = 0.0;
    SolverStatus solver_status = SolverStatus::InvalidInput;
};

class MpcPlanner {
public:
    explicit MpcPlanner(Config config);
    PlannedCommand plan(const sp_core::AimResult& reference,
                        const FeedbackState& feedback,
                        double dt_s);
    const Config& config() const noexcept { return config_; }

private:
    Config config_;
    std::optional<FeedbackState> previous_feedback_;
};

sp_core::AimResult selectAim(const sp_core::AimResult& geometric,
                             const PlannedCommand& planned) noexcept;
double wrapAngle(double angle) noexcept;

}  // namespace planner
