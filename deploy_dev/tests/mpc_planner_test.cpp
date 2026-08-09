#include "planner/mpc_planner.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

sp_core::AimResult reference(double yaw, double pitch) {
    sp_core::AimResult result;
    result.valid = true;
    result.yaw = yaw;
    result.pitch = pitch;
    return result;
}

planner::Config enabledConfig() {
    planner::Config config;
    config.enabled = true;
    config.dt_s = 0.01;
    config.horizon_steps = 20;
    config.max_yaw_acc_rad_s2 = 5.0;
    config.max_pitch_acc_rad_s2 = 7.0;
    return config;
}

}  // namespace

int main() {
    planner::FeedbackState feedback;
    feedback.yaw_velocity_rad_s = 0.0;
    feedback.pitch_velocity_rad_s = 0.0;

    planner::MpcPlanner static_planner(enabledConfig());
    auto plan = static_planner.plan(reference(0.0, 0.0), feedback, 0.01);
    require(plan.valid && plan.solver_status == planner::SolverStatus::Solved,
            "static reference solves");
    require(std::abs(plan.yaw) < 1e-12 && std::abs(plan.pitch) < 1e-12,
            "static command remains static");

    plan = static_planner.plan(reference(1.0, -0.5), feedback, 0.01);
    require(plan.valid, "step reference solves");
    require(plan.yaw > 0.0 && plan.yaw < 1.0 && plan.pitch < 0.0 && plan.pitch > -0.5,
            "step emits first executable point rather than reference endpoint");
    require(std::abs(plan.yaw_acc) <= 5.0 + 1e-9 &&
            std::abs(plan.pitch_acc) <= 7.0 + 1e-9,
            "acceleration constraints hold");

    planner::MpcPlanner wrap_planner(enabledConfig());
    feedback.yaw_rad = 3.13;
    plan = wrap_planner.plan(reference(-3.13, 0.0), feedback, 0.01);
    require(plan.valid && plan.yaw_acc > 0.0, "yaw wrap takes the short positive path");

    planner::MpcPlanner nonfinite_planner(enabledConfig());
    feedback = {};
    feedback.yaw_velocity_rad_s = std::numeric_limits<double>::quiet_NaN();
    plan = nonfinite_planner.plan(reference(0.5, 0.0), feedback, 0.01);
    require(!plan.valid && plan.solver_status == planner::SolverStatus::InvalidInput,
            "non-finite feedback is rejected");
    const auto geometric = reference(0.5, -0.2);
    const auto fallback = planner::selectAim(geometric, plan);
    require(fallback.yaw == geometric.yaw && fallback.pitch == geometric.pitch,
            "invalid or missing-feedback plan falls back to Aimer");

    planner::MpcPlanner disabled(planner::Config{});
    feedback = {};
    plan = disabled.plan(geometric, feedback, 0.01);
    require(!plan.valid && plan.solver_status == planner::SolverStatus::Disabled,
            "planner defaults disabled");
    return 0;
}
