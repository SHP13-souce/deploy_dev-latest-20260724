#include "mpc_planner.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace planner {
namespace {

struct AxisResult {
    bool solved = false;
    double position = 0.0;
    double velocity = 0.0;
    double acceleration = 0.0;
    SolverStatus status = SolverStatus::NotConverged;
};

bool finiteConfig(const Config& config) {
    return std::isfinite(config.dt_s) && config.dt_s > 0.0 &&
           config.horizon_steps >= 2 && config.horizon_steps <= 500 &&
           std::isfinite(config.q_angle) && config.q_angle >= 0.0 &&
           std::isfinite(config.q_velocity) && config.q_velocity >= 0.0 &&
           std::isfinite(config.r_acceleration) && config.r_acceleration > 0.0 &&
           std::isfinite(config.max_yaw_acc_rad_s2) && config.max_yaw_acc_rad_s2 > 0.0 &&
           std::isfinite(config.max_pitch_acc_rad_s2) && config.max_pitch_acc_rad_s2 > 0.0 &&
           std::isfinite(config.feedback_timeout_s) && config.feedback_timeout_s > 0.0;
}

AxisResult solveAxis(double position, double velocity, double reference,
                     double dt, int horizon, double q_position,
                     double q_velocity, double r_acceleration,
                     double max_acceleration) {
    AxisResult result;
    if (!std::isfinite(position) || !std::isfinite(velocity) ||
        !std::isfinite(reference) || !std::isfinite(dt) || dt <= 0.0) {
        result.status = SolverStatus::InvalidInput;
        return result;
    }

    const int controls = horizon - 1;
    Eigen::MatrixXd influence = Eigen::MatrixXd::Zero(2 * controls, controls);
    Eigen::VectorXd free_error(2 * controls);
    for (int step = 1; step <= controls; ++step) {
        free_error[2 * (step - 1)] = position + step * dt * velocity - reference;
        free_error[2 * (step - 1) + 1] = velocity;
        for (int input = 0; input < step; ++input) {
            influence(2 * (step - 1), input) = (step - input - 0.5) * dt * dt;
            influence(2 * (step - 1) + 1, input) = dt;
        }
    }

    Eigen::VectorXd weights(2 * controls);
    for (int i = 0; i < controls; ++i) {
        weights[2 * i] = q_position;
        weights[2 * i + 1] = q_velocity;
    }
    const Eigen::MatrixXd hessian = influence.transpose() * weights.asDiagonal() * influence +
                                    r_acceleration * Eigen::MatrixXd::Identity(controls, controls);
    const Eigen::VectorXd linear = influence.transpose() * weights.asDiagonal() * free_error;
    if (!hessian.allFinite() || !linear.allFinite()) {
        result.status = SolverStatus::NonFinite;
        return result;
    }

    const double lipschitz = hessian.cwiseAbs().rowwise().sum().maxCoeff();
    if (!std::isfinite(lipschitz) || lipschitz <= 0.0) {
        result.status = SolverStatus::NonFinite;
        return result;
    }
    Eigen::VectorXd input = Eigen::VectorXd::Zero(controls);
    bool converged = false;
    for (int iteration = 0; iteration < 500; ++iteration) {
        const Eigen::VectorXd next = (input - (hessian * input + linear) / lipschitz)
                                         .cwiseMax(-max_acceleration)
                                         .cwiseMin(max_acceleration);
        if (!next.allFinite()) {
            result.status = SolverStatus::NonFinite;
            return result;
        }
        if ((next - input).lpNorm<Eigen::Infinity>() < 1e-7) {
            input = next;
            converged = true;
            break;
        }
        input = next;
    }
    if (!converged) return result;

    result.acceleration = input[0];
    result.position = position + velocity * dt + 0.5 * result.acceleration * dt * dt;
    result.velocity = velocity + result.acceleration * dt;
    result.solved = std::isfinite(result.position) && std::isfinite(result.velocity) &&
                    std::isfinite(result.acceleration);
    result.status = result.solved ? SolverStatus::Solved : SolverStatus::NonFinite;
    return result;
}

}  // namespace

double wrapAngle(double angle) noexcept {
    if (!std::isfinite(angle)) return angle;
    constexpr double pi = 3.14159265358979323846;
    return std::remainder(angle, 2.0 * pi);
}

MpcPlanner::MpcPlanner(Config config) : config_(config) {
    if (!finiteConfig(config_)) throw std::invalid_argument("invalid MPC planner configuration");
}

PlannedCommand MpcPlanner::plan(const sp_core::AimResult& reference,
                                const FeedbackState& feedback,
                                double dt_s) {
    PlannedCommand command;
    command.reference_yaw = reference.yaw;
    command.reference_pitch = reference.pitch;
    command.yaw = reference.yaw;
    command.pitch = reference.pitch;
    if (!config_.enabled) {
        command.solver_status = SolverStatus::Disabled;
        return command;
    }
    if (!reference.valid || !std::isfinite(reference.yaw) || !std::isfinite(reference.pitch) ||
        !std::isfinite(feedback.yaw_rad) || !std::isfinite(feedback.pitch_rad) ||
        (feedback.yaw_velocity_rad_s && !std::isfinite(*feedback.yaw_velocity_rad_s)) ||
        (feedback.pitch_velocity_rad_s && !std::isfinite(*feedback.pitch_velocity_rad_s)) ||
        !std::isfinite(dt_s) || dt_s <= 0.0) {
        command.solver_status = SolverStatus::InvalidInput;
        previous_feedback_.reset();
        return command;
    }

    auto velocity = [&](const std::optional<double>& supplied, double current,
                        const std::optional<double>& previous, bool yaw) {
        if (supplied && std::isfinite(*supplied)) return *supplied;
        if (!previous || !std::isfinite(*previous)) return 0.0;
        const double delta = yaw ? wrapAngle(current - *previous) : current - *previous;
        return delta / dt_s;
    };
    const std::optional<double> previous_yaw = previous_feedback_ ?
        std::optional<double>(previous_feedback_->yaw_rad) : std::nullopt;
    const std::optional<double> previous_pitch = previous_feedback_ ?
        std::optional<double>(previous_feedback_->pitch_rad) : std::nullopt;
    const double yaw_velocity = velocity(feedback.yaw_velocity_rad_s,
                                         feedback.yaw_rad, previous_yaw, true);
    const double pitch_velocity = velocity(feedback.pitch_velocity_rad_s,
                                           feedback.pitch_rad, previous_pitch, false);
    previous_feedback_ = feedback;

    const double yaw_reference = feedback.yaw_rad + wrapAngle(reference.yaw - feedback.yaw_rad);
    const auto yaw = solveAxis(feedback.yaw_rad, yaw_velocity, yaw_reference,
                               dt_s, config_.horizon_steps, config_.q_angle,
                               config_.q_velocity, config_.r_acceleration,
                               config_.max_yaw_acc_rad_s2);
    const auto pitch = solveAxis(feedback.pitch_rad, pitch_velocity, reference.pitch,
                                 dt_s, config_.horizon_steps, config_.q_angle,
                                 config_.q_velocity, config_.r_acceleration,
                                 config_.max_pitch_acc_rad_s2);
    if (!yaw.solved || !pitch.solved) {
        command.solver_status = yaw.solved ? pitch.status : yaw.status;
        return command;
    }

    command.valid = true;
    command.yaw = wrapAngle(yaw.position);
    command.pitch = pitch.position;
    command.yaw_vel = yaw.velocity;
    command.pitch_vel = pitch.velocity;
    command.yaw_acc = yaw.acceleration;
    command.pitch_acc = pitch.acceleration;
    command.solver_status = SolverStatus::Solved;
    return command;
}

sp_core::AimResult selectAim(const sp_core::AimResult& geometric,
                             const PlannedCommand& planned) noexcept {
    if (!planned.valid || !std::isfinite(planned.yaw) || !std::isfinite(planned.pitch)) return geometric;
    sp_core::AimResult selected = geometric;
    selected.yaw = planned.yaw;
    selected.pitch = planned.pitch;
    return selected;
}

}  // namespace planner
