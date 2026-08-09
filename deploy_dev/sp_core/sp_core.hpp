#pragma once

#include "shared/types.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace sp_core {

enum class Color { RED, BLUE, EXTINGUISH, PURPLE };
enum class ArmorType { SMALL, LARGE };
enum class ArmorName { SENTRY, ONE, TWO, THREE, FOUR, FIVE, OUTPOST, BASE, UNKNOWN };

struct Armor {
    Color color = Color::RED;
    ArmorType type = ArmorType::SMALL;
    ArmorName name = ArmorName::UNKNOWN;
    int class_id = -1;
    float confidence = 0.0F;
    cv::Rect box;
    std::vector<cv::Point2f> points;
    Eigen::Vector3d xyz_camera = Eigen::Vector3d::Zero();
    Eigen::Vector3d xyz_gimbal = Eigen::Vector3d::Zero();
    Eigen::Vector3d xyz_world = Eigen::Vector3d::Zero();
    double yaw_world = 0.0;
    double reprojection_error = 0.0;
    bool pnp_valid = false;
};

Armor adapt(const hnu25::DetectedArmor& detected);

struct SolverConfig {
    Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
    std::vector<double> distortion{0.0, 0.0, 0.0, 0.0, 0.0};
    Eigen::Matrix3d r_camera_to_gimbal = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_camera_to_gimbal = Eigen::Vector3d::Zero();
    Eigen::Matrix3d r_gimbal_to_imu_body = Eigen::Matrix3d::Identity();
    double max_reprojection_error_px = 5.0;
};

class Solver {
public:
    explicit Solver(const std::string& config_path);
    explicit Solver(const SolverConfig& config);
    void setImuQuaternion(const Eigen::Quaterniond& q);
    bool solve(Armor& armor) const;

private:
    cv::Mat camera_matrix_;
    cv::Mat distortion_;
    Eigen::Matrix3d r_camera_to_gimbal_ = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t_camera_to_gimbal_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d r_gimbal_to_imu_body_ = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d r_gimbal_to_world_ = Eigen::Matrix3d::Identity();
    double max_reprojection_error_px_ = 5.0;
};

struct TrackerConfig {
    Color enemy_color = Color::BLUE;
    int min_detect_count = 5;
    int max_temp_lost_count = 15;
    int outpost_max_temp_lost_count = 75;
    double max_dt = 0.1;
    double nis_threshold = 13.277;
    double nis_failure_ratio = 0.4;
    int nis_window = 100;
    double process_accel_variance = 100.0;
    double process_yaw_accel_variance = 400.0;
    Eigen::Vector4d measurement_variance{0.01, 0.01, 0.01, 0.09};
    cv::Point2d image_center{720.0, 540.0};
};

class Target {
public:
    Target() = default;
    Target(const Armor& armor, std::chrono::steady_clock::time_point timestamp,
           const TrackerConfig& config);

    void predict(std::chrono::steady_clock::time_point timestamp);
    void predict(double dt);
    bool update(const Armor& armor);
    double associationCost(const Armor& armor) const;
    std::vector<Eigen::Vector4d> armorStates() const;
    bool diverged() const;
    const Eigen::Matrix<double, 11, 1>& state() const { return x_; }
    double lastNis() const { return last_nis_; }
    double nisFailureRatio() const;
    std::size_t nisSampleCount() const { return nis_failures_.size(); }
    ArmorName name() const { return name_; }
    ArmorType type() const { return type_; }
    bool jumped() const { return jumped_; }

private:
    Eigen::Vector3d armorPosition(const Eigen::Matrix<double, 11, 1>& x, int id) const;
    Eigen::Matrix<double, 4, 11> measurementJacobian(int id) const;
    int armor_num_ = 4;
    ArmorName name_ = ArmorName::UNKNOWN;
    ArmorType type_ = ArmorType::SMALL;
    bool jumped_ = false;
    int last_id_ = 0;
    TrackerConfig config_;
    Eigen::Matrix<double, 11, 1> x_ = Eigen::Matrix<double, 11, 1>::Zero();
    Eigen::Matrix<double, 11, 11> p_ = Eigen::Matrix<double, 11, 11>::Identity();
    std::chrono::steady_clock::time_point timestamp_{};
    double last_nis_ = 0.0;
    std::deque<bool> nis_failures_;
};

enum class TrackState { LOST, DETECTING, TRACKING, TEMP_LOST };

class Tracker {
public:
    explicit Tracker(const std::string& config_path);
    explicit Tracker(const TrackerConfig& config);
    void reset();
    std::optional<Target> update(const std::vector<Armor>& armors,
                                 std::chrono::steady_clock::time_point timestamp);
    TrackState state() const { return state_; }
    const char* stateName() const;
    double lastNis() const { return target_ ? target_->lastNis() : 0.0; }
    double nisFailureRatio() const { return target_ ? target_->nisFailureRatio() : 0.0; }

private:
    void transition(bool found);
    TrackerConfig config_;
    TrackState state_ = TrackState::LOST;
    int detect_count_ = 0;
    int temp_lost_count_ = 0;
    std::optional<Target> target_;
    std::optional<std::chrono::steady_clock::time_point> last_timestamp_;
};

struct AimResult {
    bool valid = false;
    bool fire = false;
    double yaw = 0.0;
    double pitch = 0.0;
    double fly_time = 0.0;
    int iterations = 0;
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
};

struct TrajectoryResult {
    bool valid = false;
    double pitch = 0.0;
    double fly_time = 0.0;
};

TrajectoryResult solveTrajectory(double speed, double distance, double height);

struct AimerConfig {
    double yaw_offset = 0.0;
    double pitch_offset = 0.0;
    double coming_angle = 60.0;
    double leaving_angle = 20.0;
    double decision_speed = 8.0;
    double high_speed_delay = 0.03;
    double low_speed_delay = 0.015;
};

class Aimer {
public:
    explicit Aimer(const std::string& config_path);
    explicit Aimer(const AimerConfig& config);
    AimResult aim(const Target& target, double bullet_speed) const;

private:
    std::optional<Eigen::Vector4d> chooseAimPoint(const Target& target) const;
    double yaw_offset_ = 0.0;
    double pitch_offset_ = 0.0;
    double coming_angle_ = 0.0;
    double leaving_angle_ = 0.0;
    double decision_speed_ = 8.0;
    double high_speed_delay_ = 0.03;
    double low_speed_delay_ = 0.015;
};

}  // namespace sp_core
