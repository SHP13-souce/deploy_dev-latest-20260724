#include "sp_core.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace sp_core {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kGravity = 9.7833;

double wrap(double angle) {
    return std::remainder(angle, 2.0 * kPi);
}

template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor> matrixFromYaml(
    const YAML::Node& yaml, const char* key) {
    const auto values = yaml[key].as<std::vector<double>>();
    if (values.size() != static_cast<std::size_t>(Rows * Cols)) {
        throw std::runtime_error(std::string(key) + " has an invalid size");
    }
    return Eigen::Map<const Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor>>(values.data());
}

Color colorFromString(const std::string& value) {
    return value == "red" ? Color::RED : Color::BLUE;
}

TrajectoryResult trajectory(double speed, double distance, double height) {
    TrajectoryResult result;
    if (!std::isfinite(speed) || !std::isfinite(distance) || !std::isfinite(height) ||
        speed <= 0.0 || distance <= 1e-6) return result;
    const double a = kGravity * distance * distance / (2.0 * speed * speed);
    const double discriminant = distance * distance - 4.0 * a * (a + height);
    if (discriminant < 0.0) return result;
    const double tan1 = (distance + std::sqrt(discriminant)) / (2.0 * a);
    const double tan2 = (distance - std::sqrt(discriminant)) / (2.0 * a);
    const double pitch1 = std::atan(tan1);
    const double pitch2 = std::atan(tan2);
    const double time1 = distance / (speed * std::cos(pitch1));
    const double time2 = distance / (speed * std::cos(pitch2));
    if (time1 < time2) {
        result.pitch = pitch1;
        result.fly_time = time1;
    } else {
        result.pitch = pitch2;
        result.fly_time = time2;
    }
    result.valid = std::isfinite(result.pitch) && std::isfinite(result.fly_time);
    return result;
}

}  // namespace

TrajectoryResult solveTrajectory(double speed, double distance, double height) {
    return trajectory(speed, distance, height);
}

Armor adapt(const hnu25::DetectedArmor& detected) {
    Armor armor;
    switch (detected.color) {
        case hnu25::Color::RED: armor.color = Color::RED; break;
        case hnu25::Color::BLUE: armor.color = Color::BLUE; break;
        case hnu25::Color::PURPLE: armor.color = Color::PURPLE; break;
        default: armor.color = Color::EXTINGUISH; break;
    }
    armor.type = detected.kind == hnu25::ArmorKind::LARGE ? ArmorType::LARGE : ArmorType::SMALL;
    switch (detected.label) {
        case hnu25::ArmorLabel::SENTRY: armor.name = ArmorName::SENTRY; break;
        case hnu25::ArmorLabel::ONE: armor.name = ArmorName::ONE; break;
        case hnu25::ArmorLabel::TWO: armor.name = ArmorName::TWO; break;
        case hnu25::ArmorLabel::THREE: armor.name = ArmorName::THREE; break;
        case hnu25::ArmorLabel::FOUR: armor.name = ArmorName::FOUR; break;
        case hnu25::ArmorLabel::FIVE: armor.name = ArmorName::FIVE; break;
        case hnu25::ArmorLabel::OUTPOST: armor.name = ArmorName::OUTPOST; break;
        case hnu25::ArmorLabel::BASE: armor.name = ArmorName::BASE; break;
        default: armor.name = ArmorName::UNKNOWN; break;
    }
    armor.class_id = detected.class_id;
    armor.confidence = detected.confidence;
    armor.box = detected.box;
    armor.points = detected.keypoints;
    return armor;
}

Solver::Solver(const std::string& config_path) {
    const YAML::Node yaml = YAML::LoadFile(config_path);
    const auto camera = matrixFromYaml<3, 3>(yaml, "camera_matrix");
    camera_matrix_ = cv::Mat(3, 3, CV_64F);
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) camera_matrix_.at<double>(row, col) = camera(row, col);
    const auto distortion = yaml["distort_coeffs"].as<std::vector<double>>();
    distortion_ = cv::Mat(1, static_cast<int>(distortion.size()), CV_64F);
    for (std::size_t i = 0; i < distortion.size(); ++i)
        distortion_.at<double>(0, static_cast<int>(i)) = distortion[i];
    r_camera_to_gimbal_ = matrixFromYaml<3, 3>(yaml, "R_camera2gimbal");
    r_gimbal_to_imu_body_ = matrixFromYaml<3, 3>(yaml, "R_gimbal2imubody");
    const auto translation = yaml["t_camera2gimbal"].as<std::vector<double>>();
    if (translation.size() != 3) throw std::runtime_error("t_camera2gimbal has an invalid size");
    t_camera_to_gimbal_ = Eigen::Map<const Eigen::Vector3d>(translation.data());
    max_reprojection_error_px_ = yaml["max_reprojection_error_px"].as<double>(5.0);
    if (!(max_reprojection_error_px_ > 0.0)) {
        throw std::runtime_error("max_reprojection_error_px must be positive");
    }
}

Solver::Solver(const SolverConfig& config)
    : camera_matrix_(3, 3, CV_64F),
      distortion_(1, static_cast<int>(config.distortion.size()), CV_64F),
      r_camera_to_gimbal_(config.r_camera_to_gimbal),
      t_camera_to_gimbal_(config.t_camera_to_gimbal),
      r_gimbal_to_imu_body_(config.r_gimbal_to_imu_body),
      max_reprojection_error_px_(config.max_reprojection_error_px) {
    if (config.distortion.empty()) throw std::runtime_error("distortion must not be empty");
    if (!(max_reprojection_error_px_ > 0.0)) {
        throw std::runtime_error("max_reprojection_error_px must be positive");
    }
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col)
            camera_matrix_.at<double>(row, col) = config.camera_matrix(row, col);
    for (std::size_t i = 0; i < config.distortion.size(); ++i)
        distortion_.at<double>(0, static_cast<int>(i)) = config.distortion[i];
}

void Solver::setImuQuaternion(const Eigen::Quaterniond& q) {
    const Eigen::Quaterniond normalized = q.normalized();
    r_gimbal_to_world_ = r_gimbal_to_imu_body_.transpose() * normalized.toRotationMatrix() *
                         r_gimbal_to_imu_body_;
}

bool Solver::solve(Armor& armor) const {
    armor.pnp_valid = false;
    if (armor.points.size() != 4) return false;
    const auto midpoint = [&](int a, int b) { return (armor.points[a] + armor.points[b]) * 0.5F; };
    const double cross = (armor.points[1].x - armor.points[0].x) *
                             (armor.points[2].y - armor.points[1].y) -
                         (armor.points[1].y - armor.points[0].y) *
                             (armor.points[2].x - armor.points[1].x);
    if (!(cross > 0.0) || !(midpoint(0, 1).y < midpoint(2, 3).y) ||
        !(midpoint(0, 3).x < midpoint(1, 2).x)) return false;
    const double width = armor.type == ArmorType::LARGE ? 0.230 : 0.135;
    constexpr double height = 0.056;
    const std::vector<cv::Point3f> object_points{
        {0.0F, static_cast<float>(width / 2), static_cast<float>(height / 2)},
        {0.0F, static_cast<float>(-width / 2), static_cast<float>(height / 2)},
        {0.0F, static_cast<float>(-width / 2), static_cast<float>(-height / 2)},
        {0.0F, static_cast<float>(width / 2), static_cast<float>(-height / 2)}};
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool success = false;
    try {
        success = cv::solvePnP(object_points, armor.points, camera_matrix_, distortion_, rvec, tvec,
                               false, cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception&) {
        return false;
    }
    if (!success || !cv::checkRange(tvec)) return false;

    armor.xyz_camera = {tvec[0], tvec[1], tvec[2]};
    armor.xyz_gimbal = r_camera_to_gimbal_ * armor.xyz_camera + t_camera_to_gimbal_;
    armor.xyz_world = r_gimbal_to_world_ * armor.xyz_gimbal;
    cv::Mat rotation_cv;
    cv::Rodrigues(rvec, rotation_cv);
    Eigen::Matrix3d rotation;
    for (int row = 0; row < 3; ++row)
        for (int col = 0; col < 3; ++col) rotation(row, col) = rotation_cv.at<double>(row, col);
    const Eigen::Matrix3d armor_to_world = r_gimbal_to_world_ * r_camera_to_gimbal_ * rotation;
    armor.yaw_world = wrap(std::atan2(armor_to_world(1, 0), armor_to_world(0, 0)));

    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distortion_, projected);
    double squared_error = 0.0;
    for (std::size_t i = 0; i < projected.size(); ++i) {
        const double error = cv::norm(projected[i] - armor.points[i]);
        squared_error += error * error;
    }
    armor.reprojection_error = std::sqrt(squared_error / projected.size());
    armor.pnp_valid = armor.xyz_camera.allFinite() && armor.xyz_camera.z() > 0.0 &&
                       std::isfinite(armor.reprojection_error) &&
                       armor.reprojection_error <= max_reprojection_error_px_;
    return armor.pnp_valid;
}

Target::Target(const Armor& armor, std::chrono::steady_clock::time_point timestamp,
               const TrackerConfig& config)
    : name_(armor.name), type_(armor.type), config_(config), timestamp_(timestamp) {
    double radius = 0.2;
    armor_num_ = 4;
    if (name_ == ArmorName::OUTPOST) { radius = 0.2765; armor_num_ = 3; }
    if (name_ == ArmorName::BASE) { radius = 0.3205; armor_num_ = 3; }
    if (type_ == ArmorType::LARGE &&
        (name_ == ArmorName::THREE || name_ == ArmorName::FOUR || name_ == ArmorName::FIVE)) {
        armor_num_ = 2;
    }
    x_ << armor.xyz_world.x() + radius * std::cos(armor.yaw_world), 0.0,
          armor.xyz_world.y() + radius * std::sin(armor.yaw_world), 0.0,
          armor.xyz_world.z(), 0.0, armor.yaw_world, 0.0, radius, 0.0, 0.0;
    Eigen::Matrix<double, 11, 1> diagonal;
    diagonal << 1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1;
    if (name_ == ArmorName::OUTPOST || name_ == ArmorName::BASE) {
        diagonal[8] = 1e-4;
        diagonal[9] = 0.0;
        diagonal[10] = 0.0;
    }
    p_ = diagonal.asDiagonal();
}

void Target::predict(std::chrono::steady_clock::time_point timestamp) {
    double dt = std::chrono::duration<double>(timestamp - timestamp_).count();
    predict(std::clamp(dt, 0.0, config_.max_dt));
    timestamp_ = timestamp;
}

void Target::predict(double dt) {
    Eigen::Matrix<double, 11, 11> f = Eigen::Matrix<double, 11, 11>::Identity();
    f(0, 1) = f(2, 3) = f(4, 5) = f(6, 7) = dt;
    x_ = f * x_;
    x_[6] = wrap(x_[6]);
    Eigen::Matrix<double, 11, 11> q = Eigen::Matrix<double, 11, 11>::Zero();
    const double accel = name_ == ArmorName::OUTPOST ? 10.0 : config_.process_accel_variance;
    const double yaw_accel = name_ == ArmorName::OUTPOST ? 0.1 : config_.process_yaw_accel_variance;
    const auto add_noise = [&](int position, double variance) {
        q(position, position) = dt * dt * dt * dt * variance / 4.0;
        q(position, position + 1) = q(position + 1, position) = dt * dt * dt * variance / 2.0;
        q(position + 1, position + 1) = dt * dt * variance;
    };
    add_noise(0, accel);
    add_noise(2, accel);
    add_noise(4, accel);
    add_noise(6, yaw_accel);
    p_ = f * p_ * f.transpose() + q;
    p_ = 0.5 * (p_ + p_.transpose());
}

Eigen::Vector3d Target::armorPosition(const Eigen::Matrix<double, 11, 1>& x, int id) const {
    const double angle = wrap(x[6] + id * 2.0 * kPi / armor_num_);
    const bool alternate = armor_num_ == 4 && (id == 1 || id == 3);
    const double radius = x[8] + (alternate ? x[9] : 0.0);
    return {x[0] - radius * std::cos(angle), x[2] - radius * std::sin(angle),
            x[4] + (alternate ? x[10] : 0.0)};
}

Eigen::Matrix<double, 4, 11> Target::measurementJacobian(int id) const {
    Eigen::Matrix<double, 4, 11> h = Eigen::Matrix<double, 4, 11>::Zero();
    const double angle = wrap(x_[6] + id * 2.0 * kPi / armor_num_);
    const bool alternate = armor_num_ == 4 && (id == 1 || id == 3);
    const double radius = x_[8] + (alternate ? x_[9] : 0.0);
    h(0, 0) = h(1, 2) = h(2, 4) = h(3, 6) = 1.0;
    h(0, 6) = radius * std::sin(angle);
    h(1, 6) = -radius * std::cos(angle);
    h(0, 8) = -std::cos(angle);
    h(1, 8) = -std::sin(angle);
    if (alternate) {
        h(0, 9) = -std::cos(angle);
        h(1, 9) = -std::sin(angle);
        h(2, 10) = 1.0;
    }
    return h;
}

bool Target::update(const Armor& armor) {
    int best_id = 0;
    double best_cost = std::numeric_limits<double>::max();
    for (int id = 0; id < armor_num_; ++id) {
        const double expected_yaw = wrap(x_[6] + id * 2.0 * kPi / armor_num_);
        const double cost = (armorPosition(x_, id) - armor.xyz_world).norm() +
                            std::abs(wrap(armor.yaw_world - expected_yaw));
        if (cost < best_cost) { best_cost = cost; best_id = id; }
    }
    Eigen::Vector4d z;
    z << armor.xyz_world, armor.yaw_world;
    Eigen::Vector4d predicted;
    predicted << armorPosition(x_, best_id), wrap(x_[6] + best_id * 2.0 * kPi / armor_num_);
    Eigen::Vector4d residual = z - predicted;
    residual[3] = wrap(residual[3]);
    const auto h = measurementJacobian(best_id);
    const Eigen::Matrix4d r = config_.measurement_variance.asDiagonal();
    const Eigen::Matrix4d s = h * p_ * h.transpose() + r;
    const Eigen::LDLT<Eigen::Matrix4d> ldlt(s);
    if (ldlt.info() != Eigen::Success || !ldlt.isPositive()) return false;
    last_nis_ = residual.dot(ldlt.solve(residual));
    const bool gated = !std::isfinite(last_nis_) || last_nis_ > config_.nis_threshold;
    nis_failures_.push_back(gated);
    while (nis_failures_.size() > static_cast<std::size_t>(config_.nis_window)) nis_failures_.pop_front();
    if (gated) return false;

    const Eigen::Matrix<double, 11, 4> pht = p_ * h.transpose();
    const Eigen::Matrix<double, 11, 4> gain = ldlt.solve(pht.transpose()).transpose();
    x_ += gain * residual;
    x_[6] = wrap(x_[6]);
    const Eigen::Matrix<double, 11, 11> identity = Eigen::Matrix<double, 11, 11>::Identity();
    const auto correction = identity - gain * h;
    p_ = correction * p_ * correction.transpose() + gain * r * gain.transpose();
    p_ = 0.5 * (p_ + p_.transpose());
    jumped_ = jumped_ || best_id != 0;
    last_id_ = best_id;
    return true;
}

std::vector<Eigen::Vector4d> Target::armorStates() const {
    std::vector<Eigen::Vector4d> armors;
    armors.reserve(armor_num_);
    for (int id = 0; id < armor_num_; ++id) {
        Eigen::Vector4d value;
        value << armorPosition(x_, id), wrap(x_[6] + id * 2.0 * kPi / armor_num_);
        armors.push_back(value);
    }
    return armors;
}

bool Target::diverged() const {
    return !x_.allFinite() || x_[8] <= 0.05 || x_[8] >= 0.5 ||
           x_[8] + x_[9] <= 0.05 || x_[8] + x_[9] >= 0.5;
}

double Target::nisFailureRatio() const {
    if (nis_failures_.empty()) return 0.0;
    return static_cast<double>(std::count(nis_failures_.begin(), nis_failures_.end(), true)) /
           nis_failures_.size();
}

double Target::associationCost(const Armor& armor) const {
    double best_cost = std::numeric_limits<double>::max();
    for (int id = 0; id < armor_num_; ++id) {
        const double expected_yaw = wrap(x_[6] + id * 2.0 * kPi / armor_num_);
        const double cost = (armorPosition(x_, id) - armor.xyz_world).norm() +
                            std::abs(wrap(armor.yaw_world - expected_yaw));
        best_cost = std::min(best_cost, cost);
    }
    return best_cost;
}

void validateTrackerConfig(const TrackerConfig& config) {
    if (config.min_detect_count <= 0 || config.max_temp_lost_count < 0 ||
        config.outpost_max_temp_lost_count < 0) {
        throw std::runtime_error("tracker frame counts are invalid");
    }
    if (!(config.max_dt > 0.0) || !(config.nis_threshold > 0.0) ||
        config.nis_window <= 0 || config.nis_failure_ratio < 0.0 ||
        config.nis_failure_ratio > 1.0 || !(config.process_accel_variance >= 0.0) ||
        !(config.process_yaw_accel_variance >= 0.0) ||
        !config.measurement_variance.allFinite() ||
        (config.measurement_variance.array() <= 0.0).any()) {
        throw std::runtime_error("tracker covariance or gate configuration is invalid");
    }
}

Tracker::Tracker(const std::string& config_path) {
    const YAML::Node yaml = YAML::LoadFile(config_path);
    config_.enemy_color = colorFromString(yaml["enemy_color"].as<std::string>("blue"));
    config_.min_detect_count = yaml["min_detect_count"].as<int>(5);
    config_.max_temp_lost_count = yaml["max_temp_lost_count"].as<int>(15);
    config_.outpost_max_temp_lost_count = yaml["outpost_max_temp_lost_count"].as<int>(75);
    config_.max_dt = yaml["max_tracker_dt"].as<double>(0.1);
    config_.nis_threshold = yaml["nis_threshold"].as<double>(13.277);
    config_.nis_failure_ratio = yaml["nis_failure_ratio"].as<double>(0.4);
    config_.nis_window = yaml["nis_window"].as<int>(100);
    config_.process_accel_variance = yaml["process_accel_variance"].as<double>(100.0);
    config_.process_yaw_accel_variance = yaml["process_yaw_accel_variance"].as<double>(400.0);
    if (yaml["measurement_variance"]) {
        const auto values = yaml["measurement_variance"].as<std::vector<double>>();
        if (values.size() != 4) throw std::runtime_error("measurement_variance has an invalid size");
        config_.measurement_variance = Eigen::Map<const Eigen::Vector4d>(values.data());
    }
    const auto camera_values = yaml["camera_matrix"].as<std::vector<double>>(std::vector<double>{});
    if (camera_values.size() == 9) config_.image_center = {camera_values[2], camera_values[5]};
    validateTrackerConfig(config_);
}

Tracker::Tracker(const TrackerConfig& config) : config_(config) {
    validateTrackerConfig(config_);
}

void Tracker::reset() {
    state_ = TrackState::LOST;
    detect_count_ = 0;
    temp_lost_count_ = 0;
    target_.reset();
    last_timestamp_.reset();
}

std::optional<Target> Tracker::update(const std::vector<Armor>& armors,
                                      std::chrono::steady_clock::time_point timestamp) {
    if (last_timestamp_ && state_ != TrackState::LOST) {
        const double dt = std::chrono::duration<double>(timestamp - *last_timestamp_).count();
        if (!(dt > 0.0) || dt > config_.max_dt) reset();
    }
    last_timestamp_ = timestamp;
    std::vector<const Armor*> candidates;
    for (const auto& armor : armors)
        if (armor.pnp_valid && armor.color == config_.enemy_color) candidates.push_back(&armor);
    std::sort(candidates.begin(), candidates.end(), [&](const Armor* a, const Armor* b) {
        const cv::Point2d center = config_.image_center;
        const cv::Point2d a_center(a->box.x + a->box.width * 0.5,
                                   a->box.y + a->box.height * 0.5);
        const cv::Point2d b_center(b->box.x + b->box.width * 0.5,
                                   b->box.y + b->box.height * 0.5);
        return cv::norm(a_center - center) < cv::norm(b_center - center);
    });

    bool found = false;
    if (state_ == TrackState::LOST) {
        if (!candidates.empty()) {
            target_.emplace(*candidates.front(), timestamp, config_);
            found = true;
        }
    } else if (target_) {
        target_->predict(timestamp);
        const Armor* best = nullptr;
        double best_cost = std::numeric_limits<double>::max();
        for (const Armor* armor : candidates) {
            if (armor->name != target_->name() || armor->type != target_->type()) continue;
            const double cost = target_->associationCost(*armor);
            if (cost < best_cost) { best_cost = cost; best = armor; }
        }
        if (best) found = target_->update(*best);
    }
    transition(found);
    if (target_ && (target_->diverged() ||
        (target_->nisSampleCount() >= static_cast<std::size_t>(config_.nis_window) &&
         target_->nisFailureRatio() >= config_.nis_failure_ratio &&
         target_->nisFailureRatio() > 0.0 && state_ == TrackState::TEMP_LOST))) {
        state_ = TrackState::LOST;
        target_.reset();
    }
    return state_ == TrackState::LOST ? std::nullopt : target_;
}

void Tracker::transition(bool found) {
    switch (state_) {
        case TrackState::LOST:
            if (found) { state_ = TrackState::DETECTING; detect_count_ = 1; }
            break;
        case TrackState::DETECTING:
            if (!found) { state_ = TrackState::LOST; target_.reset(); detect_count_ = 0; }
            else if (++detect_count_ >= config_.min_detect_count) state_ = TrackState::TRACKING;
            break;
        case TrackState::TRACKING:
            if (!found) { state_ = TrackState::TEMP_LOST; temp_lost_count_ = 1; }
            break;
        case TrackState::TEMP_LOST:
            if (found) state_ = TrackState::TRACKING;
            else {
                ++temp_lost_count_;
                const int limit = target_ && target_->name() == ArmorName::OUTPOST
                                      ? config_.outpost_max_temp_lost_count
                                      : config_.max_temp_lost_count;
                if (temp_lost_count_ > limit) { state_ = TrackState::LOST; target_.reset(); }
            }
            break;
    }
}

const char* Tracker::stateName() const {
    switch (state_) {
        case TrackState::LOST: return "lost";
        case TrackState::DETECTING: return "detecting";
        case TrackState::TRACKING: return "tracking";
        case TrackState::TEMP_LOST: return "temp_lost";
    }
    return "unknown";
}

Aimer::Aimer(const std::string& config_path) {
    const YAML::Node yaml = YAML::LoadFile(config_path);
    constexpr double degrees = kPi / 180.0;
    yaw_offset_ = yaml["yaw_offset"].as<double>(0.0) * degrees;
    pitch_offset_ = yaml["pitch_offset"].as<double>(0.0) * degrees;
    coming_angle_ = yaml["comming_angle"].as<double>(60.0) * degrees;
    leaving_angle_ = yaml["leaving_angle"].as<double>(20.0) * degrees;
    decision_speed_ = yaml["decision_speed"].as<double>(8.0);
    high_speed_delay_ = yaml["high_speed_delay_time"].as<double>(0.03);
    low_speed_delay_ = yaml["low_speed_delay_time"].as<double>(0.015);
}

Aimer::Aimer(const AimerConfig& config) {
    constexpr double degrees = kPi / 180.0;
    yaw_offset_ = config.yaw_offset * degrees;
    pitch_offset_ = config.pitch_offset * degrees;
    coming_angle_ = config.coming_angle * degrees;
    leaving_angle_ = config.leaving_angle * degrees;
    decision_speed_ = config.decision_speed;
    high_speed_delay_ = config.high_speed_delay;
    low_speed_delay_ = config.low_speed_delay;
}

std::optional<Eigen::Vector4d> Aimer::chooseAimPoint(const Target& target) const {
    const auto armors = target.armorStates();
    if (armors.empty()) return std::nullopt;
    const auto& x = target.state();
    const double center_yaw = std::atan2(x[2], x[0]);
    const double speed = x[7];
    const bool spinning = std::abs(speed) > 2.0 || target.name() == ArmorName::OUTPOST;
    int best = -1;
    double best_angle = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < armors.size(); ++i) {
        const double angle = wrap(armors[i][3] - center_yaw);
        if (!spinning) {
            if (std::abs(angle) <= 60.0 * kPi / 180.0 && std::abs(angle) < best_angle) {
                best = static_cast<int>(i); best_angle = std::abs(angle);
            }
        } else {
            const double coming = target.name() == ArmorName::OUTPOST ? 70.0 * kPi / 180.0 : coming_angle_;
            const double leaving = target.name() == ArmorName::OUTPOST ? 30.0 * kPi / 180.0 : leaving_angle_;
            if (std::abs(angle) <= coming && ((speed > 0.0 && angle < leaving) ||
                                               (speed < 0.0 && angle > -leaving))) return armors[i];
        }
    }
    if (best >= 0) return armors[best];
    return std::nullopt;
}

AimResult Aimer::aim(const Target& measured_target, double bullet_speed) const {
    AimResult result;
    if (!std::isfinite(bullet_speed) || bullet_speed < 14.0) return result;
    Target target = measured_target;
    const double delay = std::abs(target.state()[7]) > decision_speed_
                             ? high_speed_delay_ : low_speed_delay_;
    target.predict(delay);
    auto point = chooseAimPoint(target);
    if (!point) return result;
    double fly_time = 0.0;
    double pitch = 0.0;
    for (int iteration = 1; iteration <= 10; ++iteration) {
        Target predicted = target;
        predicted.predict(fly_time);
        point = chooseAimPoint(predicted);
        if (!point) return result;
        const Eigen::Vector3d xyz = point->head<3>();
        const double distance = std::hypot(xyz.x(), xyz.y());
        const auto trajectory_result = trajectory(bullet_speed, distance, xyz.z());
        if (!trajectory_result.valid) return result;
        pitch = trajectory_result.pitch;
        const double next_time = trajectory_result.fly_time;
        result.iterations = iteration;
        result.point = xyz;
        if (std::abs(next_time - fly_time) < 0.001) { fly_time = next_time; break; }
        fly_time = next_time;
    }
    result.valid = true;
    // Firing is a control-layer decision. Aimer only returns a geometric solution.
    result.fire = false;
    result.fly_time = fly_time;
    result.yaw = std::atan2(result.point.y(), result.point.x()) + yaw_offset_;
    result.pitch = pitch + pitch_offset_;
    return result;
}

}  // namespace sp_core
