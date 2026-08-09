#include "sp_core/sp_core.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string& message) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

sp_core::Armor trackedArmor(double x, double y = 0.0, double z = 0.2) {
    sp_core::Armor armor;
    armor.color = sp_core::Color::BLUE;
    armor.type = sp_core::ArmorType::SMALL;
    armor.name = sp_core::ArmorName::THREE;
    armor.xyz_world = {x, y, z};
    armor.yaw_world = 0.0;
    armor.pnp_valid = true;
    armor.box = {700, 520, 40, 40};
    return armor;
}

void testAdapt() {
    const std::vector<std::pair<hnu25::Color, sp_core::Color>> colors{
        {hnu25::Color::RED, sp_core::Color::RED},
        {hnu25::Color::BLUE, sp_core::Color::BLUE},
        {hnu25::Color::EXTINGUISH, sp_core::Color::EXTINGUISH},
        {hnu25::Color::PURPLE, sp_core::Color::PURPLE}};
    const std::vector<std::pair<hnu25::ArmorLabel, sp_core::ArmorName>> labels{
        {hnu25::ArmorLabel::SENTRY, sp_core::ArmorName::SENTRY},
        {hnu25::ArmorLabel::ONE, sp_core::ArmorName::ONE},
        {hnu25::ArmorLabel::TWO, sp_core::ArmorName::TWO},
        {hnu25::ArmorLabel::THREE, sp_core::ArmorName::THREE},
        {hnu25::ArmorLabel::FOUR, sp_core::ArmorName::FOUR},
        {hnu25::ArmorLabel::FIVE, sp_core::ArmorName::FIVE},
        {hnu25::ArmorLabel::OUTPOST, sp_core::ArmorName::OUTPOST},
        {hnu25::ArmorLabel::BASE, sp_core::ArmorName::BASE},
        {hnu25::ArmorLabel::UNKNOWN, sp_core::ArmorName::UNKNOWN}};
    for (const auto& [input, expected] : colors) {
        hnu25::DetectedArmor detected;
        detected.color = input;
        check(sp_core::adapt(detected).color == expected, "armor color mapping");
    }
    for (const auto& [input, expected] : labels) {
        hnu25::DetectedArmor detected;
        detected.label = input;
        check(sp_core::adapt(detected).name == expected, "armor label mapping");
    }
    hnu25::DetectedArmor detected;
    detected.kind = hnu25::ArmorKind::LARGE;
    detected.class_id = 11;
    detected.confidence = 0.875F;
    detected.box = {1, 2, 30, 40};
    detected.keypoints = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    const auto armor = sp_core::adapt(detected);
    check(armor.type == sp_core::ArmorType::LARGE, "large armor mapping");
    check(armor.class_id == 11 && armor.confidence == detected.confidence,
          "class and confidence mapping");
    check(armor.box == detected.box && armor.points == detected.keypoints,
          "box and keypoint mapping");
}

sp_core::SolverConfig solverConfig() {
    sp_core::SolverConfig config;
    config.camera_matrix << 900.0, 0.0, 640.0, 0.0, 900.0, 512.0, 0.0, 0.0, 1.0;
    config.max_reprojection_error_px = 0.5;
    return config;
}

std::vector<cv::Point3f> largeObjectPoints() {
    return {{0.0F, 0.115F, 0.028F}, {0.0F, -0.115F, 0.028F},
            {0.0F, -0.115F, -0.028F}, {0.0F, 0.115F, -0.028F}};
}

cv::Vec3d frontRotation(double yaw, double pitch) {
    const cv::Matx33d base(0, -1, 0, 0, 0, -1, 1, 0, 0);
    const cv::Matx33d ry(std::cos(yaw), 0, std::sin(yaw), 0, 1, 0,
                         -std::sin(yaw), 0, std::cos(yaw));
    const cv::Matx33d rx(1, 0, 0, 0, std::cos(pitch), -std::sin(pitch),
                         0, std::sin(pitch), std::cos(pitch));
    cv::Vec3d rvec;
    cv::Rodrigues(ry * rx * base, rvec);
    return rvec;
}

void verifyPnpCase(const sp_core::Solver& solver, const sp_core::SolverConfig& config,
                   const cv::Vec3d& tvec, double yaw, double pitch) {
    sp_core::Armor armor;
    armor.type = sp_core::ArmorType::LARGE;
    const cv::Mat camera(3, 3, CV_64F, const_cast<double*>(config.camera_matrix.data()));
    const cv::Mat distortion = cv::Mat::zeros(1, 5, CV_64F);
    cv::projectPoints(largeObjectPoints(), frontRotation(yaw, pitch), tvec,
                      camera.t(), distortion, armor.points);
    check(solver.solve(armor), "synthetic PnP should solve");
    near(armor.xyz_camera.x(), tvec[0], 0.01, "PnP x");
    near(armor.xyz_camera.y(), tvec[1], 0.01, "PnP y");
    near(armor.xyz_camera.z(), tvec[2], 0.02, "PnP z");
    check(armor.reprojection_error < 0.1, "PnP reprojection error");
}

void testSolver() {
    const auto config = solverConfig();
    const sp_core::Solver solver(config);
    verifyPnpCase(solver, config, {0.0, 0.0, 3.0}, 0.0, 0.0);
    verifyPnpCase(solver, config, {0.45, -0.18, 4.0}, 0.0, 0.0);
    verifyPnpCase(solver, config, {-0.25, 0.1, 3.5}, 0.25, -0.12);

    sp_core::Armor wrong_order;
    wrong_order.type = sp_core::ArmorType::LARGE;
    const cv::Mat camera(3, 3, CV_64F, const_cast<double*>(config.camera_matrix.data()));
    cv::projectPoints(largeObjectPoints(), frontRotation(0.0, 0.0), cv::Vec3d(0, 0, 3),
                      camera.t(), cv::Mat::zeros(1, 5, CV_64F), wrong_order.points);
    std::rotate(wrong_order.points.begin(), wrong_order.points.begin() + 1,
                wrong_order.points.end());
    check(!solver.solve(wrong_order), "wrong point order must be rejected");

    sp_core::Armor bad_error;
    bad_error.type = sp_core::ArmorType::LARGE;
    cv::projectPoints(largeObjectPoints(), frontRotation(0.15, 0.0), cv::Vec3d(0.1, 0, 3),
                      camera.t(), cv::Mat::zeros(1, 5, CV_64F), bad_error.points);
    bad_error.points[2].x += 20.0F;
    check(!solver.solve(bad_error), "large reprojection error must be rejected");
}

void testTrajectory() {
    const auto normal = sp_core::solveTrajectory(25.0, 4.0, 0.3);
    check(normal.valid && normal.pitch > 0.0 && normal.fly_time > 0.0,
          "normal trajectory");
    check(!sp_core::solveTrajectory(2.0, 20.0, 20.0).valid, "trajectory without solution");
    check(!sp_core::solveTrajectory(0.0, 4.0, 0.0).valid, "zero speed trajectory");
    check(!sp_core::solveTrajectory(25.0, 0.0, 0.0).valid, "zero distance trajectory");
    check(!sp_core::solveTrajectory(std::numeric_limits<double>::quiet_NaN(), 4.0, 0.0).valid,
          "NaN trajectory input");
}

void testTracker() {
    sp_core::TrackerConfig config;
    config.min_detect_count = 3;
    config.max_temp_lost_count = 3;
    config.max_dt = 0.2;
    config.nis_threshold = 20.0;
    config.measurement_variance << 0.002, 0.002, 0.002, 0.02;
    sp_core::Tracker tracker(config);
    const auto start = Clock::time_point{};
    auto target = tracker.update({trackedArmor(3.0)}, start);
    check(target && tracker.state() == sp_core::TrackState::DETECTING, "tracker detecting");
    for (int i = 1; i <= 12; ++i) {
        target = tracker.update({trackedArmor(3.0 + 0.5 * i * 0.02)},
                                start + std::chrono::milliseconds(20 * i));
    }
    check(target && tracker.state() == sp_core::TrackState::TRACKING, "tracker tracking");
    check(target->state().allFinite() && target->state()[1] > 0.1, "constant velocity estimate");
    const double x_before = target->state()[0];
    const double vx_before = target->state()[1];
    target = tracker.update({}, start + std::chrono::milliseconds(280));
    check(target && tracker.state() == sp_core::TrackState::TEMP_LOST, "tracker occlusion");
    near(target->state()[0] - x_before, vx_before * 0.04, 0.03, "tracker uses real dt");
    check(target->state().allFinite(), "occlusion prediction finite");
    tracker.update({trackedArmor(3.0 + 0.5 * 0.30)}, start + std::chrono::milliseconds(300));
    check(tracker.state() == sp_core::TrackState::TRACKING, "tracker recovers after occlusion");

    const auto before_outlier = tracker.update({trackedArmor(3.16)},
        start + std::chrono::milliseconds(320));
    target = tracker.update({trackedArmor(30.0, 20.0, 10.0)},
                            start + std::chrono::milliseconds(340));
    check(target && tracker.state() == sp_core::TrackState::TEMP_LOST, "NIS rejects outlier");
    check(tracker.lastNis() > config.nis_threshold, "outlier NIS threshold");
    check(target->state().allFinite() && before_outlier->state().allFinite(), "outlier remains finite");
    check(target->state().head<3>().norm() < 10.0, "outlier does not enter state");
}

void testAimer() {
    sp_core::TrackerConfig tracker_config;
    const sp_core::Target target(trackedArmor(3.0), Clock::time_point{}, tracker_config);
    sp_core::AimerConfig aimer_config;
    const auto result = sp_core::Aimer(aimer_config).aim(target, 25.0);
    check(result.valid, "aimer result valid");
    check(std::isfinite(result.yaw) && std::isfinite(result.pitch) &&
              std::isfinite(result.fly_time) && result.point.allFinite(),
          "aimer output finite");
    check(!result.fire, "aimer auto fire disabled");
    check(!sp_core::Aimer(aimer_config).aim(target, 0.0).valid,
          "aimer rejects missing bullet speed");
    check(!sp_core::Aimer(aimer_config).aim(
              target, std::numeric_limits<double>::quiet_NaN()).valid,
          "aimer rejects non-finite bullet speed");
}

void testTrackerSafetyEdges() {
    sp_core::TrackerConfig config;
    config.min_detect_count = 1;
    config.max_dt = 0.2;
    config.nis_threshold = 1.0;
    config.nis_window = 10;
    config.nis_failure_ratio = 0.4;
    sp_core::Tracker tracker(config);
    const auto start = Clock::time_point{};
    tracker.update({trackedArmor(3.0)}, start);
    check(tracker.state() == sp_core::TrackState::DETECTING,
          "first observation enters detecting");
    tracker.update({trackedArmor(3.0)}, start + std::chrono::milliseconds(20));
    check(tracker.state() == sp_core::TrackState::TRACKING,
          "second observation reaches tracking");

    // Early outliers must not trigger ratio reset before the configured window is full.
    for (int i = 2; i < 5; ++i) {
        tracker.update({trackedArmor(30.0, 20.0, 10.0)},
                       start + std::chrono::milliseconds(20 * i));
    }
    check(tracker.state() != sp_core::TrackState::LOST,
          "partial NIS window does not force reset");

    // Time reversal is a segment boundary and must reset instead of predicting with dt=0.
    tracker.update({trackedArmor(3.0)}, start + std::chrono::milliseconds(10));
    check(tracker.state() == sp_core::TrackState::DETECTING,
          "time reversal resets and starts a new target");
}

}  // namespace

int main() {
    try {
        testAdapt();
        testSolver();
        testTrajectory();
        testTracker();
        testAimer();
        testTrackerSafetyEdges();
        std::cout << "sp_core tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "sp_core test failure: " << error.what() << '\n';
        return 1;
    }
}
