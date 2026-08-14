#include "anti_drone/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

// Shared default calibration + target geometry.
hnu25::anti_drone::PnpSolverConfig makeConfig() {
    hnu25::anti_drone::PnpSolverConfig config;
    config.camera_matrix = cv::Matx33d(
        800.0, 0.0, 320.0,
        0.0, 800.0, 240.0,
        0.0, 0.0, 1.0);
    config.distort_coeffs = {0.0, 0.0, 0.0, 0.0, 0.0};
    config.R_camera2gimbal = cv::Matx33d::eye();
    config.t_camera2gimbal = cv::Vec3d(0.0, 0.0, 0.0);
    config.target_width_m = 0.50;
    config.target_height_m = 0.50;
    config.max_reprojection_error_px = 1.0;
    return config;
}

std::vector<cv::Point3d> makeObjectPoints() {
    const double half = 0.25;
    return {
        cv::Point3d(-half, -half, 0.0),  // TL
        cv::Point3d(+half, -half, 0.0),  // TR
        cv::Point3d(+half, +half, 0.0),  // BR
        cv::Point3d(-half, +half, 0.0),  // BL
    };
}

// Test 1: recover a known non-degenerate pose (position only).
void testKnownPoseRecovery() {
    auto config = makeConfig();

    const cv::Vec3d rvec_truth(0.08, -0.10, 0.03);
    const cv::Vec3d tvec_truth(0.15, -0.08, 3.0);

    const auto object_points = makeObjectPoints();
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec_truth, tvec_truth,
                      config.camera_matrix, config.distort_coeffs, projected);

    hnu25::anti_drone::TargetObservation observation;
    observation.corners = {projected[0], projected[1], projected[2],
                           projected[3]};
    observation.corners_valid = true;

    hnu25::anti_drone::PnpSolver solver(config);
    const auto result = solver.solve(observation);

    check(result.valid, "known pose is valid");
    check(result.reprojection_error_px < 0.1, "reprojection < 0.1 px");

    check(std::fabs(result.xyz_camera[0] - tvec_truth[0]) <= 0.02,
          "xyz_camera.x within 0.02 m");
    check(std::fabs(result.xyz_camera[1] - tvec_truth[1]) <= 0.02,
          "xyz_camera.y within 0.02 m");
    check(std::fabs(result.xyz_camera[2] - tvec_truth[2]) <= 0.02,
          "xyz_camera.z within 0.02 m");
}

// Test 2: camera -> gimbal transform is applied without transpose / missing
// translation.
void testCameraToGimbalTransform() {
    auto config = makeConfig();
    // 90-degree rotation about Z.
    config.R_camera2gimbal = cv::Matx33d(
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0);
    config.t_camera2gimbal = cv::Vec3d(0.10, -0.20, 0.30);

    const cv::Vec3d rvec_truth(0.08, -0.10, 0.03);
    const cv::Vec3d tvec_truth(0.15, -0.08, 3.0);

    const auto object_points = makeObjectPoints();
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec_truth, tvec_truth,
                      config.camera_matrix, config.distort_coeffs, projected);

    hnu25::anti_drone::TargetObservation observation;
    observation.corners = {projected[0], projected[1], projected[2],
                           projected[3]};
    observation.corners_valid = true;

    hnu25::anti_drone::PnpSolver solver(config);
    const auto result = solver.solve(observation);

    check(result.valid, "transform pose is valid");

    const cv::Vec3d expected_gimbal =
        config.R_camera2gimbal * tvec_truth + config.t_camera2gimbal;

    check(std::fabs(result.xyz_gimbal[0] - expected_gimbal[0]) <= 0.02,
          "xyz_gimbal.x within 0.02 m");
    check(std::fabs(result.xyz_gimbal[1] - expected_gimbal[1]) <= 0.02,
          "xyz_gimbal.y within 0.02 m");
    check(std::fabs(result.xyz_gimbal[2] - expected_gimbal[2]) <= 0.02,
          "xyz_gimbal.z within 0.02 m");
}

// Test 3: invalid corners return an invalid result without throwing.
void testCornersInvalid() {
    auto config = makeConfig();

    hnu25::anti_drone::TargetObservation observation;
    observation.corners_valid = false;

    hnu25::anti_drone::PnpSolver solver(config);
    const auto result = solver.solve(observation);

    check(!result.valid, "corners_valid=false -> invalid");
}

// Test 4: degenerate (identical) corners return invalid without crashing.
void testInvalidGeometry() {
    auto config = makeConfig();

    hnu25::anti_drone::TargetObservation observation;
    observation.corners = {
        cv::Point2f(320.0f, 240.0f),
        cv::Point2f(320.0f, 240.0f),
        cv::Point2f(320.0f, 240.0f),
        cv::Point2f(320.0f, 240.0f),
    };
    observation.corners_valid = true;

    hnu25::anti_drone::PnpSolver solver(config);
    const auto result = solver.solve(observation);

    check(!result.valid, "degenerate corners -> invalid");
}

// Test 5: a corrupted corner exceeds the strict reprojection threshold.
void testStrictReprojectionThreshold() {
    auto config = makeConfig();
    config.max_reprojection_error_px = 0.5;

    const cv::Vec3d rvec_truth(0.08, -0.10, 0.03);
    const cv::Vec3d tvec_truth(0.15, -0.08, 3.0);

    const auto object_points = makeObjectPoints();
    std::vector<cv::Point2f> projected;
    cv::projectPoints(object_points, rvec_truth, tvec_truth,
                      config.camera_matrix, config.distort_coeffs, projected);

    // Corrupt one corner (TR) by 8 px.
    projected[1].x += 8.0f;

    hnu25::anti_drone::TargetObservation observation;
    observation.corners = {projected[0], projected[1], projected[2],
                           projected[3]};
    observation.corners_valid = true;

    hnu25::anti_drone::PnpSolver solver(config);
    const auto result = solver.solve(observation);

    check(!result.valid, "corrupted corner -> invalid");
    // If solvePnP produced a reprojection at all, it must exceed 0.5 px.
    if (result.reprojection_error_px > 0.0) {
        check(result.reprojection_error_px > 0.5,
              "corrupted corner reprojection > 0.5");
    }
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"known pose recovery", testKnownPoseRecovery},
        {"camera to gimbal transform", testCameraToGimbalTransform},
        {"corners invalid", testCornersInvalid},
        {"invalid geometry", testInvalidGeometry},
        {"strict reprojection threshold", testStrictReprojectionThreshold},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone pnp tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
