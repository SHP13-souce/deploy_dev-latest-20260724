#include "anti_drone/predictor.hpp"
#include "anti_drone/vision_solution.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

bool approx(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

bool approxVec(const cv::Vec3d& a, const cv::Vec3d& b, double eps = 1e-6) {
    return std::fabs(a[0] - b[0]) <= eps && std::fabs(a[1] - b[1]) <= eps &&
           std::fabs(a[2] - b[2]) <= eps;
}

// Test-file-only helper: the Vec3d overload does not know about
// Prediction3D.valid, so the business layer must guard before calling it.
hnu25::anti_drone::VisionSolution solutionFromPrediction(
    const hnu25::anti_drone::Prediction3D& prediction,
    const hnu25::anti_drone::VisionCompensationConfig& comp) {
    if (!prediction.valid) {
        return {};
    }
    return hnu25::anti_drone::makeVisionSolution(
        prediction.predicted_position_gimbal_m, comp);
}

// Test 1: a predicted position with zero compensation round-trips unchanged.
void testPredictedZeroCompensation() {
    hnu25::anti_drone::Prediction3D prediction;
    prediction.valid = true;
    prediction.predicted_position_gimbal_m = cv::Vec3d(3.0, 4.0, 5.0);

    const auto sol = hnu25::anti_drone::makeVisionSolution(
        prediction.predicted_position_gimbal_m,
        hnu25::anti_drone::VisionCompensationConfig{});

    check(sol.valid, "valid == true");
    check(approxVec(sol.xyz_gimbal_raw, cv::Vec3d(3.0, 4.0, 5.0)),
          "raw == (3,4,5)");
    check(approxVec(sol.xyz_gimbal_compensated, cv::Vec3d(3.0, 4.0, 5.0)),
          "compensated == (3,4,5)");
    check(approx(sol.yaw_compensated_rad, std::atan2(4.0, 3.0)),
          "yaw == atan2(4,3)");
    check(approx(sol.pitch_compensated_rad, std::atan2(5.0, 5.0)),
          "pitch == atan2(5,5)");
}

// Test 2: xyz compensation shifts the predicted position.
void testPredictedXyzCompensation() {
    hnu25::anti_drone::Prediction3D prediction;
    prediction.valid = true;
    prediction.predicted_position_gimbal_m = cv::Vec3d(3.0, 4.0, 5.0);

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.x_offset_m = 1.0;
    comp.y_offset_m = -2.0;
    comp.z_offset_m = 0.5;

    const auto sol = hnu25::anti_drone::makeVisionSolution(
        prediction.predicted_position_gimbal_m, comp);

    check(sol.valid, "valid == true");
    check(approxVec(sol.xyz_gimbal_compensated, cv::Vec3d(4.0, 2.0, 5.5)),
          "compensated == (4,2,5.5)");
    check(approx(sol.yaw_compensated_rad, std::atan2(2.0, 4.0)),
          "yaw == atan2(2,4)");
    check(approx(sol.pitch_compensated_rad,
                 std::atan2(5.5, std::hypot(4.0, 2.0))),
          "pitch == atan2(5.5, hypot(4,2))");
}

// Test 3: angular compensation on a predicted position.
void testPredictedAngularCompensation() {
    hnu25::anti_drone::Prediction3D prediction;
    prediction.valid = true;
    prediction.predicted_position_gimbal_m = cv::Vec3d(10.0, 0.0, 0.0);

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.yaw_offset_deg = 5.0;
    comp.pitch_offset_deg = -3.0;

    const auto sol = hnu25::anti_drone::makeVisionSolution(
        prediction.predicted_position_gimbal_m, comp);

    check(sol.valid, "valid == true");
    check(approx(sol.yaw_compensated_rad, 5.0 * kDegToRad),
          "yaw == +5 deg in rad");
    check(approx(sol.pitch_compensated_rad, -3.0 * kDegToRad),
          "pitch == -3 deg in rad");
}

// Test 4: compose predictTrack with the Vec3d overload (two layers combined).
void testSyntheticPredictedMovingTarget() {
    hnu25::anti_drone::TrackEstimate3D track;
    track.state = hnu25::anti_drone::TrackState::TRACKING;
    track.position_gimbal_m = cv::Vec3d(3.0, 0.0, 0.0);
    track.velocity_gimbal_m_s = cv::Vec3d(3.0, 0.0, 0.0);

    hnu25::anti_drone::PredictionConfig pconfig;
    pconfig.horizon_s = 0.1;
    const auto prediction = hnu25::anti_drone::predictTrack(track, pconfig);

    check(prediction.valid, "prediction valid");
    const auto sol = hnu25::anti_drone::makeVisionSolution(
        prediction.predicted_position_gimbal_m,
        hnu25::anti_drone::VisionCompensationConfig{});

    check(sol.valid, "solution valid");
    check(approxVec(sol.xyz_gimbal_raw, cv::Vec3d(3.3, 0.0, 0.0)),
          "raw ~ (3.3,0,0)");
    check(approx(sol.yaw_compensated_rad, 0.0), "yaw ~ 0");
    check(approx(sol.pitch_compensated_rad, 0.0), "pitch ~ 0");
}

// Test 5: an invalid prediction must not be passed through (business-layer
// guard modelled by the local helper).
void testInvalidPredictionGuarded() {
    hnu25::anti_drone::Prediction3D prediction;
    prediction.valid = false;
    prediction.predicted_position_gimbal_m = cv::Vec3d(3.0, 4.0, 5.0);

    const auto sol = solutionFromPrediction(
        prediction, hnu25::anti_drone::VisionCompensationConfig{});

    check(!sol.valid, "invalid prediction -> invalid solution");
}

// Test 6: non-finite predicted xyz yields invalid without throwing.
void testNonFinitePredictedXyz() {
    const cv::Vec3d xyz(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0);

    const auto sol = hnu25::anti_drone::makeVisionSolution(
        xyz, hnu25::anti_drone::VisionCompensationConfig{});

    check(!sol.valid, "non-finite xyz -> invalid");
}

// Test 7: x == y == 0 leaves yaw undefined -> invalid (matches VisionSolution).
void testUndefinedYawDirection() {
    const auto sol = hnu25::anti_drone::makeVisionSolution(
        cv::Vec3d(0.0, 0.0, 5.0),
        hnu25::anti_drone::VisionCompensationConfig{});

    check(!sol.valid, "undefined yaw direction -> invalid");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"predicted zero compensation", testPredictedZeroCompensation},
        {"predicted xyz compensation", testPredictedXyzCompensation},
        {"predicted angular compensation", testPredictedAngularCompensation},
        {"synthetic predicted moving target", testSyntheticPredictedMovingTarget},
        {"invalid prediction guarded", testInvalidPredictionGuarded},
        {"non-finite predicted xyz", testNonFinitePredictedXyz},
        {"undefined yaw direction", testUndefinedYawDirection},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone prediction solution tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
