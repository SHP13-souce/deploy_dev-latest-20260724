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

bool approx(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

// Test 1: zero compensation leaves the solution identical to the raw input.
void testZeroCompensation() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_gimbal = cv::Vec3d(3.0, 4.0, 5.0);

    const auto sol = hnu25::anti_drone::makeVisionSolution(
        pnp, hnu25::anti_drone::VisionCompensationConfig{});

    check(sol.valid, "valid == true");

    check(approx(sol.xyz_gimbal_raw[0], 3.0), "raw.x == 3");
    check(approx(sol.xyz_gimbal_raw[1], 4.0), "raw.y == 4");
    check(approx(sol.xyz_gimbal_raw[2], 5.0), "raw.z == 5");

    check(approx(sol.xyz_gimbal_compensated[0], 3.0), "comp.x == 3");
    check(approx(sol.xyz_gimbal_compensated[1], 4.0), "comp.y == 4");
    check(approx(sol.xyz_gimbal_compensated[2], 5.0), "comp.z == 5");

    const double expected_yaw = std::atan2(4.0, 3.0);
    const double expected_pitch = std::atan2(5.0, 5.0);
    check(approx(sol.yaw_raw_rad, expected_yaw), "yaw_raw == atan2(4,3)");
    check(approx(sol.pitch_raw_rad, expected_pitch), "pitch_raw == atan2(5,5)");

    check(approx(sol.yaw_compensated_rad, sol.yaw_raw_rad),
          "yaw_compensated == yaw_raw");
    check(approx(sol.pitch_compensated_rad, sol.pitch_raw_rad),
          "pitch_compensated == pitch_raw");
}

// Test 2: xyz residual offset shifts the position and therefore the geometry.
void testXyzCompensation() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_gimbal = cv::Vec3d(3.0, 4.0, 5.0);

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.x_offset_m = 1.0;
    comp.y_offset_m = -2.0;
    comp.z_offset_m = 0.5;

    const auto sol = hnu25::anti_drone::makeVisionSolution(pnp, comp);

    check(sol.valid, "valid == true");

    check(approx(sol.xyz_gimbal_raw[0], 3.0), "raw.x == 3");
    check(approx(sol.xyz_gimbal_raw[1], 4.0), "raw.y == 4");
    check(approx(sol.xyz_gimbal_raw[2], 5.0), "raw.z == 5");

    check(approx(sol.xyz_gimbal_compensated[0], 4.0), "comp.x == 4");
    check(approx(sol.xyz_gimbal_compensated[1], 2.0), "comp.y == 2");
    check(approx(sol.xyz_gimbal_compensated[2], 5.5), "comp.z == 5.5");

    check(approx(sol.yaw_compensated_rad, std::atan2(2.0, 4.0)),
          "comp yaw == atan2(2,4)");
    check(approx(sol.pitch_compensated_rad,
                 std::atan2(5.5, std::hypot(4.0, 2.0))),
          "comp pitch == atan2(5.5, hypot(4,2))");
}

// Test 3: angular-only offsets (expressed in degrees in the YAML/Config).
void testAngleCompensation() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_gimbal = cv::Vec3d(10.0, 0.0, 0.0);

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.yaw_offset_deg = 10.0;
    comp.pitch_offset_deg = -5.0;

    const auto sol = hnu25::anti_drone::makeVisionSolution(pnp, comp);

    check(sol.valid, "valid == true");

    check(approx(sol.yaw_raw_rad, 0.0), "raw yaw == 0");
    check(approx(sol.pitch_raw_rad, 0.0), "raw pitch == 0");

    check(approx(sol.yaw_compensated_rad, 10.0 * kDegToRad),
          "comp yaw == +10 deg in rad");
    check(approx(sol.pitch_compensated_rad, -5.0 * kDegToRad),
          "comp pitch == -5 deg in rad");
}

// Test 4: xyz compensation happens before angular compensation.
void testXyzPlusAngularCompensation() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_gimbal = cv::Vec3d(4.0, 1.0, 2.0);

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.x_offset_m = 1.0;
    comp.y_offset_m = 1.0;
    comp.z_offset_m = -0.5;
    comp.yaw_offset_deg = 3.0;
    comp.pitch_offset_deg = -2.0;

    const auto sol = hnu25::anti_drone::makeVisionSolution(pnp, comp);

    check(sol.valid, "valid == true");

    check(approx(sol.xyz_gimbal_compensated[0], 5.0), "comp.x == 5");
    check(approx(sol.xyz_gimbal_compensated[1], 2.0), "comp.y == 2");
    check(approx(sol.xyz_gimbal_compensated[2], 1.5), "comp.z == 1.5");

    const double geometric_yaw = std::atan2(2.0, 5.0);
    const double geometric_pitch = std::atan2(1.5, std::hypot(5.0, 2.0));

    check(approx(sol.yaw_compensated_rad, geometric_yaw + 3.0 * kDegToRad),
          "comp yaw == geometric_yaw + 3 deg");
    check(approx(sol.pitch_compensated_rad,
                 geometric_pitch - 2.0 * kDegToRad),
          "comp pitch == geometric_pitch - 2 deg");
}

// Test 5: makeVisionSolution must not mutate the raw PnpResult.
void testRawPnpNotModified() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_camera = cv::Vec3d(1.0, 2.0, 3.0);
    pnp.xyz_gimbal = cv::Vec3d(3.0, 4.0, 5.0);
    pnp.rvec = cv::Vec3d(0.1, 0.2, 0.3);
    pnp.reprojection_error_px = 0.42;

    const auto original_camera = pnp.xyz_camera;
    const auto original_gimbal = pnp.xyz_gimbal;
    const auto original_rvec = pnp.rvec;
    const double original_reproj = pnp.reprojection_error_px;

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.x_offset_m = 1.0;
    comp.y_offset_m = 2.0;
    comp.z_offset_m = 3.0;
    comp.yaw_offset_deg = 4.0;
    comp.pitch_offset_deg = -3.0;

    hnu25::anti_drone::makeVisionSolution(pnp, comp);

    check(pnp.xyz_gimbal[0] == original_gimbal[0] &&
              pnp.xyz_gimbal[1] == original_gimbal[1] &&
              pnp.xyz_gimbal[2] == original_gimbal[2],
          "xyz_gimbal unchanged");
    check(pnp.xyz_camera[0] == original_camera[0] &&
              pnp.xyz_camera[1] == original_camera[1] &&
              pnp.xyz_camera[2] == original_camera[2],
          "xyz_camera unchanged");
    check(pnp.rvec[0] == original_rvec[0] &&
              pnp.rvec[1] == original_rvec[1] &&
              pnp.rvec[2] == original_rvec[2],
          "rvec unchanged");
    check(pnp.reprojection_error_px == original_reproj,
          "reprojection_error_px unchanged");
}

// Test 6: an invalid PnP result yields an invalid solution without throwing.
void testInvalidPnp() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = false;

    const auto sol = hnu25::anti_drone::makeVisionSolution(
        pnp, hnu25::anti_drone::VisionCompensationConfig{});

    check(!sol.valid, "invalid pnp -> invalid solution");
}

// Test 7: non-finite compensation yields invalid without throwing.
void testNonFiniteCompensation() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_gimbal = cv::Vec3d(3.0, 4.0, 5.0);

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.yaw_offset_deg = std::numeric_limits<double>::quiet_NaN();

    const auto sol = hnu25::anti_drone::makeVisionSolution(pnp, comp);

    check(!sol.valid, "non-finite compensation -> invalid");
}

// Test 8: yaw wraps into [-pi, pi].
void testYawWrap() {
    const double angle_rad = 179.0 * kDegToRad;

    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_gimbal = cv::Vec3d(
        std::cos(angle_rad) * 10.0,
        std::sin(angle_rad) * 10.0,
        0.0);

    hnu25::anti_drone::VisionCompensationConfig comp;
    comp.yaw_offset_deg = 5.0;

    const auto sol = hnu25::anti_drone::makeVisionSolution(pnp, comp);

    check(sol.valid, "valid == true");

    // 179 + 5 = 184 deg -> wrapped to -176 deg.
    const double expected = -176.0 * kDegToRad;
    check(approx(sol.yaw_compensated_rad, expected, 1e-6),
          "yaw wrapped to ~-176 deg");
    check(sol.yaw_compensated_rad >= -kPi - 1e-9 &&
              sol.yaw_compensated_rad <= kPi + 1e-9,
          "yaw within [-pi, pi]");
}

// Test 9: x == y == 0 leaves yaw undefined -> invalid, no crash.
void testUndefinedHorizontalDirection() {
    hnu25::anti_drone::PnpResult pnp;
    pnp.valid = true;
    pnp.xyz_gimbal = cv::Vec3d(0.0, 0.0, 5.0);

    const auto sol = hnu25::anti_drone::makeVisionSolution(
        pnp, hnu25::anti_drone::VisionCompensationConfig{});

    check(!sol.valid, "undefined horizontal direction -> invalid");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"zero compensation", testZeroCompensation},
        {"xyz compensation", testXyzCompensation},
        {"angle compensation", testAngleCompensation},
        {"xyz + angular compensation", testXyzPlusAngularCompensation},
        {"raw pnp not modified", testRawPnpNotModified},
        {"invalid pnp", testInvalidPnp},
        {"non-finite compensation", testNonFiniteCompensation},
        {"yaw wrap", testYawWrap},
        {"undefined horizontal direction", testUndefinedHorizontalDirection},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone vision solution tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
