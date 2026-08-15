#include "anti_drone/gimbal_speed_filter.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "[ OK ] " << message << '\n';
    } else {
        std::cerr << "FAILED: " << message << '\n';
        ++g_failures;
    }
}

bool approx(double a, double b, double eps = 1e-4) {
    return std::fabs(a - b) <= eps;
}

constexpr double kPi = 3.14159265358979323846;
constexpr double kDeg = kPi / 180.0;

}  // namespace

int main() {
    // ── yaw wrap: +179deg -> -179deg yields the short +~2deg delta ────────
    {
        hnu25::anti_drone::GimbalSpeedFilter filter(1.0);  // alpha = 1.0
        filter.update(179.0 * kDeg, 0.0, 1.0);             // first sample: 0
        const auto s = filter.update(-179.0 * kDeg, 0.0, 1.0);
        check(approx(s.yaw_rad_s, 2.0 * kDeg, 1e-3),
              "yaw wrap +179 -> -179 gives +~2deg/s (not -358deg/s)");
    }

    // ── yaw wrap: -179deg -> +179deg yields the short -~2deg delta ────────
    {
        hnu25::anti_drone::GimbalSpeedFilter filter(1.0);
        filter.update(-179.0 * kDeg, 0.0, 1.0);
        const auto s = filter.update(179.0 * kDeg, 0.0, 1.0);
        check(approx(s.yaw_rad_s, -2.0 * kDeg, 1e-3),
              "yaw wrap -179 -> +179 gives -~2deg/s");
    }

    // ── ordinary small-angle change (no wrap involved) ─────────────────────
    {
        hnu25::anti_drone::GimbalSpeedFilter filter(1.0);
        filter.update(10.0 * kDeg, 0.0, 1.0);
        const auto s = filter.update(12.0 * kDeg, 0.0, 0.1);
        check(approx(s.yaw_rad_s, 20.0 * kDeg, 1e-3),
              "small-angle yaw rate == 2deg / 0.1s");
    }

    // ── first valid frame returns 0 ────────────────────────────────────────
    {
        hnu25::anti_drone::GimbalSpeedFilter filter(0.3);
        const auto s = filter.update(0.5, -0.2, 0.01);
        check(s.yaw_rad_s == 0.0F && s.pitch_rad_s == 0.0F,
              "first valid frame speed is 0");
    }

    // ── reset() on target loss clears state; next frame is first again ────
    {
        hnu25::anti_drone::GimbalSpeedFilter filter(1.0);
        filter.update(0.0, 0.0, 0.1);
        const auto a = filter.update(0.1, 0.0, 0.1);   // nonzero (raw 1.0)
        check(!approx(static_cast<double>(a.yaw_rad_s), 0.0),
              "second valid frame produces nonzero yaw speed");

        filter.reset();                                 // target lost
        const auto b = filter.update(0.5, 0.0, 0.1);    // reappear: first again
        check(b.yaw_rad_s == 0.0F && b.pitch_rad_s == 0.0F,
              "speed resets to 0 after target-loss reset");
    }

    // ── dt <= 0 must not advance the filter or produce a nonzero rate ──────
    {
        hnu25::anti_drone::GimbalSpeedFilter filter(1.0);
        filter.update(0.0, 0.0, 1.0);
        const auto a = filter.update(0.5, 0.0, 0.0);    // dt == 0
        check(a.yaw_rad_s == 0.0F && a.pitch_rad_s == 0.0F,
              "dt == 0 yields zero speed");
        const auto b = filter.update(0.5, 0.0, -0.1);   // dt < 0
        check(b.yaw_rad_s == 0.0F && b.pitch_rad_s == 0.0F,
              "dt < 0 yields zero speed");
    }

    // ── pitch is NOT wrapped ───────────────────────────────────────────────
    {
        hnu25::anti_drone::GimbalSpeedFilter filter(1.0);
        filter.update(0.0, 179.0 * kDeg, 1.0);
        const auto s = filter.update(0.0, -179.0 * kDeg, 1.0);
        // Raw pitch delta is -358deg/s; unlike yaw it must NOT be wrapped.
        check(approx(static_cast<double>(s.pitch_rad_s), -358.0 * kDeg, 1e-3),
              "pitch is not wrapped (-358deg/s stays raw)");
    }

    if (g_failures == 0) {
        std::cout << "All gimbal speed filter tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
