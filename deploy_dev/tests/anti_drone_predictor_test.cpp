#include "anti_drone/predictor.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <string>

namespace {

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

hnu25::anti_drone::TrackEstimate3D makeTrack(
    hnu25::anti_drone::TrackState state,
    double px, double py, double pz,
    double vx, double vy, double vz) {
    hnu25::anti_drone::TrackEstimate3D t;
    t.state = state;
    t.position_gimbal_m = cv::Vec3d(px, py, pz);
    t.velocity_gimbal_m_s = cv::Vec3d(vx, vy, vz);
    return t;
}

// Test 1: horizon == 0 keeps the current position.
void testZeroHorizon() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::TRACKING,
        3.0, 1.0, 0.5, 1.0, 2.0, 0.5);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.0;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(p.valid, "valid == true");
    check(approxVec(p.predicted_position_gimbal_m, cv::Vec3d(3.0, 1.0, 0.5)),
          "predicted == current position");
}

// Test 2: 100 ms extrapolation.
void test100msPrediction() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::TRACKING,
        3.0, 1.0, 0.5, 1.0, 2.0, 0.5);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.1;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(p.valid, "valid == true");
    check(approxVec(p.predicted_position_gimbal_m, cv::Vec3d(3.1, 1.2, 0.55)),
          "predicted == (3.1,1.2,0.55)");
}

// Test 3: synthetic 3 m/s motion.
void test3mpsSynthetic() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::TRACKING,
        3.0, 0.0, 0.0, 3.0, 0.0, 0.0);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.1;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(p.valid, "valid == true");
    check(approxVec(p.predicted_position_gimbal_m, cv::Vec3d(3.3, 0.0, 0.0)),
          "predicted == (3.3,0,0)");
}

// Test 4: a stationary target keeps its position regardless of horizon.
void testStationaryTarget() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::TRACKING,
        4.0, 1.0, 0.5, 0.0, 0.0, 0.0);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.15;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(p.valid, "valid == true");
    check(approxVec(p.predicted_position_gimbal_m, cv::Vec3d(4.0, 1.0, 0.5)),
          "predicted == current position");
}

// Test 5: TEMP_LOST is allowed because the tracker is already coasting.
void testTempLostAllowed() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::TEMP_LOST,
        3.2, 0.0, 0.0, 1.0, 0.0, 0.0);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.1;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(p.valid, "valid == true");
    check(approxVec(p.predicted_position_gimbal_m, cv::Vec3d(3.3, 0.0, 0.0)),
          "predicted == (3.3,0,0)");
}

// Test 6: DETECTING is rejected.
void testDetectingRejected() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::DETECTING,
        3.0, 0.0, 0.0, 1.0, 0.0, 0.0);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.1;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(!p.valid, "DETECTING -> invalid");
}

// Test 7: LOST is rejected.
void testLostRejected() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::LOST,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.1;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(!p.valid, "LOST -> invalid");
}

// Test 8: non-finite velocity yields invalid without throwing.
void testNonFiniteState() {
    auto track = makeTrack(
        hnu25::anti_drone::TrackState::TRACKING,
        3.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    track.velocity_gimbal_m_s[0] = std::numeric_limits<double>::quiet_NaN();

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.1;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(!p.valid, "non-finite velocity -> invalid");
}

// Test 9: overflow during extrapolation yields invalid, no inf leak.
void testOverflow() {
    const auto track = makeTrack(
        hnu25::anti_drone::TrackState::TRACKING,
        1e308, 0.0, 0.0, 1e308, 0.0, 0.0);

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 1.0;

    const auto p = hnu25::anti_drone::predictTrack(track, config);
    check(!p.valid, "overflow -> invalid");
    check(std::isfinite(p.predicted_position_gimbal_m[0]) &&
              std::isfinite(p.predicted_position_gimbal_m[1]) &&
              std::isfinite(p.predicted_position_gimbal_m[2]),
          "no non-finite value leaks into the result");
}

// Test 10: predictTrack must not mutate the input track.
void testInputUnchanged() {
    hnu25::anti_drone::TrackEstimate3D track = makeTrack(
        hnu25::anti_drone::TrackState::TRACKING,
        3.0, 1.0, 0.5, 1.0, 2.0, 0.5);
    track.missed_count = 2;
    track.consecutive_detects = 3;

    const auto pos_before = track.position_gimbal_m;
    const auto vel_before = track.velocity_gimbal_m_s;
    const auto state_before = track.state;
    const int missed_before = track.missed_count;
    const int detects_before = track.consecutive_detects;

    hnu25::anti_drone::PredictionConfig config;
    config.horizon_s = 0.1;
    hnu25::anti_drone::predictTrack(track, config);

    check(track.position_gimbal_m[0] == pos_before[0] &&
              track.position_gimbal_m[1] == pos_before[1] &&
              track.position_gimbal_m[2] == pos_before[2],
          "position unchanged");
    check(track.velocity_gimbal_m_s[0] == vel_before[0] &&
              track.velocity_gimbal_m_s[1] == vel_before[1] &&
              track.velocity_gimbal_m_s[2] == vel_before[2],
          "velocity unchanged");
    check(track.state == state_before, "state unchanged");
    check(track.missed_count == missed_before, "missed_count unchanged");
    check(track.consecutive_detects == detects_before,
          "consecutive_detects unchanged");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"zero horizon", testZeroHorizon},
        {"100ms prediction", test100msPrediction},
        {"3 m/s synthetic", test3mpsSynthetic},
        {"stationary target", testStationaryTarget},
        {"temp lost allowed", testTempLostAllowed},
        {"detecting rejected", testDetectingRejected},
        {"lost rejected", testLostRejected},
        {"non-finite state", testNonFiniteState},
        {"overflow", testOverflow},
        {"input unchanged", testInputUnchanged},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone predictor tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
