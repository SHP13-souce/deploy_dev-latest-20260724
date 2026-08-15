#include "anti_drone/tracker.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

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

// Builds a valid PnP measurement with the given gimbal-frame position.
hnu25::anti_drone::PnpResult measurement(double x, double y, double z,
                                         double reprojection_error = 0.1) {
    hnu25::anti_drone::PnpResult m;
    m.valid = true;
    m.xyz_gimbal = cv::Vec3d(x, y, z);
    m.reprojection_error_px = reprojection_error;
    return m;
}

// Test 1: acquire in DETECTING, then confirm into TRACKING.
void testAcquireAndConfirm() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 3;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    auto e0 = tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    check(e0.has_value(), "t0 produces an estimate");
    check(e0->state == hnu25::anti_drone::TrackState::DETECTING,
          "t0 state == DETECTING");
    check(approxVec(e0->position_gimbal_m, cv::Vec3d(3.0, 0.0, 0.0)),
          "t0 position == (3,0,0)");
    check(approxVec(e0->velocity_gimbal_m_s, cv::Vec3d(0.0, 0.0, 0.0)),
          "t0 velocity == (0,0,0)");
    check(e0->measurement_updated, "t0 measurement_updated == true");

    auto e1 = tracker.update({measurement(3.1, 0.0, 0.0)},
                             t0 + std::chrono::milliseconds(100));
    check(e1.has_value(), "t1 produces an estimate");
    check(e1->state == hnu25::anti_drone::TrackState::TRACKING,
          "t1 state == TRACKING");
    check(approxVec(e1->position_gimbal_m, cv::Vec3d(3.1, 0.0, 0.0)),
          "t1 position == (3.1,0,0)");
    check(approxVec(e1->velocity_gimbal_m_s, cv::Vec3d(1.0, 0.0, 0.0)),
          "t1 velocity ~ (1,0,0) m/s");
}

// Test 2: diagonal motion yields a 3D velocity estimate.
void testDiagonalVelocity() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 3;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 1.0, 0.5)}, t0);
    auto e1 = tracker.update({measurement(3.1, 1.2, 0.55)},
                             t0 + std::chrono::milliseconds(100));

    check(e1->state == hnu25::anti_drone::TrackState::TRACKING,
          "state == TRACKING");
    check(approxVec(e1->velocity_gimbal_m_s, cv::Vec3d(1.0, 2.0, 0.5)),
          "velocity ~ (1,2,0.5) m/s");
}

// Test 3: a stationary target keeps zero velocity.
void testStationaryTarget() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 3;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(4.0, 1.0, 0.5)}, t0);
    auto e1 = tracker.update({measurement(4.0, 1.0, 0.5)},
                             t0 + std::chrono::milliseconds(100));

    check(e1->state == hnu25::anti_drone::TrackState::TRACKING,
          "state == TRACKING");
    check(approxVec(e1->velocity_gimbal_m_s, cv::Vec3d(0.0, 0.0, 0.0)),
          "velocity ~ (0,0,0)");
}

// Test 4: a single miss coasts on the prediction in TEMP_LOST.
void testTempLostPrediction() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 5;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    tracker.update({measurement(3.1, 0.0, 0.0)},
                   t0 + std::chrono::milliseconds(100));

    auto e = tracker.update({}, t0 + std::chrono::milliseconds(200));
    check(e.has_value(), "miss produces an estimate");
    check(e->state == hnu25::anti_drone::TrackState::TEMP_LOST,
          "state == TEMP_LOST");
    check(approxVec(e->position_gimbal_m, cv::Vec3d(3.2, 0.0, 0.0)),
          "position ~ (3.2,0,0)");
    check(approxVec(e->velocity_gimbal_m_s, cv::Vec3d(1.0, 0.0, 0.0)),
          "velocity stays (1,0,0)");
    check(!e->measurement_updated, "measurement_updated == false");
}

// Test 5: reacquire after TEMP_LOST returns to TRACKING and clears misses.
void testReacquire() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 5;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    tracker.update({measurement(3.1, 0.0, 0.0)},
                   t0 + std::chrono::milliseconds(100));
    tracker.update({}, t0 + std::chrono::milliseconds(200));

    auto e = tracker.update({measurement(3.3, 0.0, 0.0)},
                            t0 + std::chrono::milliseconds(300));
    check(e->state == hnu25::anti_drone::TrackState::TRACKING,
          "state == TRACKING");
    check(approxVec(e->position_gimbal_m, cv::Vec3d(3.3, 0.0, 0.0)),
          "position ~ (3.3,0,0)");
    check(e->missed_count == 0, "missed_count == 0");
    check(e->measurement_updated, "measurement_updated == true");
}

// Test 6: exceeding max_missed_count resets to LOST and returns nullopt.
void testMissLimit() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 2;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    tracker.update({measurement(3.1, 0.0, 0.0)},
                   t0 + std::chrono::milliseconds(100));

    auto e1 = tracker.update({}, t0 + std::chrono::milliseconds(200));
    check(e1->state == hnu25::anti_drone::TrackState::TEMP_LOST &&
              e1->missed_count == 1,
          "miss 1 -> TEMP_LOST, missed_count == 1");

    auto e2 = tracker.update({}, t0 + std::chrono::milliseconds(300));
    check(e2->state == hnu25::anti_drone::TrackState::TEMP_LOST &&
              e2->missed_count == 2,
          "miss 2 -> TEMP_LOST, missed_count == 2");

    auto e3 = tracker.update({}, t0 + std::chrono::milliseconds(400));
    check(!e3.has_value(), "miss 3 -> nullopt");
    check(tracker.state() == hnu25::anti_drone::TrackState::LOST,
          "state == LOST after reset");
}

// Test 7: association picks the spatially nearest candidate, not the one with
// the best single-frame reprojection quality.
void testNearest3dAssociation() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 5;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    tracker.update({measurement(3.1, 0.0, 0.0)},
                   t0 + std::chrono::milliseconds(100));

    auto e = tracker.update(
        {measurement(3.21, 0.02, 0.0, 2.0),
         measurement(3.7, 0.5, 0.3, 0.01)},
        t0 + std::chrono::milliseconds(200));

    check(e->state == hnu25::anti_drone::TrackState::TRACKING,
          "state == TRACKING");
    check(approxVec(e->position_gimbal_m, cv::Vec3d(3.21, 0.02, 0.0)),
          "associates with spatially nearest candidate A");
}

// Test 8: a candidate outside the gate is rejected; track coasts.
void testFarCandidateRejected() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 5;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 0.5;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    tracker.update({measurement(3.1, 0.0, 0.0)},
                   t0 + std::chrono::milliseconds(100));

    auto e = tracker.update({measurement(5.0, 2.0, 1.0)},
                            t0 + std::chrono::milliseconds(200));
    check(e->state == hnu25::anti_drone::TrackState::TEMP_LOST,
          "far candidate -> TEMP_LOST");
    check(approxVec(e->position_gimbal_m, cv::Vec3d(3.2, 0.0, 0.0)),
          "position coasts to prediction (3.2,0,0)");
}

// Test 9: invalid / non-finite measurements never establish a track.
void testInvalidPnpIgnored() {
    hnu25::anti_drone::TrackerConfig cfg;
    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    hnu25::anti_drone::PnpResult m1 = measurement(3.0, 0.0, 0.0);
    m1.valid = false;
    check(!tracker.update({m1}, t0).has_value(), "valid=false ignored");

    hnu25::anti_drone::PnpResult m2 = measurement(3.0, 0.0, 0.0);
    m2.xyz_gimbal[0] = std::numeric_limits<double>::quiet_NaN();
    check(!tracker.update({m2}, t0).has_value(), "NaN xyz ignored");

    hnu25::anti_drone::PnpResult m3 = measurement(3.0, 0.0, 0.0);
    m3.reprojection_error_px = std::numeric_limits<double>::quiet_NaN();
    check(!tracker.update({m3}, t0).has_value(), "NaN reprojection ignored");

    check(tracker.state() == hnu25::anti_drone::TrackState::LOST,
          "state remains LOST");
}

// Test 10: a gap larger than max_dt_s resets, then reacquires immediately.
void testLargeDtReset() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 3;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    tracker.update({measurement(3.1, 0.0, 0.0)},
                   t0 + std::chrono::milliseconds(100));

    // 1000 ms gap exceeds max_dt_s (500 ms).
    auto e = tracker.update({measurement(8.0, 1.0, 2.0)},
                            t0 + std::chrono::milliseconds(1100));
    check(e.has_value(), "reacquires after reset");
    check(e->state == hnu25::anti_drone::TrackState::DETECTING,
          "state == DETECTING after reacquire");
    check(approxVec(e->position_gimbal_m, cv::Vec3d(8.0, 1.0, 2.0)),
          "position == (8,1,2)");
    check(approxVec(e->velocity_gimbal_m_s, cv::Vec3d(0.0, 0.0, 0.0)),
          "velocity reset to (0,0,0)");
}

// Test 11: in LOST, acquisition picks the lowest reprojection error.
void testLostQualitySelection() {
    hnu25::anti_drone::TrackerConfig cfg;
    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    auto e = tracker.update(
        {measurement(3.0, 0.0, 0.0, 3.0), measurement(5.0, 0.0, 0.0, 0.5)},
        t0);
    check(e.has_value(), "acquires from LOST");
    check(e->state == hnu25::anti_drone::TrackState::DETECTING,
          "state == DETECTING");
    check(approxVec(e->position_gimbal_m, cv::Vec3d(5.0, 0.0, 0.0)),
          "picks lower-error measurement (5,0,0)");
}

// Test 12: a ~3 m/s constant-velocity scenario is tracked and predicted.
void testSynthetic3mps() {
    hnu25::anti_drone::TrackerConfig cfg;
    cfg.min_detect_count = 2;
    cfg.max_missed_count = 5;
    cfg.max_dt_s = 0.5;
    cfg.max_association_distance_m = 1.0;
    cfg.position_gain = 1.0;
    cfg.velocity_gain = 1.0;

    hnu25::anti_drone::TargetTracker3D tracker(cfg);
    const auto t0 = Clock::now();

    tracker.update({measurement(3.0, 0.0, 0.0)}, t0);
    auto e1 = tracker.update({measurement(3.3, 0.0, 0.0)},
                             t0 + std::chrono::milliseconds(100));
    check(e1->state == hnu25::anti_drone::TrackState::TRACKING,
          "state == TRACKING");
    check(approxVec(e1->velocity_gimbal_m_s, cv::Vec3d(3.0, 0.0, 0.0)),
          "velocity ~ (3,0,0) m/s");

    auto e2 = tracker.update({}, t0 + std::chrono::milliseconds(200));
    check(e2->state == hnu25::anti_drone::TrackState::TEMP_LOST,
          "state == TEMP_LOST");
    check(approxVec(e2->position_gimbal_m, cv::Vec3d(3.6, 0.0, 0.0)),
          "prediction ~ (3.6,0,0)");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"acquire + confirm", testAcquireAndConfirm},
        {"diagonal velocity", testDiagonalVelocity},
        {"stationary target", testStationaryTarget},
        {"temp lost prediction", testTempLostPrediction},
        {"reacquire", testReacquire},
        {"miss limit", testMissLimit},
        {"nearest 3d association", testNearest3dAssociation},
        {"far candidate rejected", testFarCandidateRejected},
        {"invalid pnp ignored", testInvalidPnpIgnored},
        {"large dt reset", testLargeDtReset},
        {"lost quality selection", testLostQualitySelection},
        {"synthetic 3 m/s motion", testSynthetic3mps},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone tracker tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
