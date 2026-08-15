#include "anti_drone/diagnostic_pipeline.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

bool finiteVec(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

bool approx(double a, double b, double eps = 1e-6) {
    return std::fabs(a - b) <= eps;
}

bool approxVec(const cv::Vec3d& a, const cv::Vec3d& b, double eps = 1e-6) {
    return std::fabs(a[0] - b[0]) <= eps && std::fabs(a[1] - b[1]) <= eps &&
           std::fabs(a[2] - b[2]) <= eps;
}

hnu25::anti_drone::PnpResult measurement(
    double x, double y, double z, double reprojection_error = 0.2) {
    hnu25::anti_drone::PnpResult m;
    m.valid = true;
    m.xyz_gimbal = cv::Vec3d(x, y, z);
    m.reprojection_error_px = reprojection_error;
    return m;
}

// Shared tracker config used by most tests: a reasonable constant-velocity
// setup with alpha = 1.0 (position follows measurement) and beta = 0.2.
hnu25::anti_drone::DiagnosticPipelineConfig baseConfig() {
    hnu25::anti_drone::DiagnosticPipelineConfig cfg;
    cfg.tracker.min_detect_count = 2;
    cfg.tracker.max_missed_count = 5;
    cfg.tracker.max_dt_s = 0.2;
    cfg.tracker.max_association_distance_m = 0.8;
    cfg.tracker.position_gain = 1.0;
    cfg.tracker.velocity_gain = 0.2;
    cfg.prediction.horizon_s = 0.08;
    return cfg;
}

// Test 1: a DETECTING track is available but is never extrapolated.
void testDetectingGating() {
    hnu25::anti_drone::DiagnosticPipeline pipeline(baseConfig());
    const auto t0 = Clock::now();

    const auto r = pipeline.update({measurement(4.0, 0.0, 0.5)}, t0);

    check(r.track_available, "track_available == true");
    check(r.track_state == hnu25::anti_drone::TrackState::DETECTING,
          "track_state == DETECTING");
    check(r.measurement_updated, "measurement_updated == true");
    check(!r.prediction_valid, "prediction_valid == false");
    check(!r.solution_valid, "solution_valid == false");
    check(approxVec(r.tracked_position_gimbal_m, cv::Vec3d(4.0, 0.0, 0.5)),
          "tracked == (4,0,0.5)");
}

// Test 2: two consecutive observations confirm the track and the static
// prediction flows end-to-end.
void testConfirmedStaticPipeline() {
    hnu25::anti_drone::DiagnosticPipeline pipeline(baseConfig());
    const auto t0 = Clock::now();

    pipeline.update({measurement(4.0, 0.0, 0.5)}, t0);
    const auto r = pipeline.update(
        {measurement(4.0, 0.0, 0.5)}, t0 + std::chrono::milliseconds(20));

    check(r.track_available, "track_available == true");
    check(r.track_state == hnu25::anti_drone::TrackState::TRACKING,
          "track_state == TRACKING");
    check(r.measurement_updated, "measurement_updated == true");
    check(r.prediction_valid, "prediction_valid == true");
    check(r.solution_valid, "solution_valid == true");

    check(approxVec(r.velocity_gimbal_m_s, cv::Vec3d(0.0, 0.0, 0.0)),
          "velocity ~ (0,0,0)");
    check(approxVec(r.predicted_position_gimbal_m, cv::Vec3d(4.0, 0.0, 0.5)),
          "predicted ~ (4,0,0.5)");
    check(approxVec(r.compensated_position_gimbal_m, cv::Vec3d(4.0, 0.0, 0.5)),
          "compensated ~ (4,0,0.5)");
    check(approx(r.predicted_yaw_rad, 0.0), "yaw ~ 0");
    check(approx(r.predicted_pitch_rad, std::atan2(0.5, 4.0)),
          "pitch ~ atan2(0.5,4)");
    check(approx(r.prediction_horizon_s, 0.08), "horizon == 0.08");
}

// Test 3: compensation is applied exactly once, to the predicted position.
void testCompensationPropagatedOnce() {
    hnu25::anti_drone::DiagnosticPipelineConfig cfg = baseConfig();
    cfg.prediction.horizon_s = 0.0;
    cfg.compensation.x_offset_m = 0.10;
    cfg.compensation.y_offset_m = -0.05;
    cfg.compensation.z_offset_m = 0.02;
    cfg.compensation.yaw_offset_deg = 2.0;
    cfg.compensation.pitch_offset_deg = -1.0;

    hnu25::anti_drone::DiagnosticPipeline pipeline(cfg);
    const auto t0 = Clock::now();

    pipeline.update({measurement(4.0, 1.0, 0.5)}, t0);
    const auto r = pipeline.update(
        {measurement(4.0, 1.0, 0.5)}, t0 + std::chrono::milliseconds(20));

    check(r.track_state == hnu25::anti_drone::TrackState::TRACKING,
          "track_state == TRACKING");
    check(approxVec(r.tracked_position_gimbal_m, cv::Vec3d(4.0, 1.0, 0.5)),
          "tracked raw == (4,1,0.5)");
    check(approxVec(r.predicted_position_gimbal_m, cv::Vec3d(4.0, 1.0, 0.5)),
          "predicted raw == (4,1,0.5)");
    check(approxVec(r.compensated_position_gimbal_m, cv::Vec3d(4.10, 0.95, 0.52)),
          "compensated == (4.10,0.95,0.52)");

    // Cross-check against the pure VisionSolution for the same raw position.
    const auto expected = hnu25::anti_drone::makeVisionSolution(
        cv::Vec3d(4.0, 1.0, 0.5), cfg.compensation);
    check(expected.valid, "reference solution valid");
    check(approxVec(r.compensated_position_gimbal_m,
                    expected.xyz_gimbal_compensated),
          "compensated matches reference");
    check(approx(r.predicted_yaw_rad, expected.yaw_compensated_rad),
          "yaw matches reference");
    check(approx(r.predicted_pitch_rad, expected.pitch_compensated_rad),
          "pitch matches reference");
}

// Test 4: a stationary target over 60 frames stays finite and confirmed.
void testStationary60Frames() {
    hnu25::anti_drone::DiagnosticPipeline pipeline(baseConfig());
    const auto t0 = Clock::now();

    hnu25::anti_drone::DiagnosticFrameResult last;
    for (int i = 0; i < 60; ++i) {
        const auto ts = t0 + std::chrono::milliseconds(i * 20);
        last = pipeline.update({measurement(4.0, 0.5, 0.4)}, ts);

        check(finiteVec(last.tracked_position_gimbal_m), "tracked finite");
        check(finiteVec(last.velocity_gimbal_m_s), "velocity finite");
        check(finiteVec(last.predicted_position_gimbal_m), "predicted finite");
        check(finiteVec(last.compensated_position_gimbal_m), "compensated finite");
        check(std::isfinite(last.predicted_yaw_rad), "yaw finite");
        check(std::isfinite(last.predicted_pitch_rad), "pitch finite");

        if (i == 0) {
            check(last.track_state == hnu25::anti_drone::TrackState::DETECTING,
                  "frame 0 DETECTING");
        } else {
            check(last.track_state == hnu25::anti_drone::TrackState::TRACKING,
                  "frame >0 TRACKING");
            check(last.track_available && last.prediction_valid &&
                      last.solution_valid,
                  "frame >0 fully valid");
        }
    }

    check(approxVec(last.velocity_gimbal_m_s, cv::Vec3d(0.0, 0.0, 0.0), 1e-9),
          "final velocity ~ (0,0,0)");
    check(approxVec(last.predicted_position_gimbal_m, cv::Vec3d(4.0, 0.5, 0.4)),
          "final predicted ~ (4,0.5,0.4)");
}

// Test 5: 120-frame moving synthetic sequence with deterministic jitter, three
// consecutive misses, and a nearer-reprojection distractor.
void testMoving120Frames() {
    hnu25::anti_drone::DiagnosticPipelineConfig cfg = baseConfig();
    cfg.tracker.max_dt_s = 0.1;
    cfg.tracker.max_association_distance_m = 0.60;
    cfg.prediction.horizon_s = 0.08;

    hnu25::anti_drone::DiagnosticPipeline pipeline(cfg);
    const auto t0 = Clock::now();
    constexpr double kDtS = 0.02;

    const auto truthAt = [](double t) {
        return cv::Vec3d(4.0, -1.0 + 3.0 * t, 0.4);
    };

    hnu25::anti_drone::DiagnosticFrameResult last;
    std::vector<double> prediction_errors;
    prediction_errors.reserve(120);

    for (int i = 0; i < 120; ++i) {
        const double t = i * kDtS;
        const auto ts = t0 + std::chrono::milliseconds(i * 20);

        const bool is_miss = (i == 40 || i == 41 || i == 42);
        std::vector<hnu25::anti_drone::PnpResult> meas;
        if (!is_miss) {
            const cv::Vec3d truth = truthAt(t);
            const double jx = ((i % 5) - 2) * 0.002;
            const double jy = ((i % 7) - 3) * 0.002;
            const double jz = ((i % 3) - 1) * 0.001;
            meas.push_back(measurement(truth[0] + jx, truth[1] + jy,
                                       truth[2] + jz, 1.0));
            if (i >= 15) {
                meas.push_back(measurement(truth[0] + 0.30, truth[1] + 0.30,
                                           truth[2] + 0.10, 0.01));
            }
        }

        last = pipeline.update(meas, ts);

        if (i >= 1) {
            check(last.track_state != hnu25::anti_drone::TrackState::LOST,
                  "no LOST after frame 1");
        }
        check(finiteVec(last.tracked_position_gimbal_m), "tracked finite");
        check(finiteVec(last.velocity_gimbal_m_s), "velocity finite");
        check(finiteVec(last.predicted_position_gimbal_m), "predicted finite");
        check(finiteVec(last.compensated_position_gimbal_m), "compensated finite");
        check(std::isfinite(last.predicted_yaw_rad), "yaw finite");
        check(std::isfinite(last.predicted_pitch_rad), "pitch finite");

        if (last.track_state == hnu25::anti_drone::TrackState::TRACKING ||
            last.track_state == hnu25::anti_drone::TrackState::TEMP_LOST) {
            check(last.prediction_valid && last.solution_valid,
                  "TRACKING/TEMP_LOST -> prediction + solution valid");
        }

        if (is_miss) {
            check(last.track_available, "miss frame track_available");
            check(last.track_state == hnu25::anti_drone::TrackState::TEMP_LOST,
                  "miss frame TEMP_LOST");
            check(!last.measurement_updated, "miss frame measurement_updated false");
        }
        if (i == 43) {
            check(last.track_state == hnu25::anti_drone::TrackState::TRACKING,
                  "frame 43 re-acquired TRACKING");
            check(last.measurement_updated, "frame 43 measurement_updated");
            check(last.missed_count == 0, "frame 43 missed_count == 0");
        }

        if (!is_miss) {
            const double future_t = t + cfg.prediction.horizon_s;
            const cv::Vec3d future_truth = truthAt(future_t);
            const cv::Vec3d err = last.predicted_position_gimbal_m - future_truth;
            const double e = std::sqrt(err[0] * err[0] + err[1] * err[1] +
                                       err[2] * err[2]);
            check(std::isfinite(e), "prediction error finite");
            prediction_errors.push_back(e);
        }
    }

    check(last.track_state == hnu25::anti_drone::TrackState::TRACKING,
          "final frame TRACKING");
    check(std::fabs(last.velocity_gimbal_m_s[0]) < 0.20, "final vx < 0.20");
    check(std::fabs(last.velocity_gimbal_m_s[1] - 3.0) < 0.20,
          "final vy ~ 3.0 m/s");
    check(std::fabs(last.velocity_gimbal_m_s[2]) < 0.20, "final vz < 0.20");

    check(prediction_errors.size() >= 20, ">= 20 prediction samples");
    double sum = 0.0;
    for (std::size_t k = prediction_errors.size() - 20;
         k < prediction_errors.size(); ++k) {
        sum += prediction_errors[k];
    }
    const double avg = sum / 20.0;
    check(avg < 0.10, "average prediction error < 0.10 m");
}

// Test 6: reset() drops back to LOST so the next frame re-acquires.
void testResetPipeline() {
    hnu25::anti_drone::DiagnosticPipeline pipeline(baseConfig());
    const auto t0 = Clock::now();

    pipeline.update({measurement(4.0, 0.0, 0.5)}, t0);
    pipeline.update({measurement(4.0, 0.0, 0.5)},
                    t0 + std::chrono::milliseconds(20));

    pipeline.reset();

    const auto r = pipeline.update(
        {measurement(8.0, 2.0, 1.0)}, t0 + std::chrono::milliseconds(40));
    check(r.track_available, "track_available == true");
    check(r.track_state == hnu25::anti_drone::TrackState::DETECTING,
          "DETECTING after reset");
    check(approxVec(r.tracked_position_gimbal_m, cv::Vec3d(8.0, 2.0, 1.0)),
          "tracked == (8,2,1)");
    check(approxVec(r.velocity_gimbal_m_s, cv::Vec3d(0.0, 0.0, 0.0)),
          "velocity == (0,0,0)");
    check(!r.prediction_valid, "prediction_valid == false after reset");
}

// Test 7: non-finite / negative configuration is rejected at construction.
void testInvalidConfig() {
    {
        hnu25::anti_drone::DiagnosticPipelineConfig cfg = baseConfig();
        cfg.prediction.horizon_s = std::numeric_limits<double>::quiet_NaN();
        bool threw = false;
        try {
            hnu25::anti_drone::DiagnosticPipeline p(cfg);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "NaN horizon throws invalid_argument");
    }
    {
        hnu25::anti_drone::DiagnosticPipelineConfig cfg = baseConfig();
        cfg.prediction.horizon_s = -0.1;
        bool threw = false;
        try {
            hnu25::anti_drone::DiagnosticPipeline p(cfg);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "negative horizon throws invalid_argument");
    }
    {
        hnu25::anti_drone::DiagnosticPipelineConfig cfg = baseConfig();
        cfg.compensation.yaw_offset_deg = std::numeric_limits<double>::quiet_NaN();
        bool threw = false;
        try {
            hnu25::anti_drone::DiagnosticPipeline p(cfg);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "NaN compensation throws invalid_argument");
    }
}

// Test 8: a valid prediction whose direction is undefined (x == y == 0) still
// yields prediction_valid but solution_valid == false.
void testValidPredictionInvalidSolution() {
    hnu25::anti_drone::DiagnosticPipelineConfig cfg = baseConfig();
    cfg.prediction.horizon_s = 0.0;

    hnu25::anti_drone::DiagnosticPipeline pipeline(cfg);
    const auto t0 = Clock::now();

    pipeline.update({measurement(0.0, 0.0, 5.0)}, t0);
    const auto r = pipeline.update(
        {measurement(0.0, 0.0, 5.0)}, t0 + std::chrono::milliseconds(20));

    check(r.track_available, "track_available == true");
    check(r.track_state == hnu25::anti_drone::TrackState::TRACKING,
          "track_state == TRACKING");
    check(r.prediction_valid, "prediction_valid == true");
    check(!r.solution_valid, "solution_valid == false");
}

// Test 9: the adapter maps every tracker / prediction / compensation field
// exactly, without touching detector or calibration.
void testExactConfigMapping() {
    hnu25::anti_drone::AntiDroneConfig config;
    config.tracker.min_detect_count = 4;
    config.tracker.max_missed_count = 8;
    config.tracker.max_dt_s = 0.12;
    config.tracker.max_association_distance_m = 0.55;
    config.tracker.position_gain = 0.75;
    config.tracker.velocity_gain = 0.30;
    config.prediction.horizon_s = 0.09;
    config.vision_compensation.yaw_offset_deg = 1.5;
    config.vision_compensation.pitch_offset_deg = -0.8;
    config.vision_compensation.x_offset_m = 0.01;
    config.vision_compensation.y_offset_m = -0.02;
    config.vision_compensation.z_offset_m = 0.03;

    const auto pipeline_config =
        hnu25::anti_drone::makeDiagnosticPipelineConfig(config);

    const auto& tracker = pipeline_config.tracker;
    check(tracker.min_detect_count == 4, "min_detect_count == 4");
    check(tracker.max_missed_count == 8, "max_missed_count == 8");
    check(approx(tracker.max_dt_s, 0.12), "max_dt_s == 0.12");
    check(approx(tracker.max_association_distance_m, 0.55),
          "max_association_distance_m == 0.55");
    check(approx(tracker.position_gain, 0.75), "position_gain == 0.75");
    check(approx(tracker.velocity_gain, 0.30), "velocity_gain == 0.30");

    check(approx(pipeline_config.prediction.horizon_s, 0.09),
          "horizon_s == 0.09");

    const auto& comp = pipeline_config.compensation;
    check(approx(comp.yaw_offset_deg, 1.5), "yaw_offset_deg == 1.5");
    check(approx(comp.pitch_offset_deg, -0.8), "pitch_offset_deg == -0.8");
    check(approx(comp.x_offset_m, 0.01), "x_offset_m == 0.01");
    check(approx(comp.y_offset_m, -0.02), "y_offset_m == -0.02");
    check(approx(comp.z_offset_m, 0.03), "z_offset_m == 0.03");
}

// Test 10: a default AntiDroneConfig maps to the current struct defaults.
void testDefaultMapping() {
    hnu25::anti_drone::AntiDroneConfig config;

    const auto pipeline_config =
        hnu25::anti_drone::makeDiagnosticPipelineConfig(config);

    const auto& tracker = pipeline_config.tracker;
    check(tracker.min_detect_count == 2, "min_detect_count default == 2");
    check(tracker.max_missed_count == 5, "max_missed_count default == 5");
    check(approx(tracker.max_dt_s, 0.25), "max_dt_s default == 0.25");
    check(approx(tracker.max_association_distance_m, 1.0),
          "max_association_distance_m default == 1.0");
    check(approx(tracker.position_gain, 1.0), "position_gain default == 1.0");
    check(approx(tracker.velocity_gain, 1.0), "velocity_gain default == 1.0");
    check(pipeline_config.prediction.horizon_s == 0.0,
          "horizon_s default == 0.0");

    const auto& comp = pipeline_config.compensation;
    check(comp.yaw_offset_deg == 0.0, "yaw default == 0");
    check(comp.pitch_offset_deg == 0.0, "pitch default == 0");
    check(comp.x_offset_m == 0.0, "x default == 0");
    check(comp.y_offset_m == 0.0, "y default == 0");
    check(comp.z_offset_m == 0.0, "z default == 0");
}

// Test 11: the adapter is a read-only pure conversion and never mutates the
// source config.
void testSourceConfigUnchanged() {
    hnu25::anti_drone::AntiDroneConfig config;
    config.tracker.min_detect_count = 4;
    config.tracker.max_missed_count = 8;
    config.tracker.max_dt_s = 0.12;
    config.tracker.max_association_distance_m = 0.55;
    config.tracker.position_gain = 0.75;
    config.tracker.velocity_gain = 0.30;
    config.prediction.horizon_s = 0.09;
    config.vision_compensation.yaw_offset_deg = 1.5;
    config.vision_compensation.pitch_offset_deg = -0.8;
    config.vision_compensation.x_offset_m = 0.01;
    config.vision_compensation.y_offset_m = -0.02;
    config.vision_compensation.z_offset_m = 0.03;

    const auto tracker_before = config.tracker;
    const auto prediction_before = config.prediction;
    const auto compensation_before = config.vision_compensation;

    hnu25::anti_drone::makeDiagnosticPipelineConfig(config);

    check(config.tracker.min_detect_count == tracker_before.min_detect_count,
          "tracker.min_detect_count unchanged");
    check(config.tracker.max_missed_count == tracker_before.max_missed_count,
          "tracker.max_missed_count unchanged");
    check(approx(config.tracker.max_dt_s, tracker_before.max_dt_s),
          "tracker.max_dt_s unchanged");
    check(approx(config.tracker.max_association_distance_m,
                 tracker_before.max_association_distance_m),
          "tracker.max_association_distance_m unchanged");
    check(approx(config.tracker.position_gain, tracker_before.position_gain),
          "tracker.position_gain unchanged");
    check(approx(config.tracker.velocity_gain, tracker_before.velocity_gain),
          "tracker.velocity_gain unchanged");
    check(approx(config.prediction.horizon_s, prediction_before.horizon_s),
          "prediction.horizon_s unchanged");
    check(approx(config.vision_compensation.yaw_offset_deg,
                 compensation_before.yaw_offset_deg),
          "compensation.yaw_offset_deg unchanged");
    check(approx(config.vision_compensation.pitch_offset_deg,
                 compensation_before.pitch_offset_deg),
          "compensation.pitch_offset_deg unchanged");
    check(approx(config.vision_compensation.x_offset_m,
                 compensation_before.x_offset_m),
          "compensation.x_offset_m unchanged");
    check(approx(config.vision_compensation.y_offset_m,
                 compensation_before.y_offset_m),
          "compensation.y_offset_m unchanged");
    check(approx(config.vision_compensation.z_offset_m,
                 compensation_before.z_offset_m),
          "compensation.z_offset_m unchanged");
}

// Test 12: the adapter result constructs a working DiagnosticPipeline.
void testAdapterConstructsPipeline() {
    hnu25::anti_drone::AntiDroneConfig config;
    config.tracker.min_detect_count = 2;
    config.tracker.max_missed_count = 5;
    config.tracker.max_dt_s = 0.2;
    config.tracker.max_association_distance_m = 0.8;
    config.tracker.position_gain = 1.0;
    config.tracker.velocity_gain = 0.2;
    config.prediction.horizon_s = 0.05;
    // vision_compensation stays all-zero.

    const auto pipeline_config =
        hnu25::anti_drone::makeDiagnosticPipelineConfig(config);
    hnu25::anti_drone::DiagnosticPipeline pipeline(pipeline_config);

    const auto t0 = Clock::now();
    pipeline.update({measurement(4.0, 0.0, 0.5)}, t0);
    const auto r = pipeline.update(
        {measurement(4.0, 0.0, 0.5)}, t0 + std::chrono::milliseconds(20));

    check(r.track_state == hnu25::anti_drone::TrackState::TRACKING,
          "track_state == TRACKING");
    check(r.prediction_valid, "prediction_valid == true");
    check(r.solution_valid, "solution_valid == true");
    check(approx(r.prediction_horizon_s, 0.05), "horizon_s ~= 0.05");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"detecting gating", testDetectingGating},
        {"confirmed static pipeline", testConfirmedStaticPipeline},
        {"compensation propagated once", testCompensationPropagatedOnce},
        {"stationary 60 frames", testStationary60Frames},
        {"moving 120 frames", testMoving120Frames},
        {"reset pipeline", testResetPipeline},
        {"invalid config", testInvalidConfig},
        {"valid prediction invalid solution", testValidPredictionInvalidSolution},
        {"exact config mapping", testExactConfigMapping},
        {"default mapping", testDefaultMapping},
        {"source config unchanged", testSourceConfigUnchanged},
        {"adapter constructs pipeline", testAdapterConstructsPipeline},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone diagnostic pipeline tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
