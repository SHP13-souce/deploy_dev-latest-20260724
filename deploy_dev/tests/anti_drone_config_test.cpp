#include "anti_drone/config.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

// Writes a YAML string to a temporary file and returns its path. The caller
// is responsible for removing it afterwards.
std::filesystem::path writeTempYaml(const std::string& name,
                                    const std::string& content) {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("anti_drone_config_test_" + name + ".yaml");
    std::ofstream out(path);
    out << content;
    return path;
}

// Test 1: overriding min_cv_score and all five compensation fields; other
// traditional_detector fields keep their defaults.
void testCompensationOverride() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.61\n"
        "\n"
        "vision_compensation:\n"
        "  yaw_offset_deg: 1.25\n"
        "  pitch_offset_deg: -0.75\n"
        "  x_offset_m: 0.01\n"
        "  y_offset_m: -0.02\n"
        "  z_offset_m: 0.03\n";

    const auto path = writeTempYaml("override", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(approx(config.traditional_detector.min_cv_score, 0.61),
          "min_cv_score overridden to 0.61");
    check(config.traditional_detector.white_saturation_max == 80,
          "unspecified white_saturation_max keeps default 80");

    const auto& vc = config.vision_compensation;
    check(approx(vc.yaw_offset_deg, 1.25), "yaw_offset_deg == 1.25");
    check(approx(vc.pitch_offset_deg, -0.75), "pitch_offset_deg == -0.75");
    check(approx(vc.x_offset_m, 0.01), "x_offset_m == 0.01");
    check(approx(vc.y_offset_m, -0.02), "y_offset_m == -0.02");
    check(approx(vc.z_offset_m, 0.03), "z_offset_m == 0.03");
}

// Test 2: a missing vision_compensation section keeps all-zero defaults.
void testCompensationMissing() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.60\n";

    const auto path = writeTempYaml("missing", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& vc = config.vision_compensation;
    check(vc.yaw_offset_deg == 0.0, "yaw_offset_deg default == 0");
    check(vc.pitch_offset_deg == 0.0, "pitch_offset_deg default == 0");
    check(vc.x_offset_m == 0.0, "x_offset_m default == 0");
    check(vc.y_offset_m == 0.0, "y_offset_m default == 0");
    check(vc.z_offset_m == 0.0, "z_offset_m default == 0");
}

// Test 3: the legacy loadTraditionalDetectorConfig wrapper still works.
void testOldApiCompatible() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.61\n"
        "\n"
        "vision_compensation:\n"
        "  yaw_offset_deg: 1.25\n"
        "  pitch_offset_deg: -0.75\n"
        "  x_offset_m: 0.01\n"
        "  y_offset_m: -0.02\n"
        "  z_offset_m: 0.03\n";

    const auto path = writeTempYaml("oldapi", yaml);
    const auto config =
        hnu25::anti_drone::loadTraditionalDetectorConfig(path);
    std::filesystem::remove(path);

    check(approx(config.min_cv_score, 0.61),
          "old API returns min_cv_score 0.61");
}

// Test 4: a non-finite compensation value must throw std::runtime_error.
void testInvalidCompensation() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "vision_compensation:\n"
        "  yaw_offset_deg: .nan\n";

    const auto path = writeTempYaml("nan", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);

    check(threw, "non-finite compensation throws std::runtime_error");
}

// Test 5: a non-mapping vision_compensation must throw std::runtime_error.
void testWrongNodeType() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "vision_compensation: 123\n";

    const auto path = writeTempYaml("wrongtype", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);

    check(threw, "non-mapping vision_compensation throws std::runtime_error");
}

// Test 6: a complete root-level calibration section loads into
// CalibrationConfig, while compensation stays independent.
void testFullCalibrationLoad() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.61\n"
        "\n"
        "camera_matrix:\n"
        "  [900.0, 0.0, 320.0,\n"
        "   0.0, 910.0, 240.0,\n"
        "   0.0, 0.0, 1.0]\n"
        "\n"
        "distort_coeffs:\n"
        "  [0.10, -0.05, 0.001, -0.002, 0.01]\n"
        "\n"
        "R_camera2gimbal:\n"
        "  [0.0, -1.0, 0.0,\n"
        "   1.0,  0.0, 0.0,\n"
        "   0.0,  0.0, 1.0]\n"
        "\n"
        "t_camera2gimbal:\n"
        "  [0.10, -0.20, 0.30]\n"
        "\n"
        "R_gimbal2imubody:\n"
        "  [1.0, 0.0,  0.0,\n"
        "   0.0, 0.0, -1.0,\n"
        "   0.0, 1.0,  0.0]\n"
        "\n"
        "max_reprojection_error_px: 2.5\n"
        "\n"
        "vision_compensation:\n"
        "  yaw_offset_deg: 1.25\n";

    const auto path = writeTempYaml("calib_full", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.calibration.has_value(), "calibration present");

    const auto& pnp = config.calibration->pnp;
    check(approx(pnp.camera_matrix(0, 0), 900.0), "camera_matrix(0,0) == 900");
    check(approx(pnp.camera_matrix(1, 1), 910.0), "camera_matrix(1,1) == 910");
    check(approx(pnp.camera_matrix(0, 2), 320.0), "camera_matrix(0,2) == 320");
    check(approx(pnp.camera_matrix(1, 2), 240.0), "camera_matrix(1,2) == 240");

    check(pnp.distort_coeffs.size() == 5, "distort_coeffs size == 5");
    check(approx(pnp.distort_coeffs[0], 0.10), "distort_coeffs[0] == 0.10");
    check(approx(pnp.distort_coeffs[4], 0.01), "distort_coeffs[4] == 0.01");

    check(approx(pnp.R_camera2gimbal(0, 1), -1.0),
          "R_camera2gimbal(0,1) == -1");
    check(approx(pnp.R_camera2gimbal(1, 0), 1.0),
          "R_camera2gimbal(1,0) == 1");

    check(approx(pnp.t_camera2gimbal[0], 0.10), "t_camera2gimbal[0] == 0.10");
    check(approx(pnp.t_camera2gimbal[1], -0.20), "t_camera2gimbal[1] == -0.20");
    check(approx(pnp.t_camera2gimbal[2], 0.30), "t_camera2gimbal[2] == 0.30");

    const auto& R_g2i = config.calibration->R_gimbal2imubody;
    check(approx(R_g2i(1, 2), -1.0), "R_gimbal2imubody(1,2) == -1");
    check(approx(R_g2i(2, 1), 1.0), "R_gimbal2imubody(2,1) == 1");

    check(approx(pnp.max_reprojection_error_px, 2.5),
          "max_reprojection_error_px == 2.5");
    check(approx(pnp.target_width_m, 0.50), "target_width_m == 0.50");
    check(approx(pnp.target_height_m, 0.50), "target_height_m == 0.50");

    check(approx(config.vision_compensation.yaw_offset_deg, 1.25),
          "yaw_offset_deg == 1.25 (calibration + compensation coexist)");
}

// Test 7: no calibration keys -> calibration is nullopt, load succeeds.
void testCalibrationAbsent() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.60\n"
        "\n"
        "vision_compensation:\n"
        "  yaw_offset_deg: 0.5\n";

    const auto path = writeTempYaml("calib_absent", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(!config.calibration.has_value(), "calibration absent");
    check(approx(config.vision_compensation.yaw_offset_deg, 0.5),
          "vision compensation still loads");
}

// Test 8: a partial calibration (only camera_matrix) must be rejected.
void testPartialCalibrationRejected() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "camera_matrix:\n"
        "  [800, 0, 320,\n"
        "   0, 800, 240,\n"
        "   0, 0, 1]\n";

    const auto path = writeTempYaml("calib_partial", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);

    check(threw, "partial calibration throws std::runtime_error");
}

// Test 9: an 8-element rotation matrix must be rejected.
void testInvalidMatrixSize() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "camera_matrix:\n"
        "  [800, 0, 320,\n"
        "   0, 800, 240,\n"
        "   0, 0, 1]\n"
        "\n"
        "distort_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "\n"
        "R_camera2gimbal:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0]\n"
        "\n"
        "t_camera2gimbal: [0.0, 0.0, 0.0]\n"
        "\n"
        "R_gimbal2imubody:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 1]\n";

    const auto path = writeTempYaml("calib_badmat", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);

    check(threw, "8-element R_camera2gimbal throws std::runtime_error");
}

// Test 10: fx == 0 in the camera matrix must be rejected.
void testInvalidFx() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "camera_matrix:\n"
        "  [0.0, 0, 320,\n"
        "   0, 800, 240,\n"
        "   0, 0, 1]\n"
        "\n"
        "distort_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "\n"
        "R_camera2gimbal:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 1]\n"
        "\n"
        "t_camera2gimbal: [0.0, 0.0, 0.0]\n"
        "\n"
        "R_gimbal2imubody:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 1]\n";

    const auto path = writeTempYaml("calib_badfx", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);

    check(threw, "fx == 0 throws std::runtime_error");
}

// Test 11: omitting max_reprojection_error_px keeps the 5.0 default.
void testDefaultReprojectionThreshold() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "camera_matrix:\n"
        "  [800, 0, 320,\n"
        "   0, 800, 240,\n"
        "   0, 0, 1]\n"
        "\n"
        "distort_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "\n"
        "R_camera2gimbal:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 1]\n"
        "\n"
        "t_camera2gimbal: [0.0, 0.0, 0.0]\n"
        "\n"
        "R_gimbal2imubody:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 1]\n";

    const auto path = writeTempYaml("calib_default", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.calibration.has_value(), "calibration present");
    check(approx(config.calibration->pnp.max_reprojection_error_px, 5.0),
          "max_reprojection_error_px defaults to 5.0");
}

// Test 12: full tracker + prediction override coexists with compensation.
void testFullTrackerPredictionOverride() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.60\n"
        "\n"
        "tracker:\n"
        "  min_detect_count: 3\n"
        "  max_missed_count: 7\n"
        "  max_dt_s: 0.15\n"
        "  max_association_distance_m: 0.65\n"
        "  position_gain: 0.8\n"
        "  velocity_gain: 0.25\n"
        "\n"
        "prediction:\n"
        "  horizon_s: 0.075\n"
        "\n"
        "vision_compensation:\n"
        "  yaw_offset_deg: 1.0\n";

    const auto path = writeTempYaml("tracker_pred", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& tracker = config.tracker;
    check(tracker.min_detect_count == 3, "min_detect_count == 3");
    check(tracker.max_missed_count == 7, "max_missed_count == 7");
    check(approx(tracker.max_dt_s, 0.15), "max_dt_s == 0.15");
    check(approx(tracker.max_association_distance_m, 0.65),
          "max_association_distance_m == 0.65");
    check(approx(tracker.position_gain, 0.8), "position_gain == 0.8");
    check(approx(tracker.velocity_gain, 0.25), "velocity_gain == 0.25");
    check(approx(config.prediction.horizon_s, 0.075), "horizon_s == 0.075");
    check(approx(config.vision_compensation.yaw_offset_deg, 1.0),
          "yaw_offset_deg == 1.0 (coexists)");
}

// Test 13: absent tracker/prediction sections keep struct defaults.
void testSectionsAbsentKeepDefaults() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.60\n";

    const auto path = writeTempYaml("defaults", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& tracker = config.tracker;
    check(tracker.min_detect_count == 2, "min_detect_count default == 2");
    check(tracker.max_missed_count == 5, "max_missed_count default == 5");
    check(approx(tracker.max_dt_s, 0.25), "max_dt_s default == 0.25");
    check(approx(tracker.max_association_distance_m, 1.0),
          "max_association_distance_m default == 1.0");
    check(approx(tracker.position_gain, 1.0), "position_gain default == 1.0");
    check(approx(tracker.velocity_gain, 1.0), "velocity_gain default == 1.0");
    check(config.prediction.horizon_s == 0.0, "horizon_s default == 0.0");
}

// Test 14: a partial tracker override changes only that field.
void testPartialTrackerOverride() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.60\n"
        "\n"
        "tracker:\n"
        "  velocity_gain: 0.3\n";

    const auto path = writeTempYaml("partial", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& tracker = config.tracker;
    check(approx(tracker.velocity_gain, 0.3), "velocity_gain == 0.3");
    check(tracker.min_detect_count == 2, "min_detect_count default == 2");
    check(tracker.max_missed_count == 5, "max_missed_count default == 5");
    check(approx(tracker.max_dt_s, 0.25), "max_dt_s default == 0.25");
    check(approx(tracker.max_association_distance_m, 1.0),
          "max_association_distance_m default == 1.0");
    check(approx(tracker.position_gain, 1.0), "position_gain default == 1.0");
}

// Test 15: a non-mapping tracker section must throw std::runtime_error.
void testTrackerWrongNodeType() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "tracker: 123\n";

    const auto path = writeTempYaml("tracker_wrongtype", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);

    check(threw, "non-mapping tracker throws std::runtime_error");
}

// Test 16: each invalid tracker field value must throw std::runtime_error.
void testInvalidTrackerValues() {
    const char* fields[] = {
        "min_detect_count: 0\n",
        "max_missed_count: -1\n",
        "max_dt_s: 0.0\n",
        "max_association_distance_m: -0.1\n",
        "position_gain: 1.1\n",
        "velocity_gain: -0.1\n",
        "velocity_gain: .nan\n",
    };

    int index = 0;
    for (const char* field : fields) {
        const std::string yaml =
            std::string("traditional_detector:\n"
                        "  min_cv_score: 0.55\n"
                        "\n"
                        "tracker:\n"
                        "  ") +
            field;

        const auto path =
            writeTempYaml("invalid_tracker_" + std::to_string(index), yaml);
        bool threw = false;
        try {
            hnu25::anti_drone::loadAntiDroneConfig(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::filesystem::remove(path);

        check(threw, std::string("invalid tracker value throws: ") + field);
        ++index;
    }
}

// Test 17: a non-mapping prediction section must throw std::runtime_error.
void testPredictionWrongNodeType() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "prediction: 123\n";

    const auto path = writeTempYaml("prediction_wrongtype", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);

    check(threw, "non-mapping prediction throws std::runtime_error");
}

// Test 18: negative and non-finite prediction horizons must throw.
void testInvalidPredictionHorizon() {
    {
        const std::string yaml =
            "traditional_detector:\n"
            "  min_cv_score: 0.55\n"
            "\n"
            "prediction:\n"
            "  horizon_s: -0.01\n";

        const auto path = writeTempYaml("prediction_neg", yaml);
        bool threw = false;
        try {
            hnu25::anti_drone::loadAntiDroneConfig(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::filesystem::remove(path);
        check(threw, "negative horizon throws std::runtime_error");
    }
    {
        const std::string yaml =
            "traditional_detector:\n"
            "  min_cv_score: 0.55\n"
            "\n"
            "prediction:\n"
            "  horizon_s: .nan\n";

        const auto path = writeTempYaml("prediction_nan", yaml);
        bool threw = false;
        try {
            hnu25::anti_drone::loadAntiDroneConfig(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::filesystem::remove(path);
        check(threw, "NaN horizon throws std::runtime_error");
    }
}

// Test 19: horizon_s == 0.0 is a valid way to disable prediction.
void testZeroPredictionHorizonValid() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "prediction:\n"
        "  horizon_s: 0.0\n";

    const auto path = writeTempYaml("prediction_zero", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.prediction.horizon_s == 0.0, "horizon_s == 0.0");
}

// Test 20: an illegal telemetry mode must throw std::runtime_error.
void testInvalidTelemetryMode() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  mode: \"bogus\"\n";

    const auto path = writeTempYaml("telemetry_mode", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "illegal telemetry mode throws std::runtime_error");
}

// Test 21: an illegal baud rate must throw std::runtime_error.
void testInvalidTelemetryBaud() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  baud_rate: 12345\n";

    const auto path = writeTempYaml("telemetry_baud", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "illegal telemetry baud_rate throws std::runtime_error");
}

// Test 22: SERIAL_DIAGNOSTIC with an empty device must throw.
void testSerialDiagnosticEmptyDevice() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  mode: \"serial_diagnostic\"\n"
        "  device: \"\"\n";

    const auto path = writeTempYaml("telemetry_empty_device", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "serial_diagnostic with empty device throws");
}

// Test 23: max_consecutive_failures <= 0 must throw.
void testInvalidTelemetryMaxFailures() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  max_consecutive_failures: 0\n";

    const auto path = writeTempYaml("telemetry_max_failures", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "max_consecutive_failures <= 0 throws");
}

// Test 24: a valid SERIAL_DIAGNOSTIC config loads with every field set.
void testSerialDiagnosticLoad() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  mode: \"serial_diagnostic\"\n"
        "  device: \"/dev/ttyUSB0\"\n"
        "  baud_rate: 57600\n"
        "  max_consecutive_failures: 7\n"
        "  flush_after_write: true\n";

    const auto path = writeTempYaml("telemetry_serial", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.telemetry.mode ==
              hnu25::anti_drone::TelemetryTransportMode::SERIAL_DIAGNOSTIC,
          "telemetry mode == SERIAL_DIAGNOSTIC");
    check(config.telemetry.device == "/dev/ttyUSB0",
          "telemetry device == /dev/ttyUSB0");
    check(config.telemetry.baud_rate == 57600, "telemetry baud_rate == 57600");
    check(config.telemetry.max_consecutive_failures == 7,
          "telemetry max_consecutive_failures == 7");
    check(config.telemetry.flush_after_write == true,
          "telemetry flush_after_write == true");
}

// Test 25: a valid SERIAL config loads with every field set.
void testSerialLoad() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  mode: \"serial\"\n"
        "  device: \"/dev/ttyUSB0\"\n"
        "  baud_rate: 115200\n"
        "  max_consecutive_failures: 5\n"
        "  flush_after_write: false\n";

    const auto path = writeTempYaml("telemetry_serial_prod", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.telemetry.mode ==
              hnu25::anti_drone::TelemetryTransportMode::SERIAL,
          "telemetry mode == SERIAL");
    check(config.telemetry.device == "/dev/ttyUSB0",
          "telemetry device == /dev/ttyUSB0");
    check(config.telemetry.baud_rate == 115200, "telemetry baud_rate == 115200");
}

// Test 26: SERIAL with an empty device must throw.
void testSerialEmptyDevice() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  mode: \"serial\"\n"
        "  device: \"\"\n";

    const auto path = writeTempYaml("telemetry_serial_empty_device", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "serial with empty device throws");
}

// Test 27: gimbal overrides load, unspecified fields keep defaults.
void testGimbalOverride() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "gimbal:\n"
        "  enable: false\n"
        "  send_speed: true\n"
        "  speed_filter_alpha: 0.7\n";

    const auto path = writeTempYaml("gimbal_override", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& gimbal = config.gimbal;
    check(gimbal.enable == false, "gimbal enable == false");
    check(gimbal.send_speed == true, "gimbal send_speed == true");
    check(approx(gimbal.speed_filter_alpha, 0.7),
          "gimbal speed_filter_alpha == 0.7");
}

// Test 28: absent gimbal section keeps struct defaults.
void testGimbalDefaults() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n";

    const auto path = writeTempYaml("gimbal_defaults", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& gimbal = config.gimbal;
    check(gimbal.enable == true, "gimbal enable default == true");
    check(gimbal.send_speed == true, "gimbal send_speed default == true");
    check(approx(gimbal.speed_filter_alpha, 0.3),
          "gimbal speed_filter_alpha default == 0.3");
}

// Test 29: speed_filter_alpha outside [0,1] or non-finite must throw.
void testInvalidGimbalAlpha() {
    const char* alphas[] = {"1.5", "-0.1", ".nan"};
    int index = 0;
    for (const char* alpha : alphas) {
        const std::string yaml =
            std::string("traditional_detector:\n"
                        "  min_cv_score: 0.55\n"
                        "\n"
                        "gimbal:\n"
                        "  speed_filter_alpha: ") +
            alpha + "\n";

        const auto path =
            writeTempYaml("gimbal_alpha_" + std::to_string(index), yaml);
        bool threw = false;
        try {
            hnu25::anti_drone::loadAntiDroneConfig(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::filesystem::remove(path);

        check(threw,
              std::string("invalid gimbal speed_filter_alpha throws: ") +
                  alpha);
        ++index;
    }
}

// A minimal valid calibration body (four core keys; R_gimbal2imubody is now
// optional and deliberately omitted).
std::string baseCalibrationYaml() {
    return
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "camera_matrix:\n"
        "  [800, 0, 320,\n"
        "   0, 800, 240,\n"
        "   0, 0, 1]\n"
        "\n"
        "distort_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "\n"
        "R_camera2gimbal:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 1]\n"
        "\n"
        "t_camera2gimbal: [0.0, 0.0, 0.0]\n";
}

// Builds a calibration YAML whose R_camera2gimbal is replaced by `r_block`
// (the 3 rows of the YAML sequence), then reports whether loading threw.
bool loadWithRcamera2gimbalThrows(const std::string& r_block) {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "camera_matrix:\n"
        "  [800, 0, 320,\n"
        "   0, 800, 240,\n"
        "   0, 0, 1]\n"
        "\n"
        "distort_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "\n"
        "R_camera2gimbal:\n" +
        r_block +
        "\n"
        "t_camera2gimbal: [0.0, 0.0, 0.0]\n";

    const auto path = writeTempYaml("calib_rmat", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    return threw;
}

// Test 30: write_timeout_ms defaults to 20 when absent.
void testWriteTimeoutDefault() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  mode: \"serial\"\n"
        "  device: \"/dev/ttyUSB0\"\n";

    const auto path = writeTempYaml("write_timeout_default", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.telemetry.write_timeout_ms == 20,
          "write_timeout_ms defaults to 20");
}

// Test 31: write_timeout_ms override loads.
void testWriteTimeoutOverride() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "telemetry:\n"
        "  write_timeout_ms: 35\n";

    const auto path = writeTempYaml("write_timeout_override", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.telemetry.write_timeout_ms == 35,
          "write_timeout_ms overrides to 35");
}

// Test 32: write_timeout_ms <= 0 must throw.
void testInvalidWriteTimeout() {
    const char* values[] = {"0", "-5"};
    int index = 0;
    for (const char* v : values) {
        const std::string yaml =
            std::string("traditional_detector:\n"
                        "  min_cv_score: 0.55\n"
                        "\n"
                        "telemetry:\n"
                        "  write_timeout_ms: ") +
            v + "\n";

        const auto path =
            writeTempYaml("write_timeout_invalid_" + std::to_string(index),
                          yaml);
        bool threw = false;
        try {
            hnu25::anti_drone::loadAntiDroneConfig(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::filesystem::remove(path);

        check(threw, std::string("write_timeout_ms <= 0 throws: ") + v);
        ++index;
    }
}

// Test 33: R_gimbal2imubody is optional; absent keeps identity.
void testCalibrationWithoutGimbal2Imu() {
    const std::string yaml = baseCalibrationYaml();
    const auto path = writeTempYaml("calib_no_g2i", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.calibration.has_value(),
          "calibration present without R_gimbal2imubody");
    const auto& R = config.calibration->R_gimbal2imubody;
    const bool is_eye =
        approx(R(0, 0), 1.0) && approx(R(1, 1), 1.0) && approx(R(2, 2), 1.0) &&
        approx(R(0, 1), 0.0) && approx(R(0, 2), 0.0) && approx(R(1, 0), 0.0) &&
        approx(R(1, 2), 0.0) && approx(R(2, 0), 0.0) && approx(R(2, 1), 0.0);
    check(is_eye, "R_gimbal2imubody defaults to identity when absent");
}

// Test 34: a core calibration set missing t_camera2gimbal must throw.
void testCoreCalibrationPartial() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.55\n"
        "\n"
        "camera_matrix:\n"
        "  [800, 0, 320,\n"
        "   0, 800, 240,\n"
        "   0, 0, 1]\n"
        "\n"
        "distort_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]\n"
        "\n"
        "R_camera2gimbal:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 1]\n";

    const auto path = writeTempYaml("calib_core_partial", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "missing t_camera2gimbal (core set) throws");
}

// Test 35: a valid identity R_camera2gimbal loads.
void testRotationIdentityValid() {
    check(!loadWithRcamera2gimbalThrows(
              "  [1, 0, 0,\n"
              "   0, 1, 0,\n"
              "   0, 0, 1]\n"),
          "identity R_camera2gimbal is a valid rotation");
}

// Test 36: a valid 90deg rotation R_camera2gimbal loads.
void testRotationValid() {
    check(!loadWithRcamera2gimbalThrows(
              "  [0, -1, 0,\n"
              "   1,  0, 0,\n"
              "   0,  0, 1]\n"),
          "90deg-z R_camera2gimbal is a valid rotation");
}

// Test 37: a non-orthogonal matrix must throw.
void testRotationNonOrthogonal() {
    check(loadWithRcamera2gimbalThrows(
              "  [1, 0, 0,\n"
              "   0, 1, 0,\n"
              "   0, 0, 2]\n"),
          "non-orthogonal R_camera2gimbal throws");
}

// Test 38: a reflection (det = -1) matrix must throw.
void testRotationReflection() {
    check(loadWithRcamera2gimbalThrows(
              "  [1, 0,  0,\n"
              "   0, 1,  0,\n"
              "   0, 0, -1]\n"),
          "reflection (det == -1) R_camera2gimbal throws");
}

// Test 39: a NaN matrix must throw.
void testRotationNan() {
    check(loadWithRcamera2gimbalThrows(
              "  [1, 0, 0,\n"
              "   0, 1, 0,\n"
              "   0, 0, .nan]\n"),
          "NaN R_camera2gimbal throws");
}

// Test 40: an invalid R_gimbal2imubody (non-rotation) must throw.
void testInvalidGimbal2ImuRotation() {
    const std::string yaml =
        baseCalibrationYaml() +
        "\n"
        "R_gimbal2imubody:\n"
        "  [1, 0, 0,\n"
        "   0, 1, 0,\n"
        "   0, 0, 2]\n";

    const auto path = writeTempYaml("calib_bad_g2i", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "non-rotation R_gimbal2imubody throws");
}

// Test 41: calibration resolution defaults to 0/0 (no check).
void testCalibrationResolutionDefaults() {
    const std::string yaml = baseCalibrationYaml();
    const auto path = writeTempYaml("calib_res_default", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.calibration.has_value(), "calibration present");
    check(config.calibration->calibration_image_width == 0 &&
              config.calibration->calibration_image_height == 0,
          "resolution defaults to 0/0");
    check(hnu25::anti_drone::calibrationResolutionMatches(
              *config.calibration, 640, 480),
          "0/0 pins nothing: any runtime size matches");
}

// Test 42: a valid resolution loads and matches exactly.
void testCalibrationResolutionValid() {
    const std::string yaml =
        baseCalibrationYaml() +
        "\n"
        "calibration_image_width: 1440\n"
        "calibration_image_height: 1080\n";

    const auto path = writeTempYaml("calib_res_valid", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    check(config.calibration->calibration_image_width == 1440 &&
              config.calibration->calibration_image_height == 1080,
          "resolution loaded as 1440x1080");
    check(hnu25::anti_drone::calibrationResolutionMatches(
              *config.calibration, 1440, 1080),
          "1440x1080 runtime matches");
    check(!hnu25::anti_drone::calibrationResolutionMatches(
              *config.calibration, 720, 540),
          "720x540 runtime does not match");
}

// Test 43: only width configured -> config error.
void testCalibrationResolutionWidthOnly() {
    const std::string yaml =
        baseCalibrationYaml() +
        "\n"
        "calibration_image_width: 1440\n";

    const auto path = writeTempYaml("calib_res_width_only", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "width-only resolution throws");
}

// Test 44: only height configured -> config error.
void testCalibrationResolutionHeightOnly() {
    const std::string yaml =
        baseCalibrationYaml() +
        "\n"
        "calibration_image_height: 1080\n";

    const auto path = writeTempYaml("calib_res_height_only", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "height-only resolution throws");
}

// Test 45: concentric-ring config fields keep production defaults when absent.
void testRingDefaults() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.65\n";

    const auto path = writeTempYaml("ring_defaults", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& td = config.traditional_detector;
    check(td.require_bullseye == true, "require_bullseye default == true");
    check(td.require_valid_corners == true,
          "require_valid_corners default == true");
    check(td.ring_pattern_enabled == true,
          "ring_pattern_enabled default == true");
    check(td.ring_warp_size == 192, "ring_warp_size default == 192");
    check(td.ring_radial_bins == 48, "ring_radial_bins default == 48");
    check(approx(td.ring_red_low, 0.25), "ring_red_low default == 0.25");
    check(approx(td.ring_red_high, 0.55), "ring_red_high default == 0.55");
    check(approx(td.ring_center_radius_ratio, 0.08),
          "ring_center_radius_ratio default == 0.08");
    check(approx(td.ring_center_red_min, 0.50),
          "ring_center_red_min default == 0.50");
    check(td.ring_min_transitions == 4, "ring_min_transitions default == 4");
    check(approx(td.ring_outer_red_max, 0.40),
          "ring_outer_red_max default == 0.40");
    check(approx(td.min_ring_score, 0.60), "min_ring_score default == 0.60");
    check(approx(td.ring_weight, 0.55), "ring_weight default == 0.55");
}

// Test 46: concentric-ring overrides load through the real production API.
void testRingOverrideLoad() {
    const std::string yaml =
        "traditional_detector:\n"
        "  min_cv_score: 0.65\n"
        "  require_bullseye: false\n"
        "  require_valid_corners: false\n"
        "  ring_pattern_enabled: false\n"
        "  ring_warp_size: 128\n"
        "  ring_radial_bins: 24\n"
        "  ring_red_low: 0.20\n"
        "  ring_red_high: 0.60\n"
        "  ring_center_radius_ratio: 0.10\n"
        "  ring_center_red_min: 0.45\n"
        "  ring_min_transitions: 3\n"
        "  ring_outer_red_max: 0.35\n"
        "  min_ring_score: 0.55\n"
        "  ring_weight: 0.4\n";

    const auto path = writeTempYaml("ring_override", yaml);
    const auto config = hnu25::anti_drone::loadAntiDroneConfig(path);
    std::filesystem::remove(path);

    const auto& td = config.traditional_detector;
    check(td.require_bullseye == false, "require_bullseye overridden to false");
    check(td.require_valid_corners == false,
          "require_valid_corners overridden to false");
    check(td.ring_pattern_enabled == false,
          "ring_pattern_enabled overridden to false");
    check(td.ring_warp_size == 128, "ring_warp_size == 128");
    check(td.ring_radial_bins == 24, "ring_radial_bins == 24");
    check(approx(td.ring_red_low, 0.20), "ring_red_low == 0.20");
    check(approx(td.ring_red_high, 0.60), "ring_red_high == 0.60");
    check(approx(td.ring_center_radius_ratio, 0.10),
          "ring_center_radius_ratio == 0.10");
    check(approx(td.ring_center_red_min, 0.45), "ring_center_red_min == 0.45");
    check(td.ring_min_transitions == 3, "ring_min_transitions == 3");
    check(approx(td.ring_outer_red_max, 0.35), "ring_outer_red_max == 0.35");
    check(approx(td.min_ring_score, 0.55), "min_ring_score == 0.55");
    check(approx(td.ring_weight, 0.40), "ring_weight == 0.40");
}

// Test 47: each invalid concentric-ring value must throw std::runtime_error.
void testInvalidRingValues() {
    const char* fields[] = {
        "ring_warp_size: 63\n",
        "ring_radial_bins: 7\n",
        "ring_red_low: -0.1\n",
        "ring_red_low: 0.6\n",   // low >= high (high stays default 0.55)
        "ring_red_high: 1.1\n",
        "ring_center_radius_ratio: 0.0\n",
        "ring_center_radius_ratio: 0.5\n",
        "ring_center_red_min: -0.1\n",
        "ring_center_red_min: 1.1\n",
        "ring_min_transitions: 0\n",
        "ring_outer_red_max: -0.1\n",
        "ring_outer_red_max: 1.1\n",
        "min_ring_score: -0.1\n",
        "min_ring_score: 1.1\n",
        "ring_weight: -0.1\n",
    };

    int index = 0;
    for (const char* field : fields) {
        const std::string yaml =
            std::string("traditional_detector:\n"
                        "  min_cv_score: 0.65\n"
                        "  ") +
            field;

        const auto path =
            writeTempYaml("invalid_ring_" + std::to_string(index), yaml);
        bool threw = false;
        try {
            hnu25::anti_drone::loadAntiDroneConfig(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        std::filesystem::remove(path);

        check(threw, std::string("invalid ring value throws: ") + field);
        ++index;
    }
}

// Test 48: all three weights zero must throw (normalization would divide by 0).
void testAllWeightsZeroRejected() {
    const std::string yaml =
        "traditional_detector:\n"
        "  geometry_weight: 0.0\n"
        "  color_weight: 0.0\n"
        "  ring_weight: 0.0\n";

    const auto path = writeTempYaml("zero_weights", yaml);
    bool threw = false;
    try {
        hnu25::anti_drone::loadAntiDroneConfig(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    std::filesystem::remove(path);
    check(threw, "all-zero weights throw std::runtime_error");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"compensation override", testCompensationOverride},
        {"compensation missing", testCompensationMissing},
        {"old api compatible", testOldApiCompatible},
        {"invalid compensation", testInvalidCompensation},
        {"wrong node type", testWrongNodeType},
        {"full calibration load", testFullCalibrationLoad},
        {"calibration absent", testCalibrationAbsent},
        {"partial calibration rejected", testPartialCalibrationRejected},
        {"invalid matrix size", testInvalidMatrixSize},
        {"invalid fx", testInvalidFx},
        {"default reprojection threshold", testDefaultReprojectionThreshold},
        {"full tracker prediction override", testFullTrackerPredictionOverride},
        {"sections absent keep defaults", testSectionsAbsentKeepDefaults},
        {"partial tracker override", testPartialTrackerOverride},
        {"tracker wrong node type", testTrackerWrongNodeType},
        {"invalid tracker values", testInvalidTrackerValues},
        {"prediction wrong node type", testPredictionWrongNodeType},
        {"invalid prediction horizon", testInvalidPredictionHorizon},
        {"zero prediction horizon valid", testZeroPredictionHorizonValid},
        {"invalid telemetry mode", testInvalidTelemetryMode},
        {"invalid telemetry baud", testInvalidTelemetryBaud},
        {"serial diagnostic empty device", testSerialDiagnosticEmptyDevice},
        {"invalid telemetry max failures", testInvalidTelemetryMaxFailures},
        {"serial diagnostic load", testSerialDiagnosticLoad},
        {"serial load", testSerialLoad},
        {"serial empty device", testSerialEmptyDevice},
        {"gimbal override", testGimbalOverride},
        {"gimbal defaults", testGimbalDefaults},
        {"invalid gimbal alpha", testInvalidGimbalAlpha},
        {"write timeout default", testWriteTimeoutDefault},
        {"write timeout override", testWriteTimeoutOverride},
        {"invalid write timeout", testInvalidWriteTimeout},
        {"calibration without gimbal2imu", testCalibrationWithoutGimbal2Imu},
        {"core calibration partial", testCoreCalibrationPartial},
        {"rotation identity valid", testRotationIdentityValid},
        {"rotation valid", testRotationValid},
        {"rotation non-orthogonal", testRotationNonOrthogonal},
        {"rotation reflection", testRotationReflection},
        {"rotation nan", testRotationNan},
        {"invalid gimbal2imu rotation", testInvalidGimbal2ImuRotation},
        {"calibration resolution defaults", testCalibrationResolutionDefaults},
        {"calibration resolution valid", testCalibrationResolutionValid},
        {"calibration resolution width only", testCalibrationResolutionWidthOnly},
        {"calibration resolution height only", testCalibrationResolutionHeightOnly},
        {"ring defaults", testRingDefaults},
        {"ring override load", testRingOverrideLoad},
        {"invalid ring values", testInvalidRingValues},
        {"all-zero weights rejected", testAllWeightsZeroRejected},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone config tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
