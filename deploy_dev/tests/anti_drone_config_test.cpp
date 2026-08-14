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
