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
