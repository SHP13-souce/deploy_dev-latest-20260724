#include "anti_drone/camera_extrinsic_calibration_math.hpp"
#include "anti_drone/camera_extrinsic_calibration_validation.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Camera -> gimbal extrinsic (hand-eye) calibration tool.
//
//   anti_drone_camera_extrinsic_calibration \
//       <intrinsics_yaml> <samples_csv> <gimbal_kinematics_yaml> \
//       <board_cols> <board_rows> <square_size_m> [output_yaml]
//
// This is an OFFLINE tool: it reads already-captured chessboard images plus a
// CSV of recorded gimbal yaw/pitch angles. It does NOT open a camera, a serial
// port, or any control / telemetry path, and it does NOT touch anti_drone.yaml.
//
// It solves the standard hand-eye problem with a FIXED chessboard and a MOVING
// camera (mounted on the gimbal):
//
//   ^B T_G(i)  gimbal pose in the base frame      (from recorded yaw/pitch)
//   ^C T_T(i)  fixed board pose in the camera     (solvePnP + Rodrigues)
//
// cv::calibrateHandEye() returns ^G T_C = X, i.e. exactly the production
//
//   p_gimbal = R_camera2gimbal * p_camera + t_camera2gimbal
//
// transform. The result is NOT inverted before being written out.
//
// Exit codes:
//   0 normal
//   1 bad command line
//   2 runtime / config / insufficient-sample / hand-eye failure

namespace {

using hnu25::anti_drone::GimbalKinematics;
using hnu25::anti_drone::GimbalPose;
using hnu25::anti_drone::RotationAxis;

// Formats a double with 15 significant digits (>= 12 as required) so that the
// calibrated rotation / translation survive a YAML round-trip without loss.
std::string formatDouble(double value) {
    std::ostringstream oss;
    oss << std::setprecision(15) << std::defaultfloat << value;
    return oss.str();
}

void printUsage() {
    std::cout << "Usage:\n"
              << "  anti_drone_camera_extrinsic_calibration <intrinsics_yaml> "
                 "<samples_csv> <gimbal_kinematics_yaml> <board_cols> "
                 "<board_rows> <square_size_m> [output_yaml]\n"
              << "\n"
              << "Arguments:\n"
              << "  intrinsics_yaml          standalone camera_intrinsics.yaml\n"
              << "  samples_csv              image_path,yaw_deg,pitch_deg rows\n"
              << "  gimbal_kinematics_yaml   yaw/pitch axis + sign + order\n"
              << "  board_cols               chessboard INNER corner count "
                 "(width)\n"
              << "  board_rows               chessboard INNER corner count "
                 "(height)\n"
              << "  square_size_m            chessboard square side length in "
                 "meters\n"
              << "  output_yaml              result (default: "
                 "config/camera_extrinsics.yaml)\n"
              << "\n"
              << "Example:\n"
              << "  anti_drone_camera_extrinsic_calibration "
                 "config/camera_intrinsics.yaml\n"
              << "      calib_extrinsics/samples.csv "
                 "config/gimbal_kinematics.yaml 6 8 0.029\n"
              << "      config/camera_extrinsics.yaml\n";
}

// ── Intrinsics (standalone camera_intrinsics.yaml) ──────────────────────────
// This file only carries camera_matrix / distort_coeffs / resolution; it is NOT
// a full production config, so loadAntiDroneConfig() must not be used.
struct Intrinsics {
    cv::Matx33d camera_matrix = cv::Matx33d::eye();
    std::vector<double> distort_coeffs;
    int calibration_image_width = 0;
    int calibration_image_height = 0;
};

Intrinsics loadIntrinsics(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);

    if (!root["camera_matrix"] || !root["distort_coeffs"]) {
        throw std::runtime_error(
            "intrinsics file must contain camera_matrix and distort_coeffs");
    }

    Intrinsics intrinsics;

    const std::vector<double> k =
        root["camera_matrix"].as<std::vector<double>>();
    if (k.size() != 9) {
        throw std::runtime_error("camera_matrix must have exactly 9 elements");
    }
    for (double v : k) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("camera_matrix must be finite");
        }
    }
    intrinsics.camera_matrix = cv::Matx33d(
        k[0], k[1], k[2],
        k[3], k[4], k[5],
        k[6], k[7], k[8]);
    if (!(intrinsics.camera_matrix(0, 0) > 0.0)) {
        throw std::runtime_error("camera_matrix fx must be > 0");
    }
    if (!(intrinsics.camera_matrix(1, 1) > 0.0)) {
        throw std::runtime_error("camera_matrix fy must be > 0");
    }

    const std::vector<double> dist =
        root["distort_coeffs"].as<std::vector<double>>();
    if (dist.size() < 5) {
        throw std::runtime_error(
            "distort_coeffs must have at least 5 elements");
    }
    for (double v : dist) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("distort_coeffs must be finite");
        }
    }
    intrinsics.distort_coeffs = dist;

    const bool has_width = static_cast<bool>(root["calibration_image_width"]);
    const bool has_height = static_cast<bool>(root["calibration_image_height"]);
    int width = 0;
    int height = 0;
    if (has_width) {
        width = root["calibration_image_width"].as<int>();
    }
    if (has_height) {
        height = root["calibration_image_height"].as<int>();
    }
    hnu25::anti_drone::validateCalibrationResolution(has_width, has_height,
                                                     width, height);
    intrinsics.calibration_image_width = width;
    intrinsics.calibration_image_height = height;

    return intrinsics;
}

cv::Mat asCameraMatrix(const Intrinsics& intrinsics) {
    cv::Mat K(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            K.at<double>(r, c) = intrinsics.camera_matrix(r, c);
        }
    }
    return K;
}

cv::Mat asDistCoeffs(const Intrinsics& intrinsics) {
    const std::size_t n = intrinsics.distort_coeffs.size();
    cv::Mat dist(1, static_cast<int>(n), CV_64F);
    for (std::size_t i = 0; i < n; ++i) {
        dist.at<double>(0, static_cast<int>(i)) = intrinsics.distort_coeffs[i];
    }
    return dist;
}

// ── Gimbal kinematics (gimbal_kinematics.yaml) ──────────────────────────────
// Every field is required explicitly: the mechanical axis directions have not
// been confirmed on real hardware, so the tool must never invent a "reasonable"
// default for yaw_axis / pitch_axis / rotation_order / signs.
struct KinematicsConfig {
    GimbalKinematics kin;
    std::string handeye_method_name;
    int handeye_method = cv::CALIB_HAND_EYE_PARK;
};

const YAML::Node require(const YAML::Node& root, const char* key) {
    const YAML::Node node = root[key];
    if (!node) {
        throw std::runtime_error(std::string("gimbal_kinematics.yaml is "
                                             "missing required field: ") +
                                 key);
    }
    return node;
}

RotationAxis parseAxis(const YAML::Node& node, const char* key) {
    const std::string value = node.as<std::string>();
    if (value == "x") {
        return RotationAxis::X;
    }
    if (value == "y") {
        return RotationAxis::Y;
    }
    if (value == "z") {
        return RotationAxis::Z;
    }
    throw std::runtime_error(std::string(key) +
                             " must be \"x\", \"y\", or \"z\"");
}

int parseHandEyeMethod(const std::string& name) {
    if (name == "TSAI") {
        return cv::CALIB_HAND_EYE_TSAI;
    }
    if (name == "PARK") {
        return cv::CALIB_HAND_EYE_PARK;
    }
    if (name == "HORAUD") {
        return cv::CALIB_HAND_EYE_HORAUD;
    }
    if (name == "ANDREFF") {
        return cv::CALIB_HAND_EYE_ANDREFF;
    }
    if (name == "DANIILIDIS") {
        return cv::CALIB_HAND_EYE_DANIILIDIS;
    }
    throw std::runtime_error(
        "handeye_method must be one of TSAI, PARK, HORAUD, ANDREFF, "
        "DANIILIDIS");
}

KinematicsConfig loadKinematics(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);

    KinematicsConfig cfg;

    cfg.kin.yaw_axis = parseAxis(require(root, "yaw_axis"), "yaw_axis");
    cfg.kin.pitch_axis = parseAxis(require(root, "pitch_axis"), "pitch_axis");

    const std::string order = require(root, "rotation_order").as<std::string>();
    if (order == "yaw_pitch") {
        cfg.kin.rotation_order = GimbalKinematics::RotationOrder::YawPitch;
    } else {
        throw std::runtime_error(
            "rotation_order must be \"yaw_pitch\" (only supported order in "
            "this version)");
    }

    cfg.kin.yaw_sign = require(root, "yaw_sign").as<double>();
    hnu25::anti_drone::requireGimbalSign(cfg.kin.yaw_sign, "yaw_sign");
    cfg.kin.pitch_sign = require(root, "pitch_sign").as<double>();
    hnu25::anti_drone::requireGimbalSign(cfg.kin.pitch_sign, "pitch_sign");
    cfg.kin.yaw_zero_deg = require(root, "yaw_zero_deg").as<double>();
    cfg.kin.pitch_zero_deg = require(root, "pitch_zero_deg").as<double>();

    const std::vector<double> t =
        require(root, "t_gimbal2base_m").as<std::vector<double>>();
    if (t.size() != 3) {
        throw std::runtime_error("t_gimbal2base_m must have exactly 3 "
                                 "elements");
    }
    for (double v : t) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("t_gimbal2base_m must be finite");
        }
    }
    cfg.kin.t_gimbal2base_m = cv::Vec3d(t[0], t[1], t[2]);

    for (double v : {cfg.kin.yaw_zero_deg, cfg.kin.pitch_zero_deg}) {
        if (!std::isfinite(v)) {
            throw std::runtime_error(
                "yaw_zero_deg / pitch_zero_deg must be finite");
        }
    }

    cfg.handeye_method_name = require(root, "handeye_method").as<std::string>();
    cfg.handeye_method = parseHandEyeMethod(cfg.handeye_method_name);

    return cfg;
}

// ── samples.csv ─────────────────────────────────────────────────────────────
struct CsvRow {
    std::string image_path;
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
};

std::string trim(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

std::vector<CsvRow> parseSamplesCsv(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open samples CSV: " + path);
    }

    std::vector<CsvRow> rows;
    std::string line;
    int line_number = 0;
    while (std::getline(in, line)) {
        ++line_number;
        const std::string t = trim(line);
        if (t.empty() || t[0] == '#') {
            continue;  // blank / comment
        }

        // Header row is tolerated (e.g. "image_path,yaw_deg,pitch_deg").
        const std::size_t first_comma = t.find(',');
        const std::string first_field = trim(t.substr(0, first_comma));
        if (first_field == "image_path") {
            continue;
        }

        // Split on commas into exactly three fields.
        std::vector<std::string> fields;
        std::size_t pos = 0;
        while (true) {
            const std::size_t comma = t.find(',', pos);
            if (comma == std::string::npos) {
                fields.push_back(trim(t.substr(pos)));
                break;
            }
            fields.push_back(trim(t.substr(pos, comma - pos)));
            pos = comma + 1;
        }
        if (fields.size() != 3) {
            throw std::runtime_error("samples.csv line " +
                                     std::to_string(line_number) +
                                     " must have exactly 3 fields "
                                     "(image_path, yaw_deg, pitch_deg)");
        }

        CsvRow row;
        row.image_path = fields[0];
        if (row.image_path.empty()) {
            throw std::runtime_error("samples.csv line " +
                                     std::to_string(line_number) +
                                     " has an empty image_path");
        }
        try {
            row.yaw_deg = std::stod(fields[1]);
            row.pitch_deg = std::stod(fields[2]);
        } catch (const std::exception&) {
            throw std::runtime_error("samples.csv line " +
                                     std::to_string(line_number) +
                                     " has a non-numeric yaw/pitch");
        }
        if (!std::isfinite(row.yaw_deg) || !std::isfinite(row.pitch_deg)) {
            throw std::runtime_error("samples.csv line " +
                                     std::to_string(line_number) +
                                     " has a non-finite yaw/pitch");
        }

        rows.push_back(row);
    }

    if (rows.empty()) {
        throw std::runtime_error("samples.csv contains no data rows");
    }
    return rows;
}

// ── Chessboard geometry ─────────────────────────────────────────────────────
// Physical object points, Z = 0, meters. Row-major order (top-left to
// bottom-right), matching cv::findChessboardCorners(). board_cols / board_rows
// are INNER corner counts.
std::vector<cv::Point3f> makeObjectPoints(int board_cols,
                                          int board_rows,
                                          double square_size_m) {
    std::vector<cv::Point3f> points;
    points.reserve(static_cast<std::size_t>(board_cols) * board_rows);
    for (int r = 0; r < board_rows; ++r) {
        for (int c = 0; c < board_cols; ++c) {
            points.emplace_back(static_cast<float>(c * square_size_m),
                                static_cast<float>(r * square_size_m), 0.0F);
        }
    }
    return points;
}

double det33(const cv::Matx33d& R) {
    return R(0, 0) * (R(1, 1) * R(2, 2) - R(1, 2) * R(2, 1)) -
           R(0, 1) * (R(1, 0) * R(2, 2) - R(1, 2) * R(2, 0)) +
           R(0, 2) * (R(1, 0) * R(2, 1) - R(1, 1) * R(2, 0));
}

double maxOrthoError(const cv::Matx33d& R) {
    const cv::Matx33d prod = R * R.t();
    double max_err = 0.0;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            const double expected = (r == c) ? 1.0 : 0.0;
            max_err = std::max(max_err, std::fabs(prod(r, c) - expected));
        }
    }
    return max_err;
}

double maxAbsDiff(const cv::Matx33d& A, const cv::Matx33d& B) {
    double max_diff = 0.0;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            max_diff = std::max(max_diff, std::fabs(A(r, c) - B(r, c)));
        }
    }
    return max_diff;
}

// ── Result printing / YAML ──────────────────────────────────────────────────
void printResult(const std::string& handeye_method_name,
                 std::size_t samples_total,
                 std::size_t samples_valid,
                 std::size_t samples_rejected,
                 double yaw_span_deg,
                 double pitch_span_deg,
                 const cv::Matx33d& R,
                 const cv::Vec3d& t,
                 double mean_pnp_rms,
                 double max_pnp_rms,
                 const hnu25::anti_drone::ConsistencyStats& consistency) {
    std::cout << "\n=== Camera -> Gimbal Extrinsic Calibration Result ===\n\n";

    std::cout << "samples_total:\n" << samples_total << "\n\n";
    std::cout << "samples_valid:\n" << samples_valid << "\n\n";
    std::cout << "samples_rejected:\n" << samples_rejected << "\n\n";

    std::cout << "handeye_method:\n" << handeye_method_name << "\n\n";

    std::cout << "yaw_span_deg:\n" << formatDouble(yaw_span_deg) << "\n\n";
    std::cout << "pitch_span_deg:\n" << formatDouble(pitch_span_deg) << "\n\n";

    std::cout << "R_camera2gimbal:\n\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << "[";
        for (int c = 0; c < 3; ++c) {
            if (c != 0) {
                std::cout << " ";
            }
            std::cout << formatDouble(R(r, c));
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    std::cout << "det_R:\n" << formatDouble(det33(R)) << "\n\n";

    std::cout << "t_camera2gimbal_m:\n\n";
    std::cout << "[" << formatDouble(t[0]) << " " << formatDouble(t[1]) << " "
              << formatDouble(t[2]) << "]\n\n";

    std::cout << "mean_board_pnp_rms_px:\n"
              << formatDouble(mean_pnp_rms) << "\n\n";
    std::cout << "max_board_pnp_rms_px:\n"
              << formatDouble(max_pnp_rms) << "\n\n";

    std::cout << "board_pose_translation_rms_m:\n"
              << formatDouble(consistency.translation_rms_m) << "\n\n";
    std::cout << "board_pose_translation_max_m:\n"
              << formatDouble(consistency.translation_max_m) << "\n\n";

    std::cout << "board_pose_rotation_rms_deg:\n"
              << formatDouble(consistency.rotation_rms_deg) << "\n\n";
    std::cout << "board_pose_rotation_max_deg:\n"
              << formatDouble(consistency.rotation_max_deg) << "\n";
}

bool writeResultYaml(const std::string& path,
                     const std::string& handeye_method_name,
                     std::size_t sample_count,
                     const cv::Matx33d& R,
                     const cv::Vec3d& t,
                     double mean_pnp_rms,
                     double max_pnp_rms,
                     const hnu25::anti_drone::ConsistencyStats& consistency,
                     double yaw_span_deg,
                     double pitch_span_deg) {
    const std::filesystem::path out_path(path);
    const std::filesystem::path parent = out_path.parent_path();
    if (!parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            std::cerr << "Failed to create output directory " << parent << ": "
                      << ec.message() << '\n';
            return false;
        }
    }

    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to write YAML: " << path << '\n';
        return false;
    }

    out << "# Camera -> gimbal extrinsic calibration result.\n";
    out << "# p_gimbal = R_camera2gimbal * p_camera + t_camera2gimbal "
           "(meters).\n";
    out << "R_camera2gimbal:\n";
    out << "  [" << formatDouble(R(0, 0)) << ", " << formatDouble(R(0, 1))
        << ", " << formatDouble(R(0, 2)) << ",\n";
    out << "   " << formatDouble(R(1, 0)) << ", " << formatDouble(R(1, 1))
        << ", " << formatDouble(R(1, 2)) << ",\n";
    out << "   " << formatDouble(R(2, 0)) << ", " << formatDouble(R(2, 1))
        << ", " << formatDouble(R(2, 2)) << "]\n";
    out << "t_camera2gimbal:\n";
    out << "  [" << formatDouble(t[0]) << ", " << formatDouble(t[1]) << ", "
        << formatDouble(t[2]) << "]\n";
    out << "handeye_method: " << handeye_method_name << "\n";
    out << "sample_count: " << sample_count << "\n";
    out << "mean_board_pnp_rms_px: " << formatDouble(mean_pnp_rms) << "\n";
    out << "max_board_pnp_rms_px: " << formatDouble(max_pnp_rms) << "\n";
    out << "board_pose_translation_rms_m: "
        << formatDouble(consistency.translation_rms_m) << "\n";
    out << "board_pose_translation_max_m: "
        << formatDouble(consistency.translation_max_m) << "\n";
    out << "board_pose_rotation_rms_deg: "
        << formatDouble(consistency.rotation_rms_deg) << "\n";
    out << "board_pose_rotation_max_deg: "
        << formatDouble(consistency.rotation_max_deg) << "\n";
    out << "yaw_span_deg: " << formatDouble(yaw_span_deg) << "\n";
    out << "pitch_span_deg: " << formatDouble(pitch_span_deg) << "\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 7 && argc != 8) {
        printUsage();
        return 1;
    }

    const std::string intrinsics_path = argv[1];
    const std::string samples_csv_path = argv[2];
    const std::string kinematics_path = argv[3];

    int board_cols = 0;
    int board_rows = 0;
    double square_size_m = 0.0;
    try {
        board_cols = std::stoi(argv[4]);
        board_rows = std::stoi(argv[5]);
        square_size_m = std::stod(argv[6]);
    } catch (const std::exception&) {
        std::cerr << "Invalid numeric argument.\n\n";
        printUsage();
        return 1;
    }

    const std::string output_path =
        (argc == 8) ? argv[7] : "config/camera_extrinsics.yaml";

    if (board_cols < 3 || board_rows < 3 || !(square_size_m > 0.0) ||
        !std::isfinite(square_size_m)) {
        std::cerr << "Invalid arguments: board_cols and board_rows must be "
                     ">= 3, square_size_m must be > 0 and finite.\n\n";
        printUsage();
        return 1;
    }

    try {
        std::cout << "=== Camera -> Gimbal Extrinsic Calibration ===\n\n";
        std::cout << "intrinsics: " << intrinsics_path << '\n';
        std::cout << "samples: " << samples_csv_path << '\n';
        std::cout << "kinematics: " << kinematics_path << '\n';
        std::cout << "board: " << board_cols << " x " << board_rows
                  << " inner corners\n";
        std::cout << "square_size: " << formatDouble(square_size_m) << " m\n";
        std::cout << "output: " << output_path << "\n\n";

        // ── Load configs ───────────────────────────────────────────────────
        const Intrinsics intrinsics = loadIntrinsics(intrinsics_path);
        const KinematicsConfig kinematics = loadKinematics(kinematics_path);
        const std::vector<CsvRow> csv_rows = parseSamplesCsv(samples_csv_path);

        const cv::Mat camera_matrix = asCameraMatrix(intrinsics);
        const cv::Mat dist_coeffs = asDistCoeffs(intrinsics);
        const std::vector<cv::Point3f> object_points =
            makeObjectPoints(board_cols, board_rows, square_size_m);

        const bool pin_resolution = intrinsics.calibration_image_width > 0 &&
                                    intrinsics.calibration_image_height > 0;

        std::cout << "handeye_method: " << kinematics.handeye_method_name
                  << '\n';
        if (pin_resolution) {
            std::cout << "resolution pinned to "
                      << intrinsics.calibration_image_width << " x "
                      << intrinsics.calibration_image_height << '\n';
        }
        std::cout << "\n";

        std::cout << "WARNING: this model assumes a fixed gimbal-frame origin "
                     "in the base frame.\n"
                  << "If yaw/pitch axes have significant mechanical offsets, "
                     "use a full\n"
                  << "kinematic pose model in a future calibration version.\n\n";

        // ── Per-sample processing ──────────────────────────────────────────
        struct ValidSample {
            cv::Matx33d R_gimbal2base;
            cv::Vec3d t_gimbal2base;
            cv::Matx33d R_target2cam;
            cv::Vec3d t_target2cam;
            double pnp_rms_px = 0.0;
        };

        std::vector<ValidSample> valid_samples;
        std::vector<double> valid_yaw_deg;
        std::vector<double> valid_pitch_deg;
        double sum_pnp_rms = 0.0;
        double max_pnp_rms = 0.0;

        std::size_t samples_rejected = 0;

        std::cout << "--- per-sample results ---\n";
        for (std::size_t idx = 0; idx < csv_rows.size(); ++idx) {
            const CsvRow& row = csv_rows[idx];
            std::ostringstream label;
            label << "sample_" << std::setw(3) << std::setfill('0')
                  << (idx + 1);

            std::string reject_reason;
            bool ok = true;

            const cv::Mat image = cv::imread(row.image_path);
            if (image.empty()) {
                reject_reason = "image not readable";
                ok = false;
            }

            if (ok && pin_resolution &&
                (image.cols != intrinsics.calibration_image_width ||
                 image.rows != intrinsics.calibration_image_height)) {
                reject_reason = "resolution mismatch";
                ok = false;
            }

            cv::Mat gray;
            std::vector<cv::Point2f> corners;
            bool corners_found = false;
            if (ok) {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
                corners_found = cv::findChessboardCorners(
                    gray, cv::Size(board_cols, board_rows), corners,
                    cv::CALIB_CB_ADAPTIVE_THRESH |
                        cv::CALIB_CB_NORMALIZE_IMAGE);
                if (!corners_found) {
                    reject_reason = "chessboard not detected";
                    ok = false;
                }
            }

            if (ok) {
                const cv::TermCriteria criteria(
                    cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30,
                    0.001);
                cv::cornerSubPix(gray, corners, cv::Size(11, 11),
                                 cv::Size(-1, -1), criteria);
            }

            cv::Vec3d rvec;
            cv::Vec3d tvec;
            bool pnp_ok = false;
            if (ok) {
                try {
                    pnp_ok = cv::solvePnP(
                        object_points, corners, camera_matrix, dist_coeffs,
                        rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                } catch (const cv::Exception&) {
                    pnp_ok = false;
                }
                if (!pnp_ok) {
                    reject_reason = "PnP failed";
                    ok = false;
                }
            }

            if (ok && (!std::isfinite(rvec[0]) || !std::isfinite(rvec[1]) ||
                       !std::isfinite(rvec[2]) || !std::isfinite(tvec[0]) ||
                       !std::isfinite(tvec[1]) || !std::isfinite(tvec[2]))) {
                reject_reason = "non-finite PnP result";
                ok = false;
            }
            if (ok && !(tvec[2] > 0.0)) {
                reject_reason = "tvec.z <= 0";
                ok = false;
            }

            double pnp_rms_px = 0.0;
            if (ok) {
                cv::Mat R_target2cam_mat;
                cv::Rodrigues(rvec, R_target2cam_mat);
                std::vector<cv::Point2f> projected;
                try {
                    cv::projectPoints(object_points, rvec, tvec, camera_matrix,
                                      dist_coeffs, projected);
                } catch (const cv::Exception&) {
                    reject_reason = "reprojection failed";
                    ok = false;
                }
                if (ok && projected.size() == corners.size()) {
                    double sq_sum = 0.0;
                    for (std::size_t j = 0; j < projected.size(); ++j) {
                        const double dx =
                            static_cast<double>(projected[j].x) - corners[j].x;
                        const double dy =
                            static_cast<double>(projected[j].y) - corners[j].y;
                        sq_sum += dx * dx + dy * dy;
                    }
                    pnp_rms_px = std::sqrt(sq_sum /
                                           static_cast<double>(projected.size()));
                    if (!std::isfinite(pnp_rms_px)) {
                        reject_reason = "non-finite reprojection RMS";
                        ok = false;
                    }
                } else if (ok) {
                    reject_reason = "reprojection size mismatch";
                    ok = false;
                }

                if (ok) {
                    ValidSample sample;
                    const GimbalPose pose = hnu25::anti_drone::makeGimbalPose(
                        row.yaw_deg, row.pitch_deg, kinematics.kin);
                    sample.R_gimbal2base = pose.R_gimbal2base;
                    sample.t_gimbal2base = pose.t_gimbal2base;
                    cv::Matx33d R_target2cam;
                    for (int r = 0; r < 3; ++r) {
                        for (int c = 0; c < 3; ++c) {
                            R_target2cam(r, c) =
                                R_target2cam_mat.at<double>(r, c);
                        }
                    }
                    sample.R_target2cam = R_target2cam;
                    sample.t_target2cam = tvec;
                    sample.pnp_rms_px = pnp_rms_px;

                    valid_samples.push_back(sample);
                    valid_yaw_deg.push_back(row.yaw_deg);
                    valid_pitch_deg.push_back(row.pitch_deg);
                    sum_pnp_rms += pnp_rms_px;
                    max_pnp_rms = std::max(max_pnp_rms, pnp_rms_px);
                }
            }

            if (!ok) {
                ++samples_rejected;
                std::cout << label.str() << ": " << row.image_path << " yaw="
                          << formatDouble(row.yaw_deg) << " pitch="
                          << formatDouble(row.pitch_deg)
                          << " REJECTED: " << reject_reason << '\n';
            } else {
                std::cout << label.str() << ": " << row.image_path << " yaw="
                          << formatDouble(row.yaw_deg) << " pitch="
                          << formatDouble(row.pitch_deg) << " corners=detected"
                          << " pnp_rms=" << formatDouble(pnp_rms_px) << " px\n";
            }
        }
        std::cout << "--------------------------\n\n";

        const std::size_t samples_total = csv_rows.size();
        const std::size_t samples_valid = valid_samples.size();

        // ── Sample-count gates ─────────────────────────────────────────────
        if (samples_valid < 5) {
            std::cerr << "ERROR: only " << samples_valid
                      << " valid poses (< 5). Hand-eye calibration is "
                         "forbidden with so few samples.\n";
            return 2;
        }
        if (samples_valid < 12) {
            std::cout << "WARNING: only " << samples_valid
                      << " valid poses (< 12). 15-30 diverse poses are "
                         "recommended.\n";
        }

        // ── Pose diversity ─────────────────────────────────────────────────
        const auto minmax_yaw = std::minmax_element(valid_yaw_deg.begin(),
                                                    valid_yaw_deg.end());
        const auto minmax_pitch = std::minmax_element(valid_pitch_deg.begin(),
                                                      valid_pitch_deg.end());
        const double yaw_span_deg = *minmax_yaw.second - *minmax_yaw.first;
        const double pitch_span_deg = *minmax_pitch.second - *minmax_pitch.first;

        std::cout << "yaw_span_deg: " << formatDouble(yaw_span_deg) << '\n';
        std::cout << "pitch_span_deg: " << formatDouble(pitch_span_deg)
                  << '\n';

        if (yaw_span_deg < 20.0) {
            std::cout << "WARNING: yaw span < 20 deg; gimbal yaw rotation may "
                         "be insufficiently varied for a stable hand-eye "
                         "solve.\n";
        }
        if (pitch_span_deg < 15.0) {
            std::cout << "WARNING: pitch span < 15 deg; gimbal pitch rotation "
                         "may be insufficiently varied for a stable hand-eye "
                         "solve.\n";
        }

        for (std::size_t i = 1; i < valid_yaw_deg.size(); ++i) {
            if (std::fabs(valid_yaw_deg[i] - valid_yaw_deg[i - 1]) < 1.0 &&
                std::fabs(valid_pitch_deg[i] - valid_pitch_deg[i - 1]) < 1.0) {
                std::cout << "near-duplicate pose warning: samples " << i
                          << " and " << (i + 1)
                          << " have almost identical yaw/pitch.\n";
            }
        }

        // ── Hand-eye solve ─────────────────────────────────────────────────
        std::vector<cv::Matx33d> R_gripper2base;
        std::vector<cv::Vec3d> t_gripper2base;
        std::vector<cv::Matx33d> R_target2cam;
        std::vector<cv::Vec3d> t_target2cam;
        R_gripper2base.reserve(valid_samples.size());
        t_gripper2base.reserve(valid_samples.size());
        R_target2cam.reserve(valid_samples.size());
        t_target2cam.reserve(valid_samples.size());
        for (const ValidSample& s : valid_samples) {
            R_gripper2base.push_back(s.R_gimbal2base);
            t_gripper2base.push_back(s.t_gimbal2base);
            R_target2cam.push_back(s.R_target2cam);
            t_target2cam.push_back(s.t_target2cam);
        }

        cv::Matx33d R_camera2gimbal;
        cv::Vec3d t_camera2gimbal;
        const bool solved = hnu25::anti_drone::solveHandEye(
            R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
            kinematics.handeye_method, R_camera2gimbal, t_camera2gimbal);
        if (!solved) {
            std::cerr << "Hand-eye calibration failed.\n";
            return 2;
        }

        // ── Result legality ────────────────────────────────────────────────
        {
            const double ortho_err = maxOrthoError(R_camera2gimbal);
            const double det = det33(R_camera2gimbal);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    if (!std::isfinite(R_camera2gimbal(r, c))) {
                        std::cerr << "Hand-eye result has a non-finite "
                                     "rotation entry.\n";
                        return 2;
                    }
                }
            }
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(t_camera2gimbal[i])) {
                    std::cerr << "Hand-eye result has a non-finite "
                                 "translation entry.\n";
                    return 2;
                }
            }
            if (ortho_err > 1e-3 || std::fabs(det - 1.0) > 1e-3) {
                std::cerr << "Hand-eye result is not a proper rotation "
                             "(ortho_err="
                          << formatDouble(ortho_err)
                          << ", det=" << formatDouble(det)
                          << "). Refusing to write a corrupted result.\n";
                return 2;
            }

            // Optional cleanup of only numerically tiny error, and only with
            // an explicit message. OpenCV normally returns an already-proper
            // rotation, so this branch is rarely taken.
            const cv::Matx33d R_projected =
                hnu25::anti_drone::projectToSO3(R_camera2gimbal);
            const double proj_diff =
                maxAbsDiff(R_camera2gimbal, R_projected);
            if (proj_diff > 1e-9) {
                std::cout << "rotation orthonormalized (raw det="
                          << formatDouble(det)
                          << ", raw ortho_err=" << formatDouble(ortho_err)
                          << ")\n";
                R_camera2gimbal = R_projected;
            }
        }

        // ── Fixed-target consistency ───────────────────────────────────────
        const hnu25::anti_drone::ConsistencyStats consistency =
            hnu25::anti_drone::evaluateFixedTargetConsistency(
                R_gripper2base, t_gripper2base, R_camera2gimbal,
                t_camera2gimbal, R_target2cam, t_target2cam);

        const double mean_pnp_rms =
            sum_pnp_rms / static_cast<double>(samples_valid);

        printResult(kinematics.handeye_method_name, samples_total,
                    samples_valid, samples_rejected, yaw_span_deg,
                    pitch_span_deg, R_camera2gimbal, t_camera2gimbal,
                    mean_pnp_rms, max_pnp_rms, consistency);

        if (!writeResultYaml(output_path, kinematics.handeye_method_name,
                             samples_valid, R_camera2gimbal, t_camera2gimbal,
                             mean_pnp_rms, max_pnp_rms, consistency,
                             yaw_span_deg, pitch_span_deg)) {
            std::cerr << "ERROR: failed to write calibration result to "
                      << output_path << '\n';
            return 2;
        }

        std::cout << "\nWrote " << output_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Camera extrinsic calibration failed: " << error.what()
                  << '\n';
        return 2;
    }
}
