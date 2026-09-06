#include "anti_drone/config.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/hik_frame_source.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#endif

// Guided, semi-automatic camera -> gimbal extrinsic calibration tool.
//
//   anti_drone_camera_extrinsic_guided_calibration \
//       <runtime_config> <intrinsics_yaml> <gimbal_kinematics_yaml> \
//       <board_cols> <board_rows> <square_size_m> <output_yaml>
//
// This is an ACQUISITION layer that sits on top of the existing OFFLINE solver
// anti_drone_camera_extrinsic_calibration. It opens the live Hik camera, detects
// the fixed chessboard, runs solvePnP per frame to decide when the gimbal has
// stopped moving, freezes a full-resolution frame, prompts the operator for the
// ACTUAL gimbal feedback yaw/pitch, and appends image_path,yaw_deg,pitch_deg to
// a per-session samples.csv. When enough diverse samples are collected it
// spawns the existing solver (found next to this executable) as a subprocess.
//
// Safety scope: this executable is calibration-only. It reads the camera and
// the keyboard, writes images + CSV, and calls the offline solver. It does NOT
// open a serial port, send telemetry, or issue any gimbal / yaw / pitch / speed
// / actuator command.
//
// Exit codes:
//   0 normal exit (solve succeeded, or a graceful early stop)
//   1 bad command line
//   2 runtime / config / start-camera / solve failure
//   3 Hik MVS support unavailable in this build

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

// ── Field-tunable thresholds (top of file, adjust on site) ─────────────────
constexpr double kMaxCapturePnpRmsPx = 1.0;   // reject pose if PnP RMS above this
constexpr double kStableWindowMs = 400.0;     // pose-history window for stability
constexpr int kStableMinSamples = 8;          // min valid poses in the window
constexpr double kStableMinDurationMs = 300.0; // min history time span to trust "stable"
constexpr double kStableTranslationM = 0.005; // max tvec spread while "stopped"
constexpr double kStableRotationDeg = 0.30;   // max R spread while "stopped"
constexpr double kMoveTranslationM = 0.02;    // pose must move this far to re-arm
constexpr double kMoveRotationDeg = 2.0;      // ...or rotate this far
constexpr double kMinAnglePoseDistanceDeg = 3.0; // angle dedup distance
constexpr int kMinSamples = 20;               // minimum sample count to solve
constexpr double kMinYawSpanDeg = 25.0;       // minimum yaw diversity to solve
constexpr double kMinPitchSpanDeg = 18.0;     // minimum pitch diversity to solve
constexpr int kMaxSamples = 30;               // hard cap, never exceed

std::string formatNumber(double value) {
    std::ostringstream oss;
    oss << std::setprecision(12) << std::defaultfloat << value;
    return oss.str();
}

std::string trim(const std::string& s) {
    const std::size_t first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    const std::size_t last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

// Single-quote shell quoting so the solver subprocess handles paths containing
// spaces. Escapes embedded single quotes as '\''.
std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// session directory suffix: YYYYMMDD_HHMMSS local time.
std::string timestampString() {
    const std::time_t now = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &now);
#else
    localtime_r(&now, &tm_buf);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return std::string(buf);
}

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

// Geodesic rotation angle in degrees: acos(clamp((trace(Ra^T * Rb) - 1) / 2)).
double rotationErrorDeg(const cv::Matx33d& Ra, const cv::Matx33d& Rb) {
    const cv::Matx33d d = Ra.t() * Rb;
    const double trace = d(0, 0) + d(1, 1) + d(2, 2);
    const double c = (trace - 1.0) / 2.0;
    return std::acos(std::max(-1.0, std::min(1.0, c))) * kRadToDeg;
}

cv::Matx33d rvecToMatx(const cv::Vec3d& rvec) {
    cv::Mat Rm;
    cv::Rodrigues(rvec, Rm);
    cv::Matx33d R;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            R(r, c) = Rm.at<double>(r, c);
        }
    }
    return R;
}

// ── Intrinsics (standalone camera_intrinsics.yaml) ──────────────────────────
struct Intrinsics {
    cv::Mat camera_matrix;  // 3x3 CV_64F
    cv::Mat dist_coeffs;    // 1xN CV_64F
    int calibration_image_width = 0;
    int calibration_image_height = 0;
};

Intrinsics loadIntrinsics(const std::string& path) {
    const YAML::Node root = YAML::LoadFile(path);

    if (!root["camera_matrix"] || !root["distort_coeffs"]) {
        throw std::runtime_error(
            "intrinsics file must contain camera_matrix and distort_coeffs");
    }

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

    Intrinsics intr;
    intr.camera_matrix = cv::Mat(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            intr.camera_matrix.at<double>(r, c) = k[static_cast<std::size_t>(r) * 3 + c];
        }
    }
    if (!(intr.camera_matrix.at<double>(0, 0) > 0.0) ||
        !(intr.camera_matrix.at<double>(1, 1) > 0.0)) {
        throw std::runtime_error("camera_matrix fx / fy must be > 0");
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
    intr.dist_coeffs = cv::Mat(1, static_cast<int>(dist.size()), CV_64F);
    for (std::size_t i = 0; i < dist.size(); ++i) {
        intr.dist_coeffs.at<double>(0, static_cast<int>(i)) = dist[i];
    }

    // The guided tool pins the live image to the calibrated resolution, so the
    // intrinsics file must carry a concrete resolution.
    if (!root["calibration_image_width"] || !root["calibration_image_height"]) {
        throw std::runtime_error(
            "intrinsics file must specify calibration_image_width and "
            "calibration_image_height");
    }
    intr.calibration_image_width = root["calibration_image_width"].as<int>();
    intr.calibration_image_height = root["calibration_image_height"].as<int>();
    if (intr.calibration_image_width <= 0 ||
        intr.calibration_image_height <= 0) {
        throw std::runtime_error(
            "calibration_image_width / calibration_image_height must be > 0");
    }

    return intr;
}

// ── Gimbal kinematics strict validation ────────────────────────────────────
// The guided tool checks every field the offline solver will later consume,
// BEFORE the camera opens, so a bad kinematics file fails fast instead of after
// N samples. This only checks presence + value legality; it does NOT reimplement
// the solver's hand-eye math (the solver remains the authority on the meaning of
// yaw/pitch axes, signs, order, and method).
void validateKinematicsYaml(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        throw std::runtime_error("gimbal_kinematics.yaml not found: " + path);
    }

    const YAML::Node root = YAML::LoadFile(path);
    if (!root.IsDefined() || root.IsNull()) {
        throw std::runtime_error("gimbal_kinematics.yaml is empty or invalid");
    }

    const auto require = [&root](const char* key) {
        const YAML::Node node = root[key];
        if (!node) {
            throw std::runtime_error(
                std::string("gimbal_kinematics.yaml is missing required "
                            "field: ") +
                key);
        }
        return node;
    };

    const auto checkAxis = [](const std::string& value, const char* key) {
        if (value != "x" && value != "y" && value != "z") {
            throw std::runtime_error(std::string(key) +
                                     " must be \"x\", \"y\", or \"z\"");
        }
    };
    checkAxis(require("yaw_axis").as<std::string>(), "yaw_axis");
    checkAxis(require("pitch_axis").as<std::string>(), "pitch_axis");

    const std::string order = require("rotation_order").as<std::string>();
    if (order != "yaw_pitch") {
        throw std::runtime_error(
            "rotation_order must be \"yaw_pitch\" (only supported order)");
    }

    const double yaw_sign = require("yaw_sign").as<double>();
    if (yaw_sign != 1.0 && yaw_sign != -1.0) {
        throw std::runtime_error("yaw_sign must be +1 or -1");
    }
    const double pitch_sign = require("pitch_sign").as<double>();
    if (pitch_sign != 1.0 && pitch_sign != -1.0) {
        throw std::runtime_error("pitch_sign must be +1 or -1");
    }

    const double yaw_zero_deg = require("yaw_zero_deg").as<double>();
    const double pitch_zero_deg = require("pitch_zero_deg").as<double>();
    if (!std::isfinite(yaw_zero_deg) || !std::isfinite(pitch_zero_deg)) {
        throw std::runtime_error("yaw_zero_deg / pitch_zero_deg must be finite");
    }

    const std::vector<double> t =
        require("t_gimbal2base_m").as<std::vector<double>>();
    if (t.size() != 3) {
        throw std::runtime_error("t_gimbal2base_m must have exactly 3 elements");
    }
    for (double v : t) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("t_gimbal2base_m must be finite");
        }
    }

    const std::string method = require("handeye_method").as<std::string>();
    if (method != "TSAI" && method != "PARK" && method != "HORAUD" &&
        method != "ANDREFF" && method != "DANIILIDIS") {
        throw std::runtime_error(
            "handeye_method must be one of TSAI, PARK, HORAUD, ANDREFF, "
            "DANIILIDIS");
    }
}

void printUsage() {
    std::cout << "Usage:\n"
              << "  anti_drone_camera_extrinsic_guided_calibration "
                 "<runtime_config> <intrinsics_yaml> <gimbal_kinematics_yaml> "
                 "<board_cols> <board_rows> <square_size_m> <output_yaml>\n"
              << "\n"
              << "Arguments:\n"
              << "  runtime_config          production anti_drone.yaml (camera "
                 "exposure/gain/frame_rate)\n"
              << "  intrinsics_yaml         standalone camera_intrinsics.yaml\n"
              << "  gimbal_kinematics_yaml  yaw/pitch axis + sign + order\n"
              << "  board_cols              chessboard INNER corner count "
                 "(width)\n"
              << "  board_rows              chessboard INNER corner count "
                 "(height)\n"
              << "  square_size_m           chessboard square side length in "
                 "meters\n"
              << "  output_yaml             result (config/camera_extrinsics.yaml)\n"
              << "\n"
              << "Example:\n"
              << "  anti_drone_camera_extrinsic_guided_calibration "
                 "config/anti_drone.yaml\n"
              << "      config/camera_intrinsics.yaml "
                 "config/gimbal_kinematics.yaml 6 8 0.029\n"
              << "      config/camera_extrinsics.yaml\n";
}

// ── Operator prompts (stdin, terminal) ──────────────────────────────────────
enum class AnglePrompt { Got, Skip, Quit };

AnglePrompt promptAngle(double& yaw, double& pitch) {
    while (true) {
        std::cout << "\n------------------------------------------------\n"
                  << "Stable calibration pose detected.\n\n"
                  << "Enter ACTUAL gimbal feedback angles in degrees.\n"
                  << "Format:\n"
                  << "yaw pitch\n\n"
                  << "Example:\n"
                  << "15.42 -7.83\n\n"
                  << "yaw pitch > " << std::flush;

        std::string line;
        if (!std::getline(std::cin, line)) {
            return AnglePrompt::Quit;  // EOF
        }
        const std::string t = trim(line);
        if (t == "q" || t == "Q") {
            return AnglePrompt::Quit;
        }
        if (t == "skip" || t == "Skip" || t == "SKIP") {
            return AnglePrompt::Skip;
        }

        std::istringstream iss(t);
        double a = 0.0;
        double b = 0.0;
        if (!(iss >> a >> b)) {
            std::cout << "Invalid angle input. Please enter:\nyaw pitch\n";
            continue;
        }
        std::string extra;
        if (iss >> extra) {
            std::cout << "Invalid angle input. Please enter:\nyaw pitch\n";
            continue;
        }
        if (!std::isfinite(a) || !std::isfinite(b)) {
            std::cout << "Invalid angle input. Please enter:\nyaw pitch\n";
            continue;
        }
        yaw = a;
        pitch = b;
        return AnglePrompt::Got;
    }
}

enum class ReadyPrompt { Solve, Continue, Quit };

ReadyPrompt promptReady(bool allow_continue) {
    while (true) {
        std::cout << "\nEnough samples collected.\n";
        std::cout << "Press ENTER to solve now";
        if (allow_continue) {
            std::cout << ",\nor type \"continue\" to collect more samples.";
        } else {
            std::cout << ".";
        }
        std::cout << "\n> " << std::flush;

        std::string line;
        if (!std::getline(std::cin, line)) {
            return ReadyPrompt::Quit;
        }
        const std::string t = trim(line);
        if (t == "q" || t == "Q") {
            return ReadyPrompt::Quit;
        }
        if (t.empty()) {
            return ReadyPrompt::Solve;
        }
        if (allow_continue &&
            (t == "continue" || t == "Continue" || t == "CONTINUE")) {
            return ReadyPrompt::Continue;
        }
        if (!allow_continue) {
            std::cout << "Maximum samples reached; press ENTER to solve, or "
                         "\"q\" to quit.\n";
        } else {
            std::cout << "Please press ENTER to solve, or type \"continue\".\n";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        printUsage();
        return 1;
    }

    const std::string runtime_config = argv[1];
    const std::string intrinsics_yaml = argv[2];
    const std::string kinematics_yaml = argv[3];

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
    const std::string output_yaml = argv[7];

    if (board_cols < 3 || board_rows < 3 || !(square_size_m > 0.0) ||
        !std::isfinite(square_size_m)) {
        std::cerr << "Invalid arguments: board_cols and board_rows must be "
                     ">= 3, square_size_m must be > 0 and finite.\n\n";
        printUsage();
        return 1;
    }

#if HNU25_HAS_MVS
    try {
        std::cout << "=== Guided Camera -> Gimbal Extrinsic Calibration ===\n\n";

        // ── Gimbal kinematics: strict validation before the camera opens ───
        try {
            validateKinematicsYaml(kinematics_yaml);
        } catch (const std::exception& error) {
            std::cerr << "gimbal_kinematics.yaml is invalid: " << error.what()
                      << '\n';
            return 2;
        }

        // ── Intrinsics (used by this tool for live PnP) ─────────────────────
        const Intrinsics intrinsics = loadIntrinsics(intrinsics_yaml);
        const int expected_width = intrinsics.calibration_image_width;
        const int expected_height = intrinsics.calibration_image_height;

        // ── Production config (camera only) ─────────────────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(runtime_config);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config: " << runtime_config
                      << '\n'
                      << error.what() << '\n';
            return 2;
        }

        std::cout << "runtime config: " << runtime_config << '\n';
        std::cout << "intrinsics: " << intrinsics_yaml << '\n';
        std::cout << "kinematics: " << kinematics_yaml << '\n';
        std::cout << "board: " << board_cols << " x " << board_rows
                  << " inner corners\n";
        std::cout << "square_size: " << formatNumber(square_size_m) << " m\n";
        std::cout << "expected resolution: " << expected_width << " x "
                  << expected_height << '\n';
        std::cout << "output: " << output_yaml << "\n\n";

        std::cout << "IMPORTANT:\n"
                  << "Enter actual gimbal feedback / encoder angles.\n"
                  << "Do NOT enter target or command angles.\n"
                  << "Unit: degrees.\n"
                  << "Sign and zero offset are handled by gimbal_kinematics.yaml; "
                     "enter raw feedback angles here.\n\n";

        // ── Map RuntimeCameraConfig onto the camera module (same as app) ────
        hnu25::camera::HikConfig camera_config;
        camera_config.serial_number = config.camera.serial_number;
        camera_config.exposure = config.camera.exposure;
        camera_config.gain = config.camera.gain;
        camera_config.frame_rate = config.camera.frame_rate;

        hnu25::camera::HikFrameSource source(camera_config);
        try {
            source.start();
        } catch (const std::exception& error) {
            std::cerr << "Failed to start Hik camera:\n"
                      << error.what() << '\n';
            return 2;
        }

        // ── Per-session output directory + CSV ──────────────────────────────
        const std::string session_dir =
            "calib_extrinsics_guided/session_" + timestampString();
        const std::filesystem::path session_path(session_dir);
        const std::filesystem::path frames_dir = session_path / "frames";
        std::error_code dir_ec;
        std::filesystem::create_directories(frames_dir, dir_ec);
        if (dir_ec) {
            std::cerr << "Failed to create directory " << frames_dir << ": "
                      << dir_ec.message() << '\n';
            return 2;
        }
        const std::string csv_path = (session_path / "samples.csv").string();
        std::ofstream csv(csv_path, std::ios::out | std::ios::trunc);
        if (!csv) {
            std::cerr << "Failed to create samples.csv: " << csv_path << '\n';
            return 2;
        }
        csv << "image_path,yaw_deg,pitch_deg\n";
        csv.flush();

        std::cout << "session: " << session_dir << "\n\n";

        const int frame_timeout_ms = config.camera.frame_timeout_ms;
        const int max_consecutive_timeouts =
            config.camera.max_consecutive_timeouts;

        const std::vector<cv::Point3f> object_points =
            makeObjectPoints(board_cols, board_rows, square_size_m);

        // ── Live PnP helper (captures intrinsics / object points) ──────────
        struct PnpOutcome {
            bool ok = false;
            cv::Vec3d rvec;
            cv::Vec3d tvec;
            double rms_px = 0.0;
        };
        auto runPnp = [&](const std::vector<cv::Point2f>& corners) {
            PnpOutcome o;
            cv::Vec3d rvec;
            cv::Vec3d tvec;
            try {
                const bool solved = cv::solvePnP(
                    object_points, corners, intrinsics.camera_matrix,
                    intrinsics.dist_coeffs, rvec, tvec, false,
                    cv::SOLVEPNP_ITERATIVE);
                if (!solved) {
                    return o;
                }
            } catch (const cv::Exception&) {
                return o;
            }
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(rvec[i]) || !std::isfinite(tvec[i])) {
                    return o;
                }
            }
            if (!(tvec[2] > 0.0)) {
                return o;
            }
            std::vector<cv::Point2f> projected;
            try {
                cv::projectPoints(object_points, rvec, tvec,
                                  intrinsics.camera_matrix,
                                  intrinsics.dist_coeffs, projected);
            } catch (const cv::Exception&) {
                return o;
            }
            if (projected.size() != corners.size()) {
                return o;
            }
            double sq_sum = 0.0;
            for (std::size_t j = 0; j < projected.size(); ++j) {
                const double dx =
                    static_cast<double>(projected[j].x) - corners[j].x;
                const double dy =
                    static_cast<double>(projected[j].y) - corners[j].y;
                sq_sum += dx * dx + dy * dy;
            }
            o.rms_px =
                std::sqrt(sq_sum / static_cast<double>(projected.size()));
            if (!std::isfinite(o.rms_px)) {
                return o;
            }
            o.rvec = rvec;
            o.tvec = tvec;
            o.ok = true;
            return o;
        };

        // ── Stability / sample state ────────────────────────────────────────
        struct PoseStamp {
            std::chrono::steady_clock::time_point t;
            cv::Vec3d tvec;
            cv::Matx33d R;
        };
        struct PendingSample {
            cv::Mat frame;
            std::vector<cv::Point2f> corners;
            cv::Vec3d rvec;
            cv::Vec3d tvec;
            cv::Matx33d R;
            std::chrono::steady_clock::time_point timestamp;
        };

        std::deque<PoseStamp> history;
        std::optional<PoseStamp> move_away_from;
        bool waiting_for_move = false;

        std::vector<double> saved_yaw;
        std::vector<double> saved_pitch;
        double min_yaw = std::numeric_limits<double>::infinity();
        double max_yaw = -std::numeric_limits<double>::infinity();
        double min_pitch = std::numeric_limits<double>::infinity();
        double max_pitch = -std::numeric_limits<double>::infinity();

        int consecutive_timeouts = 0;
        double actual_fps = 0.0;
        std::size_t fps_window_frames = 0;
        auto fps_window_start = std::chrono::steady_clock::now();

        const std::string window_name =
            "Anti-Drone Guided Extrinsic Calibration";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, 1280, 720);

        bool quit = false;
        bool stopped_no_solve = false;

        while (!quit && !stopped_no_solve) {
            hnu25::camera::Frame frame;
            const bool received = source.waitForFrame(
                frame, std::chrono::milliseconds(frame_timeout_ms));
            if (!received) {
                ++consecutive_timeouts;
                std::cout << "Frame timeout " << consecutive_timeouts << "/"
                          << max_consecutive_timeouts << '\n';
                if (consecutive_timeouts >= max_consecutive_timeouts) {
                    std::cerr << "ERROR: too many consecutive frame timeouts ("
                              << consecutive_timeouts
                              << "). Stopping acquisition; session data is "
                                 "preserved and the solver will NOT run.\n";
                    stopped_no_solve = true;
                    break;
                }
                continue;
            }
            consecutive_timeouts = 0;

            if (frame.image.empty()) {
                continue;
            }

            // ── Resolution pinning ─────────────────────────────────────────
            if (frame.image.cols != expected_width ||
                frame.image.rows != expected_height) {
                std::cerr << "ERROR: live image resolution " << frame.image.cols
                          << " x " << frame.image.rows
                          << " does not match intrinsics resolution "
                          << expected_width << " x " << expected_height
                          << ". Refusing to run a mis-calibrated pipeline.\n";
                return 2;
            }

            // ── Measured FPS ───────────────────────────────────────────────
            ++fps_window_frames;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - fps_window_start).count();
            if (elapsed >= 1.0) {
                actual_fps = static_cast<double>(fps_window_frames) / elapsed;
                fps_window_frames = 0;
                fps_window_start = now;
            }

            // ── Chessboard detection (downscaled, same as intrinsic tool) ──
            cv::Mat gray;
            cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);

            const double detection_scale =
                std::min(1.0, 960.0 / static_cast<double>(gray.cols));

            cv::Mat detection_gray;
            if (detection_scale < 1.0) {
                cv::resize(gray, detection_gray, cv::Size(), detection_scale,
                           detection_scale, cv::INTER_AREA);
            } else {
                detection_gray = gray;
            }

            std::vector<cv::Point2f> corners_small;
            const bool found = cv::findChessboardCorners(
                detection_gray, cv::Size(board_cols, board_rows), corners_small,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                    cv::CALIB_CB_FAST_CHECK);

            std::vector<cv::Point2f> corners;
            bool detected = false;
            if (found) {
                corners.resize(corners_small.size());
                for (std::size_t i = 0; i < corners_small.size(); ++i) {
                    corners[i].x = corners_small[i].x / detection_scale;
                    corners[i].y = corners_small[i].y / detection_scale;
                }
                const cv::TermCriteria criteria(
                    cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30,
                    0.001);
                cv::cornerSubPix(gray, corners, cv::Size(11, 11),
                                 cv::Size(-1, -1), criteria);
                detected = true;
            }

            // ── Live PnP on full-resolution corners ────────────────────────
            PnpOutcome pnp;
            if (detected) {
                pnp = runPnp(corners);
            }
            const bool quality_ok = pnp.ok && pnp.rms_px <= kMaxCapturePnpRmsPx;

            // ── Prune the pose history to the stability window ─────────────
            while (!history.empty()) {
                const double age_ms =
                    std::chrono::duration<double, std::milli>(now -
                                                              history.front().t)
                        .count();
                if (age_ms > kStableWindowMs) {
                    history.pop_front();
                } else {
                    break;
                }
            }

            // ── Move detection (only while waiting for the gimbal to move) ─
            if (waiting_for_move && quality_ok && move_away_from) {
                const cv::Matx33d R = rvecToMatx(pnp.rvec);
                const double dt = cv::norm(pnp.tvec - move_away_from->tvec);
                const double dr = rotationErrorDeg(R, move_away_from->R);
                if (dt >= kMoveTranslationM || dr >= kMoveRotationDeg) {
                    waiting_for_move = false;
                    move_away_from.reset();
                    history.clear();
                    std::cout << "MOVE DETECTED: waiting for the gimbal to "
                                 "settle at the new pose.\n";
                }
            }

            // ── Stability accumulation (not while waiting for a move) ──────
            bool stable = false;
            if (quality_ok) {
                if (!waiting_for_move) {
                    const cv::Matx33d R = rvecToMatx(pnp.rvec);
                    history.push_back(PoseStamp{now, pnp.tvec, R});
                    if (static_cast<int>(history.size()) >= kStableMinSamples) {
                        const double span_ms =
                            std::chrono::duration<double, std::milli>(
                                history.back().t - history.front().t)
                                .count();
                        if (span_ms >= kStableMinDurationMs) {
                            stable = true;
                            const PoseStamp& ref = history.front();
                            for (const PoseStamp& p : history) {
                                const double dt = cv::norm(p.tvec - ref.tvec);
                                const double dr = rotationErrorDeg(p.R, ref.R);
                                if (dt > kStableTranslationM ||
                                    dr > kStableRotationDeg) {
                                    stable = false;
                                    break;
                                }
                            }
                        }
                    }
                }
            } else {
                history.clear();
            }

            // ── Freeze a pending frame and prompt for the actual angles ────
            if (stable && !waiting_for_move) {
                PendingSample pending;
                pending.frame = frame.image.clone();
                pending.corners = corners;
                pending.rvec = pnp.rvec;
                pending.tvec = pnp.tvec;
                pending.R = rvecToMatx(pnp.rvec);
                pending.timestamp = frame.captured_at;

                // Flush a frozen preview BEFORE blocking on stdin so the image
                // and the "ANGLE INPUT REQUIRED" overlay are visible.
                {
                    cv::Mat frozen = pending.frame.clone();
                    cv::drawChessboardCorners(
                        frozen, cv::Size(board_cols, board_rows),
                        pending.corners, true);
                    cv::putText(frozen, "ANGLE INPUT REQUIRED",
                                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                                0.9, cv::Scalar(0, 0, 255), 2);
                    cv::imshow(window_name, frozen);
                    cv::waitKey(1);
                }

                double yaw = 0.0;
                double pitch = 0.0;
                const AnglePrompt angle_result = promptAngle(yaw, pitch);

                if (angle_result == AnglePrompt::Quit) {
                    quit = true;
                    break;
                }

                if (angle_result == AnglePrompt::Skip) {
                    // Abandon this pending frame and require the gimbal to move
                    // before re-arming (otherwise the still-stable pose would
                    // immediately re-trigger the prompt).
                    move_away_from =
                        PoseStamp{pending.timestamp, pending.tvec, pending.R};
                    waiting_for_move = true;
                    continue;
                }

                // ── Angle-pose dedup against already-saved samples ─────────
                bool too_close = false;
                for (std::size_t i = 0; i < saved_yaw.size(); ++i) {
                    const double dy = yaw - saved_yaw[i];
                    const double dp = pitch - saved_pitch[i];
                    const double dist = std::sqrt(dy * dy + dp * dp);
                    if (dist < kMinAnglePoseDistanceDeg) {
                        too_close = true;
                        break;
                    }
                }
                if (too_close) {
                    std::cout << "Angle pose too close to an existing "
                                 "sample.\n"
                              << "Sample discarded.\n";
                    move_away_from =
                        PoseStamp{pending.timestamp, pending.tvec, pending.R};
                    waiting_for_move = true;
                    continue;
                }

                // ── Save the frozen full-resolution image ──────────────────
                const std::size_t frame_index = saved_yaw.size() + 1;
                std::ostringstream name;
                name << "frame_" << std::setw(6) << std::setfill('0')
                     << frame_index << ".png";
                const std::filesystem::path image_path = frames_dir / name.str();
                const bool wrote_image = cv::imwrite(image_path.string(),
                                                     pending.frame);
                if (!wrote_image) {
                    std::cerr << "ERROR: failed to save image " << image_path
                              << ". Sample not recorded.\n";
                    move_away_from =
                        PoseStamp{pending.timestamp, pending.tvec, pending.R};
                    waiting_for_move = true;
                    continue;
                }

                // Only after imwrite succeeds do we commit the CSV row.
                const std::string absolute_image =
                    std::filesystem::absolute(image_path).string();
                csv << absolute_image << "," << formatNumber(yaw) << ","
                    << formatNumber(pitch) << "\n";
                csv.flush();

                saved_yaw.push_back(yaw);
                saved_pitch.push_back(pitch);
                min_yaw = std::min(min_yaw, yaw);
                max_yaw = std::max(max_yaw, yaw);
                min_pitch = std::min(min_pitch, pitch);
                max_pitch = std::max(max_pitch, pitch);

                move_away_from =
                    PoseStamp{pending.timestamp, pending.tvec, pending.R};
                waiting_for_move = true;

                std::cout << "Saved sample " << saved_yaw.size() << ": "
                          << image_path.string() << " yaw=" << formatNumber(yaw)
                          << " pitch=" << formatNumber(pitch)
                          << " pnp_rms=" << formatNumber(pnp.rms_px)
                          << " px\n";

                // ── Dataset-ready / completion evaluation ──────────────────
                const double yaw_span = max_yaw - min_yaw;
                const double pitch_span = max_pitch - min_pitch;
                const bool diverse = yaw_span >= kMinYawSpanDeg &&
                                     pitch_span >= kMinPitchSpanDeg;

                if (static_cast<int>(saved_yaw.size()) >= kMaxSamples) {
                    if (!diverse) {
                        std::cout << "\nWARNING:\n"
                                  << kMaxSamples
                                  << " samples reached but pose diversity is "
                                     "insufficient.\n\n"
                                  << "Current:\n"
                                  << "  yaw span = " << formatNumber(yaw_span)
                                  << " deg\n"
                                  << "  pitch span = "
                                  << formatNumber(pitch_span) << " deg\n\n"
                                  << "Required:\n"
                                  << "  yaw >= " << kMinYawSpanDeg << " deg\n"
                                  << "  pitch >= " << kMinPitchSpanDeg
                                  << " deg\n\n"
                                  << "Session data preserved. Not writing a "
                                     "possibly-wrong extrinsic result.\n";
                        stopped_no_solve = true;
                        break;
                    }
                    // At the cap with sufficient diversity: solve only.
                    const ReadyPrompt r = promptReady(false);
                    if (r == ReadyPrompt::Quit) {
                        quit = true;
                        break;
                    }
                    // Solve (no "continue" at the cap).
                    break;  // handled below after the loop
                } else if (static_cast<int>(saved_yaw.size()) >= kMinSamples &&
                           diverse) {
                    const ReadyPrompt r = promptReady(true);
                    if (r == ReadyPrompt::Quit) {
                        quit = true;
                        break;
                    }
                    if (r == ReadyPrompt::Continue) {
                        // Stay in the loop; already waiting_for_move.
                        continue;
                    }
                    // Solve.
                    break;  // handled below after the loop
                }
            }

            // ── Live status overlay ────────────────────────────────────────
            cv::Mat display = frame.image.clone();
            if (detected) {
                cv::drawChessboardCorners(display,
                                          cv::Size(board_cols, board_rows),
                                          corners, true);
            }

            std::string state_text;
            if (!detected) {
                state_text = "SEARCHING BOARD";
            } else if (!quality_ok) {
                state_text = "QUALITY REJECT";
            } else if (waiting_for_move) {
                state_text = "WAITING FOR MOVE";
            } else {
                state_text = "WAITING FOR STABLE";
            }

            const cv::Scalar overlay_color(0, 255, 0);
            int overlay_y = 30;
            auto putLine = [&](const std::string& text) {
                cv::putText(display, text, cv::Point(10, overlay_y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, overlay_color, 2);
                overlay_y += 26;
            };
            {
                std::ostringstream line;
                line << "FPS: " << std::fixed << std::setprecision(1)
                     << actual_fps;
                putLine(line.str());
            }
            putLine("Image: " + std::to_string(frame.image.cols) + " x " +
                    std::to_string(frame.image.rows));
            putLine("Detection image: " + std::to_string(detection_gray.cols) +
                    " x " + std::to_string(detection_gray.rows));
            {
                std::ostringstream line;
                line << "Detection scale: " << std::fixed
                     << std::setprecision(3) << detection_scale;
                putLine(line.str());
            }
            putLine("Board: " + std::to_string(board_cols) + " x " +
                    std::to_string(board_rows));
            putLine(std::string("Detected: ") + (detected ? "YES" : "NO"));
            if (pnp.ok) {
                std::ostringstream line;
                line << "PnP RMS: " << std::fixed << std::setprecision(2)
                     << pnp.rms_px << " px";
                putLine(line.str());
            } else {
                putLine("PnP RMS: --");
            }
            putLine(std::string("Stable: ") +
                    (stable && !waiting_for_move ? "YES" : "NO"));
            putLine("Samples: " + std::to_string(saved_yaw.size()));
            {
                const double yaw_span =
                    saved_yaw.empty() ? 0.0 : (max_yaw - min_yaw);
                std::ostringstream line;
                line << "Yaw span: " << std::fixed << std::setprecision(1)
                     << yaw_span << " deg";
                putLine(line.str());
            }
            {
                const double pitch_span =
                    saved_pitch.empty() ? 0.0 : (max_pitch - min_pitch);
                std::ostringstream line;
                line << "Pitch span: " << std::fixed << std::setprecision(1)
                     << pitch_span << " deg";
                putLine(line.str());
            }
            putLine("Status: " + state_text);
            putLine("Q/ESC: quit");

            cv::imshow(window_name, display);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                quit = true;
            }
        }

        source.stop();
        cv::destroyAllWindows();
        csv.flush();
        csv.close();

        if (quit || stopped_no_solve) {
            const double yaw_span =
                saved_yaw.empty() ? 0.0 : (max_yaw - min_yaw);
            const double pitch_span =
                saved_pitch.empty() ? 0.0 : (max_pitch - min_pitch);
            std::cout << "\n=== Guided Calibration Session Summary ===\n\n";
            std::cout << "samples: " << saved_yaw.size() << '\n';
            std::cout << "yaw_span_deg: " << formatNumber(yaw_span) << '\n';
            std::cout << "pitch_span_deg: " << formatNumber(pitch_span) << '\n';
            std::cout << "session: " << session_dir << '\n';
            std::cout << "solved: no\n";
            return 0;
        }

        // ── Invoke the existing offline solver as a subprocess ─────────────
        const std::filesystem::path exe_dir =
            std::filesystem::path(argv[0]).parent_path();
        const std::filesystem::path solver =
            exe_dir / "anti_drone_camera_extrinsic_calibration";

        std::ostringstream cmd;
        cmd << shellQuote(solver.string()) << " " << shellQuote(intrinsics_yaml)
            << " " << shellQuote(csv_path) << " "
            << shellQuote(kinematics_yaml) << " " << board_cols << " "
            << board_rows << " " << formatNumber(square_size_m) << " "
            << shellQuote(output_yaml);

        std::cout << "\nRunning solver:\n" << cmd.str() << "\n\n";

        const int rc = std::system(cmd.str().c_str());
        bool success = false;
#ifndef _WIN32
        if (rc != -1 && WIFEXITED(rc)) {
            success = (WEXITSTATUS(rc) == 0);
        }
#else
        success = (rc == 0);
#endif

        if (success) {
            std::cout << "\n========================================\n"
                      << "GUIDED EXTRINSIC CALIBRATION SUCCESS\n"
                      << "========================================\n\n"
                      << "Output:\n"
                      << output_yaml << "\n\n"
                      << "Session:\n"
                      << session_dir << "\n\n"
                      << "Review these result fields in the output YAML:\n"
                      << "  mean_board_pnp_rms_px\n"
                      << "  max_board_pnp_rms_px\n"
                      << "  board_pose_translation_rms_m\n"
                      << "  board_pose_rotation_rms_deg\n";
            return 0;
        }

        std::cerr << "\nGUIDED EXTRINSIC CALIBRATION SOLVE FAILED\n";
        std::cerr << "The session data was NOT deleted:\n  " << session_dir
                  << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Guided extrinsic calibration failed: " << error.what()
                  << '\n';
        return 2;
    }
#else
    std::cerr << "Hik MVS support is not available in this build.\n"
              << "Check /opt/MVS and rebuild the project.\n";
    return 3;
#endif
}
