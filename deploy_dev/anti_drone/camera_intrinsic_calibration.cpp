#include "anti_drone/config.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/hik_frame_source.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

// Hik industrial-camera intrinsic calibration tool.
//
//   anti_drone_camera_intrinsic_calibration \
//       <config_yaml> <board_cols> <board_rows> <square_size_m> [output_yaml]
//
// This is a standalone, read-only calibration tool: it reads frames from the
// SAME HikFrameSource / RuntimeCameraConfig path as anti_drone_app (so the
// exposure / gain / frame rate / resolution are identical to production), and
// collects chessboard corner samples into an OpenCV calibrateCamera intrinsic
// solve. It starts no detector, PnP, tracker, predictor, telemetry, serial
// transport, or any control path, and it never writes intrinsics (or fake
// extrinsics) into the production anti_drone.yaml.
//
// Exit codes:
//   0 normal exit
//   1 bad command line
//   2 runtime / start-camera exception (and config load failure)
//   3 Hik MVS support unavailable in this build

namespace {

// Formats a double with 15 significant digits (>= 12 as required) so that
// calibrated focal lengths / distortion coefficients survive a round-trip
// through the generated YAML without precision loss.
std::string formatDouble(double value) {
    std::ostringstream oss;
    oss << std::setprecision(15) << std::defaultfloat << value;
    return oss.str();
}

// Path of the saved PNG for one sample. Six-digit zero-padded index so the
// files sort naturally.
std::string framePath(std::size_t number) {
    std::ostringstream oss;
    oss << "calib_intrinsics/frames/frame_" << std::setw(6) << std::setfill('0')
        << number << ".png";
    return oss.str();
}

// Physical chessboard object points, Z = 0, in meters. Corner order is
// row-major (top-left to bottom-right), which matches the order
// cv::findChessboardCorners() returns.
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

// Normalizes calibrateCamera()'s distortion coefficients — which may be a 1x5
// row, a 5x1 column, or another container — into a guaranteed 1xN CV_64F row
// vector. Returns an empty Mat when the source is not a valid CV_64F vector of
// at least 5 coefficients, so the caller reports a clear error instead of
// relying on any particular row/column layout.
cv::Mat safeDistCoeffs(const cv::Mat& dist) {
    cv::Mat row = dist.reshape(1, 1);
    if (row.empty() || row.type() != CV_64F || row.total() < 5) {
        return cv::Mat();
    }
    return row;
}

// Scan the existing frame_*.png files so a new run continues from the highest
// index + 1 instead of overwriting previous images.
std::size_t scanNextFrameNumber(const std::string& frames_dir) {
    std::size_t next = 1;
    std::error_code ec;
    std::filesystem::directory_iterator it(frames_dir, ec);
    if (ec) {
        return next;
    }
    for (const auto& entry : it) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        // "frame_" + 6 digits + ".png" == 16 characters.
        if (name.size() != 16 || name.compare(0, 6, "frame_") != 0 ||
            name.compare(12, 4, ".png") != 0) {
            continue;
        }
        const std::string digits = name.substr(6, 6);
        const bool all_digits = std::all_of(
            digits.begin(), digits.end(),
            [](char c) { return c >= '0' && c <= '9'; });
        if (!all_digits) {
            continue;
        }
        try {
            const int num = std::stoi(digits);
            if (num >= 0) {
                next = std::max(next, static_cast<std::size_t>(num) + 1);
            }
        } catch (const std::exception&) {
            // Ignore an unparsable name and keep scanning.
        }
    }
    return next;
}

void printUsage() {
    std::cout << "Usage:\n"
              << "  anti_drone_camera_intrinsic_calibration <config_yaml> "
                 "<board_cols> <board_rows> <square_size_m> [output_yaml]\n"
              << "\n"
              << "Arguments:\n"
              << "  config_yaml     production anti_drone.yaml (camera, "
                 "exposure/gain/frame_rate)\n"
              << "  board_cols      chessboard INNER corner count along the "
                 "width (>= 3)\n"
              << "  board_rows      chessboard INNER corner count along the "
                 "height (>= 3)\n"
              << "  square_size_m   chessboard square side length in meters "
                 "(> 0, finite)\n"
              << "  output_yaml     result path (default: "
                 "config/camera_intrinsics.yaml)\n"
              << "\n"
              << "Example:\n"
              << "  anti_drone_camera_intrinsic_calibration "
                 "config/anti_drone.yaml 9 6 0.025 "
                 "config/camera_intrinsics.yaml\n";
}

void printCalibrationResult(const cv::Size& image_size,
                            std::size_t sample_count,
                            const cv::Mat& camera_matrix,
                            const cv::Mat& dist_coeffs,
                            double opencv_rms_px,
                            double mean_reproj,
                            double max_reproj) {
    std::cout << "\n=== Camera Intrinsic Calibration Result ===\n\n";

    std::cout << "image_size:\n";
    std::cout << image_size.width << " x " << image_size.height << "\n\n";

    std::cout << "samples:\n" << sample_count << "\n\n";

    std::cout << "camera_matrix:\n\n";
    std::cout << "fx " << formatDouble(camera_matrix.at<double>(0, 0)) << "\n";
    std::cout << "fy " << formatDouble(camera_matrix.at<double>(1, 1)) << "\n";
    std::cout << "cx " << formatDouble(camera_matrix.at<double>(0, 2)) << "\n";
    std::cout << "cy " << formatDouble(camera_matrix.at<double>(1, 2)) << "\n\n";
    for (int r = 0; r < 3; ++r) {
        std::cout << "[";
        for (int c = 0; c < 3; ++c) {
            if (c != 0) {
                std::cout << ", ";
            }
            std::cout << formatDouble(camera_matrix.at<double>(r, c));
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    std::cout << "distort_coeffs:\n\n";
    std::cout << "k1 " << formatDouble(dist_coeffs.at<double>(0, 0)) << "\n";
    std::cout << "k2 " << formatDouble(dist_coeffs.at<double>(0, 1)) << "\n";
    std::cout << "p1 " << formatDouble(dist_coeffs.at<double>(0, 2)) << "\n";
    std::cout << "p2 " << formatDouble(dist_coeffs.at<double>(0, 3)) << "\n";
    std::cout << "k3 " << formatDouble(dist_coeffs.at<double>(0, 4)) << "\n\n";

    std::cout << "opencv_rms_px:\n" << formatDouble(opencv_rms_px) << "\n\n";
    std::cout << "mean_reprojection_error_px:\n"
              << formatDouble(mean_reproj) << "\n\n";
    std::cout << "max_reprojection_error_px:\n"
              << formatDouble(max_reproj) << "\n";
}

// Writes the standalone intrinsic result file. Deliberately NOT a full
// production config: the extrinsics (R_camera2gimbal / t_camera2gimbal) are
// absent because they are not calibrated yet. It must not be fabricated.
bool writeResultYaml(const std::string& path,
                     const cv::Mat& camera_matrix,
                     const cv::Mat& dist_coeffs,
                     const cv::Size& image_size,
                     double opencv_rms_px,
                     double mean_reproj,
                     double max_reproj,
                     int board_cols,
                     int board_rows,
                     double square_size_m,
                     std::size_t sample_count) {
    const std::filesystem::path out_path(path);
    const std::filesystem::path parent = out_path.parent_path();
    if (!parent.empty()) {
        std::error_code dir_ec;
        std::filesystem::create_directories(parent, dir_ec);
        if (dir_ec) {
            std::cerr << "Failed to create output directory " << parent
                      << ": " << dir_ec.message() << '\n';
            return false;
        }
    }

    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to write YAML: " << path << '\n';
        return false;
    }

    const double fx = camera_matrix.at<double>(0, 0);
    const double fy = camera_matrix.at<double>(1, 1);
    const double cx = camera_matrix.at<double>(0, 2);
    const double cy = camera_matrix.at<double>(1, 2);

    out << "# Camera intrinsic calibration result (standalone).\n";
    out << "# NOT a full production config: R_camera2gimbal / t_camera2gimbal\n";
    out << "# are not calibrated yet and are intentionally absent. Merge these\n";
    out << "# intrinsics into anti_drone.yaml only after real extrinsic\n";
    out << "# calibration is completed.\n";
    out << "camera_matrix:\n";
    out << "  [" << formatDouble(fx) << ", 0.0, " << formatDouble(cx) << ",\n";
    out << "   0.0, " << formatDouble(fy) << ", " << formatDouble(cy) << ",\n";
    out << "   0.0, 0.0, 1.0]\n";
    out << "distort_coeffs:\n";
    out << "  [" << formatDouble(dist_coeffs.at<double>(0, 0)) << ", "
        << formatDouble(dist_coeffs.at<double>(0, 1)) << ", "
        << formatDouble(dist_coeffs.at<double>(0, 2)) << ", "
        << formatDouble(dist_coeffs.at<double>(0, 3)) << ", "
        << formatDouble(dist_coeffs.at<double>(0, 4)) << "]\n";
    out << "calibration_image_width: " << image_size.width << "\n";
    out << "calibration_image_height: " << image_size.height << "\n";
    out << "opencv_rms_px: " << formatDouble(opencv_rms_px) << "\n";
    out << "mean_reprojection_error_px: " << formatDouble(mean_reproj) << "\n";
    out << "max_reprojection_error_px: " << formatDouble(max_reproj) << "\n";
    out << "board_cols_inner: " << board_cols << "\n";
    out << "board_rows_inner: " << board_rows << "\n";
    out << "square_size_m: " << formatDouble(square_size_m) << "\n";
    out << "sample_count: " << sample_count << "\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        printUsage();
        return 1;
    }

    const std::string config_path = argv[1];

    int board_cols = 0;
    int board_rows = 0;
    double square_size_m = 0.0;
    try {
        board_cols = std::stoi(argv[2]);
        board_rows = std::stoi(argv[3]);
        square_size_m = std::stod(argv[4]);
    } catch (const std::exception&) {
        std::cerr << "Invalid numeric argument.\n\n";
        printUsage();
        return 1;
    }

    const std::string output_path =
        (argc == 6) ? argv[5] : "config/camera_intrinsics.yaml";

    if (board_cols < 3 || board_rows < 3 || !(square_size_m > 0.0) ||
        !std::isfinite(square_size_m)) {
        std::cerr << "Invalid arguments: board_cols and board_rows must be "
                     ">= 3, square_size_m must be > 0 and finite.\n\n";
        printUsage();
        return 1;
    }

#if HNU25_HAS_MVS
    try {
        // ── Load config through the production API ─────────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config: " << config_path
                      << '\n'
                      << error.what() << '\n';
            return 2;
        }

        std::cout << "=== Camera Intrinsic Calibration ===\n\n";
        std::cout << "config: " << config_path << '\n';
        std::cout << "serial_number: "
                  << (config.camera.serial_number.empty()
                          ? "(auto) first detected camera"
                          : config.camera.serial_number)
                  << '\n';
        std::cout << "exposure: " << config.camera.exposure << " us\n";
        std::cout << "gain: " << config.camera.gain << '\n';
        std::cout << "frame_rate: " << config.camera.frame_rate << " FPS\n";
        std::cout << "board: " << board_cols << " x " << board_rows
                  << " inner corners\n";
        std::cout << "square_size: " << square_size_m << " m\n";
        std::cout << "output: " << output_path << "\n\n";
        std::cout << "Starting Hik camera...\n";

        // ── Map RuntimeCameraConfig onto the camera module ─────────────────
        // Identical mapping to anti_drone_app, so the calibration uses the
        // exact production resolution / exposure / gain / lens.
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

        // ── Sample output directory (continue numbering, never overwrite) ──
        const std::string frames_dir = "calib_intrinsics/frames";
        std::error_code dir_ec;
        std::filesystem::create_directories(frames_dir, dir_ec);
        if (dir_ec) {
            std::cerr << "Failed to create directory " << frames_dir << ": "
                      << dir_ec.message() << '\n';
        }
        std::size_t next_frame_number = scanNextFrameNumber(frames_dir);

        const int frame_timeout_ms = config.camera.frame_timeout_ms;
        const int max_consecutive_timeouts =
            config.camera.max_consecutive_timeouts;

        // ── Shared physical object points (identical for every sample) ─────
        const std::vector<cv::Point3f> object_template =
            makeObjectPoints(board_cols, board_rows, square_size_m);

        // ── Calibration state ──────────────────────────────────────────────
        std::vector<std::vector<cv::Point2f>> image_points;
        std::vector<std::size_t> sample_frame_numbers;
        cv::Size image_size{0, 0};
        bool image_size_set = false;

        bool calibrated = false;
        cv::Mat camera_matrix;
        cv::Mat dist_coeffs;
        double opencv_rms_px = 0.0;
        double mean_reproj = 0.0;
        double max_reproj = 0.0;
        bool undistort_view = false;

        int consecutive_timeouts = 0;

        double actual_fps = 0.0;
        std::size_t fps_window_frames = 0;
        auto fps_window_start = std::chrono::steady_clock::now();

        const std::string window_name = "Camera Intrinsic Calibration";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, 1280, 720);

        bool quit = false;
        while (!quit) {
            hnu25::camera::Frame frame;
            const bool received = source.waitForFrame(
                frame, std::chrono::milliseconds(frame_timeout_ms));
            if (!received) {
                ++consecutive_timeouts;
                std::cout << "Frame timeout " << consecutive_timeouts << "/"
                          << max_consecutive_timeouts << '\n';
                if (consecutive_timeouts >= max_consecutive_timeouts) {
                    std::cout << "Too many consecutive frame timeouts; "
                                 "stopping.\n";
                    break;
                }
                continue;
            }
            consecutive_timeouts = 0;

            if (frame.image.empty()) {
                std::cerr << "Frame empty (warning); skipping.\n";
                continue;
            }

            // ── Resolution consistency ─────────────────────────────────────
            if (!image_size_set) {
                image_size = cv::Size(frame.image.cols, frame.image.rows);
                image_size_set = true;
            } else if (frame.image.cols != image_size.width ||
                       frame.image.rows != image_size.height) {
                std::cerr << "Camera image size changed during calibration.\n";
                break;
            }

            // ── Measured FPS ──────────────────────────────────────────────
            ++fps_window_frames;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed =
                std::chrono::duration<double>(now - fps_window_start).count();
            if (elapsed >= 1.0) {
                actual_fps = static_cast<double>(fps_window_frames) / elapsed;
                fps_window_frames = 0;
                fps_window_start = now;
            }

            // ── Chessboard detection (raw native resolution, never resized)─
            cv::Mat gray;
            cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Point2f> corners;
            const bool found = cv::findChessboardCorners(
                gray, cv::Size(board_cols, board_rows), corners,
                cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

            bool detected = false;
            if (found) {
                const cv::TermCriteria criteria(
                    cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30,
                    0.001);
                cv::cornerSubPix(gray, corners, cv::Size(11, 11),
                                 cv::Size(-1, -1), criteria);
                detected = true;
            }

            // ── Display: a clone (or undistorted view), never the raw frame ─
            cv::Mat display;
            if (undistort_view && calibrated) {
                cv::undistort(frame.image, display, camera_matrix,
                              dist_coeffs);
            } else {
                display = frame.image.clone();
                if (detected) {
                    cv::drawChessboardCorners(display,
                                              cv::Size(board_cols, board_rows),
                                              corners, true);
                }
            }

            const cv::Scalar overlay_color(0, 255, 0);
            int overlay_y = 30;
            auto putLine = [&](const std::string& text) {
                cv::putText(display, text, cv::Point(10, overlay_y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, overlay_color, 2);
                overlay_y += 30;
            };
            {
                std::ostringstream line;
                line << "FPS: " << std::fixed << std::setprecision(1)
                     << actual_fps;
                putLine(line.str());
            }
            putLine("Image: " + std::to_string(frame.image.cols) + " x " +
                    std::to_string(frame.image.rows));
            putLine("Board: " + std::to_string(board_cols) + " x " +
                    std::to_string(board_rows));
            putLine(std::string("Detected: ") + (detected ? "YES" : "NO"));
            putLine("Samples: " + std::to_string(image_points.size()));
            putLine(std::string("View: ") +
                    ((undistort_view && calibrated) ? "UNDISTORTED" : "RAW"));
            putLine("S:save  U:undo  C:calibrate  V:view  Q/ESC:quit");

            cv::imshow(window_name, display);

            // ── Key handling ──────────────────────────────────────────────
            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                quit = true;
            } else if (key == 's' || key == 'S') {
                if (detected) {
                    const std::size_t this_number = next_frame_number;
                    const std::string path = framePath(this_number);
                    if (cv::imwrite(path, frame.image)) {
                        image_points.push_back(corners);
                        sample_frame_numbers.push_back(this_number);
                        ++next_frame_number;
                        std::cout << "Saved sample " << image_points.size()
                                  << ": " << path << '\n';
                    } else {
                        std::cout << "Failed to save: " << path << '\n';
                    }
                } else {
                    std::cout << "Cannot save sample: chessboard not "
                                 "detected.\n";
                }
            } else if (key == 'u' || key == 'U') {
                if (image_points.empty()) {
                    std::cout << "No sample to undo.\n";
                } else {
                    image_points.pop_back();
                    const std::size_t number = sample_frame_numbers.back();
                    sample_frame_numbers.pop_back();
                    const std::string path = framePath(number);
                    std::error_code rm_ec;
                    if (std::filesystem::remove(path, rm_ec)) {
                        std::cout << "Removed " << path << '\n';
                    }
                    std::cout << "Samples: " << image_points.size() << '\n';
                }
            } else if (key == 'c' || key == 'C') {
                if (image_points.size() < 2) {
                    std::cerr << "Cannot calibrate: need at least 2 samples "
                                 "(have "
                              << image_points.size() << ").\n";
                } else {
                    if (image_points.size() < 15) {
                        std::cout << "WARNING: fewer than 15 samples.\n"
                                  << "20-30 diverse views are recommended.\n";
                    }

                    const std::vector<std::vector<cv::Point3f>> object_points(
                        image_points.size(), object_template);

                    cv::Mat K;
                    cv::Mat dist;
                    std::vector<cv::Mat> rvecs;
                    std::vector<cv::Mat> tvecs;

                    double opencv_rms = 0.0;
                    try {
                        opencv_rms = cv::calibrateCamera(
                            object_points, image_points, image_size, K, dist,
                            rvecs, tvecs);
                    } catch (const cv::Exception& error) {
                        std::cerr << "Calibration failed: " << error.what()
                                  << '\n';
                        continue;
                    }

                    // Normalize the distortion vector once, up front, so the
                    // reprojection, terminal report, YAML and undistort all
                    // consume a guaranteed 1xN CV_64F row vector. This never
                    // depends on whether calibrateCamera returned 1x5 or 5x1.
                    const cv::Mat dist_row = safeDistCoeffs(dist);
                    if (dist_row.empty()) {
                        std::cerr << "Calibration failed: calibrateCamera "
                                     "returned an unexpected distortion "
                                     "coefficient vector.\n";
                        continue;
                    }

                    // Per-sample RMS reprojection error (in addition to the
                    // value calibrateCamera returns). Lets the operator spot
                    // individually bad images; nothing is auto-deleted.
                    double sum_err = 0.0;
                    double max_err = 0.0;
                    try {
                        for (std::size_t i = 0; i < image_points.size(); ++i) {
                            std::vector<cv::Point2f> projected;
                            cv::projectPoints(object_points[i], rvecs[i],
                                              tvecs[i], K, dist_row,
                                              projected);
                            double sq_sum = 0.0;
                            for (std::size_t j = 0; j < projected.size();
                                 ++j) {
                                const double dx =
                                    static_cast<double>(projected[j].x) -
                                    image_points[i][j].x;
                                const double dy =
                                    static_cast<double>(projected[j].y) -
                                    image_points[i][j].y;
                                sq_sum += dx * dx + dy * dy;
                            }
                            const double rms_i =
                                std::sqrt(sq_sum /
                                          static_cast<double>(projected.size()));
                            sum_err += rms_i;
                            max_err = std::max(max_err, rms_i);
                            std::cout << "sample_" << std::setw(3)
                                      << std::setfill('0') << (i + 1) << ": "
                                      << rms_i << " px\n";
                        }
                    } catch (const cv::Exception& error) {
                        std::cerr << "Reprojection failed: " << error.what()
                                  << '\n';
                        continue;
                    }

                    camera_matrix = K.clone();
                    dist_coeffs = dist_row.clone();
                    opencv_rms_px = opencv_rms;
                    mean_reproj =
                        sum_err / static_cast<double>(image_points.size());
                    max_reproj = max_err;
                    calibrated = true;

                    printCalibrationResult(image_size, image_points.size(),
                                           camera_matrix, dist_coeffs,
                                           opencv_rms_px, mean_reproj,
                                           max_reproj);

                    if (writeResultYaml(output_path, camera_matrix,
                                        dist_coeffs, image_size, opencv_rms_px,
                                        mean_reproj, max_reproj, board_cols,
                                        board_rows, square_size_m,
                                        image_points.size())) {
                        std::cout << "Wrote " << output_path << '\n';
                    }
                }
            } else if (key == 'v' || key == 'V') {
                if (!calibrated) {
                    std::cout << "Calibration not available yet.\n";
                } else {
                    undistort_view = !undistort_view;
                }
            }
        }

        source.stop();
        cv::destroyAllWindows();

        std::cout << "\n=== Calibration Session Summary ===\n\n";
        std::cout << "samples: " << image_points.size() << '\n';
        std::cout << "calibrated: " << (calibrated ? "yes" : "no") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Camera intrinsic calibration failed: " << error.what()
                  << '\n';
        return 2;
    }
#else
    std::cerr << "Hik MVS support is not available in this build.\n"
              << "Check /opt/MVS and rebuild the project.\n";
    return 3;
#endif
}
