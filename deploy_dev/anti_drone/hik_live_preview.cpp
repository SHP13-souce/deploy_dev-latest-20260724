#include "anti_drone/config.hpp"
#include "camera/frame.hpp"
#include "camera/frame_source.hpp"
#include "camera/hik_frame_source.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// Read-only Hik industrial camera live preview entry point:
//
//   HikFrameSource -> Frame.image (BGR cv::Mat) -> OpenCV HighGUI -> screen.
//
// Press S to save the current *raw* frame as a PNG for later camera intrinsic
// calibration. This tool starts no detector, PnP, tracker, predictor,
// telemetry, serial transport, or any network / control path. It is safe to
// run on-site even when the electrical control unit is not connected.
//
// Exit codes:
//   0 normal exit
//   1 bad command line
//   2 runtime / start-camera exception (and config load failure)
//   3 Hik MVS support unavailable in this build

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage:\n"
                  << "  anti_drone_hik_live_preview <config_yaml>\n"
                  << "\n"
                  << "Example:\n"
                  << "  anti_drone_hik_live_preview config/anti_drone.yaml\n";
        return 1;
    }

    const std::string config_path = argv[1];

#if HNU25_HAS_MVS
    try {
        // ── Load config through the production API ────────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config: " << config_path
                      << '\n'
                      << error.what() << '\n';
            return 2;
        }

        std::cout << "=== Hik Live Preview ===\n\n";
        std::cout << "config: " << config_path << '\n';
        std::cout << "serial_number: "
                  << (config.camera.serial_number.empty()
                          ? "(auto) first detected camera"
                          : config.camera.serial_number)
                  << '\n';
        std::cout << "exposure: " << config.camera.exposure << " us\n";
        std::cout << "gain: " << config.camera.gain << '\n';
        std::cout << "frame_rate: " << config.camera.frame_rate << " FPS\n\n";
        std::cout << "Starting Hik camera...\n";

        // ── Map RuntimeCameraConfig onto the camera module ────────────────
        // This mapping lives only here: the anti_drone library does not
        // include camera/hik_frame_source.hpp. It uses the exact same camera
        // parameter source as anti_drone_app.
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

        // ── Calibration frame output directory ────────────────────────────
        std::filesystem::create_directories("calib_frames");

        const int frame_timeout_ms = config.camera.frame_timeout_ms;
        const int max_consecutive_timeouts =
            config.camera.max_consecutive_timeouts;

        const std::string window_name = "Hik Camera Live Preview";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        // Only the GUI window is resized; the camera frame keeps its native
        // resolution (frame.image.cols / frame.image.rows are never changed).
        cv::resizeWindow(window_name, 1280, 720);

        std::size_t received_frames = 0;
        std::size_t saved_frames = 0;
        std::size_t next_save_number = 1;
        int consecutive_timeouts = 0;

        // Measured FPS, computed from actually received frames (steady_clock,
        // never system_clock), refreshed once per second.
        double actual_fps = 0.0;
        std::size_t fps_window_frames = 0;
        auto fps_window_start = std::chrono::steady_clock::now();

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
                                 "stopping preview.\n";
                    break;
                }
                continue;
            }
            consecutive_timeouts = 0;

            if (frame.image.empty()) {
                std::cerr << "Frame empty (warning); skipping.\n";
                continue;
            }

            ++received_frames;

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

            // ── Overlay is drawn only on a clone, never on the raw frame ──
            cv::Mat display = frame.image.clone();
            {
                std::ostringstream line1;
                line1 << "FPS: " << std::fixed << std::setprecision(1)
                      << actual_fps;
                cv::putText(display, line1.str(), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 255, 0), 2);

                std::ostringstream line2;
                line2 << "Frame: " << received_frames << "   "
                      << frame.image.cols << "x" << frame.image.rows;
                cv::putText(display, line2.str(), cv::Point(10, 65),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8,
                            cv::Scalar(0, 255, 0), 2);
            }

            cv::imshow(window_name, display);

            const int key = cv::waitKey(1);
            if (key == 'q' || key == 'Q' || key == 27) {
                quit = true;
                break;
            }
            if (key == 's' || key == 'S') {
                std::ostringstream name;
                name << "calib_frames/frame_" << std::setw(6)
                     << std::setfill('0') << next_save_number << ".png";
                const std::string path = name.str();
                ++next_save_number;
                // Save the raw frame (frame.image), not the overlaid display.
                if (cv::imwrite(path, frame.image)) {
                    ++saved_frames;
                    std::cout << "Saved calibration frame:\n"
                              << path << '\n';
                } else {
                    std::cout << "Failed to save:\n" << path << '\n';
                }
            }
        }

        source.stop();
        cv::destroyAllWindows();

        std::cout << "\n=== Preview Summary ===\n\n";
        std::cout << "received_frames: " << received_frames << '\n';
        std::cout << "saved_frames: " << saved_frames << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Hik live preview failed: " << error.what() << '\n';
        return 2;
    }
#else
    std::cerr << "Hik MVS support is not available in this build.\n"
              << "Check /opt/MVS and rebuild the project.\n";
    return 3;
#endif
}
