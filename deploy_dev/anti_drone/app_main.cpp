#include "anti_drone/config.hpp"
#include "anti_drone/diagnostic_csv.hpp"
#include "anti_drone/diagnostic_frame_processor.hpp"
#include "anti_drone/vision_telemetry.hpp"

#include "camera/frame.hpp"
#include "camera/hik_frame_source.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// Formal Anti-Drone vision application entry point.
//
// Data chain (normal mode):
//   HikFrameSource -> Frame -> DiagnosticFrameProcessor::process()
//     -> DiagnosticFrameProcessorResult -> makeVisionTelemetry()
//     -> VisionTelemetry -> encodeVisionTelemetry() -> 50-byte packet.
//
// The 50-byte packet is produced as the module's final diagnostic output, but
// this entry point deliberately does NOT open a serial port, connect to the
// communication module, or emit any gimbal / fire / control command. It reuses
// the existing modules only and reimplements none of the math.
//
// Return codes:
//   1 usage error
//   2 config load failure
//   3 diagnostic frame processor construction failure
//   4 camera start failure
//   5 consecutive frame-timeout limit reached
//   6 unexpected exception
//   7 Hik MVS support unavailable in this build

namespace {

// Graceful-shutdown flag. The signal handler below only sets this flag; the
// frame loop observes it.
std::atomic<bool> g_stop_requested{false};

void handleSignal(int) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

// steady_clock timestamps are monotonic (no wall clock); VisionTelemetry
// carries microseconds since the clock epoch.
std::uint64_t toMicroseconds(std::chrono::steady_clock::time_point tp) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            tp.time_since_epoch())
            .count());
}

void printStartupSummary(
    const std::string& config_path,
    const hnu25::anti_drone::AntiDroneConfig& config) {
    std::cout << "=== Anti-Drone Vision Application ===\n\n";

    std::cout << "Config:\n  " << config_path << "\n\n";

    std::cout << "Detector:\n  READY\n\n";

    std::cout << "Camera:\n";
    std::cout << "  backend: Hik MVS\n";
    std::cout << "  serial_number: "
              << (config.camera.serial_number.empty()
                      ? "(auto)"
                      : config.camera.serial_number)
              << '\n';
    std::cout << "  exposure: " << config.camera.exposure << '\n';
    std::cout << "  gain: " << config.camera.gain << '\n';
    std::cout << "  frame_rate: " << config.camera.frame_rate << '\n';
    std::cout << "  frame_timeout_ms: " << config.camera.frame_timeout_ms
              << '\n';
    std::cout << "  max_consecutive_timeouts: "
              << config.camera.max_consecutive_timeouts << '\n';
    std::cout << '\n';

    std::cout << "Calibration: ";
    if (config.calibration.has_value()) {
        std::cout << "CONFIGURED\n";
        std::cout << "  fx: "
                  << config.calibration->pnp.camera_matrix(0, 0) << '\n';
        std::cout << "  fy: "
                  << config.calibration->pnp.camera_matrix(1, 1) << '\n';
        std::cout << "3D diagnostic solution: ENABLED\n";
    } else {
        std::cout << "NOT CONFIGURED\n";
        std::cout << "3D diagnostic solution: DISABLED\n";
    }
    std::cout << '\n';

    std::cout << "VisionTelemetry:\n";
    std::cout << "  version: "
              << static_cast<int>(
                     hnu25::anti_drone::kVisionTelemetryVersion)
              << '\n';
    std::cout << "  packet_size: "
              << hnu25::anti_drone::kVisionTelemetryPacketSize << '\n';
    std::cout << '\n';

    std::cout << "Communication: DISABLED\n\n";
}

void printFrameStatus(std::uint64_t frame_index,
                      std::uint64_t camera_frame_number,
                      const hnu25::anti_drone::VisionTelemetry& telemetry) {
    std::cout << "Frame " << frame_index << ":\n";
    std::cout << "  camera_frame_number: " << camera_frame_number << '\n';
    std::cout << "  detections: " << telemetry.detection_count << '\n';
    std::cout << "  pnp_measurements: " << telemetry.pnp_measurement_count
              << '\n';
    std::cout << "  track_state: "
              << hnu25::anti_drone::trackStateName(telemetry.track_state)
              << '\n';
    std::cout << "  vision_valid: " << std::boolalpha
              << telemetry.vision_valid << '\n';
    if (telemetry.vision_valid) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  yaw_rad: " << telemetry.yaw_rad << '\n';
        std::cout << "  pitch_rad: " << telemetry.pitch_rad << '\n';
        std::cout << "  x_m: " << telemetry.x_m << '\n';
        std::cout << "  y_m: " << telemetry.y_m << '\n';
        std::cout << "  z_m: " << telemetry.z_m << '\n';
    } else {
        std::cout << "  yaw/pitch: INVALID\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    // ── CLI parsing ────────────────────────────────────────────────────────
    bool check_only = false;
    std::string config_path;
    if (argc == 2) {
        config_path = argv[1];
    } else if (argc == 3 && std::string(argv[2]) == "--check") {
        config_path = argv[1];
        check_only = true;
    } else {
        std::cout << "Usage: anti_drone_app <config_yaml> [--check]\n";
        return 1;
    }

    // ── Graceful shutdown on SIGINT / SIGTERM ─────────────────────────────
    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    try {
        // ── Load config through the production API ────────────────────────
        hnu25::anti_drone::AntiDroneConfig config;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
        } catch (const std::exception& error) {
            std::cerr << "Failed to load anti-drone config:\n"
                      << config_path << '\n'
                      << error.what() << '\n';
            return 2;
        }

        // ── Construct the reusable per-frame core ─────────────────────────
        // DiagnosticFrameProcessor is not default-constructible, so it is
        // constructed in place here and owned for the lifetime of main.
        std::optional<hnu25::anti_drone::DiagnosticFrameProcessor> processor;
        try {
            processor.emplace(config);
        } catch (const std::exception& error) {
            std::cerr << "Failed to construct diagnostic frame processor: "
                      << error.what() << '\n';
            return 3;
        }

        printStartupSummary(config_path, config);

        // ── --check mode: no camera, no frame loop ────────────────────────
        if (check_only) {
            std::cout << "Hik MVS build support: ";
#if HNU25_HAS_MVS
            std::cout << "YES\n";
#else
            std::cout << "NO\n";
#endif
            std::cout << "\nConfiguration and runtime construction check "
                         "passed.\n";
            return 0;
        }

#if HNU25_HAS_MVS
        // ── Map RuntimeCameraConfig onto the camera module ────────────────
        // This mapping lives only here: the anti_drone library does not
        // include camera/hik_frame_source.hpp.
        hnu25::camera::HikConfig camera_config;
        camera_config.serial_number = config.camera.serial_number;
        camera_config.exposure = config.camera.exposure;
        camera_config.gain = config.camera.gain;
        camera_config.frame_rate = config.camera.frame_rate;

        std::cout << "Starting Hik camera...\n";

        hnu25::camera::HikFrameSource source(camera_config);
        try {
            source.start();
        } catch (const std::exception& error) {
            std::cerr << "Failed to start Hik camera:\n"
                      << error.what() << '\n';
            return 4;
        }
        std::cout << "Camera started.\n";
        std::cout << "Vision frame loop started.\n";

        const int frame_timeout_ms = config.camera.frame_timeout_ms;
        const int max_consecutive_timeouts =
            config.camera.max_consecutive_timeouts;
        const std::uint64_t log_every_n_frames =
            static_cast<std::uint64_t>(config.runtime.log_every_n_frames);

        std::uint32_t sequence = 0;
        std::uint64_t frame_index = 0;
        int consecutive_timeouts = 0;

        // State-change logging baseline.
        hnu25::anti_drone::TrackState previous_track_state =
            hnu25::anti_drone::TrackState::LOST;
        bool previous_vision_valid = false;
        bool have_previous = false;

        int exit_code = 0;

        while (!g_stop_requested.load(std::memory_order_relaxed)) {
            hnu25::camera::Frame frame;
            const bool received = source.waitForFrame(
                frame, std::chrono::milliseconds(frame_timeout_ms));
            if (!received) {
                std::cout << "Frame timeout (" << frame_timeout_ms
                          << " ms)\n";
                if (++consecutive_timeouts >= max_consecutive_timeouts) {
                    std::cerr << max_consecutive_timeouts
                              << " consecutive frame timeouts; stopping.\n";
                    exit_code = 5;
                    break;
                }
                continue;
            }
            consecutive_timeouts = 0;

            if (frame.image.empty()) {
                std::cerr << "Frame empty (warning); skipping.\n";
                continue;
            }

            // ── Per-frame processing ──────────────────────────────────────
            const auto result =
                processor->process(frame.image, frame.captured_at);

            const std::uint64_t timestamp_us =
                toMicroseconds(frame.captured_at);

            const auto telemetry =
                hnu25::anti_drone::makeVisionTelemetry(
                    result, sequence, timestamp_us);

            // Produce the fixed 50-byte packet. It is the module's final
            // diagnostic output; nothing here transmits it (no serial /
            // communication / control / fire semantics).
            const std::vector<std::uint8_t> packet =
                hnu25::anti_drone::encodeVisionTelemetry(telemetry);
            (void)packet;

            ++sequence;  // uint32_t rollover is acceptable.
            ++frame_index;

            const bool state_changed =
                !have_previous ||
                telemetry.track_state != previous_track_state ||
                telemetry.vision_valid != previous_vision_valid;
            const bool periodic =
                (frame_index % log_every_n_frames) == 0;

            if (state_changed || periodic) {
                printFrameStatus(
                    frame_index, frame.frame_number, telemetry);
            }

            previous_track_state = telemetry.track_state;
            previous_vision_valid = telemetry.vision_valid;
            have_previous = true;
        }

        source.stop();
        std::cout << "Vision frame loop stopped.\n";
        std::cout << "Camera stopped.\n";
        std::cout << "Anti-Drone Vision Application exited cleanly.\n";
        return exit_code;
#else
        std::cerr << "Hik MVS support is not available in this build.\n";
        return 7;
#endif
    } catch (const std::exception& error) {
        std::cerr << "Anti-Drone Vision Application failed: " << error.what()
                  << '\n';
        return 6;
    }
}
