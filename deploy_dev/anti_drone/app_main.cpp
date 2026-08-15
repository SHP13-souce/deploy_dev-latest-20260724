#include "anti_drone/config.hpp"
#include "anti_drone/diagnostic_csv.hpp"
#include "anti_drone/diagnostic_frame_processor.hpp"
#include "anti_drone/gimbal_speed_filter.hpp"
#include "anti_drone/vision_telemetry.hpp"
#include "anti_drone/vision_telemetry_loopback.hpp"
#include "anti_drone/vision_telemetry_stream.hpp"
#include "anti_drone/vision_telemetry_transport.hpp"
#if HNU25_HAS_SERIAL_TELEMETRY
#include "anti_drone/vision_telemetry_serial_transport.hpp"
#endif

#include "camera/frame.hpp"
#include "camera/hik_frame_source.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Formal Anti-Drone vision application entry point.
//
// Data chain (normal mode):
//   HikFrameSource -> Frame -> DiagnosticFrameProcessor::process()
//     -> DiagnosticFrameProcessorResult -> makeVisionTelemetry()
//     -> VisionTelemetry -> encodeVisionTelemetry() -> fixed-size packet.
//
// The fixed-size packet carries the vision result plus a gimbal-following
// angular-rate output. Depending on the configured telemetry mode it is either
// verified in-memory (loopback) or forwarded over a real serial device to the
// electrical control unit. It never emits any fire / actuator command; gimbal
// following is expressed only as pointing direction + angular rate. The entry
// point reuses the existing modules only and reimplements none of their math.
//
// Return codes:
//   1 usage error
//   2 config load failure
//   3 diagnostic frame processor construction failure
//   4 camera start failure
//   5 consecutive frame-timeout limit reached
//   6 unexpected exception
//   7 Hik MVS support unavailable in this build
//   8 loopback smoke failure (--check)
//   9 telemetry transport open failure
//  10 telemetry consecutive send-failure limit reached
//  11 calibration resolution mismatch at runtime

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

// Builds the configured transport. Serial modes (SERIAL, SERIAL_DIAGNOSTIC)
// are only available when the build links the serial transport
// (HNU25_HAS_SERIAL_TELEMETRY); otherwise requesting one is a configuration
// error surfaced here as a throw.
std::unique_ptr<hnu25::anti_drone::VisionTelemetryTransport>
makeTelemetryTransport(
    const hnu25::anti_drone::TelemetryTransportConfig& config) {
    if (hnu25::anti_drone::telemetryTransportModeIsSerial(config.mode)) {
#if HNU25_HAS_SERIAL_TELEMETRY
        hnu25::anti_drone::VisionTelemetrySerialTransportConfig serial;
        serial.device = config.device;
        serial.baud_rate = config.baud_rate;
        serial.flush_after_write = config.flush_after_write;
        serial.write_timeout_ms = config.write_timeout_ms;
        return std::make_unique<
            hnu25::anti_drone::VisionTelemetrySerialTransport>(
            std::move(serial));
#else
        throw std::runtime_error(
            "serial telemetry transport is not available in this build");
#endif
    }
    return std::make_unique<
        hnu25::anti_drone::VisionTelemetryLoopbackTransport>();
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

    const bool serial_mode =
        hnu25::anti_drone::telemetryTransportModeIsSerial(
            config.telemetry.mode);

    std::cout << "Vision telemetry transport:\n";
    std::cout << "  mode: "
              << hnu25::anti_drone::telemetryTransportModeName(
                     config.telemetry.mode)
              << '\n';
    if (serial_mode) {
        std::cout << "  device: " << config.telemetry.device << '\n';
        std::cout << "  baud_rate: " << config.telemetry.baud_rate << '\n';
    }
    std::cout << "  packet_size: "
              << hnu25::anti_drone::kVisionTelemetryPacketSize << '\n';
    std::cout << "  protocol_version: "
              << static_cast<int>(
                     hnu25::anti_drone::kVisionTelemetryVersion)
              << '\n';
    std::cout << "  READY\n";
    std::cout << '\n';

    std::cout << "Gimbal following:\n";
    std::cout << "  enabled: " << std::boolalpha
              << config.gimbal.enable << '\n';
    std::cout << "  send_speed: " << config.gimbal.send_speed << '\n';
    std::cout << "  speed_filter_alpha: " << config.gimbal.speed_filter_alpha
              << '\n';
    std::cout << '\n';

    std::cout << "External serial output: "
              << (serial_mode ? "ENABLED" : "DISABLED") << '\n';
    std::cout << "Loopback verification: "
              << (serial_mode ? "startup check only" : "ENABLED")
              << "\n\n";
}

void printFrameStatus(
    std::uint64_t frame_index,
    std::uint64_t camera_frame_number,
    const hnu25::anti_drone::VisionTelemetry& telemetry,
    const hnu25::anti_drone::VisionTelemetryTransportStats& transport_stats) {
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
        std::cout << "  yaw_speed_rad_s: " << telemetry.yaw_speed_rad_s << '\n';
        std::cout << "  pitch_speed_rad_s: " << telemetry.pitch_speed_rad_s
                  << '\n';
        std::cout << "  x_m: " << telemetry.x_m << '\n';
        std::cout << "  y_m: " << telemetry.y_m << '\n';
        std::cout << "  z_m: " << telemetry.z_m << '\n';
    } else {
        std::cout << "  yaw/pitch: INVALID\n";
    }
    std::cout << "  telemetry_packets_submitted: "
              << transport_stats.packets_submitted << '\n';
    std::cout << "  telemetry_packets_accepted: "
              << transport_stats.packets_accepted << '\n';
    std::cout << "  telemetry_bytes_accepted: "
              << transport_stats.bytes_accepted << '\n';
    std::cout << "  telemetry_transport_failures: "
              << transport_stats.failures << '\n';
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

        // ── --check mode: no camera, no frame loop, no serial / /dev/* ─────
        if (check_only) {
            std::cout << "Configured telemetry mode: "
                      << hnu25::anti_drone::telemetryTransportModeName(
                             config.telemetry.mode)
                      << '\n';
            if (hnu25::anti_drone::telemetryTransportModeIsSerial(
                    config.telemetry.mode)) {
                std::cout << "  device: " << config.telemetry.device << '\n';
                std::cout << "  baud_rate: " << config.telemetry.baud_rate
                          << '\n';
            }
            std::cout << "External serial open: SKIPPED (--check)\n";

            std::cout << "Hik MVS build support: ";
#if HNU25_HAS_MVS
            std::cout << "YES\n";
#else
            std::cout << "NO\n";
#endif

            // Internal loopback smoke: encode a minimal legal telemetry, send
            // it through the loopback, and confirm the receiver re-parses it
            // with the same sequence. Always uses the in-memory loopback (no
            // serial / /dev/*) regardless of the configured mode.
            hnu25::anti_drone::VisionTelemetryLoopbackTransport loopback;
            hnu25::anti_drone::VisionTelemetry sample;
            sample.version = hnu25::anti_drone::kVisionTelemetryVersion;
            sample.sequence = 1;
            sample.timestamp_us = 1;
            sample.vision_valid = false;

            bool loopback_pass = false;
            const auto sample_packet =
                hnu25::anti_drone::encodeVisionTelemetry(sample);
            if (loopback.send(sample_packet)) {
                hnu25::anti_drone::VisionTelemetry received;
                if (loopback.popReceived(received) &&
                    received.sequence == 1) {
                    loopback_pass = true;
                }
            }

            if (loopback_pass) {
                std::cout << "Vision telemetry loopback: PASS\n";
            } else {
                std::cout << "Vision telemetry loopback: FAIL\n";
            }

            std::cout << "\nConfiguration and runtime construction check "
                         "passed.\n";
            return loopback_pass ? 0 : 8;
        }

        // ── Runtime transport: LOOPBACK, SERIAL, or SERIAL_DIAGNOSTIC ──────
        auto transport = makeTelemetryTransport(config.telemetry);

#if HNU25_HAS_SERIAL_TELEMETRY
        // A serial transport must open before the frame loop; an open failure
        // is a startup error, not a silent mid-loop failure.
        if (hnu25::anti_drone::telemetryTransportModeIsSerial(
                config.telemetry.mode)) {
            auto* serial_transport =
                dynamic_cast<
                    hnu25::anti_drone::VisionTelemetrySerialTransport*>(
                        transport.get());
            if (serial_transport == nullptr || !serial_transport->open()) {
                std::cerr << "Failed to open telemetry serial device '"
                          << config.telemetry.device << "'\n";
                return 9;
            }
            std::cout << "Telemetry serial transport opened.\n";
        }
#endif

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

        // Gimbal-following angular-rate filter. The rate is the low-pass-
        // filtered first difference of the compensated pointing direction
        // across valid frames; the yaw difference is wrapped to [-pi, pi], and
        // the estimate resets to 0 whenever the solution becomes invalid so a
        // stale rate never spans a target-loss gap.
        hnu25::anti_drone::GimbalSpeedFilter gimbal_speed_filter(
            config.gimbal.speed_filter_alpha);
        std::chrono::steady_clock::time_point prev_captured_at{};

        int exit_code = 0;
        int consecutive_transport_failures = 0;

        // One-time calibration resolution check on the first valid frame.
        bool calibration_resolution_checked = false;

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

            // ── Calibration resolution guard ─────────────────────────────
            // When the calibration pins an image resolution, reject a runtime
            // size mismatch on the first valid frame instead of silently
            // running a mis-calibrated pipeline. No camera_matrix rescaling is
            // attempted: a mismatch is a configuration error, not a fix-up.
            if (!calibration_resolution_checked) {
                calibration_resolution_checked = true;
                if (config.calibration.has_value() &&
                    !hnu25::anti_drone::calibrationResolutionMatches(
                        *config.calibration,
                        frame.image.cols,
                        frame.image.rows)) {
                    const auto& calib = *config.calibration;
                    std::cerr << "Calibration resolution mismatch:\n"
                              << "calibrated: "
                              << calib.calibration_image_width << "x"
                              << calib.calibration_image_height << '\n'
                              << "runtime:    " << frame.image.cols
                              << "x" << frame.image.rows << '\n';
                    exit_code = 11;
                    break;
                }
            }

            // ── Per-frame processing ──────────────────────────────────────
            const auto result =
                processor->process(frame.image, frame.captured_at);

            const std::uint64_t timestamp_us =
                toMicroseconds(frame.captured_at);

            auto telemetry =
                hnu25::anti_drone::makeVisionTelemetry(
                    result, sequence, timestamp_us);

            // Gimbal-following angular-rate output: filtered first difference
            // of the compensated pointing direction. 0 on the first valid
            // frame and whenever the solution is invalid.
            if (config.gimbal.enable && config.gimbal.send_speed &&
                telemetry.vision_valid) {
                const double dt = std::chrono::duration<double>(
                    frame.captured_at - prev_captured_at).count();
                const auto speed = gimbal_speed_filter.update(
                    static_cast<double>(telemetry.yaw_rad),
                    static_cast<double>(telemetry.pitch_rad),
                    dt);
                prev_captured_at = frame.captured_at;
                telemetry.yaw_speed_rad_s = speed.yaw_rad_s;
                telemetry.pitch_speed_rad_s = speed.pitch_rad_s;
            } else {
                gimbal_speed_filter.reset();
            }

            // Produce the fixed-size packet, then feed it through the
            // configured transport (loopback or serial). The transport
            // expresses only "send VisionTelemetry bytes" — no control / fire
            // / gimbal semantics.
            const std::vector<std::uint8_t> packet =
                hnu25::anti_drone::encodeVisionTelemetry(telemetry);
            if (transport->send(packet)) {
                consecutive_transport_failures = 0;
            } else if (++consecutive_transport_failures >=
                       config.telemetry.max_consecutive_failures) {
                std::cerr << "Telemetry transport failed "
                          << consecutive_transport_failures
                          << " consecutive times; stopping.\n";
                exit_code = 10;
                break;
            }

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
                    frame_index, frame.frame_number, telemetry,
                    transport->stats());
            }

            previous_track_state = telemetry.track_state;
            previous_vision_valid = telemetry.vision_valid;
            have_previous = true;
        }

        source.stop();
        std::cout << "Vision frame loop stopped.\n";
        std::cout << "Camera stopped.\n";

#if HNU25_HAS_SERIAL_TELEMETRY
        if (hnu25::anti_drone::telemetryTransportModeIsSerial(
                config.telemetry.mode)) {
            auto* serial_transport =
                dynamic_cast<
                    hnu25::anti_drone::VisionTelemetrySerialTransport*>(
                        transport.get());
            if (serial_transport != nullptr) {
                serial_transport->close();
            }
        }
#endif

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
