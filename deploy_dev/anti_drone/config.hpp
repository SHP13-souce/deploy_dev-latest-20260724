#pragma once

#include "anti_drone/pnp_solver.hpp"
#include "anti_drone/predictor.hpp"
#include "anti_drone/tracker.hpp"
#include "anti_drone/traditional_detector.hpp"

#include <filesystem>
#include <optional>
#include <string>

namespace hnu25::anti_drone {

// Residual compensation for the downstream PnP / gimbal-direction stage.
// These are NOT geometric calibration parameters; they are small manual
// adjustments applied on-site after real calibration.
//
// yaw_offset_deg / pitch_offset_deg:
//   Manual residual correction for the PnP / gimbal pointing direction,
//   in degrees.
//
// x/y/z_offset_m:
//   After the camera -> gimbal coordinate transform, a residual translation
//   correction applied to the gimbal-frame position, in meters.
//
// This stage only loads and stores these values; it does not apply them.
struct VisionCompensationConfig {
    double yaw_offset_deg = 0.0;
    double pitch_offset_deg = 0.0;

    double x_offset_m = 0.0;
    double y_offset_m = 0.0;
    double z_offset_m = 0.0;
};

// Camera / PnP calibration. The PnP solver only consumes the fields stored in
// PnpSolverConfig (camera_matrix, distort_coeffs, R_camera2gimbal,
// t_camera2gimbal, max_reprojection_error_px). R_gimbal2imubody is part of the
// original project's real extrinsic chain and is kept here for later stages,
// even though the PnP solver does not use it yet.
struct CalibrationConfig {
    PnpSolverConfig pnp;
    cv::Matx33d R_gimbal2imubody = cv::Matx33d::eye();

    // Optional runtime resolution pinning. 0 means "unspecified / no check";
    // when both are > 0, the runtime image size must match the calibration
    // image size exactly.
    int calibration_image_width = 0;
    int calibration_image_height = 0;
};

// Camera runtime settings for the production application. This is a plain
// value store only: the anti_drone library must NOT include
// camera/hik_frame_source.hpp. app_main.cpp maps this onto a
// hnu25::camera::HikConfig.
struct RuntimeCameraConfig {
    // Empty means "use the first camera the MVS SDK discovers".
    std::string serial_number;

    double exposure = 3000.0;
    double gain = 0.0;
    double frame_rate = 100.0;

    int frame_timeout_ms = 1000;
    int max_consecutive_timeouts = 5;
};

// Application behaviour for the production entry point.
struct AntiDroneRuntimeConfig {
    // Status lines are printed every N valid frames (and always on state
    // changes). Must be >= 1.
    int log_every_n_frames = 10;

    // When true, the production app opens a real-time preview window and
    // overlays the current detector output on a clone of each frame. The
    // processing chain and telemetry are unaffected. Default false keeps the
    // app headless (no DISPLAY required).
    bool show_preview = false;
};

// Gimbal-following behaviour for the production entry point. This covers only
// the vision side's output for gimbal following: whether to emit the
// yaw/pitch angular-rate fields and how strongly to low-pass them. It
// expresses no fire / actuator / control command.
struct GimbalConfig {
    // Master switch for gimbal-following speed output. When false, the speed
    // fields stay at 0 and the speed filter is not advanced.
    bool enable = true;

    // Emit the yaw_speed_rad_s / pitch_speed_rad_s fields. Ignored when
    // enable is false.
    bool send_speed = true;

    // Low-pass coefficient for the angular-rate filter: the new-sample weight
    // in [0, 1]. filtered = (1 - alpha) * old + alpha * raw.
    double speed_filter_alpha = 0.3;
};

// Which concrete transport the production application uses to deliver the
// fixed-size VisionTelemetry packet. LOOPBACK is an in-memory self-check only;
// SERIAL and SERIAL_DIAGNOSTIC both forward the packet over a real serial
// device (SERIAL for production gimbal following, SERIAL_DIAGNOSTIC for a
// read-only diagnostic receiver). No mode carries fire / actuator / control
// semantics.
enum class TelemetryTransportMode {
    LOOPBACK,
    SERIAL_DIAGNOSTIC,
    SERIAL,
};

// Human-readable name for a transport mode. Unknown enum values (future
// extensions) map to "UNKNOWN" rather than throwing.
const char* telemetryTransportModeName(TelemetryTransportMode mode) noexcept;

// True for modes that open a real serial device (SERIAL, SERIAL_DIAGNOSTIC).
bool telemetryTransportModeIsSerial(TelemetryTransportMode mode) noexcept;

// Transport selection for the production application.
struct TelemetryTransportConfig {
    TelemetryTransportMode mode = TelemetryTransportMode::LOOPBACK;
    std::string device;                  // serial device path (empty for LOOPBACK)
    int baud_rate = 115200;              // one of {9600,19200,38400,57600,115200}
    int write_timeout_ms = 20;           // finite poll() timeout per write, > 0
    int max_consecutive_failures = 5;    // consecutive send failures stop the app
    bool flush_after_write = false;      // tcdrain after each accepted write
};

// Aggregates all anti-drone configuration into a single loadable unit.
struct AntiDroneConfig {
    TraditionalDetectorConfig traditional_detector;

    // Runtime algorithm tuning for the 3D tracker and prediction stages. These
    // are independent of camera calibration and vision compensation.
    TrackerConfig tracker;
    PredictionConfig prediction;

    // Optional: real camera calibration may not be available yet. When the
    // YAML contains none of the calibration keys this stays nullopt, so the
    // existing detector / image_preview workflow keeps working unchanged.
    std::optional<CalibrationConfig> calibration;

    VisionCompensationConfig vision_compensation;

    // Production runtime camera + application behaviour. Loaded from the
    // optional "camera:" / "runtime:" YAML sections; the defaults keep older
    // config files compatible.
    RuntimeCameraConfig camera;
    AntiDroneRuntimeConfig runtime;

    // Production vision-telemetry transport selection. Loaded from the
    // optional "telemetry:" YAML section; the defaults keep older config
    // files compatible (LOOPBACK, no serial access).
    TelemetryTransportConfig telemetry;

    // Gimbal-following output behaviour. Loaded from the optional "gimbal:"
    // YAML section; the defaults (enable + send_speed, alpha 0.3) keep older
    // config files compatible.
    GimbalConfig gimbal;
};

// Loads the full anti-drone configuration from a YAML file. The file must
// contain a top-level "traditional_detector:" mapping; the optional
// "vision_compensation:" mapping defaults to all-zero when absent. Invalid
// values throw std::runtime_error.
AntiDroneConfig loadAntiDroneConfig(const std::filesystem::path& path);

// Returns true when the runtime image size is acceptable under the
// calibration's optional resolution pinning. A calibration with width/height
// both <= 0 pins nothing, so any size is accepted; otherwise both must match
// exactly.
bool calibrationResolutionMatches(
    const CalibrationConfig& calibration,
    int image_cols,
    int image_rows);

// Convenience wrapper that returns only the traditional detector portion.
// Kept for backward compatibility (e.g. image_preview).
TraditionalDetectorConfig loadTraditionalDetectorConfig(
    const std::filesystem::path& path);

}  // namespace hnu25::anti_drone
