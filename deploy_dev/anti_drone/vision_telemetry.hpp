#pragma once

#include "anti_drone/diagnostic_frame_processor.hpp"
#include "anti_drone/tracker.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

// VisionTelemetry is a diagnostic/result transport format.
// It contains NO actuator/fire/control command semantics.
//
// Units:
//   yaw_rad / pitch_rad : radians
//   x_m / y_m / z_m     : meters
//   timestamp_us        : microseconds
//   prediction_horizon_s: seconds
//   yaw_speed_rad_s / pitch_speed_rad_s : radians/second
//
// The wire format is a fixed-size little-endian packet with a CRC-16 trailer.
// Every field is encoded explicitly (no struct reinterpret_cast / #pragma
// pack), so padding, ABI, and host-endian differences never leak onto the
// wire.
namespace hnu25::anti_drone {

constexpr std::uint8_t kVisionTelemetryVersion = 1;
constexpr std::uint8_t kVisionTelemetryMagic0 = 0x41;  // 'A'
constexpr std::uint8_t kVisionTelemetryMagic1 = 0x44;  // 'D'

// header(4 bytes) + payload(52 bytes) + crc16(2 bytes).
constexpr std::size_t kVisionTelemetryPacketSize = 58;

// payload byte count between the 4-byte header and the 2-byte CRC.
constexpr std::size_t kVisionTelemetryPayloadLength = 52;

struct VisionTelemetry {
    std::uint8_t version = 1;

    std::uint32_t sequence = 0;
    std::uint64_t timestamp_us = 0;

    bool calibration_available = false;
    bool vision_valid = false;
    bool track_available = false;
    bool prediction_valid = false;
    bool measurement_updated = false;

    TrackState track_state = TrackState::LOST;

    // Final compensated diagnostic pointing direction, radians.
    float yaw_rad = 0.0F;
    float pitch_rad = 0.0F;

    // Final compensated diagnostic position, meters.
    float x_m = 0.0F;
    float y_m = 0.0F;
    float z_m = 0.0F;

    // Prediction horizon currently in use, seconds.
    float prediction_horizon_s = 0.0F;

    std::uint16_t detection_count = 0;
    std::uint16_t pnp_measurement_count = 0;

    // Gimbal-following angular rate of the compensated pointing direction,
    // radians/second. 0 on the first valid frame and whenever vision_valid is
    // false. Filled by the application's cross-frame speed filter; it is NOT
    // derived from a single-frame solution.
    float yaw_speed_rad_s = 0.0F;
    float pitch_speed_rad_s = 0.0F;
};

// Builds a telemetry from a per-frame processing result. yaw/pitch come from
// DiagnosticFrameResult::predicted_yaw_rad / predicted_pitch_rad; position
// comes from compensated_position_gimbal_m. No atan2 / compensation /
// prediction math is reimplemented here. The yaw/pitch speed fields are left
// at 0: they are filled by the caller's cross-frame speed filter.
VisionTelemetry makeVisionTelemetry(
    const DiagnosticFrameProcessorResult& frame_result,
    std::uint32_t sequence,
    std::uint64_t timestamp_us);

// Serializes a telemetry into a fixed-size packet. Throws std::invalid_argument
// if vision_valid is true but any yaw/pitch/xyz value is non-finite, or if
// prediction_horizon_s is non-finite / negative.
std::vector<std::uint8_t> encodeVisionTelemetry(
    const VisionTelemetry& telemetry);

// Parses a packet. On any failure (size / magic / version / payload length /
// CRC / track-state / non-finite float) returns false and leaves `telemetry`
// untouched.
bool decodeVisionTelemetry(
    const std::vector<std::uint8_t>& bytes,
    VisionTelemetry& telemetry);

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, refin false, refout false,
// xorout 0x0000.
std::uint16_t visionTelemetryCrc16(
    const std::uint8_t* data,
    std::size_t size);

}  // namespace hnu25::anti_drone
