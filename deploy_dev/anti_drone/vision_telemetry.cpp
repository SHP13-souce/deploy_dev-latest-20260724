#include "anti_drone/vision_telemetry.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace hnu25::anti_drone {
namespace {

// Payload byte count: everything between the 4-byte header and the 2-byte CRC.
//
//   uint32 sequence                 4
//   uint64 timestamp_us             8
//   uint16 status_flags             2
//   uint8  track_state              1
//   uint8  reserved                 1
//   float  yaw_rad                  4
//   float  pitch_rad                4
//   float  x_m                      4
//   float  y_m                      4
//   float  z_m                      4
//   float  prediction_horizon_s     4
//   uint16 detection_count          2
//   uint16 pnp_measurement_count    2
//                                  --
//                                  44
constexpr std::size_t kVisionTelemetryPayloadLength = 44;

static_assert(kVisionTelemetryPacketSize ==
                  4 + kVisionTelemetryPayloadLength + 2,
              "packet size must equal header + payload + crc");

// ── little-endian writers (field-by-field; no whole-struct memcpy) ──────────

void appendU8(std::vector<std::uint8_t>& out, std::uint8_t value) {
    out.push_back(value);
}

void appendU16(std::vector<std::uint8_t>& out, std::uint16_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFu));
}

void appendU32(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFFu));
    out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFFu));
}

void appendU64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        out.push_back(
            static_cast<std::uint8_t>((value >> (8 * i)) & 0xFFull));
    }
}

// float32: move the IEEE-754 bit pattern into a uint32_t via std::memcpy (no
// reinterpret_cast of the float* itself), then emit that integer little-endian.
void appendF32(std::vector<std::uint8_t>& out, float value) {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    appendU32(out, bits);
}

// ── little-endian readers (caller guarantees the pointer is in-bounds) ──────

std::uint8_t readU8(const std::uint8_t* p) {
    return p[0];
}

std::uint16_t readU16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(p[1]) << 8);
}

std::uint32_t readU32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint64_t readU64(const std::uint8_t* p) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(p[i]) << (8 * i);
    }
    return value;
}

float readF32(const std::uint8_t* p) {
    static_assert(sizeof(float) == 4, "float must be 4 bytes");
    const std::uint32_t bits = readU32(p);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ── status flags (bits 0..4) ────────────────────────────────────────────────

std::uint16_t encodeStatusFlags(const VisionTelemetry& telemetry) {
    std::uint16_t flags = 0;
    if (telemetry.calibration_available) flags |= 0x01u;
    if (telemetry.vision_valid) flags |= 0x02u;
    if (telemetry.track_available) flags |= 0x04u;
    if (telemetry.prediction_valid) flags |= 0x08u;
    if (telemetry.measurement_updated) flags |= 0x10u;
    return flags;
}

void decodeStatusFlags(std::uint16_t flags, VisionTelemetry& telemetry) {
    telemetry.calibration_available = (flags & 0x01u) != 0;
    telemetry.vision_valid = (flags & 0x02u) != 0;
    telemetry.track_available = (flags & 0x04u) != 0;
    telemetry.prediction_valid = (flags & 0x08u) != 0;
    telemetry.measurement_updated = (flags & 0x10u) != 0;
}

// ── TrackState wire mapping ─────────────────────────────────────────────────

std::uint8_t encodeTrackState(TrackState state) {
    switch (state) {
        case TrackState::LOST: return 0;
        case TrackState::DETECTING: return 1;
        case TrackState::TRACKING: return 2;
        case TrackState::TEMP_LOST: return 3;
    }
    return 0;  // unreachable for valid enum values
}

bool decodeTrackState(std::uint8_t value, TrackState& state) {
    switch (value) {
        case 0: state = TrackState::LOST; return true;
        case 1: state = TrackState::DETECTING; return true;
        case 2: state = TrackState::TRACKING; return true;
        case 3: state = TrackState::TEMP_LOST; return true;
        default: return false;
    }
}

// ── uint16 saturation ───────────────────────────────────────────────────────

std::uint16_t saturateU16(std::size_t value) {
    if (value > 0xFFFFu) {
        return 0xFFFFu;
    }
    return static_cast<std::uint16_t>(value);
}

}  // namespace

// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, refin false, refout false,
// xorout 0x0000. Non-reflected (MSB-first) bitwise implementation.
std::uint16_t visionTelemetryCrc16(const std::uint8_t* data, std::size_t size) {
    std::uint16_t crc = 0xFFFF;
    for (std::size_t i = 0; i < size; ++i) {
        crc ^= static_cast<std::uint16_t>(data[i]) << 8;
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc & 0x8000u)
                      ? static_cast<std::uint16_t>((crc << 1) ^ 0x1021u)
                      : static_cast<std::uint16_t>(crc << 1);
        }
    }
    return crc;
}

VisionTelemetry makeVisionTelemetry(
    const DiagnosticFrameProcessorResult& frame_result,
    std::uint32_t sequence,
    std::uint64_t timestamp_us) {
    VisionTelemetry telemetry;
    telemetry.version = kVisionTelemetryVersion;
    telemetry.sequence = sequence;
    telemetry.timestamp_us = timestamp_us;

    telemetry.calibration_available = frame_result.calibration_available;
    telemetry.detection_count = saturateU16(frame_result.observations.size());
    telemetry.pnp_measurement_count =
        saturateU16(frame_result.measurements.size());

    if (!frame_result.diagnostic_enabled) {
        // Detector-only: no stale direction / position / state is emitted.
        telemetry.vision_valid = false;
        telemetry.track_available = false;
        telemetry.prediction_valid = false;
        telemetry.measurement_updated = false;
        telemetry.track_state = TrackState::LOST;
        // yaw / pitch / xyz / prediction_horizon_s stay at their 0 defaults.
        return telemetry;
    }

    const auto& diagnostic = frame_result.diagnostic;

    telemetry.track_available = diagnostic.track_available;
    telemetry.prediction_valid = diagnostic.prediction_valid;
    telemetry.measurement_updated = diagnostic.measurement_updated;
    telemetry.track_state = diagnostic.track_state;

    // prediction_horizon_s is a config-validated value, but guard the
    // double -> float conversion anyway (check finite before converting).
    const double horizon = diagnostic.prediction_horizon_s;
    telemetry.prediction_horizon_s =
        (std::isfinite(horizon) && horizon >= 0.0)
            ? static_cast<float>(horizon)
            : 0.0F;

    telemetry.vision_valid = diagnostic.solution_valid;
    if (!diagnostic.solution_valid) {
        // Invalid solution: zero out direction / position, keep the rest.
        telemetry.yaw_rad = 0.0F;
        telemetry.pitch_rad = 0.0F;
        telemetry.x_m = 0.0F;
        telemetry.y_m = 0.0F;
        telemetry.z_m = 0.0F;
        return telemetry;
    }

    const cv::Vec3d& position = diagnostic.compensated_position_gimbal_m;
    if (!std::isfinite(diagnostic.predicted_yaw_rad) ||
        !std::isfinite(diagnostic.predicted_pitch_rad) ||
        !std::isfinite(position[0]) || !std::isfinite(position[1]) ||
        !std::isfinite(position[2])) {
        // Non-finite solution values: mark invalid and zero them out.
        telemetry.vision_valid = false;
        telemetry.yaw_rad = 0.0F;
        telemetry.pitch_rad = 0.0F;
        telemetry.x_m = 0.0F;
        telemetry.y_m = 0.0F;
        telemetry.z_m = 0.0F;
        return telemetry;
    }

    telemetry.yaw_rad = static_cast<float>(diagnostic.predicted_yaw_rad);
    telemetry.pitch_rad = static_cast<float>(diagnostic.predicted_pitch_rad);
    telemetry.x_m = static_cast<float>(position[0]);
    telemetry.y_m = static_cast<float>(position[1]);
    telemetry.z_m = static_cast<float>(position[2]);
    return telemetry;
}

std::vector<std::uint8_t> encodeVisionTelemetry(
    const VisionTelemetry& telemetry) {
    if (!std::isfinite(telemetry.prediction_horizon_s) ||
        telemetry.prediction_horizon_s < 0.0F) {
        throw std::invalid_argument(
            "prediction_horizon_s must be finite and >= 0");
    }
    if (telemetry.vision_valid &&
        (!std::isfinite(telemetry.yaw_rad) ||
         !std::isfinite(telemetry.pitch_rad) ||
         !std::isfinite(telemetry.x_m) ||
         !std::isfinite(telemetry.y_m) ||
         !std::isfinite(telemetry.z_m))) {
        throw std::invalid_argument(
            "vision_valid requires finite yaw/pitch/xyz");
    }

    std::vector<std::uint8_t> out;
    out.reserve(kVisionTelemetryPacketSize);

    // header
    appendU8(out, kVisionTelemetryMagic0);
    appendU8(out, kVisionTelemetryMagic1);
    appendU8(out, telemetry.version);
    appendU8(out, static_cast<std::uint8_t>(kVisionTelemetryPayloadLength));

    // payload
    appendU32(out, telemetry.sequence);
    appendU64(out, telemetry.timestamp_us);
    appendU16(out, encodeStatusFlags(telemetry));
    appendU8(out, encodeTrackState(telemetry.track_state));
    appendU8(out, 0);  // reserved
    appendF32(out, telemetry.yaw_rad);
    appendF32(out, telemetry.pitch_rad);
    appendF32(out, telemetry.x_m);
    appendF32(out, telemetry.y_m);
    appendF32(out, telemetry.z_m);
    appendF32(out, telemetry.prediction_horizon_s);
    appendU16(out, telemetry.detection_count);
    appendU16(out, telemetry.pnp_measurement_count);

    // CRC covers every byte from the magic up to (not including) the CRC.
    appendU16(out, visionTelemetryCrc16(out.data(), out.size()));

    return out;
}

bool decodeVisionTelemetry(const std::vector<std::uint8_t>& bytes,
                           VisionTelemetry& telemetry) {
    if (bytes.size() != kVisionTelemetryPacketSize) {
        return false;
    }
    if (bytes[0] != kVisionTelemetryMagic0 ||
        bytes[1] != kVisionTelemetryMagic1) {
        return false;
    }
    if (bytes[2] != kVisionTelemetryVersion) {
        return false;
    }
    if (bytes[3] != kVisionTelemetryPayloadLength) {
        return false;
    }

    const std::uint8_t* p = bytes.data() + 4;

    VisionTelemetry temp;
    temp.version = bytes[2];
    temp.sequence = readU32(p); p += 4;
    temp.timestamp_us = readU64(p); p += 8;
    const std::uint16_t status_flags = readU16(p); p += 2;
    const std::uint8_t track_state_wire = readU8(p); p += 1;
    // reserved byte: read and discard (not validated).
    p += 1;
    temp.yaw_rad = readF32(p); p += 4;
    temp.pitch_rad = readF32(p); p += 4;
    temp.x_m = readF32(p); p += 4;
    temp.y_m = readF32(p); p += 4;
    temp.z_m = readF32(p); p += 4;
    temp.prediction_horizon_s = readF32(p); p += 4;
    temp.detection_count = readU16(p); p += 2;
    temp.pnp_measurement_count = readU16(p); p += 2;

    decodeStatusFlags(status_flags, temp);

    if (!decodeTrackState(track_state_wire, temp.track_state)) {
        return false;
    }

    if (!std::isfinite(temp.yaw_rad) || !std::isfinite(temp.pitch_rad) ||
        !std::isfinite(temp.x_m) || !std::isfinite(temp.y_m) ||
        !std::isfinite(temp.z_m) || !std::isfinite(temp.prediction_horizon_s)) {
        return false;
    }

    const std::uint16_t expected =
        visionTelemetryCrc16(bytes.data(), kVisionTelemetryPacketSize - 2);
    const std::uint16_t wire =
        readU16(bytes.data() + kVisionTelemetryPacketSize - 2);
    if (expected != wire) {
        return false;
    }

    telemetry = temp;
    return true;
}

}  // namespace hnu25::anti_drone
