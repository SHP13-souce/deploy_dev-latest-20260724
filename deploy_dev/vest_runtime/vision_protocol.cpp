#include "vest_runtime/vision_protocol.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

static_assert(sizeof(float) == 4,
              "vision protocol requires 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559,
              "vision protocol requires IEEE-754 float");

namespace {

constexpr std::uint8_t kMagic0 = 'V';
constexpr std::uint8_t kMagic1 = 'O';
constexpr std::uint8_t kMagic2 = 'B';
constexpr std::uint8_t kMagic3 = 'S';

constexpr std::uint8_t kFlagHasTarget = 0x01;
constexpr std::uint8_t kFlagTargetValid = 0x02;
constexpr std::uint8_t kFlagReservedMask = 0xFC;

constexpr std::size_t kOffsetMagic = 0;
constexpr std::size_t kOffsetVersion = 4;
constexpr std::size_t kOffsetMessageType = 5;
constexpr std::size_t kOffsetFlags = 6;
constexpr std::size_t kOffsetTrackingState = 7;
constexpr std::size_t kOffsetPayloadSize = 8;
constexpr std::size_t kOffsetReserved = 10;
constexpr std::size_t kOffsetFrameId = 12;
constexpr std::size_t kOffsetTimestampUs = 20;
constexpr std::size_t kOffsetTrackId = 28;
constexpr std::size_t kOffsetCenterX = 32;
constexpr std::size_t kOffsetCenterY = 36;
constexpr std::size_t kOffsetErrorX = 40;
constexpr std::size_t kOffsetErrorY = 44;
constexpr std::size_t kOffsetVelocityX = 48;
constexpr std::size_t kOffsetVelocityY = 52;
constexpr std::size_t kOffsetPredictedX = 56;
constexpr std::size_t kOffsetPredictedY = 60;
constexpr std::size_t kOffsetBboxW = 64;
constexpr std::size_t kOffsetBboxH = 68;
constexpr std::size_t kOffsetConfidence = 72;
constexpr std::size_t kOffsetCrc32 = 76;

constexpr std::size_t kCrcCoverage = 76;

// ── Little-endian write helpers ─────────────────────────────────────

void writeU16LE(std::uint8_t* dst, std::uint16_t value) {
    dst[0] = static_cast<std::uint8_t>(value);
    dst[1] = static_cast<std::uint8_t>(value >> 8);
}

void writeU32LE(std::uint8_t* dst, std::uint32_t value) {
    dst[0] = static_cast<std::uint8_t>(value);
    dst[1] = static_cast<std::uint8_t>(value >> 8);
    dst[2] = static_cast<std::uint8_t>(value >> 16);
    dst[3] = static_cast<std::uint8_t>(value >> 24);
}

void writeU64LE(std::uint8_t* dst, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        dst[i] = static_cast<std::uint8_t>(value >> (i * 8));
    }
}

void writeFloat32LE(std::uint8_t* dst, float value) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    writeU32LE(dst, bits);
}

// ── Little-endian read helpers ──────────────────────────────────────

std::uint16_t readU16LE(const std::uint8_t* src) {
    std::uint16_t value = 0;
    value |= static_cast<std::uint16_t>(src[0]);
    value |= static_cast<std::uint16_t>(src[1]) << 8;
    return value;
}

std::uint32_t readU32LE(const std::uint8_t* src) {
    std::uint32_t value = 0;
    value |= static_cast<std::uint32_t>(src[0]);
    value |= static_cast<std::uint32_t>(src[1]) << 8;
    value |= static_cast<std::uint32_t>(src[2]) << 16;
    value |= static_cast<std::uint32_t>(src[3]) << 24;
    return value;
}

std::uint64_t readU64LE(const std::uint8_t* src) {
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(src[i]) << (i * 8);
    }
    return value;
}

float readFloat32LE(const std::uint8_t* src) {
    std::uint32_t bits = readU32LE(src);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// ── CRC-32/IEEE (reflected) ─────────────────────────────────────────

std::uint32_t crc32Table[256];
bool crcTableInitialized = false;

void initCrc32Table() {
    for (int i = 0; i < 256; ++i) {
        std::uint32_t crc = static_cast<std::uint32_t>(i);
        for (int j = 0; j < 8; ++j) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320u;
            } else {
                crc >>= 1;
            }
        }
        crc32Table[i] = crc;
    }
    crcTableInitialized = true;
}

std::uint32_t computeCrc32(const std::uint8_t* data, std::size_t len) {
    if (!crcTableInitialized) {
        initCrc32Table();
    }
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < len; ++i) {
        crc = crc32Table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

// ── Validation ──────────────────────────────────────────────────────

void validateObservation(
    const hnu25::vest::VisionTargetObservation& obs) {
    using hnu25::vest::VisionTrackingState;

    // tracking_state must be valid
    if (obs.tracking_state != VisionTrackingState::NoTarget &&
        obs.tracking_state != VisionTrackingState::Tracking &&
        obs.tracking_state != VisionTrackingState::TemporarilyLost) {
        throw std::invalid_argument(
            "vision protocol: invalid tracking_state");
    }

    // All 11 floats must be finite
    if (!std::isfinite(obs.center_x) ||
        !std::isfinite(obs.center_y) ||
        !std::isfinite(obs.error_x) ||
        !std::isfinite(obs.error_y) ||
        !std::isfinite(obs.velocity_x) ||
        !std::isfinite(obs.velocity_y) ||
        !std::isfinite(obs.predicted_x) ||
        !std::isfinite(obs.predicted_y) ||
        !std::isfinite(obs.bbox_w) ||
        !std::isfinite(obs.bbox_h) ||
        !std::isfinite(obs.confidence)) {
        throw std::invalid_argument(
            "vision protocol: observation contains non-finite float");
    }

    // target_valid implies has_target and Tracking
    if (obs.target_valid) {
        if (!obs.has_target) {
            throw std::invalid_argument(
                "vision protocol: target_valid requires has_target");
        }
        if (obs.tracking_state != VisionTrackingState::Tracking) {
            throw std::invalid_argument(
                "vision protocol: target_valid requires Tracking state");
        }
    }

    // State-specific constraints
    switch (obs.tracking_state) {
        case VisionTrackingState::NoTarget:
            if (obs.has_target) {
                throw std::invalid_argument(
                    "vision protocol: NoTarget requires has_target=false");
            }
            if (obs.target_valid) {
                throw std::invalid_argument(
                    "vision protocol: NoTarget requires target_valid=false");
            }
            if (obs.track_id != -1) {
                throw std::invalid_argument(
                    "vision protocol: NoTarget requires track_id=-1");
            }
            break;

        case VisionTrackingState::Tracking:
            if (!obs.has_target) {
                throw std::invalid_argument(
                    "vision protocol: Tracking requires has_target");
            }
            if (obs.track_id < 0) {
                throw std::invalid_argument(
                    "vision protocol: has_target requires track_id >= 0");
            }
            break;

        case VisionTrackingState::TemporarilyLost:
            if (!obs.has_target) {
                throw std::invalid_argument(
                    "vision protocol: TemporarilyLost requires has_target");
            }
            if (obs.target_valid) {
                throw std::invalid_argument(
                    "vision protocol: TemporarilyLost requires "
                    "target_valid=false");
            }
            if (obs.track_id < 0) {
                throw std::invalid_argument(
                    "vision protocol: has_target requires track_id >= 0");
            }
            break;
    }

    // has_target cross-check (redundant with state checks but explicit)
    if (obs.has_target && obs.track_id < 0) {
        throw std::invalid_argument(
            "vision protocol: has_target requires track_id >= 0");
    }
}

}  // namespace

namespace hnu25::vest_runtime {

VisionProtocolPacket encodeVisionObservation(
    const hnu25::vest::VisionTargetObservation& observation) {
    validateObservation(observation);

    VisionProtocolPacket packet{};

    // ── Header ──────────────────────────────────────────────────────
    packet[kOffsetMagic + 0] = kMagic0;
    packet[kOffsetMagic + 1] = kMagic1;
    packet[kOffsetMagic + 2] = kMagic2;
    packet[kOffsetMagic + 3] = kMagic3;

    packet[kOffsetVersion] = kVisionProtocolVersion;
    packet[kOffsetMessageType] = kVisionObservationMessageType;

    std::uint8_t flags = 0;
    if (observation.has_target) {
        flags |= kFlagHasTarget;
    }
    if (observation.target_valid) {
        flags |= kFlagTargetValid;
    }
    packet[kOffsetFlags] = flags;

    packet[kOffsetTrackingState] =
        static_cast<std::uint8_t>(observation.tracking_state);

    writeU16LE(&packet[kOffsetPayloadSize],
               kVisionObservationPayloadSize);

    writeU16LE(&packet[kOffsetReserved], 0);

    // ── Payload ─────────────────────────────────────────────────────
    writeU64LE(&packet[kOffsetFrameId], observation.frame_id);
    writeU64LE(&packet[kOffsetTimestampUs],
               observation.measurement_timestamp_us);
    writeU32LE(&packet[kOffsetTrackId],
               static_cast<std::uint32_t>(observation.track_id));

    writeFloat32LE(&packet[kOffsetCenterX], observation.center_x);
    writeFloat32LE(&packet[kOffsetCenterY], observation.center_y);
    writeFloat32LE(&packet[kOffsetErrorX], observation.error_x);
    writeFloat32LE(&packet[kOffsetErrorY], observation.error_y);
    writeFloat32LE(&packet[kOffsetVelocityX], observation.velocity_x);
    writeFloat32LE(&packet[kOffsetVelocityY], observation.velocity_y);
    writeFloat32LE(&packet[kOffsetPredictedX], observation.predicted_x);
    writeFloat32LE(&packet[kOffsetPredictedY], observation.predicted_y);
    writeFloat32LE(&packet[kOffsetBboxW], observation.bbox_w);
    writeFloat32LE(&packet[kOffsetBboxH], observation.bbox_h);
    writeFloat32LE(&packet[kOffsetConfidence], observation.confidence);

    // ── CRC-32 ──────────────────────────────────────────────────────
    const std::uint32_t crc =
        computeCrc32(packet.data(), kCrcCoverage);
    writeU32LE(&packet[kOffsetCrc32], crc);

    return packet;
}

hnu25::vest::VisionTargetObservation decodeVisionObservation(
    const std::uint8_t* data,
    std::size_t size) {
    // ── Null / size check ───────────────────────────────────────────
    if (data == nullptr) {
        throw std::invalid_argument(
            "vision protocol data must not be null");
    }

    if (size != kVisionProtocolPacketSize) {
        throw std::runtime_error(
            "vision protocol invalid packet size: expected " +
            std::to_string(kVisionProtocolPacketSize) +
            ", got " + std::to_string(size));
    }

    // ── Header validation ───────────────────────────────────────────
    if (data[kOffsetMagic + 0] != kMagic0 ||
        data[kOffsetMagic + 1] != kMagic1 ||
        data[kOffsetMagic + 2] != kMagic2 ||
        data[kOffsetMagic + 3] != kMagic3) {
        throw std::runtime_error(
            "vision protocol invalid magic");
    }

    if (data[kOffsetVersion] != kVisionProtocolVersion) {
        throw std::runtime_error(
            "vision protocol unsupported version: " +
            std::to_string(data[kOffsetVersion]));
    }

    if (data[kOffsetMessageType] != kVisionObservationMessageType) {
        throw std::runtime_error(
            "vision protocol unexpected message type: " +
            std::to_string(data[kOffsetMessageType]));
    }

    const std::uint16_t payload_size =
        readU16LE(&data[kOffsetPayloadSize]);
    if (payload_size != kVisionObservationPayloadSize) {
        throw std::runtime_error(
            "vision protocol invalid payload size: expected " +
            std::to_string(kVisionObservationPayloadSize) +
            ", got " + std::to_string(payload_size));
    }

    const std::uint16_t reserved =
        readU16LE(&data[kOffsetReserved]);
    if (reserved != 0) {
        throw std::runtime_error(
            "vision protocol reserved field must be 0");
    }

    const std::uint8_t flags = data[kOffsetFlags];
    if ((flags & kFlagReservedMask) != 0) {
        throw std::runtime_error(
            "vision protocol reserved flag bits must be 0");
    }

    const std::uint8_t tracking_state_byte = data[kOffsetTrackingState];
    if (tracking_state_byte > 2) {
        throw std::runtime_error(
            "vision protocol invalid tracking_state: " +
            std::to_string(tracking_state_byte));
    }

    // ── CRC-32 ──────────────────────────────────────────────────────
    const std::uint32_t wire_crc = readU32LE(&data[kOffsetCrc32]);
    const std::uint32_t computed_crc =
        computeCrc32(data, kCrcCoverage);
    if (wire_crc != computed_crc) {
        throw std::runtime_error(
            "vision protocol CRC mismatch");
    }

    // ── Decode payload ──────────────────────────────────────────────
    hnu25::vest::VisionTargetObservation observation;

    observation.frame_id = readU64LE(&data[kOffsetFrameId]);
    observation.measurement_timestamp_us =
        readU64LE(&data[kOffsetTimestampUs]);

    {
        const std::uint32_t track_id_bits =
            readU32LE(&data[kOffsetTrackId]);
        observation.track_id =
            static_cast<std::int32_t>(track_id_bits);
    }

    observation.has_target = (flags & kFlagHasTarget) != 0;
    observation.target_valid = (flags & kFlagTargetValid) != 0;

    observation.tracking_state =
        static_cast<hnu25::vest::VisionTrackingState>(
            tracking_state_byte);

    observation.center_x = readFloat32LE(&data[kOffsetCenterX]);
    observation.center_y = readFloat32LE(&data[kOffsetCenterY]);
    observation.error_x = readFloat32LE(&data[kOffsetErrorX]);
    observation.error_y = readFloat32LE(&data[kOffsetErrorY]);
    observation.velocity_x = readFloat32LE(&data[kOffsetVelocityX]);
    observation.velocity_y = readFloat32LE(&data[kOffsetVelocityY]);
    observation.predicted_x = readFloat32LE(&data[kOffsetPredictedX]);
    observation.predicted_y = readFloat32LE(&data[kOffsetPredictedY]);
    observation.bbox_w = readFloat32LE(&data[kOffsetBboxW]);
    observation.bbox_h = readFloat32LE(&data[kOffsetBboxH]);
    observation.confidence = readFloat32LE(&data[kOffsetConfidence]);

    // ── Semantic validation ─────────────────────────────────────────
    validateObservation(observation);

    return observation;
}

}  // namespace hnu25::vest_runtime
