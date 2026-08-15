#include "anti_drone/vision_telemetry.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using hnu25::anti_drone::DiagnosticFrameProcessorResult;
using hnu25::anti_drone::TrackState;
using hnu25::anti_drone::VisionTelemetry;

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

bool approxFloat(float a, float b, float eps = 1e-5F) {
    return std::fabs(a - b) <= eps;
}

// Overwrites 4 bytes at `offset` with the little-endian IEEE-754 bits of
// `value`, matching the encoder's float32 layout.
void putFloat(std::vector<std::uint8_t>& bytes, std::size_t offset,
              float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (int i = 0; i < 4; ++i) {
        bytes[offset + i] =
            static_cast<std::uint8_t>((bits >> (8 * i)) & 0xFFu);
    }
}

// A hand-built, encode-ready telemetry with every field set to a distinctive
// value so a round-trip test is unambiguous.
VisionTelemetry makeTestTelemetry() {
    VisionTelemetry telemetry;
    telemetry.version = 1;
    telemetry.sequence = 0xDEADBEEFu;
    telemetry.timestamp_us = 0x0123456789ABCDEFull;
    telemetry.calibration_available = true;
    telemetry.vision_valid = true;
    telemetry.track_available = true;
    telemetry.prediction_valid = true;
    telemetry.measurement_updated = false;
    telemetry.track_state = TrackState::TRACKING;
    telemetry.yaw_rad = 0.125F;
    telemetry.pitch_rad = -0.25F;
    telemetry.x_m = 1.5F;
    telemetry.y_m = -2.5F;
    telemetry.z_m = 3.5F;
    telemetry.prediction_horizon_s = 0.05F;
    telemetry.detection_count = 2;
    telemetry.pnp_measurement_count = 1;
    telemetry.yaw_speed_rad_s = 0.75F;
    telemetry.pitch_speed_rad_s = -0.5F;
    return telemetry;
}

// ── Tests ───────────────────────────────────────────────────────────────────

// Test 1: detector-only result -> no vision/track/prediction data emitted.
void testBuilderDetectorOnly() {
    DiagnosticFrameProcessorResult result;
    result.calibration_available = false;
    result.diagnostic_enabled = false;
    result.observations.push_back({});

    const VisionTelemetry telemetry =
        hnu25::anti_drone::makeVisionTelemetry(result, 0, 0);

    check(!telemetry.calibration_available, "calibration_available == false");
    check(!telemetry.vision_valid, "vision_valid == false");
    check(!telemetry.track_available, "track_available == false");
    check(!telemetry.prediction_valid, "prediction_valid == false");
    check(!telemetry.measurement_updated, "measurement_updated == false");
    check(telemetry.track_state == TrackState::LOST, "track_state == LOST");
    check(telemetry.yaw_rad == 0.0F, "yaw_rad == 0");
    check(telemetry.pitch_rad == 0.0F, "pitch_rad == 0");
    check(telemetry.x_m == 0.0F && telemetry.y_m == 0.0F &&
              telemetry.z_m == 0.0F,
          "xyz == 0");
    check(telemetry.prediction_horizon_s == 0.0F, "prediction_horizon_s == 0");
    check(telemetry.detection_count == 1, "detection_count == 1");
    check(telemetry.pnp_measurement_count == 0, "pnp_measurement_count == 0");
}

// Test 2: valid diagnostic -> every populated field is copied across.
void testBuilderValidDiagnostic() {
    DiagnosticFrameProcessorResult result;
    result.calibration_available = true;
    result.diagnostic_enabled = true;
    result.diagnostic.solution_valid = true;
    result.diagnostic.track_available = true;
    result.diagnostic.prediction_valid = true;
    result.diagnostic.measurement_updated = true;
    result.diagnostic.track_state = TrackState::TRACKING;
    result.diagnostic.compensated_position_gimbal_m = cv::Vec3d(2.0, 0.3, -0.1);
    result.diagnostic.predicted_yaw_rad = 0.15;
    result.diagnostic.predicted_pitch_rad = -0.05;
    result.diagnostic.prediction_horizon_s = 0.02;

    const VisionTelemetry telemetry =
        hnu25::anti_drone::makeVisionTelemetry(result, 123, 456789);

    check(telemetry.version == 1, "version == 1");
    check(telemetry.sequence == 123, "sequence == 123");
    check(telemetry.timestamp_us == 456789, "timestamp_us == 456789");
    check(telemetry.calibration_available, "calibration_available == true");
    check(telemetry.vision_valid, "vision_valid == true");
    check(telemetry.track_available, "track_available == true");
    check(telemetry.prediction_valid, "prediction_valid == true");
    check(telemetry.measurement_updated, "measurement_updated == true");
    check(telemetry.track_state == TrackState::TRACKING,
          "track_state == TRACKING");
    check(approxFloat(telemetry.yaw_rad, 0.15F), "yaw_rad ~ 0.15");
    check(approxFloat(telemetry.pitch_rad, -0.05F), "pitch_rad ~ -0.05");
    check(approxFloat(telemetry.x_m, 2.0F), "x_m ~ 2.0");
    check(approxFloat(telemetry.y_m, 0.3F), "y_m ~ 0.3");
    check(approxFloat(telemetry.z_m, -0.1F), "z_m ~ -0.1");
    check(approxFloat(telemetry.prediction_horizon_s, 0.02F),
          "prediction_horizon_s ~ 0.02");
}

// Test 3: solution_valid == false -> direction / position are zeroed, but the
// tracker-derived flags are still carried through.
void testBuilderInvalidSolutionZeroed() {
    DiagnosticFrameProcessorResult result;
    result.calibration_available = true;
    result.diagnostic_enabled = true;
    result.diagnostic.solution_valid = false;
    result.diagnostic.track_available = true;
    result.diagnostic.track_state = TrackState::TRACKING;
    // Deliberately non-zero values that must be cleared.
    result.diagnostic.predicted_yaw_rad = 0.9;
    result.diagnostic.predicted_pitch_rad = -0.7;
    result.diagnostic.compensated_position_gimbal_m = cv::Vec3d(5.0, 6.0, 7.0);

    const VisionTelemetry telemetry =
        hnu25::anti_drone::makeVisionTelemetry(result, 0, 0);

    check(!telemetry.vision_valid, "vision_valid == false");
    check(telemetry.yaw_rad == 0.0F, "yaw_rad == 0");
    check(telemetry.pitch_rad == 0.0F, "pitch_rad == 0");
    check(telemetry.x_m == 0.0F && telemetry.y_m == 0.0F &&
              telemetry.z_m == 0.0F,
          "xyz == 0");
    check(telemetry.track_available, "track_available == true");
    check(telemetry.track_state == TrackState::TRACKING,
          "track_state == TRACKING");
}

// Test 4: encode -> decode round-trip preserves every field.
void testRoundTrip() {
    const VisionTelemetry telemetry = makeTestTelemetry();
    const std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(telemetry);

    check(bytes.size() == hnu25::anti_drone::kVisionTelemetryPacketSize,
          "packet size fixed");
    check(bytes[0] == 0x41, "magic[0] == 0x41");
    check(bytes[1] == 0x44, "magic[1] == 0x44");
    check(bytes[2] == 1, "version == 1");

    VisionTelemetry decoded;
    check(hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "decode returns true");

    check(decoded.version == 1, "decoded version == 1");
    check(decoded.sequence == 0xDEADBEEFu, "decoded sequence");
    check(decoded.timestamp_us == 0x0123456789ABCDEFull, "decoded timestamp");
    check(decoded.calibration_available, "decoded calibration_available");
    check(decoded.vision_valid, "decoded vision_valid");
    check(decoded.track_available, "decoded track_available");
    check(decoded.prediction_valid, "decoded prediction_valid");
    check(!decoded.measurement_updated, "decoded measurement_updated == false");
    check(decoded.track_state == TrackState::TRACKING, "decoded track_state");
    check(approxFloat(decoded.yaw_rad, 0.125F), "decoded yaw_rad");
    check(approxFloat(decoded.pitch_rad, -0.25F), "decoded pitch_rad");
    check(approxFloat(decoded.x_m, 1.5F), "decoded x_m");
    check(approxFloat(decoded.y_m, -2.5F), "decoded y_m");
    check(approxFloat(decoded.z_m, 3.5F), "decoded z_m");
    check(approxFloat(decoded.prediction_horizon_s, 0.05F),
          "decoded prediction_horizon_s");
    check(decoded.detection_count == 2, "decoded detection_count");
    check(decoded.pnp_measurement_count == 1, "decoded pnp_measurement_count");
    check(approxFloat(decoded.yaw_speed_rad_s, 0.75F),
          "decoded yaw_speed_rad_s");
    check(approxFloat(decoded.pitch_speed_rad_s, -0.5F),
          "decoded pitch_speed_rad_s");
}

// Test 5: flipping a payload byte (before the CRC) breaks the CRC check.
void testCrcCorruption() {
    std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(makeTestTelemetry());
    bytes[4] ^= 0xFFu;  // sequence low byte

    VisionTelemetry decoded;
    check(!hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "corrupt sequence -> decode false");
}

// Test 6: bad magic byte -> decode false.
void testBadMagic() {
    std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(makeTestTelemetry());
    bytes[0] = 0x00u;

    VisionTelemetry decoded;
    check(!hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "bad magic -> decode false");
}

// Test 7: bad version byte -> decode false (rejected before CRC).
void testBadVersion() {
    std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(makeTestTelemetry());
    bytes[2] = 99u;

    VisionTelemetry decoded;
    check(!hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "bad version -> decode false");
}

// Test 8: unknown track-state wire value (with a recomputed valid CRC) is
// rejected.
void testBadTrackState() {
    std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(makeTestTelemetry());
    bytes[18] = 99u;  // track_state byte

    const std::uint16_t crc = hnu25::anti_drone::visionTelemetryCrc16(
        bytes.data(), bytes.size() - 2);
    bytes[bytes.size() - 2] = static_cast<std::uint8_t>(crc & 0xFFu);
    bytes[bytes.size() - 1] = static_cast<std::uint8_t>((crc >> 8) & 0xFFu);

    VisionTelemetry decoded;
    check(!hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "bad track state -> decode false");
}

// Test 9: non-finite solution value -> builder marks vision invalid and zeroes
// direction / position instead of emitting NaN.
void testBuilderNonFinite() {
    DiagnosticFrameProcessorResult result;
    result.calibration_available = true;
    result.diagnostic_enabled = true;
    result.diagnostic.solution_valid = true;
    result.diagnostic.predicted_yaw_rad =
        std::numeric_limits<double>::quiet_NaN();
    result.diagnostic.predicted_pitch_rad = 0.0;
    result.diagnostic.compensated_position_gimbal_m = cv::Vec3d(1.0, 2.0, 3.0);

    const VisionTelemetry telemetry =
        hnu25::anti_drone::makeVisionTelemetry(result, 0, 0);

    check(!telemetry.vision_valid, "vision_valid == false");
    check(telemetry.yaw_rad == 0.0F, "yaw_rad == 0");
    check(telemetry.pitch_rad == 0.0F, "pitch_rad == 0");
    check(telemetry.x_m == 0.0F && telemetry.y_m == 0.0F &&
              telemetry.z_m == 0.0F,
          "xyz == 0");
}

// Test 10: count values above uint16_t max saturate to 65535 rather than
// wrapping.
void testCountSaturation() {
    DiagnosticFrameProcessorResult result;
    result.calibration_available = false;
    result.diagnostic_enabled = false;
    result.observations.resize(65536);
    result.measurements.resize(65536);

    const VisionTelemetry telemetry =
        hnu25::anti_drone::makeVisionTelemetry(result, 0, 0);

    check(telemetry.detection_count == 65535,
          "detection_count saturated to 65535");
    check(telemetry.pnp_measurement_count == 65535,
          "pnp_measurement_count saturated to 65535");
}

// Test 11: encode refuses a version the decoder itself would reject.
void testEncodeRejectsBadVersion() {
    VisionTelemetry telemetry = makeTestTelemetry();
    telemetry.version = 99;

    bool threw = false;
    try {
        hnu25::anti_drone::encodeVisionTelemetry(telemetry);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "version 99 -> encode throws invalid_argument");
}

// Test 12: a finite double that overflows to +inf in float must be treated as
// an invalid solution (no Inf is emitted).
void testBuilderOverflowToInfinity() {
    DiagnosticFrameProcessorResult result;
    result.calibration_available = true;
    result.diagnostic_enabled = true;
    result.diagnostic.solution_valid = true;
    result.diagnostic.predicted_yaw_rad = std::numeric_limits<double>::max();
    result.diagnostic.predicted_pitch_rad = 0.0;
    result.diagnostic.compensated_position_gimbal_m = cv::Vec3d(1.0, 2.0, 3.0);

    const VisionTelemetry telemetry =
        hnu25::anti_drone::makeVisionTelemetry(result, 0, 0);

    check(!telemetry.vision_valid, "vision_valid == false");
    check(telemetry.yaw_rad == 0.0F, "yaw_rad == 0");
    check(telemetry.pitch_rad == 0.0F, "pitch_rad == 0");
    check(telemetry.x_m == 0.0F && telemetry.y_m == 0.0F &&
              telemetry.z_m == 0.0F,
          "xyz == 0");
}

// Test 13: a negative prediction_horizon_s on the wire (with a valid CRC) is
// rejected, mirroring the encode-side finite && >= 0 contract.
void testDecodeRejectsNegativeHorizon() {
    std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(makeTestTelemetry());
    // prediction_horizon_s float32 lives at payload offset 40:
    // header(4) + seq(4) + ts(8) + flags(2) + track(1) + reserved(1)
    // + yaw(4) + pitch(4) + x(4) + y(4) + z(4) = 40.
    putFloat(bytes, 40, -1.0F);

    const std::uint16_t crc = hnu25::anti_drone::visionTelemetryCrc16(
        bytes.data(), bytes.size() - 2);
    bytes[bytes.size() - 2] = static_cast<std::uint8_t>(crc & 0xFFu);
    bytes[bytes.size() - 1] = static_cast<std::uint8_t>((crc >> 8) & 0xFFu);

    VisionTelemetry decoded;
    check(!hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "negative horizon -> decode false");
}

// Test 14: status_flags bit 5 (undefined in v1) must be rejected, not ignored.
void testDecodeRejectsUnknownStatusBits() {
    std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(makeTestTelemetry());
    // status_flags uint16 at payload offset 0 (byte offset 16); bit 5 lives in
    // its low byte.
    bytes[16] |= 0x20u;

    const std::uint16_t crc = hnu25::anti_drone::visionTelemetryCrc16(
        bytes.data(), bytes.size() - 2);
    bytes[bytes.size() - 2] = static_cast<std::uint8_t>(crc & 0xFFu);
    bytes[bytes.size() - 1] = static_cast<std::uint8_t>((crc >> 8) & 0xFFu);

    VisionTelemetry decoded;
    check(!hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "unknown status bit -> decode false");
}

// Test 15: a non-zero reserved byte is rejected.
void testDecodeRejectsNonZeroReserved() {
    std::vector<std::uint8_t> bytes =
        hnu25::anti_drone::encodeVisionTelemetry(makeTestTelemetry());
    // reserved byte at payload offset 2 (byte offset 19).
    bytes[19] = 1u;

    const std::uint16_t crc = hnu25::anti_drone::visionTelemetryCrc16(
        bytes.data(), bytes.size() - 2);
    bytes[bytes.size() - 2] = static_cast<std::uint8_t>(crc & 0xFFu);
    bytes[bytes.size() - 1] = static_cast<std::uint8_t>((crc >> 8) & 0xFFu);

    VisionTelemetry decoded;
    check(!hnu25::anti_drone::decodeVisionTelemetry(bytes, decoded),
          "non-zero reserved -> decode false");
}

// Test 16: an out-of-range TrackState cannot be encoded.
void testEncodeRejectsInvalidTrackState() {
    VisionTelemetry telemetry = makeTestTelemetry();
    telemetry.track_state = static_cast<TrackState>(99);

    bool threw = false;
    try {
        hnu25::anti_drone::encodeVisionTelemetry(telemetry);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "invalid TrackState -> encode throws invalid_argument");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"builder detector-only", testBuilderDetectorOnly},
        {"builder valid diagnostic", testBuilderValidDiagnostic},
        {"builder invalid solution zeroed", testBuilderInvalidSolutionZeroed},
        {"encode/decode round trip", testRoundTrip},
        {"crc corruption", testCrcCorruption},
        {"bad magic", testBadMagic},
        {"bad version", testBadVersion},
        {"bad track state", testBadTrackState},
        {"builder non-finite", testBuilderNonFinite},
        {"count saturation", testCountSaturation},
        {"encode rejects bad version", testEncodeRejectsBadVersion},
        {"builder double->float overflow", testBuilderOverflowToInfinity},
        {"decode rejects negative horizon", testDecodeRejectsNegativeHorizon},
        {"decode rejects unknown status bits", testDecodeRejectsUnknownStatusBits},
        {"decode rejects non-zero reserved", testDecodeRejectsNonZeroReserved},
        {"encode rejects invalid track state", testEncodeRejectsInvalidTrackState},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone vision telemetry tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
