#include "vest_runtime/vision_protocol.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using hnu25::vest::VisionTargetObservation;
using hnu25::vest::VisionTrackingState;
using hnu25::vest_runtime::VisionProtocolPacket;
using hnu25::vest_runtime::kVisionProtocolPacketSize;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("test failure: " + message);
    }
}

template <typename Exception, typename Func>
void requireThrows(Func&& func, const std::string& name) {
    try {
        func();
        throw std::runtime_error(
            "test failure: " + name + " did not throw");
    } catch (const Exception&) {
        return;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "test failure: " + name +
            " threw wrong exception type: " + e.what());
    } catch (...) {
        throw std::runtime_error(
            "test failure: " + name + " threw unknown exception");
    }
}

// ── Test-only reference CRC-32 ──────────────────────────────────────

std::uint32_t referenceCrc32(const std::uint8_t* data,
                              std::size_t len) {
    std::uint32_t crc = 0xFFFFFFFFu;

    for (std::size_t i = 0; i < len; ++i) {
        crc ^= static_cast<std::uint32_t>(data[i]);

        for (int bit = 0; bit < 8; ++bit) {
            if ((crc & 1u) != 0u) {
                crc = (crc >> 1) ^ 0xEDB88320u;
            } else {
                crc >>= 1;
            }
        }
    }

    return crc ^ 0xFFFFFFFFu;
}

void writeU32LEForTest(std::uint8_t* dst, std::uint32_t value) {
    dst[0] = static_cast<std::uint8_t>(value);
    dst[1] = static_cast<std::uint8_t>(value >> 8);
    dst[2] = static_cast<std::uint8_t>(value >> 16);
    dst[3] = static_cast<std::uint8_t>(value >> 24);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: Golden Tracking packet
// ═══════════════════════════════════════════════════════════════════════

void testGoldenTrackingPacket() {
    VisionTargetObservation obs;

    obs.frame_id = 0x0102030405060708ULL;
    obs.measurement_timestamp_us = 0x1112131415161718ULL;

    obs.has_target = true;
    obs.target_valid = true;
    obs.track_id = static_cast<std::int32_t>(0x01020304u);
    obs.tracking_state = VisionTrackingState::Tracking;

    obs.center_x = 0.5F;
    obs.center_y = 0.25F;
    obs.error_x = -0.125F;
    obs.error_y = 0.125F;
    obs.velocity_x = 1.5F;
    obs.velocity_y = -2.0F;
    obs.predicted_x = 1.25F;
    obs.predicted_y = -0.5F;
    obs.bbox_w = 0.25F;
    obs.bbox_h = 0.5F;
    obs.confidence = 0.75F;

    const auto packet =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    require(packet.size() == kVisionProtocolPacketSize,
            "packet size must be 80");

    const VisionProtocolPacket expected = {
        0x56, 0x4F, 0x42, 0x53, 0x01, 0x01, 0x03, 0x01,
        0x40, 0x00, 0x00, 0x00, 0x08, 0x07, 0x06, 0x05,
        0x04, 0x03, 0x02, 0x01, 0x18, 0x17, 0x16, 0x15,
        0x14, 0x13, 0x12, 0x11, 0x04, 0x03, 0x02, 0x01,
        0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x80, 0x3E,
        0x00, 0x00, 0x00, 0xBE, 0x00, 0x00, 0x00, 0x3E,
        0x00, 0x00, 0xC0, 0x3F, 0x00, 0x00, 0x00, 0xC0,
        0x00, 0x00, 0xA0, 0x3F, 0x00, 0x00, 0x00, 0xBF,
        0x00, 0x00, 0x80, 0x3E, 0x00, 0x00, 0x00, 0x3F,
        0x00, 0x00, 0x40, 0x3F, 0x36, 0x0D, 0x53, 0x26
    };

    require(packet == expected, "golden Tracking packet mismatch");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: Tracking round-trip
// ═══════════════════════════════════════════════════════════════════════

void testTrackingRoundTrip() {
    VisionTargetObservation obs;

    obs.frame_id = 0x0102030405060708ULL;
    obs.measurement_timestamp_us = 0x1112131415161718ULL;
    obs.has_target = true;
    obs.target_valid = true;
    obs.track_id = static_cast<std::int32_t>(0x01020304u);
    obs.tracking_state = VisionTrackingState::Tracking;
    obs.center_x = 0.5F;
    obs.center_y = 0.25F;
    obs.error_x = -0.125F;
    obs.error_y = 0.125F;
    obs.velocity_x = 1.5F;
    obs.velocity_y = -2.0F;
    obs.predicted_x = 1.25F;
    obs.predicted_y = -0.5F;
    obs.bbox_w = 0.25F;
    obs.bbox_h = 0.5F;
    obs.confidence = 0.75F;

    const auto packet =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    const auto decoded =
        hnu25::vest_runtime::decodeVisionObservation(packet);

    require(decoded.frame_id == obs.frame_id,
            "round-trip: frame_id");
    require(decoded.measurement_timestamp_us ==
                obs.measurement_timestamp_us,
            "round-trip: timestamp");
    require(decoded.has_target == obs.has_target,
            "round-trip: has_target");
    require(decoded.target_valid == obs.target_valid,
            "round-trip: target_valid");
    require(decoded.track_id == obs.track_id,
            "round-trip: track_id");
    require(decoded.tracking_state == obs.tracking_state,
            "round-trip: tracking_state");

    require(decoded.center_x == obs.center_x,
            "round-trip: center_x");
    require(decoded.center_y == obs.center_y,
            "round-trip: center_y");
    require(decoded.error_x == obs.error_x,
            "round-trip: error_x");
    require(decoded.error_y == obs.error_y,
            "round-trip: error_y");
    require(decoded.velocity_x == obs.velocity_x,
            "round-trip: velocity_x");
    require(decoded.velocity_y == obs.velocity_y,
            "round-trip: velocity_y");
    require(decoded.predicted_x == obs.predicted_x,
            "round-trip: predicted_x");
    require(decoded.predicted_y == obs.predicted_y,
            "round-trip: predicted_y");
    require(decoded.bbox_w == obs.bbox_w,
            "round-trip: bbox_w");
    require(decoded.bbox_h == obs.bbox_h,
            "round-trip: bbox_h");
    require(decoded.confidence == obs.confidence,
            "round-trip: confidence");

    const auto reencoded =
        hnu25::vest_runtime::encodeVisionObservation(decoded);

    require(reencoded == packet,
            "round-trip: re-encoded packet must match original");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: NoTarget + track_id=-1 round-trip
// ═══════════════════════════════════════════════════════════════════════

void testNoTargetRoundTrip() {
    VisionTargetObservation obs;
    obs.frame_id = 99;

    const auto packet =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    const auto decoded =
        hnu25::vest_runtime::decodeVisionObservation(packet);

    require(decoded.frame_id == 99,
            "NoTarget round-trip: frame_id");
    require(decoded.has_target == false,
            "NoTarget round-trip: has_target");
    require(decoded.target_valid == false,
            "NoTarget round-trip: target_valid");
    require(decoded.track_id == -1,
            "NoTarget round-trip: track_id");
    require(decoded.tracking_state == VisionTrackingState::NoTarget,
            "NoTarget round-trip: state");

    // Verify -1 wire bit pattern
    require(packet[28] == 0xFF, "NoTarget: track_id byte 0");
    require(packet[29] == 0xFF, "NoTarget: track_id byte 1");
    require(packet[30] == 0xFF, "NoTarget: track_id byte 2");
    require(packet[31] == 0xFF, "NoTarget: track_id byte 3");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 4: TemporarilyLost round-trip
// ═══════════════════════════════════════════════════════════════════════

void testTemporarilyLostRoundTrip() {
    VisionTargetObservation obs;
    obs.frame_id = 123;
    obs.measurement_timestamp_us = 456789;
    obs.has_target = true;
    obs.target_valid = false;
    obs.track_id = 7;
    obs.tracking_state = VisionTrackingState::TemporarilyLost;

    const auto packet =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    const auto decoded =
        hnu25::vest_runtime::decodeVisionObservation(packet);

    require(decoded.has_target == true,
            "TemporarilyLost round-trip: has_target");
    require(decoded.target_valid == false,
            "TemporarilyLost round-trip: target_valid");
    require(decoded.track_id == 7,
            "TemporarilyLost round-trip: track_id");
    require(decoded.tracking_state ==
                VisionTrackingState::TemporarilyLost,
            "TemporarilyLost round-trip: state");
    require(decoded.measurement_timestamp_us == 456789,
            "TemporarilyLost round-trip: timestamp");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 5: Tracking + target_valid=false is legal
// ═══════════════════════════════════════════════════════════════════════

void testTrackingWithoutValidMeasurement() {
    VisionTargetObservation obs;
    obs.has_target = true;
    obs.target_valid = false;
    obs.track_id = 3;
    obs.tracking_state = VisionTrackingState::Tracking;

    const auto packet =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    const auto decoded =
        hnu25::vest_runtime::decodeVisionObservation(packet);

    require(decoded.has_target == true,
            "Tracking+!valid: has_target");
    require(decoded.target_valid == false,
            "Tracking+!valid: target_valid");
    require(decoded.track_id == 3,
            "Tracking+!valid: track_id");
    require(decoded.tracking_state == VisionTrackingState::Tracking,
            "Tracking+!valid: state");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 6: encode rejects invalid observations
// ═══════════════════════════════════════════════════════════════════════

void testEncodeValidation() {
    // A: has_target=false, target_valid=true, Tracking
    requireThrows<std::invalid_argument>([]() {
        VisionTargetObservation obs;
        obs.has_target = false;
        obs.target_valid = true;
        obs.tracking_state = VisionTrackingState::Tracking;
        hnu25::vest_runtime::encodeVisionObservation(obs);
    }, "encode: !has_target + valid + Tracking");

    // B: has_target=false, Tracking
    requireThrows<std::invalid_argument>([]() {
        VisionTargetObservation obs;
        obs.has_target = false;
        obs.tracking_state = VisionTrackingState::Tracking;
        hnu25::vest_runtime::encodeVisionObservation(obs);
    }, "encode: !has_target + Tracking");

    // C: NaN center_x
    requireThrows<std::invalid_argument>([]() {
        VisionTargetObservation obs;
        obs.has_target = true;
        obs.target_valid = false;
        obs.track_id = 1;
        obs.tracking_state = VisionTrackingState::TemporarilyLost;
        obs.center_x = std::numeric_limits<float>::quiet_NaN();
        hnu25::vest_runtime::encodeVisionObservation(obs);
    }, "encode: NaN center_x");

    // D: infinity confidence
    requireThrows<std::invalid_argument>([]() {
        VisionTargetObservation obs;
        obs.has_target = true;
        obs.target_valid = true;
        obs.track_id = 1;
        obs.tracking_state = VisionTrackingState::Tracking;
        obs.confidence =
            std::numeric_limits<float>::infinity();
        hnu25::vest_runtime::encodeVisionObservation(obs);
    }, "encode: infinity confidence");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 7-8: null / size validation
// ═══════════════════════════════════════════════════════════════════════

void testNullAndSizeValidation() {
    // nullptr
    requireThrows<std::invalid_argument>([]() {
        hnu25::vest_runtime::decodeVisionObservation(
            nullptr, kVisionProtocolPacketSize);
    }, "decode: nullptr");

    // size 79
    requireThrows<std::runtime_error>([&]() {
        VisionTargetObservation obs;
        obs.frame_id = 1;
        const auto packet =
            hnu25::vest_runtime::encodeVisionObservation(obs);
        hnu25::vest_runtime::decodeVisionObservation(
            packet.data(), 79);
    }, "decode: size 79");

    // size 81
    requireThrows<std::runtime_error>([&]() {
        VisionTargetObservation obs;
        obs.frame_id = 1;
        const auto packet =
            hnu25::vest_runtime::encodeVisionObservation(obs);
        hnu25::vest_runtime::decodeVisionObservation(
            packet.data(), 81);
    }, "decode: size 81");
}

// ═══════════════════════════════════════════════════════════════════════
// Test 9-12,15-16: header validation
// ═══════════════════════════════════════════════════════════════════════

VisionProtocolPacket makeValidTrackingPacket() {
    VisionTargetObservation obs;
    obs.has_target = true;
    obs.target_valid = true;
    obs.track_id = 1;
    obs.tracking_state = VisionTrackingState::Tracking;
    return hnu25::vest_runtime::encodeVisionObservation(obs);
}

void testHeaderValidation() {
    // magic corruption
    {
        auto corrupted = makeValidTrackingPacket();
        corrupted[0] ^= 0x01;
        requireThrows<std::runtime_error>([&]() {
            hnu25::vest_runtime::decodeVisionObservation(corrupted);
        }, "decode: magic corruption");
    }

    // version
    {
        auto corrupted = makeValidTrackingPacket();
        corrupted[4] = 2;
        requireThrows<std::runtime_error>([&]() {
            hnu25::vest_runtime::decodeVisionObservation(corrupted);
        }, "decode: version corruption");
    }

    // reserved flag bit
    {
        auto corrupted = makeValidTrackingPacket();
        corrupted[6] |= 0x04;
        requireThrows<std::runtime_error>([&]() {
            hnu25::vest_runtime::decodeVisionObservation(corrupted);
        }, "decode: reserved flag bit");
    }

    // invalid tracking_state
    {
        auto corrupted = makeValidTrackingPacket();
        corrupted[7] = 3;
        requireThrows<std::runtime_error>([&]() {
            hnu25::vest_runtime::decodeVisionObservation(corrupted);
        }, "decode: invalid tracking_state");
    }

    // reserved uint16
    {
        auto corrupted = makeValidTrackingPacket();
        corrupted[10] = 1;
        requireThrows<std::runtime_error>([&]() {
            hnu25::vest_runtime::decodeVisionObservation(corrupted);
        }, "decode: reserved uint16");
    }

    // payload_size
    {
        auto corrupted = makeValidTrackingPacket();
        corrupted[8] = 63;
        corrupted[9] = 0;
        requireThrows<std::runtime_error>([&]() {
            hnu25::vest_runtime::decodeVisionObservation(corrupted);
        }, "decode: payload_size");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test 13: CRC corruption
// ═══════════════════════════════════════════════════════════════════════

void testCrcValidation() {
    auto corrupted = makeValidTrackingPacket();

    corrupted[32] ^= 0x01;

    requireThrows<std::runtime_error>([&]() {
        hnu25::vest_runtime::decodeVisionObservation(corrupted);
    }, "decode: CRC corruption");

    try {
        hnu25::vest_runtime::decodeVisionObservation(corrupted);
        throw std::runtime_error(
            "test failure: CRC corruption did not throw");
    } catch (const std::runtime_error& e) {
        const std::string what(e.what());
        require(what.find("CRC") != std::string::npos,
                "CRC error should mention 'CRC'");
    } catch (...) {
        throw std::runtime_error(
            "test failure: CRC corruption threw wrong type");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Test 14: semantic-invalid wire with valid CRC
// ═══════════════════════════════════════════════════════════════════════

void testSemanticWireValidation() {
    auto packet = makeValidTrackingPacket();

    // Clear flags: has_target=false, target_valid=false
    packet[6] = 0;

    // Recompute CRC with test-only reference
    const std::uint32_t new_crc =
        referenceCrc32(packet.data(), 76);
    writeU32LEForTest(&packet[76], new_crc);

    try {
        hnu25::vest_runtime::decodeVisionObservation(packet);
        throw std::runtime_error(
            "test failure: semantic-invalid wire did not throw");
    } catch (const std::runtime_error& e) {
        const std::string what(e.what());
        require(
            what.find("invalid observation semantics") !=
                std::string::npos,
            "semantic error should mention 'invalid observation semantics'");
    } catch (...) {
        throw std::runtime_error(
            "test failure: semantic-invalid wire threw wrong type");
    }
}

}  // namespace

int main() {
    try {
        testGoldenTrackingPacket();
        testTrackingRoundTrip();
        testNoTargetRoundTrip();
        testTemporarilyLostRoundTrip();
        testTrackingWithoutValidMeasurement();
        testEncodeValidation();
        testNullAndSizeValidation();
        testHeaderValidation();
        testCrcValidation();
        testSemanticWireValidation();

        std::cout << "vest vision protocol tests passed\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "vest vision protocol test failed: "
                  << e.what() << '\n';

        return 1;
    }
}
