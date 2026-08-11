#pragma once

#include "vest/vision_target_core.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace hnu25::vest_runtime {

inline constexpr std::size_t kVisionProtocolPacketSize = 80;

inline constexpr std::uint8_t kVisionProtocolVersion = 1;

inline constexpr std::uint8_t kVisionObservationMessageType = 1;

inline constexpr std::uint16_t kVisionObservationPayloadSize = 64;

using VisionProtocolPacket =
    std::array<std::uint8_t, kVisionProtocolPacketSize>;

VisionProtocolPacket encodeVisionObservation(
    const hnu25::vest::VisionTargetObservation& observation);

hnu25::vest::VisionTargetObservation decodeVisionObservation(
    const std::uint8_t* data,
    std::size_t size);

inline hnu25::vest::VisionTargetObservation decodeVisionObservation(
    const VisionProtocolPacket& packet) {
    return decodeVisionObservation(packet.data(), packet.size());
}

}  // namespace hnu25::vest_runtime
