#pragma once

#include <cstdint>
#include <vector>

namespace hnu25::anti_drone {

// Accumulated counters for a VisionTelemetryTransport. These express only the
// transport of vision-result bytes — no control / fire / gimbal / aim
// semantics.
struct VisionTelemetryTransportStats {
    std::uint64_t packets_submitted = 0;
    std::uint64_t packets_accepted = 0;
    std::uint64_t bytes_accepted = 0;
    std::uint64_t failures = 0;
};

// Abstract transport for handing encoded VisionTelemetry packets (50 bytes,
// v1 wire format) to a downstream consumer. A concrete transport may be
// in-memory (loopback), a serial link, a socket, etc. — but this interface
// itself expresses only "send VisionTelemetry bytes". It never expresses
// control / fire / gimbal / aim commands.
class VisionTelemetryTransport {
public:
    virtual ~VisionTelemetryTransport() = default;

    // Sends one encoded packet. Returns true on acceptance by the transport,
    // false otherwise.
    virtual bool send(const std::vector<std::uint8_t>& packet) = 0;

    virtual const VisionTelemetryTransportStats& stats() const noexcept = 0;
};

}  // namespace hnu25::anti_drone
