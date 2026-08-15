#pragma once

#include "anti_drone/vision_telemetry.hpp"
#include "anti_drone/vision_telemetry_stream.hpp"
#include "anti_drone/vision_telemetry_transport.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace hnu25::anti_drone {

// In-memory transport that feeds every sent packet straight into a
// VisionTelemetryStreamParser, verifying that the runtime's encoded output can
// be re-parsed by the receiving side. It touches no serial port, socket,
// /dev/*, or real hardware.
class VisionTelemetryLoopbackTransport : public VisionTelemetryTransport {
public:
    VisionTelemetryLoopbackTransport() = default;

    bool send(const std::vector<std::uint8_t>& packet) override;

    const VisionTelemetryTransportStats& stats() const noexcept override;

    // Reads the next successfully received (decoded) telemetry in FIFO order.
    bool popReceived(VisionTelemetry& telemetry);

    // Receiver-side stream statistics (what the internal parser observed).
    const VisionTelemetryStreamStats& receiverStats() const noexcept;

private:
    VisionTelemetryStreamParser parser_;
    VisionTelemetryTransportStats stats_;
    std::deque<VisionTelemetry> received_;
};

}  // namespace hnu25::anti_drone
