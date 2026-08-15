#include "anti_drone/vision_telemetry_loopback.hpp"

#include <cstdint>
#include <vector>

namespace hnu25::anti_drone {

bool VisionTelemetryLoopbackTransport::send(
    const std::vector<std::uint8_t>& packet) {
    ++stats_.packets_submitted;

    if (packet.size() != kVisionTelemetryPacketSize) {
        ++stats_.failures;
        return false;
    }

    parser_.push(packet);

    VisionTelemetry telemetry;
    if (parser_.pop(telemetry)) {
        ++stats_.packets_accepted;
        stats_.bytes_accepted += kVisionTelemetryPacketSize;
        received_.push_back(telemetry);
        return true;
    }

    ++stats_.failures;
    return false;
}

const VisionTelemetryTransportStats&
VisionTelemetryLoopbackTransport::stats() const noexcept {
    return stats_;
}

bool VisionTelemetryLoopbackTransport::popReceived(
    VisionTelemetry& telemetry) {
    if (received_.empty()) {
        return false;
    }
    telemetry = received_.front();
    received_.pop_front();
    return true;
}

const VisionTelemetryStreamStats&
VisionTelemetryLoopbackTransport::receiverStats() const noexcept {
    return parser_.stats();
}

}  // namespace hnu25::anti_drone
