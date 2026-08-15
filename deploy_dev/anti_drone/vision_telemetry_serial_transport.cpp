#include "anti_drone/vision_telemetry_serial_transport.hpp"

#include "anti_drone/vision_telemetry.hpp"

#include <utility>

namespace hnu25::anti_drone {

VisionTelemetrySerialTransport::VisionTelemetrySerialTransport(
    VisionTelemetrySerialTransportConfig config)
    : config_(std::move(config)) {}

VisionTelemetrySerialTransport::~VisionTelemetrySerialTransport() {
    close();
}

bool VisionTelemetrySerialTransport::open() {
    return serial_.open(config_.device, config_.baud_rate);
}

void VisionTelemetrySerialTransport::close() noexcept {
    serial_.close();
}

bool VisionTelemetrySerialTransport::isOpen() const noexcept {
    return serial_.isOpen();
}

bool VisionTelemetrySerialTransport::send(
    const std::vector<std::uint8_t>& packet) {
    ++stats_.packets_submitted;

    if (packet.size() != kVisionTelemetryPacketSize) {
        ++stats_.failures;
        return false;
    }
    if (!serial_.isOpen()) {
        ++stats_.failures;
        return false;
    }
    if (!serial_.write(packet)) {
        ++stats_.failures;
        return false;
    }

    ++stats_.packets_accepted;
    stats_.bytes_accepted += kVisionTelemetryPacketSize;
    if (config_.flush_after_write) {
        serial_.flush();
    }
    return true;
}

const VisionTelemetryTransportStats&
VisionTelemetrySerialTransport::stats() const noexcept {
    return stats_;
}

}  // namespace hnu25::anti_drone
