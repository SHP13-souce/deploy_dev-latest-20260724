#pragma once

#include "anti_drone/vision_telemetry_transport.hpp"
#include "anti_drone/serial_port.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace hnu25::anti_drone {

// Configuration for the diagnostic serial transport. It mirrors only the
// fields this transport needs, independently of the full AntiDroneConfig, so
// this header does not depend on config.hpp.
struct VisionTelemetrySerialTransportConfig {
    std::string device;
    int baud_rate = 115200;
    bool flush_after_write = false;
};

// Serial transport for the fixed-size VisionTelemetry v1 packet. It only
// validates packet size, manages the open/close lifecycle, delegates the actual
// I/O to hnu25::SerialPort, and accumulates statistics. It performs no
// retry/resend (a stale visual result has no value), and it expresses no
// control / fire / gimbal / actuator semantics.
class VisionTelemetrySerialTransport final : public VisionTelemetryTransport {
public:
    explicit VisionTelemetrySerialTransport(
        VisionTelemetrySerialTransportConfig config);

    ~VisionTelemetrySerialTransport() override;

    // Opens the configured device. Returns false on failure (does not throw).
    bool open();

    void close() noexcept;

    bool isOpen() const noexcept;

    bool send(const std::vector<std::uint8_t>& packet) override;

    const VisionTelemetryTransportStats& stats() const noexcept override;

private:
    VisionTelemetrySerialTransportConfig config_;
    hnu25::SerialPort serial_;
    VisionTelemetryTransportStats stats_;
};

}  // namespace hnu25::anti_drone
