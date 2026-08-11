#include "vest_runtime/serial_observation_sink.hpp"
#include "vest_runtime/serial_port.hpp"
#include "vest_runtime/vision_protocol.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <pty.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using hnu25::vest::VisionTargetObservation;
using hnu25::vest::VisionTrackingState;
using hnu25::vest_runtime::kVisionProtocolPacketSize;
using hnu25::vest_runtime::SerialObservationSink;
using hnu25::vest_runtime::SerialPort;
using hnu25::vest_runtime::SerialPortConfig;
using hnu25::vest_runtime::VisionProtocolPacket;

// ═══════════════════════════════════════════════════════════════════════════
// Test helpers
// ═══════════════════════════════════════════════════════════════════════════

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("test failure: " + message);
    }
}

template <typename Exception, typename Func>
void requireThrows(Func&& func, const std::string& name) {
    bool caught_expected = false;

    try {
        func();
    } catch (const Exception&) {
        caught_expected = true;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "test failure: " + name +
            " threw wrong exception type: " + e.what());
    } catch (...) {
        throw std::runtime_error(
            "test failure: " + name +
            " threw unknown exception");
    }

    if (!caught_expected) {
        throw std::runtime_error(
            "test failure: " + name + " did not throw");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PtyPair — RAII pseudo-terminal pair
// ═══════════════════════════════════════════════════════════════════════════

class PtyPair {
public:
    PtyPair() {
        int master = -1;
        int slave = -1;

        if (::openpty(&master, &slave, nullptr, nullptr, nullptr) < 0) {
            const int error_number = errno;
            throw std::runtime_error(
                std::string("openpty failed: ") +
                std::strerror(error_number));
        }

        // Get slave device path, then close the slave fd.
        // SerialPort will reopen the slave by path.
        char name_buffer[256]{};
        const int ttyname_result =
            ::ttyname_r(slave, name_buffer, sizeof(name_buffer));
        if (ttyname_result != 0) {
            ::close(slave);
            ::close(master);
            throw std::runtime_error(
                std::string("ttyname_r failed: ") +
                std::strerror(ttyname_result));
        }

        slave_path_ = name_buffer;

        if (::close(slave) < 0) {
            const int error_number = errno;
            ::close(master);
            throw std::runtime_error(
                std::string("close slave fd failed: ") +
                std::strerror(error_number));
        }

        master_fd_ = master;
    }

    ~PtyPair() noexcept {
        if (master_fd_ >= 0) {
            ::close(master_fd_);
            master_fd_ = -1;
        }
    }

    PtyPair(const PtyPair&) = delete;
    PtyPair& operator=(const PtyPair&) = delete;

    int masterFd() const noexcept { return master_fd_; }

    const std::string& slavePath() const noexcept {
        return slave_path_;
    }

private:
    int master_fd_ = -1;
    std::string slave_path_;
};

// ═══════════════════════════════════════════════════════════════════════════
// readExactlyWithTimeout — blocking read with poll-based timeout
// ═══════════════════════════════════════════════════════════════════════════

void readExactlyWithTimeout(int fd,
                            std::uint8_t* data,
                            std::size_t size) {
    constexpr int kTimeoutMs = 1000;

    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::milliseconds(kTimeoutMs);

    std::size_t offset = 0;

    while (offset < size) {
        // Calculate remaining timeout
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
            throw std::runtime_error(
                "timed out waiting for PTY data");
        }

        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now);

        struct pollfd pfd {};
        pfd.fd = fd;
        pfd.events = POLLIN;

        const int poll_ret =
            ::poll(&pfd, 1, static_cast<int>(remaining.count()));

        if (poll_ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            const int error_number = errno;
            throw std::runtime_error(
                std::string("poll failed: ") +
                std::strerror(error_number));
        }

        if (poll_ret == 0) {
            throw std::runtime_error(
                "timed out waiting for PTY data");
        }

        if ((pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) != 0) {
            throw std::runtime_error(
                "PTY master fd error/hangup");
        }

        if ((pfd.revents & POLLIN) == 0) {
            continue;
        }

        const ssize_t n =
            ::read(fd, data + offset, size - offset);

        if (n > 0) {
            offset += static_cast<std::size_t>(n);
            continue;
        }

        if (n < 0 && errno == EINTR) {
            continue;
        }

        if (n == 0) {
            throw std::runtime_error(
                "PTY master read returned EOF");
        }

        const int error_number = errno;
        throw std::runtime_error(
            std::string("PTY master read failed: ") +
            std::strerror(error_number));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: SerialPort config validation
// ═══════════════════════════════════════════════════════════════════════════

void testSerialConfigValidation() {
    // A: empty device
    requireThrows<std::invalid_argument>([]() {
        SerialPortConfig cfg;
        cfg.device = "";
        cfg.baud_rate = 115200;
        SerialPort port(cfg);
    }, "empty device");

    // B: baud_rate <= 0
    requireThrows<std::invalid_argument>([]() {
        SerialPortConfig cfg;
        cfg.device = "/dev/null";
        cfg.baud_rate = 0;
        SerialPort port(cfg);
    }, "baud_rate <= 0");

    // C: unsupported positive baud
    requireThrows<std::invalid_argument>([]() {
        SerialPortConfig cfg;
        cfg.device = "/dev/null";
        cfg.baud_rate = 123456;
        SerialPort port(cfg);
    }, "unsupported baud rate");

    // D: nonexistent device
    requireThrows<std::runtime_error>([]() {
        SerialPortConfig cfg;
        cfg.device = "/dev/hnu25_this_device_should_not_exist";
        cfg.baud_rate = 115200;
        SerialPort port(cfg);
    }, "nonexistent device");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: SerialPort raw write through PTY
// ═══════════════════════════════════════════════════════════════════════════

void testSerialPortRawWrite() {
    PtyPair pty;

    SerialPortConfig config;
    config.device = pty.slavePath();
    config.baud_rate = 115200;

    SerialPort port(config);

    // writeAll(nullptr, 0) — must succeed
    port.writeAll(nullptr, 0);

    // writeAll(nullptr, 1) — must throw
    requireThrows<std::invalid_argument>([&port]() {
        port.writeAll(nullptr, 1);
    }, "writeAll(nullptr, size>0)");

    // Write known bytes
    const std::array<std::uint8_t, 8> tx = {
        0x00, 0x01, 0x02, 0x7F,
        0x80, 0xFE, 0xFF, 0x55
    };

    port.writeAll(tx.data(), tx.size());

    // Read back from master
    std::array<std::uint8_t, 8> rx{};
    readExactlyWithTimeout(pty.masterFd(), rx.data(), rx.size());

    require(rx == tx, "raw write: received bytes must match sent");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: SerialObservationSink Tracking packet
// ═══════════════════════════════════════════════════════════════════════════

void testTrackingObservationSink() {
    PtyPair pty;

    SerialPortConfig config;
    config.device = pty.slavePath();
    config.baud_rate = 115200;

    SerialObservationSink sink(config);

    VisionTargetObservation obs;
    obs.frame_id = 42;
    obs.measurement_timestamp_us = 123456789;
    obs.has_target = true;
    obs.target_valid = true;
    obs.track_id = 7;
    obs.tracking_state = VisionTrackingState::Tracking;
    obs.center_x = 0.5F;
    obs.center_y = 0.25F;
    obs.error_x = 0.0F;
    obs.error_y = -0.25F;
    obs.velocity_x = 0.1F;
    obs.velocity_y = -0.2F;
    obs.predicted_x = 0.6F;
    obs.predicted_y = 0.3F;
    obs.bbox_w = 0.2F;
    obs.bbox_h = 0.4F;
    obs.confidence = 0.9F;

    const VisionProtocolPacket expected =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    sink.onObservation(obs);

    VisionProtocolPacket received{};
    readExactlyWithTimeout(pty.masterFd(),
                           received.data(),
                           received.size());

    require(received == expected,
            "Tracking: full 80-byte packet must match expected");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: NoTarget observation is sent (not filtered)
// ═══════════════════════════════════════════════════════════════════════════

void testNoTargetObservationSink() {
    PtyPair pty;

    SerialPortConfig config;
    config.device = pty.slavePath();
    config.baud_rate = 115200;

    SerialObservationSink sink(config);

    VisionTargetObservation obs;
    obs.frame_id = 100;
    // has_target=false, target_valid=false, track_id=-1, NoTarget (defaults)

    const VisionProtocolPacket expected =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    sink.onObservation(obs);

    VisionProtocolPacket received{};
    readExactlyWithTimeout(pty.masterFd(),
                           received.data(),
                           received.size());

    require(received == expected,
            "NoTarget: full 80-byte packet must match expected");
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: TemporarilyLost observation is sent (not filtered)
// ═══════════════════════════════════════════════════════════════════════════

void testTemporarilyLostObservationSink() {
    PtyPair pty;

    SerialPortConfig config;
    config.device = pty.slavePath();
    config.baud_rate = 115200;

    SerialObservationSink sink(config);

    VisionTargetObservation obs;
    obs.frame_id = 101;
    obs.measurement_timestamp_us = 555;
    obs.has_target = true;
    obs.target_valid = false;
    obs.track_id = 3;
    obs.tracking_state = VisionTrackingState::TemporarilyLost;
    // all floats zero (default, finite)

    const VisionProtocolPacket expected =
        hnu25::vest_runtime::encodeVisionObservation(obs);

    sink.onObservation(obs);

    VisionProtocolPacket received{};
    readExactlyWithTimeout(pty.masterFd(),
                           received.data(),
                           received.size());

    require(received == expected,
            "TemporarilyLost: full 80-byte packet must match expected");
}

}  // namespace

int main() {
    try {
        testSerialConfigValidation();
        testSerialPortRawWrite();
        testTrackingObservationSink();
        testNoTargetObservationSink();
        testTemporarilyLostObservationSink();

        std::cout << "vest serial transport tests passed\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "vest serial transport test failed: "
                  << e.what() << '\n';

        return 1;
    }
}
