// Linux-only integration test for VisionTelemetrySerialTransport.
//
// It exercises the transport against a pseudo-terminal (PTY) so the full
// open -> write -> read -> parse round-trip is verified without any real
// serial hardware. It also covers the transport's defensive behaviour for
// wrong-size packets, send-before-open, send-after-close, and a nonexistent
// device path.
//
// This test never opens a real camera, a real serial port, or any other
// hardware; the PTY is created entirely in software via posix_openpt().

#ifdef __linux__
#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // posix_openpt / grantpt / unlockpt / ptsname
#endif
#endif

#include "anti_drone/vision_telemetry.hpp"
#include "anti_drone/vision_telemetry_serial_transport.hpp"
#include "anti_drone/vision_telemetry_stream.hpp"

#ifdef __linux__
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#endif

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "[ OK ] " << message << '\n';
    } else {
        std::cerr << "FAILED: " << message << '\n';
        ++g_failures;
    }
}

// A minimal, legal telemetry with a distinct sequence number. yaw/pitch/xyz
// are finite so encodeVisionTelemetry accepts the vision_valid=true packet.
hnu25::anti_drone::VisionTelemetry makePacket(std::uint32_t sequence) {
    hnu25::anti_drone::VisionTelemetry telemetry;
    telemetry.version = hnu25::anti_drone::kVisionTelemetryVersion;
    telemetry.sequence = sequence;
    telemetry.timestamp_us = static_cast<std::uint64_t>(sequence) * 1000;
    telemetry.vision_valid = true;
    telemetry.track_state = hnu25::anti_drone::TrackState::TRACKING;
    telemetry.yaw_rad = 0.1F * static_cast<float>(sequence);
    telemetry.pitch_rad = 0.2F;
    telemetry.x_m = 1.0F;
    telemetry.y_m = 2.0F;
    telemetry.z_m = 3.0F;
    telemetry.prediction_horizon_s = 0.0F;
    telemetry.detection_count = 1;
    telemetry.pnp_measurement_count = 4;
    return telemetry;
}

#ifdef __linux__

struct Pty {
    int master_fd = -1;
    std::string slave_path;
};

bool makePty(Pty& pty) {
    pty.master_fd = posix_openpt(O_RDWR | O_NOCTTY);
    if (pty.master_fd < 0) {
        return false;
    }
    if (grantpt(pty.master_fd) != 0 || unlockpt(pty.master_fd) != 0) {
        ::close(pty.master_fd);
        pty.master_fd = -1;
        return false;
    }
    char* name = ptsname(pty.master_fd);
    if (name == nullptr) {
        ::close(pty.master_fd);
        pty.master_fd = -1;
        return false;
    }
    pty.slave_path = name;
    return true;
}

bool setNonBlocking(int fd) {
    const int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) {
        return false;
    }
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

// Drains every buffered byte from the master side (non-blocking).
std::vector<std::uint8_t> readAll(int fd) {
    std::vector<std::uint8_t> out;
    std::uint8_t buf[512];
    for (;;) {
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n > 0) {
            out.insert(out.end(), buf, buf + n);
            continue;
        }
        break;  // EOF, EAGAIN, or error: nothing more is buffered.
    }
    return out;
}

struct RoundTripResult {
    std::vector<std::uint32_t> sequences;
    hnu25::anti_drone::VisionTelemetryTransportStats transport;
    hnu25::anti_drone::VisionTelemetryStreamStats parser;
};

// Opens a serial transport on a fresh PTY slave, sends one packet per
// sequence number, then reads the master side and parses the bytes back.
RoundTripResult roundTrip(const std::vector<std::uint32_t>& sequences) {
    RoundTripResult result;

    Pty pty;
    if (!makePty(pty)) {
        return result;  // empty sequences signals the caller this failed
    }
    setNonBlocking(pty.master_fd);

    hnu25::anti_drone::VisionTelemetrySerialTransport transport(
        hnu25::anti_drone::VisionTelemetrySerialTransportConfig{
            pty.slave_path, 115200, false});
    if (!transport.open()) {
        ::close(pty.master_fd);
        return result;
    }

    for (const std::uint32_t sequence : sequences) {
        const auto packet =
            hnu25::anti_drone::encodeVisionTelemetry(makePacket(sequence));
        transport.send(packet);
    }

    const std::vector<std::uint8_t> bytes = readAll(pty.master_fd);
    hnu25::anti_drone::VisionTelemetryStreamParser parser;
    parser.push(bytes);

    hnu25::anti_drone::VisionTelemetry decoded;
    while (parser.pop(decoded)) {
        result.sequences.push_back(decoded.sequence);
    }

    result.transport = transport.stats();
    result.parser = parser.stats();
    transport.close();
    ::close(pty.master_fd);
    return result;
}

#endif  // __linux__

}  // namespace

int main() {
#ifdef __linux__
    // Sanity: the software PTY must be available (no hardware involved).
    {
        Pty pty;
        check(makePty(pty), "PTY (posix_openpt) available");
        if (pty.master_fd >= 0) {
            ::close(pty.master_fd);
        }
    }

    // ── A: single packet round-trip ──────────────────────────────────────
    {
        const auto r = roundTrip({42});
        check(r.sequences.size() == 1, "A: one packet decoded");
        check(!r.sequences.empty() && r.sequences[0] == 42,
              "A: sequence == 42");
        check(r.parser.crc_or_format_errors == 0,
              "A: crc_or_format_errors == 0");
        check(r.transport.packets_submitted == 1 &&
                  r.transport.packets_accepted == 1 &&
                  r.transport.bytes_accepted ==
                      hnu25::anti_drone::kVisionTelemetryPacketSize &&
                  r.transport.failures == 0,
              "A: transport stats correct");
    }

    // ── B: consecutive multi-packet round-trip ───────────────────────────
    {
        const auto r = roundTrip({7, 8, 9});
        check(r.sequences.size() == 3, "B: three packets decoded");
        check(r.sequences == std::vector<std::uint32_t>({7, 8, 9}),
              "B: order preserved (7, 8, 9)");
        check(r.parser.crc_or_format_errors == 0,
              "B: crc_or_format_errors == 0");
        check(r.transport.packets_submitted == 3 &&
                  r.transport.packets_accepted == 3 &&
                  r.transport.bytes_accepted ==
                      3 * hnu25::anti_drone::kVisionTelemetryPacketSize &&
                  r.transport.failures == 0,
              "B: transport stats correct");
    }

    // ── G: real PTY CRC + sequence correctness (100..104) ────────────────
    {
        const auto r = roundTrip({100, 101, 102, 103, 104});
        check(r.sequences.size() == 5, "G: five packets decoded");
        check(r.sequences ==
                  std::vector<std::uint32_t>({100, 101, 102, 103, 104}),
              "G: order preserved (100..104)");
        check(r.parser.crc_or_format_errors == 0,
              "G: crc_or_format_errors == 0");
        check(r.parser.sequence_gaps == 0, "G: sequence_gaps == 0");
        check(r.transport.packets_submitted == 5 &&
                  r.transport.packets_accepted == 5 &&
                  r.transport.bytes_accepted ==
                      5 * hnu25::anti_drone::kVisionTelemetryPacketSize &&
                  r.transport.failures == 0,
              "G: transport stats correct");
    }

    // ── C: wrong-size packet -> false + failure ──────────────────────────
    {
        hnu25::anti_drone::VisionTelemetrySerialTransport transport(
            hnu25::anti_drone::VisionTelemetrySerialTransportConfig{
                "", 115200, false});
        const std::vector<std::uint8_t> bad(49, 0);
        check(!transport.send(bad), "C: wrong-size send rejected");
        const auto& st = transport.stats();
        check(st.packets_submitted == 1 && st.failures == 1 &&
                  st.packets_accepted == 0,
              "C: wrong-size increments failure only");
    }

    // ── D: send before open -> false + failure ───────────────────────────
    {
        hnu25::anti_drone::VisionTelemetrySerialTransport transport(
            hnu25::anti_drone::VisionTelemetrySerialTransportConfig{
                "", 115200, false});
        const auto packet =
            hnu25::anti_drone::encodeVisionTelemetry(makePacket(1));
        check(!transport.send(packet), "D: send-before-open rejected");
        const auto& st = transport.stats();
        check(st.packets_submitted == 1 && st.failures == 1 &&
                  st.packets_accepted == 0,
              "D: send-before-open increments failure only");
    }

    // ── E: send after close -> false ─────────────────────────────────────
    {
        Pty pty;
        check(makePty(pty), "E: PTY created");
        hnu25::anti_drone::VisionTelemetrySerialTransport transport(
            hnu25::anti_drone::VisionTelemetrySerialTransportConfig{
                pty.slave_path, 115200, false});
        if (transport.open()) {
            transport.close();
            const auto packet =
                hnu25::anti_drone::encodeVisionTelemetry(makePacket(2));
            check(!transport.send(packet), "E: send-after-close rejected");
        } else {
            check(false, "E: transport opened before close");
        }
        if (pty.master_fd >= 0) {
            ::close(pty.master_fd);
        }
    }

    // ── F: open nonexistent path -> false, no crash ──────────────────────
    {
        hnu25::anti_drone::VisionTelemetrySerialTransport transport(
            hnu25::anti_drone::VisionTelemetrySerialTransportConfig{
                "/nonexistent/anti_drone_no_such_device", 115200, false});
        check(!transport.open(), "F: open nonexistent path returns false");
        check(!transport.isOpen(), "F: not open after failed open");
    }

    // ── H: finite write timeout configured -> normal writes still work ───
    {
        Pty pty;
        check(makePty(pty), "H: PTY created");
        setNonBlocking(pty.master_fd);

        hnu25::anti_drone::VisionTelemetrySerialTransportConfig cfg;
        cfg.device = pty.slave_path;
        cfg.baud_rate = 115200;
        cfg.flush_after_write = false;
        cfg.write_timeout_ms = 5;

        hnu25::anti_drone::VisionTelemetrySerialTransport transport(cfg);
        if (!transport.open()) {
            check(false, "H: transport opened with write_timeout_ms");
        } else {
            const auto packet =
                hnu25::anti_drone::encodeVisionTelemetry(makePacket(200));
            check(transport.send(packet),
                  "H: send succeeds with finite write timeout");

            const std::vector<std::uint8_t> bytes = readAll(pty.master_fd);
            hnu25::anti_drone::VisionTelemetryStreamParser parser;
            parser.push(bytes);
            hnu25::anti_drone::VisionTelemetry decoded;
            check(parser.pop(decoded) && decoded.sequence == 200,
                  "H: packet round-trips with finite write timeout");
            transport.close();
        }
        if (pty.master_fd >= 0) {
            ::close(pty.master_fd);
        }
    }

    if (g_failures == 0) {
        std::cout << "All serial transport integration checks passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
#else
    std::cout << "Serial transport integration test skipped (non-Linux).\n";
    return 0;
#endif
}
