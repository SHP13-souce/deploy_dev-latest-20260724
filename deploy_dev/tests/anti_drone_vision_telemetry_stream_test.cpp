#include "anti_drone/vision_telemetry.hpp"
#include "anti_drone/vision_telemetry_loopback.hpp"
#include "anti_drone/vision_telemetry_stream.hpp"

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

using hnu25::anti_drone::VisionTelemetry;
using hnu25::anti_drone::VisionTelemetryLoopbackTransport;
using hnu25::anti_drone::VisionTelemetryStreamParser;

// Builds a valid 50-byte packet carrying `sequence` (all other fields default:
// vision_valid false, track LOST, zeros).
std::vector<std::uint8_t> makePacket(std::uint32_t sequence) {
    VisionTelemetry t;
    t.sequence = sequence;
    return hnu25::anti_drone::encodeVisionTelemetry(t);
}

// Builds a 50-byte frame with a correct magic/version/payload-length header but
// an all-zero payload and CRC, guaranteed to fail decode and to contain no
// stray 0x41 0x44 marker.
std::vector<std::uint8_t> corruptPacket() {
    std::vector<std::uint8_t> p(
        hnu25::anti_drone::kVisionTelemetryPacketSize, 0x00);
    p[0] = hnu25::anti_drone::kVisionTelemetryMagic0;
    p[1] = hnu25::anti_drone::kVisionTelemetryMagic1;
    p[2] = hnu25::anti_drone::kVisionTelemetryVersion;
    p[3] = 44;
    return p;
}

}  // namespace

int main() {
    // ── Test 1: exact one packet ──────────────────────────────────────────
    {
        VisionTelemetryStreamParser parser;
        parser.push(makePacket(0));
        VisionTelemetry t;
        const bool ok = parser.pop(t);
        check(ok, "exact one packet pops");
        check(ok && t.sequence == 0, "exact one packet sequence correct");
        check(!parser.pop(t), "no second packet");
        check(parser.stats().packets_received == 1, "packets_received == 1");
        check(parser.stats().packets_valid == 1, "packets_valid == 1");
    }

    // ── Test 2: packet split into multiple pushes ─────────────────────────
    {
        VisionTelemetryStreamParser parser;
        const auto packet = makePacket(1);
        parser.push(packet.data(), 10);
        VisionTelemetry t;
        check(!parser.pop(t), "no packet after 10 bytes");
        parser.push(packet.data() + 10, 15);
        check(!parser.pop(t), "no packet after 25 bytes");
        parser.push(packet.data() + 25, 25);
        const bool ok = parser.pop(t);
        check(ok, "packet after full 50 bytes");
        check(ok && t.sequence == 1, "split packet sequence correct");
    }

    // ── Test 3: three packets in one push ─────────────────────────────────
    {
        VisionTelemetryStreamParser parser;
        std::vector<std::uint8_t> data;
        const auto p0 = makePacket(10);
        const auto p1 = makePacket(11);
        const auto p2 = makePacket(12);
        data.insert(data.end(), p0.begin(), p0.end());
        data.insert(data.end(), p1.begin(), p1.end());
        data.insert(data.end(), p2.begin(), p2.end());
        parser.push(data);
        VisionTelemetry a, b, c;
        const bool ok = parser.pop(a) && parser.pop(b) && parser.pop(c);
        check(ok, "three packets popped");
        check(ok && a.sequence == 10 && b.sequence == 11 && c.sequence == 12,
              "three packets sequence order correct");
        check(!parser.pop(a), "no fourth packet");
    }

    // ── Test 4: leading garbage then valid packet ─────────────────────────
    {
        VisionTelemetryStreamParser parser;
        std::vector<std::uint8_t> data = {0x00, 0xFF, 0x12, 0x00, 0x00};
        const auto good = makePacket(3);
        data.insert(data.end(), good.begin(), good.end());
        parser.push(data);
        VisionTelemetry t;
        const bool ok = parser.pop(t);
        check(ok, "leading garbage recovered");
        check(ok && t.sequence == 3, "leading garbage sequence correct");
        check(parser.stats().discarded_bytes == 5,
              "garbage bytes discarded");
    }

    // ── Test 5: corrupted packet then valid packet recovery ───────────────
    {
        VisionTelemetryStreamParser parser;
        std::vector<std::uint8_t> data;
        const auto bad = corruptPacket();
        const auto good = makePacket(8);
        data.insert(data.end(), bad.begin(), bad.end());
        data.insert(data.end(), good.begin(), good.end());
        parser.push(data);
        VisionTelemetry t;
        const bool ok = parser.pop(t);
        check(ok, "corrupted-then-valid recovered a packet");
        check(ok && t.sequence == 8, "recovered packet sequence correct");
        check(parser.stats().crc_or_format_errors == 1,
              "one crc/format error recorded");
    }

    // ── Test 6: sequence gap detection ────────────────────────────────────
    {
        VisionTelemetryStreamParser parser;
        std::vector<std::uint8_t> data;
        for (const std::uint32_t s : {100u, 101u, 102u, 105u}) {
            const auto p = makePacket(s);
            data.insert(data.end(), p.begin(), p.end());
        }
        parser.push(data);
        VisionTelemetry t;
        int popped = 0;
        while (parser.pop(t)) {
            ++popped;
        }
        check(popped == 4, "four packets popped");
        check(parser.stats().sequence_gaps == 2, "sequence_gaps == 2");
    }

    // ── Test 7: uint32 sequence rollover ──────────────────────────────────
    {
        VisionTelemetryStreamParser parser;
        std::vector<std::uint8_t> data;
        for (const std::uint32_t s :
             {0xFFFFFFFEu, 0xFFFFFFFFu, 0x00000000u, 0x00000001u}) {
            const auto p = makePacket(s);
            data.insert(data.end(), p.begin(), p.end());
        }
        parser.push(data);
        VisionTelemetry t;
        int popped = 0;
        while (parser.pop(t)) {
            ++popped;
        }
        check(popped == 4, "rollover: four packets popped");
        check(parser.stats().sequence_gaps == 0, "rollover: no gaps");
    }

    // ── Test 8: duplicate sequence no huge gap ────────────────────────────
    {
        VisionTelemetryStreamParser parser;
        const auto p = makePacket(100);
        parser.push(p);
        parser.push(p);  // duplicate
        VisionTelemetry t;
        check(parser.pop(t), "first duplicate pop");
        check(parser.pop(t), "second duplicate pop");
        check(parser.stats().sequence_gaps == 0,
              "duplicate sequence produces no gap");
    }

    // ── Test 8a: out-of-order — forward gap then stale packet ─────────────
    {
        VisionTelemetryStreamParser parser;
        std::vector<std::uint8_t> data;
        for (const std::uint32_t s : {100u, 101u, 105u, 103u}) {
            const auto p = makePacket(s);
            data.insert(data.end(), p.begin(), p.end());
        }
        parser.push(data);
        VisionTelemetry t;
        std::vector<std::uint32_t> got;
        while (parser.pop(t)) {
            got.push_back(t.sequence);
        }
        check(got.size() == 4, "out-of-order: four packets popped");
        check(got.size() == 4 && got[0] == 100 && got[1] == 101 &&
                  got[2] == 105 && got[3] == 103,
              "out-of-order: pop order preserved");
        check(parser.stats().sequence_gaps == 3,
              "out-of-order: gaps == 3 (stale 103 ignored)");
    }

    // ── Test 8b: out-of-order — stale packet then consecutive ─────────────
    {
        VisionTelemetryStreamParser parser;
        std::vector<std::uint8_t> data;
        for (const std::uint32_t s : {100u, 99u, 101u}) {
            const auto p = makePacket(s);
            data.insert(data.end(), p.begin(), p.end());
        }
        parser.push(data);
        VisionTelemetry t;
        std::vector<std::uint32_t> got;
        while (parser.pop(t)) {
            got.push_back(t.sequence);
        }
        check(got.size() == 3, "out-of-order: three packets popped");
        check(parser.stats().sequence_gaps == 0,
              "out-of-order: gaps == 0 (stale 99 ignored, 101 consecutive)");
    }

    // ── Test 9: reset clears parser state/stats ───────────────────────────
    {
        VisionTelemetryStreamParser parser;
        parser.push(makePacket(1));
        VisionTelemetry t;
        parser.pop(t);
        parser.reset();
        const auto& s = parser.stats();
        check(s.bytes_received == 0 && s.packets_received == 0 &&
                  s.packets_valid == 0 && s.crc_or_format_errors == 0 &&
                  s.discarded_bytes == 0 && s.sequence_gaps == 0,
              "reset clears stats");
        parser.push(makePacket(2));
        check(parser.pop(t) && t.sequence == 2,
              "reset clears sequence tracking (no false gap)");
        check(parser.stats().sequence_gaps == 0, "no gap after reset");
    }

    // ── Test 10: loopback transport send/pop ──────────────────────────────
    {
        VisionTelemetryLoopbackTransport transport;
        const bool sent = transport.send(makePacket(5));
        check(sent, "loopback send accepted");
        check(transport.stats().packets_submitted == 1, "submitted == 1");
        check(transport.stats().packets_accepted == 1, "accepted == 1");
        check(transport.stats().bytes_accepted == 50, "bytes_accepted == 50");
        check(transport.stats().failures == 0, "no failures");
        VisionTelemetry received;
        const bool popped = transport.popReceived(received);
        check(popped, "loopback popReceived returns");
        check(popped && received.sequence == 5, "loopback received sequence");
    }

    // ── Test 11: wrong packet size rejected by loopback ───────────────────
    {
        VisionTelemetryLoopbackTransport transport;
        std::vector<std::uint8_t> short_packet(10, 0x00);
        const bool ok = transport.send(short_packet);
        check(!ok, "wrong-size packet rejected");
        check(transport.stats().packets_submitted == 1, "submitted == 1");
        check(transport.stats().packets_accepted == 0, "accepted == 0");
        check(transport.stats().failures == 1, "failures == 1");
    }

    // ── Test 12: multiple loopback packets preserve order ─────────────────
    {
        VisionTelemetryLoopbackTransport transport;
        transport.send(makePacket(10));
        transport.send(makePacket(11));
        transport.send(makePacket(12));
        VisionTelemetry a, b, c;
        const bool ok = transport.popReceived(a) &&
                        transport.popReceived(b) &&
                        transport.popReceived(c);
        check(ok, "three loopback packets popped");
        check(ok && a.sequence == 10 && b.sequence == 11 &&
                  c.sequence == 12,
              "loopback packet order preserved");
        check(!transport.popReceived(a), "no extra loopback packets");
    }

    if (g_failures == 0) {
        std::cout << "All vision telemetry stream tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
