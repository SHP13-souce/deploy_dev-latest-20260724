#pragma once

#include "anti_drone/vision_telemetry.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hnu25::anti_drone {

// Counters accumulated by VisionTelemetryStreamParser.
struct VisionTelemetryStreamStats {
    std::uint64_t bytes_received = 0;        // total bytes pushed in
    std::uint64_t packets_received = 0;      // complete fixed-size candidates seen
    std::uint64_t packets_valid = 0;         // candidates that decoded cleanly
    std::uint64_t crc_or_format_errors = 0;  // candidates that failed to decode
    std::uint64_t discarded_bytes = 0;       // bytes dropped while resyncing
    std::uint64_t sequence_gaps = 0;         // missing sequence numbers
};

// Continuous byte-stream parser for the fixed-size VisionTelemetry v1 wire
// format. It accepts an arbitrary stream of bytes (as a real serial port would
// deliver them: split, coalesced, or with garbage interleaved), resyncs on the
// 0x41 0x44 magic marker, and pops complete, decoded packets one at a time.
//
// A single corrupt packet does not poison the stream: only the minimum
// necessary bytes are discarded before searching for the next marker, so the
// stream recovers after occasional corruption.
class VisionTelemetryStreamParser {
public:
    VisionTelemetryStreamParser() = default;

    // Appends raw bytes to the internal buffer.
    void push(const std::uint8_t* data, std::size_t size);
    void push(const std::vector<std::uint8_t>& data);

    // Attempts to extract and decode the next complete packet. Returns true
    // and fills `telemetry` on success. Returns false when no complete valid
    // packet is available yet (partial input is retained; leading garbage or
    // a trailing 0x41 is kept as needed for the next push).
    bool pop(VisionTelemetry& telemetry);

    const VisionTelemetryStreamStats& stats() const noexcept;

    // Clears the buffer, counters, and sequence tracking.
    void reset();

private:
    // Offset of the first complete 0x41 0x44 marker, or npos if absent.
    std::size_t findSync() const noexcept;

    // Discards the first `count` bytes as resync garbage.
    void discardPrefix(std::size_t count) noexcept;

    // Updates last_sequence_ / sequence_gaps for a successfully decoded
    // packet.
    void recordSequence(std::uint32_t sequence) noexcept;

    std::vector<std::uint8_t> buffer_;
    VisionTelemetryStreamStats stats_;
    bool have_last_sequence_ = false;
    std::uint32_t last_sequence_ = 0;
};

}  // namespace hnu25::anti_drone
