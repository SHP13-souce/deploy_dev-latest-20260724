#include "anti_drone/vision_telemetry_stream.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace hnu25::anti_drone {
namespace {

constexpr std::size_t kNoSync = static_cast<std::size_t>(-1);

}  // namespace

void VisionTelemetryStreamParser::push(const std::uint8_t* data,
                                       std::size_t size) {
    if (data == nullptr || size == 0) {
        return;
    }
    buffer_.insert(buffer_.end(), data, data + size);
    stats_.bytes_received += size;
}

void VisionTelemetryStreamParser::push(
    const std::vector<std::uint8_t>& data) {
    push(data.data(), data.size());
}

bool VisionTelemetryStreamParser::pop(VisionTelemetry& telemetry) {
    while (true) {
        const std::size_t sync = findSync();
        if (sync == kNoSync) {
            // No complete marker yet. Keep at most one trailing 0x41: it may
            // be the first half of a marker split across pushes.
            std::size_t keep = 0;
            if (!buffer_.empty() &&
                buffer_.back() == kVisionTelemetryMagic0) {
                keep = 1;
            }
            const std::size_t drop = buffer_.size() - keep;
            if (drop > 0) {
                stats_.discarded_bytes += drop;
                buffer_.erase(buffer_.begin(), buffer_.begin() + drop);
            }
            return false;
        }

        discardPrefix(sync);

        if (buffer_.size() < kVisionTelemetryPacketSize) {
            // Partial packet: wait for more bytes.
            return false;
        }

        std::vector<std::uint8_t> candidate(
            buffer_.begin(), buffer_.begin() + kVisionTelemetryPacketSize);
        ++stats_.packets_received;

        if (decodeVisionTelemetry(candidate, telemetry)) {
            ++stats_.packets_valid;
            buffer_.erase(buffer_.begin(),
                          buffer_.begin() + kVisionTelemetryPacketSize);
            recordSequence(telemetry.sequence);
            return true;
        }

        // Corrupt candidate: drop only the leading magic byte and keep
        // searching for the next marker (minimal discard).
        ++stats_.crc_or_format_errors;
        discardPrefix(1);
    }
}

const VisionTelemetryStreamStats&
VisionTelemetryStreamParser::stats() const noexcept {
    return stats_;
}

void VisionTelemetryStreamParser::reset() {
    buffer_.clear();
    stats_ = VisionTelemetryStreamStats{};
    have_last_sequence_ = false;
    last_sequence_ = 0;
}

std::size_t VisionTelemetryStreamParser::findSync() const noexcept {
    for (std::size_t i = 0; i + 1 < buffer_.size(); ++i) {
        if (buffer_[i] == kVisionTelemetryMagic0 &&
            buffer_[i + 1] == kVisionTelemetryMagic1) {
            return i;
        }
    }
    return kNoSync;
}

void VisionTelemetryStreamParser::discardPrefix(
    std::size_t count) noexcept {
    if (count == 0) {
        return;
    }
    stats_.discarded_bytes += count;
    buffer_.erase(buffer_.begin(), buffer_.begin() + count);
}

void VisionTelemetryStreamParser::recordSequence(
    std::uint32_t sequence) noexcept {
    if (!have_last_sequence_) {
        last_sequence_ = sequence;
        have_last_sequence_ = true;
        return;
    }

    // Forward delta modulo 2^32 (uint32 subtraction wraps), so rollover
    // (0xFFFFFFFF -> 0) yields delta == 1 and reads as consecutive.
    const std::uint32_t delta = sequence - last_sequence_;

    if (delta == 0) {
        // Duplicate: no gap, baseline unchanged.
        return;
    }
    if (delta == 1) {
        // Consecutive: no gap, advance baseline.
        last_sequence_ = sequence;
        return;
    }
    if (delta < 0x80000000u) {
        // Forward advance with a gap: delta - 1 missing sequence numbers.
        stats_.sequence_gaps += delta - 1;
        last_sequence_ = sequence;
        return;
    }
    // delta >= 0x80000000: an old / out-of-order packet. Do not count a
    // (huge wrapped) gap and do not move the baseline backward.
}

}  // namespace hnu25::anti_drone
