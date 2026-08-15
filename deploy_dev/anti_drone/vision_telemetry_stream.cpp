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
    const std::uint32_t expected = last_sequence_ + 1;  // uint32 wraps.
    if (sequence != expected && sequence != last_sequence_) {
        // Forward gap or reorder: the wrapped difference counts the number of
        // missing sequence numbers (uint32 rollover is naturally continuous).
        stats_.sequence_gaps +=
            static_cast<std::uint32_t>(sequence - expected);
    }
    // A duplicate (sequence == last_sequence_) contributes no gap.
    last_sequence_ = sequence;
}

}  // namespace hnu25::anti_drone
