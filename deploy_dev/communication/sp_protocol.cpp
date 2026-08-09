#include "sp_protocol.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace hnu25::sp {
namespace {

constexpr double RAD_TO_DEG = 57.295779513082320876;

void appendI32(std::vector<uint8_t>& out, int32_t value) {
    const uint32_t bits = static_cast<uint32_t>(value);
    out.push_back(static_cast<uint8_t>(bits));
    out.push_back(static_cast<uint8_t>(bits >> 8));
    out.push_back(static_cast<uint8_t>(bits >> 16));
    out.push_back(static_cast<uint8_t>(bits >> 24));
}

int32_t readI32(const std::vector<uint8_t>& data, size_t offset) {
    const uint32_t bits = static_cast<uint32_t>(data[offset]) |
                          (static_cast<uint32_t>(data[offset + 1]) << 8) |
                          (static_cast<uint32_t>(data[offset + 2]) << 16) |
                          (static_cast<uint32_t>(data[offset + 3]) << 24);
    return static_cast<int32_t>(bits);
}

int32_t radiansToMilliDegrees(double radians) {
    const double value = radians * RAD_TO_DEG * 1000.0;
    return static_cast<int32_t>(std::clamp(
        value,
        static_cast<double>(std::numeric_limits<int32_t>::min()),
        static_cast<double>(std::numeric_limits<int32_t>::max())));
}

std::pair<uint8_t, uint8_t> checksums(const uint8_t* data, size_t size) {
    uint8_t sum = 0;
    uint8_t add = 0;
    for (size_t i = 0; i < size; ++i) {
        sum = static_cast<uint8_t>(sum + data[i]);
        add = static_cast<uint8_t>(add + sum);
    }
    return {sum, add};
}

}  // namespace

std::vector<uint8_t> encodeFrame(const Frame& frame) {
    if (frame.data.size() > std::numeric_limits<uint8_t>::max()) return {};
    std::vector<uint8_t> bytes;
    bytes.reserve(frame.data.size() + 6);
    bytes.push_back(HEAD);
    bytes.push_back(frame.d_addr);
    bytes.push_back(frame.id);
    bytes.push_back(static_cast<uint8_t>(frame.data.size()));
    bytes.insert(bytes.end(), frame.data.begin(), frame.data.end());
    const auto [sum, add] = checksums(bytes.data(), bytes.size());
    bytes.push_back(sum);
    bytes.push_back(add);
    return bytes;
}

std::vector<uint8_t> encodeGimbalCommand(const GimbalCommand& command) {
    if (!std::isfinite(command.yaw_rad) || !std::isfinite(command.pitch_rad)) return {};
    Frame frame;
    frame.d_addr = D_ADDR_CBOARD;
    frame.id = ID_GIMBAL;
    frame.data.reserve(GIMBAL_DATA_SIZE);
    frame.data.push_back(command.mode);
    appendI32(frame.data, radiansToMilliDegrees(command.yaw_rad));
    appendI32(frame.data, radiansToMilliDegrees(command.pitch_rad));
    appendI32(frame.data, command.fire ? 1000 : 0);       // Roll is the fire channel.
    appendI32(frame.data, command.control ? 1000 : 0);    // YawSpd is the takeover channel.
    appendI32(frame.data, 0);
    appendI32(frame.data, 0);
    return encodeFrame(frame);
}

bool decodeGimbalFeedback(const Frame& frame, GimbalFeedback& feedback,
                          std::optional<uint8_t> expected_address) {
    if (frame.id != ID_GIMBAL || frame.data.size() != GIMBAL_DATA_SIZE ||
        (expected_address && frame.d_addr != *expected_address)) return false;
    feedback.mode = frame.data[0];
    feedback.yaw_deg = readI32(frame.data, 1) / 1000.0;
    feedback.pitch_deg = readI32(frame.data, 5) / 1000.0;
    feedback.roll_deg = readI32(frame.data, 9) / 1000.0;
    feedback.yaw_speed = readI32(frame.data, 13) / 1000.0;
    feedback.pitch_speed = readI32(frame.data, 17) / 1000.0;
    feedback.roll_speed = readI32(frame.data, 21) / 1000.0;
    return true;
}

bool feedbackFresh(const GimbalFeedback& feedback, std::chrono::milliseconds max_age,
                   std::chrono::steady_clock::time_point now) {
    if (max_age < std::chrono::milliseconds::zero() || feedback.received_at.time_since_epoch().count() == 0)
        return false;
    const auto age = now - feedback.received_at;
    return age >= std::chrono::steady_clock::duration::zero() && age <= max_age;
}

EchoRejector::EchoRejector(bool enabled, std::chrono::milliseconds window)
    : enabled_(enabled), window_(window) {
    if (window < std::chrono::milliseconds::zero())
        throw std::invalid_argument("echo rejection window must not be negative");
}

void EchoRejector::recordSent(const std::vector<uint8_t>& bytes,
                              std::chrono::steady_clock::time_point sent_at) {
    std::lock_guard<std::mutex> lock(mutex_);
    last_sent_ = bytes;
    last_sent_at_ = sent_at;
}

bool EchoRejector::reject(const Frame& frame,
                          std::chrono::steady_clock::time_point received_at) const {
    if (!enabled_) return false;
    const auto bytes = encodeFrame(frame);
    std::lock_guard<std::mutex> lock(mutex_);
    const auto age = received_at - last_sent_at_;
    return !last_sent_.empty() && age >= std::chrono::steady_clock::duration::zero() &&
           age <= window_ && bytes == last_sent_;
}

std::vector<Frame> StreamParser::push(const uint8_t* data, size_t size) {
    if (data != nullptr && size != 0) buffer_.insert(buffer_.end(), data, data + size);
    std::vector<Frame> frames;
    while (true) {
        const auto head = std::find(buffer_.begin(), buffer_.end(), HEAD);
        buffer_.erase(buffer_.begin(), head);
        if (buffer_.size() < 4) break;
        const size_t data_size = buffer_[3];
        if (data_size > MAX_DATA_SIZE) {
            buffer_.erase(buffer_.begin());
            continue;
        }
        const size_t frame_size = data_size + 6;
        if (buffer_.size() < frame_size) break;
        const auto [sum, add] = checksums(buffer_.data(), frame_size - 2);
        if (buffer_[frame_size - 2] != sum || buffer_[frame_size - 1] != add) {
            buffer_.erase(buffer_.begin());
            continue;
        }
        Frame frame;
        frame.d_addr = buffer_[1];
        frame.id = buffer_[2];
        frame.data.assign(buffer_.begin() + 4, buffer_.begin() + 4 + data_size);
        frames.push_back(std::move(frame));
        buffer_.erase(buffer_.begin(), buffer_.begin() + frame_size);
    }
    return frames;
}

void StreamParser::reset() {
    buffer_.clear();
}

SendLimiter::SendLimiter(std::chrono::milliseconds interval) : interval_(interval) {}

bool SendLimiter::allow(std::chrono::steady_clock::time_point now) {
    if (sent_ && now - last_send_ < interval_) return false;
    last_send_ = now;
    sent_ = true;
    return true;
}

}  // namespace hnu25::sp
