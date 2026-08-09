#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace hnu25::sp {

constexpr uint8_t HEAD = 0xFF;
constexpr uint8_t D_ADDR_CBOARD = 0x02;
constexpr uint8_t ID_GIMBAL = 0x20;
constexpr size_t GIMBAL_DATA_SIZE = 25;
constexpr size_t MAX_DATA_SIZE = 100;
constexpr const char* DEFAULT_DEVICE = "/dev/rm_cboard";

struct Frame {
    uint8_t d_addr = 0;
    uint8_t id = 0;
    std::vector<uint8_t> data;
};

struct GimbalCommand {
    bool control = false;
    bool fire = false;
    double yaw_rad = 0.0;
    double pitch_rad = 0.0;
    uint8_t mode = 1;
};

struct GimbalFeedback {
    uint8_t mode = 0;
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
    double roll_deg = 0.0;
    double yaw_speed = 0.0;
    double pitch_speed = 0.0;
    double roll_speed = 0.0;
    std::chrono::steady_clock::time_point received_at{};
};

std::vector<uint8_t> encodeFrame(const Frame& frame);
std::vector<uint8_t> encodeGimbalCommand(const GimbalCommand& command);
bool decodeGimbalFeedback(const Frame& frame, GimbalFeedback& feedback,
                          std::optional<uint8_t> expected_address = std::nullopt);
bool feedbackFresh(const GimbalFeedback& feedback, std::chrono::milliseconds max_age,
                   std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

class EchoRejector {
public:
    explicit EchoRejector(bool enabled = true,
                          std::chrono::milliseconds window = std::chrono::milliseconds(20));
    void recordSent(const std::vector<uint8_t>& bytes,
                    std::chrono::steady_clock::time_point sent_at = std::chrono::steady_clock::now());
    bool reject(const Frame& frame,
                std::chrono::steady_clock::time_point received_at = std::chrono::steady_clock::now()) const;

private:
    bool enabled_;
    std::chrono::steady_clock::duration window_;
    mutable std::mutex mutex_;
    std::vector<uint8_t> last_sent_;
    std::chrono::steady_clock::time_point last_sent_at_{};
};

class StreamParser {
public:
    std::vector<Frame> push(const uint8_t* data, size_t size);
    std::vector<Frame> push(const std::vector<uint8_t>& data) {
        return push(data.data(), data.size());
    }
    void reset();

private:
    std::vector<uint8_t> buffer_;
};

class SendLimiter {
public:
    explicit SendLimiter(std::chrono::milliseconds interval = std::chrono::milliseconds(10));
    bool allow(std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

private:
    std::chrono::steady_clock::duration interval_;
    std::chrono::steady_clock::time_point last_send_;
    bool sent_ = false;
};

}  // namespace hnu25::sp
