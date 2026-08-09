#pragma once

#include "sp_protocol.hpp"

#include <cstdint>
#include <vector>

namespace hnu25::bcp {

constexpr uint8_t HEAD = sp::HEAD;
constexpr uint8_t ADDR_GIMBAL = sp::D_ADDR_CBOARD;
constexpr uint8_t ID_GIMBAL = sp::ID_GIMBAL;
constexpr uint8_t FRAME_HEADER_SIZE = 4;
constexpr uint8_t FRAME_TAIL_SIZE = 2;
constexpr uint8_t GIMBAL_DATA_SIZE = static_cast<uint8_t>(sp::GIMBAL_DATA_SIZE);
constexpr uint8_t GIMBAL_FRAME_SIZE = FRAME_HEADER_SIZE + GIMBAL_DATA_SIZE + FRAME_TAIL_SIZE;

struct GimbalPayload {
    uint8_t mode = 1;
    int32_t yaw = 0;
    int32_t pitch = 0;
    int32_t roll = 0;
};

inline int32_t radToInt1000(double rad) {
    return static_cast<int32_t>(rad * 57.295779513082320876 * 1000.0);
}

inline std::vector<uint8_t> buildGimbalFrame(const GimbalPayload& payload) {
    sp::Frame frame;
    frame.d_addr = ADDR_GIMBAL;
    frame.id = ID_GIMBAL;
    frame.data.reserve(GIMBAL_DATA_SIZE);
    frame.data.push_back(payload.mode);
    const auto append = [&frame](int32_t value) {
        const uint32_t bits = static_cast<uint32_t>(value);
        frame.data.push_back(static_cast<uint8_t>(bits));
        frame.data.push_back(static_cast<uint8_t>(bits >> 8));
        frame.data.push_back(static_cast<uint8_t>(bits >> 16));
        frame.data.push_back(static_cast<uint8_t>(bits >> 24));
    };
    append(payload.yaw);
    append(payload.pitch);
    append(payload.roll);
    append(0);
    append(0);
    append(0);
    return sp::encodeFrame(frame);
}

}  // namespace hnu25::bcp
