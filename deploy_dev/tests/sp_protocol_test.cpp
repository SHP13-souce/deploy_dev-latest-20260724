#include "communication/sp_protocol.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

using namespace hnu25::sp;

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}
void requireNear(double actual, double expected, const char* message) {
    require(std::abs(actual - expected) < 1e-9, message);
}
}  // namespace

int main() {
    GimbalCommand command;
    command.control = true;
    command.fire = true;
    command.yaw_rad = 3.14159265358979323846 / 2.0;
    command.pitch_rad = -3.14159265358979323846 / 6.0;
    const std::vector<uint8_t> golden = {
        0xFF, 0x02, 0x20, 0x19, 0x01,
        0x90, 0x5F, 0x01, 0x00, 0xD1, 0x8A, 0xFF, 0xFF,
        0xE8, 0x03, 0x00, 0x00, 0xE8, 0x03, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x5A, 0x5A,
    };
    const auto encoded = encodeGimbalCommand(command);
    require(encoded == golden, "gimbal command golden bytes");

    command.yaw_rad = std::numeric_limits<double>::quiet_NaN();
    require(encodeGimbalCommand(command).empty(), "NaN yaw is rejected");
    command.yaw_rad = 0.0;
    command.pitch_rad = std::numeric_limits<double>::infinity();
    require(encodeGimbalCommand(command).empty(), "infinite pitch is rejected");

    StreamParser parser;
    const std::vector<uint8_t> prefix = {0x12, 0x34, 0xFF};
    require(parser.push(prefix).empty(), "noise and partial header produce no frame");
    require(parser.push(encoded.data() + 1, 7).empty(), "partial frame produces no frame");
    std::vector<uint8_t> rest(encoded.begin() + 8, encoded.end());
    rest.insert(rest.end(), encoded.begin(), encoded.end());
    const auto frames = parser.push(rest);
    require(frames.size() == 2, "sticky input yields two frames");
    require(frames[0].d_addr == D_ADDR_CBOARD && frames[0].id == ID_GIMBAL,
            "decoded header fields");

    GimbalFeedback gimbal;
    require(decodeGimbalFeedback(frames[0], gimbal), "decode gimbal feedback");
    require(gimbal.mode == 1, "decode control mode");
    requireNear(gimbal.yaw_deg, 90.0, "decode yaw");
    requireNear(gimbal.pitch_deg, -29.999, "decode pitch");
    requireNear(gimbal.roll_deg, 1.0, "decode Roll/fire field");
    requireNear(gimbal.yaw_speed, 1.0, "decode YawSpd/takeover field");
    require(!decodeGimbalFeedback(frames[0], gimbal, static_cast<uint8_t>(0x03)),
            "unexpected feedback address rejected");
    require(decodeGimbalFeedback(frames[0], gimbal, D_ADDR_CBOARD),
            "expected feedback address accepted");

    const auto received_at = std::chrono::steady_clock::time_point{} + std::chrono::seconds(1);
    gimbal.received_at = received_at;
    require(feedbackFresh(gimbal, std::chrono::milliseconds(50), received_at + std::chrono::milliseconds(50)),
            "feedback fresh at age limit");
    require(!feedbackFresh(gimbal, std::chrono::milliseconds(50), received_at + std::chrono::milliseconds(51)),
            "feedback expires after max age");

    EchoRejector echo_rejector(true, std::chrono::milliseconds(20));
    echo_rejector.recordSent(encoded, received_at);
    require(echo_rejector.reject(frames[0], received_at + std::chrono::milliseconds(5)),
            "recent identical sent frame rejected as echo");
    require(!echo_rejector.reject(frames[0], received_at + std::chrono::milliseconds(21)),
            "identical frame accepted outside echo window");
    EchoRejector echo_disabled(false, std::chrono::milliseconds(20));
    echo_disabled.recordSent(encoded, received_at);
    require(!echo_disabled.reject(frames[0], received_at), "echo rejection can be disabled");

    auto bad_sum = encoded;
    bad_sum[bad_sum.size() - 2] ^= 0x01;
    auto bad_add = encoded;
    bad_add.back() ^= 0x01;
    std::vector<uint8_t> recovery = bad_sum;
    recovery.insert(recovery.end(), bad_add.begin(), bad_add.end());
    recovery.insert(recovery.end(), encoded.begin(), encoded.end());
    parser.reset();
    require(parser.push(recovery).size() == 1, "both checksums enforced and stream recovers");

    SendLimiter limiter;
    const auto t0 = std::chrono::steady_clock::time_point{};
    require(limiter.allow(t0), "first send allowed");
    require(!limiter.allow(t0 + std::chrono::milliseconds(9)), "send blocked before 10ms");
    require(limiter.allow(t0 + std::chrono::milliseconds(10)), "send allowed at 10ms");
    return 0;
}
