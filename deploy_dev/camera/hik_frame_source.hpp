#pragma once

#include "camera/frame_source.hpp"
#include "camera/latest_frame_buffer.hpp"

#include <atomic>
#include <string>
#include <thread>

namespace hnu25::camera {

struct HikConfig {
    std::string serial_number;
    double exposure = 3000.0;
    double gain = 0.0;
    double frame_rate = 100.0;
};

class HikFrameSource final : public FrameSource {
public:
    explicit HikFrameSource(HikConfig config);
    ~HikFrameSource() override;

    void start() override;
    void stop() noexcept override;
    bool waitForFrame(Frame& frame, std::chrono::milliseconds timeout) override;

private:
    enum class State { Empty, HandleCreated, DeviceOpen, Grabbing };

    void configure();
    void captureLoop();
    void cleanup() noexcept;
    void setEnum(const char* name, unsigned int value);
    void setFloatAndReadBack(const char* name, double value);

    HikConfig config_;
    void* handle_ = nullptr;
    State state_ = State::Empty;
    LatestFrameBuffer frames_;
    std::atomic<bool> stop_requested_{false};
    std::thread capture_thread_;
};

}  // namespace hnu25::camera
