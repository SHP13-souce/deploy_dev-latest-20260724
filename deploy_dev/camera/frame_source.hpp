#pragma once

#include "camera/frame.hpp"

#include <chrono>

namespace hnu25::camera {

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual void start() = 0;
    virtual void stop() noexcept = 0;
    virtual bool waitForFrame(Frame& frame, std::chrono::milliseconds timeout) = 0;
};

}  // namespace hnu25::camera
