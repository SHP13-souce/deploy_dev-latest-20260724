#pragma once

#include "camera/frame.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <utility>

namespace hnu25::camera {

class LatestFrameBuffer {
public:
    void push(Frame frame) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopped_) return;
            latest_ = std::move(frame);
        }
        ready_.notify_one();
    }

    bool waitPop(Frame& frame, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!ready_.wait_for(lock, timeout, [this] { return latest_.has_value() || stopped_; })) return false;
        if (!latest_) return false;
        frame = std::move(*latest_);
        latest_.reset();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
            latest_.reset();
        }
        ready_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_;
    std::optional<Frame> latest_;
    bool stopped_ = false;
};

}  // namespace hnu25::camera
