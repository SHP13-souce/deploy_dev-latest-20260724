#pragma once

#include <memory>
#include <string>

#include <opencv2/core/mat.hpp>

namespace debug_web {

struct Config {
    bool enabled = false;
    std::string shm_name = "/hnu_vision_debug";
    int quality = 80;
    double publish_hz = 15.0;
};

class DebugWebPublisher {
public:
    explicit DebugWebPublisher(Config config) noexcept;
    ~DebugWebPublisher();
    DebugWebPublisher(DebugWebPublisher&&) noexcept;
    DebugWebPublisher& operator=(DebugWebPublisher&&) noexcept;
    DebugWebPublisher(const DebugWebPublisher&) = delete;
    DebugWebPublisher& operator=(const DebugWebPublisher&) = delete;

    void publish(const cv::Mat& frame, const std::string& state_json) noexcept;
    bool ready() const noexcept;
    bool due() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::string statePathForShm(const std::string& shm_name);

}  // namespace debug_web
