#pragma once

#include "camera/frame_source.hpp"
#include "camera/latest_frame_buffer.hpp"

#include <opencv2/videoio.hpp>

#include <atomic>
#include <thread>

namespace hnu25::camera {

struct OpenCvConfig {
    int camera_id = 0;
    int width = 1280;
    int height = 1024;
    double exposure = 0.0;
    double gain = 0.0;
    double frame_rate = 0.0;
};

class OpenCvFrameSource final : public FrameSource {
public:
    explicit OpenCvFrameSource(OpenCvConfig config);
    ~OpenCvFrameSource() override;

    void start() override;
    void stop() noexcept override;
    bool waitForFrame(Frame& frame, std::chrono::milliseconds timeout) override;

private:
    void captureLoop();

    OpenCvConfig config_;
    cv::VideoCapture capture_;
    LatestFrameBuffer frames_;
    std::atomic<bool> stop_requested_{false};
    std::thread capture_thread_;
    bool started_ = false;
};

}  // namespace hnu25::camera
