#include "camera/opencv_frame_source.hpp"

#include <opencv2/imgproc.hpp>

#include <iostream>
#include <stdexcept>
#include <utility>

namespace hnu25::camera {

OpenCvFrameSource::OpenCvFrameSource(OpenCvConfig config) : config_(std::move(config)) {}

OpenCvFrameSource::~OpenCvFrameSource() {
    stop();
}

void OpenCvFrameSource::start() {
    if (started_) throw std::logic_error("OpenCvFrameSource already started");
    if (!capture_.open(config_.camera_id))
        throw std::runtime_error("cannot open camera " + std::to_string(config_.camera_id));
    if (!capture_.set(cv::CAP_PROP_FRAME_WIDTH, config_.width) ||
        !capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config_.height)) {
        capture_.release();
        throw std::runtime_error("camera rejected requested width/height");
    }
    if (config_.frame_rate > 0.0 && !capture_.set(cv::CAP_PROP_FPS, config_.frame_rate))
        std::cerr << "[Camera] OpenCV backend rejected frame_rate\n";
    if (config_.gain > 0.0 && !capture_.set(cv::CAP_PROP_GAIN, config_.gain))
        std::cerr << "[Camera] OpenCV backend rejected gain\n";
    // CAP_PROP_EXPOSURE has backend-specific units, so the generic source must not reinterpret microseconds.
    if (config_.exposure > 0.0)
        std::cerr << "[Camera] OpenCV exposure is not set because CAP_PROP_EXPOSURE units are backend-specific\n";

    stop_requested_.store(false, std::memory_order_relaxed);
    started_ = true;
    capture_thread_ = std::thread(&OpenCvFrameSource::captureLoop, this);
}

void OpenCvFrameSource::stop() noexcept {
    if (!started_) return;
    stop_requested_.store(true, std::memory_order_relaxed);
    frames_.stop();
    if (capture_thread_.joinable()) capture_thread_.join();
    capture_.release();
    started_ = false;
}

bool OpenCvFrameSource::waitForFrame(Frame& frame, std::chrono::milliseconds timeout) {
    return frames_.waitPop(frame, timeout);
}

void OpenCvFrameSource::captureLoop() {
    std::uint64_t frame_number = 0;
    while (!stop_requested_.load(std::memory_order_relaxed)) {
        cv::Mat input;
        if (!capture_.read(input) || input.empty()) {
            if (!stop_requested_.load(std::memory_order_relaxed))
                std::cerr << "[Camera] OpenCV frame read failed; stopping source\n";
            break;
        }

        cv::Mat bgr;
        if (input.type() == CV_8UC3) bgr = input.clone();
        else if (input.type() == CV_8UC1) cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
        else if (input.type() == CV_8UC4) cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
        else {
            std::cerr << "[Camera] dropping unsupported OpenCV frame type " << input.type() << '\n';
            continue;
        }

        const auto received_at = std::chrono::steady_clock::now();
        Frame frame;
        frame.image = std::move(bgr);
        frame.received_at = received_at;
        frame.captured_at = received_at;
        frame.frame_number = frame_number++;
        frame.timestamp_quality = TimestampQuality::Received;
        frames_.push(std::move(frame));
    }
    frames_.stop();
}

}  // namespace hnu25::camera
