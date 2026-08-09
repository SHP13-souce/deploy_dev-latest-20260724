#include "camera/frame_source.hpp"
#include "camera/latest_frame_buffer.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

class MockFrameSource final : public hnu25::camera::FrameSource {
public:
    void start() override {
        worker_ = std::thread([this] {
            hnu25::camera::Frame frame;
            frame.image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(1, 2, 3)).clone();
            frame.frame_number = 10;
            frames_.push(std::move(frame));
            hnu25::camera::Frame latest;
            latest.image = cv::Mat(2, 2, CV_8UC3, cv::Scalar(4, 5, 6)).clone();
            latest.frame_number = 11;
            frames_.push(std::move(latest));
        });
    }

    void stop() noexcept override {
        frames_.stop();
        if (worker_.joinable()) worker_.join();
    }

    bool waitForFrame(hnu25::camera::Frame& frame, std::chrono::milliseconds timeout) override {
        return frames_.waitPop(frame, timeout);
    }

    ~MockFrameSource() override { stop(); }

private:
    hnu25::camera::LatestFrameBuffer frames_;
    std::thread worker_;
};

}  // namespace

int main() {
    using namespace std::chrono_literals;
    hnu25::camera::Frame frame;
    const auto now = std::chrono::steady_clock::now();
    frame.captured_at = now - 1ms;
    frame.received_at = now;
    frame.frame_number = 7;
    frame.timestamp_quality = hnu25::camera::TimestampQuality::ApproximateExposureCenter;
    require(frame.captured_at < frame.received_at, "Frame retains capture and receive times");

    hnu25::camera::LatestFrameBuffer buffer;
    frame.frame_number = 1;
    buffer.push(std::move(frame));
    hnu25::camera::Frame latest;
    latest.frame_number = 2;
    buffer.push(std::move(latest));
    require(buffer.waitPop(frame, 10ms) && frame.frame_number == 2, "new frame overwrites old frame");

    hnu25::camera::LatestFrameBuffer stopped;
    bool woke = false;
    std::thread waiter([&] { woke = !stopped.waitPop(frame, 5s); });
    std::this_thread::sleep_for(10ms);
    stopped.stop();
    waiter.join();
    require(woke, "stop wakes blocked consumer");

    MockFrameSource mock;
    mock.start();
    std::this_thread::sleep_for(10ms);
    require(mock.waitForFrame(frame, 100ms) && frame.frame_number == 11,
            "mock source uses latest-frame behavior");
    mock.stop();
    return 0;
}
