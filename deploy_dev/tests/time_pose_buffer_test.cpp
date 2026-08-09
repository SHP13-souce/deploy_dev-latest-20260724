#include "transform/time_pose_buffer.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <atomic>
#include <thread>

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAILED: " << message << '\n'; std::exit(1); }
}
}

int main() {
    using namespace std::chrono_literals;
    const transform::TimePoint epoch{};
    transform::TimePoseBuffer buffer({4, 10s, 50ms});
    require(buffer.query(epoch).status == transform::PoseStatus::Empty, "empty status");
    require(!buffer.push(epoch, Eigen::Quaterniond(0, 0, 0, 0)), "zero rejected");
    require(!buffer.push(epoch, Eigen::Quaterniond(std::numeric_limits<double>::quiet_NaN(), 0, 0, 0)),
            "non-finite rejected");
    const Eigen::Quaterniond start = Eigen::Quaterniond::Identity();
    const Eigen::Quaterniond end(Eigen::AngleAxisd(1.0, Eigen::Vector3d::UnitZ()));
    require(buffer.push(epoch, start), "first push");
    require(!buffer.push(epoch, start), "duplicate timestamp rejected");
    require(buffer.push(epoch + 100ms, end), "second push");
    require(!buffer.push(epoch + 50ms, start), "out of order rejected");
    auto result = buffer.query(epoch + 50ms);
    require(result.status == transform::PoseStatus::Interpolated, "interpolated status");
    require(std::abs(Eigen::AngleAxisd(result.quaternion).angle() - 0.5) < 1e-9, "slerp midpoint");
    require(buffer.query(epoch + 100ms).status == transform::PoseStatus::Exact, "exact status");

    transform::TimePoseBuffer double_cover;
    require(double_cover.push(epoch, start), "double cover first");
    require(double_cover.push(epoch + 100ms, Eigen::Quaterniond(-end.w(), -end.x(), -end.y(), -end.z())),
            "double cover second");
    result = double_cover.query(epoch + 50ms);
    require(std::abs(Eigen::AngleAxisd(result.quaternion).angle() - 0.5) < 1e-9,
            "short path for quaternion double cover");
    require(buffer.query(epoch + 120ms).status == transform::PoseStatus::Extrapolated, "bounded extrapolation");
    require(buffer.query(epoch + 151ms).status == transform::PoseStatus::TooOld, "expired extrapolation");
    require(buffer.query(epoch - 1ms).status == transform::PoseStatus::TooOld, "before oldest");
    buffer.reset();
    require(buffer.query(epoch).status == transform::PoseStatus::Empty, "reset empties buffer");

    transform::TimePoseBuffer concurrent({1024, 10s, 50ms});
    std::atomic<bool> writer_done{false};
    std::thread writer([&] {
        for (int i = 1; i <= 1000; ++i) {
            require(concurrent.push(epoch + std::chrono::microseconds(i), start),
                    "concurrent ordered push");
        }
        writer_done.store(true);
    });
    std::thread reader([&] {
        while (!writer_done.load()) {
            const auto query = concurrent.query(epoch + 500us);
            require(query.status == transform::PoseStatus::Empty ||
                    query.status == transform::PoseStatus::TooOld || query.valid(),
                    "concurrent query returns a valid status");
            (void)concurrent.size();
        }
    });
    writer.join();
    reader.join();
    require(concurrent.size() == 1000, "concurrent buffer retains all samples");
    return 0;
}
