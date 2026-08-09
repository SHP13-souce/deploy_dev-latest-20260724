#pragma once

#include <Eigen/Geometry>

#include <chrono>
#include <cstddef>
#include <deque>
#include <limits>
#include <mutex>

namespace transform {

using TimePoint = std::chrono::steady_clock::time_point;

enum class PoseStatus { Exact, Interpolated, Extrapolated, TooOld, Empty };

struct PoseQuery {
    Eigen::Quaterniond quaternion{std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN(),
                                  std::numeric_limits<double>::quiet_NaN()};
    PoseStatus status = PoseStatus::Empty;
    std::chrono::steady_clock::duration age{};

    bool valid() const {
        return status == PoseStatus::Exact || status == PoseStatus::Interpolated ||
               status == PoseStatus::Extrapolated;
    }
};

struct TimePoseBufferConfig {
    std::size_t capacity = 256;
    std::chrono::steady_clock::duration time_window = std::chrono::seconds(2);
    std::chrono::steady_clock::duration max_extrapolation = std::chrono::milliseconds(50);
};

class TimePoseBuffer {
public:
    explicit TimePoseBuffer(TimePoseBufferConfig config = {});

    bool push(TimePoint timestamp, const Eigen::Quaterniond& quaternion);
    PoseQuery query(TimePoint timestamp) const;
    void reset();
    std::size_t size() const;

private:
    struct Sample {
        TimePoint timestamp;
        Eigen::Quaterniond quaternion;
    };

    TimePoseBufferConfig config_;
    mutable std::mutex mutex_;
    std::deque<Sample> samples_;
};

}  // namespace transform
