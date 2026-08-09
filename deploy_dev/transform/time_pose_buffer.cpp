#include "transform/time_pose_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace transform {
namespace {

bool finite(const Eigen::Quaterniond& q) {
    return std::isfinite(q.w()) && std::isfinite(q.x()) &&
           std::isfinite(q.y()) && std::isfinite(q.z());
}

}  // namespace

TimePoseBuffer::TimePoseBuffer(TimePoseBufferConfig config) : config_(config) {
    if (config_.capacity == 0 || config_.time_window < std::chrono::steady_clock::duration::zero() ||
        config_.max_extrapolation < std::chrono::steady_clock::duration::zero()) {
        throw std::invalid_argument("invalid time pose buffer configuration");
    }
}

bool TimePoseBuffer::push(TimePoint timestamp, const Eigen::Quaterniond& quaternion) {
    const double norm = quaternion.norm();
    if (!finite(quaternion) || !std::isfinite(norm) || norm <= 0.0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!samples_.empty() && timestamp <= samples_.back().timestamp) return false;
    samples_.push_back({timestamp, quaternion.normalized()});
    while (samples_.size() > config_.capacity) samples_.pop_front();
    while (samples_.size() > 1 &&
           samples_.back().timestamp - samples_.front().timestamp > config_.time_window) {
        samples_.pop_front();
    }
    return true;
}

PoseQuery TimePoseBuffer::query(TimePoint timestamp) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (samples_.empty()) return {};
    if (timestamp < samples_.front().timestamp) {
        return {samples_.front().quaternion, PoseStatus::TooOld,
                samples_.front().timestamp - timestamp};
    }
    if (timestamp > samples_.back().timestamp) {
        const auto age = timestamp - samples_.back().timestamp;
        return {samples_.back().quaternion,
                age <= config_.max_extrapolation ? PoseStatus::Extrapolated : PoseStatus::TooOld,
                age};
    }

    const auto upper = std::lower_bound(
        samples_.begin(), samples_.end(), timestamp,
        [](const Sample& sample, TimePoint value) { return sample.timestamp < value; });
    if (upper->timestamp == timestamp) return {upper->quaternion, PoseStatus::Exact, {}};

    const auto lower = std::prev(upper);
    Eigen::Quaterniond right = upper->quaternion;
    if (lower->quaternion.dot(right) < 0.0) right.coeffs() *= -1.0;
    const double ratio = std::chrono::duration<double>(timestamp - lower->timestamp).count() /
                         std::chrono::duration<double>(upper->timestamp - lower->timestamp).count();
    return {lower->quaternion.slerp(ratio, right).normalized(), PoseStatus::Interpolated, {}};
}

void TimePoseBuffer::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    samples_.clear();
}

std::size_t TimePoseBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_.size();
}

}  // namespace transform
