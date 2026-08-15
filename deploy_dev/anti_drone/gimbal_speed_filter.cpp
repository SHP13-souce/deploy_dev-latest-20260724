#include "anti_drone/gimbal_speed_filter.hpp"

#include <cmath>

namespace hnu25::anti_drone {

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

GimbalSpeedFilter::GimbalSpeedFilter(double alpha) : alpha_(alpha) {}

GimbalSpeed GimbalSpeedFilter::update(
    double current_yaw,
    double current_pitch,
    double dt_s) {
    GimbalSpeed out;

    if (have_prev_ && dt_s > 0.0) {
        // Wrap the yaw difference into [-pi, pi] before differencing so a
        // +/-pi seam crossing yields the short delta. Pitch is not periodic.
        const double delta_yaw =
            std::remainder(current_yaw - prev_yaw_, 2.0 * kPi);
        const double raw_yaw = delta_yaw / dt_s;
        const double raw_pitch = (current_pitch - prev_pitch_) / dt_s;

        filt_yaw_ = (1.0 - alpha_) * filt_yaw_ + alpha_ * raw_yaw;
        filt_pitch_ = (1.0 - alpha_) * filt_pitch_ + alpha_ * raw_pitch;

        if (std::isfinite(filt_yaw_) && std::isfinite(filt_pitch_)) {
            out.yaw_rad_s = static_cast<float>(filt_yaw_);
            out.pitch_rad_s = static_cast<float>(filt_pitch_);
        }
    }

    prev_yaw_ = current_yaw;
    prev_pitch_ = current_pitch;
    have_prev_ = true;
    return out;
}

void GimbalSpeedFilter::reset() noexcept {
    have_prev_ = false;
    prev_yaw_ = 0.0;
    prev_pitch_ = 0.0;
    filt_yaw_ = 0.0;
    filt_pitch_ = 0.0;
}

}  // namespace hnu25::anti_drone
