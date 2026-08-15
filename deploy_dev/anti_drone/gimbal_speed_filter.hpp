#pragma once

namespace hnu25::anti_drone {

// One sample of the cross-frame gimbal-following angular rate.
struct GimbalSpeed {
    float yaw_rad_s = 0.0F;
    float pitch_rad_s = 0.0F;
};

// Cross-frame gimbal-following angular-rate estimator. It computes the
// low-pass-filtered first difference of the compensated pointing direction:
//
//     raw_yaw   = wrap_pi(current_yaw   - prev_yaw)   / dt
//     raw_pitch =            (current_pitch - prev_pitch) / dt
//     filtered  = (1 - alpha) * old + alpha * raw
//
// The yaw difference is wrapped to [-pi, pi] before differencing so a crossing
// of the +/-pi seam (e.g. +179deg -> -179deg) produces the short ~+2deg delta
// instead of the raw ~-358deg delta. Pitch is intentionally NOT wrapped.
//
// The estimate is 0 on the first valid sample and is reset to 0 by reset();
// a dt <= 0 sample does not advance the filter and yields 0.
class GimbalSpeedFilter {
public:
    explicit GimbalSpeedFilter(double alpha);

    // Advances the filter with a new valid pointing sample. `current_yaw` /
    // `current_pitch` are radians; `dt_s` is the elapsed time in seconds since
    // the previous valid sample. Returns {yaw_rad_s, pitch_rad_s}: 0 on the
    // first sample, on dt_s <= 0, and whenever a filtered value is non-finite.
    GimbalSpeed update(double current_yaw, double current_pitch, double dt_s);

    // Marks the solution invalid / target lost: clears the prior-sample state
    // and resets the filtered estimate to 0 so a stale rate never spans a
    // target-loss gap. The next update() is again treated as a first sample.
    void reset() noexcept;

private:
    double alpha_;

    bool have_prev_ = false;
    double prev_yaw_ = 0.0;
    double prev_pitch_ = 0.0;

    double filt_yaw_ = 0.0;
    double filt_pitch_ = 0.0;
};

}  // namespace hnu25::anti_drone
