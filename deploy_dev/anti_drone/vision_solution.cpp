#include "anti_drone/vision_solution.hpp"

#include <cmath>

namespace hnu25::anti_drone {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;

bool finite(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

// Wraps an angle into the range [-pi, pi] when finite; leaves non-finite
// values untouched so the caller's final finite gate can reject them.
double wrapAngle(double angle) {
    if (!std::isfinite(angle)) {
        return angle;
    }
    return std::remainder(angle, 2.0 * kPi);
}

}  // namespace

VisionSolution makeVisionSolution(
    const cv::Vec3d& xyz_gimbal_raw,
    const VisionCompensationConfig& compensation) {
    VisionSolution result;

    if (!finite(xyz_gimbal_raw)) {
        return result;
    }

    // Defensive validation: the YAML loader already rejects non-finite
    // compensation, but this function may be called with a programmatically
    // built Config object.
    if (!std::isfinite(compensation.yaw_offset_deg) ||
        !std::isfinite(compensation.pitch_offset_deg) ||
        !std::isfinite(compensation.x_offset_m) ||
        !std::isfinite(compensation.y_offset_m) ||
        !std::isfinite(compensation.z_offset_m)) {
        return result;
    }

    const cv::Vec3d raw = xyz_gimbal_raw;
    result.xyz_gimbal_raw = raw;

    // ── Raw geometric yaw / pitch (line-of-sight, no ballistics) ─────────
    const double raw_horizontal = std::hypot(raw[0], raw[1]);
    if (!(raw_horizontal > 1e-9) || !std::isfinite(raw_horizontal)) {
        return result;
    }
    const double yaw_raw = std::atan2(raw[1], raw[0]);
    const double pitch_raw = std::atan2(raw[2], raw_horizontal);
    if (!std::isfinite(yaw_raw) || !std::isfinite(pitch_raw)) {
        return result;
    }
    result.yaw_raw_rad = yaw_raw;
    result.pitch_raw_rad = pitch_raw;

    // ── xyz residual translation (gimbal frame, after camera->gimbal) ────
    const cv::Vec3d xyz_compensated(
        raw[0] + compensation.x_offset_m,
        raw[1] + compensation.y_offset_m,
        raw[2] + compensation.z_offset_m);
    if (!finite(xyz_compensated)) {
        return result;
    }
    result.xyz_gimbal_compensated = xyz_compensated;

    // ── Recompute geometric angles from the compensated position ─────────
    const double compensated_horizontal =
        std::hypot(xyz_compensated[0], xyz_compensated[1]);
    if (!(compensated_horizontal > 1e-9) ||
        !std::isfinite(compensated_horizontal)) {
        return result;
    }
    const double geometric_yaw =
        std::atan2(xyz_compensated[1], xyz_compensated[0]);
    const double geometric_pitch =
        std::atan2(xyz_compensated[2], compensated_horizontal);

    // ── Apply yaw / pitch residual offsets (degrees -> radians) ──────────
    const double yaw_offset_rad = compensation.yaw_offset_deg * kDegToRad;
    const double pitch_offset_rad =
        compensation.pitch_offset_deg * kDegToRad;

    const double yaw_compensated = geometric_yaw + yaw_offset_rad;
    const double pitch_compensated = geometric_pitch + pitch_offset_rad;

    result.yaw_compensated_rad = wrapAngle(yaw_compensated);
    result.pitch_compensated_rad = pitch_compensated;

    // ── Final finite gate before marking valid ───────────────────────────
    if (!finite(result.xyz_gimbal_raw) ||
        !finite(result.xyz_gimbal_compensated) ||
        !std::isfinite(result.yaw_raw_rad) ||
        !std::isfinite(result.pitch_raw_rad) ||
        !std::isfinite(result.yaw_compensated_rad) ||
        !std::isfinite(result.pitch_compensated_rad)) {
        return result;
    }

    result.valid = true;
    return result;
}

VisionSolution makeVisionSolution(
    const PnpResult& pnp,
    const VisionCompensationConfig& compensation) {
    if (!pnp.valid) {
        return {};
    }
    return makeVisionSolution(pnp.xyz_gimbal, compensation);
}

}  // namespace hnu25::anti_drone
