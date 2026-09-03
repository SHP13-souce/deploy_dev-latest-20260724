#pragma once

#include <stdexcept>
#include <string>

// Pure config-validation helpers shared between the camera -> gimbal extrinsic
// calibration executable and its unit tests. This header deliberately has no
// dependency on yaml-cpp, OpenCV, or the runtime: it is header-only so that it
// can be exercised by the test target without dragging in the parser's YAML
// machinery.

namespace hnu25::anti_drone {

// ── Gimbal kinematics sign validation ───────────────────────────────────────
// The gimbal yaw/pitch sign fields must be exactly +1.0 or -1.0. A value of 0,
// a fractional magnitude (0.5), an arbitrary magnitude (2.0), or a non-finite
// value is a configuration error and is rejected with a clear message naming
// the offending key.
inline void requireGimbalSign(double sign, const char* field_name) {
    if (sign != 1.0 && sign != -1.0) {
        throw std::runtime_error(std::string(field_name) +
                                 " must be +1 or -1");
    }
}

// ── Camera calibration resolution validation ────────────────────────────────
// The optional calibration_image_width / calibration_image_height pair must be
// either (a) both absent/0, or (b) both strictly > 0. Any other combination —
// only one present, one > 0 with the other <= 0, or a negative value — is a
// configuration error.
inline void validateCalibrationResolution(bool has_width,
                                          bool has_height,
                                          int width,
                                          int height) {
    if (has_width != has_height) {
        throw std::runtime_error(
            "calibration_image_width and calibration_image_height must be "
            "specified together (both present or both absent)");
    }
    if (!has_width) {
        return;  // both absent -> both default to 0
    }
    if (width < 0 || height < 0) {
        throw std::runtime_error(
            "calibration_image_width and calibration_image_height must be "
            "non-negative");
    }
    if ((width > 0) != (height > 0)) {
        throw std::runtime_error(
            "calibration_image_width and calibration_image_height must both "
            "be > 0, or both be 0");
    }
}

}  // namespace hnu25::anti_drone
