#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace hnu25::anti_drone {

// Pure-math helpers for the camera -> gimbal (hand-eye) extrinsic calibration
// tool. This header deliberately has NO dependency on yaml-cpp, serial,
// telemetry, tracker, predictor, or the camera runtime: it is shared between
// the offline calibration executable and its unit tests.
//
// Frame conventions (must match pnp_solver / vision_solution):
//
//   B = fixed base / world frame
//   G = moving gimbal frame
//   C = camera frame (fixed to the gimbal, moves with it)
//   T = fixed chessboard target frame
//
// A transform ^A T_B maps a point from frame B into frame A:
//
//   p_A = ^A R_B * p_B + ^A t_B
//
// The production PnP chain uses
//
//   p_gimbal = R_camera2gimbal * p_camera + t_camera2gimbal
//
// i.e. R_camera2gimbal = ^G R_C and t_camera2gimbal = ^G t_C (meters). This is
// exactly what cv::calibrateHandEye() returns as R_cam2gripper / t_cam2gripper
// when "gripper" = gimbal and "target" = chessboard. The tool MUST NOT invert
// this result.

enum class RotationAxis {
    X,
    Y,
    Z,
};

// How yaw and pitch combine into the gimbal's base-frame orientation.
//
//   R_gimbal2base = R_yaw * R_pitch
//
// yaw is applied first (outermost); the pitch axis is understood in the moving
// (yaw-rotated) frame. The base-frame origin is chosen at the gimbal reference
// rotation center, so in the first version t_gimbal2base_m is [0,0,0].
struct GimbalKinematics {
    RotationAxis yaw_axis = RotationAxis::Z;
    RotationAxis pitch_axis = RotationAxis::Y;

    enum class RotationOrder {
        YawPitch,  // R = R_yaw * R_pitch
    };
    RotationOrder rotation_order = RotationOrder::YawPitch;

    double yaw_sign = 1.0;
    double pitch_sign = 1.0;

    double yaw_zero_deg = 0.0;
    double pitch_zero_deg = 0.0;

    cv::Vec3d t_gimbal2base_m{0.0, 0.0, 0.0};
};

// Right-handed 3x3 rotation of `angle_rad` about `axis`.
cv::Matx33d axisRotation(RotationAxis axis, double angle_rad);

struct GimbalPose {
    cv::Matx33d R_gimbal2base = cv::Matx33d::eye();
    cv::Vec3d t_gimbal2base{0.0, 0.0, 0.0};
};

// Builds the gimbal pose ^B T_G from recorded yaw/pitch (degrees) using the
// kinematics definition above. Degrees are converted to radians internally.
GimbalPose makeGimbalPose(double yaw_deg,
                          double pitch_deg,
                          const GimbalKinematics& kin);

// Standard hand-eye solve. `method` is one of cv::CALIB_HAND_EYE_* (kept as an
// int so this header does not need calib3d).
//
// Inputs (per sample i):
//   R_gripper2base[i] = ^B R_G(i)   (gimbal pose in the fixed base frame)
//   t_gripper2base[i] = ^B t_G(i)
//   R_target2cam[i]   = ^C R_T(i)   (fixed chessboard pose in the camera frame,
//                                    from solvePnP + Rodrigues)
//   t_target2cam[i]   = ^C t_T(i)
//
// Outputs (NOT inverted):
//   R_cam2gripper = ^G R_C = R_camera2gimbal
//   t_cam2gripper = ^G t_C = t_camera2gimbal
//
// so that
//   p_gimbal = R_camera2gimbal * p_camera + t_camera2gimbal.
bool solveHandEye(const std::vector<cv::Matx33d>& R_gripper2base,
                  const std::vector<cv::Vec3d>& t_gripper2base,
                  const std::vector<cv::Matx33d>& R_target2cam,
                  const std::vector<cv::Vec3d>& t_target2cam,
                  int method,
                  cv::Matx33d& R_cam2gripper,
                  cv::Vec3d& t_cam2gripper);

// Fixed-target consistency. The chessboard is fixed in the world, so the
// recovered board pose in the base frame,
//
//   ^B T_T(i) = ^B T_G(i) * ^G T_C * ^C T_T(i),
//
// must be the same for every sample. Scatter statistics are returned; a good
// calibration shows values close to zero.
struct ConsistencyStats {
    bool valid = false;

    cv::Vec3d translation_mean{0.0, 0.0, 0.0};
    cv::Matx33d rotation_mean = cv::Matx33d::eye();

    double translation_rms_m = 0.0;
    double translation_max_m = 0.0;

    double rotation_rms_deg = 0.0;
    double rotation_max_deg = 0.0;
};

ConsistencyStats evaluateFixedTargetConsistency(
    const std::vector<cv::Matx33d>& R_gimbal2base,
    const std::vector<cv::Vec3d>& t_gimbal2base,
    const cv::Matx33d& R_cam2gimbal,
    const cv::Vec3d& t_cam2gimbal,
    const std::vector<cv::Matx33d>& R_target2cam,
    const std::vector<cv::Vec3d>& t_target2cam);

// Projects a 3x3 matrix onto SO(3) via SVD (closest rotation). Used for the
// mean rotation and to clean a near-proper hand-eye result.
cv::Matx33d projectToSO3(const cv::Matx33d& M);

// Geodesic angle between two rotations, in radians:
//   acos(clamp((trace(Ra^T * Rb) - 1) / 2, -1, 1))
double rotationErrorRad(const cv::Matx33d& Ra, const cv::Matx33d& Rb);

}  // namespace hnu25::anti_drone
