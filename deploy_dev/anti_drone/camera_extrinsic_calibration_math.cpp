#include "anti_drone/camera_extrinsic_calibration_math.hpp"

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hnu25::anti_drone {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

// 3x3 CV_64F Mat built elementwise (unambiguous, no layout assumptions).
cv::Mat matFromMatx33(const cv::Matx33d& m) {
    cv::Mat out(3, 3, CV_64F);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out.at<double>(r, c) = m(r, c);
        }
    }
    return out;
}

cv::Mat matFromVec3(const cv::Vec3d& v) {
    cv::Mat out(3, 1, CV_64F);
    out.at<double>(0, 0) = v[0];
    out.at<double>(1, 0) = v[1];
    out.at<double>(2, 0) = v[2];
    return out;
}

cv::Matx33d matx33FromMat(const cv::Mat& m) {
    cv::Matx33d out;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out(r, c) = m.at<double>(r, c);
        }
    }
    return out;
}

cv::Vec3d vec3FromMat(const cv::Mat& m) {
    return cv::Vec3d(m.at<double>(0, 0), m.at<double>(1, 0),
                     m.at<double>(2, 0));
}

}  // namespace

cv::Matx33d axisRotation(RotationAxis axis, double angle_rad) {
    const double c = std::cos(angle_rad);
    const double s = std::sin(angle_rad);
    switch (axis) {
        case RotationAxis::X:
            return cv::Matx33d(1.0, 0.0, 0.0,
                               0.0, c, -s,
                               0.0, s, c);
        case RotationAxis::Y:
            return cv::Matx33d(c, 0.0, s,
                               0.0, 1.0, 0.0,
                               -s, 0.0, c);
        case RotationAxis::Z:
            return cv::Matx33d(c, -s, 0.0,
                               s, c, 0.0,
                               0.0, 0.0, 1.0);
    }
    return cv::Matx33d::eye();
}

GimbalPose makeGimbalPose(double yaw_deg,
                          double pitch_deg,
                          const GimbalKinematics& kin) {
    const double yaw_rad =
        kin.yaw_sign * (yaw_deg - kin.yaw_zero_deg) * kDegToRad;
    const double pitch_rad =
        kin.pitch_sign * (pitch_deg - kin.pitch_zero_deg) * kDegToRad;

    const cv::Matx33d R_yaw = axisRotation(kin.yaw_axis, yaw_rad);
    const cv::Matx33d R_pitch = axisRotation(kin.pitch_axis, pitch_rad);

    GimbalPose pose;
    // R_gimbal2base = R_yaw * R_pitch: yaw outermost, pitch in the moving
    // (yaw-rotated) frame. Do not silently reorder this multiplication.
    switch (kin.rotation_order) {
        case GimbalKinematics::RotationOrder::YawPitch:
            pose.R_gimbal2base = R_yaw * R_pitch;
            break;
        default:
            throw std::invalid_argument(
                "unsupported gimbal rotation_order");
    }
    pose.t_gimbal2base = kin.t_gimbal2base_m;
    return pose;
}

bool solveHandEye(const std::vector<cv::Matx33d>& R_gripper2base,
                  const std::vector<cv::Vec3d>& t_gripper2base,
                  const std::vector<cv::Matx33d>& R_target2cam,
                  const std::vector<cv::Vec3d>& t_target2cam,
                  int method,
                  cv::Matx33d& R_cam2gripper,
                  cv::Vec3d& t_cam2gripper) {
    const std::size_t n = R_gripper2base.size();
    if (n != t_gripper2base.size() || n != R_target2cam.size() ||
        n != t_target2cam.size()) {
        return false;
    }

    std::vector<cv::Mat> Rg;
    std::vector<cv::Mat> tg;
    std::vector<cv::Mat> Rt;
    std::vector<cv::Mat> tt;
    Rg.reserve(n);
    tg.reserve(n);
    Rt.reserve(n);
    tt.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        Rg.push_back(matFromMatx33(R_gripper2base[i]));
        tg.push_back(matFromVec3(t_gripper2base[i]));
        Rt.push_back(matFromMatx33(R_target2cam[i]));
        tt.push_back(matFromVec3(t_target2cam[i]));
    }

    cv::Mat Rcg;
    cv::Mat tcg;
    try {
        cv::calibrateHandEye(Rg, tg, Rt, tt, Rcg, tcg, method);
    } catch (const cv::Exception&) {
        return false;
    }

    if (Rcg.empty() || tcg.empty() || Rcg.rows != 3 || Rcg.cols != 3 ||
        tcg.rows != 3 || tcg.cols != 1) {
        return false;
    }

    // The result is ^G T_C (camera -> gimbal). It is returned as-is; it is NOT
    // inverted. This is the single most important direction contract in the
    // whole calibration chain.
    R_cam2gripper = matx33FromMat(Rcg);
    t_cam2gripper = vec3FromMat(tcg);
    return true;
}

cv::Matx33d projectToSO3(const cv::Matx33d& M) {
    cv::SVD svd(matFromMatx33(M), cv::SVD::FULL_UV);
    cv::Mat R = svd.u * svd.vt;
    if (cv::determinant(R) < 0.0) {
        // Flip the last column of U (i.e. U * diag(1,1,-1) * V^T) to force
        // det(R) = +1.
        cv::Mat U = svd.u.clone();
        U.col(2) *= -1.0;
        R = U * svd.vt;
    }
    return matx33FromMat(R);
}

double rotationErrorRad(const cv::Matx33d& Ra, const cv::Matx33d& Rb) {
    const cv::Matx33d R_error = Ra.t() * Rb;
    const double trace =
        R_error(0, 0) + R_error(1, 1) + R_error(2, 2);
    const double c = (trace - 1.0) / 2.0;
    return std::acos(std::max(-1.0, std::min(1.0, c)));
}

ConsistencyStats evaluateFixedTargetConsistency(
    const std::vector<cv::Matx33d>& R_gimbal2base,
    const std::vector<cv::Vec3d>& t_gimbal2base,
    const cv::Matx33d& R_cam2gimbal,
    const cv::Vec3d& t_cam2gimbal,
    const std::vector<cv::Matx33d>& R_target2cam,
    const std::vector<cv::Vec3d>& t_target2cam) {
    ConsistencyStats stats;
    const std::size_t n = R_gimbal2base.size();
    if (n == 0 || n != t_gimbal2base.size() || n != R_target2cam.size() ||
        n != t_target2cam.size()) {
        return stats;
    }

    std::vector<cv::Matx33d> R_board(n);
    std::vector<cv::Vec3d> t_board(n);
    for (std::size_t i = 0; i < n; ++i) {
        // ^B R_T(i) = R_gimbal2base * R_cam2gimbal * R_target2cam
        R_board[i] =
            R_gimbal2base[i] * R_cam2gimbal * R_target2cam[i];
        // ^B t_T(i) = R_gimbal2base * (R_cam2gimbal * t_target2cam +
        //                              t_cam2gimbal) + t_gimbal2base
        t_board[i] = R_gimbal2base[i] *
                         (R_cam2gimbal * t_target2cam[i] + t_cam2gimbal) +
                     t_gimbal2base[i];
    }

    // Translation mean + scatter.
    cv::Vec3d t_mean(0.0, 0.0, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        t_mean += t_board[i];
    }
    t_mean = t_mean / static_cast<double>(n);
    stats.translation_mean = t_mean;

    double sum_t_sq = 0.0;
    double max_t = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = cv::norm(t_board[i] - t_mean);
        sum_t_sq += d * d;
        max_t = std::max(max_t, d);
    }
    stats.translation_rms_m =
        std::sqrt(sum_t_sq / static_cast<double>(n));
    stats.translation_max_m = max_t;

    // Rotation mean: sum the matrices then project onto SO(3).
    cv::Matx33d R_sum = cv::Matx33d::zeros();
    for (std::size_t i = 0; i < n; ++i) {
        R_sum += R_board[i];
    }
    stats.rotation_mean = projectToSO3(R_sum);

    // Rotation scatter (geodesic).
    double sum_r_sq = 0.0;
    double max_r = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double ang = rotationErrorRad(stats.rotation_mean, R_board[i]);
        sum_r_sq += ang * ang;
        max_r = std::max(max_r, ang);
    }
    stats.rotation_rms_deg =
        std::sqrt(sum_r_sq / static_cast<double>(n)) * kRadToDeg;
    stats.rotation_max_deg = max_r * kRadToDeg;

    stats.valid = true;
    return stats;
}

}  // namespace hnu25::anti_drone
