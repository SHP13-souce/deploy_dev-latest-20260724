#include "anti_drone/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace hnu25::anti_drone {

namespace {

bool finite(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

}  // namespace

PnpSolver::PnpSolver(const PnpSolverConfig& config) : config_(config) {
    // ── Target geometry ──────────────────────────────────────────────────
    if (!(config_.target_width_m > 0.0)) {
        throw std::invalid_argument("target_width_m must be > 0");
    }
    if (!(config_.target_height_m > 0.0)) {
        throw std::invalid_argument("target_height_m must be > 0");
    }
    if (!(config_.max_reprojection_error_px > 0.0) ||
        !std::isfinite(config_.max_reprojection_error_px)) {
        throw std::invalid_argument(
            "max_reprojection_error_px must be finite and > 0");
    }

    // ── Camera matrix ────────────────────────────────────────────────────
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(config_.camera_matrix(r, c))) {
                throw std::invalid_argument("camera_matrix must be finite");
            }
        }
    }
    if (!(config_.camera_matrix(0, 0) > 0.0)) {
        throw std::invalid_argument("camera_matrix fx must be > 0");
    }
    if (!(config_.camera_matrix(1, 1) > 0.0)) {
        throw std::invalid_argument("camera_matrix fy must be > 0");
    }

    // ── Distortion ───────────────────────────────────────────────────────
    if (config_.distort_coeffs.empty()) {
        throw std::invalid_argument("distort_coeffs must not be empty");
    }
    for (const double d : config_.distort_coeffs) {
        if (!std::isfinite(d)) {
            throw std::invalid_argument("distort_coeffs must be finite");
        }
    }

    // ── Camera -> gimbal transform ───────────────────────────────────────
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (!std::isfinite(config_.R_camera2gimbal(r, c))) {
                throw std::invalid_argument("R_camera2gimbal must be finite");
            }
        }
    }
    if (!finite(config_.t_camera2gimbal)) {
        throw std::invalid_argument("t_camera2gimbal must be finite");
    }

    // ── Cache owned cv::Mat copies (never dangling headers) ─────────────
    camera_matrix_ = cv::Mat(config_.camera_matrix);  // CV_64F 3x3

    const std::size_t n = config_.distort_coeffs.size();
    distort_coeffs_ = cv::Mat(1, static_cast<int>(n), CV_64F);
    for (std::size_t i = 0; i < n; ++i) {
        distort_coeffs_.at<double>(0, static_cast<int>(i)) =
            config_.distort_coeffs[i];
    }
}

PnpResult PnpSolver::solve(const TargetObservation& observation) const {
    PnpResult result;

    if (!observation.corners_valid) {
        return result;
    }

    // ── Image corner safety ──────────────────────────────────────────────
    for (const auto& p : observation.corners) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y)) {
            return result;
        }
    }

    // ── Object points (target board frame, center at origin) ────────────
    const double half_width = config_.target_width_m / 2.0;
    const double half_height = config_.target_height_m / 2.0;

    const std::vector<cv::Point3d> object_points{
        cv::Point3d(-half_width, -half_height, 0.0),  // TL
        cv::Point3d(+half_width, -half_height, 0.0),  // TR
        cv::Point3d(+half_width, +half_height, 0.0),  // BR
        cv::Point3d(-half_width, +half_height, 0.0),  // BL
    };

    const std::vector<cv::Point2f> image_points{
        observation.corners[0],
        observation.corners[1],
        observation.corners[2],
        observation.corners[3],
    };

    // ── solvePnP ─────────────────────────────────────────────────────────
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    bool solved = false;
    try {
        solved = cv::solvePnP(
            object_points,
            image_points,
            camera_matrix_,
            distort_coeffs_,
            rvec,
            tvec,
            false,
            cv::SOLVEPNP_IPPE);
    } catch (const cv::Exception&) {
        return result;
    }

    if (!solved) {
        return result;
    }
    if (!finite(rvec) || !finite(tvec)) {
        return result;
    }
    if (!(tvec[2] > 0.0)) {
        return result;
    }

    // ── Reprojection error ───────────────────────────────────────────────
    std::vector<cv::Point2f> projected_points;
    try {
        cv::projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix_,
            distort_coeffs_,
            projected_points);
    } catch (const cv::Exception&) {
        return result;
    }

    if (projected_points.size() != image_points.size()) {
        return result;
    }

    double sum_squared_error = 0.0;
    for (std::size_t i = 0; i < image_points.size(); ++i) {
        const double dx = static_cast<double>(projected_points[i].x) -
                          static_cast<double>(image_points[i].x);
        const double dy = static_cast<double>(projected_points[i].y) -
                          static_cast<double>(image_points[i].y);
        sum_squared_error += dx * dx + dy * dy;
    }
    const double rms = std::sqrt(sum_squared_error / 4.0);

    if (!std::isfinite(rms)) {
        return result;
    }

    // ── Fill result (kept even when invalid, for debug) ─────────────────
    result.reprojection_error_px = rms;
    result.rvec = rvec;
    result.xyz_camera = tvec;
    result.xyz_gimbal =
        config_.R_camera2gimbal * result.xyz_camera +
        config_.t_camera2gimbal;

    if (rms <= config_.max_reprojection_error_px) {
        result.valid = true;
    }

    return result;
}

}  // namespace hnu25::anti_drone
