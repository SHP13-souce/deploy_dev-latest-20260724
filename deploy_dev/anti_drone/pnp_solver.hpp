#pragma once

#include "anti_drone/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace hnu25::anti_drone {

// Calibration + target geometry for the PnP solver. Field names map directly
// to the original project's calibration YAML keys.
struct PnpSolverConfig {
    cv::Matx33d camera_matrix = cv::Matx33d::eye();

    std::vector<double> distort_coeffs{0.0, 0.0, 0.0, 0.0, 0.0};

    cv::Matx33d R_camera2gimbal = cv::Matx33d::eye();

    cv::Vec3d t_camera2gimbal{0.0, 0.0, 0.0};

    double target_width_m = 0.50;
    double target_height_m = 0.50;

    double max_reprojection_error_px = 5.0;
};

// Pure geometric PnP output. No yaw/pitch/distance/compensation yet.
struct PnpResult {
    bool valid = false;

    cv::Vec3d xyz_camera{0.0, 0.0, 0.0};
    cv::Vec3d xyz_gimbal{0.0, 0.0, 0.0};

    cv::Vec3d rvec{0.0, 0.0, 0.0};

    double reprojection_error_px = 0.0;
};

class PnpSolver {
public:
    explicit PnpSolver(const PnpSolverConfig& config);

    PnpResult solve(const TargetObservation& observation) const;

private:
    PnpSolverConfig config_;
    cv::Mat camera_matrix_;
    cv::Mat distort_coeffs_;
};

}  // namespace hnu25::anti_drone
