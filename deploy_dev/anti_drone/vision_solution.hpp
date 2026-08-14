#pragma once

#include "anti_drone/config.hpp"
#include "anti_drone/pnp_solver.hpp"

#include <opencv2/core.hpp>

namespace hnu25::anti_drone {

// Downstream, derived vision solution: keeps the raw PnP result alongside a
// residual-compensated position and geometric pointing direction. This is
// purely for visual localization and debugging — no control, serial, or
// actuation logic lives here.
struct VisionSolution {
    bool valid = false;

    cv::Vec3d xyz_gimbal_raw{0.0, 0.0, 0.0};
    cv::Vec3d xyz_gimbal_compensated{0.0, 0.0, 0.0};

    double yaw_raw_rad = 0.0;
    double pitch_raw_rad = 0.0;

    double yaw_compensated_rad = 0.0;
    double pitch_compensated_rad = 0.0;
};

// Pure function deriving a compensated vision solution from a raw PnP result
// and the residual compensation parameters. Never mutates its inputs.
VisionSolution makeVisionSolution(
    const PnpResult& pnp,
    const VisionCompensationConfig& compensation);

}  // namespace hnu25::anti_drone
