# ─── Solve module: PnP + EKF Predictor ────────────────────────────────────
# PnP: 解算组自研 (160行 solver.cpp 移植)
# EKF: 解算组 (kalman_filter + motion_model + predictor)

add_library(solve STATIC
    pnp_solver.cpp
    motion_model.cpp
    kalman_filter.cpp
    predictor.cpp
)

target_include_directories(solve PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(solve PUBLIC
    opencv_core opencv_calib3d
    Eigen3::Eigen
)
