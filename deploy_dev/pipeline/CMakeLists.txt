# ─── Pipeline: 主程序 ─────────────────────────────────────────────────────

add_executable(hnu_vision main.cpp)

target_include_directories(hnu_vision PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/detection
    ${CMAKE_SOURCE_DIR}/solve
    ${CMAKE_SOURCE_DIR}/communication
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenVINO_INCLUDE_DIRS}
)

target_link_libraries(hnu_vision PRIVATE
    detection solve communication
    opencv_core opencv_imgproc opencv_calib3d opencv_videoio opencv_highgui
    openvino::runtime
    Eigen3::Eigen
)

add_library(sp_standard_config STATIC sp_standard_config.cpp runtime_tuning.cpp)
target_include_directories(sp_standard_config PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(sp_standard_config PUBLIC sp_core fire_gate mpc_planner communication telemetry yaml-cpp)

find_package(Threads REQUIRED)
add_executable(sp_standard sp_standard.cpp)
target_include_directories(sp_standard PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenVINO_INCLUDE_DIRS}
)
target_link_libraries(sp_standard PRIVATE
    sp_standard_config
    camera
    detection
    sp_core
    communication
    telemetry
    mpc_planner
    debug_web
    time_pose_buffer
    opencv_core
    opencv_imgproc
    opencv_calib3d
    opencv_videoio
    openvino::runtime
    Eigen3::Eigen
    Threads::Threads
)
