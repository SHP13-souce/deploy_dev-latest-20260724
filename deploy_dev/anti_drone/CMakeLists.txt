add_library(anti_drone STATIC
    traditional_detector.cpp
    config.cpp
    pnp_solver.cpp
    vision_solution.cpp
    tracker.cpp
    predictor.cpp
)

target_include_directories(anti_drone PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(anti_drone PUBLIC
    opencv_core
    opencv_imgproc
    opencv_calib3d
    yaml-cpp
)

add_executable(
    anti_drone_image_preview
    image_preview.cpp
)

target_include_directories(
    anti_drone_image_preview
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_image_preview
    PRIVATE
    anti_drone
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)
