add_library(anti_drone STATIC
    traditional_detector.cpp
    config.cpp
    pnp_solver.cpp
    vision_solution.cpp
    tracker.cpp
    predictor.cpp
    diagnostic_pipeline.cpp
    diagnostic_csv.cpp
    diagnostic_frame_processor.cpp
    gimbal_speed_filter.cpp
    vision_telemetry.cpp
    vision_telemetry_stream.cpp
    vision_telemetry_loopback.cpp
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

add_executable(
    anti_drone_runtime_preview
    runtime_preview.cpp
)

target_include_directories(
    anti_drone_runtime_preview
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_runtime_preview
    PRIVATE
    anti_drone
    opencv_core
)

add_executable(
    anti_drone_single_frame_preview
    single_frame_preview.cpp
)

target_include_directories(
    anti_drone_single_frame_preview
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_single_frame_preview
    PRIVATE
    anti_drone
    opencv_core
    opencv_imgcodecs
)

add_executable(
    anti_drone_hik_detection_preview
    hik_detection_preview.cpp
)

target_include_directories(
    anti_drone_hik_detection_preview
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_hik_detection_preview
    PRIVATE
    anti_drone
    camera
    opencv_core
)

add_executable(
    anti_drone_hik_live_preview
    hik_live_preview.cpp
)

target_include_directories(
    anti_drone_hik_live_preview
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_hik_live_preview
    PRIVATE
    anti_drone
    camera
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
)

add_executable(
    anti_drone_camera_intrinsic_calibration
    camera_intrinsic_calibration.cpp
)

target_include_directories(
    anti_drone_camera_intrinsic_calibration
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_camera_intrinsic_calibration
    PRIVATE
    anti_drone
    camera
    opencv_core
    opencv_imgproc
    opencv_calib3d
    opencv_imgcodecs
    opencv_highgui
)

add_executable(
    anti_drone_sequence_preview
    sequence_preview.cpp
)

target_include_directories(
    anti_drone_sequence_preview
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_sequence_preview
    PRIVATE
    anti_drone
    opencv_core
    opencv_imgcodecs
)

add_executable(
    anti_drone_app
    app_main.cpp
)

target_include_directories(
    anti_drone_app
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_app
    PRIVATE
    anti_drone
    camera
    opencv_core
    opencv_imgproc
    opencv_highgui
)

# Diagnostic serial transport. Built unconditionally. It reuses the migrated
# SerialPort implementation (anti_drone/serial_port.cpp) directly as a source.
add_library(
    anti_drone_telemetry_serial STATIC
    vision_telemetry_serial_transport.cpp
    serial_port.cpp
)

target_include_directories(
    anti_drone_telemetry_serial
    PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(
    anti_drone_telemetry_serial
    PUBLIC
    anti_drone
)

target_compile_definitions(
    anti_drone_telemetry_serial
    PUBLIC
    HNU25_HAS_SERIAL_TELEMETRY=1
)

target_link_libraries(
    anti_drone_app
    PRIVATE
    anti_drone_telemetry_serial
)

add_executable(
    anti_drone_vision_telemetry_receiver
    vision_telemetry_receiver.cpp
)

target_include_directories(
    anti_drone_vision_telemetry_receiver
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(
    anti_drone_vision_telemetry_receiver
    PRIVATE
    anti_drone
    anti_drone_telemetry_serial
    opencv_core
)
