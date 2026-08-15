# ─── Tests ────────────────────────────────────────────────────────────────

find_package(Threads REQUIRED)

add_executable(camera_test camera_test.cpp)
target_link_libraries(camera_test PRIVATE camera Threads::Threads)
add_test(NAME camera_frame_source COMMAND camera_test)

add_executable(
    anti_drone_geometry_test
    anti_drone_geometry_test.cpp
)

target_link_libraries(
    anti_drone_geometry_test
    PRIVATE
    anti_drone
    opencv_core
    opencv_imgproc
)

add_test(
    NAME anti_drone_geometry
    COMMAND anti_drone_geometry_test
)

add_executable(
    anti_drone_color_test
    anti_drone_color_test.cpp
)

target_link_libraries(
    anti_drone_color_test
    PRIVATE
    anti_drone
    opencv_core
    opencv_imgproc
)

add_test(
    NAME anti_drone_color
    COMMAND anti_drone_color_test
)

add_executable(
    anti_drone_nms_test
    anti_drone_nms_test.cpp
)

target_link_libraries(
    anti_drone_nms_test
    PRIVATE
    anti_drone
    opencv_core
    opencv_imgproc
)

add_test(
    NAME anti_drone_nms
    COMMAND anti_drone_nms_test
)

add_executable(
    anti_drone_config_test
    anti_drone_config_test.cpp
)

target_link_libraries(
    anti_drone_config_test
    PRIVATE
    anti_drone
)

add_test(
    NAME anti_drone_config
    COMMAND anti_drone_config_test
)

add_executable(
    anti_drone_gimbal_speed_filter_test
    anti_drone_gimbal_speed_filter_test.cpp
)

target_link_libraries(
    anti_drone_gimbal_speed_filter_test
    PRIVATE
    anti_drone
)

add_test(
    NAME anti_drone_gimbal_speed_filter
    COMMAND anti_drone_gimbal_speed_filter_test
)

add_executable(
    anti_drone_pnp_test
    anti_drone_pnp_test.cpp
)

target_link_libraries(
    anti_drone_pnp_test
    PRIVATE
    anti_drone
    opencv_core
    opencv_calib3d
)

add_test(
    NAME anti_drone_pnp
    COMMAND anti_drone_pnp_test
)

add_executable(
    anti_drone_vision_solution_test
    anti_drone_vision_solution_test.cpp
)

target_link_libraries(
    anti_drone_vision_solution_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_vision_solution
    COMMAND anti_drone_vision_solution_test
)

add_executable(
    anti_drone_tracker_test
    anti_drone_tracker_test.cpp
)

target_link_libraries(
    anti_drone_tracker_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_tracker
    COMMAND anti_drone_tracker_test
)

add_executable(
    anti_drone_predictor_test
    anti_drone_predictor_test.cpp
)

target_link_libraries(
    anti_drone_predictor_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_predictor
    COMMAND anti_drone_predictor_test
)

add_executable(
    anti_drone_prediction_solution_test
    anti_drone_prediction_solution_test.cpp
)

target_link_libraries(
    anti_drone_prediction_solution_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_prediction_solution
    COMMAND anti_drone_prediction_solution_test
)

add_executable(
    anti_drone_diagnostic_pipeline_test
    anti_drone_diagnostic_pipeline_test.cpp
)

target_link_libraries(
    anti_drone_diagnostic_pipeline_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_diagnostic_pipeline
    COMMAND anti_drone_diagnostic_pipeline_test
)

add_executable(
    anti_drone_diagnostic_csv_test
    anti_drone_diagnostic_csv_test.cpp
)

target_link_libraries(
    anti_drone_diagnostic_csv_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_diagnostic_csv
    COMMAND anti_drone_diagnostic_csv_test
)

add_executable(
    anti_drone_production_config_smoke_test
    anti_drone_production_config_smoke_test.cpp
)

target_link_libraries(
    anti_drone_production_config_smoke_test
    PRIVATE
    anti_drone
    opencv_core
)

target_compile_definitions(
    anti_drone_production_config_smoke_test
    PRIVATE
    ANTI_DRONE_SOURCE_DIR=\"${CMAKE_SOURCE_DIR}\"
)

add_test(
    NAME anti_drone_production_config_smoke
    COMMAND anti_drone_production_config_smoke_test
)

add_test(
    NAME anti_drone_runtime_preview_smoke
    COMMAND
        anti_drone_runtime_preview
        ${CMAKE_SOURCE_DIR}/config/anti_drone.yaml
)

add_test(
    NAME anti_drone_app_check_smoke
    COMMAND
        anti_drone_app
        ${CMAKE_SOURCE_DIR}/config/anti_drone.yaml
        --check
)

add_executable(
    anti_drone_diagnostic_frame_processor_test
    anti_drone_diagnostic_frame_processor_test.cpp
)

target_link_libraries(
    anti_drone_diagnostic_frame_processor_test
    PRIVATE
    anti_drone
    opencv_core
    opencv_imgproc
)

add_test(
    NAME anti_drone_diagnostic_frame_processor
    COMMAND anti_drone_diagnostic_frame_processor_test
)

add_executable(
    anti_drone_vision_telemetry_test
    anti_drone_vision_telemetry_test.cpp
)

target_link_libraries(
    anti_drone_vision_telemetry_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_vision_telemetry
    COMMAND anti_drone_vision_telemetry_test
)

add_executable(
    anti_drone_vision_telemetry_stream_test
    anti_drone_vision_telemetry_stream_test.cpp
)

target_link_libraries(
    anti_drone_vision_telemetry_stream_test
    PRIVATE
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_vision_telemetry_stream
    COMMAND anti_drone_vision_telemetry_stream_test
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
add_executable(
    anti_drone_vision_telemetry_serial_integration_test
    anti_drone_vision_telemetry_serial_integration_test.cpp
)

target_link_libraries(
    anti_drone_vision_telemetry_serial_integration_test
    PRIVATE
    anti_drone_telemetry_serial
    anti_drone
    opencv_core
)

add_test(
    NAME anti_drone_vision_telemetry_serial_integration
    COMMAND anti_drone_vision_telemetry_serial_integration_test
)
endif()
