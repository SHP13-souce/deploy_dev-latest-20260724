# ─── Tests ────────────────────────────────────────────────────────────────

add_executable(sp_core_tests sp_core_tests.cpp)
target_link_libraries(sp_core_tests PRIVATE sp_core opencv_calib3d)
add_test(NAME sp_core_tests COMMAND sp_core_tests)

add_executable(time_pose_buffer_test time_pose_buffer_test.cpp)
find_package(Threads REQUIRED)
target_link_libraries(time_pose_buffer_test PRIVATE time_pose_buffer Threads::Threads)
add_test(NAME time_pose_buffer COMMAND time_pose_buffer_test)

add_executable(fire_gate_test fire_gate_test.cpp)
target_link_libraries(fire_gate_test PRIVATE fire_gate)
add_test(NAME fire_gate COMMAND fire_gate_test)

add_executable(mpc_planner_test mpc_planner_test.cpp)
target_link_libraries(mpc_planner_test PRIVATE mpc_planner)
add_test(NAME mpc_planner COMMAND mpc_planner_test)

add_executable(debug_web_shm_test debug_web_shm_test.cpp)
target_link_libraries(debug_web_shm_test PRIVATE debug_web)
add_test(NAME debug_web_shm_protocol COMMAND debug_web_shm_test)

add_executable(camera_test camera_test.cpp)
target_link_libraries(camera_test PRIVATE camera Threads::Threads)
add_test(NAME camera_frame_source COMMAND camera_test)

if(BUILD_APPLICATIONS)
add_executable(vest_logic_test vest_logic_test.cpp)
target_link_libraries(vest_logic_test PRIVATE vest_detection)
add_test(NAME vest_logic COMMAND vest_logic_test)

add_executable(detector_benchmark detector_benchmark.cpp)
target_link_libraries(detector_benchmark PRIVATE detection opencv_core opencv_videoio)

add_executable(sp_protocol_test sp_protocol_test.cpp)
target_link_libraries(sp_protocol_test PRIVATE communication)
add_test(NAME sp_protocol_golden COMMAND sp_protocol_test)

add_executable(sp_standard_test sp_standard_test.cpp)
target_link_libraries(sp_standard_test PRIVATE sp_standard_config)
add_test(NAME sp_standard_safety COMMAND sp_standard_test)

add_executable(runtime_tuning_test runtime_tuning_test.cpp)
target_link_libraries(runtime_tuning_test PRIVATE sp_standard_config)
add_test(NAME runtime_tuning_safety COMMAND runtime_tuning_test)

add_executable(test_video test_video.cpp)

target_include_directories(test_video PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/detection
    ${CMAKE_SOURCE_DIR}/solve
    ${CMAKE_SOURCE_DIR}/communication
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenVINO_INCLUDE_DIRS}
)

target_link_libraries(test_video PRIVATE
    detection solve communication
    opencv_core opencv_imgproc opencv_calib3d opencv_videoio opencv_highgui
    openvino::runtime
    Eigen3::Eigen
)

add_executable(sp_demo_replay sp_demo_replay.cpp)
target_include_directories(sp_demo_replay PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(sp_demo_replay PRIVATE
    detection
    sp_core
    fire_gate
    time_pose_buffer
    telemetry
    debug_web
    opencv_core
    opencv_videoio
    openvino::runtime
)

add_executable(test_telemetry ${CMAKE_SOURCE_DIR}/telemetry/test_telemetry.cpp)
target_link_libraries(test_telemetry PRIVATE telemetry)
add_test(NAME telemetry_encoding COMMAND test_telemetry)

set(SP_DEMO_ASSET_DIR "/home/robomaster/projects/sp_vision_25-main/assets/demo"
    CACHE PATH "Directory containing demo.avi and demo.txt for the replay smoke test")
if(EXISTS "${SP_DEMO_ASSET_DIR}/demo.avi" AND EXISTS "${SP_DEMO_ASSET_DIR}/demo.txt")
    add_test(NAME sp_demo_replay_smoke COMMAND sp_demo_replay
        "${SP_DEMO_ASSET_DIR}/demo.avi"
        "${SP_DEMO_ASSET_DIR}/demo.txt"
        "${CMAKE_SOURCE_DIR}/config/sp_demo.yaml"
        10)
endif()
endif()
