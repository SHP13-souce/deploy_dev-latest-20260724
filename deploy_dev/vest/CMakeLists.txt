# ─── Vest detection module: target_v3 + OpenVINO CPU ───────────────────────

add_library(vest_detection STATIC
    detector.cpp
    tracker.cpp
    target_selector.cpp
    vision_target_core.cpp
)

target_include_directories(vest_detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
)

target_link_libraries(vest_detection PUBLIC
    opencv_core
    opencv_imgproc
    openvino::runtime
)
