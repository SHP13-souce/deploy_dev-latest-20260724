# ─── Detection module: Yolo26Detector ──────────────────────────────────────
# 河北科技大学 Actor&Thinker YOLO26n-Pose + OpenVINO CPU

add_library(detection STATIC
    detector.cpp
    awakening_detector.cpp
    yolo26_detector.cpp
)

target_include_directories(detection PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${OpenVINO_INCLUDE_DIRS}
)

target_link_libraries(detection PUBLIC
    opencv_core opencv_imgproc
    openvino::runtime
)
