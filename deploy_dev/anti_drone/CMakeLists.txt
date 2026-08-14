add_library(anti_drone STATIC
    traditional_detector.cpp
)

target_include_directories(anti_drone PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(anti_drone PUBLIC
    opencv_core
)
