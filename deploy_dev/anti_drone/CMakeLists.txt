add_library(anti_drone INTERFACE)

target_include_directories(anti_drone INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(anti_drone INTERFACE
    opencv_core
)
