add_library(sp_core STATIC sp_core.cpp)

target_include_directories(sp_core PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(sp_core PUBLIC
    Eigen3::Eigen
    yaml-cpp
    opencv_core
    opencv_calib3d
)
