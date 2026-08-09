add_library(time_pose_buffer STATIC time_pose_buffer.cpp)
target_include_directories(time_pose_buffer PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(time_pose_buffer PUBLIC Eigen3::Eigen)
