add_library(mpc_planner STATIC mpc_planner.cpp)
target_include_directories(mpc_planner PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(mpc_planner PUBLIC sp_core Eigen3::Eigen)
