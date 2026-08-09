add_library(fire_gate STATIC fire_gate.cpp)
target_include_directories(fire_gate PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(fire_gate PUBLIC sp_core)
