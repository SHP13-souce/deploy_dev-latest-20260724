add_library(debug_web STATIC debug_web_publisher.cpp)
target_include_directories(debug_web PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(debug_web PUBLIC opencv_core opencv_imgcodecs)

if(UNIX AND NOT APPLE)
    target_link_libraries(debug_web PUBLIC rt)
endif()
