add_library(camera STATIC opencv_frame_source.cpp)
target_include_directories(camera PUBLIC ${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera PUBLIC opencv_core opencv_imgproc opencv_videoio Threads::Threads)

set(HNU25_MVS_ROOT "/opt/MVS" CACHE PATH "Hikrobot MVS installation root")
find_path(HNU25_MVS_INCLUDE_DIR MvCameraControl.h
    HINTS "${HNU25_MVS_ROOT}/include" /opt/MVS/include /usr/include /usr/local/include)
find_library(HNU25_MVS_LIBRARY NAMES MvCameraControl
    HINTS /opt/MVS/lib/64 /opt/MVS/lib "${HNU25_MVS_ROOT}/lib/64"
          "${HNU25_MVS_ROOT}/lib/amd64" "${HNU25_MVS_ROOT}/lib/arm64"
          "${HNU25_MVS_ROOT}/lib" /usr/lib /usr/local/lib)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND HNU25_MVS_INCLUDE_DIR AND HNU25_MVS_LIBRARY)
    target_sources(camera PRIVATE hik_frame_source.cpp)
    target_include_directories(camera PRIVATE "${HNU25_MVS_INCLUDE_DIR}")
    target_link_libraries(camera PRIVATE "${HNU25_MVS_LIBRARY}")
    target_compile_definitions(camera PUBLIC HNU25_HAS_MVS=1)
    message(STATUS "Hik FrameSource enabled: ${HNU25_MVS_LIBRARY}")
else()
    target_compile_definitions(camera PUBLIC HNU25_HAS_MVS=0)
    message(STATUS "Hik FrameSource disabled (Linux and an MVS installation are required)")
endif()
