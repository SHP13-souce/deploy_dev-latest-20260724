cmake_minimum_required(VERSION 3.16)
project(HNU_NHS_Vision_25 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Find packages ──────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d videoio highgui)
find_package(yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

# ─── Include directories ───────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}
                    ${OpenCV_INCLUDE_DIRS})

# ─── Sanitizers (debug only) ───────────────────────────────────────────────
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    link_libraries(-fsanitize=address)
endif()

# ─── Subdirectories ────────────────────────────────────────────────────────
add_subdirectory(camera)
add_subdirectory(anti_drone)

# ─── Tests (optional) ──────────────────────────────────────────────────────
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
