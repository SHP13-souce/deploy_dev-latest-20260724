# ─── Vest Runtime Core Library ────────────────────────────────────────────
add_library(vest_runtime_core STATIC
    runtime_config.cpp
    vest_runtime.cpp
)

target_include_directories(vest_runtime_core PUBLIC
    ${CMAKE_SOURCE_DIR}
)

target_link_libraries(vest_runtime_core PUBLIC
    vest_detection
    camera
    yaml-cpp
)

# ─── Vest Runtime Executable ──────────────────────────────────────────────
add_executable(vest_runtime
    main.cpp
)

target_link_libraries(vest_runtime PRIVATE
    vest_runtime_core
)

set_target_properties(vest_runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
