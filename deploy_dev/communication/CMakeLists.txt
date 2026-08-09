# ─── Communication module: BCP Serial ─────────────────────────────────────
# BCP 协议定义: 通信组 (基于 SMBU-PolarBear 标准)
# 串口封装: 精简自 fused_project/io/serial/

add_library(communication STATIC
    sp_protocol.cpp
    serial_port.cpp
)

target_include_directories(communication PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}
)

# Linux 串口不需要额外库; Windows 需要 setupapi
if(WIN32)
    target_link_libraries(communication PUBLIC setupapi)
endif()
