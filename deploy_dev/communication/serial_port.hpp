#pragma once
/// @file   communication/serial_port.hpp
/// @brief  精简串口封装 (Linux) —— 通信组
///
/// 基于同项目 serial 库精简，仅保留 Linux 端核心功能

#include <string>
#include <vector>
#include <cstdint>

namespace hnu25 {

class SerialPort {
public:
    SerialPort();
    ~SerialPort();

    // 打开串口
    bool open(const std::string& device = "/dev/rm_cboard", int baud_rate = 115200);

    // 关闭串口
    void close();

    // 是否已打开
    bool isOpen() const;

    // 写入全部数据 (处理 partial write/EINTR/EAGAIN)
    bool write(const std::vector<uint8_t>& data);

    // 写入原始字节 (阻塞)
    bool write(const uint8_t* data, size_t len);

    // 读取最多 max_bytes 字节 (非阻塞, 返回实际读取)
    std::vector<uint8_t> read(size_t max_bytes = 256);

    // 等待已写数据发送完成，不丢弃收发缓冲区
    void flush();

private:
    int fd_ = -1;
    std::string device_;
};

}  // namespace hnu25
