#pragma once
/// @file   anti_drone/serial_port.hpp
/// @brief  Linux serial-port wrapper used by the Anti-Drone diagnostic telemetry.

#include <string>
#include <vector>
#include <cstdint>

namespace hnu25 {

class SerialPort {
public:
    SerialPort();
    ~SerialPort();

    // 打开串口
    bool open(const std::string& device, int baud_rate = 115200);

    // 关闭串口
    void close();

    // 是否已打开
    bool isOpen() const;

    // 设置单次 write 的 poll 超时(毫秒)。值必须 > 0；<= 0 的入参被忽略，
    // 保留当前值。超时后 write 返回 false。
    void setWriteTimeoutMs(int timeout_ms);

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
    int write_timeout_ms_ = 20;  // poll() timeout per write attempt
};

}  // namespace hnu25
