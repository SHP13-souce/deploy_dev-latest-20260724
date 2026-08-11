#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace hnu25::vest_runtime {

struct SerialPortConfig {
    std::string device;
    int baud_rate = 115200;
};

class SerialPort {
public:
    explicit SerialPort(SerialPortConfig config);
    ~SerialPort() noexcept;

    SerialPort(const SerialPort&) = delete;
    SerialPort& operator=(const SerialPort&) = delete;

    SerialPort(SerialPort&&) = delete;
    SerialPort& operator=(SerialPort&&) = delete;

    void writeAll(const std::uint8_t* data, std::size_t size);

private:
    SerialPortConfig config_;
    int fd_ = -1;

    void openPort();
    void closePort() noexcept;
};

}  // namespace hnu25::vest_runtime
