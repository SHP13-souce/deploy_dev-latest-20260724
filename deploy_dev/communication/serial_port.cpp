#include "serial_port.hpp"

#ifdef __linux__
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <poll.h>
#include <cstring>
#include <cerrno>
#include <iostream>
#endif

namespace hnu25 {

SerialPort::SerialPort() = default;

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const std::string& device, int baud_rate) {
#ifdef __linux__
    close();

    speed_t speed;
    switch (baud_rate) {
        case 9600:   speed = B9600;   break;
        case 19200:  speed = B19200;  break;
        case 38400:  speed = B38400;  break;
        case 57600:  speed = B57600;  break;
        case 115200: speed = B115200; break;
        default:
            std::cerr << "[Serial] unsupported baud rate: " << baud_rate << std::endl;
            return false;
    }

    fd_ = ::open(device.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::cerr << "[Serial] open '" << device << "' failed: "
                  << std::strerror(errno) << std::endl;
        return false;
    }

    struct termios tty;
    std::memset(&tty, 0, sizeof(tty));
    if (tcgetattr(fd_, &tty) != 0) {
        std::cerr << "[Serial] tcgetattr failed" << std::endl;
        close();
        return false;
    }

    // 设置波特率
    if (cfsetospeed(&tty, speed) != 0 || cfsetispeed(&tty, speed) != 0) {
        std::cerr << "[Serial] failed to set baud rate " << baud_rate << ": "
                  << std::strerror(errno) << std::endl;
        close();
        return false;
    }

    // 8N1, 无流控
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;

    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_oflag &= ~OPOST;

    // 超时: 0.1s
    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::cerr << "[Serial] tcsetattr failed" << std::endl;
        close();
        return false;
    }

    device_ = device;
    return true;
#else
    // Windows: 不在本文件支持 (部署目标为 Linux)
    std::cerr << "[Serial] Windows not supported in this build" << std::endl;
    return false;
#endif
}

void SerialPort::close() {
#ifdef __linux__
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#endif
}

bool SerialPort::isOpen() const {
#ifdef __linux__
    return fd_ >= 0;
#else
    return false;
#endif
}

bool SerialPort::write(const std::vector<uint8_t>& data) {
    return write(data.data(), data.size());
}

bool SerialPort::write(const uint8_t* data, size_t len) {
#ifdef __linux__
    if (fd_ < 0 || (data == nullptr && len != 0)) return false;
    size_t offset = 0;
    while (offset < len) {
        const ssize_t written = ::write(fd_, data + offset, len - offset);
        if (written > 0) {
            offset += static_cast<size_t>(written);
            continue;
        }
        if (written < 0 && errno == EINTR) continue;
        if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            struct pollfd pfd {fd_, POLLOUT, 0};
            int ready;
            do {
                ready = ::poll(&pfd, 1, -1);
            } while (ready < 0 && errno == EINTR);
            if (ready > 0) continue;
        }
        std::cerr << "[Serial] write failed: " << std::strerror(errno) << std::endl;
        return false;
    }
    return true;
#else
    return false;
#endif
}

std::vector<uint8_t> SerialPort::read(size_t max_bytes) {
#ifdef __linux__
    std::vector<uint8_t> buf(max_bytes);
    if (fd_ < 0) return {};

    ssize_t n = ::read(fd_, buf.data(), max_bytes);
    if (n < 0) {
        if (errno != EAGAIN && errno != EWOULDBLOCK)
            std::cerr << "[Serial] read error: " << std::strerror(errno) << std::endl;
        return {};
    }
    buf.resize(static_cast<size_t>(n));
    return buf;
#else
    return {};
#endif
}

void SerialPort::flush() {
#ifdef __linux__
    if (fd_ >= 0 && tcdrain(fd_) != 0)
        std::cerr << "[Serial] tcdrain failed: " << std::strerror(errno) << std::endl;
#endif
}

}  // namespace hnu25
