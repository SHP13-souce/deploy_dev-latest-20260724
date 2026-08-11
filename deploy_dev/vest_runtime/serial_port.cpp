#include "vest_runtime/serial_port.hpp"

#if defined(__linux__)
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

#include <stdexcept>
#include <string>
#include <utility>

#if defined(__linux__)
namespace {

speed_t baudRateToTermios(int baud_rate) {
    switch (baud_rate) {
        case 9600:
            return B9600;
        case 19200:
            return B19200;
        case 38400:
            return B38400;
        case 57600:
            return B57600;
        case 115200:
            return B115200;
        case 230400:
            return B230400;
#ifdef B460800
        case 460800:
            return B460800;
#endif
#ifdef B500000
        case 500000:
            return B500000;
#endif
#ifdef B576000
        case 576000:
            return B576000;
#endif
#ifdef B921600
        case 921600:
            return B921600;
#endif
#ifdef B1000000
        case 1000000:
            return B1000000;
#endif
        default:
            throw std::invalid_argument(
                "unsupported serial baud rate: " +
                std::to_string(baud_rate));
    }
}

}  // namespace
#endif

namespace hnu25::vest_runtime {

// ── Constructor / Destructor ───────────────────────────────────────────

SerialPort::SerialPort(SerialPortConfig config)
    : config_(std::move(config)) {
    openPort();
}

SerialPort::~SerialPort() noexcept {
    closePort();
}

// ── closePort ──────────────────────────────────────────────────────────

void SerialPort::closePort() noexcept {
#if defined(__linux__)
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
#else
    fd_ = -1;
#endif
}

// ── openPort ───────────────────────────────────────────────────────────

void SerialPort::openPort() {
#if !defined(__linux__)
    (void)config_;
    throw std::runtime_error(
        "serial port is only supported on Linux");
#else

    if (config_.device.empty()) {
        throw std::invalid_argument(
            "serial device must not be empty");
    }

    if (config_.baud_rate <= 0) {
        throw std::invalid_argument(
            "serial baud rate must be positive");
    }

    // ── Validate baud rate before acquiring fd ───────────────────
    const speed_t speed = baudRateToTermios(config_.baud_rate);

    // ── Open device ──────────────────────────────────────────────
    fd_ = ::open(config_.device.c_str(),
                 O_RDWR | O_NOCTTY | O_CLOEXEC);

    if (fd_ < 0) {
        const int error_number = errno;
        throw std::runtime_error(
            "failed to open serial device " + config_.device +
            ": " + std::strerror(error_number));
    }

    // ── termios ──────────────────────────────────────────────────
    struct termios tty {};
    if (::tcgetattr(fd_, &tty) < 0) {
        const int error_number = errno;
        closePort();
        throw std::runtime_error(
            "tcgetattr " + config_.device + ": " +
            std::strerror(error_number));
    }

    cfmakeraw(&tty);

    // 8N1
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cflag |= CLOCAL;
    tty.c_cflag |= CREAD;

    // No software flow control
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);

    // No hardware flow control
#ifdef CRTSCTS
    tty.c_cflag &= ~CRTSCTS;
#endif

    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 0;

    // ── Baud rate ────────────────────────────────────────────────
    if (::cfsetispeed(&tty, speed) < 0) {
        const int error_number = errno;
        closePort();
        throw std::runtime_error(
            "cfsetispeed " + config_.device + ": " +
            std::strerror(error_number));
    }

    if (::cfsetospeed(&tty, speed) < 0) {
        const int error_number = errno;
        closePort();
        throw std::runtime_error(
            "cfsetospeed " + config_.device + ": " +
            std::strerror(error_number));
    }

    // ── Apply ────────────────────────────────────────────────────
    if (::tcsetattr(fd_, TCSANOW, &tty) < 0) {
        const int error_number = errno;
        closePort();
        throw std::runtime_error(
            "tcsetattr " + config_.device + ": " +
            std::strerror(error_number));
    }

#endif  // __linux__
}

// ── writeAll ───────────────────────────────────────────────────────────

void SerialPort::writeAll(const std::uint8_t* data,
                          std::size_t size) {
    if (size == 0) {
        return;
    }

    if (data == nullptr) {
        throw std::invalid_argument(
            "serial write data must not be null");
    }

#if !defined(__linux__)
    (void)data;
    (void)size;
    throw std::runtime_error(
        "serial port is only supported on Linux");
#else

    if (fd_ < 0) {
        throw std::runtime_error("serial port is not open");
    }

    std::size_t offset = 0;

    while (offset < size) {
        const ssize_t written =
            ::write(fd_, data + offset, size - offset);

        if (written > 0) {
            offset += static_cast<std::size_t>(written);
            continue;
        }

        if (written < 0 && errno == EINTR) {
            continue;
        }

        if (written == 0) {
            throw std::runtime_error(
                "serial write returned zero bytes on " +
                config_.device);
        }

        const int error_number = errno;
        throw std::runtime_error(
            "serial write failed on " + config_.device +
            ": " + std::strerror(error_number));
    }

#endif  // __linux__
}

}  // namespace hnu25::vest_runtime
