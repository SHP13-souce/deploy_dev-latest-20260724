#include "debug_web/debug_web_publisher.hpp"
#include "debug_web/shm_protocol.hpp"

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

int main() {
    const std::string name = "/hnu_debug_web_test_" + std::to_string(getpid());
    debug_web::Config config;
    config.enabled = true;
    config.shm_name = name;
    config.quality = 75;
    config.publish_hz = 0.0;
    debug_web::DebugWebPublisher publisher(config);
    if (!publisher.ready()) return 1;
    publisher.publish(cv::Mat(48, 64, CV_8UC3, cv::Scalar(20, 80, 160)), "{\"ok\":true}");

    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    const std::size_t size = sizeof(debug_web::ShmHeader) + debug_web::kJpegCapacity;
    void* mapping = fd >= 0 ? mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
    if (mapping == MAP_FAILED) return 2;
    const auto* header = static_cast<const debug_web::ShmHeader*>(mapping);
    const auto before = header->sequence.load(std::memory_order_acquire);
    const auto* jpeg = static_cast<const unsigned char*>(mapping) + header->header_size;
    const bool valid = std::memcmp(header->magic, debug_web::kMagic, 8) == 0 &&
        before != 0 && (before & 1U) == 0 && header->version == debug_web::kVersion &&
        header->width == 64 && header->height == 48 && header->jpeg_size > 4 &&
        jpeg[0] == 0xff && jpeg[1] == 0xd8 && jpeg[header->jpeg_size - 2] == 0xff &&
        jpeg[header->jpeg_size - 1] == 0xd9 &&
        before == header->sequence.load(std::memory_order_acquire);
    munmap(mapping, size);
    close(fd);
    shm_unlink(name.c_str());
    unlink(debug_web::statePathForShm(name).c_str());
    if (!valid) std::cerr << "invalid shared-memory snapshot\n";
    return valid ? 0 : 3;
}
