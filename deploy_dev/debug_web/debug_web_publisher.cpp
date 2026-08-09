#include "debug_web/debug_web_publisher.hpp"

#include "debug_web/shm_protocol.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <new>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace debug_web {
namespace {

std::string normalizeShmName(std::string name) {
    if (name.empty()) name = "/hnu_vision_debug";
    if (name.front() != '/') name.insert(name.begin(), '/');
    std::replace(name.begin() + 1, name.end(), '/', '_');
    return name;
}

}  // namespace

std::string statePathForShm(const std::string& shm_name) {
    std::string safe = normalizeShmName(shm_name).substr(1);
    return "/tmp/" + safe + ".json";
}

struct DebugWebPublisher::Impl {
    explicit Impl(Config value) : config(std::move(value)) {
        if (!config.enabled) return;
        config.shm_name = normalizeShmName(config.shm_name);
        config.quality = std::clamp(config.quality, 1, 100);
        const std::size_t mapping_size = sizeof(ShmHeader) + kJpegCapacity;
        fd = shm_open(config.shm_name.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd < 0 || ftruncate(fd, static_cast<off_t>(mapping_size)) != 0) return;
        mapping = mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapping == MAP_FAILED) {
            mapping = nullptr;
            return;
        }
        header = new (mapping) ShmHeader{};
        std::memcpy(header->magic, kMagic, sizeof(kMagic));
        header->version = kVersion;
        header->header_size = sizeof(ShmHeader);
        header->capacity = kJpegCapacity;
        header->sequence.store(0, std::memory_order_release);
        ready = true;
    }

    ~Impl() {
        if (mapping) munmap(mapping, sizeof(ShmHeader) + kJpegCapacity);
        if (fd >= 0) close(fd);
    }

    void publish(const cv::Mat& frame, const std::string& state_json) {
        if (!ready || frame.empty()) return;
        const auto now = std::chrono::steady_clock::now();
        if (config.publish_hz > 0.0 && last_publish.time_since_epoch().count() != 0 &&
            std::chrono::duration<double>(now - last_publish).count() < 1.0 / config.publish_hz) return;

        std::vector<unsigned char> jpeg;
        if (!cv::imencode(".jpg", frame, jpeg, {cv::IMWRITE_JPEG_QUALITY, config.quality}) ||
            jpeg.size() > kJpegCapacity) return;

        const std::uint64_t odd = header->sequence.fetch_add(1, std::memory_order_acq_rel) + 1;
        header->timestamp_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
        header->jpeg_size = static_cast<std::uint32_t>(jpeg.size());
        header->width = static_cast<std::uint32_t>(frame.cols);
        header->height = static_cast<std::uint32_t>(frame.rows);
        std::memcpy(static_cast<unsigned char*>(mapping) + sizeof(ShmHeader), jpeg.data(), jpeg.size());
        std::atomic_thread_fence(std::memory_order_release);
        header->sequence.store(odd + 1, std::memory_order_release);

        const std::filesystem::path final_path(statePathForShm(config.shm_name));
        const auto temporary = final_path.string() + ".tmp." + std::to_string(getpid());
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            output << state_json;
            output.flush();
            if (!output) {
                std::filesystem::remove(temporary);
                return;
            }
        }
        std::filesystem::rename(temporary, final_path);
        last_publish = now;
    }

    bool due() const {
        if (!ready) return false;
        if (!(config.publish_hz > 0.0) || last_publish.time_since_epoch().count() == 0) return true;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - last_publish).count()
            >= 1.0 / config.publish_hz;
    }

    Config config;
    int fd = -1;
    void* mapping = nullptr;
    ShmHeader* header = nullptr;
    bool ready = false;
    std::chrono::steady_clock::time_point last_publish{};
};

DebugWebPublisher::DebugWebPublisher(Config config) noexcept {
    try { impl_ = std::make_unique<Impl>(std::move(config)); } catch (...) {}
}
DebugWebPublisher::~DebugWebPublisher() = default;
DebugWebPublisher::DebugWebPublisher(DebugWebPublisher&&) noexcept = default;
DebugWebPublisher& DebugWebPublisher::operator=(DebugWebPublisher&&) noexcept = default;
void DebugWebPublisher::publish(const cv::Mat& frame, const std::string& state_json) noexcept {
    try { if (impl_) impl_->publish(frame, state_json); } catch (...) {}
}
bool DebugWebPublisher::ready() const noexcept { return impl_ && impl_->ready; }
bool DebugWebPublisher::due() const noexcept {
    try { return impl_ && impl_->due(); } catch (...) { return false; }
}

}  // namespace debug_web
