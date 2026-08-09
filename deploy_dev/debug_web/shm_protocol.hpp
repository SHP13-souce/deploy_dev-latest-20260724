#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace debug_web {

inline constexpr char kMagic[8] = {'H', 'N', 'U', 'D', 'B', 'G', '1', '\0'};
inline constexpr std::uint32_t kVersion = 1;
inline constexpr std::size_t kJpegCapacity = 8 * 1024 * 1024;

struct alignas(8) ShmHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t header_size;
    std::uint64_t capacity;
    std::atomic<std::uint64_t> sequence;
    std::uint64_t timestamp_ns;
    std::uint32_t jpeg_size;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t reserved;
};

static_assert(sizeof(ShmHeader) == 56, "shared-memory header layout changed");
static_assert(offsetof(ShmHeader, sequence) == 24, "Python protocol expects this offset");

}  // namespace debug_web
