#pragma once

#include "detection/detector.hpp"
#include "sp_core/sp_core.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace hnu25::standard {

struct RuntimeParameters {
    float conf_threshold = 0.25F;
    int binary_threshold = 120;
    double bullet_speed = 23.0;
    sp_core::AimerConfig aimer;
};

struct TuningRequest {
    std::uint64_t id = 0;
    bool save = false;
    RuntimeParameters parameters;
};

class RuntimeTuning {
public:
    RuntimeTuning(std::filesystem::path config_path,
                  std::string shm_name,
                  RuntimeParameters initial);

    std::optional<TuningRequest> poll();
    void acknowledge(const TuningRequest& request,
                     const RuntimeParameters& active,
                     bool ok,
                     const std::string& message);
    void save(const RuntimeParameters& parameters) const;
    const std::filesystem::path& requestPath() const { return request_path_; }

private:
    void writeState(const RuntimeParameters& parameters,
                    std::uint64_t id,
                    bool saved,
                    bool ok,
                    const std::string& message) const;

    std::filesystem::path config_path_;
    std::filesystem::path request_path_;
    std::filesystem::path state_path_;
    RuntimeParameters initial_;
    std::uint64_t last_id_ = 0;
};

}  // namespace hnu25::standard
