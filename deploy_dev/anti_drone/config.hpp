#pragma once

#include "anti_drone/traditional_detector.hpp"

#include <filesystem>

namespace hnu25::anti_drone {

// Loads a TraditionalDetectorConfig from a YAML file. The file must contain a
// top-level "traditional_detector:" mapping; keys absent from the file keep
// the C++ struct defaults. Invalid values throw std::runtime_error.
TraditionalDetectorConfig loadTraditionalDetectorConfig(
    const std::filesystem::path& path);

}  // namespace hnu25::anti_drone
