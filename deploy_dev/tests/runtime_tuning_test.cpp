#include "pipeline/runtime_tuning.hpp"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <cmath>
#include <iostream>
#include <unistd.h>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main() {
    const std::string suffix = std::to_string(getpid());
    const auto config_path = std::filesystem::temp_directory_path() /
        ("hnu_runtime_tuning_" + suffix + ".yaml");
    {
        std::ofstream output(config_path);
        output << "detector:\n  conf_threshold: 0.2\n"
                  "web:\n  binary_threshold: 120\n"
                  "bullet_speed: 23\n"
                  "yaw_offset: -1\npitch_offset: -1.4\n"
                  "comming_angle: 60\nleaving_angle: 20\n"
                  "decision_speed: 8\nhigh_speed_delay_time: 0.03\n"
                  "low_speed_delay_time: 0.015\ncommand_enabled: false\n";
    }
    hnu25::standard::RuntimeParameters initial;
    initial.conf_threshold = 0.2F;
    initial.binary_threshold = 120;
    initial.bullet_speed = 23.0;
    const std::string shm = "/hnu_runtime_tuning_" + suffix;
    hnu25::standard::RuntimeTuning tuning(config_path, shm, initial);
    {
        std::ofstream request(tuning.requestPath());
        request << "id: 1\nsave: false\nconf_threshold: 0.3\nbinary_threshold: 140\n"
                   "bullet_speed: 24\nyaw_offset: 0.5\npitch_offset: -0.5\n"
                   "comming_angle: 55\nleaving_angle: 18\ndecision_speed: 7\n"
                   "high_speed_delay_time: 0.02\nlow_speed_delay_time: 0.01\n";
    }
    auto request = tuning.poll();
    require(request && !request->save && request->parameters.binary_threshold == 140,
            "temporary request parsed");
    tuning.acknowledge(*request, request->parameters, true, "temporarily applied");
    require(YAML::LoadFile(config_path.string())["detector"]["conf_threshold"].as<double>() == 0.2,
            "temporary apply does not modify YAML");

    request->id = 2;
    request->save = true;
    tuning.save(request->parameters);
    tuning.acknowledge(*request, request->parameters, true, "applied and saved");
    const YAML::Node saved = YAML::LoadFile(config_path.string());
    require(std::abs(saved["detector"]["conf_threshold"].as<double>() - 0.3) < 1e-6,
            "save updates allowlisted value");
    require(!saved["command_enabled"].as<bool>(), "save preserves safety setting");

    std::filesystem::remove(config_path);
    std::filesystem::remove("/tmp/hnu_runtime_tuning_" + suffix + ".tuning_state.json");
    std::filesystem::remove(tuning.requestPath());
    return 0;
}
