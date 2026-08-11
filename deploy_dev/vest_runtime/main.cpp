#include "vest_runtime/runtime_config.hpp"
#include "vest_runtime/vest_runtime.hpp"

#include <atomic>
#include <csignal>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::atomic_bool g_stop_requested{false};

static_assert(
    std::atomic_bool::is_always_lock_free,
    "vest_runtime requires lock-free atomic_bool for signal handling");

void handleSignal(int) noexcept {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

class NoopObservationSink final
    : public hnu25::vest_runtime::ObservationSink {
public:
    void onObservation(
        const hnu25::vest::VisionTargetObservation&) override {}
};

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string config_path = "config/vest.yaml";

        if (argc == 3 && std::string(argv[1]) == "--config") {
            config_path = argv[2];
        } else if (argc != 1) {
            std::cerr << "Usage: vest_runtime [--config <path>]\n";
            return 2;
        }

        if (std::signal(SIGINT, handleSignal) == SIG_ERR ||
            std::signal(SIGTERM, handleSignal) == SIG_ERR) {
            throw std::runtime_error(
                "failed to install signal handler");
        }

        auto config =
            hnu25::vest_runtime::RuntimeConfig::loadFromFile(
                config_path);

        std::cerr << "[vest_runtime] config=" << config_path << '\n';

        hnu25::vest_runtime::VestRuntime runtime(std::move(config));

        NoopObservationSink sink;

        runtime.run(g_stop_requested, sink);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[vest_runtime] fatal: " << e.what() << '\n';
        return 1;
    }
}
