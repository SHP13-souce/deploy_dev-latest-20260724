#include "anti_drone/diagnostic_csv.hpp"
#include "anti_drone/vision_telemetry.hpp"
#include "anti_drone/vision_telemetry_stream.hpp"

#include "anti_drone/serial_port.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Read-only diagnostic receiver for the anti-drone VisionTelemetry serial
// stream. It opens a serial device, feeds every read chunk into a
// VisionTelemetryStreamParser, and prints each decoded packet.
//
// This tool is deliberately read-only: it never issues any control, fire, or
// actuation command, and it never touches any hardware path. It only decodes
// and displays the incoming vision-result stream.
//
// Usage: anti_drone_vision_telemetry_receiver <device> [baud_rate]
//   baud_rate defaults to 115200.

namespace {

std::atomic<bool> g_stop_requested{false};

void handleSignal(int) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

void printTelemetry(const hnu25::anti_drone::VisionTelemetry& telemetry) {
    std::cout << "seq=" << telemetry.sequence
              << " ts_us=" << telemetry.timestamp_us
              << " vision_valid=" << (telemetry.vision_valid ? "true" : "false")
              << " track_state="
              << hnu25::anti_drone::trackStateName(telemetry.track_state);
    if (telemetry.vision_valid) {
        std::cout << std::fixed << std::setprecision(4)
                  << " yaw_rad=" << telemetry.yaw_rad
                  << " pitch_rad=" << telemetry.pitch_rad
                  << " x_m=" << telemetry.x_m
                  << " y_m=" << telemetry.y_m
                  << " z_m=" << telemetry.z_m;
    } else {
        std::cout << " vision_result: INVALID";
    }
    std::cout << " detections=" << telemetry.detection_count
              << " pnp_measurements=" << telemetry.pnp_measurement_count
              << '\n';
}

void printParserStats(
    const hnu25::anti_drone::VisionTelemetryStreamStats& stats) {
    std::cout << "[parser] bytes_received=" << stats.bytes_received
              << " packets_received=" << stats.packets_received
              << " packets_valid=" << stats.packets_valid
              << " crc_or_format_errors=" << stats.crc_or_format_errors
              << " discarded_bytes=" << stats.discarded_bytes
              << " sequence_gaps=" << stats.sequence_gaps
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: anti_drone_vision_telemetry_receiver "
                     "<device> [baud_rate]\n";
        return 1;
    }

    const std::string device = argv[1];
    int baud_rate = 115200;
    if (argc == 3) {
        baud_rate = std::atoi(argv[2]);
    }

    hnu25::SerialPort serial;
    if (!serial.open(device, baud_rate)) {
        std::cerr << "Failed to open serial device '" << device << "'\n";
        return 2;
    }
    std::cout << "Opened " << device << " @ " << baud_rate << " baud\n";

    std::signal(SIGINT, handleSignal);
    std::signal(SIGTERM, handleSignal);

    hnu25::anti_drone::VisionTelemetryStreamParser parser;
    std::uint64_t packets_printed = 0;

    while (!g_stop_requested.load(std::memory_order_relaxed)) {
        const std::vector<std::uint8_t> bytes = serial.read(256);
        if (!bytes.empty()) {
            parser.push(bytes);
        }

        hnu25::anti_drone::VisionTelemetry telemetry;
        while (parser.pop(telemetry)) {
            ++packets_printed;
            printTelemetry(telemetry);
            if (packets_printed % 50 == 0) {
                printParserStats(parser.stats());
            }
        }

        if (bytes.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::cout << "Stopping receiver...\n";
    printParserStats(parser.stats());
    serial.close();
    std::cout << "Receiver exited cleanly.\n";
    return 0;
}
