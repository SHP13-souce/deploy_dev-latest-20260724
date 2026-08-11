#include "vest_runtime/serial_observation_sink.hpp"
#include "vest_runtime/vision_protocol.hpp"

#include <utility>

namespace hnu25::vest_runtime {

SerialObservationSink::SerialObservationSink(SerialPortConfig config)
    : serial_(std::move(config)) {}

void SerialObservationSink::onObservation(
    const hnu25::vest::VisionTargetObservation& observation) {
    const auto packet = encodeVisionObservation(observation);

    serial_.writeAll(packet.data(), packet.size());
}

}  // namespace hnu25::vest_runtime
