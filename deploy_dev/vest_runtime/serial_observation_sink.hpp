#pragma once

#include "vest_runtime/serial_port.hpp"
#include "vest_runtime/vest_runtime.hpp"

namespace hnu25::vest_runtime {

class SerialObservationSink final : public ObservationSink {
public:
    explicit SerialObservationSink(SerialPortConfig config);

    void onObservation(
        const hnu25::vest::VisionTargetObservation& observation) override;

private:
    SerialPort serial_;
};

}  // namespace hnu25::vest_runtime
