#pragma once

#include "anti_drone/diagnostic_pipeline.hpp"

#include <cstdint>
#include <ostream>

namespace hnu25::anti_drone {

// Human-readable name for a tracking state. Unknown enum values (future
// extensions) map to "UNKNOWN" rather than throwing.
const char* trackStateName(TrackState state) noexcept;

// Writes the fixed CSV header line, terminated by '\n'. Does not alter the
// stream's formatting flags or precision.
void writeDiagnosticCsvHeader(std::ostream& out);

// Writes one CSV data row. This is serialization only — it never recomputes
// any xyz / yaw / pitch and never reads the valid flags to drop columns. The
// stream's formatting flags and precision are restored on exit. Throws
// std::invalid_argument if timestamp_s or any written double is non-finite.
void writeDiagnosticCsvRow(
    std::ostream& out,
    std::uint64_t frame_index,
    double timestamp_s,
    const DiagnosticFrameResult& result);

}  // namespace hnu25::anti_drone
