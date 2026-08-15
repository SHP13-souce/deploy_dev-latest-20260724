#include "anti_drone/diagnostic_csv.hpp"

#include <cmath>
#include <stdexcept>

namespace hnu25::anti_drone {

namespace {

bool finite(const cv::Vec3d& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

// The logger does not judge algorithm validity, but it must never serialize a
// non-finite double (which would corrupt the CSV for downstream pandas/Excel).
void validateResultFinite(const DiagnosticFrameResult& result) {
    if (!finite(result.tracked_position_gimbal_m) ||
        !finite(result.velocity_gimbal_m_s) ||
        !finite(result.predicted_position_gimbal_m) ||
        !finite(result.compensated_position_gimbal_m) ||
        !std::isfinite(result.prediction_horizon_s) ||
        !std::isfinite(result.predicted_yaw_rad) ||
        !std::isfinite(result.predicted_pitch_rad)) {
        throw std::invalid_argument(
            "DiagnosticFrameResult contains a non-finite value");
    }
}

void writeBool(std::ostream& out, bool value) {
    out << (value ? '1' : '0');
}

}  // namespace

const char* trackStateName(TrackState state) noexcept {
    switch (state) {
        case TrackState::LOST:
            return "LOST";
        case TrackState::DETECTING:
            return "DETECTING";
        case TrackState::TRACKING:
            return "TRACKING";
        case TrackState::TEMP_LOST:
            return "TEMP_LOST";
        default:
            return "UNKNOWN";
    }
}

void writeDiagnosticCsvHeader(std::ostream& out) {
    out << "frame_index,timestamp_s,track_available,track_state,"
           "measurement_updated,consecutive_detects,missed_count,"
           "tracked_x_m,tracked_y_m,tracked_z_m,"
           "velocity_x_m_s,velocity_y_m_s,velocity_z_m_s,"
           "prediction_valid,prediction_horizon_s,"
           "predicted_x_m,predicted_y_m,predicted_z_m,"
           "solution_valid,compensated_x_m,compensated_y_m,compensated_z_m,"
           "predicted_yaw_rad,predicted_pitch_rad\n";
}

void writeDiagnosticCsvRow(
    std::ostream& out,
    std::uint64_t frame_index,
    double timestamp_s,
    const DiagnosticFrameResult& result) {
    if (!std::isfinite(timestamp_s)) {
        throw std::invalid_argument("timestamp_s must be finite");
    }
    validateResultFinite(result);

    // Preserve the caller's formatting so the logger never leaks precision or
    // float flags into the surrounding stream.
    const auto old_flags = out.flags();
    const auto old_precision = out.precision();
    out.precision(17);

    out << frame_index << ',' << timestamp_s << ',';
    writeBool(out, result.track_available);
    out << ',' << trackStateName(result.track_state) << ',';
    writeBool(out, result.measurement_updated);
    out << ',' << result.consecutive_detects << ',' << result.missed_count << ',';
    out << result.tracked_position_gimbal_m[0] << ','
        << result.tracked_position_gimbal_m[1] << ','
        << result.tracked_position_gimbal_m[2] << ',';
    out << result.velocity_gimbal_m_s[0] << ','
        << result.velocity_gimbal_m_s[1] << ','
        << result.velocity_gimbal_m_s[2] << ',';
    writeBool(out, result.prediction_valid);
    out << ',' << result.prediction_horizon_s << ',';
    out << result.predicted_position_gimbal_m[0] << ','
        << result.predicted_position_gimbal_m[1] << ','
        << result.predicted_position_gimbal_m[2] << ',';
    writeBool(out, result.solution_valid);
    out << ',' << result.compensated_position_gimbal_m[0] << ','
        << result.compensated_position_gimbal_m[1] << ','
        << result.compensated_position_gimbal_m[2] << ',';
    out << result.predicted_yaw_rad << ',' << result.predicted_pitch_rad << '\n';

    out.flags(old_flags);
    out.precision(old_precision);
}

}  // namespace hnu25::anti_drone
