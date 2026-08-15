#include "anti_drone/diagnostic_csv.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

bool approx(double a, double b, double eps = 1e-9) {
    return std::fabs(a - b) <= eps;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> fields;
    std::size_t start = 0;
    while (true) {
        const std::size_t pos = s.find(delim, start);
        if (pos == std::string::npos) {
            fields.push_back(s.substr(start));
            break;
        }
        fields.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return fields;
}

// Removes a single trailing '\n' so the remaining text can be comma-split
// without a stray newline glued to the last column.
std::string stripTrailingNewline(std::string s) {
    if (!s.empty() && s.back() == '\n') {
        s.pop_back();
    }
    return s;
}

using hnu25::anti_drone::DiagnosticFrameResult;
using hnu25::anti_drone::TrackState;
using hnu25::anti_drone::trackStateName;
using hnu25::anti_drone::writeDiagnosticCsvHeader;
using hnu25::anti_drone::writeDiagnosticCsvRow;

const char* kHeader =
    "frame_index,timestamp_s,track_available,track_state,"
    "measurement_updated,consecutive_detects,missed_count,"
    "tracked_x_m,tracked_y_m,tracked_z_m,"
    "velocity_x_m_s,velocity_y_m_s,velocity_z_m_s,"
    "prediction_valid,prediction_horizon_s,"
    "predicted_x_m,predicted_y_m,predicted_z_m,"
    "solution_valid,compensated_x_m,compensated_y_m,compensated_z_m,"
    "predicted_yaw_rad,predicted_pitch_rad";

// Test 1: the header string matches the fixed spec exactly and ends with '\n'.
void testHeaderExact() {
    std::ostringstream out;
    writeDiagnosticCsvHeader(out);
    const std::string expected = std::string(kHeader) + '\n';
    check(out.str() == expected, "header string exactly matches");
}

// Test 2: a confirmed TRACKING frame serializes to one 24-column row with the
// expected key values.
void testTrackingRow() {
    DiagnosticFrameResult r;
    r.track_available = true;
    r.prediction_valid = true;
    r.solution_valid = true;
    r.track_state = TrackState::TRACKING;
    r.measurement_updated = true;
    r.consecutive_detects = 4;
    r.missed_count = 0;
    r.tracked_position_gimbal_m = cv::Vec3d(4.0, 1.0, 0.5);
    r.velocity_gimbal_m_s = cv::Vec3d(0.0, 3.0, 0.0);
    r.prediction_horizon_s = 0.08;
    r.predicted_position_gimbal_m = cv::Vec3d(4.0, 1.24, 0.5);
    r.compensated_position_gimbal_m = cv::Vec3d(4.1, 1.2, 0.52);
    r.predicted_yaw_rad = 0.3;
    r.predicted_pitch_rad = 0.1;

    std::ostringstream out;
    writeDiagnosticCsvRow(out, 42, 1.25, r);

    const std::string text = out.str();
    check(!text.empty() && text.back() == '\n', "row ends with newline");

    const auto cols = split(stripTrailingNewline(text), ',');
    check(cols.size() == 24, "row has exactly 24 columns");

    check(cols[0] == "42", "frame_index == 42");
    check(approx(std::stod(cols[1]), 1.25), "timestamp_s == 1.25");
    check(cols[2] == "1", "track_available == 1");
    check(cols[3] == "TRACKING", "track_state == TRACKING");
    check(cols[4] == "1", "measurement_updated == 1");
    check(approx(std::stod(cols[11]), 3.0), "velocity_y == 3");
    check(cols[13] == "1", "prediction_valid == 1");
    check(cols[18] == "1", "solution_valid == 1");
    check(approx(std::stod(cols[22]), 0.3), "yaw == 0.3");
    check(approx(std::stod(cols[23]), 0.1), "pitch == 0.1");
}

// Test 3: a DETECTING frame keeps 24 columns and writes the default predicted
// fields as 0 rather than leaving them empty.
void testDetectingRow() {
    DiagnosticFrameResult r;
    r.track_available = true;
    r.track_state = TrackState::DETECTING;
    // prediction_valid / solution_valid remain false.

    std::ostringstream out;
    writeDiagnosticCsvRow(out, 0, 0.0, r);

    const auto cols = split(stripTrailingNewline(out.str()), ',');
    check(cols.size() == 24, "row has exactly 24 columns");
    check(cols[3] == "DETECTING", "track_state == DETECTING");
    check(cols[13] == "0", "prediction_valid == 0");
    check(cols[18] == "0", "solution_valid == 0");
    check(cols[15] == "0", "predicted_x == 0 (not empty)");
    check(cols[16] == "0", "predicted_y == 0 (not empty)");
    check(cols[17] == "0", "predicted_z == 0 (not empty)");
}

// Test 4: a TEMP_LOST prediction-only frame records measurement_updated == 0
// and its missed_count.
void testTempLostRow() {
    DiagnosticFrameResult r;
    r.track_available = true;
    r.prediction_valid = true;
    r.solution_valid = true;
    r.track_state = TrackState::TEMP_LOST;
    r.measurement_updated = false;
    r.missed_count = 3;
    r.tracked_position_gimbal_m = cv::Vec3d(4.0, 1.0, 0.5);
    r.velocity_gimbal_m_s = cv::Vec3d(0.0, 3.0, 0.0);
    r.prediction_horizon_s = 0.08;
    r.predicted_position_gimbal_m = cv::Vec3d(4.0, 1.24, 0.5);
    r.compensated_position_gimbal_m = cv::Vec3d(4.1, 1.2, 0.52);
    r.predicted_yaw_rad = 0.3;
    r.predicted_pitch_rad = 0.1;

    std::ostringstream out;
    writeDiagnosticCsvRow(out, 41, 0.82, r);

    const auto cols = split(stripTrailingNewline(out.str()), ',');
    check(cols.size() == 24, "row has exactly 24 columns");
    check(cols[3] == "TEMP_LOST", "track_state == TEMP_LOST");
    check(cols[4] == "0", "measurement_updated == 0");
    check(cols[6] == "3", "missed_count == 3");
}

// Test 5: a default-constructed (LOST) result serializes with all bools 0.
void testLostRow() {
    DiagnosticFrameResult r;  // all defaults.

    std::ostringstream out;
    writeDiagnosticCsvRow(out, 0, 0.0, r);

    const auto cols = split(stripTrailingNewline(out.str()), ',');
    check(cols.size() == 24, "row has exactly 24 columns");
    check(cols[3] == "LOST", "track_state == LOST");
    check(cols[2] == "0", "track_available == 0");
    check(cols[4] == "0", "measurement_updated == 0");
    check(cols[13] == "0", "prediction_valid == 0");
    check(cols[18] == "0", "solution_valid == 0");
}

// Test 6: non-finite timestamp_s is rejected.
void testInvalidTimestamp() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    DiagnosticFrameResult r;
    std::ostringstream out;

    bool threw_nan = false;
    try {
        writeDiagnosticCsvRow(out, 0, nan, r);
    } catch (const std::invalid_argument&) {
        threw_nan = true;
    }
    check(threw_nan, "NaN timestamp throws std::invalid_argument");

    bool threw_inf = false;
    try {
        writeDiagnosticCsvRow(out, 0, inf, r);
    } catch (const std::invalid_argument&) {
        threw_inf = true;
    }
    check(threw_inf, "+Inf timestamp throws std::invalid_argument");
}

// Test 7: non-finite result doubles are rejected.
void testInvalidResultValue() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    std::ostringstream out;

    {
        DiagnosticFrameResult r;
        r.velocity_gimbal_m_s[1] = nan;
        bool threw = false;
        try {
            writeDiagnosticCsvRow(out, 0, 0.0, r);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "NaN velocity throws std::invalid_argument");
    }
    {
        DiagnosticFrameResult r;
        r.predicted_yaw_rad = inf;
        bool threw = false;
        try {
            writeDiagnosticCsvRow(out, 0, 0.0, r);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "Inf yaw throws std::invalid_argument");
    }
}

// Test 8: the logger restores the caller's formatting flags and precision.
void testFormattingRestored() {
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    const auto flags_before = out.flags();
    const auto precision_before = out.precision();

    DiagnosticFrameResult r;
    writeDiagnosticCsvRow(out, 0, 0.0, r);

    check(out.flags() == flags_before, "flags restored");
    check(out.precision() == precision_before, "precision restored");
}

// Test 9: TrackState names, including the unknown-enum fallback.
void testTrackStateNames() {
    check(std::string(trackStateName(TrackState::LOST)) == "LOST", "LOST");
    check(std::string(trackStateName(TrackState::DETECTING)) == "DETECTING",
          "DETECTING");
    check(std::string(trackStateName(TrackState::TRACKING)) == "TRACKING",
          "TRACKING");
    check(std::string(trackStateName(TrackState::TEMP_LOST)) == "TEMP_LOST",
          "TEMP_LOST");
    check(std::string(trackStateName(static_cast<TrackState>(999))) == "UNKNOWN",
          "unknown -> UNKNOWN");
}

// Test 10: 1 header + 100 data rows, every data row exactly 24 columns.
void testMultiFrameCsv() {
    std::ostringstream out;
    writeDiagnosticCsvHeader(out);

    for (int i = 0; i < 100; ++i) {
        DiagnosticFrameResult r;
        r.track_available = true;
        r.prediction_valid = true;
        r.solution_valid = true;
        r.track_state = TrackState::TRACKING;
        r.measurement_updated = true;
        r.consecutive_detects = i;
        r.missed_count = 0;
        r.tracked_position_gimbal_m = cv::Vec3d(4.0, 0.01 * i, 0.5);
        r.velocity_gimbal_m_s = cv::Vec3d(0.0, 3.0, 0.0);
        r.prediction_horizon_s = 0.08;
        r.predicted_position_gimbal_m = cv::Vec3d(4.0, 0.01 * i + 0.24, 0.5);
        r.compensated_position_gimbal_m = r.predicted_position_gimbal_m;
        r.predicted_yaw_rad = 0.01 * i;
        r.predicted_pitch_rad = 0.1;
        writeDiagnosticCsvRow(out, static_cast<std::uint64_t>(i), i * 0.02, r);
    }

    const std::string all = out.str();
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (true) {
        const std::size_t pos = all.find('\n', start);
        if (pos == std::string::npos) {
            break;
        }
        lines.push_back(all.substr(start, pos - start));
        start = pos + 1;
    }

    check(lines.size() == 101, "1 header + 100 rows == 101 lines");
    check(split(lines[0], ',').size() == 24, "header has 24 columns");
    for (std::size_t i = 1; i < lines.size(); ++i) {
        const auto cols = split(lines[i], ',');
        check(cols.size() == 24, "data row has 24 columns");
    }
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"header exact", testHeaderExact},
        {"tracking row", testTrackingRow},
        {"detecting row", testDetectingRow},
        {"temp lost row", testTempLostRow},
        {"lost row", testLostRow},
        {"invalid timestamp", testInvalidTimestamp},
        {"invalid result value", testInvalidResultValue},
        {"formatting restored", testFormattingRestored},
        {"track state names", testTrackStateNames},
        {"multi-frame csv", testMultiFrameCsv},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone diagnostic csv tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
