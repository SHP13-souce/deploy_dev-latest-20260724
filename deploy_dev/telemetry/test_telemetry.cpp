#include "telemetry/telemetry.hpp"

#include <cmath>
#include <iostream>
#include <string>

int main() {
    telemetry::FrameSample sample;
    sample.frame_index = 42;
    sample.video_time_s = 1.25;
    sample.detection_count = 2;
    sample.tracker_state_code = 2;
    sample.tracker_state_name = "tracking\nlocked";
    sample.target_valid = true;
    sample.state[7] = 3.5;
    sample.aim_yaw_rad = std::nan("");
    sample.fire_reason = "no_feedback";
    sample.planner_valid = true;
    sample.planner_solver_status = 1;
    sample.planner_yaw_acceleration_rad_s2 = 12.5;
    sample.total_ms = 12.75;

    const std::string json = telemetry::encodeJson(sample);
    const char* expected[] = {
        "\"frame\":{\"index\":42", "\"detection\":{\"count\":2",
        "\"state_code\":2", "\"state_name\":\"tracking\\nlocked\"",
         "\"target_valid\":true", "\"yaw_rate\":3.5", "\"yaw_rad\":null",
         "\"planner\":{\"valid\":true,\"solver_status\":1",
         "\"yaw_acceleration_rad_s2\":12.5",
         "\"fire_gate\":{\"allowed\":false,\"reason\":\"no_feedback\"",
        "\"total\":12.75"};
    for (const char* value : expected) {
        if (json.find(value) == std::string::npos) {
            std::cerr << "missing JSON fragment: " << value << '\n' << json << '\n';
            return 1;
        }
    }
    if (json.empty() || json.front() != '{' || json.back() != '}') return 1;

    telemetry::UdpPublisher disabled({false, "not-an-address", 0, 0.0});
    disabled.publish(sample);
    if (disabled.ready()) return 1;
    return 0;
}
