#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace telemetry {

struct Config {
    bool enabled = true;
    std::string address = "127.0.0.1";
    std::uint16_t port = 9870;
    double publish_hz = 30.0;
};

struct FrameSample {
    std::uint64_t frame_index = 0;
    double video_time_s = 0.0;
    double timeline_s = 0.0;
    double dt_s = 0.0;
    bool pose_valid = false;
    bool timeline_reset = false;

    int detection_count = 0;
    double detection_confidence_max = 0.0;
    int pnp_valid_count = 0;
    int pnp_rejected_count = 0;
    double reprojection_mean_px = 0.0;
    double reprojection_max_px = 0.0;

    int tracker_state_code = 0;
    std::string tracker_state_name = "lost";
    bool target_valid = false;
    std::array<double, 11> state{};
    double nis = 0.0;
    double nis_failure_ratio = 0.0;

    bool aim_valid = false;
    bool fire = false;
    std::string fire_reason = "disabled";
    double fire_yaw_error_rad = 0.0;
    double fire_pitch_error_rad = 0.0;
    double fire_yaw_tolerance_rad = 0.0;
    double fire_pitch_tolerance_rad = 0.0;
    double aim_yaw_rad = 0.0;
    double aim_pitch_rad = 0.0;
    double fly_time_s = 0.0;

    bool planner_valid = false;
    int planner_solver_status = 0;
    double planner_reference_yaw_rad = 0.0;
    double planner_reference_pitch_rad = 0.0;
    double planner_yaw_rad = 0.0;
    double planner_pitch_rad = 0.0;
    double planner_yaw_velocity_rad_s = 0.0;
    double planner_pitch_velocity_rad_s = 0.0;
    double planner_yaw_acceleration_rad_s2 = 0.0;
    double planner_pitch_acceleration_rad_s2 = 0.0;

    bool gimbal_feedback_valid = false;
    double gimbal_yaw_deg = 0.0;
    double gimbal_pitch_deg = 0.0;
    double gimbal_yaw_speed = 0.0;
    double gimbal_pitch_speed = 0.0;

    double detection_ms = 0.0;
    double pnp_ms = 0.0;
    double tracker_ms = 0.0;
    double aim_ms = 0.0;
    double total_ms = 0.0;
};

std::string encodeJson(const FrameSample& sample);

class UdpPublisher {
public:
    explicit UdpPublisher(Config config) noexcept;
    ~UdpPublisher();
    UdpPublisher(UdpPublisher&&) noexcept;
    UdpPublisher& operator=(UdpPublisher&&) noexcept;
    UdpPublisher(const UdpPublisher&) = delete;
    UdpPublisher& operator=(const UdpPublisher&) = delete;

    void publish(const FrameSample& sample) noexcept;
    bool ready() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace telemetry
