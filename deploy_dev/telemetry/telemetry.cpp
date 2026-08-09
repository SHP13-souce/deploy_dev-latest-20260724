#include "telemetry.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace telemetry {
namespace {

void number(std::ostringstream& out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "null";
}

void escaped(std::ostringstream& out, const std::string& value) {
    out << '"';
    for (const unsigned char c : value) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\b': out << "\\b"; break;
            case '\f': out << "\\f"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c) << std::dec << std::setfill(' ');
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
}

}  // namespace

std::string encodeJson(const FrameSample& s) {
    std::ostringstream out;
    out << std::boolalpha << std::setprecision(10)
        << "{\"frame\":{\"index\":" << s.frame_index << ",\"video_time_s\":";
    number(out, s.video_time_s);
    out << ",\"timeline_s\":"; number(out, s.timeline_s);
    out << ",\"dt_s\":"; number(out, s.dt_s);
    out << ",\"pose_valid\":" << s.pose_valid
        << ",\"timeline_reset\":" << s.timeline_reset
        << "},\"detection\":{\"count\":" << s.detection_count
        << ",\"confidence_max\":"; number(out, s.detection_confidence_max);
    out << "},\"pnp\":{\"valid_count\":" << s.pnp_valid_count
        << ",\"rejected_count\":" << s.pnp_rejected_count
        << ",\"reprojection_mean_px\":"; number(out, s.reprojection_mean_px);
    out << ",\"reprojection_max_px\":"; number(out, s.reprojection_max_px);
    out << "},\"tracker\":{\"state_code\":" << s.tracker_state_code
        << ",\"state_name\":"; escaped(out, s.tracker_state_name);
    out << ",\"target_valid\":" << s.target_valid << ",\"nis\":"; number(out, s.nis);
    out << ",\"nis_failure_ratio\":"; number(out, s.nis_failure_ratio);
    out << "},\"state\":{\"x\":"; number(out, s.state[0]);
    out << ",\"vx\":"; number(out, s.state[1]);
    out << ",\"y\":"; number(out, s.state[2]);
    out << ",\"vy\":"; number(out, s.state[3]);
    out << ",\"z\":"; number(out, s.state[4]);
    out << ",\"vz\":"; number(out, s.state[5]);
    out << ",\"yaw\":"; number(out, s.state[6]);
    out << ",\"yaw_rate\":"; number(out, s.state[7]);
    out << ",\"radius_1\":"; number(out, s.state[8]);
    out << ",\"radius_delta\":"; number(out, s.state[9]);
    out << ",\"height_delta\":"; number(out, s.state[10]);
    out << "},\"aim\":{\"valid\":" << s.aim_valid << ",\"fire\":" << s.fire
        << ",\"yaw_rad\":"; number(out, s.aim_yaw_rad);
    out << ",\"pitch_rad\":"; number(out, s.aim_pitch_rad);
    out << ",\"fly_time_s\":"; number(out, s.fly_time_s);
    out << "},\"planner\":{\"valid\":" << s.planner_valid
        << ",\"solver_status\":" << s.planner_solver_status
        << ",\"reference_yaw_rad\":"; number(out, s.planner_reference_yaw_rad);
    out << ",\"reference_pitch_rad\":"; number(out, s.planner_reference_pitch_rad);
    out << ",\"yaw_rad\":"; number(out, s.planner_yaw_rad);
    out << ",\"pitch_rad\":"; number(out, s.planner_pitch_rad);
    out << ",\"yaw_velocity_rad_s\":"; number(out, s.planner_yaw_velocity_rad_s);
    out << ",\"pitch_velocity_rad_s\":"; number(out, s.planner_pitch_velocity_rad_s);
    out << ",\"yaw_acceleration_rad_s2\":"; number(out, s.planner_yaw_acceleration_rad_s2);
    out << ",\"pitch_acceleration_rad_s2\":"; number(out, s.planner_pitch_acceleration_rad_s2);
    out << "},\"fire_gate\":{\"allowed\":" << s.fire << ",\"reason\":";
    escaped(out, s.fire_reason);
    out << ",\"yaw_error_rad\":"; number(out, s.fire_yaw_error_rad);
    out << ",\"pitch_error_rad\":"; number(out, s.fire_pitch_error_rad);
    out << ",\"yaw_tolerance_rad\":"; number(out, s.fire_yaw_tolerance_rad);
    out << ",\"pitch_tolerance_rad\":"; number(out, s.fire_pitch_tolerance_rad);
    out << "},\"gimbal\":{\"feedback_valid\":" << s.gimbal_feedback_valid
        << ",\"yaw_deg\":"; number(out, s.gimbal_yaw_deg);
    out << ",\"pitch_deg\":"; number(out, s.gimbal_pitch_deg);
    out << ",\"yaw_speed\":"; number(out, s.gimbal_yaw_speed);
    out << ",\"pitch_speed\":"; number(out, s.gimbal_pitch_speed);
    out << "},\"latency_ms\":{\"detection\":"; number(out, s.detection_ms);
    out << ",\"pnp\":"; number(out, s.pnp_ms);
    out << ",\"tracker\":"; number(out, s.tracker_ms);
    out << ",\"aim\":"; number(out, s.aim_ms);
    out << ",\"total\":"; number(out, s.total_ms);
    out << "}}";
    return out.str();
}

struct UdpPublisher::Impl {
    Config config;
    int socket = -1;
    sockaddr_in destination{};
    double last_publish_s = -std::numeric_limits<double>::infinity();

    explicit Impl(Config value) : config(std::move(value)) {
        if (!config.enabled || config.port == 0 || !(config.publish_hz > 0.0)) return;
        socket = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket < 0) return;
        const int flags = ::fcntl(socket, F_GETFL, 0);
        if (flags < 0 || ::fcntl(socket, F_SETFL, flags | O_NONBLOCK) < 0) {
            ::close(socket);
            socket = -1;
            return;
        }
        destination.sin_family = AF_INET;
        destination.sin_port = htons(config.port);
        if (::inet_pton(AF_INET, config.address.c_str(), &destination.sin_addr) != 1) {
            ::close(socket);
            socket = -1;
        }
    }

    ~Impl() {
        if (socket >= 0) ::close(socket);
    }
};

UdpPublisher::UdpPublisher(Config config) noexcept {
    try { impl_ = std::make_unique<Impl>(std::move(config)); } catch (...) {}
}

UdpPublisher::~UdpPublisher() = default;
UdpPublisher::UdpPublisher(UdpPublisher&&) noexcept = default;
UdpPublisher& UdpPublisher::operator=(UdpPublisher&&) noexcept = default;

bool UdpPublisher::ready() const noexcept { return impl_ && impl_->socket >= 0; }

void UdpPublisher::publish(const FrameSample& sample) noexcept {
    try {
        if (!ready()) return;
        const double period_s = 1.0 / impl_->config.publish_hz;
        if (sample.timeline_s + 1e-9 < impl_->last_publish_s + period_s) return;
        const std::string payload = encodeJson(sample);
        ::sendto(impl_->socket, payload.data(), payload.size(), 0,
                 reinterpret_cast<const sockaddr*>(&impl_->destination),
                 sizeof(impl_->destination));
        impl_->last_publish_s = sample.timeline_s;
    } catch (...) {
        // Telemetry is best-effort and must never affect replay processing.
    }
}

}  // namespace telemetry
