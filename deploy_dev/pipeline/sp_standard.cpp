#include "communication/serial_port.hpp"
#include "aim/fire_gate.hpp"
#include "camera/frame_source.hpp"
#include "camera/opencv_frame_source.hpp"
#if HNU25_HAS_MVS
#include "camera/hik_frame_source.hpp"
#endif
#include "communication/sp_protocol.hpp"
#include "debug_web/debug_web_publisher.hpp"
#include "detection/detector.hpp"
#include "pipeline/sp_standard_config.hpp"
#include "pipeline/runtime_tuning.hpp"
#include "planner/mpc_planner.hpp"
#include "sp_core/sp_core.hpp"
#include "telemetry/telemetry.hpp"
#include "transform/time_pose_buffer.hpp"

#include <Eigen/Geometry>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <opencv2/imgproc.hpp>

namespace {

volatile std::sig_atomic_t stop_requested = 0;

void requestStop(int) {
    stop_requested = 1;
}

struct FeedbackSnapshot {
    hnu25::sp::GimbalFeedback value;
    bool valid = false;
};

std::unique_ptr<hnu25::camera::FrameSource> makeFrameSource(
    const hnu25::standard::CameraConfig& config) {
    if (config.type == "opencv") {
        return std::make_unique<hnu25::camera::OpenCvFrameSource>(hnu25::camera::OpenCvConfig{
            config.camera_id, config.width, config.height, config.exposure, config.gain, config.frame_rate});
    }
#if HNU25_HAS_MVS
    return std::make_unique<hnu25::camera::HikFrameSource>(hnu25::camera::HikConfig{
        config.serial_number, config.exposure, config.gain, config.frame_rate});
#else
    throw std::runtime_error("camera.type=hik requested, but this build has no MVS support");
#endif
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: sp_standard <config.yaml>\n";
        return 2;
    }

    std::atomic<bool> reader_stop{false};
    std::thread serial_reader;
    hnu25::SerialPort serial;
    try {
        const std::filesystem::path config_path = std::filesystem::absolute(argv[1]);
        const auto config = hnu25::standard::loadConfig(config_path);
        auto detector = hnu25::makeDetector(config.detector);
        sp_core::Solver solver(config_path.string());
        sp_core::Tracker tracker(config_path.string());
        sp_core::Aimer aimer(config.aimer);
        planner::MpcPlanner mpc_planner(config.planner);
        aim::FireGate fire_gate(config.fire_gate);
        telemetry::UdpPublisher publisher(config.telemetry);
        debug_web::DebugWebPublisher web_publisher(config.web);
        auto viewConfig = [&](const char* suffix) {
            auto value = config.web;
            value.shm_name += suffix;
            return value;
        };
        debug_web::DebugWebPublisher annotated_publisher(viewConfig("_annotated"));
        debug_web::DebugWebPublisher gray_publisher(viewConfig("_gray"));
        debug_web::DebugWebPublisher binary_publisher(viewConfig("_binary"));
        hnu25::standard::RuntimeParameters runtime_parameters{
            config.detector.conf_threshold, config.binary_threshold,
            config.bullet_speed, config.aimer};
        hnu25::standard::RuntimeTuning tuning(
            config_path, config.web.shm_name, runtime_parameters);

        const auto& q = config.imu_quaternion_wxyz;
        if (!config.feedback_pose.enabled) {
            solver.setImuQuaternion(Eigen::Quaterniond(q[0], q[1], q[2], q[3]));
            std::cerr << "[Safety] feedback pose disabled; static quaternion fallback is invalid for firing\n";
        }
        std::cerr << "[Safety] command_enabled=" << std::boolalpha << config.command_enabled
                   << " fire_gate_enabled=" << config.fire_gate.enabled << '\n';

        auto frame_source = makeFrameSource(config.camera);
        frame_source->start();

        FeedbackSnapshot feedback;
        std::mutex feedback_mutex;
        transform::TimePoseBuffer pose_buffer(
            {256, std::chrono::seconds(2),
             std::chrono::milliseconds(config.feedback_pose.max_age_ms)});
        hnu25::sp::EchoRejector echo_rejector(
            config.feedback_pose.reject_echo,
            std::chrono::milliseconds(config.feedback_pose.echo_window_ms));
        if (config.serial.enabled) {
            if (!serial.open(config.serial.device, config.serial.baud))
                throw std::runtime_error("cannot open configured serial device");
            serial_reader = std::thread([&] {
                hnu25::sp::StreamParser parser;
                while (!reader_stop.load(std::memory_order_relaxed)) {
                    const auto bytes = serial.read();
                    if (bytes.empty()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        continue;
                    }
                    for (const auto& frame : parser.push(bytes)) {
                        const auto received_at = std::chrono::steady_clock::now();
                        if (echo_rejector.reject(frame, received_at)) continue;
                        hnu25::sp::GimbalFeedback decoded;
                        if (hnu25::sp::decodeGimbalFeedback(
                                frame, decoded, config.feedback_pose.expected_address)) {
                            decoded.received_at = received_at;
                            if (config.feedback_pose.enabled) {
                                const auto pose_at = received_at +
                                    std::chrono::milliseconds(config.feedback_pose.time_offset_ms);
                                pose_buffer.push(
                                    pose_at,
                                    hnu25::standard::feedbackQuaternion(decoded, config.feedback_pose));
                            }
                            std::lock_guard<std::mutex> lock(feedback_mutex);
                            feedback.value = decoded;
                            feedback.valid = true;
                        }
                    }
                }
            });
        }

        std::signal(SIGINT, requestStop);
        hnu25::sp::SendLimiter limiter;
        const auto start = std::chrono::steady_clock::now();
        auto previous_capture = start;
        std::uint64_t frame_index = 0;
        double latency_sum_ms = 0.0;
        while (!stop_requested) {
            if (const auto request = tuning.poll()) {
                try {
                    detector->setConfidenceThreshold(request->parameters.conf_threshold);
                    aimer = sp_core::Aimer(request->parameters.aimer);
                    runtime_parameters = request->parameters;
                    if (request->save) tuning.save(runtime_parameters);
                    tuning.acknowledge(*request, runtime_parameters, true,
                                       request->save ? "applied and saved" : "temporarily applied");
                } catch (const std::exception& error) {
                    tuning.acknowledge(*request, runtime_parameters, false, error.what());
                }
            }
            hnu25::camera::Frame camera_frame;
            if (!frame_source->waitForFrame(camera_frame, std::chrono::milliseconds(1000))) {
                if (stop_requested) break;
                throw std::runtime_error("camera frame wait timed out");
            }
            const auto capture_timestamp = camera_frame.captured_at;
            const cv::Mat& frame = camera_frame.image;
            const double dt_s = frame_index == 0
                                    ? 0.0
                                    : std::chrono::duration<double>(capture_timestamp - previous_capture).count();
            previous_capture = capture_timestamp;

            const auto pose_query = pose_buffer.query(capture_timestamp);
            const bool pose_valid = config.feedback_pose.enabled && pose_query.valid() &&
                pose_query.age <= std::chrono::milliseconds(config.feedback_pose.max_age_ms);
            if (pose_valid) solver.setImuQuaternion(pose_query.quaternion);

            const auto detected = detector->detect(frame);
            const auto after_detection = std::chrono::steady_clock::now();
            std::vector<sp_core::Armor> armors;
            armors.reserve(detected.size());
            int pnp_valid = 0;
            int pnp_rejected = 0;
            double confidence_max = 0.0;
            double reprojection_sum = 0.0;
            double reprojection_max = 0.0;
            for (const auto& detection : detected) {
                confidence_max = std::max(confidence_max, static_cast<double>(detection.confidence));
                auto armor = sp_core::adapt(detection);
                if (!pose_valid && config.feedback_pose.enabled) {
                    ++pnp_rejected;
                } else if (solver.solve(armor)) {
                    ++pnp_valid;
                    reprojection_sum += armor.reprojection_error;
                    reprojection_max = std::max(reprojection_max, armor.reprojection_error);
                } else if (armor.reprojection_error > 0.0) {
                    ++pnp_rejected;
                }
                armors.push_back(std::move(armor));
            }
            const auto after_pnp = std::chrono::steady_clock::now();
            const auto target = tracker.update(armors, capture_timestamp);
            const auto after_tracker = std::chrono::steady_clock::now();
            sp_core::AimResult aim;
            if (target) aim = aimer.aim(*target, runtime_parameters.bullet_speed);
            const auto after_aim = std::chrono::steady_clock::now();

            FeedbackSnapshot feedback_copy;
            {
                std::lock_guard<std::mutex> lock(feedback_mutex);
                feedback_copy = feedback;
            }
            feedback_copy.valid = feedback_copy.valid && hnu25::sp::feedbackFresh(
                feedback_copy.value, std::chrono::milliseconds(config.feedback_pose.max_age_ms), after_aim);
            constexpr double degrees_to_radians = 0.01745329251994329577;
            const bool planner_feedback_fresh = feedback_copy.valid && hnu25::sp::feedbackFresh(
                feedback_copy.value,
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::duration<double>(config.planner.feedback_timeout_s)),
                after_aim);
            planner::PlannedCommand planned;
            planned.reference_yaw = aim.yaw;
            planned.reference_pitch = aim.pitch;
            planned.yaw = aim.yaw;
            planned.pitch = aim.pitch;
            planned.solver_status = config.planner.enabled
                ? planner::SolverStatus::InvalidInput : planner::SolverStatus::Disabled;
            if (planner_feedback_fresh) {
                planner::FeedbackState state;
                state.yaw_rad = feedback_copy.value.yaw_deg * degrees_to_radians;
                state.pitch_rad = feedback_copy.value.pitch_deg * degrees_to_radians;
                if (config.planner.use_feedback_velocity &&
                    std::isfinite(feedback_copy.value.yaw_speed))
                    state.yaw_velocity_rad_s = feedback_copy.value.yaw_speed * degrees_to_radians;
                if (config.planner.use_feedback_velocity &&
                    std::isfinite(feedback_copy.value.pitch_speed))
                    state.pitch_velocity_rad_s = feedback_copy.value.pitch_speed * degrees_to_radians;
                planned = mpc_planner.plan(aim, state, dt_s > 0.0 ? dt_s : config.planner.dt_s);
            }
            const sp_core::AimResult command_aim = planner::selectAim(aim, planned);
            aim::FireGateInput gate_input;
            gate_input.aim = command_aim;
            gate_input.track_state = tracker.state();
            gate_input.target_valid = target.has_value();
            gate_input.pose_valid = pose_valid;
            gate_input.feedback_valid = feedback_copy.valid;
            gate_input.measurement_age = std::chrono::duration<double>(pose_query.age).count();
            gate_input.reprojection_error = reprojection_max;
            gate_input.nis = tracker.lastNis();
            gate_input.feedback_yaw = feedback_copy.value.yaw_deg * degrees_to_radians;
            gate_input.feedback_pitch = feedback_copy.value.pitch_deg * degrees_to_radians;
            if (target) gate_input.armor_type = target->type();
            const auto fire_decision = fire_gate.evaluate(gate_input);

            if (limiter.allow(after_aim)) {
                const auto command = hnu25::standard::mapCommand(
                    command_aim, config.command_enabled, fire_decision);
                if (command) {
                    const auto bytes = hnu25::sp::encodeGimbalCommand(*command);
                    if (bytes.empty()) throw std::runtime_error("refusing invalid gimbal command");
                    if (!serial.isOpen()) throw std::runtime_error("command enabled but serial is not open");
                    echo_rejector.recordSent(bytes, after_aim);
                    if (!serial.write(bytes)) throw std::runtime_error("serial command write failed");
                }
            }

            const auto completed = std::chrono::steady_clock::now();
            const double detection_ms = std::chrono::duration<double, std::milli>(after_detection - capture_timestamp).count();
            const double pnp_ms = std::chrono::duration<double, std::milli>(after_pnp - after_detection).count();
            const double tracker_ms = std::chrono::duration<double, std::milli>(after_tracker - after_pnp).count();
            const double aim_ms = std::chrono::duration<double, std::milli>(after_aim - after_tracker).count();
            const double total_ms = std::chrono::duration<double, std::milli>(completed - capture_timestamp).count();
            latency_sum_ms += total_ms;

            telemetry::FrameSample sample;
            sample.frame_index = camera_frame.frame_number;
            sample.video_time_s = std::chrono::duration<double>(capture_timestamp - start).count();
            sample.timeline_s = sample.video_time_s;
            sample.dt_s = dt_s;
            sample.pose_valid = pose_valid;
            sample.detection_count = static_cast<int>(detected.size());
            sample.detection_confidence_max = confidence_max;
            sample.pnp_valid_count = pnp_valid;
            sample.pnp_rejected_count = pnp_rejected;
            sample.reprojection_mean_px = pnp_valid ? reprojection_sum / pnp_valid : 0.0;
            sample.reprojection_max_px = reprojection_max;
            sample.tracker_state_code = static_cast<int>(tracker.state());
            sample.tracker_state_name = tracker.stateName();
            sample.target_valid = target.has_value();
            if (target) {
                const auto& state = target->state();
                for (int i = 0; i < state.rows(); ++i) sample.state[i] = state[i];
            }
            sample.nis = tracker.lastNis();
            sample.nis_failure_ratio = tracker.nisFailureRatio();
            sample.aim_valid = command_aim.valid;
            sample.fire = fire_decision.allowed;
            sample.fire_reason = aim::toString(fire_decision.reason);
            sample.fire_yaw_error_rad = fire_decision.yaw_error;
            sample.fire_pitch_error_rad = fire_decision.pitch_error;
            sample.fire_yaw_tolerance_rad = fire_decision.yaw_tolerance;
            sample.fire_pitch_tolerance_rad = fire_decision.pitch_tolerance;
            sample.aim_yaw_rad = command_aim.yaw;
            sample.aim_pitch_rad = command_aim.pitch;
            sample.fly_time_s = aim.fly_time;
            sample.planner_valid = planned.valid;
            sample.planner_solver_status = static_cast<int>(planned.solver_status);
            sample.planner_reference_yaw_rad = planned.reference_yaw;
            sample.planner_reference_pitch_rad = planned.reference_pitch;
            sample.planner_yaw_rad = planned.yaw;
            sample.planner_pitch_rad = planned.pitch;
            sample.planner_yaw_velocity_rad_s = planned.yaw_vel;
            sample.planner_pitch_velocity_rad_s = planned.pitch_vel;
            sample.planner_yaw_acceleration_rad_s2 = planned.yaw_acc;
            sample.planner_pitch_acceleration_rad_s2 = planned.pitch_acc;
            sample.gimbal_feedback_valid = feedback_copy.valid;
            sample.gimbal_yaw_deg = feedback_copy.value.yaw_deg;
            sample.gimbal_pitch_deg = feedback_copy.value.pitch_deg;
            sample.gimbal_yaw_speed = feedback_copy.value.yaw_speed;
            sample.gimbal_pitch_speed = feedback_copy.value.pitch_speed;
            sample.detection_ms = detection_ms;
            sample.pnp_ms = pnp_ms;
            sample.tracker_ms = tracker_ms;
            sample.aim_ms = aim_ms;
            sample.total_ms = total_ms;
            publisher.publish(sample);
            const std::string state_json = telemetry::encodeJson(sample);
            web_publisher.publish(frame, state_json);
            if (annotated_publisher.due()) {
                cv::Mat annotated = frame.clone();
                for (const auto& value : detected) {
                    const cv::Scalar color = value.color == hnu25::Color::BLUE
                        ? cv::Scalar(255, 120, 40) : cv::Scalar(60, 80, 255);
                    cv::rectangle(annotated, value.box, color, 2);
                    for (const auto& point : value.keypoints)
                        cv::circle(annotated, point, 4, cv::Scalar(80, 255, 150), -1);
                    cv::putText(annotated, cv::format("%.2f", value.confidence),
                                value.box.tl() + cv::Point(0, -5),
                                cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
                }
                annotated_publisher.publish(annotated, state_json);
            }
            if (gray_publisher.due() || binary_publisher.due()) {
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                gray_publisher.publish(gray, state_json);
                cv::Mat binary;
                cv::threshold(gray, binary, runtime_parameters.binary_threshold, 255, cv::THRESH_BINARY);
                binary_publisher.publish(binary, state_json);
            }

            ++frame_index;
            if (frame_index % 100 == 0) {
                const double elapsed_s = std::chrono::duration<double>(completed - start).count();
                std::cout << "frames=" << frame_index << " fps=" << std::fixed << std::setprecision(1)
                          << (elapsed_s > 0.0 ? frame_index / elapsed_s : 0.0)
                          << " latency_ms=" << total_ms
                          << " mean_latency_ms=" << latency_sum_ms / frame_index
                          << " tracker=" << tracker.stateName() << '\n';
            }
        }

        reader_stop.store(true, std::memory_order_relaxed);
        if (serial_reader.joinable()) serial_reader.join();
        serial.flush();
        serial.close();
        frame_source->stop();
        std::cout << "stopped cleanly after " << frame_index << " frames\n";
    } catch (const std::exception& error) {
        reader_stop.store(true, std::memory_order_relaxed);
        if (serial_reader.joinable()) serial_reader.join();
        serial.close();
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
