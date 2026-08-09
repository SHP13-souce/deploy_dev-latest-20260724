#include "detection/detector.hpp"
#include "aim/fire_gate.hpp"
#include "debug_web/debug_web_publisher.hpp"
#include "sp_core/sp_core.hpp"
#include "telemetry/telemetry.hpp"
#include "transform/time_pose_buffer.hpp"

#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

struct Pose {
    double timestamp = 0.0;
    Eigen::Quaterniond quaternion = Eigen::Quaterniond::Identity();
};

std::vector<Pose> loadPoses(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open pose file: " + path);
    std::vector<Pose> poses;
    Pose pose;
    double qw, qx, qy, qz;
    while (input >> pose.timestamp >> qw >> qx >> qy >> qz) {
        pose.quaternion = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
        poses.push_back(pose);
    }
    if (poses.empty()) throw std::runtime_error("pose file is empty: " + path);
    return poses;
}

std::filesystem::path resolveConfigPath(const std::filesystem::path& config,
                                         const std::string& value) {
    std::filesystem::path path(value);
    return path.is_absolute() ? path : std::filesystem::weakly_canonical(config.parent_path() / path);
}

telemetry::Config loadTelemetryConfig(const YAML::Node& yaml) {
    telemetry::Config config;
    try {
        const YAML::Node node = yaml["telemetry"];
        if (!node) return config;
        config.enabled = node["enabled"].as<bool>(config.enabled);
        config.address = node["address"].as<std::string>(config.address);
        config.port = node["port"].as<std::uint16_t>(config.port);
        config.publish_hz = node["publish_hz"].as<double>(config.publish_hz);
    } catch (const YAML::Exception&) {
        // Invalid telemetry settings fall back to safe defaults and never stop replay.
    }
    return config;
}

debug_web::Config loadWebConfig(const YAML::Node& yaml) {
    debug_web::Config config;
    try {
        const YAML::Node node = yaml["web"];
        if (!node) return config;
        config.enabled = node["enabled"].as<bool>(config.enabled);
        config.shm_name = node["shm_name"].as<std::string>(config.shm_name);
        config.quality = node["quality"].as<int>(config.quality);
        config.publish_hz = node["publish_hz"].as<double>(config.publish_hz);
    } catch (const YAML::Exception&) {
        // Debug output must never prevent an offline replay from running.
        config.enabled = false;
    }
    return config;
}

hnu25::DetectorConfig loadDetectorConfig(const std::filesystem::path& config_path,
                                          const YAML::Node& yaml) {
    hnu25::DetectorConfig config;
    const YAML::Node detector = yaml["detector"];
    if (detector) {
        config.backend = detector["backend"].as<std::string>(config.backend);
        config.model_path = resolveConfigPath(
            config_path, detector["model_path"].as<std::string>()).string();
        if (detector["classifier_path"])
            config.classifier_path = resolveConfigPath(
                config_path, detector["classifier_path"].as<std::string>()).string();
        config.conf_threshold = detector["conf_threshold"].as<float>(
            config.backend == "awakening" ? 0.2F : config.conf_threshold);
        config.performance_mode = detector["performance_mode"].as<std::string>(
            config.performance_mode);
    } else {
        config.model_path = resolveConfigPath(
            config_path, yaml["model_path"].as<std::string>()).string();
        config.conf_threshold = yaml["conf_threshold"].as<float>(config.conf_threshold);
    }
    return config;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: sp_demo_replay <demo.avi> <demo.txt> <sp_demo.yaml> [max_frames]\n";
        return 2;
    }
    try {
        const std::filesystem::path config_path = std::filesystem::absolute(argv[3]);
        const YAML::Node yaml = YAML::LoadFile(config_path.string());
        auto detector = hnu25::makeDetector(loadDetectorConfig(config_path, yaml));
        sp_core::Solver solver(config_path.string());
        sp_core::Tracker tracker(config_path.string());
        sp_core::Aimer aimer(config_path.string());
        telemetry::UdpPublisher telemetry_publisher(loadTelemetryConfig(yaml));
        debug_web::DebugWebPublisher web_publisher(loadWebConfig(yaml));
        const auto poses = loadPoses(argv[2]);
        cv::VideoCapture capture(argv[1]);
        if (!capture.isOpened()) throw std::runtime_error("cannot open video: " + std::string(argv[1]));
        const double fps = capture.get(cv::CAP_PROP_FPS) > 0.0 ? capture.get(cv::CAP_PROP_FPS) : 30.0;
        const int total_frames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
        const int max_frames = argc >= 5 ? std::stoi(argv[4])
                                         : (total_frames > 0 ? total_frames
                                                             : std::numeric_limits<int>::max());
        const double bullet_speed = yaml["bullet_speed"].as<double>(23.0);
        const auto epoch = std::chrono::steady_clock::time_point{};
        const double first_pose_time = poses.front().timestamp;
        double timeline_seconds = 0.0;
        double previous_pose_time = first_pose_time;
        transform::TimePoseBuffer pose_buffer({256, std::chrono::seconds(2),
                                                std::chrono::milliseconds(50)});
        aim::FireGateConfig replay_gate_config;
        replay_gate_config.enabled = true;
        aim::FireGate replay_fire_gate(replay_gate_config);

        int frames = 0, detection_frames = 0, detections = 0, pnp_ok = 0, pnp_rejected = 0;
        int pnp_skipped_no_pose = 0;
        int pose_frames = 0, pose_missing_frames = 0, timeline_resets = 0;
        int tracking_frames = 0, aim_ok = 0, safe_aim_frames = 0;
        double detection_ms = 0.0, pnp_ms = 0.0, tracking_ms = 0.0, aimer_ms = 0.0;
        double reprojection_sum = 0.0, reprojection_max = 0.0, nis_sum = 0.0;
        int nis_count = 0;
        cv::Mat frame;
        while (frames < max_frames && capture.read(frame)) {
            double video_time = capture.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
            if (!(video_time >= 0.0)) video_time = frames / fps;
            const bool has_pose_sample = static_cast<std::size_t>(frames) < poses.size();
            bool timeline_reset = false;
            double frame_dt = frames == 0 ? 0.0 : 1.0 / fps;
            if (has_pose_sample) {
                const auto& pose = poses[frames];
                if (frames > 0) {
                    const double raw_dt = pose.timestamp - previous_pose_time;
                    if (raw_dt > 0.0 && raw_dt <= 0.1) {
                        timeline_seconds += raw_dt;
                        frame_dt = raw_dt;
                    } else {
                        // The recorder can pause between clips. Advance by one video frame so the
                        // tracker sees a clean segment boundary rather than a multi-second jump.
                        timeline_seconds += 1.0 / fps;
                        timeline_reset = true;
                        ++timeline_resets;
                        tracker.reset();
                        pose_buffer.reset();
                    }
                }
                previous_pose_time = pose.timestamp;
                ++pose_frames;
            } else {
                timeline_seconds += 1.0 / fps;
                ++pose_missing_frames;
            }
            const auto timeline = epoch + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                               std::chrono::duration<double>(timeline_seconds));
            if (has_pose_sample && !pose_buffer.push(timeline, poses[frames].quaternion))
                throw std::runtime_error("invalid or out-of-order replay pose sample");
            const auto pose_query = pose_buffer.query(timeline);
            const bool pose_valid = pose_query.valid();
            if (pose_valid) {
                solver.setImuQuaternion(pose_query.quaternion);
            } else if (pose_missing_frames == 1 || pose_query.status == transform::PoseStatus::TooOld) {
                tracker.reset();
            }

            const auto begin = std::chrono::steady_clock::now();
            const auto detected = detector->detect(frame);
            const auto after_detection = std::chrono::steady_clock::now();
            double confidence_max = 0.0;
            for (const auto& value : detected) confidence_max = std::max(
                confidence_max, static_cast<double>(value.confidence));
            detections += static_cast<int>(detected.size());
            detection_frames += !detected.empty();
            std::vector<sp_core::Armor> armors;
            armors.reserve(detected.size());
            int frame_pnp_ok = 0, frame_pnp_rejected = 0;
            double frame_reprojection_sum = 0.0, frame_reprojection_max = 0.0;
            for (const auto& value : detected) {
                auto armor = sp_core::adapt(value);
                if (!pose_valid) {
                    ++pnp_skipped_no_pose;
                } else if (solver.solve(armor)) {
                    ++pnp_ok;
                    ++frame_pnp_ok;
                    reprojection_sum += armor.reprojection_error;
                    reprojection_max = std::max(reprojection_max, armor.reprojection_error);
                    frame_reprojection_sum += armor.reprojection_error;
                    frame_reprojection_max = std::max(frame_reprojection_max, armor.reprojection_error);
                }
                else {
                    ++pnp_rejected;
                    ++frame_pnp_rejected;
                }
                armors.push_back(std::move(armor));
            }
            const auto after_pnp = std::chrono::steady_clock::now();
            // World-frame tracking is invalid without a pose sample. A segment boundary naturally
            // resets the tracker through max_tracker_dt on the next valid sample.
            const auto target = pose_valid ? tracker.update(armors, timeline) : std::nullopt;
            const auto after_tracking = std::chrono::steady_clock::now();
            sp_core::AimResult aim;
            if (target) aim = aimer.aim(*target, bullet_speed);
            const auto after_aim = std::chrono::steady_clock::now();
            const double frame_detection_ms =
                std::chrono::duration<double, std::milli>(after_detection - begin).count();
            const double frame_pnp_ms =
                std::chrono::duration<double, std::milli>(after_pnp - after_detection).count();
            const double frame_tracking_ms =
                std::chrono::duration<double, std::milli>(after_tracking - after_pnp).count();
            const double frame_aimer_ms =
                std::chrono::duration<double, std::milli>(after_aim - after_tracking).count();
            tracking_frames += pose_valid && tracker.state() == sp_core::TrackState::TRACKING;
            aim_ok += aim.valid;
            safe_aim_frames += aim.valid && tracker.state() == sp_core::TrackState::TRACKING;
            if (target && tracker.lastNis() > 0.0) { nis_sum += tracker.lastNis(); ++nis_count; }
            detection_ms += frame_detection_ms;
            pnp_ms += frame_pnp_ms;
            tracking_ms += frame_tracking_ms;
            aimer_ms += frame_aimer_ms;

            telemetry::FrameSample telemetry_sample;
            telemetry_sample.frame_index = static_cast<std::uint64_t>(frames);
            telemetry_sample.video_time_s = video_time;
            telemetry_sample.timeline_s = timeline_seconds;
            telemetry_sample.dt_s = frame_dt;
            telemetry_sample.pose_valid = pose_valid;
            telemetry_sample.timeline_reset = timeline_reset;
            telemetry_sample.detection_count = static_cast<int>(detected.size());
            telemetry_sample.detection_confidence_max = confidence_max;
            telemetry_sample.pnp_valid_count = frame_pnp_ok;
            telemetry_sample.pnp_rejected_count = frame_pnp_rejected;
            telemetry_sample.reprojection_mean_px =
                frame_pnp_ok ? frame_reprojection_sum / frame_pnp_ok : 0.0;
            telemetry_sample.reprojection_max_px = frame_reprojection_max;
            telemetry_sample.tracker_state_code = static_cast<int>(tracker.state());
            telemetry_sample.tracker_state_name = tracker.stateName();
            telemetry_sample.target_valid = target.has_value();
            if (target) {
                const auto& state = target->state();
                for (int i = 0; i < state.rows(); ++i) telemetry_sample.state[i] = state[i];
            }
            telemetry_sample.nis = tracker.lastNis();
            telemetry_sample.nis_failure_ratio = tracker.nisFailureRatio();
            telemetry_sample.aim_valid = aim.valid;
            aim::FireGateInput gate_input;
            gate_input.aim = aim;
            gate_input.track_state = tracker.state();
            gate_input.target_valid = target.has_value();
            gate_input.pose_valid = pose_valid;
            gate_input.feedback_valid = false;
            gate_input.measurement_age = std::chrono::duration<double>(pose_query.age).count();
            gate_input.reprojection_error = frame_reprojection_max;
            gate_input.nis = tracker.lastNis();
            if (target) gate_input.armor_type = target->type();
            const auto fire_decision = replay_fire_gate.evaluate(gate_input);
            telemetry_sample.fire = fire_decision.allowed;
            telemetry_sample.fire_reason = aim::toString(fire_decision.reason);
            telemetry_sample.fire_yaw_error_rad = fire_decision.yaw_error;
            telemetry_sample.fire_pitch_error_rad = fire_decision.pitch_error;
            telemetry_sample.fire_yaw_tolerance_rad = fire_decision.yaw_tolerance;
            telemetry_sample.fire_pitch_tolerance_rad = fire_decision.pitch_tolerance;
            telemetry_sample.aim_yaw_rad = aim.yaw;
            telemetry_sample.aim_pitch_rad = aim.pitch;
            telemetry_sample.fly_time_s = aim.fly_time;
            telemetry_sample.detection_ms = frame_detection_ms;
            telemetry_sample.pnp_ms = frame_pnp_ms;
            telemetry_sample.tracker_ms = frame_tracking_ms;
            telemetry_sample.aim_ms = frame_aimer_ms;
            telemetry_sample.total_ms = frame_detection_ms + frame_pnp_ms +
                                        frame_tracking_ms + frame_aimer_ms;
            telemetry_publisher.publish(telemetry_sample);
            web_publisher.publish(frame, telemetry::encodeJson(telemetry_sample));
            ++frames;
            if (frames % 50 == 0 || frames == max_frames ||
                (total_frames > 0 && frames == total_frames)) {
                std::cout << "frame=" << frames << " t=" << std::fixed << std::setprecision(3)
                          << video_time << " det=" << detected.size() << " pnp="
                          << std::count_if(armors.begin(), armors.end(), [](const auto& a) { return a.pnp_valid; })
                            << " pose=" << pose_valid << " reset=" << timeline_reset
                           << " tracking=" << tracker.stateName() << " aim=" << aim.valid
                          << " reproj=" << (armors.empty() ? 0.0 : armors.front().reprojection_error)
                           << " nis=" << tracker.lastNis()
                           << " fire_reason=" << aim::toString(fire_decision.reason) << '\n';
            }
        }
        const double divisor = std::max(frames, 1);
        std::cout << "\nSP demo replay summary\n"
                  << "frames=" << frames << "/" << total_frames << " pose_samples=" << poses.size()
                  << " pose_frames=" << pose_frames << " pose_missing=" << pose_missing_frames
                  << " timeline_resets=" << timeline_resets << '\n'
                  << "detection_frames=" << detection_frames << " detections=" << detections << '\n'
                  << "pnp_ok=" << pnp_ok << " pnp_rejected=" << pnp_rejected
                  << " pnp_skipped_no_pose=" << pnp_skipped_no_pose
                  << " tracking_frames=" << tracking_frames << " aim_ok=" << aim_ok
                  << " safe_aim_frames=" << safe_aim_frames << '\n'
                  << "reprojection_mean_px=" << (pnp_ok ? reprojection_sum / pnp_ok : 0.0)
                  << " reprojection_max_px=" << reprojection_max << '\n'
                  << "nis_mean=" << (nis_count ? nis_sum / nis_count : 0.0) << " nis_samples=" << nis_count
                  << " nis_failure_ratio=" << tracker.nisFailureRatio() << '\n'
                  << "mean_ms detection=" << detection_ms / divisor << " pnp=" << pnp_ms / divisor
                  << " tracking=" << tracking_ms / divisor << " aim=" << aimer_ms / divisor
                  << " total=" << (detection_ms + pnp_ms + tracking_ms + aimer_ms) / divisor << '\n';
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
