#include "vest_runtime/runtime_config.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void validateAllowedKeys(const YAML::Node& node,
                         const std::vector<std::string>& allowed,
                         const std::string& section) {
    if (!node.IsMap()) {
        return;
    }
    for (const auto& kv : node) {
        const std::string key = kv.first.as<std::string>();
        if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
            throw std::runtime_error(
                "unknown config key '" + key + "' in section '" + section +
                "'");
        }
    }
}

YAML::Node requireMapSection(const YAML::Node& parent,
                             const std::string& key,
                             const std::string& parent_path) {
    const std::string full_path =
        parent_path.empty() ? key : parent_path + "." + key;
    YAML::Node node = parent[key];
    if (!node) {
        throw std::runtime_error("missing required config section: " +
                                 full_path);
    }
    if (!node.IsMap()) {
        throw std::runtime_error("config section '" + full_path +
                                 "' must be a map");
    }
    return node;
}

template <typename T>
T readRequired(const YAML::Node& parent, const std::string& key,
               const std::string& section) {
    const auto& node = parent[key];
    if (!node) {
        throw std::runtime_error("missing required config key '" + section +
                                 "." + key + "'");
    }
    try {
        return node.as<T>();
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("invalid value for '" + section + "." + key +
                                 "': " + e.what());
    }
}

}  // namespace

namespace hnu25::vest_runtime {

RuntimeConfig RuntimeConfig::loadFromFile(const std::string& path) {
    if (path.empty()) {
        throw std::invalid_argument(
            "vest runtime config path must not be empty");
    }

    YAML::Node root;
    try {
        root = YAML::LoadFile(path);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error(
            "failed to load vest runtime config '" + path + "': " + e.what());
    }

    if (!root.IsMap()) {
        throw std::runtime_error(
            "vest runtime config root must be a map");
    }

    // ── Root allowed keys ──────────────────────────────────────────
    validateAllowedKeys(root,
                        {"runtime", "detector", "tracker", "camera"},
                        "root");

    // ── Top-level sections ─────────────────────────────────────────
    const YAML::Node runtime_node = requireMapSection(root, "runtime", "");
    const YAML::Node detector_node = requireMapSection(root, "detector", "");
    const YAML::Node tracker_node = requireMapSection(root, "tracker", "");
    const YAML::Node camera_node = requireMapSection(root, "camera", "");

    RuntimeConfig config;

    // ═══════════════════════════════════════════════════════════════
    // runtime
    // ═══════════════════════════════════════════════════════════════
    validateAllowedKeys(runtime_node,
                        {"camera_backend", "frame_timeout_ms",
                         "max_consecutive_timeouts", "log_interval_frames"},
                        "runtime");

    {
        const std::string backend_str =
            readRequired<std::string>(runtime_node, "camera_backend",
                                      "runtime");
        if (backend_str == "hik") {
            config.runtime.camera_backend = CameraBackend::Hik;
        } else if (backend_str == "opencv") {
            config.runtime.camera_backend = CameraBackend::OpenCv;
        } else {
            throw std::runtime_error(
                "runtime.camera_backend must be 'hik' or 'opencv'");
        }
    }

    config.runtime.frame_timeout_ms =
        readRequired<int>(runtime_node, "frame_timeout_ms", "runtime");
    if (config.runtime.frame_timeout_ms <= 0) {
        throw std::runtime_error(
            "runtime.frame_timeout_ms must be positive");
    }

    config.runtime.max_consecutive_timeouts =
        readRequired<int>(runtime_node, "max_consecutive_timeouts",
                          "runtime");
    if (config.runtime.max_consecutive_timeouts <= 0) {
        throw std::runtime_error(
            "runtime.max_consecutive_timeouts must be positive");
    }

    config.runtime.log_interval_frames =
        readRequired<int>(runtime_node, "log_interval_frames", "runtime");
    if (config.runtime.log_interval_frames <= 0) {
        throw std::runtime_error(
            "runtime.log_interval_frames must be positive");
    }

    // ═══════════════════════════════════════════════════════════════
    // detector
    // ═══════════════════════════════════════════════════════════════
    validateAllowedKeys(detector_node,
                        {"model_path", "conf_threshold", "nms_threshold",
                         "performance_mode"},
                        "detector");

    config.detector.model_path =
        readRequired<std::string>(detector_node, "model_path", "detector");
    if (config.detector.model_path.empty()) {
        throw std::runtime_error("detector.model_path must not be empty");
    }

    config.detector.conf_threshold =
        readRequired<float>(detector_node, "conf_threshold", "detector");
    if (!std::isfinite(config.detector.conf_threshold)) {
        throw std::runtime_error("detector.conf_threshold must be finite");
    }
    if (config.detector.conf_threshold < 0.0F ||
        config.detector.conf_threshold > 1.0F) {
        throw std::runtime_error(
            "detector.conf_threshold must be in [0.0, 1.0]");
    }

    config.detector.nms_threshold =
        readRequired<float>(detector_node, "nms_threshold", "detector");
    if (!std::isfinite(config.detector.nms_threshold)) {
        throw std::runtime_error("detector.nms_threshold must be finite");
    }
    if (config.detector.nms_threshold < 0.0F ||
        config.detector.nms_threshold > 1.0F) {
        throw std::runtime_error(
            "detector.nms_threshold must be in [0.0, 1.0]");
    }

    config.detector.performance_mode =
        readRequired<std::string>(detector_node, "performance_mode",
                                  "detector");
    if (config.detector.performance_mode != "latency" &&
        config.detector.performance_mode != "throughput") {
        throw std::runtime_error(
            "detector.performance_mode must be 'latency' or 'throughput'");
    }

    // ═══════════════════════════════════════════════════════════════
    // tracker
    // ═══════════════════════════════════════════════════════════════
    validateAllowedKeys(tracker_node,
                        {"confirm_hits", "max_lost_frames", "min_iou",
                         "max_center_distance_ratio"},
                        "tracker");

    config.tracker.confirm_hits =
        readRequired<int>(tracker_node, "confirm_hits", "tracker");
    if (config.tracker.confirm_hits <= 0) {
        throw std::runtime_error(
            "tracker.confirm_hits must be positive");
    }

    config.tracker.max_lost_frames =
        readRequired<int>(tracker_node, "max_lost_frames", "tracker");
    if (config.tracker.max_lost_frames < 0) {
        throw std::runtime_error(
            "tracker.max_lost_frames must be >= 0");
    }

    config.tracker.min_iou =
        readRequired<float>(tracker_node, "min_iou", "tracker");
    if (!std::isfinite(config.tracker.min_iou)) {
        throw std::runtime_error("tracker.min_iou must be finite");
    }
    if (config.tracker.min_iou < 0.0F || config.tracker.min_iou > 1.0F) {
        throw std::runtime_error(
            "tracker.min_iou must be in [0.0, 1.0]");
    }

    config.tracker.max_center_distance_ratio =
        readRequired<float>(tracker_node, "max_center_distance_ratio",
                            "tracker");
    if (!std::isfinite(config.tracker.max_center_distance_ratio)) {
        throw std::runtime_error(
            "tracker.max_center_distance_ratio must be finite");
    }
    if (config.tracker.max_center_distance_ratio < 0.0F) {
        throw std::runtime_error(
            "tracker.max_center_distance_ratio must be >= 0.0");
    }

    // ═══════════════════════════════════════════════════════════════
    // camera
    // ═══════════════════════════════════════════════════════════════
    validateAllowedKeys(camera_node, {"hik", "opencv"}, "camera");

    // ── camera.hik ─────────────────────────────────────────────────
    const YAML::Node hik_node = requireMapSection(camera_node, "hik", "camera");
    validateAllowedKeys(hik_node,
                        {"serial_number", "exposure", "gain", "frame_rate"},
                        "camera.hik");

    config.hik_camera.serial_number =
        readRequired<std::string>(hik_node, "serial_number", "camera.hik");

    config.hik_camera.exposure =
        readRequired<double>(hik_node, "exposure", "camera.hik");
    if (!std::isfinite(config.hik_camera.exposure)) {
        throw std::runtime_error("camera.hik.exposure must be finite");
    }
    if (config.hik_camera.exposure <= 0.0) {
        throw std::runtime_error("camera.hik.exposure must be positive");
    }

    config.hik_camera.gain =
        readRequired<double>(hik_node, "gain", "camera.hik");
    if (!std::isfinite(config.hik_camera.gain)) {
        throw std::runtime_error("camera.hik.gain must be finite");
    }
    if (config.hik_camera.gain < 0.0) {
        throw std::runtime_error("camera.hik.gain must be >= 0.0");
    }

    config.hik_camera.frame_rate =
        readRequired<double>(hik_node, "frame_rate", "camera.hik");
    if (!std::isfinite(config.hik_camera.frame_rate)) {
        throw std::runtime_error("camera.hik.frame_rate must be finite");
    }
    if (config.hik_camera.frame_rate < 0.0) {
        throw std::runtime_error("camera.hik.frame_rate must be >= 0.0");
    }

    // ── camera.opencv ──────────────────────────────────────────────
    const YAML::Node opencv_node =
        requireMapSection(camera_node, "opencv", "camera");
    validateAllowedKeys(opencv_node,
                        {"camera_id", "width", "height", "exposure", "gain",
                         "frame_rate"},
                        "camera.opencv");

    config.opencv_camera.camera_id =
        readRequired<int>(opencv_node, "camera_id", "camera.opencv");
    if (config.opencv_camera.camera_id < 0) {
        throw std::runtime_error("camera.opencv.camera_id must be >= 0");
    }

    config.opencv_camera.width =
        readRequired<int>(opencv_node, "width", "camera.opencv");
    if (config.opencv_camera.width <= 0) {
        throw std::runtime_error("camera.opencv.width must be positive");
    }

    config.opencv_camera.height =
        readRequired<int>(opencv_node, "height", "camera.opencv");
    if (config.opencv_camera.height <= 0) {
        throw std::runtime_error("camera.opencv.height must be positive");
    }

    config.opencv_camera.exposure =
        readRequired<double>(opencv_node, "exposure", "camera.opencv");
    if (!std::isfinite(config.opencv_camera.exposure)) {
        throw std::runtime_error("camera.opencv.exposure must be finite");
    }
    if (config.opencv_camera.exposure < 0.0) {
        throw std::runtime_error("camera.opencv.exposure must be >= 0.0");
    }

    config.opencv_camera.gain =
        readRequired<double>(opencv_node, "gain", "camera.opencv");
    if (!std::isfinite(config.opencv_camera.gain)) {
        throw std::runtime_error("camera.opencv.gain must be finite");
    }
    if (config.opencv_camera.gain < 0.0) {
        throw std::runtime_error("camera.opencv.gain must be >= 0.0");
    }

    config.opencv_camera.frame_rate =
        readRequired<double>(opencv_node, "frame_rate", "camera.opencv");
    if (!std::isfinite(config.opencv_camera.frame_rate)) {
        throw std::runtime_error("camera.opencv.frame_rate must be finite");
    }
    if (config.opencv_camera.frame_rate < 0.0) {
        throw std::runtime_error("camera.opencv.frame_rate must be >= 0.0");
    }

    return config;
}

}  // namespace hnu25::vest_runtime
