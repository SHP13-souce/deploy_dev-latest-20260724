#include "anti_drone/config.hpp"
#include "anti_drone/diagnostic_pipeline.hpp"

#include <cmath>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#ifndef ANTI_DRONE_SOURCE_DIR
#error "ANTI_DRONE_SOURCE_DIR is not defined"
#endif

namespace {

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "[ OK ] " << message << '\n';
    } else {
        std::cerr << "FAILED: " << message << '\n';
        ++g_failures;
    }
}

}  // namespace

int main() {
    try {
        const std::filesystem::path config_path =
            std::filesystem::path(ANTI_DRONE_SOURCE_DIR) /
            "config" /
            "anti_drone.yaml";

        // ── Test 1: the production file must exist ────────────────────────
        const bool exists = std::filesystem::exists(config_path);
        check(exists, "production config exists");
        if (!exists) {
            std::cerr << "    config_path: " << config_path << '\n';
            std::cerr << g_failures << " check(s) failed.\n";
            return 1;
        }

        // ── Test 2: it must load through the real production API ──────────
        hnu25::anti_drone::AntiDroneConfig config;
        bool loaded = false;
        std::string load_error;
        try {
            config = hnu25::anti_drone::loadAntiDroneConfig(config_path);
            loaded = true;
        } catch (const std::exception& e) {
            load_error = e.what();
        }
        if (!loaded) {
            check(false,
                  std::string("production config loaded: ") + load_error);
            std::cerr << g_failures << " check(s) failed.\n";
            return 1;
        }
        check(true, "production config loaded");

        // ── Test 3: traditional detector sanity (not exhaustive) ──────────
        const auto& td = config.traditional_detector;
        check(std::isfinite(td.min_cv_score) && td.min_cv_score >= 0.0F &&
                  td.min_cv_score <= 1.0F,
              "traditional detector min_cv_score sane");
        check(td.white_saturation_max >= 0 && td.white_saturation_max <= 255,
              "traditional detector white_saturation_max sane");
        check(td.white_value_min >= 0 && td.white_value_min <= 255,
              "traditional detector white_value_min sane");

        // ── Test 4: tracker sanity (no hard-coded tuning values) ──────────
        const auto& tracker = config.tracker;
        check(tracker.min_detect_count >= 1, "tracker min_detect_count sane");
        check(tracker.max_missed_count >= 0, "tracker max_missed_count sane");
        check(std::isfinite(tracker.max_dt_s) && tracker.max_dt_s > 0.0,
              "tracker max_dt_s sane");
        check(std::isfinite(tracker.max_association_distance_m) &&
                  tracker.max_association_distance_m > 0.0,
              "tracker max_association_distance_m sane");
        check(std::isfinite(tracker.position_gain),
              "tracker position_gain sane");
        check(std::isfinite(tracker.velocity_gain),
              "tracker velocity_gain sane");

        // ── Test 5: prediction horizon is the intentional default ─────────
        // 0.0 is the current project-stage production default until real
        // end-to-end latency is measured. When that changes and the YAML is
        // updated, update this assertion together with it — do not silently
        // let it go stale. Do NOT use the synthetic test's 0.08 here.
        check(config.prediction.horizon_s == 0.0,
              "prediction default sane (horizon_s == 0.0)");

        // ── Test 6: compensation fields are finite (not forced to 0) ──────
        const auto& vc = config.vision_compensation;
        check(std::isfinite(vc.yaw_offset_deg) &&
                  std::isfinite(vc.pitch_offset_deg) &&
                  std::isfinite(vc.x_offset_m) &&
                  std::isfinite(vc.y_offset_m) &&
                  std::isfinite(vc.z_offset_m),
              "compensation finite");

        // ── Test 7: the adapter preserves every runtime field ─────────────
        const auto pipeline_config =
            hnu25::anti_drone::makeDiagnosticPipelineConfig(config);
        check(pipeline_config.tracker.min_detect_count ==
                      config.tracker.min_detect_count &&
                  pipeline_config.tracker.max_missed_count ==
                      config.tracker.max_missed_count &&
                  pipeline_config.tracker.max_dt_s ==
                      config.tracker.max_dt_s &&
                  pipeline_config.tracker.max_association_distance_m ==
                      config.tracker.max_association_distance_m &&
                  pipeline_config.tracker.position_gain ==
                      config.tracker.position_gain &&
                  pipeline_config.tracker.velocity_gain ==
                      config.tracker.velocity_gain,
              "pipeline config tracker mapped");
        check(pipeline_config.prediction.horizon_s ==
                  config.prediction.horizon_s,
              "pipeline config prediction mapped");
        check(pipeline_config.compensation.yaw_offset_deg ==
                      config.vision_compensation.yaw_offset_deg &&
                  pipeline_config.compensation.pitch_offset_deg ==
                      config.vision_compensation.pitch_offset_deg &&
                  pipeline_config.compensation.x_offset_m ==
                      config.vision_compensation.x_offset_m &&
                  pipeline_config.compensation.y_offset_m ==
                      config.vision_compensation.y_offset_m &&
                  pipeline_config.compensation.z_offset_m ==
                      config.vision_compensation.z_offset_m,
              "pipeline config compensation mapped");

        // ── Test 8: the assembled config constructs the pipeline ──────────
        bool constructed = false;
        std::string ctor_error;
        try {
            hnu25::anti_drone::DiagnosticPipeline pipeline(pipeline_config);
            constructed = true;
        } catch (const std::exception& e) {
            ctor_error = e.what();
        }
        if (constructed) {
            check(true, "DiagnosticPipeline constructed");
        } else {
            check(false,
                  std::string("DiagnosticPipeline constructed: ") + ctor_error);
        }

        // ── Test 9: production camera runtime sanity ──────────────────────
        const auto& camera = config.camera;
        check(std::isfinite(camera.exposure) && camera.exposure > 0.0,
              "camera exposure sane (> 0)");
        check(std::isfinite(camera.frame_rate) && camera.frame_rate > 0.0,
              "camera frame_rate sane (> 0)");
        check(camera.frame_timeout_ms > 0,
              "camera frame_timeout_ms sane (> 0)");
        check(camera.max_consecutive_timeouts >= 1,
              "camera max_consecutive_timeouts sane (>= 1)");

        // ── Test 10: production runtime sanity ────────────────────────────
        check(config.runtime.log_every_n_frames >= 1,
              "runtime log_every_n_frames sane (>= 1)");

        // ── Test 11: telemetry transport defaults (LOOPBACK) ──────────────
        const auto& telemetry = config.telemetry;
        check(telemetry.mode ==
                  hnu25::anti_drone::TelemetryTransportMode::LOOPBACK,
              "telemetry mode defaults to LOOPBACK");
        check(telemetry.baud_rate == 115200,
              "telemetry baud_rate defaults to 115200");
        check(telemetry.max_consecutive_failures == 5,
              "telemetry max_consecutive_failures defaults to 5");
        check(telemetry.flush_after_write == false,
              "telemetry flush_after_write defaults to false");

        if (g_failures == 0) {
            std::cout << "All production config smoke checks passed.\n";
            return 0;
        }
        std::cerr << g_failures << " check(s) failed.\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Production config smoke test exception: " << e.what()
                  << '\n';
        return 1;
    }
}
