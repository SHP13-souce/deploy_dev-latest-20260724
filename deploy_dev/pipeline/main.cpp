/**
 * @file    pipeline/main.cpp
 * @brief   HNU_NHS_Vision-25 主流水线 —— 三组融合版本
 *
 * 管线:
 *   相机取帧 -> Yolo26Detector (识别组) -> PnPSolver (解算组)
 *   -> Predictor EKF (解算组) -> BCP 串口发送 (通信组)
 *
 * 编译:  cmake -B build && cmake --build build
 * 运行:  ./build/hnu_vision [config_path] [camera_id]
 */

#include "shared/types.hpp"
#include "detection/yolo26_detector.hpp"
#include "solve/pnp_solver.hpp"
#include "solve/predictor.hpp"
#include "communication/bcp_packet.hpp"
#include "communication/serial_port.hpp"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <thread>
#include <fstream>
#include <sstream>

using namespace hnu25;
using namespace std::chrono;

// ─── 全局标志位 (信号处理) ────────────────────────────────────────────────

static std::atomic<bool> g_running{true};

static void signalHandler(int) {
    g_running = false;
}

// ─── 简易配置解析 ─────────────────────────────────────────────────────────

struct PipelineConfig {
    // 相机
    int    camera_id = 0;
    int    camera_width  = 1440;
    int    camera_height = 1080;

    // 检测
    std::string model_path  = "assets/models/hebei_at_nn/praysky_coord_noe2e_0331_640x640.onnx";
    float conf_threshold    = 0.25f;
    Color  enemy_color      = Color::RED;

    // PnP
    double fx = 1100.0, fy = 1100.0;
    double cx = 720.0,  cy = 540.0;

    // EKF
    double ekf_position_sigma = 0.01;
    double ekf_velocity_sigma = 0.5;
    double ekf_measurement_sigma = 0.05;
    double predict_time  = 0.05;    // 50ms
    double max_lost_time = 0.30;    // 300ms

    // 通信
    std::string serial_device = "/dev/ttyUSB0";
    int    serial_baud = 115200;
    bool   serial_enabled = true;

    // 开火条件
    double fire_distance_max = 8.0;  // 最大开火距离 (m)
    double fire_yaw_tol      = 0.05;  // yaw 偏差容忍 (rad, ~3度)
    int    fire_delay_frames = 3;     // 连续检测帧数

    // 调试
    bool   show_window = false;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            std::cerr << "[Config] cannot open '" << path << "', using defaults\n";
            return false;
        }
        std::string line;
        while (std::getline(f, line)) {
            auto p = line.find(':');
            if (p == std::string::npos) continue;
            std::string k = line.substr(0, p);
            std::string v = line.substr(p + 1);
            // trim
            k.erase(0, k.find_first_not_of(" \t"));
            k.erase(k.find_last_not_of(" \t\r\n") + 1);
            v.erase(0, v.find_first_not_of(" \t"));
            v.erase(v.find_last_not_of(" \t\r\n") + 1);

            try {
                if (k == "camera_id")          camera_id = std::stoi(v);
                else if (k == "camera_width")  camera_width = std::stoi(v);
                else if (k == "camera_height") camera_height = std::stoi(v);
                else if (k == "model_path")    model_path = v;
                else if (k == "conf_threshold")  conf_threshold = std::stof(v);
                else if (k == "enemy_color") {
                    if (v == "blue" || v == "BLUE") enemy_color = Color::BLUE;
                    else enemy_color = Color::RED;
                }
                else if (k == "fx") fx = std::stod(v);
                else if (k == "fy") fy = std::stod(v);
                else if (k == "cx") cx = std::stod(v);
                else if (k == "cy") cy = std::stod(v);
                else if (k == "ekf_position_sigma")  ekf_position_sigma  = std::stod(v);
                else if (k == "ekf_velocity_sigma")  ekf_velocity_sigma  = std::stod(v);
                else if (k == "ekf_measurement_sigma") ekf_measurement_sigma = std::stod(v);
                else if (k == "predict_time")  predict_time  = std::stod(v);
                else if (k == "max_lost_time") max_lost_time = std::stod(v);
                else if (k == "serial_device")  serial_device = v;
                else if (k == "serial_baud")    serial_baud = std::stoi(v);
                else if (k == "serial_enabled") serial_enabled = (v == "true" || v == "1" || v == "yes");
                else if (k == "fire_distance_max") fire_distance_max = std::stod(v);
                else if (k == "fire_yaw_tol")      fire_yaw_tol      = std::stod(v);
                else if (k == "show_window") show_window = (v == "true" || v == "1" || v == "yes");
            } catch (...) {}
        }
        return true;
    }
};

// ─── 辅助函数 ─────────────────────────────────────────────────────────────

inline double rad2deg(double r) { return r * 57.29577951308232; }

// 选择优先级最高的装甲板 (最近的 RED 目标)
inline const DetectedArmor* pickBest(const std::vector<DetectedArmor>& armors, Color enemy) {
    const DetectedArmor* best = nullptr;
    double best_dist = 1e9;
    for (const auto& a : armors) {
        if (a.color != enemy) continue;
        double d = cv::norm(cv::Point2f(
            static_cast<float>(a.box.x + a.box.width  / 2.0),
            static_cast<float>(a.box.y + a.box.height / 2.0))
            - cv::Point2f(720, 540));
        if (d < best_dist) { best_dist = d; best = &a; }
    }
    // 如果没有检测到敌方颜色，退回到最近的目标
    if (!best) {
        for (const auto& a : armors) {
            double d = cv::norm(cv::Point2f(
                static_cast<float>(a.box.x + a.box.width  / 2.0),
                static_cast<float>(a.box.y + a.box.height / 2.0))
                - cv::Point2f(720, 540));
            if (d < best_dist) { best_dist = d; best = &a; }
        }
    }
    return best;
}

// ─── 主函数 ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    // 信号处理
    std::signal(SIGINT,  signalHandler);
    std::signal(SIGTERM, signalHandler);

    // 解析参数
    std::string config_path = "config/car.yaml";
    if (argc >= 2) config_path = argv[1];

    PipelineConfig cfg;
    cfg.load(config_path);

    std::cout << "=== HNU_NHS_Vision-25 ===\n";
    std::cout << "config: " << config_path << "\n";
    std::cout << "enemy:  " << color_name(cfg.enemy_color) << "方\n";

    // ─── 初始化各模块 ────────────────────────────────────────────────────

    // 1. 识别组: YOLO26 检测器
    std::cout << "[Init] detector: " << cfg.model_path << "\n";
    Yolo26Detector detector(cfg.model_path, cfg.conf_threshold);

    // 2. 解算组: PnP + EKF
    PnPSolver solver(cfg.fx, cfg.fy, cfg.cx, cfg.cy);
    solver.setEnemyColor(cfg.enemy_color);

    Predictor predictor;
    predictor.setPredictTime(cfg.predict_time);
    predictor.setMaxLostTime(cfg.max_lost_time);

    // 3. 通信组: 串口 + BCP
    SerialPort serial;
    if (cfg.serial_enabled) {
        std::cout << "[Init] serial: " << cfg.serial_device
                  << " " << cfg.serial_baud << "\n";
        if (!serial.open(cfg.serial_device, cfg.serial_baud)) {
            std::cerr << "[Warn] serial open failed, will skip send\n";
            cfg.serial_enabled = false;
        }
    }

    // 4. 相机
    cv::VideoCapture cap(cfg.camera_id);
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.camera_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.camera_height);
    if (!cap.isOpened()) {
        std::cerr << "[Fatal] cannot open camera\n";
        return 1;
    }
    std::cout << "[Init] camera: " << cfg.camera_width << "x" << cfg.camera_height << "\n";

    // ─── 主循环 ──────────────────────────────────────────────────────────

    int frame_count = 0;
    int detect_count = 0;
    int fire_count   = 0;
    int consecutive_detect = 0;
    int target_id    = 0;
    auto last_status_time = steady_clock::now();

    std::cout << "\n[Run] main loop started, press Ctrl+C to stop\n\n";

    while (g_running) {
        // ── 0. 取帧 ──
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            std::this_thread::sleep_for(milliseconds(5));
            continue;
        }
        auto t0 = steady_clock::now();
        frame_count++;

        // ── 1. 识别组: YOLO26 检测 ──
        auto armors = detector.detect(frame);
        bool has_det = !armors.empty();

        // ── 2. 选择最佳目标 ──
        const auto* best = pickBest(armors, cfg.enemy_color);

        TargetMeasurement meas;
        meas.timestamp = t0;
        meas.detected  = (best != nullptr);

        // ── 3. 解算组: PnP 解算 ──
        if (best) {
            auto pnp = solver.solve(*best);
            if (pnp.valid) {
                meas.position = pnp.position;
                meas.yaw      = pnp.yaw;
                meas.id       = target_id;
            } else {
                meas.detected = false;
            }
        }

        // ── 4. 解算组: EKF 预测 ──
        auto state = predictor.update(meas);

        // ── 5. 通信组: 生成云台指令 → BCP 发送 ──
        GimbalCommand cmd;
        if (state.valid) {
            double px = state.predicted_position.x();
            double py = state.predicted_position.y();
            double pz = state.predicted_position.z();

            cmd.yaw       = std::atan2(py, px);
            cmd.pitch     = std::atan2(pz, std::hypot(px, py));
            cmd.yaw_vel   = state.velocity(1);  // 简化
            cmd.pitch_vel = state.velocity(2);

            double dist = state.predicted_position.norm();
            cmd.fire = (dist > 0.1 && dist < cfg.fire_distance_max &&
                        std::abs(cmd.yaw) < cfg.fire_yaw_tol);

            if (cmd.fire) {
                consecutive_detect++;
                if (consecutive_detect < cfg.fire_delay_frames)
                    cmd.fire = false;
            } else {
                consecutive_detect = 0;
            }
        }

        // 串口发送
        if (cfg.serial_enabled) {
            bcp::GimbalPayload payload;
            payload.mode  = 1;
            payload.yaw   = bcp::radToInt1000(cmd.yaw);
            payload.pitch = bcp::radToInt1000(cmd.pitch);
            payload.roll  = 0;

            auto frame_data = bcp::buildGimbalFrame(payload);
            serial.write(frame_data);
            serial.flush();
        }

        // ── 6. 统计 + 显示 ──
        if (has_det) detect_count++;
        if (cmd.fire) fire_count++;

        // 每秒输出一次状态
        auto now = steady_clock::now();
        if (duration_cast<milliseconds>(now - last_status_time).count() >= 1000) {
            double fps = frame_count / duration_cast<duration<double>>(now - t0).count();
            // 估算 (使用粗略时间)
            auto elapsed = duration_cast<milliseconds>(now - t0).count();

            std::cout << "\r[Frame " << frame_count << "] "
                      << "det=" << detect_count
                      << " fire=" << fire_count
                      << " | yaw=" << int(rad2deg(cmd.yaw))
                      << "deg pitch=" << int(rad2deg(cmd.pitch)) << "deg"
                      << " | dist=" << int(state.predicted_position.norm() * 100) << "cm"
                      << " | " << (cmd.fire ? "FIRE" : "hold")
                      << "        \r" << std::flush;

            last_status_time = now;
        }

        // 可选窗口显示
        if (cfg.show_window) {
            cv::Mat display = frame.clone();
            if (best) {
                cv::rectangle(display, best->box, cv::Scalar(0, 255, 0), 2);
                for (const auto& kp : best->keypoints)
                    cv::circle(display, kp, 3, cv::Scalar(0, 0, 255), -1);
            }
            if (state.valid) {
                char buf[128];
                snprintf(buf, sizeof(buf), "dist=%.1fm yaw=%.1f",
                         state.predicted_position.norm(), rad2deg(cmd.yaw));
                cv::putText(display, buf, cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 255, 255), 2);
            }
            cv::imshow("HNU_NHS_Vision-25", display);
            if (cv::waitKey(1) == 27) break;
        }
    }

    // ─── 清理 ─────────────────────────────────────────────────────────────

    std::cout << "\n\n[Exit] " << frame_count << " frames, "
              << detect_count << " detections, "
              << fire_count << " fire commands\n";

    serial.close();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
