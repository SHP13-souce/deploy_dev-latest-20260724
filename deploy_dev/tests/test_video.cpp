/**
 * @file    tests/test_video.cpp
 * @brief   视频文件测试 —— 输出标注视频用于评估
 *
 * 用法: ./build/test_video [video.avi] [config.yaml] [output.avi]
 */

#include "shared/types.hpp"
#include "detection/yolo26_detector.hpp"
#include "solve/pnp_solver.hpp"
#include "solve/predictor.hpp"
#include "communication/bcp_packet.hpp"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace hnu25;
using namespace std::chrono;

// ─── 配置 (同 pipeline/main.cpp) ─────────────────────────────────────────

struct TestConfig {
    std::string model_path  = "assets/models/hebei_at_nn/praysky_coord_noe2e_0331_640x640.onnx";
    float conf_threshold    = 0.25f;
    Color  enemy_color      = Color::RED;
    double fx = 1100.0, fy = 1100.0, cx = 720.0, cy = 540.0;
    double predict_time  = 0.05;
    double max_lost_time = 0.30;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line;
        while (std::getline(f, line)) {
            auto p = line.find(':');
            if (p == std::string::npos) continue;
            std::string k = line.substr(0, p), v = line.substr(p + 1);
            k.erase(0, k.find_first_not_of(" \t"));
            k.erase(k.find_last_not_of(" \t\r\n") + 1);
            v.erase(0, v.find_first_not_of(" \t"));
            v.erase(v.find_last_not_of(" \t\r\n") + 1);
            try {
                if (k == "model_path") model_path = v;
                else if (k == "conf_threshold") conf_threshold = std::stof(v);
                else if (k == "enemy_color") {
                    enemy_color = (v == "blue"||v=="BLUE") ? Color::BLUE : Color::RED;
                }
                else if (k == "fx") fx=std::stod(v);
                else if (k == "fy") fy=std::stod(v);
                else if (k == "cx") cx=std::stod(v);
                else if (k == "cy") cy=std::stod(v);
                else if (k == "predict_time") predict_time=std::stod(v);
                else if (k == "max_lost_time") max_lost_time=std::stod(v);
            } catch(...) {}
        }
        return true;
    }
};

inline double rad2deg(double r) { return r * 57.29577951308232; }

inline const DetectedArmor* pickBest(const std::vector<DetectedArmor>& armors, Color enemy) {
    const DetectedArmor* best = nullptr;
    double best_dist = 1e9;
    for (const auto& a : armors) {
        if (a.color != enemy) continue;
        auto cx = static_cast<float>(a.box.x + a.box.width/2.0);
        auto cy = static_cast<float>(a.box.y + a.box.height/2.0);
        double d = cv::norm(cv::Point2f(cx, cy) - cv::Point2f(720, 540));
        if (d < best_dist) { best_dist = d; best = &a; }
    }
    if (!best) {
        for (const auto& a : armors) {
            auto cx = static_cast<float>(a.box.x + a.box.width/2.0);
            auto cy = static_cast<float>(a.box.y + a.box.height/2.0);
            double d = cv::norm(cv::Point2f(cx, cy) - cv::Point2f(720, 540));
            if (d < best_dist) { best_dist = d; best = &a; }
        }
    }
    return best;
}

int main(int argc, char** argv) {
    std::string video_path  = (argc >= 2) ? argv[1] : "demo.avi";
    std::string config_path = (argc >= 3) ? argv[2] : "config/test.yaml";
    std::string output_path = (argc >= 4) ? argv[3] : "demo_result.avi";

    TestConfig cfg;
    cfg.load(config_path);

    std::cout << "=== HNU_NHS_Vision-25 Test ===\n";
    std::cout << "video:  " << video_path << "\n";
    std::cout << "config: " << config_path << "\n";

    // Init
    Yolo26Detector detector(cfg.model_path, cfg.conf_threshold);
    PnPSolver solver(cfg.fx, cfg.fy, cfg.cx, cfg.cy);
    solver.setEnemyColor(cfg.enemy_color);
    Predictor predictor;
    predictor.setPredictTime(cfg.predict_time);
    predictor.setMaxLostTime(cfg.max_lost_time);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[Fatal] cannot open video\n";
        return 1;
    }

    auto frame_size = cv::Size(
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),
        static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
    double fps     = cap.get(cv::CAP_PROP_FPS);
    auto total     = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    const bool write_video = output_path != "-";
    cv::VideoWriter writer;
    if (write_video) {
        writer.open(output_path, cv::VideoWriter::fourcc('M','J','P','G'),
                    fps > 0 ? fps : 30, frame_size);
    }

    std::cout << "video: " << frame_size.width << "x" << frame_size.height
              << " @ " << fps << "fps, " << total << " frames\n";
    std::cout << "output: " << (write_video ? output_path : "disabled") << "\n\n";

    int frame_count = 0, detect_count = 0, track_count = 0, target_id = 0;
    double detect_total_ms = 0.0, algorithm_total_ms = 0.0;
    auto last_status = steady_clock::now();

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        auto t0 = steady_clock::now();
        frame_count++;

        // Detection
        auto armors = detector.detect(frame);
        auto t_detect = steady_clock::now();
        detect_total_ms += duration<double, std::milli>(t_detect - t0).count();
        bool has_det = !armors.empty();
        if (has_det) detect_count++;

        // Pick best
        const auto* best = pickBest(armors, cfg.enemy_color);

        TargetMeasurement meas;
        meas.timestamp = t0;
        meas.detected  = (best != nullptr);

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

        auto state = predictor.update(meas);
        if (state.valid) track_count++;

        // Draw
        cv::Mat display = frame.clone();
        if (best) {
            cv::Scalar box_color = (best->color == cfg.enemy_color)
                                       ? cv::Scalar(0, 255, 0)
                                       : cv::Scalar(0, 165, 255);
            cv::rectangle(display, best->box, box_color, 2);
            for (const auto& kp : best->keypoints)
                cv::circle(display, kp, 3, cv::Scalar(0, 0, 255), -1);

            char conf_text[32];
            snprintf(conf_text, sizeof(conf_text), "%.2f %s %s",
                     best->confidence,
                     color_name(best->color),
                     best->kind == ArmorKind::SMALL ? "small" : "LARGE");
            cv::putText(display, conf_text,
                        cv::Point(best->box.x, best->box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
        }

        if (state.valid) {
            double px = state.predicted_position.x();
            double py = state.predicted_position.y();
            double pz = state.predicted_position.z();
            double dist = state.predicted_position.norm();

            char buf[128];
            snprintf(buf, sizeof(buf), "D=%.1fm yaw=%.1fdeg",
                     dist, rad2deg(std::atan2(py, px)));
            cv::putText(display, buf, cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 255), 2);

            // 预测点在图像中心
            cv::circle(display, cv::Point(720, 540), 5,
                       cv::Scalar(0, 0, 255), -1);
        }

        // 来源标签
        cv::putText(display, "YOLO26 | PnP+EKF | BCP",
                    cv::Point(10, frame_size.height - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(255, 255, 255), 1);

        auto t_algorithm = steady_clock::now();
        algorithm_total_ms += duration<double, std::milli>(t_algorithm - t0).count();
        if (write_video) writer.write(display);

        if (frame_count % 100 == 0 || frame_count == total) {
            auto t1 = steady_clock::now();
            double avg_ms = duration_cast<duration<double, std::milli>>(t1 - t0).count();
            std::cout << "[frame " << frame_count << "/" << total << "] "
                      << "det=" << detect_count << " (" << int(100.0*detect_count/frame_count) << "%) "
                      << "track=" << track_count << " (" << int(100.0*track_count/frame_count) << "%) "
                      << "| " << int(avg_ms) << "ms\n";
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "video test complete: " << frame_count << " frames\n";
    std::cout << "frames with detection: " << detect_count
              << " (" << int(100.0*detect_count/frame_count) << "%)\n";
    std::cout << "frames with tracking: " << track_count
              << " (" << int(100.0*track_count/frame_count) << "%)\n";
    std::cout << "mean detection: " << detect_total_ms/frame_count << " ms ("
              << 1000.0*frame_count/detect_total_ms << " FPS)\n";
    std::cout << "mean algorithm: " << algorithm_total_ms/frame_count << " ms ("
              << 1000.0*frame_count/algorithm_total_ms << " FPS)\n";
    if (write_video) std::cout << "output saved to: " << output_path << "\n";
    std::cout << "========================================\n";

    cap.release();
    if (write_video) writer.release();
    return 0;
}
