/**
 * @file    shared/types.hpp
 * @brief   统一类型定义 —— HNU_NHS_Vision-25 三组融合项目
 *
 * 这个文件定义了检测→解算→通信三阶段之间传递的核心数据结构，
 * 避免各组使用不同结构体导致接口混乱。
 *
 * 来源:
 *   - DetectedArmor:  deploy_dev 检测器与下游解算之间的统一格式
 *   - PnPResult:      解算组 PnP 解算的中间结果
 *   - TargetMeasurement / TargetState: 解算组 EKF 预测器 (kalman_filter + motion_model)
 *   - GimbalCommand:  通信组 BCP 协议需要的数据
 */

#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <string>
#include <vector>

namespace hnu25 {

// ─── 装甲板颜色 ────────────────────────────────────────────────────────────
enum class Color : uint8_t { RED = 0, BLUE = 1, EXTINGUISH = 2, PURPLE = 3 };

inline const char* color_name(Color c) {
    switch (c) {
        case Color::RED:         return "红";
        case Color::BLUE:        return "蓝";
        case Color::EXTINGUISH:  return "灭灯";
        case Color::PURPLE:      return "紫";
        default:                 return "未知";
    }
}

// ─── 装甲板大小 ────────────────────────────────────────────────────────────
enum class ArmorKind : uint8_t { SMALL = 0, LARGE = 1 };

// ─── 装甲板编号 ─────────────────────────────────────────────────────────────
enum class ArmorLabel : uint8_t {
    SENTRY = 0,   // 哨兵
    ONE    = 1,   // 1号 (英雄, 大眼睛)
    TWO    = 2,
    THREE  = 3,
    FOUR   = 4,
    FIVE   = 5,
    OUTPOST= 6,   // 前哨站
    BASE   = 7,   // 基地
    UNKNOWN= 99
};

// ─── 检测结果：单个装甲板 ─────────────────────────────────────────────────
struct DetectedArmor {
    std::vector<cv::Point2f> keypoints;  // 4个关键点: TL, TR, BR, BL
    cv::Rect box;                        // 外接框 (原图坐标)
    float confidence = 0.0f;             // 检测置信度
    int   class_id   = 0;                // Backend-native class ID; use label for semantics.
    Color color     = Color::RED;
    ArmorKind kind  = ArmorKind::SMALL;
    ArmorLabel label = ArmorLabel::UNKNOWN;
    std::chrono::steady_clock::time_point timestamp;
};

// ─── PnP 解算结果 ──────────────────────────────────────────────────────────
struct PnPResult {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();  // 世界坐标系 (m)
    double yaw    = 0.0;   // yaw 角 (rad)
    double pitch  = 0.0;   // pitch 角 (rad)
    double distance = 0.0; // 距离 (m)
    bool valid = false;
};

// ─── 观测／测量（给 EKF） ──────────────────────────────────────────────────
struct TargetMeasurement {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    double yaw = 0.0;
    std::chrono::steady_clock::time_point timestamp;
    bool detected = false;
    int  id = -1;
};

// ─── EKF 滤波后状态 ────────────────────────────────────────────────────────
struct TargetState {
    Eigen::Vector3d position          = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity          = Eigen::Vector3d::Zero();
    Eigen::Vector3d predicted_position = Eigen::Vector3d::Zero();
    double yaw   = 0.0;
    bool   valid = false;
    int    id    = -1;
};

// ─── 发给云台的指令 ────────────────────────────────────────────────────────
struct GimbalCommand {
    double yaw       = 0.0;   // yaw 角度 (rad)
    double pitch     = 0.0;   // pitch 角度 (rad)
    double yaw_vel   = 0.0;   // yaw 角速度 (rad/s)
    double pitch_vel = 0.0;   // pitch 角速度 (rad/s)
    bool   fire      = false; // 是否开火
};

}  // namespace hnu25
