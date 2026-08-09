#pragma once
/// @file   solve/predictor.hpp
/// @brief  EKF 预测器 —— 解算组
///
/// 功能: 使用卡尔曼滤波器对目标位置/速度进行预测
///       无检测时纯预测（丢失容错），有检测时预测+更新

#include "shared/types.hpp"
#include "motion_model.hpp"
#include "kalman_filter.hpp"

namespace hnu25 {

class Predictor {
public:
    Predictor();

    /// 核心接口: 输入测量值 -> 输出滤波+预测后的状态
    TargetState update(const TargetMeasurement& measurement);

    /// 重置所有状态
    void reset();

    /// 前向预测时间 (默认 50ms，影响 predicted_position)
    void setPredictTime(double seconds);

    /// 最大丢失容忍时间 (默认 300ms，超时自动重置)
    void setMaxLostTime(double seconds);

private:
    void initFilter(const TargetMeasurement& m);

    MotionModel  model_;
    KalmanFilter kf_;

    bool initialized_ = false;
    int  current_id_  = -1;
    std::chrono::steady_clock::time_point last_timestamp_;

    double predict_time_  = 0.05;   // 50ms 弹道预测
    double max_lost_time_ = 0.30;   // 300ms 丢失容忍
    double lost_time_     = 0.0;
};

}  // namespace hnu25
