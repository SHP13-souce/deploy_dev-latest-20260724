#pragma once
/// @file   solve/motion_model.hpp
/// @brief  6-state 匀速运动模型 —— 解算组
///
/// 状态: x = [px, py, pz, vx, vy, vz]^T
/// 观测: z = [px, py, pz]^T
///
/// 提供 F(6x6), H(3x6), Q(6x6), R(3x3) 四个矩阵

#include <Eigen/Dense>

namespace hnu25 {

class MotionModel {
public:
    MotionModel();

    /// 根据帧间隔更新状态转移矩阵 F 和噪声矩阵 Q
    void update(double dt);

    /// 调参: 过程噪声 (默认 position=0.01, velocity=0.5)
    void setProcessNoise(double position_sigma, double velocity_sigma);

    /// 调参: 测量噪声 (默认 0.05)
    void setMeasurementNoise(double measurement_sigma);

    const Eigen::Matrix<double, 6, 6>& F() const { return F_; }
    const Eigen::Matrix<double, 3, 6>& H() const { return H_; }
    const Eigen::Matrix<double, 6, 6>& Q() const { return Q_; }
    const Eigen::Matrix<double, 3, 3>& R() const { return R_; }

private:
    void buildH();
    void buildQ();
    void buildR();

    double dt_;
    double position_sigma_  = 0.01;
    double velocity_sigma_  = 0.5;
    double measurement_sigma_ = 0.05;

    Eigen::Matrix<double, 6, 6> F_;
    Eigen::Matrix<double, 3, 6> H_;
    Eigen::Matrix<double, 6, 6> Q_;
    Eigen::Matrix<double, 3, 3> R_;
};

}  // namespace hnu25
