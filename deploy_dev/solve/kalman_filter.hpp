#pragma once
/// @file   solve/kalman_filter.hpp
/// @brief  线性卡尔曼滤波器 (6-state, 3-measurement) —— 解算组

#include <Eigen/Dense>

namespace hnu25 {

class KalmanFilter {
public:
    KalmanFilter();

    /// 初始化状态和协方差
    void init(const Eigen::Matrix<double, 6, 1>& x0,
              const Eigen::Matrix<double, 6, 6>& P0);

    /// 预测步: x = F*x, P = F*P*F' + Q
    void predict(const Eigen::Matrix<double, 6, 6>& F,
                 const Eigen::Matrix<double, 6, 6>& Q);

    /// 更新步: K = P*H'*(H*P*H' + R)^-1, x = x + K*(z - H*x)
    void update(const Eigen::Vector3d& z,
                const Eigen::Matrix<double, 3, 6>& H,
                const Eigen::Matrix<double, 3, 3>& R);

    const Eigen::Matrix<double, 6, 1>& state()      const { return x_; }
    const Eigen::Matrix<double, 6, 6>& covariance() const { return P_; }

private:
    Eigen::Matrix<double, 6, 1> x_;
    Eigen::Matrix<double, 6, 6> P_;
    bool initialized_ = false;
};

}  // namespace hnu25
