#include "kalman_filter.hpp"

namespace hnu25 {

KalmanFilter::KalmanFilter() : initialized_(false) {
    x_.setZero();
    P_.setIdentity();
}

void KalmanFilter::init(const Eigen::Matrix<double, 6, 1>& x0,
                        const Eigen::Matrix<double, 6, 6>& P0) {
    x_ = x0;
    P_ = P0;
    initialized_ = true;
}

void KalmanFilter::predict(const Eigen::Matrix<double, 6, 6>& F,
                           const Eigen::Matrix<double, 6, 6>& Q) {
    if (!initialized_) return;
    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q;
}

void KalmanFilter::update(const Eigen::Vector3d& z,
                          const Eigen::Matrix<double, 3, 6>& H,
                          const Eigen::Matrix<double, 3, 3>& R) {
    if (!initialized_) return;

    Eigen::Vector3d y = z - H * x_;
    Eigen::Matrix3d S = H * P_ * H.transpose() + R;
    Eigen::Matrix<double, 6, 3> K = P_ * H.transpose() * S.inverse();

    x_ = x_ + K * y;

    Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
    P_ = (I - K * H) * P_;
}

}  // namespace hnu25
