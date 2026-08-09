#include "motion_model.hpp"

#include <algorithm>

namespace hnu25 {

MotionModel::MotionModel() : dt_(0.01) {
    F_.setIdentity();
    H_.setZero();
    Q_.setZero();
    R_.setZero();
    buildH();
    buildQ();
    buildR();
}

void MotionModel::update(double dt) {
    dt_ = std::max(1e-4, std::min(dt, 0.2));

    F_.setIdentity();
    F_(0, 3) = dt_;
    F_(1, 4) = dt_;
    F_(2, 5) = dt_;

    buildQ();
}

void MotionModel::setProcessNoise(double position_sigma, double velocity_sigma) {
    position_sigma_ = std::max(1e-6, position_sigma);
    velocity_sigma_ = std::max(1e-6, velocity_sigma);
    buildQ();
}

void MotionModel::setMeasurementNoise(double measurement_sigma) {
    measurement_sigma_ = std::max(1e-6, measurement_sigma);
    buildR();
}

void MotionModel::buildH() {
    H_.setZero();
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;
}

void MotionModel::buildQ() {
    Q_.setZero();
    Q_(0, 0) = position_sigma_;
    Q_(1, 1) = position_sigma_;
    Q_(2, 2) = position_sigma_;
    Q_(3, 3) = velocity_sigma_ * dt_;
    Q_(4, 4) = velocity_sigma_ * dt_;
    Q_(5, 5) = velocity_sigma_ * dt_;
}

void MotionModel::buildR() {
    R_.setZero();
    R_(0, 0) = measurement_sigma_;
    R_(1, 1) = measurement_sigma_;
    R_(2, 2) = measurement_sigma_;
}

}  // namespace hnu25
