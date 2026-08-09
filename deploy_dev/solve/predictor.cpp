#include "predictor.hpp"

#include <algorithm>

namespace hnu25 {

Predictor::Predictor() {}

void Predictor::reset() {
    initialized_ = false;
    current_id_  = -1;
    lost_time_   = 0.0;
}

void Predictor::setPredictTime(double seconds) {
    predict_time_ = std::max(0.0, seconds);
}

void Predictor::setMaxLostTime(double seconds) {
    max_lost_time_ = std::max(0.0, seconds);
}

void Predictor::initFilter(const TargetMeasurement& m) {
    Eigen::Matrix<double, 6, 1> x0;
    x0 << m.position.x(), m.position.y(), m.position.z(),
          0.0, 0.0, 0.0;

    Eigen::Matrix<double, 6, 6> P0 = Eigen::Matrix<double, 6, 6>::Identity();
    P0(0, 0) = 0.1;  P0(1, 1) = 0.1;  P0(2, 2) = 0.1;
    P0(3, 3) = 10.0; P0(4, 4) = 10.0; P0(5, 5) = 10.0;

    kf_.init(x0, P0);

    initialized_    = true;
    current_id_     = m.id;
    last_timestamp_ = m.timestamp;
    lost_time_      = 0.0;
}

TargetState Predictor::update(const TargetMeasurement& m) {
    TargetState result;
    result.id  = m.id;
    result.yaw = m.yaw;

    if (!m.detected) {
        if (!initialized_) {
            result.valid = false;
            return result;
        }

        double dt = 0.02;  // 无检测时用固定 dt
        model_.update(dt);
        kf_.predict(model_.F(), model_.Q());

        lost_time_ += dt;
        if (lost_time_ > max_lost_time_) {
            reset();
            result.valid = false;
            return result;
        }
    } else {
        if (!initialized_ || m.id != current_id_) {
            initFilter(m);
        } else {
            double dt = std::chrono::duration<double>(
                m.timestamp - last_timestamp_).count();
            dt = std::max(1e-4, std::min(dt, 0.2));

            model_.update(dt);
            kf_.predict(model_.F(), model_.Q());
            kf_.update(m.position, model_.H(), model_.R());

            last_timestamp_ = m.timestamp;
            lost_time_ = 0.0;
        }
    }

    const auto& x = kf_.state();
    result.position = Eigen::Vector3d(x(0), x(1), x(2));
    result.velocity = Eigen::Vector3d(x(3), x(4), x(5));
    result.predicted_position = result.position + result.velocity * predict_time_;
    result.valid = initialized_;

    return result;
}

}  // namespace hnu25
