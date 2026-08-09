#pragma once
/// @file   solve/pnp_solver.hpp
/// @brief  PnP 姿态解算器 —— 解算组
///
/// 输入: DetectedArmor (4个图像关键点)
/// 输出: PnPResult (3D 位置 + yaw 角)

#include "shared/types.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <string>

namespace hnu25 {

class PnPSolver {
public:
    /// @param fx,fy,cx,cy  相机内参 (默认值可在配置文件覆盖)
    explicit PnPSolver(double fx = 1100.0, double fy = 1100.0,
                       double cx = 720.0,  double cy = 540.0);

    /// 从配置文件加载参数
    explicit PnPSolver(const std::string& config_path);

    /// 设置是否使用敌方颜色过滤
    void setEnemyColor(Color color) { enemy_color_ = color; }

    /// 执行 PnP 解算
    PnPResult solve(const DetectedArmor& armor);

    /// 获取相机内参矩阵 (3x3, CV_64F)
    const cv::Mat& cameraMatrix() const { return camera_matrix_; }

private:
    static std::vector<cv::Point3f> smallArmorPoints();
    static std::vector<cv::Point3f> largeArmorPoints();

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    Color   enemy_color_ = Color::RED;
    double  max_reproj_error_ = 15.0;
};

}  // namespace hnu25
