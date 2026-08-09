#include "pnp_solver.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace hnu25 {

namespace {

// 模型关键点按两侧灯条端点标注，因此 PnP 使用灯条中心间距与灯条长度，
// 不是 AM01/AM11 装甲板组件的 140x125 / 235x127 mm 整体外形尺寸。
constexpr double SMALL_ARMOR_WIDTH = 0.135;
constexpr double LARGE_ARMOR_WIDTH = 0.230;
constexpr double LIGHTBAR_LENGTH = 0.056;

}  // namespace

// ─── Constructor ────────────────────────────────────────────────────────────

PnPSolver::PnPSolver(double fx, double fy, double cx, double cy) {
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix_.at<double>(0, 0) = fx;
    camera_matrix_.at<double>(1, 1) = fy;
    camera_matrix_.at<double>(0, 2) = cx;
    camera_matrix_.at<double>(1, 2) = cy;
    dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
}

PnPSolver::PnPSolver(const std::string& path) {
    // 简易 YAML 逐行解析 (避免 yaml-cpp 依赖)
    camera_matrix_ = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix_.at<double>(0, 0) = 1100.0;
    camera_matrix_.at<double>(1, 1) = 1100.0;
    camera_matrix_.at<double>(0, 2) = 720.0;
    camera_matrix_.at<double>(1, 2) = 540.0;
    dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);

    std::ifstream file(path);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        // 简单 key: value 格式
        auto pos = line.find(':');
        if (pos == std::string::npos) continue;
        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);
        // 去除空格
        key.erase(0, key.find_first_not_of(" \t"));
        val.erase(0, val.find_first_not_of(" \t"));
        val.erase(val.find_last_not_of(" \t\r\n") + 1);

        try {
            double v = std::stod(val);
            if      (key == "fx") camera_matrix_.at<double>(0, 0) = v;
            else if (key == "fy") camera_matrix_.at<double>(1, 1) = v;
            else if (key == "cx") camera_matrix_.at<double>(0, 2) = v;
            else if (key == "cy") camera_matrix_.at<double>(1, 2) = v;
            else if (key == "max_reproj_error") max_reproj_error_ = v;
        } catch (...) {}
    }
}

// ─── Solve ──────────────────────────────────────────────────────────────────

PnPResult PnPSolver::solve(const DetectedArmor& armor) {
    PnPResult result;

    if (armor.keypoints.size() != 4) return result;

    const auto& obj_pts = (armor.kind == ArmorKind::LARGE)
                              ? largeArmorPoints() : smallArmorPoints();

    std::vector<cv::Point2f> img_pts;
    img_pts.reserve(4);
    for (const auto& p : armor.keypoints)
        img_pts.emplace_back(p.x, p.y);

    cv::Mat rvec, tvec;
    if (!cv::solvePnP(obj_pts, img_pts, camera_matrix_, dist_coeffs_,
                      rvec, tvec, false, cv::SOLVEPNP_IPPE)) {
        return result;
    }

    // 重投影误差
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj_pts, rvec, tvec, camera_matrix_, dist_coeffs_, proj);
    double err = 0.0;
    for (size_t i = 0; i < 4; ++i)
        err += cv::norm(img_pts[i] - proj[i]);
    err /= 4.0;
    if (err > max_reproj_error_) return result;

    result.position = Eigen::Vector3d(tvec.at<double>(0),
                                      tvec.at<double>(1),
                                      tvec.at<double>(2));
    result.distance = result.position.norm();

    // 解 Euler 角
    cv::Mat rot;
    cv::Rodrigues(rvec, rot);
    double pitch = std::atan2(-rot.at<double>(2, 0),
        std::sqrt(rot.at<double>(2, 1) * rot.at<double>(2, 1) +
                  rot.at<double>(2, 2) * rot.at<double>(2, 2)));
    double yaw = std::atan2(rot.at<double>(1, 0), rot.at<double>(0, 0));

    // yaw 去歧义
    Eigen::Vector3d normal(rot.at<double>(0, 2),
                           rot.at<double>(1, 2),
                           rot.at<double>(2, 2));
    if (normal.dot(Eigen::Vector3d::UnitZ()) < 0)
        yaw += 3.141592653589793;

    const double PI = 3.141592653589793;
    while (yaw >  PI) yaw -= 2.0 * PI;
    while (yaw < -PI) yaw += 2.0 * PI;

    result.yaw   = yaw;
    result.pitch = pitch;
    result.valid = true;

    return result;
}

// ─── Static helpers ────────────────────────────────────────────────────────

std::vector<cv::Point3f> PnPSolver::smallArmorPoints() {
    const double hw = SMALL_ARMOR_WIDTH / 2.0;
    const double hh = LIGHTBAR_LENGTH / 2.0;
    return {
        cv::Point3f(-hw, -hh, 0),
        cv::Point3f( hw, -hh, 0),
        cv::Point3f( hw,  hh, 0),
        cv::Point3f(-hw,  hh, 0)
    };
}

std::vector<cv::Point3f> PnPSolver::largeArmorPoints() {
    const double hw = LARGE_ARMOR_WIDTH / 2.0;
    const double hh = LIGHTBAR_LENGTH / 2.0;
    return {
        cv::Point3f(-hw, -hh, 0),
        cv::Point3f( hw, -hh, 0),
        cv::Point3f( hw,  hh, 0),
        cv::Point3f(-hw,  hh, 0)
    };
}

}  // namespace hnu25
