#include "anti_drone/config.hpp"
#include "anti_drone/traditional_detector.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

void printPoint2f(const char* name, const cv::Point2f& p) {
    std::cout << "  " << name << ": (" << std::fixed << std::setprecision(2)
              << p.x << ", " << p.y << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cout << "Usage: anti_drone_image_preview "
                     "<input_image> <output_image> <config_yaml>\n";
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const std::string config_path = argv[3];

    // ── Load config ───────────────────────────────────────────────────────
    hnu25::anti_drone::TraditionalDetectorConfig config;
    try {
        config = hnu25::anti_drone::loadTraditionalDetectorConfig(config_path);
    } catch (const std::exception& error) {
        std::cerr << "Failed to load config: " << config_path << ": "
                  << error.what() << '\n';
        return 4;
    }

    // ── Read image ────────────────────────────────────────────────────────
    const cv::Mat image = cv::imread(input_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cout << "Failed to read input image: " << input_path << '\n';
        return 2;
    }
    // imread with IMREAD_COLOR guarantees a non-empty result is CV_8UC3 BGR.

    // ── Detect ────────────────────────────────────────────────────────────
    hnu25::anti_drone::TraditionalTargetDetector detector(config);
    const auto observations = detector.detect(image);

    // ── Console summary ───────────────────────────────────────────────────
    std::cout << "Input: " << input_path << '\n';
    std::cout << "Config: " << config_path << '\n';
    std::cout << "Image: " << image.cols << 'x' << image.rows << '\n';
    std::cout << "Detections: " << observations.size() << '\n';

    // ── Per-target console output ─────────────────────────────────────────
    for (std::size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];
        std::cout << "Target " << i << '\n';

        std::cout << "  cv_score: " << std::fixed << std::setprecision(2)
                  << obs.cv_score << '\n';

        std::cout << "  bbox: x=" << obs.box.x << " y=" << obs.box.y
                  << " width=" << obs.box.width
                  << " height=" << obs.box.height << '\n';

        printPoint2f("center", obs.center);

        std::cout << "  corners_valid: " << std::boolalpha
                  << obs.corners_valid << '\n';
        std::cout << "  bullseye_valid: " << obs.bullseye_valid << '\n';
        std::cout << "  from_cv: " << obs.from_cv << '\n';
        std::cout << "  from_yolo: " << obs.from_yolo << '\n';

        if (obs.corners_valid) {
            printPoint2f("TL", obs.corners[0]);
            printPoint2f("TR", obs.corners[1]);
            printPoint2f("BR", obs.corners[2]);
            printPoint2f("BL", obs.corners[3]);
        }

        if (obs.bullseye_valid) {
            printPoint2f("bullseye_center", obs.bullseye_center);
        }
    }

    // ── Visualization (drawn only on a clone) ─────────────────────────────
    cv::Mat visualization = image.clone();

    for (std::size_t i = 0; i < observations.size(); ++i) {
        const auto& obs = observations[i];

        // Bounding box.
        cv::rectangle(visualization, obs.box, cv::Scalar(0, 255, 0), 2);

        // Corners + outline, strictly from the stored corners.
        if (obs.corners_valid) {
            const cv::Point tl(cvRound(obs.corners[0].x),
                               cvRound(obs.corners[0].y));
            const cv::Point tr(cvRound(obs.corners[1].x),
                               cvRound(obs.corners[1].y));
            const cv::Point br(cvRound(obs.corners[2].x),
                               cvRound(obs.corners[2].y));
            const cv::Point bl(cvRound(obs.corners[3].x),
                               cvRound(obs.corners[3].y));

            cv::circle(visualization, tl, 4, cv::Scalar(0, 0, 255),
                       cv::FILLED);
            cv::circle(visualization, tr, 4, cv::Scalar(0, 0, 255),
                       cv::FILLED);
            cv::circle(visualization, br, 4, cv::Scalar(0, 0, 255),
                       cv::FILLED);
            cv::circle(visualization, bl, 4, cv::Scalar(0, 0, 255),
                       cv::FILLED);

            cv::line(visualization, tl, tr, cv::Scalar(0, 255, 255), 2);
            cv::line(visualization, tr, br, cv::Scalar(0, 255, 255), 2);
            cv::line(visualization, br, bl, cv::Scalar(0, 255, 255), 2);
            cv::line(visualization, bl, tl, cv::Scalar(0, 255, 255), 2);
        }

        // Board center cross marker.
        cv::drawMarker(
            visualization,
            cv::Point(cvRound(obs.center.x), cvRound(obs.center.y)),
            cv::Scalar(255, 255, 255),
            cv::MARKER_CROSS,
            12,
            2);

        // Bullseye marker (distinct style).
        if (obs.bullseye_valid) {
            cv::drawMarker(
                visualization,
                cv::Point(cvRound(obs.bullseye_center.x),
                          cvRound(obs.bullseye_center.y)),
                cv::Scalar(0, 0, 255),
                cv::MARKER_TILTED_CROSS,
                12,
                2);
        }

        // Short text label.
        std::ostringstream label;
        label << '#' << i
              << " cv=" << std::fixed << std::setprecision(2) << obs.cv_score
              << " B=" << (obs.bullseye_valid ? 1 : 0)
              << " C=" << (obs.corners_valid ? 1 : 0);

        const int label_y = std::max(15, obs.box.y - 8);
        cv::putText(
            visualization,
            label.str(),
            cv::Point(obs.box.x, label_y),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1);
    }

    // ── Save ──────────────────────────────────────────────────────────────
    if (!cv::imwrite(output_path, visualization)) {
        std::cout << "Failed to write output image: " << output_path << '\n';
        return 3;
    }

    std::cout << "Saved visualization: " << output_path << '\n';
    return 0;
}
