#include "anti_drone/traditional_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kImageWidth = 640;
constexpr int kImageHeight = 480;

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

// 640 x 480 CV_8UC3 background, deliberately a non-white "grass-like" green.
cv::Mat makeBackground() {
    return cv::Mat(
        kImageHeight,
        kImageWidth,
        CV_8UC3,
        cv::Scalar(40, 100, 40));
}

double euclidean(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return std::sqrt(dx * dx + dy * dy);
}

// Paint one target: a white board with a red/white concentric-ring bullseye
// (the real 50 cm x 50 cm competition target). Ring radii are derived from the
// board size so the target scales with board size.
void drawTarget(cv::Mat& image, const cv::Point& tl, const cv::Point& br) {
    cv::rectangle(image, tl, br, cv::Scalar(255, 255, 255), cv::FILLED);

    const cv::Point center((tl.x + br.x) / 2, (tl.y + br.y) / 2);
    const int half = std::max(1, (br.x - tl.x) / 2);

    const cv::Scalar red(0, 0, 255);
    const cv::Scalar white(255, 255, 255);

    cv::circle(image, center, static_cast<int>(std::lround(0.72 * half)), red,
               cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.60 * half)), white,
               cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.50 * half)), red,
               cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.38 * half)), white,
               cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.24 * half)), red,
               cv::FILLED);
}

// Test 1: a single target yields exactly one observation with a valid
// bullseye.
void testSingleTarget() {
    cv::Mat image = makeBackground();
    drawTarget(image, cv::Point(120, 140), cv::Point(280, 300));

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(result.size() == 1, "single target -> one observation");
    if (result.empty()) {
        return;
    }
    check(result.front().bullseye_valid, "single target bullseye_valid == true");
}

// Two clearly separated targets must both survive NMS.
cv::Mat makeTwoSeparatedTargets() {
    cv::Mat image = makeBackground();
    // Left target.
    drawTarget(image, cv::Point(60, 170), cv::Point(200, 310));
    // Right target.
    drawTarget(image, cv::Point(400, 170), cv::Point(540, 310));
    return image;
}

void checkTwoTargets(const std::vector<hnu25::anti_drone::TargetObservation>& result,
                     const std::string& context) {
    check(result.size() == 2, context + ": two separated targets survive");
    if (result.size() != 2) {
        return;
    }

    for (const auto& obs : result) {
        check(obs.from_cv, context + ": from_cv == true");
        check(obs.bullseye_valid, context + ": bullseye_valid == true");
    }

    check(euclidean(result[0].center, result[1].center) > 200.0,
          context + ": two centers are clearly separated");
}

// Test 2: default detector keeps both separated targets.
void testTwoSeparatedTargets() {
    cv::Mat image = makeTwoSeparatedTargets();

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    checkTwoTargets(result, "default threshold");
}

// Test 3: threshold == 1.0 must keep both (IoU=0, 0 > 1 is false).
void testNmsThresholdOne() {
    cv::Mat image = makeTwoSeparatedTargets();

    hnu25::anti_drone::TraditionalDetectorConfig config;
    config.nms_iou_threshold = 1.0F;
    hnu25::anti_drone::TraditionalTargetDetector detector(config);

    const auto result = detector.detect(image);
    checkTwoTargets(result, "threshold 1.0");
}

// Test 4: an illegal high threshold clamps to 1.0 and keeps both.
void testNmsIllegalHighThreshold() {
    cv::Mat image = makeTwoSeparatedTargets();

    hnu25::anti_drone::TraditionalDetectorConfig config;
    config.nms_iou_threshold = 5.0F;
    hnu25::anti_drone::TraditionalTargetDetector detector(config);

    const auto result = detector.detect(image);
    checkTwoTargets(result, "threshold 5.0 clamped to 1.0");
}

// Test 5: an illegal negative threshold clamps to 0.0; with IoU == 0 the
// strict `>` test keeps both, proving `>=` is not used.
void testNmsIllegalNegativeThreshold() {
    cv::Mat image = makeTwoSeparatedTargets();

    hnu25::anti_drone::TraditionalDetectorConfig config;
    config.nms_iou_threshold = -1.0F;
    hnu25::anti_drone::TraditionalTargetDetector detector(config);

    const auto result = detector.detect(image);
    checkTwoTargets(result, "threshold -1.0 clamped to 0.0");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"single target", testSingleTarget},
        {"two separated targets", testTwoSeparatedTargets},
        {"nms threshold 1.0", testNmsThresholdOne},
        {"nms illegal high threshold", testNmsIllegalHighThreshold},
        {"nms illegal negative threshold", testNmsIllegalNegativeThreshold},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone nms tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
