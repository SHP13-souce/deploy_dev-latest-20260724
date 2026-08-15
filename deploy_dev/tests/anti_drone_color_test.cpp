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

// 640 x 480 CV_8UC3 background, deliberately a non-white "grass-like" green
// so the white-mask stage has to actually separate the target from background.
cv::Mat makeBackground() {
    return cv::Mat(
        kImageHeight,
        kImageWidth,
        CV_8UC3,
        cv::Scalar(40, 100, 40));
}

// A clean 200x200 white board centered at (320, 240), with no red painted.
cv::Mat makeBoardWithoutRed() {
    cv::Mat image = makeBackground();
    cv::rectangle(image, cv::Point(220, 140), cv::Point(420, 340),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    return image;
}

double euclidean(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return std::sqrt(dx * dx + dy * dy);
}

// Test 1: a centered red bullseye on the board is a strong, valid target.
void testCenteredBullseye() {
    cv::Mat image = makeBoardWithoutRed();
    cv::circle(image, cv::Point(320, 240), 35, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(!result.empty(), "centered bullseye is detected");
    if (result.empty()) {
        return;
    }

    const auto& obs = result.front();
    check(obs.from_cv, "from_cv == true");
    check(!obs.from_yolo, "from_yolo == false");
    check(obs.corners_valid, "corners_valid == true");
    check(obs.bullseye_valid, "bullseye_valid == true");

    check(euclidean(obs.bullseye_center, cv::Point2f(320.0F, 240.0F)) <= 3.0,
          "bullseye_center near (320, 240)");

    check(std::isfinite(obs.cv_score), "cv_score is finite");
    check(obs.cv_score >= 0.0F && obs.cv_score <= 1.0F, "cv_score in [0, 1]");
    check(obs.cv_score > 0.90F, "cv_score > 0.90 for ideal bullseye");
}

// Test 2: a board with no red is not a final target under default fusion.
void testNoRedRejectedByDefaultFusion() {
    cv::Mat image = makeBoardWithoutRed();

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(result.empty(), "no-red board is rejected by default fusion");
}

// Test 3: a red blob clearly off-center must not be a valid bullseye.
void testOffCenterRedRejected() {
    cv::Mat image = makeBoardWithoutRed();
    cv::circle(image, cv::Point(385, 305), 20, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    // Preferred: filtered entirely. Acceptable: survives but without a
    // valid bullseye.
    if (!result.empty()) {
        check(!result.front().bullseye_valid,
              "off-center red is not a valid bullseye");
    }
}

// Test 4: red outside the board must be isolated by the board mask.
void testExternalRedIgnored() {
    cv::Mat image = makeBoardWithoutRed();
    cv::circle(image, cv::Point(500, 240), 45, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(result.empty(), "external red is ignored by board mask");
}

// Test 5: the second (high) hue red range participates in detection.
void testHighHueRedRange() {
    // Build a saturated red in the second hue range via HSV -> BGR.
    const cv::Mat hsv_pixel(1, 1, CV_8UC3, cv::Scalar(175, 255, 255));
    cv::Mat bgr_pixel;
    cv::cvtColor(hsv_pixel, bgr_pixel, cv::COLOR_HSV2BGR);
    const cv::Vec3b px = bgr_pixel.at<cv::Vec3b>(0, 0);
    const cv::Scalar color(px[0], px[1], px[2]);

    cv::Mat image = makeBoardWithoutRed();
    cv::circle(image, cv::Point(320, 240), 35, color, cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(!result.empty(), "high-hue red bullseye is detected");
    if (result.empty()) {
        return;
    }

    const auto& obs = result.front();
    check(obs.bullseye_valid, "high-hue red bullseye_valid == true");
    check(euclidean(obs.bullseye_center, cv::Point2f(320.0F, 240.0F)) <= 3.0,
          "high-hue bullseye_center near (320, 240)");
}

// Test 6: a tiny red speck is not enough red evidence to be a bullseye.
void testTinyRedEvidence() {
    cv::Mat image = makeBoardWithoutRed();
    cv::circle(image, cv::Point(320, 240), 2, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    // Preferred: filtered entirely. Acceptable: survives but without a
    // valid bullseye.
    if (!result.empty()) {
        check(!result.front().bullseye_valid,
              "tiny red is not a reliable bullseye");
    }
}

// Test 7: a mildly off-center red blob still forms a valid bullseye.
void testSlightlyOffCenterBullseye() {
    cv::Mat image = makeBoardWithoutRed();
    cv::circle(image, cv::Point(335, 250), 30, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(!result.empty(), "slightly off-center bullseye is detected");
    if (result.empty()) {
        return;
    }

    const auto& obs = result.front();
    check(obs.bullseye_valid, "slightly off-center bullseye_valid == true");
    check(euclidean(obs.bullseye_center, cv::Point2f(335.0F, 250.0F)) <= 3.0,
          "bullseye_center near (335, 250)");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"centered bullseye", testCenteredBullseye},
        {"no red rejected", testNoRedRejectedByDefaultFusion},
        {"off-center red rejected", testOffCenterRedRejected},
        {"external red ignored", testExternalRedIgnored},
        {"high hue red range", testHighHueRedRange},
        {"tiny red evidence", testTinyRedEvidence},
        {"slightly off-center bullseye", testSlightlyOffCenterBullseye},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone color tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
