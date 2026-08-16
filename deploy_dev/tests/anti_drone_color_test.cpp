#include "anti_drone/traditional_detector.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
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

double euclidean(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return std::sqrt(dx * dx + dy * dy);
}

// A clean white board (no red) centered in the image. Used as the base for the
// negative tests.
cv::Mat makeWhiteBoard(int board_half = 100) {
    cv::Mat image = makeBackground();
    const cv::Point center(kImageWidth / 2, kImageHeight / 2);
    cv::rectangle(image,
                  cv::Point(center.x - board_half, center.y - board_half),
                  cv::Point(center.x + board_half, center.y + board_half),
                  cv::Scalar(255, 255, 255), cv::FILLED);
    return image;
}

// Draws the red/white concentric-ring bullseye (no board) centered at `center`.
// Radii are fractions of `board_half`, largest first, so each smaller filled
// circle overwrites the interior of the previous one and leaves rings behind:
//
//   center RED -> WHITE -> RED -> WHITE -> RED (outer ring)
//
// which from the center outward is exactly the real competition target's
// R W R W R alternation (followed by the white board margin).
void drawConcentricRings(cv::Mat& image,
                         const cv::Point& center,
                         int board_half,
                         const cv::Scalar& red = cv::Scalar(0, 0, 255)) {
    const cv::Scalar white(255, 255, 255);
    cv::circle(image, center, static_cast<int>(std::lround(0.72 * board_half)),
               red, cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.60 * board_half)),
               white, cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.50 * board_half)),
               red, cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.38 * board_half)),
               white, cv::FILLED);
    cv::circle(image, center, static_cast<int>(std::lround(0.24 * board_half)),
               red, cv::FILLED);
}

// A complete centered target: white board + concentric bullseye.
cv::Mat makeConcentricTarget(int board_half = 100) {
    cv::Mat image = makeWhiteBoard(board_half);
    drawConcentricRings(image, cv::Point(kImageWidth / 2, kImageHeight / 2),
                        board_half);
    return image;
}

// Blits a concentric target onto a perspective / rotated quadrilateral `quad`
// (order TL, TR, BR, BL). The target is drawn on a canonical board then warped
// into the image, so the detector must recover the structure through its own
// perspective normalization.
void blitConcentricTarget(cv::Mat& image,
                          const std::array<cv::Point2f, 4>& quad,
                          int board_half) {
    const int side = 2 * board_half;
    cv::Mat canvas(side, side, CV_8UC3, cv::Scalar(255, 255, 255));
    drawConcentricRings(canvas, cv::Point(board_half, board_half), board_half);

    const std::vector<cv::Point2f> src = {
        cv::Point2f(0.0F, 0.0F),
        cv::Point2f(static_cast<float>(side - 1), 0.0F),
        cv::Point2f(static_cast<float>(side - 1), static_cast<float>(side - 1)),
        cv::Point2f(0.0F, static_cast<float>(side - 1)),
    };
    const std::vector<cv::Point2f> dst(quad.begin(), quad.end());
    const cv::Mat H = cv::getPerspectiveTransform(src, dst);
    // Fill the unmapped surround with the same green background so there is no
    // black border artifact.
    cv::warpPerspective(canvas, image, H, image.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, cv::Scalar(40, 100, 40));
}

// Red in the second (high) hue range, so the high-hue branch of the red mask
// is exercised by the warp stage too.
cv::Scalar highHueRed() {
    const cv::Mat hsv_pixel(1, 1, CV_8UC3, cv::Scalar(175, 255, 255));
    cv::Mat bgr_pixel;
    cv::cvtColor(hsv_pixel, bgr_pixel, cv::COLOR_HSV2BGR);
    const cv::Vec3b px = bgr_pixel.at<cv::Vec3b>(0, 0);
    return cv::Scalar(px[0], px[1], px[2]);
}

// ── Positive tests ──────────────────────────────────────────────────────────

// Standard frontal concentric target is detected with valid corners/bullseye.
void testStandardConcentricTarget() {
    cv::Mat image = makeConcentricTarget(100);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(!result.empty(), "standard concentric target is detected");
    if (result.empty()) {
        return;
    }

    const auto& obs = result.front();
    check(obs.from_cv, "from_cv == true");
    check(!obs.from_yolo, "from_yolo == false");
    check(obs.corners_valid, "corners_valid == true");
    check(obs.bullseye_valid, "bullseye_valid == true");
    check(std::isfinite(obs.cv_score), "cv_score is finite");
    check(obs.cv_score >= 0.0F && obs.cv_score <= 1.0F, "cv_score in [0, 1]");
    check(obs.cv_score > 0.65F, "cv_score > 0.65 for the real target");

    const cv::Point2f expected(320.0F, 240.0F);
    check(euclidean(obs.bullseye_center, expected) <= 3.0,
          "bullseye_center near (320, 240)");
}

// Mild perspective is still detected.
void testSlightPerspective() {
    cv::Mat image = makeBackground();
    blitConcentricTarget(
        image,
        {cv::Point2f(235, 140), cv::Point2f(415, 148), cv::Point2f(408, 335),
         cv::Point2f(228, 328)},
        100);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(!result.empty(), "slightly perspective target is detected");
}

// Obvious perspective is still detected (within a reasonable range).
void testObviousPerspective() {
    cv::Mat image = makeBackground();
    blitConcentricTarget(
        image,
        {cv::Point2f(210, 125), cv::Point2f(440, 160), cv::Point2f(425, 355),
         cv::Point2f(190, 335)},
        100);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(!result.empty(), "obviously perspective target is detected");
}

// A rotated target is still detected.
void testRotatedTarget() {
    constexpr double kPi = 3.14159265358979323846;
    const double theta = 30.0 * kPi / 180.0;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);
    const cv::Point2f center(320.0F, 240.0F);
    constexpr double kHalf = 85.0;

    const auto rotate = [&](double x, double y) {
        return cv::Point2f(
            static_cast<float>(center.x + x * cos_t - y * sin_t),
            static_cast<float>(center.y + x * sin_t + y * cos_t));
    };

    cv::Mat image = makeBackground();
    blitConcentricTarget(
        image,
        {rotate(-kHalf, -kHalf), rotate(kHalf, -kHalf), rotate(kHalf, kHalf),
         rotate(-kHalf, kHalf)},
        100);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(!result.empty(), "rotated target is detected");
}

// Light blur (motion-like) should still be detected.
void testMotionBlur() {
    cv::Mat image = makeConcentricTarget(100);
    cv::GaussianBlur(image, image, cv::Size(5, 5), 1.2);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(!result.empty(), "lightly blurred target is detected");
}

// A smaller target (still well within the candidate size range) is detected.
void testSmallerTarget() {
    cv::Mat image = makeConcentricTarget(60);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(!result.empty(), "smaller target is detected");
}

// The high-hue red range participates in the warp-stage red mask.
void testHighHueConcentricTarget() {
    cv::Mat image = makeWhiteBoard(100);
    drawConcentricRings(image, cv::Point(320, 240), 100, highHueRed());

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(!result.empty(), "high-hue concentric target is detected");
}

// ── Negative tests (office false positives) ─────────────────────────────────

// 1. A plain white board (no red) is rejected.
void testPureWhiteBoardRejected() {
    cv::Mat image = makeWhiteBoard(100);
    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "pure white board is rejected");
}

// 2. White board + a single solid red circle is rejected: it has a valid
//    bullseye but no red/white concentric-ring structure.
void testSingleRedCircleRejected() {
    cv::Mat image = makeWhiteBoard(100);
    cv::circle(image, cv::Point(320, 240), 30, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(),
          "white board + single red circle is rejected");
}

// 3. White board + a small red dot is rejected (not enough red evidence).
void testSmallRedDotRejected() {
    cv::Mat image = makeWhiteBoard(100);
    cv::circle(image, cv::Point(320, 240), 3, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "white board + small red dot rejected");
}

// 4. White board + scattered red regions (no concentric structure, center
//    white) is rejected.
void testScatteredRedRejected() {
    cv::Mat image = makeWhiteBoard(100);
    const cv::Scalar red(0, 0, 255);
    cv::circle(image, cv::Point(260, 180), 22, red, cv::FILLED);
    cv::circle(image, cv::Point(380, 180), 22, red, cv::FILLED);
    cv::circle(image, cv::Point(260, 300), 22, red, cv::FILLED);
    cv::circle(image, cv::Point(380, 300), 22, red, cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "scattered red regions rejected");
}

// 5. White board + a centered red rectangle is rejected (no ring alternation).
void testCenterRedRectangleRejected() {
    cv::Mat image = makeWhiteBoard(100);
    cv::rectangle(image, cv::Point(290, 210), cv::Point(350, 270),
                  cv::Scalar(0, 0, 255), cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(),
          "center red rectangle rejected");
}

// 6. White board + a red horizontal line is rejected (center is not mostly
//    red, and there is no alternation).
void testRedLineRejected() {
    cv::Mat image = makeWhiteBoard(100);
    cv::line(image, cv::Point(220, 240), cv::Point(420, 240),
             cv::Scalar(0, 0, 255), 3);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "red horizontal line rejected");
}

// 7. White board + two red regions (left and right of center) is rejected.
void testTwoRedRegionsRejected() {
    cv::Mat image = makeWhiteBoard(100);
    cv::circle(image, cv::Point(250, 240), 24, cv::Scalar(0, 0, 255),
               cv::FILLED);
    cv::circle(image, cv::Point(390, 240), 24, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "two red regions rejected");
}

// 8. Off-center concentric rings are rejected (the board center is not red).
void testOffCenterConcentricRejected() {
    cv::Mat image = makeWhiteBoard(100);
    drawConcentricRings(image, cv::Point(365, 285), 100);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "off-center concentric rings rejected");
}

// 9. Red outside the board is rejected (board mask isolates it).
void testRedOutsideBoardRejected() {
    cv::Mat image = makeWhiteBoard(100);
    cv::circle(image, cv::Point(520, 240), 45, cv::Scalar(0, 0, 255),
               cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "red outside board rejected");
}

// 10. A plain wide white rectangle is rejected (aspect ratio gate).
void testWideWhiteRectangleRejected() {
    cv::Mat image = makeBackground();
    cv::rectangle(image, cv::Point(100, 200), cv::Point(540, 280),
                  cv::Scalar(255, 255, 255), cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    check(detector.detect(image).empty(), "wide white rectangle rejected");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"standard concentric target", testStandardConcentricTarget},
        {"slight perspective", testSlightPerspective},
        {"obvious perspective", testObviousPerspective},
        {"rotated target", testRotatedTarget},
        {"motion blur", testMotionBlur},
        {"smaller target", testSmallerTarget},
        {"high-hue concentric target", testHighHueConcentricTarget},
        {"pure white board rejected", testPureWhiteBoardRejected},
        {"single red circle rejected", testSingleRedCircleRejected},
        {"small red dot rejected", testSmallRedDotRejected},
        {"scattered red rejected", testScatteredRedRejected},
        {"center red rectangle rejected", testCenterRedRectangleRejected},
        {"red line rejected", testRedLineRejected},
        {"two red regions rejected", testTwoRedRegionsRejected},
        {"off-center concentric rejected", testOffCenterConcentricRejected},
        {"red outside board rejected", testRedOutsideBoardRejected},
        {"wide white rectangle rejected", testWideWhiteRectangleRejected},
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
