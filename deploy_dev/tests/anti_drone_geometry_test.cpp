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

double euclidean(const cv::Point2f& a, const cv::Point2f& b) {
    const double dx = static_cast<double>(a.x) - static_cast<double>(b.x);
    const double dy = static_cast<double>(a.y) - static_cast<double>(b.y);
    return std::sqrt(dx * dx + dy * dy);
}

// Test 1: a clean 200x200 white square centered at (320, 240).
void testStandardSquare() {
    cv::Mat image = makeBackground();
    check(image.cols == kImageWidth, "image width is 640");
    check(image.rows == kImageHeight, "image height is 480");

    cv::rectangle(image, cv::Point(220, 140), cv::Point(420, 340),
                  cv::Scalar(255, 255, 255), cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(!result.empty(), "standard square is detected");
    if (result.empty()) {
        return;
    }

    const auto& obs = result.front();
    check(obs.from_cv, "from_cv == true");
    check(!obs.from_yolo, "from_yolo == false");
    check(!obs.bullseye_valid, "bullseye_valid == false");
    check(obs.corners_valid, "corners_valid == true");

    check(std::abs(obs.center.x - 320.0F) <= 3.0F, "center.x near 320");
    check(std::abs(obs.center.y - 240.0F) <= 3.0F, "center.y near 240");

    check(std::isfinite(obs.cv_score), "cv_score is finite");
    check(obs.cv_score >= 0.0F && obs.cv_score <= 1.0F, "cv_score in [0, 1]");
    check(obs.cv_score > 0.90F, "cv_score > 0.90 for clean square");

    // Corner semantics: [0] TL, [1] TR, [2] BR, [3] BL.
    const cv::Point2f tl(220.0F, 140.0F);
    const cv::Point2f tr(420.0F, 140.0F);
    const cv::Point2f br(420.0F, 340.0F);
    const cv::Point2f bl(220.0F, 340.0F);

    check(std::abs(obs.corners[0].x - tl.x) <= 3.0F &&
              std::abs(obs.corners[0].y - tl.y) <= 3.0F,
          "corner[0] is TL");
    check(std::abs(obs.corners[1].x - tr.x) <= 3.0F &&
              std::abs(obs.corners[1].y - tr.y) <= 3.0F,
          "corner[1] is TR");
    check(std::abs(obs.corners[2].x - br.x) <= 3.0F &&
              std::abs(obs.corners[2].y - br.y) <= 3.0F,
          "corner[2] is BR");
    check(std::abs(obs.corners[3].x - bl.x) <= 3.0F &&
              std::abs(obs.corners[3].y - bl.y) <= 3.0F,
          "corner[3] is BL");

    check(obs.corners[0].x < obs.corners[1].x, "TL.x < TR.x");
    check(obs.corners[3].x < obs.corners[2].x, "BL.x < BR.x");
    check(obs.corners[0].y < obs.corners[3].y, "TL.y < BL.y");
    check(obs.corners[1].y < obs.corners[2].y, "TR.y < BR.y");
}

// Test 2: a mildly perspective-distorted white convex quad.
void testPerspectiveQuad() {
    const cv::Point2f tl(230.0F, 130.0F);
    const cv::Point2f tr(430.0F, 160.0F);
    const cv::Point2f br(400.0F, 350.0F);
    const cv::Point2f bl(210.0F, 320.0F);

    cv::Mat image = makeBackground();
    const std::vector<cv::Point> quad = {
        cv::Point(230, 130),  // TL
        cv::Point(430, 160),  // TR
        cv::Point(400, 350),  // BR
        cv::Point(210, 320),  // BL
    };
    cv::fillConvexPoly(image, quad, cv::Scalar(255, 255, 255));

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(!result.empty(), "perspective quad is detected");
    if (result.empty()) {
        return;
    }

    const auto& obs = result.front();
    check(obs.corners_valid, "perspective quad corners_valid == true");

    check(euclidean(obs.corners[0], tl) <= 6.0, "corner[0] TL within 6px");
    check(euclidean(obs.corners[1], tr) <= 6.0, "corner[1] TR within 6px");
    check(euclidean(obs.corners[2], br) <= 6.0, "corner[2] BR within 6px");
    check(euclidean(obs.corners[3], bl) <= 6.0, "corner[3] BL within 6px");

    const cv::Point2f expected_center(
        (tl.x + tr.x + br.x + bl.x) / 4.0F,
        (tl.y + tr.y + br.y + bl.y) / 4.0F);
    check(euclidean(obs.center, expected_center) <= 5.0,
          "center near quad mean within 5px");
}

// Test 7: a white square rotated 35 degrees around (320, 240).
void testRotatedSquare() {
    constexpr double kPi = 3.14159265358979323846;
    const double theta = 35.0 * kPi / 180.0;
    const double cos_t = std::cos(theta);
    const double sin_t = std::sin(theta);

    const cv::Point2f center(320.0F, 240.0F);
    constexpr double kHalf = 90.0;

    // Local square corners before rotation.
    const cv::Point2f local_tl(static_cast<float>(-kHalf),
                               static_cast<float>(-kHalf));
    const cv::Point2f local_tr(static_cast<float>(kHalf),
                               static_cast<float>(-kHalf));
    const cv::Point2f local_br(static_cast<float>(kHalf),
                               static_cast<float>(kHalf));
    const cv::Point2f local_bl(static_cast<float>(-kHalf),
                               static_cast<float>(kHalf));

    // 2D rotation in image coordinates (x right, y down).
    const auto rotate = [&](const cv::Point2f& p) {
        return cv::Point2f(
            static_cast<float>(center.x + p.x * cos_t - p.y * sin_t),
            static_cast<float>(center.y + p.x * sin_t + p.y * cos_t));
    };

    const cv::Point2f expected_tl = rotate(local_tl);
    const cv::Point2f expected_tr = rotate(local_tr);
    const cv::Point2f expected_br = rotate(local_br);
    const cv::Point2f expected_bl = rotate(local_bl);

    cv::Mat image = makeBackground();
    const std::vector<cv::Point> quad = {
        cv::Point(static_cast<int>(std::round(expected_tl.x)),
                  static_cast<int>(std::round(expected_tl.y))),
        cv::Point(static_cast<int>(std::round(expected_tr.x)),
                  static_cast<int>(std::round(expected_tr.y))),
        cv::Point(static_cast<int>(std::round(expected_br.x)),
                  static_cast<int>(std::round(expected_br.y))),
        cv::Point(static_cast<int>(std::round(expected_bl.x)),
                  static_cast<int>(std::round(expected_bl.y))),
    };
    cv::fillConvexPoly(image, quad, cv::Scalar(255, 255, 255));

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);

    check(!result.empty(), "rotated square is detected");
    if (result.empty()) {
        return;
    }

    const auto& obs = result.front();
    check(obs.from_cv, "from_cv == true");
    check(!obs.from_yolo, "from_yolo == false");
    check(!obs.bullseye_valid, "bullseye_valid == false");
    check(obs.corners_valid, "corners_valid == true");
    check(std::isfinite(obs.cv_score), "cv_score is finite");
    check(obs.cv_score >= 0.0F && obs.cv_score <= 1.0F, "cv_score in [0, 1]");
    check(obs.cv_score > 0.85F, "cv_score > 0.85 for rotated square");

    check(euclidean(obs.center, cv::Point2f(320.0F, 240.0F)) <= 4.0,
          "center near (320, 240) within 4px");

    check(euclidean(obs.corners[0], expected_tl) <= 6.0, "corner[0] is TL");
    check(euclidean(obs.corners[1], expected_tr) <= 6.0, "corner[1] is TR");
    check(euclidean(obs.corners[2], expected_br) <= 6.0, "corner[2] is BR");
    check(euclidean(obs.corners[3], expected_bl) <= 6.0, "corner[3] is BL");
}

// Test 3: a wide white rectangle whose aspect ratio exceeds max_aspect_ratio.
void testWideRectangleFiltered() {
    cv::Mat image = makeBackground();
    cv::rectangle(image, cv::Point(100, 200), cv::Point(540, 280),
                  cv::Scalar(255, 255, 255), cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(result.empty(), "wide rectangle (aspect > 2.20) is filtered");
}

// Test 4: a 3x3 white speck, filtered by min_candidate_area_ratio.
void testTinyNoiseFiltered() {
    cv::Mat image = makeBackground();
    cv::rectangle(image, cv::Point(320, 240), cv::Point(322, 242),
                  cv::Scalar(255, 255, 255), cv::FILLED);

    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(image);
    check(result.empty(), "3x3 noise is filtered by min area ratio");
}

// Test 5: empty Mat returns empty without crashing.
void testEmptyMat() {
    cv::Mat empty;
    hnu25::anti_drone::TraditionalTargetDetector detector;
    const auto result = detector.detect(empty);
    check(result.empty(), "empty Mat returns empty");
}

// Test 6: non-BGR image types return empty without crashing.
void testWrongImageType() {
    hnu25::anti_drone::TraditionalTargetDetector detector;

    const cv::Mat gray(
        kImageHeight,
        kImageWidth,
        CV_8UC1,
        cv::Scalar(128));
    check(detector.detect(gray).empty(), "CV_8UC1 returns empty");

    const cv::Mat rgba(
        kImageHeight,
        kImageWidth,
        CV_8UC4,
        cv::Scalar(0, 0, 0, 255));
    check(detector.detect(rgba).empty(), "CV_8UC4 returns empty");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"standard square", testStandardSquare},
        {"perspective quad", testPerspectiveQuad},
        {"rotated square", testRotatedSquare},
        {"wide rectangle filtered", testWideRectangleFiltered},
        {"tiny noise filtered", testTinyNoiseFiltered},
        {"empty mat", testEmptyMat},
        {"wrong image type", testWrongImageType},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone geometry tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
