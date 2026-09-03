#include "anti_drone/camera_extrinsic_calibration_math.hpp"
#include "anti_drone/camera_extrinsic_calibration_validation.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;

int g_failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "    FAILED: " << message << '\n';
        ++g_failures;
    }
}

bool matNear(const cv::Matx33d& a, const cv::Matx33d& b, double tol) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            if (std::fabs(a(r, c) - b(r, c)) > tol) {
                return false;
            }
        }
    }
    return true;
}

bool vecNear(const cv::Vec3d& a, const cv::Vec3d& b, double tol) {
    return std::fabs(a[0] - b[0]) <= tol &&
           std::fabs(a[1] - b[1]) <= tol &&
           std::fabs(a[2] - b[2]) <= tol;
}

// ── Test 1: axisRotation produces the canonical right-handed matrices ──────
void testAxisRotation() {
    using hnu25::anti_drone::RotationAxis;
    using hnu25::anti_drone::axisRotation;

    const cv::Matx33d Rz = axisRotation(RotationAxis::Z, kPi / 2.0);
    // R_z(90) = [[0,-1,0],[1,0,0],[0,0,1]]
    check(matNear(Rz, cv::Matx33d(0.0, -1.0, 0.0,
                                  1.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0), 1e-12),
          "R_z(90) is the canonical +Z rotation");
    const cv::Vec3d ez(1.0, 0.0, 0.0);
    const cv::Vec3d Rz_ez = Rz * ez;
    check(vecNear(Rz_ez, cv::Vec3d(0.0, 1.0, 0.0), 1e-12),
          "R_z(90) maps +x to +y");

    const cv::Matx33d Ry = axisRotation(RotationAxis::Y, kPi / 2.0);
    // R_y(90) = [[0,0,1],[0,1,0],[-1,0,0]]
    check(matNear(Ry, cv::Matx33d(0.0, 0.0, 1.0,
                                  0.0, 1.0, 0.0,
                                  -1.0, 0.0, 0.0), 1e-12),
          "R_y(90) is the canonical +Y rotation");
    check(vecNear(Ry * ez, cv::Vec3d(0.0, 0.0, -1.0), 1e-12),
          "R_y(90) maps +x to -z");

    const cv::Matx33d Rx = axisRotation(RotationAxis::X, kPi / 2.0);
    check(matNear(Rx, cv::Matx33d(1.0, 0.0, 0.0,
                                  0.0, 0.0, -1.0,
                                  0.0, 1.0, 0.0), 1e-12),
          "R_x(90) is the canonical +X rotation");
    check(vecNear(Rx * cv::Vec3d(0.0, 1.0, 0.0), cv::Vec3d(0.0, 0.0, 1.0),
                  1e-12),
          "R_x(90) maps +y to +z");
}

// ── Test 2: yaw_axis/pitch_axis/order are honored exactly ──────────────────
void testAxisAndOrder() {
    using hnu25::anti_drone::GimbalKinematics;
    using hnu25::anti_drone::RotationAxis;
    using hnu25::anti_drone::axisRotation;
    using hnu25::anti_drone::makeGimbalPose;

    // Default example kinematics: yaw_axis z, pitch_axis y, order yaw_pitch,
    // sign +1, zero 0.
    GimbalKinematics kin;
    kin.yaw_axis = RotationAxis::Z;
    kin.pitch_axis = RotationAxis::Y;
    kin.rotation_order = GimbalKinematics::RotationOrder::YawPitch;
    kin.yaw_sign = 1.0;
    kin.pitch_sign = 1.0;
    kin.yaw_zero_deg = 0.0;
    kin.pitch_zero_deg = 0.0;

    // yaw=90, pitch=0 -> R = R_yaw * R_pitch = R_z(90) * I = R_z(90).
    const auto pose_yaw = makeGimbalPose(90.0, 0.0, kin);
    check(matNear(pose_yaw.R_gimbal2base,
                  axisRotation(RotationAxis::Z, kPi / 2.0), 1e-12),
          "yaw=90/pitch=0 gives R_z(90)");

    // yaw=0, pitch=90 -> R = I * R_y(90) = R_y(90).
    const auto pose_pitch = makeGimbalPose(0.0, 90.0, kin);
    check(matNear(pose_pitch.R_gimbal2base,
                  axisRotation(RotationAxis::Y, kPi / 2.0), 1e-12),
          "yaw=0/pitch=90 gives R_y(90)");

    // yaw=90, pitch=90 -> R = R_z(90) * R_y(90) (yaw outermost). The order is
    // NOT R_y * R_z.
    const auto pose_both = makeGimbalPose(90.0, 90.0, kin);
    const cv::Matx33d Rz_Ry = axisRotation(RotationAxis::Z, kPi / 2.0) *
                              axisRotation(RotationAxis::Y, kPi / 2.0);
    const cv::Matx33d Ry_Rz = axisRotation(RotationAxis::Y, kPi / 2.0) *
                              axisRotation(RotationAxis::Z, kPi / 2.0);
    check(matNear(pose_both.R_gimbal2base, Rz_Ry, 1e-12),
          "yaw=90/pitch=90 gives R_yaw * R_pitch = R_z * R_y");
    check(!matNear(Rz_Ry, Ry_Rz, 1e-6),
          "R_yaw * R_pitch differs from R_pitch * R_yaw (order is not swapped)");
}

// ── Test 3: synthetic hand-eye recovers the camera->gimbal direction ───────
//
// This is the transform-direction regression test. A KNOWN ^G T_C = X
// (rotation not identity, translation not zero) and a fixed board pose ^B T_T
// are used to generate noiseless observations
//
//   ^C T_T(i) = inverse(^B T_G(i) * ^G T_C) * ^B T_T
//
// which are fed to solveHandEye(). The recovered result must equal X. If the
// tool (or this math) accidentally output the INVERSE transform (gimbal ->
// camera instead of camera -> gimbal), the recovered rotation would differ
// from X by roughly twice X's rotation angle (tens of degrees), and the
// translation would be wrong by its full magnitude — so the tight tolerances
// below would FAIL.
void testSyntheticHandEye() {
    using hnu25::anti_drone::RotationAxis;
    using hnu25::anti_drone::axisRotation;
    using hnu25::anti_drone::rotationErrorRad;
    using hnu25::anti_drone::solveHandEye;
    using hnu25::anti_drone::evaluateFixedTargetConsistency;

    // Ground-truth camera -> gimbal transform (^G T_C), deliberately not
    // identity and not zero.
    const cv::Matx33d R_gt =
        axisRotation(RotationAxis::X, 0.30) *
        axisRotation(RotationAxis::Y, -0.20) *
        axisRotation(RotationAxis::Z, 0.10);
    const cv::Vec3d t_gt(0.04, -0.02, 0.03);

    // Fixed board pose in the base frame (^B T_T).
    const cv::Matx33d R_board = axisRotation(RotationAxis::Z, 0.40);
    const cv::Vec3d t_board(1.60, 0.10, -0.20);

    // A spread of gimbal poses (^B T_G(i)) with varied yaw AND pitch.
    const double yaw_pitch[][2] = {
        {-30.0, -15.0}, {-20.0, -10.0}, {-10.0, -5.0},  {0.0, -15.0},
        {10.0, 5.0},    {20.0, 10.0},   {30.0, 15.0},   {-25.0, 10.0},
        {-15.0, 5.0},   {5.0, -5.0},    {15.0, 10.0},   {25.0, -10.0},
    };
    const std::size_t n = sizeof(yaw_pitch) / sizeof(yaw_pitch[0]);

    std::vector<cv::Matx33d> R_gripper2base;
    std::vector<cv::Vec3d> t_gripper2base;
    std::vector<cv::Matx33d> R_target2cam;
    std::vector<cv::Vec3d> t_target2cam;

    for (std::size_t i = 0; i < n; ++i) {
        const double yaw = yaw_pitch[i][0] * kDegToRad;
        const double pitch = yaw_pitch[i][1] * kDegToRad;

        // ^B T_G(i): yaw about Z, pitch about Y, yaw outermost, zero offset.
        const cv::Matx33d R_g = axisRotation(RotationAxis::Z, yaw) *
                                axisRotation(RotationAxis::Y, pitch);
        const cv::Vec3d t_g(0.0, 0.0, 0.0);

        // A = ^B T_G * ^G T_C.
        const cv::Matx33d R_A = R_g * R_gt;
        const cv::Vec3d t_A = R_g * t_gt + t_g;

        // ^C T_T(i) = inverse(A) * ^B T_T.
        const cv::Matx33d R_ct = R_A.t() * R_board;
        const cv::Vec3d t_ct = R_A.t() * (t_board - t_A);

        R_gripper2base.push_back(R_g);
        t_gripper2base.push_back(t_g);
        R_target2cam.push_back(R_ct);
        t_target2cam.push_back(t_ct);
    }

    cv::Matx33d R_rec;
    cv::Vec3d t_rec;
    const bool solved = solveHandEye(R_gripper2base, t_gripper2base,
                                     R_target2cam, t_target2cam,
                                     cv::CALIB_HAND_EYE_PARK, R_rec, t_rec);
    check(solved, "hand-eye solve succeeds");

    const double rot_err_deg = rotationErrorRad(R_gt, R_rec) * kRadToDeg;
    check(rot_err_deg < 1e-4,
          "recovered rotation matches ground truth (camera -> gimbal, not "
          "inverted)");
    check(vecNear(t_rec, t_gt, 1e-6),
          "recovered translation matches ground truth");

    // The inverted transform must be far from the ground truth: this is what
    // a direction bug would produce, and it must be rejected.
    check(rotationErrorRad(R_gt, R_rec.t()) * kRadToDeg > 1.0,
          "inverse of the recovered rotation is far from ground truth");

    // Fixed-board consistency should be essentially zero on noiseless data.
    const auto stats = evaluateFixedTargetConsistency(
        R_gripper2base, t_gripper2base, R_rec, t_rec, R_target2cam,
        t_target2cam);
    check(stats.valid, "consistency evaluation is valid");
    check(stats.translation_rms_m < 1e-6,
          "consistency translation RMS ~ 0");
    check(stats.rotation_rms_deg < 1e-3,
          "consistency rotation RMS ~ 0");
}

// ── Test 4: gimbal yaw/pitch signs must be exactly +1 or -1 ────────────────
void testGimbalSignValidation() {
    using hnu25::anti_drone::requireGimbalSign;

    bool accepted = true;
    try {
        requireGimbalSign(1.0, "yaw_sign");
        requireGimbalSign(-1.0, "pitch_sign");
    } catch (const std::exception&) {
        accepted = false;
    }
    check(accepted, "+1.0 and -1.0 signs are accepted");

    const auto throwsWith = [](double v, const char* name,
                               const char* expected_msg) {
        try {
            requireGimbalSign(v, name);
        } catch (const std::runtime_error& e) {
            return std::string(e.what()) == expected_msg;
        } catch (...) {
            return false;
        }
        return false;
    };

    check(throwsWith(0.0, "yaw_sign", "yaw_sign must be +1 or -1"),
          "yaw_sign=0 rejected with exact message");
    check(throwsWith(0.5, "yaw_sign", "yaw_sign must be +1 or -1"),
          "yaw_sign=0.5 rejected with exact message");
    check(throwsWith(2.0, "yaw_sign", "yaw_sign must be +1 or -1"),
          "yaw_sign=2.0 rejected with exact message");
    check(throwsWith(-2.0, "yaw_sign", "yaw_sign must be +1 or -1"),
          "yaw_sign=-2.0 rejected with exact message");
    check(throwsWith(0.0, "pitch_sign", "pitch_sign must be +1 or -1"),
          "pitch_sign=0 rejected with exact message");
}

// ── Test 5: calibration resolution must be both-present or both-absent ──────
void testResolutionValidation() {
    using hnu25::anti_drone::validateCalibrationResolution;

    const auto accepts = [](bool hw, bool hh, int w, int h) {
        try {
            validateCalibrationResolution(hw, hh, w, h);
        } catch (...) {
            return false;
        }
        return true;
    };
    const auto rejects = [](bool hw, bool hh, int w, int h) {
        try {
            validateCalibrationResolution(hw, hh, w, h);
        } catch (...) {
            return true;
        }
        return false;
    };

    check(accepts(false, false, 0, 0), "both absent accepted");
    check(accepts(true, true, 1920, 1080), "both > 0 accepted");
    check(accepts(true, true, 0, 0), "both present as 0 accepted");

    check(rejects(true, false, 1920, 0), "only width rejected");
    check(rejects(false, true, 0, 1080), "only height rejected");
    check(rejects(true, true, 1920, 0), "width>0 height<=0 rejected");
    check(rejects(true, true, 0, 1080), "height>0 width<=0 rejected");
    check(rejects(true, true, -1, 1080), "negative width rejected");
    check(rejects(true, true, 1920, -1), "negative height rejected");
}

}  // namespace

int main() {
    struct TestCase {
        const char* name;
        void (*fn)();
    };
    const TestCase cases[] = {
        {"axis rotation matrices", testAxisRotation},
        {"yaw/pitch axis and order", testAxisAndOrder},
        {"synthetic hand-eye direction", testSyntheticHandEye},
        {"gimbal sign validation", testGimbalSignValidation},
        {"calibration resolution validation", testResolutionValidation},
    };

    for (const auto& c : cases) {
        const int before = g_failures;
        std::cout << "[ RUN      ] " << c.name << '\n';
        c.fn();
        std::cout << (g_failures == before ? "[       OK ] " : "[  FAILED  ] ")
                  << c.name << '\n';
    }

    if (g_failures == 0) {
        std::cout << "All anti_drone extrinsic calibration tests passed.\n";
        return 0;
    }
    std::cerr << g_failures << " check(s) failed.\n";
    return 1;
}
