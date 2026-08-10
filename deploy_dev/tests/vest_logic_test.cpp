#include "vest/tracker.hpp"
#include "vest/target_selector.hpp"
#include "vest/vision_target_core.hpp"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

bool nearlyEqual(float a, float b, float epsilon = 1e-4F) {
    return std::fabs(a - b) <= epsilon;
}

hnu25::vest::DetectedVest makeDetection(float center_x, float center_y,
                                      float width, float height,
                                      float confidence,
                                      std::chrono::steady_clock::time_point timestamp) {
    hnu25::vest::DetectedVest detection;
    detection.center = cv::Point2f(center_x, center_y);
    detection.box = cv::Rect2f(center_x - width * 0.5F,
                               center_y - height * 0.5F,
                               width, height);
    detection.confidence = confidence;
    detection.class_id = 0;
    detection.timestamp = timestamp;
    return detection;
}

hnu25::vest::TrackedVest makeTrack(int track_id,
                                hnu25::vest::VestTrackState state,
                                float center_x, float center_y,
                                float box_w, float box_h,
                                float vel_x, float vel_y,
                                float pred_x, float pred_y,
                                float confidence,
                                std::chrono::steady_clock::time_point timestamp) {
    hnu25::vest::TrackedVest track;
    track.track_id = track_id;
    track.state = state;
    track.center = cv::Point2f(center_x, center_y);
    track.box = cv::Rect2f(center_x - box_w * 0.5F,
                           center_y - box_h * 0.5F,
                           box_w, box_h);
    track.velocity = cv::Point2f(vel_x, vel_y);
    track.predicted_center = cv::Point2f(pred_x, pred_y);
    track.confidence = confidence;
    track.timestamp = timestamp;
    return track;
}

// ─── Test 1: confirm_hits=2, Track ID, velocity, predicted_center ───────────

void testTrackerConfirmationAndMotion() {
    const auto t0 = std::chrono::steady_clock::time_point{std::chrono::seconds{10}};

    hnu25::vest::VestTrackerConfig config;
    config.confirm_hits = 2;
    config.max_lost_frames = 5;
    config.min_iou = 0.10F;
    config.max_center_distance_ratio = 1.5F;

    hnu25::vest::VestTracker tracker(config);

    // Frame 1: single detection
    const auto det0 = makeDetection(100.0F, 100.0F, 40.0F, 60.0F, 0.9F, t0);
    auto outputs = tracker.update({det0});

    assert(outputs.size() == 1);
    assert(outputs[0].track_id == 0);
    assert(outputs[0].state == hnu25::vest::VestTrackState::Tentative);

    const int first_track_id = outputs[0].track_id;

    // Frame 2: same target moved slightly
    const auto t1 = t0 + std::chrono::milliseconds{100};
    const auto det1 = makeDetection(110.0F, 100.0F, 40.0F, 60.0F, 0.9F, t1);
    outputs = tracker.update({det1});

    assert(outputs.size() == 1);
    assert(outputs[0].track_id == first_track_id);
    assert(outputs[0].state == hnu25::vest::VestTrackState::Tracking);

    // Velocity: (110-100)/0.1 = 100 px/s
    const float expected_vel_x = 100.0F;
    assert(nearlyEqual(outputs[0].velocity.x, expected_vel_x));
    assert(nearlyEqual(outputs[0].velocity.y, 0.0F));

    // Predicted center: new_center + velocity * dt = 110 + 100*0.1 = 120
    const float expected_pred_x = 120.0F;
    assert(nearlyEqual(outputs[0].predicted_center.x, expected_pred_x));
    assert(nearlyEqual(outputs[0].predicted_center.y, 100.0F));
}

// ─── Test 2: TemporarilyLost lifecycle ──────────────────────────────────────

void testTrackerLostLifecycle() {
    const auto t0 = std::chrono::steady_clock::time_point{std::chrono::seconds{10}};

    hnu25::vest::VestTrackerConfig config;
    config.confirm_hits = 1;
    config.max_lost_frames = 2;
    config.min_iou = 0.10F;
    config.max_center_distance_ratio = 1.5F;

    hnu25::vest::VestTracker tracker(config);

    // Frame 1: single detection → confirm_hits=1 → Tracking immediately
    const auto det0 = makeDetection(100.0F, 100.0F, 40.0F, 60.0F, 0.9F, t0);
    auto outputs = tracker.update({det0});

    assert(outputs.size() == 1);
    assert(outputs[0].state == hnu25::vest::VestTrackState::Tracking);
    const int track_id = outputs[0].track_id;

    // Miss 1
    const auto t1 = t0 + std::chrono::milliseconds{100};
    (void)t1;  // Tracker doesn't use timestamp directly for unmatched
    outputs = tracker.update({});

    assert(outputs.size() == 1);
    assert(outputs[0].track_id == track_id);
    assert(outputs[0].state == hnu25::vest::VestTrackState::TemporarilyLost);

    // Miss 2
    outputs = tracker.update({});

    assert(outputs.size() == 1);
    assert(outputs[0].track_id == track_id);
    assert(outputs[0].state == hnu25::vest::VestTrackState::TemporarilyLost);

    // Miss 3 → Lost → erased
    outputs = tracker.update({});

    assert(outputs.empty());
}

// ─── Test 3: TargetSelector ─────────────────────────────────────────────────

void testTargetSelector() {
    const auto t0 = std::chrono::steady_clock::time_point{std::chrono::seconds{10}};

    // ── 3a: First selection – closest to image center ──────────────────

    hnu25::vest::TargetSelector selector;

    std::vector<hnu25::vest::TrackedVest> tracks = {
        makeTrack(2, hnu25::vest::VestTrackState::Tracking,
                  100.0F, 240.0F, 40.0F, 60.0F,  0.0F, 0.0F, 100.0F, 240.0F, 0.9F, t0),
        makeTrack(5, hnu25::vest::VestTrackState::Tracking,
                  300.0F, 240.0F, 40.0F, 60.0F,  0.0F, 0.0F, 300.0F, 240.0F, 0.9F, t0),
        makeTrack(8, hnu25::vest::VestTrackState::Tracking,
                  500.0F, 240.0F, 40.0F, 60.0F,  0.0F, 0.0F, 500.0F, 240.0F, 0.9F, t0),
    };

    auto selected = selector.select(tracks, cv::Size(640, 480));
    // Center at (320, 240); Track 5 is at (300, 240) — closest
    assert(selected.has_target);
    assert(selected.measurement_valid);
    assert(selected.track.track_id == 5);

    // ── 3b: Selected ID persistence (even when another is closer) ──────

    std::vector<hnu25::vest::TrackedVest> tracks2 = {
        makeTrack(5, hnu25::vest::VestTrackState::Tracking,
                  100.0F, 100.0F, 40.0F, 60.0F,  0.0F, 0.0F, 100.0F, 100.0F, 0.9F, t0),
        makeTrack(8, hnu25::vest::VestTrackState::Tracking,
                  320.0F, 240.0F, 40.0F, 60.0F,  0.0F, 0.0F, 320.0F, 240.0F, 0.9F, t0),
    };

    selected = selector.select(tracks2, cv::Size(640, 480));
    // Track 8 is now at exact center, but Track 5 should persist
    assert(selected.has_target);
    assert(selected.measurement_valid);
    assert(selected.track.track_id == 5);

    // ── 3c: Selected TemporarilyLost – keep ID, measurement_valid=false ─

    std::vector<hnu25::vest::TrackedVest> tracks3 = {
        makeTrack(5, hnu25::vest::VestTrackState::TemporarilyLost,
                  100.0F, 100.0F, 40.0F, 60.0F,  0.0F, 0.0F, 100.0F, 100.0F, 0.9F, t0),
        makeTrack(8, hnu25::vest::VestTrackState::Tracking,
                  320.0F, 240.0F, 40.0F, 60.0F,  0.0F, 0.0F, 320.0F, 240.0F, 0.9F, t0),
    };

    selected = selector.select(tracks3, cv::Size(640, 480));

    assert(selected.has_target);
    assert(!selected.measurement_valid);
    assert(selected.track.track_id == 5);
    assert(selected.track.state == hnu25::vest::VestTrackState::TemporarilyLost);

    // ── 3d: Selected disappears → same-frame re-select ────────────────

    std::vector<hnu25::vest::TrackedVest> tracks4 = {
        makeTrack(8, hnu25::vest::VestTrackState::Tracking,
                  320.0F, 240.0F, 40.0F, 60.0F,  0.0F, 0.0F, 320.0F, 240.0F, 0.9F, t0),
        makeTrack(9, hnu25::vest::VestTrackState::Tracking,
                  500.0F, 240.0F, 40.0F, 60.0F,  0.0F, 0.0F, 500.0F, 240.0F, 0.9F, t0),
    };

    selected = selector.select(tracks4, cv::Size(640, 480));
    // Track 5 is gone; Track 8 at (320, 240) = exact center should be selected
    assert(selected.has_target);
    assert(selected.measurement_valid);
    assert(selected.track.track_id == 8);
}

// ─── Test 4: VisionTargetCore normalization and state mapping ───────────────

void testVisionTargetCore() {
    const auto t0 = std::chrono::steady_clock::time_point{std::chrono::seconds{10}};

    hnu25::vest::VisionTargetCore core;

    // ── 4a: Normal Tracking normalization ──────────────────────────────

    hnu25::vest::SelectedTarget selected;
    selected.has_target = true;
    selected.measurement_valid = true;

    selected.track.track_id = 7;
    selected.track.state = hnu25::vest::VestTrackState::Tracking;
    selected.track.center = cv::Point2f(500.0F, 250.0F);
    selected.track.box = cv::Rect2f(400.0F, 200.0F, 200.0F, 100.0F);
    selected.track.velocity = cv::Point2f(100.0F, 50.0F);
    selected.track.predicted_center = cv::Point2f(600.0F, 300.0F);
    selected.track.confidence = 0.8F;
    selected.track.timestamp = t0;

    auto obs = core.build(selected, cv::Size(1000, 500), 42);

    assert(obs.frame_id == 42);
    assert(obs.has_target);
    assert(obs.target_valid);
    assert(obs.track_id == 7);
    assert(obs.tracking_state == hnu25::vest::VisionTrackingState::Tracking);

    // Normalized values
    assert(nearlyEqual(obs.center_x, 0.5F));       // 500/1000
    assert(nearlyEqual(obs.center_y, 0.5F));       // 250/500
    assert(nearlyEqual(obs.error_x, 0.0F));        // 0.5-0.5
    assert(nearlyEqual(obs.error_y, 0.0F));
    assert(nearlyEqual(obs.velocity_x, 0.1F));     // 100/1000
    assert(nearlyEqual(obs.velocity_y, 0.1F));     // 50/500
    assert(nearlyEqual(obs.predicted_x, 0.6F));    // 600/1000
    assert(nearlyEqual(obs.predicted_y, 0.6F));    // 300/500
    assert(nearlyEqual(obs.bbox_w, 0.2F));         // 200/1000
    assert(nearlyEqual(obs.bbox_h, 0.2F));         // 100/500
    assert(nearlyEqual(obs.confidence, 0.8F));

    // Timestamp must be positive
    assert(obs.measurement_timestamp_us > 0);

    // ── 4b: NoTarget ───────────────────────────────────────────────────

    hnu25::vest::SelectedTarget no_target;
    // default: has_target=false

    auto no_obs = core.build(no_target, cv::Size(1000, 500), 99);

    assert(no_obs.frame_id == 99);
    assert(!no_obs.has_target);
    assert(!no_obs.target_valid);
    assert(no_obs.track_id == -1);
    assert(no_obs.tracking_state == hnu25::vest::VisionTrackingState::NoTarget);
    assert(no_obs.measurement_timestamp_us == 0);

    // All spatial/bbox/confidence fields should be zero
    assert(no_obs.center_x == 0.0F);
    assert(no_obs.center_y == 0.0F);
    assert(no_obs.error_x == 0.0F);
    assert(no_obs.error_y == 0.0F);
    assert(no_obs.velocity_x == 0.0F);
    assert(no_obs.velocity_y == 0.0F);
    assert(no_obs.predicted_x == 0.0F);
    assert(no_obs.predicted_y == 0.0F);
    assert(no_obs.bbox_w == 0.0F);
    assert(no_obs.bbox_h == 0.0F);
    assert(no_obs.confidence == 0.0F);

    // ── 4c: TemporarilyLost – stale spatial data must be zeroed ────────

    hnu25::vest::SelectedTarget lost_selected;
    lost_selected.has_target = true;
    lost_selected.measurement_valid = false;

    lost_selected.track.track_id = 7;
    lost_selected.track.state = hnu25::vest::VestTrackState::TemporarilyLost;
    // Deliberately fill with non-zero stale data
    lost_selected.track.center = cv::Point2f(500.0F, 250.0F);
    lost_selected.track.box = cv::Rect2f(400.0F, 200.0F, 200.0F, 100.0F);
    lost_selected.track.velocity = cv::Point2f(100.0F, 50.0F);
    lost_selected.track.predicted_center = cv::Point2f(600.0F, 300.0F);
    lost_selected.track.confidence = 0.8F;
    lost_selected.track.timestamp = t0;

    auto lost_obs = core.build(lost_selected, cv::Size(1000, 500), 55);

    assert(lost_obs.has_target);
    assert(!lost_obs.target_valid);
    assert(lost_obs.track_id == 7);
    assert(lost_obs.tracking_state == hnu25::vest::VisionTrackingState::TemporarilyLost);

    // All spatial fields must be zero (no stale-data leakage)
    assert(lost_obs.center_x == 0.0F);
    assert(lost_obs.center_y == 0.0F);
    assert(lost_obs.error_x == 0.0F);
    assert(lost_obs.error_y == 0.0F);
    assert(lost_obs.velocity_x == 0.0F);
    assert(lost_obs.velocity_y == 0.0F);
    assert(lost_obs.predicted_x == 0.0F);
    assert(lost_obs.predicted_y == 0.0F);
    assert(lost_obs.bbox_w == 0.0F);
    assert(lost_obs.bbox_h == 0.0F);
    assert(lost_obs.confidence == 0.0F);

    // Timestamp should still carry the last measurement time
    assert(lost_obs.measurement_timestamp_us > 0);
}

// ─── Test 5: Invalid inputs (image_size, NaN, negative bbox) ────────────────

void testInvalidInputs() {
    const auto t0 = std::chrono::steady_clock::time_point{std::chrono::seconds{10}};

    // ── 5a: VisionTargetCore – invalid image_size ──────────────────────

    hnu25::vest::VisionTargetCore core;
    hnu25::vest::SelectedTarget valid_selected;
    valid_selected.has_target = true;
    valid_selected.measurement_valid = true;
    valid_selected.track.track_id = 1;
    valid_selected.track.state = hnu25::vest::VestTrackState::Tracking;
    valid_selected.track.center = cv::Point2f(100.0F, 100.0F);
    valid_selected.track.box = cv::Rect2f(80.0F, 80.0F, 40.0F, 40.0F);
    valid_selected.track.velocity = cv::Point2f(0.0F, 0.0F);
    valid_selected.track.predicted_center = cv::Point2f(100.0F, 100.0F);
    valid_selected.track.confidence = 0.5F;
    valid_selected.track.timestamp = t0;

    bool threw = false;
    try {
        core.build(valid_selected, cv::Size(0, 500), 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        core.build(valid_selected, cv::Size(1000, -1), 1);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // ── 5b: TargetSelector – invalid image_size ────────────────────────

    hnu25::vest::TargetSelector invalid_selector;

    threw = false;
    try {
        invalid_selector.select({}, cv::Size(0, 480));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // ── 5c: VisionTargetCore – NaN center protection ───────────────────

    hnu25::vest::SelectedTarget nan_selected;
    nan_selected.has_target = true;
    nan_selected.measurement_valid = true;
    nan_selected.track.track_id = 3;
    nan_selected.track.state = hnu25::vest::VestTrackState::Tracking;
    nan_selected.track.center.x = std::numeric_limits<float>::quiet_NaN();
    nan_selected.track.center.y = 100.0F;
    nan_selected.track.box = cv::Rect2f(80.0F, 80.0F, 40.0F, 40.0F);
    nan_selected.track.velocity = cv::Point2f(0.0F, 0.0F);
    nan_selected.track.predicted_center = cv::Point2f(100.0F, 100.0F);
    nan_selected.track.confidence = 0.5F;
    nan_selected.track.timestamp = t0;

    auto nan_obs = core.build(nan_selected, cv::Size(1000, 500), 77);

    assert(nan_obs.has_target);
    assert(!nan_obs.target_valid);
    assert(nan_obs.track_id == 3);
    assert(nan_obs.tracking_state == hnu25::vest::VisionTrackingState::Tracking);

    // All measurement fields must be zero
    assert(nan_obs.center_x == 0.0F);
    assert(nan_obs.center_y == 0.0F);
    assert(nan_obs.error_x == 0.0F);
    assert(nan_obs.error_y == 0.0F);
    assert(nan_obs.velocity_x == 0.0F);
    assert(nan_obs.velocity_y == 0.0F);
    assert(nan_obs.predicted_x == 0.0F);
    assert(nan_obs.predicted_y == 0.0F);
    assert(nan_obs.bbox_w == 0.0F);
    assert(nan_obs.bbox_h == 0.0F);
    assert(nan_obs.confidence == 0.0F);

    // ── 5d: VisionTargetCore – negative bbox protection ────────────────

    hnu25::vest::SelectedTarget neg_selected;
    neg_selected.has_target = true;
    neg_selected.measurement_valid = true;
    neg_selected.track.track_id = 5;
    neg_selected.track.state = hnu25::vest::VestTrackState::Tracking;
    neg_selected.track.center = cv::Point2f(100.0F, 100.0F);
    neg_selected.track.box = cv::Rect2f(100.0F, 100.0F, -1.0F, 40.0F);
    neg_selected.track.velocity = cv::Point2f(0.0F, 0.0F);
    neg_selected.track.predicted_center = cv::Point2f(100.0F, 100.0F);
    neg_selected.track.confidence = 0.5F;
    neg_selected.track.timestamp = t0;

    auto neg_obs = core.build(neg_selected, cv::Size(1000, 500), 88);

    assert(neg_obs.has_target);
    assert(!neg_obs.target_valid);
    assert(neg_obs.track_id == 5);
    assert(neg_obs.tracking_state == hnu25::vest::VisionTrackingState::Tracking);

    // All measurement fields must be zero
    assert(neg_obs.center_x == 0.0F);
    assert(neg_obs.center_y == 0.0F);
    assert(neg_obs.error_x == 0.0F);
    assert(neg_obs.error_y == 0.0F);
    assert(neg_obs.velocity_x == 0.0F);
    assert(neg_obs.velocity_y == 0.0F);
    assert(neg_obs.predicted_x == 0.0F);
    assert(neg_obs.predicted_y == 0.0F);
    assert(neg_obs.bbox_w == 0.0F);
    assert(neg_obs.bbox_h == 0.0F);
    assert(neg_obs.confidence == 0.0F);
}

}  // namespace

int main() {
    testTrackerConfirmationAndMotion();
    testTrackerLostLifecycle();
    testTargetSelector();
    testVisionTargetCore();
    testInvalidInputs();

    std::cout << "vest_logic_test: PASS" << std::endl;
    return 0;
}
