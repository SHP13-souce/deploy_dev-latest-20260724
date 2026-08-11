#pragma once

#include "camera/frame_source.hpp"
#include "vest/detector.hpp"
#include "vest/target_selector.hpp"
#include "vest/tracker.hpp"
#include "vest/vision_target_core.hpp"
#include "vest_runtime/runtime_config.hpp"

#include <atomic>
#include <memory>

namespace hnu25::vest_runtime {

class ObservationSink {
public:
    virtual ~ObservationSink() = default;

    virtual void onObservation(
        const hnu25::vest::VisionTargetObservation& observation) = 0;
};

class VestRuntime {
public:
    explicit VestRuntime(RuntimeConfig config);

    void run(std::atomic_bool& stop_requested,
             ObservationSink& sink);

private:
    RuntimeConfig config_;

    std::unique_ptr<hnu25::camera::FrameSource> camera_;

    hnu25::vest::VestDetector detector_;
    hnu25::vest::VestTracker tracker_;
    hnu25::vest::TargetSelector selector_;
    hnu25::vest::VisionTargetCore core_;
};

}  // namespace hnu25::vest_runtime
