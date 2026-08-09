#include "detection/detector.hpp"

#include <opencv2/videoio.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 4 || argc > 8) {
        std::cerr << "usage: detector_benchmark <hebei|awakening> <model.onnx> <video> "
                     "[max_frames] [conf_threshold] [latency|throughput] [classifier.onnx]\n";
        return 2;
    }
    try {
        hnu25::DetectorConfig config;
        config.backend = argv[1];
        config.model_path = argv[2];
        config.conf_threshold = argc >= 6 ? std::stof(argv[5])
                                          : (config.backend == "awakening" ? 0.2F : 0.25F);
        config.performance_mode = argc >= 7 ? argv[6] : "latency";
        config.classifier_path = argc >= 8 ? argv[7] : "";
        auto detector = hnu25::makeDetector(config);
        cv::VideoCapture capture(argv[3]);
        if (!capture.isOpened()) throw std::runtime_error("cannot open input video");
        const int max_frames = argc >= 5 ? std::stoi(argv[4]) : -1;
        int frames = 0;
        int detection_frames = 0;
        int detections = 0;
        double elapsed_ms = 0.0;
        cv::Mat image;
        while ((max_frames < 0 || frames < max_frames) && capture.read(image)) {
            const auto begin = std::chrono::steady_clock::now();
            const auto result = detector->detect(image);
            elapsed_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - begin).count();
            detections += static_cast<int>(result.size());
            detection_frames += !result.empty();
            ++frames;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "backend=" << config.backend << '\n'
                  << "frames=" << frames << '\n'
                  << "detections=" << detections << '\n'
                  << "detection_frames=" << detection_frames << '\n'
                  << "mean_detection_ms=" << (frames > 0 ? elapsed_ms / frames : 0.0) << '\n'
                  << "detection_fps=" << (elapsed_ms > 0.0 ? frames * 1000.0 / elapsed_ms : 0.0)
                  << '\n';
        return frames > 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "detector_benchmark: " << error.what() << '\n';
        return 1;
    }
}
