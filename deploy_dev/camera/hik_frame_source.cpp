#include "camera/hik_frame_source.hpp"

#include <MvCameraControl.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace hnu25::camera {
namespace {

std::string boundedString(const unsigned char* value, std::size_t capacity) {
    const auto* begin = reinterpret_cast<const char*>(value);
    return std::string(begin, std::find(begin, begin + capacity, '\0'));
}

std::string serialOf(const MV_CC_DEVICE_INFO& device) {
    if (device.nTLayerType == MV_USB_DEVICE)
        return boundedString(device.SpecialInfo.stUsb3VInfo.chSerialNumber, INFO_MAX_BUFFER_SIZE);
    if (device.nTLayerType == MV_GIGE_DEVICE)
        return boundedString(device.SpecialInfo.stGigEInfo.chSerialNumber,
                             sizeof(device.SpecialInfo.stGigEInfo.chSerialNumber));
    return {};
}

cv::Mat toOwnedBgr(const MV_FRAME_OUT& raw) {
    const auto& info = raw.stFrameInfo;
    const cv::Size size(info.nWidth, info.nHeight);
    cv::Mat bgr;
    switch (info.enPixelType) {
        case PixelType_Gvsp_Mono8:
            cv::cvtColor(cv::Mat(size, CV_8UC1, raw.pBufAddr), bgr, cv::COLOR_GRAY2BGR);
            break;
        case PixelType_Gvsp_BayerGR8:
            cv::cvtColor(cv::Mat(size, CV_8UC1, raw.pBufAddr), bgr, cv::COLOR_BayerGR2BGR);
            break;
        case PixelType_Gvsp_BayerRG8:
            cv::cvtColor(cv::Mat(size, CV_8UC1, raw.pBufAddr), bgr, cv::COLOR_BayerRG2BGR);
            break;
        case PixelType_Gvsp_BayerGB8:
            cv::cvtColor(cv::Mat(size, CV_8UC1, raw.pBufAddr), bgr, cv::COLOR_BayerGB2BGR);
            break;
        case PixelType_Gvsp_BayerBG8:
            cv::cvtColor(cv::Mat(size, CV_8UC1, raw.pBufAddr), bgr, cv::COLOR_BayerBG2BGR);
            break;
        case PixelType_Gvsp_YUV422_YUYV_Packed:
            cv::cvtColor(cv::Mat(size, CV_8UC2, raw.pBufAddr), bgr, cv::COLOR_YUV2BGR_YUYV);
            break;
        case PixelType_Gvsp_RGB8_Packed:
            cv::cvtColor(cv::Mat(size, CV_8UC3, raw.pBufAddr), bgr, cv::COLOR_RGB2BGR);
            break;
        case PixelType_Gvsp_BGR8_Packed:
            bgr = cv::Mat(size, CV_8UC3, raw.pBufAddr).clone();
            break;
        default:
            break;
    }
    return bgr;
}

}  // namespace

HikFrameSource::HikFrameSource(HikConfig config) : config_(std::move(config)) {}

HikFrameSource::~HikFrameSource() {
    stop();
}

void HikFrameSource::start() {
    if (state_ != State::Empty) throw std::logic_error("HikFrameSource already started");
    MV_CC_DEVICE_INFO_LIST devices{};
    int result = MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, &devices);
    if (result != MV_OK) throw std::runtime_error("MV_CC_EnumDevices failed: " + std::to_string(result));
    if (devices.nDeviceNum == 0) throw std::runtime_error("no Hik camera found");

    MV_CC_DEVICE_INFO* selected = nullptr;
    if (!config_.serial_number.empty()) {
        for (unsigned int i = 0; i < devices.nDeviceNum; ++i) {
            if (devices.pDeviceInfo[i] && serialOf(*devices.pDeviceInfo[i]) == config_.serial_number) {
                selected = devices.pDeviceInfo[i];
                break;
            }
        }
        if (!selected) throw std::runtime_error("Hik camera serial not found: " + config_.serial_number);
    } else {
        selected = devices.pDeviceInfo[0];
        std::cerr << "[Camera] serial_number is empty; selecting first Hik device (serial="
                  << serialOf(*selected) << ")\n";
    }

    try {
        result = MV_CC_CreateHandle(&handle_, selected);
        if (result != MV_OK || !handle_) throw std::runtime_error("MV_CC_CreateHandle failed: " + std::to_string(result));
        state_ = State::HandleCreated;
        result = MV_CC_OpenDevice(handle_);
        if (result != MV_OK) throw std::runtime_error("MV_CC_OpenDevice failed: " + std::to_string(result));
        state_ = State::DeviceOpen;
        configure();
        result = MV_CC_StartGrabbing(handle_);
        if (result != MV_OK) throw std::runtime_error("MV_CC_StartGrabbing failed: " + std::to_string(result));
        state_ = State::Grabbing;
        stop_requested_.store(false, std::memory_order_relaxed);
        capture_thread_ = std::thread(&HikFrameSource::captureLoop, this);
    } catch (...) {
        cleanup();
        throw;
    }
}

void HikFrameSource::stop() noexcept {
    stop_requested_.store(true, std::memory_order_relaxed);
    frames_.stop();
    if (capture_thread_.joinable()) capture_thread_.join();
    cleanup();
}

bool HikFrameSource::waitForFrame(Frame& frame, std::chrono::milliseconds timeout) {
    return frames_.waitPop(frame, timeout);
}

void HikFrameSource::configure() {
    setEnum("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF);
    setEnum("GainAuto", MV_GAIN_MODE_OFF);
    setFloatAndReadBack("ExposureTime", config_.exposure);
    setFloatAndReadBack("Gain", config_.gain);
    if (config_.frame_rate > 0.0) {
        const int enabled = MV_CC_SetBoolValue(handle_, "AcquisitionFrameRateEnable", true);
        bool read_back = false;
        const int read_result = MV_CC_GetBoolValue(handle_, "AcquisitionFrameRateEnable", &read_back);
        if (enabled != MV_OK || read_result != MV_OK || !read_back)
            std::cerr << "[Camera] could not enable/read back AcquisitionFrameRateEnable\n";
        setFloatAndReadBack("AcquisitionFrameRate", config_.frame_rate);
    }
}

void HikFrameSource::captureLoop() {
    while (!stop_requested_.load(std::memory_order_relaxed)) {
        MV_FRAME_OUT raw{};
        const int result = MV_CC_GetImageBuffer(handle_, &raw, 100);
        if (result != MV_OK) continue;

        const auto received_at = std::chrono::steady_clock::now();
        cv::Mat bgr;
        try {
            bgr = toOwnedBgr(raw);
        } catch (const cv::Exception& error) {
            std::cerr << "[Camera] Hik conversion failed: " << error.what() << '\n';
        }
        const int free_result = MV_CC_FreeImageBuffer(handle_, &raw);
        if (free_result != MV_OK) {
            std::cerr << "[Camera] MV_CC_FreeImageBuffer failed: " << free_result << '\n';
            break;
        }
        if (bgr.empty()) {
            std::cerr << "[Camera] dropping unknown Hik pixel format " << raw.stFrameInfo.enPixelType << '\n';
            continue;
        }

        const double exposure_us = raw.stFrameInfo.fExposureTime > 0.0F
                                       ? raw.stFrameInfo.fExposureTime : config_.exposure;
        Frame frame;
        frame.image = std::move(bgr);
        frame.received_at = received_at;
        frame.captured_at = received_at - std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                              std::chrono::duration<double, std::micro>(exposure_us / 2.0));
        frame.frame_number = raw.stFrameInfo.nFrameNum;
        frame.timestamp_quality = TimestampQuality::ApproximateExposureCenter;
        // nHostTimeStamp is deliberately not used: this SDK header does not define its unit or clock epoch.
        frames_.push(std::move(frame));
    }
    frames_.stop();
}

void HikFrameSource::cleanup() noexcept {
    if (!handle_) {
        state_ = State::Empty;
        return;
    }
    if (state_ == State::Grabbing) {
        if (const int result = MV_CC_StopGrabbing(handle_); result != MV_OK)
            std::cerr << "[Camera] MV_CC_StopGrabbing failed: " << result << '\n';
        state_ = State::DeviceOpen;
    }
    if (state_ == State::DeviceOpen) {
        if (const int result = MV_CC_CloseDevice(handle_); result != MV_OK)
            std::cerr << "[Camera] MV_CC_CloseDevice failed: " << result << '\n';
        state_ = State::HandleCreated;
    }
    if (state_ == State::HandleCreated) {
        if (const int result = MV_CC_DestroyHandle(handle_); result != MV_OK)
            std::cerr << "[Camera] MV_CC_DestroyHandle failed: " << result << '\n';
    }
    handle_ = nullptr;
    state_ = State::Empty;
}

void HikFrameSource::setEnum(const char* name, unsigned int value) {
    const int set_result = MV_CC_SetEnumValue(handle_, name, value);
    MVCC_ENUMVALUE read_back{};
    const int get_result = MV_CC_GetEnumValue(handle_, name, &read_back);
    if (set_result != MV_OK || get_result != MV_OK || read_back.nCurValue != value)
        std::cerr << "[Camera] could not set/read back " << name << '\n';
}

void HikFrameSource::setFloatAndReadBack(const char* name, double value) {
    const int set_result = MV_CC_SetFloatValue(handle_, name, static_cast<float>(value));
    MVCC_FLOATVALUE read_back{};
    const int get_result = MV_CC_GetFloatValue(handle_, name, &read_back);
    if (set_result != MV_OK || get_result != MV_OK)
        std::cerr << "[Camera] could not set/read back " << name << '\n';
    else
        std::cerr << "[Camera] " << name << '=' << read_back.fCurValue << '\n';
}

}  // namespace hnu25::camera
