# Anti-Drone Vision

面向 Anti-Drone 视觉检测 / 定位 / 跟踪诊断的独立 C++17 工程。

当前主要流程：

```text
Hik MVS Camera
  -> TraditionalDetector
  -> PnP
  -> 3D Tracker
  -> Predictor
  -> VisionSolution
  -> VisionTelemetry v1
  -> Loopback / Serial Diagnostic Transport
```

`VisionTelemetry` 是视觉结果 / 诊断数据协议，仅承载视觉结果与诊断信息，
不包含任何控制、执行或发射逻辑。

## Project Structure

```text
anti_drone/
  app_main.cpp
  traditional_detector.*
  pnp_solver.*
  tracker.*
  predictor.*
  vision_solution.*
  diagnostic_frame_processor.*
  vision_telemetry.*
  vision_telemetry_stream.*
  vision_telemetry_serial_transport.*
  serial_port.*
  VISION_TELEMETRY_PROTOCOL.md
  SERIAL_DIAGNOSTIC.md

camera/
config/
  anti_drone.yaml
tests/
```

## Dependencies

- C++17
- CMake >= 3.16
- OpenCV
- yaml-cpp
- Threads
- Hikrobot MVS SDK（实机 Hik camera 使用）

## Configure

```bash
cmake -S . -B build_anti_drone_vm \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=ON \
  -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
```

## Build

```bash
cmake --build build_anti_drone_vm -j$(nproc)
```

## Tests

```bash
ctest --test-dir build_anti_drone_vm \
  --output-on-failure
```

当前测试覆盖：

- camera abstraction
- detector geometry / color / NMS
- config
- PnP
- tracker / predictor
- diagnostic pipeline
- VisionTelemetry
- stream parser
- PTY serial integration

## Configuration Check

```bash
./build_anti_drone_vm/anti_drone/anti_drone_app \
  ./config/anti_drone.yaml \
  --check
```

`--check` 只做配置与链路自检，不打开真实相机或真实 serial device。

## Runtime

```bash
./build_anti_drone_vm/anti_drone/anti_drone_app \
  ./config/anti_drone.yaml
```

当前没有真实 calibration 时，程序会显示：

```text
Calibration: NOT CONFIGURED
```

此时 3D diagnostic solution 不可用是正常行为。请先完成真实内外参标定后运行。

## Vision Telemetry

`VisionTelemetry v1` 是固定 50-byte 小端数据包，使用 CRC-16/CCITT-FALSE 校验。

完整 wire format 见 [anti_drone/VISION_TELEMETRY_PROTOCOL.md](anti_drone/VISION_TELEMETRY_PROTOCOL.md)。

串口诊断说明见 [anti_drone/SERIAL_DIAGNOSTIC.md](anti_drone/SERIAL_DIAGNOSTIC.md)。

## Current Status

- Traditional target detection: implemented
- PnP pipeline: implemented
- 3D tracker: implemented
- predictor: implemented
- VisionTelemetry: implemented
- serial diagnostic transport: implemented
- Hik MVS build support: implemented

Pending real hardware work:

- camera runtime verification
- real intrinsic / extrinsic calibration
- real calibration quality verification
