/**
 * @file   Fusion Changes (融合变更记录)
 * @brief  HNU_NHS_Vision-25 三组代码融合为可部署版本的详细记录
 *
 * 本项目将 2025 级三组新人独立完成的代码融合为统一可部署的自瞄系统:
 *   - 识别组 (余东阳): YOLO26 装甲板检测器
 *   - 解算组: PnP 姿态解算 + EKF 卡尔曼预测器
 *   - 通信组: BCP 串口通信协议
 *
 * 融合原则: 尽量少改各组代码，仅做必要的接口适配和去冗余
 */

# 融合变更记录

## 一、目录结构

```
deploy/
├── CMakeLists.txt              # 顶层构建文件 [新建]
├── shared/
│   └── types.hpp               # 统一类型定义 [新建]
├── detection/
│   ├── CMakeLists.txt          # [新建]
│   ├── yolo26_detector.hpp     # [来自识别组, 重写接口适配]
│   └── yolo26_detector.cpp     # [来自识别组, 重写接口适配]
├── solve/
│   ├── CMakeLists.txt          # [新建]
│   ├── pnp_solver.hpp/cpp      # [来自解算组 solver.cpp, 重写]
│   ├── motion_model.hpp/cpp    # [来自解算组, 未改]
│   ├── kalman_filter.hpp/cpp   # [来自解算组, 未改]
│   └── predictor.hpp/cpp       # [来自解算组, 微调 include]
├── communication/
│   ├── CMakeLists.txt          # [新建]
│   ├── bcp_packet.hpp          # [来自通信组 packet_typedef.hpp, 精简]
│   ├── serial_port.hpp         # [来自同项目 serial 库, 精简]
│   └── serial_port.cpp         # [来自同项目 serial 库, 精简]
├── pipeline/
│   ├── CMakeLists.txt          # [新建]
│   └── main.cpp                # [新建] 主流水线
├── config/
│   ├── car.yaml                # [新建] 实车配置
│   └── test.yaml               # [新建] 测试配置
├── tests/
│   ├── CMakeLists.txt          # [新建]
│   └── test_video.cpp          # [新建] 视频测试
├── assets/models/
│   └── yolo26.onnx             # [来自识别组，复制]
└── docs/
    ├── fusion_changes.md       # 本文件
    └── deployment.md           # 部署指南 [新建]
```

## 二、各组修改详述

### 2.1 识别组: Yolo26Detector (余东阳)

**原始文件:**
- `fused_project/tasks/auto_aim/yolo26_detector.hpp` (50 行)
- `fused_project/tasks/auto_aim/yolo26_detector.cpp` (314 行)

**变更:**
1. **类型系统重构**: 原始代码输出 `auto_aim::Armor` (依赖 Tongji armor.hpp 80+ 行枚举/结构体/Lightbar)，改为输出 `hnu25::DetectedArmor` (仅 7 个字段)
2. **命名空间**: `auto_aim` → `hnu25`
3. **接口统一**: `preprocessToNCHW` 重命名为 `preprocessNCHW`，`formatOutputToRows` 重命名为 `formatOutput`
4. **去冗余**: 移除 `first_detect_` 控制台输出和调试 `max_score` 打印
5. **算法零改动**: letterbox、postprocess 自适应 4 格式、关键点重排逻辑完全保留

### 2.2 解算组: PnP + EKF

**原始文件:**
- `solve/HNU_NHS_Vision-dev-solve/solver.cpp` (164 行) — PnP 解算
- `solve/HNU_NHS_Vision-dev-solve/predict/kalman_filter.hpp/cpp` — 卡尔曼滤波
- `solve/HNU_NHS_Vision-dev-solve/predict/motion_model.hpp/cpp` — 运动模型
- `solve/HNU_NHS_Vision-dev-solve/predict/predictor.hpp/cpp` — EKF 预测器
- `solve/HNU_NHS_Vision-dev-solve/predict/predict_types.hpp` — 类型定义

**变更:**
1. **消除重复**: 原始有两套 solver (solver.cpp + armor__solver.cpp)，只保留 solver.cpp 算法，重写为 `pnp_solver.cpp`
2. **去依赖**: PnPSolver 不再依赖 Extrinsics 外参模块（简化为相机坐标系）
3. **命名空间**: `HNU_NHS_Vision::auto_aim` → `hnu25`
4. **类型系统**: `predict_types.hpp` 中的 `TargetMeasurement`/`TargetState` 合并到 `shared/types.hpp`
5. **kalman_filter / motion_model**: 零改动，算法完全保留
6. **predictor**: 仅修 include 和 namespace

### 2.3 通信组: BCP + 串口

**原始文件:**
- `communication/.../packet_typedef.hpp` — BCP 协议 (Apache 2.0)
- `communication/.../standard_robot_pp_ros2.cpp` — ROS2 节点 (含串口收发)

**变更:**
1. **去 ROS2 化**: 移除所有 rclcpp、serial_driver、tf2 依赖
2. **精简 BCP**: 仅保留云台控制帧 (`GimbalPayload` + `buildGimbalFrame`)，移除底盘/哨兵/心率帧
3. **串口封装**: 重写 `serial_port.cpp` (150行)，原生 Linux termios + fcntl，无外部依赖
4. **协议不变**: BCP 帧格式、校验和算法、角度编码完全一致，与实车兼容

## 三、未保留的内容

| 内容 | 原因 |
|------|------|
| armor__.solver.cpp | 与 solver.cpp 功能重复 |
| 通信组 ROS2 依赖 | 实车用纯串口，非 ROS2 |
| Tongji MPC planner | 不属于三组范围 |
| Tongji tracker | 被 EKF 替代 |
| Tongji classifier/shooter/voter | 不属于三组范围 |
| fused_project 的 legacy_predictor | 被 EKF predictor 替代 |
| 解算组 solver_predictor_adapter | 合并到 main.cpp |

## 四、文件统计

| 来源 | 文件数 | 行数 | 改动程度 |
|------|:------:|:----:|:--------:|
| 识别组 (余东阳) | 2 | ~350 行 | 接口重写, 算法不变 |
| 解算组 | 6 | ~380 行 | kalman+motion 不变, pnp 重写 |
| 通信组 | 3 | ~250 行 | 完全重写去 ROS2 |
| 新增 (胶水代码) | 7 | ~450 行 | main.cpp + test_video + CMake |
| **总计** | **18** | **~1430 行** | |

## 五、部署验证

```bash
# 编译
cd deploy
cmake -B build && cmake --build build -j$(nproc)

# 视频测试 (生成标注视频)
./build/test_video demo.avi config/test.yaml demo_result.avi

# 实车运行
./build/hnu_vision config/car.yaml
```
