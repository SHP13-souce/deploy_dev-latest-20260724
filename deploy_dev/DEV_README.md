# deploy_dev — 开发副本说明

本目录是 `deploy` 的开发副本。原始 `deploy`(使用来路不明的 `yolo26.onnx`)已封档为只读基线,
本副本用于阶段 1 及之后的开发:替换检测模型为河北科技大学开源模型 + 借鉴 Awakening。

## 封档基线(请勿在此之上继续改原 deploy)

- 快照包: `C:\Users\34741\Downloads\deploy-baseline-seal-20260714.zip`
- SHA-256: `30C31C3922972B67E53D0F243CB44CEA67BF11F8F505DC20F7FD5291DFDC6BC7`
- 内容: 封档时的源码 + 配置 + 原 `assets/models/yolo26.onnx`(排除 build)
- 原 yolo26.onnx SHA-256: `11CEB7805D588C0F31F16346FC3425B8074AA8F7C146B0E6EC6AE9D541A80443`
  - 注: 该模型来路不明、性能与 HNU_NHS_Vision 同源,阶段 0 已决定弃用。

## 新检测模型来源(河北科技大学 Actor&Thinker 战队)

- 仓库: https://github.com/PraySky1337/AT_NN_Detector
- 本地 zip: `C:\Users\34741\Downloads\AT_NN_Detector-main.zip`
- 架构: Ultralytics YOLO26n-Pose,装甲板四角点姿态检测
- 存放位置: `assets/models/hebei_at_nn/`
- 输出格式: `1x30x18`
  - `[0:4]` xyxy (letterbox 坐标)
  - `[4]`   confidence
  - `[5]`   class_id (0-11)
  - `[6:10]` color scores (B/R/G/P, 取 argmax)
  - `[10:18]` 4 个关键点 (顺时针)
- 类别 (12): `sX_oY`,s0=小装甲 s1=大装甲;o0=哨兵 o1=英雄 o2..o5=步兵2-5 o6=前哨站 o7=基地
  - 未覆盖: RM2026 基地小装甲
- 输入: RGB, [0,1] 归一化, letterbox pad=114

### ⚠️ 许可证 (AGPL-3.0)

模型基于 Ultralytics AGPL-3.0 训练。集成本模型的衍生项目必须:
- 同样采用 AGPL-3.0 开源
- 保留版权与 License 文本 (见 `assets/models/hebei_at_nn/LICENSE`)
- 任何基于它的网络服务需开源完整源码

## 阶段 0 评测结论 (OpenVINO CPU, WSL, conf>=0.25)

在 687 帧 demo.avi 上:

| 模型 | 单帧推理 | 推理FPS | 检出覆盖 |
|---|---:|---:|---:|
| **coord_noe2e_640** (选定) | ~10 ms | ~100 | 92% |
| coord_noe2e_768 | ~10.5 ms | ~96 | 92% |
| c2psa_e2e_640 | ~25 ms | ~40 | 92% |
| c2psa_e2e_768 | ~20 ms | ~49 | 93% |
| 原神秘 yolo26 (旧基准) | ~50 ms | ~20 | 52% |

- **选定 CoordAtt(noe2e)640×640**: 纯 CPU 上比 C2PSA 快 2-2.5 倍,检出率持平;
  比原模型快 5 倍,检出覆盖 52%→92%。
- 640 vs 768 速度几乎持平;768 的免-letterbox 福利针对 1440×1080 相机,
  本队海康 MV-CA020-10UC 为 2048×1536,享受不到,故选 640。
- 河北自带 3 段视频检出低(4-22%)是因视频本身多为远景/运镜、近距清晰装甲板少,
  非模型缺陷(检到时置信度 0.73-0.79 正常)。

## 开发路线 (先后顺序)

- [x] 阶段 0: 验证河北模型 + 选定 CoordAtt-640 (纯 Python, 未改 C++)
- [x] 阶段 1: 改检测器后处理适配 `1x30x18`/12类/独立颜色分支,重写 types.hpp 映射,
       内存复用优化,同基准对比新旧模型端到端表现
- [x] 阶段 2: 引入 SP 整车 EKF/弹道,补齐坐标变换、时间同步与安全开火门
- [x] 阶段 3: 接入 PlotJuggler UDP telemetry 与共享内存 Web 调试闭环
- [x] 阶段 4: 接入 Awakening opt-1208 + 可选 MLP 复分类，保留河北后端做运行时 A/B

## Awakening 检测后端 A/B

`detection::makeDetector` 现在提供 `hebei` 与 `awakening` 两个后端，统一输出
`DetectedArmor`。Awakening 适配器只移植 MIT 许可下必要的 416x416 BGR 预处理、TUP
网格解码、NMS/角点合并和可选 `mlp_finetuned.onnx` 数字复分类，不引入其 Scheduler、
C++23、TBB 或 Ceres 依赖。第三方声明见 `docs/THIRD_PARTY_NOTICES.md`。

统一 WSL、687 帧 `demo.avi` 实测：

| 后端 | 模式 | 检测数/有检测帧 | 检测耗时 | SP tracking 帧 | NIS 均值 |
|---|---|---:|---:|---:|---:|
| 河北 CoordAtt-640 | latency | 851 / 633 | 16.074 ms | 431 | 0.648 |
| Awakening opt-1208 | latency，无 MLP | 838 / 573 | 9.293 ms | 未测 | 未测 |
| Awakening opt-1208 | throughput，无 MLP | 838 / 573 | 4.595 ms | 145 | 15.823 |
| Awakening opt-1208 + MLP | throughput | 838 / 573 | 4.574 ms（全链计时中的检测段） | 283 | 1.183 |

因此 Awakening 是已接通的高速可选前端，但不是当前默认替代：河北在本录像上多覆盖 60
帧，SP 跟踪连续性也更好。实车应分别验收静止、平移和小陀螺命中率后再决定车辆配置，
不能只按平均推理耗时选模型。

## 阶段 1 实施结果

检测器已固定到河北科技大学 CoordAtt-640 的明确契约，不再保留旧模型的
17/18/52/53 列和 36 类兼容分支：

- OpenVINO 设备固定为 CPU，性能模式为 `LATENCY`
- 输入 Tensor、letterbox canvas 和 resize Mat 跨帧复用
- BGR → RGB/NCHW 直接写入 OpenVINO Tensor，不再 `split` + `memcpy`
- 输出 Tensor 直接解析，不再包装并 `clone` 完整输出
- 12 类映射严格采用官方 `PAIR_TO_ARMOR`
- 独立颜色分支映射为 B/R/G/P
- 官方关键点顺序按 `KPT_ORDER={0,3,2,1}` 转为下游 PnP 的 TL/TR/BR/BL

构建环境：WSL Ubuntu 24.04，GCC 13.3，OpenCV 4.6，OpenVINO 2024.6，Release。
统一输入：1440×1080 `demo.avi`，687 帧，置信度阈值 0.25。

| 指标 | 旧 deploy | deploy_dev 阶段 1 |
|---|---:|---:|
| 有检测帧 | 360/687 (52%) | **633/687 (92%)** |
| 跟踪有效帧 | 643/687 (93%) | **686/687 (99%)** |
| 检测阶段平均耗时 | 50.827 ms | **11.969 ms** |
| 检测阶段吞吐 | 19.675 FPS | **83.549 FPS** |
| 纯算法吞吐 | 9.591 FPS | **77.418 FPS** |
| 含 MJPEG 标注视频写出 | 9.591 FPS (旧总基准) | **23.535 FPS** |

结果：检测覆盖增加 40 个百分点；最新复跑中检测阶段耗时下降 76.5%，纯算法吞吐约为
旧版 8.07 倍。阶段 1 多次复跑的检测耗时为 11.97-14.36 ms，受 CPU 调度和模型缓存
影响；标注视频编码会将整体速度压到 23.5 FPS，不代表实车主循环性能。

基准日志：

- `/home/robomaster/bench_results/hnu25_dev_stage1_no_video.log`
- `/home/robomaster/bench_results/hnu25_dev_stage1.log`
- `/home/robomaster/bench_results/hnu25_dev_stage1.avi`

## PnP 尺寸约定

装甲板组件整体外形尺寸与神经网络关键点对应的 PnP 特征尺寸不是同一组数据：

| 类型 | 型号 | 组件整体外形 | 当前 PnP 灯条特征几何 |
|---|---|---:|---:|
| 大装甲板 | AM11 | 235 x 127 mm | 230 x 56 mm |
| 小装甲板 | AM01 | 140 x 125 mm | 135 x 56 mm |

河北科大 ATLabelMaster 的标注规范要求从左上角开始按灯条标注，模型四点对应两侧灯条
端点形成的四边形。因此 `solve/pnp_solver.cpp` 使用 SP Vision 同源的灯条中心间距和灯条
长度，而不能直接使用组件外壳的 125/127 mm 高度。原代码误将小装甲写为 230 x 54 mm，
现已修正为小 135 x 56 mm、大 230 x 56 mm。

后续仍需用实物测量和定距靶验证 PnP 距离及重投影误差，并用动态 IMU 完成实车验证。

## 阶段封档

- 原始 `deploy` 基线封档：`C:\Users\34741\Downloads\deploy-baseline-seal-20260714.zip`
- 原始基线 SHA-256：`30C31C3922972B67E53D0F243CB44CEA67BF11F8F505DC20F7FD5291DFDC6BC7`
- 河北检测器阶段封档：`C:\Users\34741\Downloads\deploy-dev-hebei-detector-20260714.zip`
- 河北检测器阶段 SHA-256：`6548FE889EE73083FB688718E63496A50857C7CB1D8369F9FEA342FDA901B31C`
- 本次最终小范围集成不再生成 zip；由主代理在合并确认后统一封档和计算最终 hash。

## 最终架构

主链保持单向、同步和可审计，不引入 Awakening Scheduler：

```text
camera frame + timestamped IMU pose
  -> configurable Hebei YOLO26 / Awakening opt-1208 detector
  -> armor adapter + IPPE PnP + reprojection gate
  -> SP whole-vehicle 11-state EKF tracker + NIS gate
  -> SP armor selection + ballistic aimer
  -> FireGate -> command mapping -> SP serial protocol
  -> telemetry JSON -> PlotJuggler UDP 127.0.0.1:9870
  -> current frame + same telemetry JSON -> debug Web shared memory
```

`transform::TimePoseBuffer` 负责姿态按帧时间查询、插值、有限外推和过期拒绝；
`aim::FireGate` 独立检查跟踪、姿态年龄、反馈、PnP、NIS、弹道和角度窗口。Web 与
PlotJuggler 都是主链末端只读旁路，不参与算法、串口或开火判定。

## SP 来源与明确修复

整车目标模型、跟踪状态机、NIS 统计门、选板和弹道行为来自本队
`/home/robomaster/projects/sp_vision_25-main` 的 SP Vision 实现，迁入 `sp_core/` 后去除
设备/UI 耦合并用可单测接口固定。不是照搬 Awakening 内核，Awakening 仅作为时间同步、
安全边界和调试闭环的设计参考。

本阶段明确完成的修复：

- 河北模型输出固定为 `1x30x18`，严格使用官方 12 类、独立 B/R/G/P 颜色分支和
  `KPT_ORDER={0,3,2,1}`，删除旧模型模糊兼容分支。
- PnP 几何改为灯条特征尺寸：小装甲 `135 x 56 mm`、大装甲 `230 x 56 mm`，并增加
  四点方向、正深度、有限值和最大重投影误差拒绝。
- 姿态与视频使用同一时间线；录像段时间跳变时重置 tracker/pose buffer，缺姿态帧跳过
  world-frame PnP/跟踪，避免使用陈旧姿态污染 EKF。
- SP 11 维整车 EKF 使用配置化 Q/R、Joseph covariance update、协方差对称化、LDLT
  检查、NIS 窗口统计、半径发散检查和最大帧间隔重置。
- 目标选择仅接受敌方且 PnP 有效的装甲；弹道失败返回无效 aim，不产生非有限指令。
- 通信采用 golden-vector 校验的 SP 帧协议、流式解析和发送限频；`mapCommand` 对关闭开关
  和非有限角度 fail closed。
- `FireGate` 与 `auto_fire`/瞄准结果解耦，缺反馈、陈旧姿态、坏 PnP/NIS、未稳定等均拒火。
- PlotJuggler 与 Web 共用 `telemetry::encodeJson(FrameSample)`；Web 使用限频 JPEG 共享内存，
  默认关闭时不创建发布资源，也不改变算法、串口或 `FireGate`。

## 模块与运行命令

| 模块 | 作用 |
|---|---|
| `detection/` | 河北 YOLO26 / Awakening opt-1208 双 OpenVINO CPU 后端 |
| `sp_core/` | 类型适配、PnP、SP 整车 EKF、状态机、选板与弹道 |
| `transform/` | `TimePoseBuffer` 姿态时间对齐和过期保护 |
| `aim/` | 独立 `FireGate` |
| `communication/` | SP 协议、流解析、串口和发送限频 |
| `telemetry/` | UDP JSON 发布和 PlotJuggler 布局 |
| `debug_web/` | JPEG 共享内存发布、状态 JSON 和 Flask 页面 |
| `pipeline/` | `sp_standard` 相机/串口实车胶水 |
| `tests/sp_demo_replay.cpp` | 视频 + IMU 文本全链路离线回放 |

WSL Release 构建与测试：

```bash
cd /mnt/d/HNU_RM/HNU_NHS_Vision-25/deploy_dev
cmake -S . -B build-final-release -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build-final-release -j$(nproc)
ctest --test-dir build-final-release --output-on-failure
```

全回放：

```bash
/usr/bin/time -f 'wall=%e s' ./build-final-release/tests/sp_demo_replay \
  /home/robomaster/projects/sp_vision_25-main/assets/demo/demo.avi \
  /home/robomaster/projects/sp_vision_25-main/assets/demo/demo.txt \
  config/sp_demo.yaml
```

实车入口：

```bash
./build-final-release/pipeline/sp_standard config/sp_standard.yaml
```

## 验证结果

Release CTest 当前共 10 项，全部通过：除原有 SP 核心、姿态、FireGate、协议、Web、
telemetry 和回放测试外，新增 `camera_frame_source` 与 `mpc_planner`。

Release 全回放最终指标：

| 指标 | 结果 |
|---|---:|
| frames | 687 |
| pose frames / missing / timeline resets | 620 / 67 / 9 |
| detections | 851 |
| PnP valid / rejected / skipped without pose | 789 / 2 / 60 |
| tracking frames | 431 |
| aim valid / safe aim | 571 / 420 |
| reprojection mean / max | 0.756 px / 4.25 px |
| NIS mean / samples | 0.648 / 560 |
| mean total latency | 10.447 ms |
| wall time | 9.87 s |

ASan Debug 中 7 个纯测试通过。OpenVINO replay smoke 未在 ASan 下运行：OpenVINO CPU
插件所用 TBB 与 AddressSanitizer 的 `RTLD_DEEPBIND` 存在已知兼容问题，属于运行时加载
限制，不是纯模块测试失败。复现纯测试的命令为：

```bash
cmake -S . -B build-final-asan -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DBUILD_TESTS=ON
cmake --build build-final-asan -j$(nproc)
ctest --test-dir build-final-asan --output-on-failure -E sp_demo_replay_smoke
```

## 调试入口

PlotJuggler 布局路径：`telemetry/sp_demo_plotjuggler.xml`。在 PlotJuggler Streaming 中
启动 UDP Server 监听 `9870`，再运行 `sp_demo_replay` 或 `sp_standard`。数据是 UDP JSON，
不是 ROS2 topic。

Web 默认关闭。将对应 YAML 的 `web.enabled` 改为 `true`，安装依赖后运行：

```bash
python3 -m venv .venv-web
.venv-web/bin/pip install -r debug_web/requirements.txt
DEBUG_WEB_PYTHON=.venv-web/bin/python debug_web/debug.sh \
  ./build-final-release/pipeline/sp_standard config/sp_standard.yaml
```

浏览器访问 `http://127.0.0.1:5000`。离线查看时把包装命令替换为上面的
`sp_demo_replay` 命令即可。

## 安全默认与待办

`config/sp_standard.yaml` 中 `command_enabled: false`、`planner.enabled: false`、
`fire_gate.enabled: false`、`web.enabled: false`。四者必须分别显式开启，Web 不具备控制
或开火能力。`planner.use_feedback_velocity` 也默认关闭，直到下位机速度字段单位得到确认。

仍需在实车完成：

1. 在目标 NUC 用实际相机序列号启用 `camera.type: hik`，验证曝光、增益、帧率和曝光中心
   时间戳；代码已完成 MVS 条件编译与参考 SDK 链接验证。
2. 用实车串口抓包确认下位机反馈帧类型、地址方向、字段与单位；当前地址校验默认不启用，
   回显拒绝默认启用，速度字段默认仅用于遥测、不进入 MPC。
3. 确认 yaw/pitch/roll 符号和 ZYX 旋转契约后，再依次开启 command、angle-only MPC 和
   FireGate。

## SP IO、海康与 MPC 状态

海康相机已从 SP Vision 提取为独立 `camera/FrameSource`，并修复原实现中 SDK buffer
释放后引用、句柄失败清理、固定选择第一台相机和旧帧积压问题。Hik 输出自有 BGR 图像、
海康帧号、接收时间和曝光中心近似时间；无 MVS 时仍可构建 OpenCV 相机源。使用 SP
仓库自带 SDK 的条件构建已成功链接 `libMvCameraControl.so`。

下位机 IO 没有原样复制旧 `CBoard`。当前以 `sp_protocol + TimePoseBuffer` 为唯一反馈
链：反馈带 `received_at`，支持可配置符号、ZYX 四元数、时间偏移、过期保护、可选地址
校验和完整帧回显拒绝；图像线程在 `Frame.captured_at` 查询姿态。

融合仓库中的 MPC 实际是“完整继承的旧 `Planner + TinyMPC + standard_mpc` 独立入口”，
不是空接口；但新人 `fusion_test/deploy` 完全旁路，因此并未真正融合进新人主链。旧实现
还存在目标轨迹代替真实云台初态、输出 index 50（约 0.5 s 后）以及当前 SP 协议丢弃
速度/加速度等问题。

当前 `planner/` 是修正后的独立 angle-only MPC：位于 Aimer 和 FireGate 之间，使用实际
云台反馈为初态，yaw 做周期展开，使用双积分模型和加速度约束，输出第一个可执行点；失败
时回退几何 Aimer。速度和加速度进入 PlotJuggler/Web 遥测，但现有协议仍只发送规划角度。
该模块采用自包含约束二次规划求解，不是旧 TinyMPC 源码黑盒照搬，默认关闭。

最终验证：普通 Release 10/10 CTest 通过；MVS 条件构建成功；不加载 OpenVINO 的
Debug+ASan 6/6 通过；687 帧功能结果保持 851 检测、789 PnP 成功、431 tracking、
571 aim、420 safe aim、重投影均值 0.756 px、NIS 均值 0.648。WSL 调度导致全核心耗时
在约 10.45-16.79 ms 间波动，功能计数一致。
