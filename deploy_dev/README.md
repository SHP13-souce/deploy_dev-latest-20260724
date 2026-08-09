# HNU_NHS_Vision-25 deploy_dev

以 2025 级三组 `deploy` 为基础继续开发的轻量自瞄系统。主链保持接近 SP Vision 的
直线式数据流；Awakening 仅作为时间同步、安全边界和调试能力的参考，不照搬 Scheduler。

```text
frame/pose -> detect(Hebei YOLO26 / Awakening opt-1208) -> PnP -> SP whole-vehicle EKF -> ballistic aim
                                                        |-> FireGate -> BCP serial
                                                        |-> UDP JSON -> PlotJuggler
                                                        `-> SHM/JPEG + JSON -> Web
```

## 快速开始

### 编译与测试 (WSL/Linux)

```bash
cd /mnt/d/HNU_RM/HNU_NHS_Vision-25/deploy_dev
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

依赖：OpenCV、Eigen3、yaml-cpp、OpenVINO 2024+。

### 检测后端

`sp_standard` 和 `sp_demo_replay` 支持两个 OpenVINO CPU 后端，输出统一为
`DetectedArmor`，因此 PnP、SP 整车 EKF、Aimer、MPC 和 FireGate 不需要分支：

```yaml
detector:
  backend: awakening
  model_path: /path/to/awakening/model/opt-1208-001.onnx
  classifier_path: /path/to/awakening/model/mlp_finetuned.onnx  # 推荐，可省略
  conf_threshold: 0.20
  performance_mode: throughput
```

- `hebei` 是默认后端，使用 640x640 `[1,30,18]` 模型，当前回放覆盖更高。
- `awakening` 使用 416x416 TUP `opt-1208`；启用 MLP 复分类可显著减少类别抖动。
- `latency` 适合低尾延迟；`throughput` 在当前 CPU 同步回放中更快，但仍应在目标 NUC
  上复测温度、P95/P99 延迟和整链命中率。
- Awakening 模型不随本目录重复分发。源码适配和模型来源声明见
  [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。

只比较检测前端时运行：

```bash
./build/tests/detector_benchmark awakening /path/to/opt-1208-001.onnx \
  /path/to/demo.avi 687 0.20 throughput /path/to/mlp_finetuned.onnx
```

### SP 全链路回放

```bash
./build/tests/sp_demo_replay \
  /home/robomaster/projects/sp_vision_25-main/assets/demo/demo.avi \
  /home/robomaster/projects/sp_vision_25-main/assets/demo/demo.txt \
  config/sp_demo.yaml
```

`sp_demo_replay` 使用录像和对应 IMU 四元数文本，无需相机和串口即可验收检测、PnP、
SP 整车 EKF 跟踪和弹道链路。只验证旧检测流水线时仍可运行：

```bash
./build/tests/test_video /path/to/demo.avi config/test.yaml -
```

### PlotJuggler

启动 PlotJuggler，在 Streaming 中添加 `UDP Server` 并监听 `9870`，再导入：

```text
telemetry/sp_demo_plotjuggler.xml
```

运行 `sp_demo_replay` 或 `sp_standard` 后，JSON key 会作为曲线出现。这里不是 ROS topic，
不需要 ROS2。

### Web 调试

先将所用 YAML 的 `web.enabled` 改为 `true`，然后用包装脚本同时启动视觉程序和 Web：

```bash
python3 -m venv .venv-web
.venv-web/bin/pip install -r debug_web/requirements.txt
DEBUG_WEB_PYTHON=.venv-web/bin/python debug_web/debug.sh \
  ./build/tests/sp_demo_replay \
  /home/robomaster/projects/sp_vision_25-main/assets/demo/demo.avi \
  /home/robomaster/projects/sp_vision_25-main/assets/demo/demo.txt \
  config/sp_demo.yaml
```

浏览器打开 `http://127.0.0.1:5000`。实车调试时将包装命令替换为
`./build/pipeline/sp_standard config/sp_standard.yaml`。

实验室 NUC `192.168.1.144` 可安装 `debug_web/hnu25-debug-view.service`，使用
`config/nuc144_safe.yaml` 在 `0.0.0.0:5000` 提供只读调车画面。该安全配置保持串口、
反馈姿态、MPC、控制指令和 FireGate 全部关闭，浏览器访问
`http://192.168.1.144:5000`。服务管理命令：

```bash
sudo systemctl status hnu25-debug-view
sudo systemctl restart hnu25-debug-view
sudo systemctl stop hnu25-debug-view
journalctl -u hnu25-debug-view -f
```

Web 支持按需开启原图、检测标注、灰度和可调阈值二值图。默认只订阅原图和检测标注，
避免手机同时拉取四路 MJPEG。参数面板只开放检测置信度、二值阈值、弹速、yaw/pitch
偏置、选板角、转速阈值和高低速延时：

- `临时应用`：在当前进程帧边界生效，不修改 YAML，服务重启后恢复已保存值。
- `确认保存`：先应用，再原子更新当前运行配置 YAML。
- 串口、反馈姿态、控制指令、MPC、FireGate 和自动开火不在 Web 白名单内，提交额外键或
  越界值会被 HTTP 层和 C++ 层双重拒绝。

四路流也可独立访问：`/video/raw`、`/video/annotated`、`/video/gray`、
`/video/binary`。参数状态与更新接口为 `GET/POST /parameters`。

### 实车 SP 主程序

确认相机、串口、内外参和敌方颜色后运行：

```bash
./build/pipeline/sp_standard config/sp_standard.yaml
```

`config/sp_standard.yaml` 的 `command_enabled`、`planner.enabled`、`fire_gate.enabled`、
`web.enabled` 均默认为 `false`。`camera.type` 可选 `opencv` 或 `hik`；启用
`feedback_pose` 后，串口云台反馈会写入姿态时间缓存并在图像曝光时刻查询。反馈地址、
速度字段单位和正负方向仍应以实车抓包确认，不得仅因程序能启动就开启指令或发射。

## 目录

| 目录 | 内容 | 来源 |
|---|---|---|
| `shared/` | 检测公共类型 | deploy_dev |
| `detection/` | 河北 YOLO26 / Awakening opt-1208 双 OpenVINO 检测后端 | 河北模型 + deploy_dev；Awakening MIT 适配 |
| `solve/` | 旧 PnP/EKF 解算 | 解算组 |
| `sp_core/` | PnP、SP 整车 EKF、状态机、选板和弹道 | SP Vision 行为移植并明确修复 |
| `camera/` | OpenCV/Hik FrameSource、曝光时间戳和 latest-frame | SP Hik 安全重构 |
| `transform/` | 姿态时间缓存、插值和过期保护 | 新建 |
| `aim/` | 独立安全开火门 `FireGate` | 新建 |
| `planner/` | 使用真实云台反馈的修正版 angle-only MPC | 基于 SP/TinyMPC 思路重写 |
| `communication/` | BCP/SP 协议与串口 | 通信组 + 安全修复 |
| `telemetry/` | PlotJuggler UDP/JSON 与布局 | 新建 |
| `debug_web/` | 共享内存帧发布和 Flask Web | 新建 |
| `pipeline/` | 旧主程序及 `sp_standard` 实车胶水 | 新建 |
| `config/` | 运行配置 | 新建 |
| `tests/` | 纯测试、协议测试和全链路回放 | 新建 |

开发基线、模型来源、许可证、架构、明确修复和最终验收结果见
[DEV_README.md](DEV_README.md)。各组早期融合记录见
[docs/fusion_changes.md](docs/fusion_changes.md)。
