# VisionTelemetry 协议 (v1)

本文档描述反无人机视觉结果的线上传输格式。它仅用于固定大小的诊断/结果传输——不携带任何控制、开火、执行器或云台命令。

## 概述

| 字段   | 值                  |
|--------|---------------------|
| 协议   | VisionTelemetry v1  |
| 包大小 | 58 字节（固定）     |
| 字节序 | 小端序              |
| 魔数   | `0x41 0x44`（"AD"） |
| CRC    | CRC-16/CCITT-FALSE  |

## CRC-16/CCITT-FALSE

| 参数   | 值     |
|--------|--------|
| poly   | 0x1021 |
| init   | 0xFFFF |
| refin  | false  |
| refout | false  |
| xorout | 0x0000 |

CRC 覆盖从魔数头开始、到 2 字节 CRC 尾部之前（不含）的每个字节（字节 0..55）。它以小端序传输（低字节在前）。

## 字节布局

| 偏移量（字节） | 字段                  | 类型    | 说明                |
|----------------|-----------------------|---------|---------------------|
| 0–1            | magic                 | uint8   | `0x41 0x44`（"AD"） |
| 2              | version               | uint8   | `1`                 |
| 3              | payload_length        | uint8   | `52`                |
| 4–7            | sequence              | uint32  | 小端序              |
| 8–15           | timestamp_us          | uint64  | 小端序              |
| 16–17          | status_flags          | uint16  | 小端序，见下文      |
| 18             | track_state           | uint8   | 见下文              |
| 19             | reserved              | uint8   | 必须为 0            |
| 20–23          | yaw_rad               | float32 | 小端序              |
| 24–27          | pitch_rad             | float32 | 小端序              |
| 28–31          | x_m                   | float32 | 小端序              |
| 32–35          | y_m                   | float32 | 小端序              |
| 36–39          | z_m                   | float32 | 小端序              |
| 40–43          | prediction_horizon_s  | float32 | 小端序              |
| 44–45          | detection_count       | uint16  | 小端序              |
| 46–47          | pnp_measurement_count | uint16  | 小端序              |
| 48–51          | yaw_speed_rad_s       | float32 | 小端序              |
| 52–55          | pitch_speed_rad_s     | float32 | 小端序              |
| 56–57          | CRC                   | uint16  | 小端序              |

## status_flags（位 0–4）

| 位   | 字段                  |
|------|-----------------------|
| 0    | calibration_available |
| 1    | vision_valid          |
| 2    | track_available       |
| 3    | prediction_valid      |
| 4    | measurement_updated   |
| 5–15 | reserved（必须为 0）  |

## track_state

| 值 | 名称      |
|----|-----------|
| 0  | LOST      |
| 1  | DETECTING |
| 2  | TRACKING  |
| 3  | TEMP_LOST |

## 单位

- `yaw_rad` / `pitch_rad`：弧度
- `x_m` / `y_m` / `z_m`：米
- `prediction_horizon_s`：秒
- `timestamp_us`：微秒
- `yaw_speed_rad_s` / `pitch_speed_rad_s`：弧度/秒

## 语义

**`vision_valid` 控制方向/位置字段的有效性。** 仅当 `vision_valid == 1` 时，`yaw_rad`、`pitch_rad`、`x_m`、`y_m`、`z_m` 才携带有效的视觉结果。当 `vision_valid == 0` 时，这些字段无意义——值 `0` 绝不能解读为“真实的零角度/零位置”。

**`yaw_speed_rad_s` / `pitch_speed_rad_s`** 是补偿后指向方向的云台跟随角速率，单位为弧度/秒。它们是连续有效帧之间 `yaw_rad` / `pitch_rad` 的低通滤波一阶差分。在第一个有效帧以及 `vision_valid == 0` 时它们为 `0`，因此速率从不跨越目标丢失的间隔。

**`sequence`** 是分配给每个成功处理、非空帧的计数器（模 2^32）。它让下游消费者能够检测丢失、重复或乱序的数据包。从 `4294967295` 到 `0` 的回绕是连续的，而非间断。

**`timestamp_us`** 是帧的 `captured_at`（`std::chrono::steady_clock`）的单调进程时间，转换为微秒。它不是 UTC 墙上时钟时间戳。

## 传输说明

- **固定大小。** 每个数据包恰好 58 字节；没有随包变化的长度或分隔符字段。
- **按字节流解析。** 串行链路传递的是字节而非数据包：单次读取可能返回部分包、多个包，或跨多次读取拆分的包。切勿假设一次读取等于一个数据包。
- **按魔数重新同步。** 接收方搜索 `0x41 0x44`（"AD"）魔数标记，将其之前的内容视为需丢弃的垃圾。这使得解析器能够从流中途开始或某个损坏字节中恢复。
- **CRC 失败丢弃 + 重新同步。** CRC 不匹配（或版本 / 负载长度 / 跟踪状态 / 浮点字段无效）的候选包会被丢弃，解析器继续搜索下一个魔数标记。一个坏包不会毒化流的其余部分。
- **用 sequence 检测丢失 / 重复 / 乱序。** `sequence` 让消费者能够检测间隔（丢失）、重复（重复包）和乱序投递。它以 2^32 为模回绕；回绕是连续的，而非间断。
- **`vision_valid` 控制方向/位置字段。** `yaw_rad`、`pitch_rad`、`x_m`、`y_m`、`z_m` 仅在 `vision_valid == 1` 时有意义（见上文“语义”）。

## 控制语义

本协议仅面向下游消费者的视觉结果 / 遥测 / 诊断结果传输。它不包含任何开火命令、射击命令、执行器命令或云台控制映射。本文档及其代码路径中的任何内容都不会发出控制或开火动作。
