# VisionTelemetry Protocol (v1)

This document describes the on-the-wire format of the anti-drone vision
result. It is a fixed-size diagnostic / result transport only — it carries no
control, fire, actuator, or gimbal commands.

## Overview

| Field        | Value                |
|--------------|----------------------|
| Protocol     | VisionTelemetry v1   |
| Packet size  | 50 bytes (fixed)     |
| Endian       | little-endian        |
| Magic        | `0x41 0x44` ("AD")   |
| CRC          | CRC-16/CCITT-FALSE   |

## CRC-16/CCITT-FALSE

| Parameter | Value  |
|-----------|--------|
| poly      | 0x1021 |
| init      | 0xFFFF |
| refin     | false  |
| refout    | false  |
| xorout    | 0x0000 |

The CRC covers every byte from the magic header up to (but not including) the
2-byte CRC trailer (bytes 0..47). It is transmitted little-endian (low byte
first).

## Byte layout

| Offset (bytes) | Field                 | Type    | Notes                       |
|----------------|-----------------------|---------|-----------------------------|
| 0–1            | magic                 | uint8   | `0x41 0x44` ("AD")          |
| 2              | version               | uint8   | `1`                         |
| 3              | payload_length        | uint8   | `44`                        |
| 4–7            | sequence              | uint32  | little-endian               |
| 8–15           | timestamp_us          | uint64  | little-endian               |
| 16–17          | status_flags          | uint16  | little-endian, see below    |
| 18             | track_state           | uint8   | see below                   |
| 19             | reserved              | uint8   | must be 0                   |
| 20–23          | yaw_rad               | float32 | little-endian               |
| 24–27          | pitch_rad             | float32 | little-endian               |
| 28–31          | x_m                   | float32 | little-endian               |
| 32–35          | y_m                   | float32 | little-endian               |
| 36–39          | z_m                   | float32 | little-endian               |
| 40–43          | prediction_horizon_s  | float32 | little-endian               |
| 44–45          | detection_count       | uint16  | little-endian               |
| 46–47          | pnp_measurement_count | uint16  | little-endian               |
| 48–49          | CRC                   | uint16  | little-endian               |

## status_flags (bits 0–4)

| Bit  | Field                 |
|------|-----------------------|
| 0    | calibration_available |
| 1    | vision_valid          |
| 2    | track_available       |
| 3    | prediction_valid      |
| 4    | measurement_updated   |
| 5–15 | reserved (must be 0)  |

## track_state

| Value | Name       |
|-------|------------|
| 0     | LOST       |
| 1     | DETECTING  |
| 2     | TRACKING   |
| 3     | TEMP_LOST  |

## Units

- `yaw_rad` / `pitch_rad`: radians
- `x_m` / `y_m` / `z_m`: meters
- `prediction_horizon_s`: seconds
- `timestamp_us`: microseconds

## Semantics

**`vision_valid` gates the direction / position fields.** Only when
`vision_valid == 1` do `yaw_rad`, `pitch_rad`, `x_m`, `y_m`, `z_m` carry a
valid vision result. When `vision_valid == 0`, those fields are meaningless — a
value of `0` must NOT be interpreted as "true zero angle / zero position".

**`sequence`** is a counter (modulo 2^32) assigned to each successfully
processed, non-empty frame. It lets a downstream consumer detect lost,
duplicate, or out-of-order packets. Rollover from `4294967295` to `0` is
continuous, not a gap.

**`timestamp_us`** is the monotonic process time of the frame's `captured_at`
(`std::chrono::steady_clock`), converted to microseconds. It is NOT a UTC
wall-clock timestamp.

## Control semantics

This protocol is a vision-result / telemetry / diagnostic-result transport
only, for a downstream consumer. It contains no fire command, shoot command,
actuator command, or gimbal-control mapping. Nothing in this document or its
code path issues control or fire actions.
