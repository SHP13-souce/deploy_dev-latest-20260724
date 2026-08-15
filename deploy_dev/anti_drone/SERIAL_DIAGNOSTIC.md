# SERIAL_DIAGNOSTIC Transport

The production anti-drone application can deliver its fixed 50-byte
`VisionTelemetry v1` packet over a real serial device, for a separate read-only
receiver. This is a **software diagnostic mode**: it forwards the vision
*result* only. It does **not** map the result onto any actuator / gimbal / fire
command, and it never links the SP control protocol.

## Mode selection

The application transport is chosen by the `telemetry:` section of
`config/anti_drone.yaml`:

```yaml
telemetry:
  mode: "loopback"            # or "serial_diagnostic"
  device: "/dev/ttyUSB0"      # serial device path (required for serial_diagnostic)
  baud_rate: 115200           # 9600 | 19200 | 38400 | 57600 | 115200
  max_consecutive_failures: 5 # consecutive send failures that stop the app
  flush_after_write: false    # tcdrain after each accepted write
```

- `mode: "loopback"` (default) keeps the in-memory self-check; no serial device
  is ever opened.
- `mode: "serial_diagnostic"` opens `device` before the frame loop and forwards
  every encoded packet over `hnu25::SerialPort`.

## What the transport does (and does not do)

`VisionTelemetrySerialTransport` only:

1. validates the packet is exactly 50 bytes,
2. manages the open/close lifecycle,
3. calls `hnu25::SerialPort::write()` (which already handles partial writes,
   `EINTR`, and `EAGAIN`/`poll`),
4. optionally flushes (`tcdrain`) after each write when `flush_after_write` is
   true, and
5. accumulates statistics.

It performs **no** retry/resend queue: a stale visual result has no value, so a
failed packet is dropped and the next frame's latest result is sent instead. It
expresses **no** control / fire / gimbal / actuator semantics.

## Error semantics

- **Open failure** (serial device cannot be opened): the application prints a
  clear error and exits non-zero (`9`) before entering the frame loop.
- **Send failure**: each failed `send()` is counted. A success resets the
  counter to zero. When `max_consecutive_failures` consecutive sends fail, the
  application prints `Telemetry transport failed N consecutive times;
  stopping.` and exits non-zero (`10`).

## Verifying without hardware (PTY)

`tests/anti_drone_vision_telemetry_serial_integration_test.cpp` opens a
pseudo-terminal (`posix_openpt`) and runs the transport over the PTY slave, so
the full `open -> write -> read -> parse` round-trip is exercised with no real
serial hardware. It verifies a 5-packet sequence `100..104` decodes in order
with zero CRC/format errors and zero sequence gaps, and checks the transport
statistics (`packets_submitted`, `packets_accepted`, `bytes_accepted`,
`failures`).

## Transport statistics

`VisionTelemetryTransportStats` accumulates:

| Counter              | Meaning                                     |
|----------------------|---------------------------------------------|
| `packets_submitted`  | every `send()` call                         |
| `packets_accepted`   | packets accepted by the serial device       |
| `bytes_accepted`     | accepted packet bytes (50 per packet)       |
| `failures`           | wrong-size, not-open, or write failures     |

## Read-only receiver

`anti_drone_vision_telemetry_receiver <device> [baud_rate]` reads a serial
stream, re-parses it with `VisionTelemetryStreamParser`, and prints each decoded
packet. It is read-only: it never constructs a `GimbalCommand`, links
`sp_protocol`, or touches any hardware / actuation path.
