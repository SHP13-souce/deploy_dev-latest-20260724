#!/usr/bin/env python3
"""Windows-native Daedalus (Talos SHM) client for deploy_dev models.

Reads frames from F:\\talos_shm\\talos_ipc_* (same layout as Daedalus talos-ipc),
runs Hebei OpenVINO armor detection, writes GimbalCmd back.

Usage:
  python tools/talos_win_client.py
  python tools/talos_win_client.py --shm-dir F:\\talos_shm --show
"""
from __future__ import annotations

import argparse
import math
import mmap
import os
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from openvino import Core

# ---- talos-ipc layout (layout.rs) ----
IMAGE_W, IMAGE_H = 1440, 1080
IMAGE_CH = 3
IMAGE_SIZE = IMAGE_W * IMAGE_H * IMAGE_CH
IMAGE_POOL_SIZE = IMAGE_SIZE * 3
META_SIZE = 3712

IMAGE_TB_OFFSET = 64
POSE_GIMBAL_OFFSET = 256  # poses[0]
GIMBAL_CMD_OFFSET = 1536
FLAG_NEW = 0x80
INDEX_MASK = 0x03

# ImageTripleBuffer: state@0, slots start @64 inside TB => absolute 128
IMAGE_SLOTS_OFFSET = IMAGE_TB_OFFSET + 64
# PoseTripleBuffer slots start @64 inside pose TB => absolute 320 for gimbal
POSE_SLOTS_OFFSET = POSE_GIMBAL_OFFSET + 64

SHM_MAGIC = 0x54414C05


def now_ns() -> int:
    return time.time_ns()


class TalosShm:
    def __init__(self, shm_dir: Path):
        self.meta_path = shm_dir / "talos_ipc_meta"
        self.img_path = shm_dir / "talos_ipc_image_pool"
        if not self.meta_path.exists() or not self.img_path.exists():
            raise FileNotFoundError(
                f"SHM missing under {shm_dir}. Start Daedalus with TALOS_SHM_DIR={shm_dir}"
            )
        self.meta_f = open(self.meta_path, "r+b", buffering=0)
        self.img_f = open(self.img_path, "r+b", buffering=0)
        self.meta = mmap.mmap(self.meta_f.fileno(), META_SIZE, access=mmap.ACCESS_WRITE)
        self.img = mmap.mmap(self.img_f.fileno(), IMAGE_POOL_SIZE, access=mmap.ACCESS_READ)
        magic = struct.unpack_from("<I", self.meta, 0)[0]
        if magic != SHM_MAGIC:
            raise RuntimeError(f"bad SHM magic {magic:#x}, expected {SHM_MAGIC:#x}")
        ver = struct.unpack_from("<I", self.meta, 4)[0]
        w, h = struct.unpack_from("<II", self.meta, 24)
        print(f"[talos] magic ok ver={ver} header={w}x{h} dir={shm_dir}")

    def close(self):
        self.meta.close()
        self.img.close()
        self.meta_f.close()
        self.img_f.close()

    def read_frame(self) -> tuple[np.ndarray | None, int, int]:
        state = self.meta[IMAGE_TB_OFFSET]
        idx = state & INDEX_MASK
        off = IMAGE_SLOTS_OFFSET + idx * 32
        seq, ts, w, h = struct.unpack_from("<QQII", self.meta, off)
        buf_id = self.meta[off + 24]
        if w != IMAGE_W or h != IMAGE_H or seq == 0:
            return None, seq, ts
        # clear NEW flag so relaxed producer can keep going (best-effort)
        self.meta[IMAGE_TB_OFFSET] = state & ~FLAG_NEW
        src = self.img[buf_id * IMAGE_SIZE : (buf_id + 1) * IMAGE_SIZE]
        rgb = np.frombuffer(src, dtype=np.uint8).reshape(IMAGE_H, IMAGE_W, 3).copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr, seq, ts

    def read_gimbal_quat_xyzw(self) -> np.ndarray | None:
        state = self.meta[POSE_GIMBAL_OFFSET]
        idx = state & INDEX_MASK
        off = POSE_SLOTS_OFFSET + idx * 64
        # PoseMeta: frame_seq u64, position f32[3], quaternion f32[4], timestamp u64
        frame_seq = struct.unpack_from("<Q", self.meta, off)[0]
        if frame_seq == 0:
            return None
        q = struct.unpack_from("<ffff", self.meta, off + 8 + 12)  # after pos
        return np.array(q, dtype=np.float64)  # x y z w

    def write_gimbal_cmd(self, yaw_deg: float, pitch_deg: float, distance_m: float, fire: bool):
        # GimbalTripleBuffer write slot 0 simply for relaxed consumer->producer
        tb = GIMBAL_CMD_OFFSET
        write_idx = self.meta[tb + 1] % 3
        slot = tb + 64 + write_idx * 32
        cmd = struct.pack(
            "<QfffB11x",
            now_ns(),
            float(yaw_deg),
            float(pitch_deg),
            float(distance_m),
            1 if fire else 0,
        )
        self.meta[slot : slot + 32] = cmd
        # publish: set state = write_idx | FLAG_NEW, rotate write_idx
        old = self.meta[tb]
        self.meta[tb] = write_idx | FLAG_NEW
        self.meta[tb + 1] = old & INDEX_MASK
        try:
            self.meta.flush()
        except Exception:
            pass


class HebeiDetector:
    """Minimal Hebei YOLO26-pose OpenVINO detector (1x30x18)."""

    def __init__(self, model_path: Path, conf: float = 0.25, device: str = "CPU"):
        core = Core()
        model = core.read_model(str(model_path))
        self.compiled = core.compile_model(model, device)
        self.input = self.compiled.input(0)
        self.output = self.compiled.output(0)
        shape = list(self.input.shape)
        # NCHW
        self.in_h, self.in_w = int(shape[2]), int(shape[3])
        self.conf = conf
        print(f"[det] {model_path.name} in={self.in_w}x{self.in_h} conf={conf} dev={device}")

    def detect(self, bgr: np.ndarray) -> list[dict]:
        h0, w0 = bgr.shape[:2]
        # letterbox
        scale = min(self.in_w / w0, self.in_h / h0)
        nw, nh = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.in_h, self.in_w, 3), 114, dtype=np.uint8)
        top = (self.in_h - nh) // 2
        left = (self.in_w - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        out = self.compiled([blob])[self.output]
        # out: 1x30x18 or 1x18xN
        arr = np.squeeze(out)
        if arr.ndim != 2:
            return []
        if arr.shape[0] == 18:
            arr = arr.T  # Nx18
        elif arr.shape[1] != 18:
            # 1x30x18 style already N=30? handle 30x18
            if arr.shape[1] == 18:
                pass
            else:
                return []
        dets = []
        for row in arr:
            conf = float(row[4])
            if conf < self.conf:
                continue
            # xyxy letterbox
            x1, y1, x2, y2 = row[0:4]
            cls_id = int(row[5]) if row[5] < 20 else int(np.argmax(row[5:6] + 0) or 0)
            # color scores at 6:10
            color_id = int(np.argmax(row[6:10])) if row.shape[0] >= 10 else 0
            kpts = row[10:18].reshape(4, 2) if row.shape[0] >= 18 else None
            # map letterbox -> original
            def unlb(x, y):
                return ((x - left) / scale, (y - top) / scale)

            box = [*unlb(x1, y1), *unlb(x2, y2)]
            pts = None
            if kpts is not None:
                pts = [unlb(px, py) for px, py in kpts]
            dets.append(
                {
                    "conf": conf,
                    "cls": cls_id,
                    "color": color_id,  # 0B 1R 2G 3P
                    "box": box,
                    "kpts": pts,
                }
            )
        return dets


def aim_from_box(box, img_w, img_h, fov_y_deg=45.0) -> tuple[float, float, float]:
    """Rough center-based aim angles (deg) relative to optical axis."""
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    # pixel offset from image center
    dx = (cx - img_w * 0.5) / (img_w * 0.5)
    dy = (cy - img_h * 0.5) / (img_h * 0.5)
    aspect = img_w / img_h
    fov_y = math.radians(fov_y_deg)
    fov_x = 2 * math.atan(math.tan(fov_y / 2) * aspect)
    yaw_deg = math.degrees(dx * (fov_x / 2))
    pitch_deg = math.degrees(-dy * (fov_y / 2))  # image y down
    # crude distance from box height
    bh = max(y2 - y1, 1.0)
    # small armor ~0.12m tall, f ~ fy
    fy = (img_h * 0.5) / math.tan(fov_y / 2)
    dist = (0.12 * fy) / bh
    dist = float(np.clip(dist, 0.5, 15.0))
    return yaw_deg, pitch_deg, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shm-dir", default=os.environ.get("TALOS_SHM_DIR", r"F:\talos_shm"))
    ap.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parents[1]
            / "assets/models/hebei_at_nn/praysky_coord_noe2e_0331_640x640.onnx"
        ),
    )
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--enemy-color", choices=["red", "blue", "any"], default="blue")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--send", action="store_true", default=True)
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--device", default="CPU")
    args = ap.parse_args()
    if args.no_send:
        args.send = False

    shm = TalosShm(Path(args.shm_dir))
    det = HebeiDetector(Path(args.model), conf=args.conf, device=args.device)

    color_map = {"blue": 0, "red": 1}
    want = color_map.get(args.enemy_color)

    last_seq = -1
    frames = 0
    t0 = time.time()
    print("[run] press Ctrl+C to stop; Daedalus must have F5 auto-aim ON to accept cmds")

    try:
        while True:
            img, seq, ts = shm.read_frame()
            if img is None or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            frames += 1

            dets = det.detect(img)
            # filter enemy color if requested
            if want is not None:
                dets = [d for d in dets if d["color"] == want]
            dets.sort(key=lambda d: d["conf"], reverse=True)

            yaw = pitch = dist = 0.0
            fire = False
            best = dets[0] if dets else None
            if best is not None:
                yaw, pitch, dist = aim_from_box(best["box"], IMAGE_W, IMAGE_H)
                fire = best["conf"] > 0.5
                if args.send:
                    shm.write_gimbal_cmd(yaw, pitch, dist, fire)

            if frames % 15 == 0:
                hz = frames / max(time.time() - t0, 1e-3)
                msg = f"seq={seq} fps~{hz:.1f} dets={len(dets)}"
                if best:
                    msg += f" conf={best['conf']:.2f} yaw={yaw:.1f} pitch={pitch:.1f} d={dist:.1f} fire={fire}"
                print(msg, flush=True)

            if args.show:
                vis = img.copy()
                for d in dets[:8]:
                    x1, y1, x2, y2 = map(int, d["box"])
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis,
                        f"{d['conf']:.2f}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
                cv2.imshow("talos_win_client", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("stop")
    finally:
        shm.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
