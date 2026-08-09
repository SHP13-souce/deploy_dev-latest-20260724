#!/usr/bin/env python3
import argparse
import json
import mmap
import os
import struct
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

MAGIC = b"HNUDBG1\0"
VERSION = 1
HEADER = struct.Struct("<8sIIQQQIIII")


class ShmReader:
    def __init__(self, name):
        self.name = name.lstrip("/").replace("/", "_")
        self.path = Path("/dev/shm") / self.name
        self.state_path = Path("/tmp") / f"{self.name}.json"
        self.mapping = None
        self.file = None

    def close(self):
        if self.mapping is not None:
            self.mapping.close()
        if self.file is not None:
            self.file.close()
        self.mapping = None
        self.file = None

    def connect(self):
        if self.mapping is not None:
            return True
        try:
            self.file = self.path.open("rb", buffering=0)
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            return True
        except (FileNotFoundError, OSError, ValueError):
            self.close()
            return False

    def frame(self):
        if not self.connect():
            return None
        try:
            for _ in range(5):
                first = HEADER.unpack_from(self.mapping, 0)
                magic, version, header_size, capacity, sequence, _, size, _, _, _ = first
                if magic != MAGIC or version != VERSION or header_size != HEADER.size:
                    self.close()
                    return None
                if sequence == 0 or sequence & 1 or size < 4 or size > capacity:
                    time.sleep(0.002)
                    continue
                jpeg = self.mapping[header_size:header_size + size]
                second = HEADER.unpack_from(self.mapping, 0)
                if sequence == second[4] and not second[4] & 1 and jpeg[:2] == b"\xff\xd8" and jpeg[-2:] == b"\xff\xd9":
                    return jpeg
            return None
        except (OSError, ValueError, struct.error):
            self.close()
            return None

    def state(self):
        try:
            with self.state_path.open("r", encoding="utf-8") as source:
                value = json.load(source)
            return value if isinstance(value, dict) else {"value": value}
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}

    def health(self):
        frame = self.frame()
        state = self.state()
        return {
            "ok": frame is not None,
            "shm": f"/{self.name}",
            "frame_available": frame is not None,
            "state_available": bool(state),
        }


PARAMETERS = {
    "conf_threshold": (0.01, 0.95), "binary_threshold": (0, 255),
    "bullet_speed": (14, 35), "yaw_offset": (-10, 10), "pitch_offset": (-10, 10),
    "comming_angle": (0, 90), "leaving_angle": (0, 90), "decision_speed": (0, 20),
    "high_speed_delay_time": (0, 0.5), "low_speed_delay_time": (0, 0.5),
}


def tuning_paths(shm_name):
    safe = shm_name.lstrip("/").replace("/", "_") or "hnu_vision_debug"
    return (Path("/tmp") / f"{safe}.tuning_request.yaml",
            Path("/tmp") / f"{safe}.tuning_state.json")


def create_app(shm_name):
    app = Flask(__name__)
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    readers = {
        "raw": ShmReader(shm_name),
        "annotated": ShmReader(f"{shm_name}_annotated"),
        "gray": ShmReader(f"{shm_name}_gray"),
        "binary": ShmReader(f"{shm_name}_binary"),
    }
    reader = readers["raw"]
    tuning_request_path, tuning_state_path = tuning_paths(shm_name)

    @app.get("/")
    def index():
        return render_template("index.html")

    def video_response(selected):
        def stream():
            last = None
            while True:
                frame = selected.frame()
                if frame is not None and frame != last:
                    last = frame
                    yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                else:
                    time.sleep(0.02 if frame is not None else 0.2)

        return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/video")
    def video():
        return video_response(reader)

    @app.get("/video/<view>")
    def video_view(view):
        if view not in readers:
            return jsonify({"error": "unknown view"}), 404
        return video_response(readers[view])

    @app.get("/data")
    def data():
        return jsonify(reader.state())

    @app.get("/health")
    def health():
        value = reader.health()
        return jsonify(value), 200 if value["ok"] else 503

    @app.get("/parameters")
    def parameters():
        try:
            with tuning_state_path.open("r", encoding="utf-8") as source:
                return jsonify(json.load(source))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return jsonify({"ok": False, "message": "vision tuning state unavailable"}), 503

    @app.post("/parameters")
    def update_parameters():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("parameters"), dict):
            return jsonify({"ok": False, "message": "parameters object required"}), 400
        values = body["parameters"]
        if set(values) != set(PARAMETERS):
            return jsonify({"ok": False, "message": "parameter whitelist mismatch"}), 400
        parsed = {}
        try:
            for key, (lower, upper) in PARAMETERS.items():
                value = float(values[key])
                if not lower <= value <= upper:
                    raise ValueError(f"{key} must be in [{lower},{upper}]")
                parsed[key] = int(value) if key == "binary_threshold" else value
        except (TypeError, ValueError) as error:
            return jsonify({"ok": False, "message": str(error)}), 400
        request_id = time.time_ns()
        lines = [f"id: {request_id}", f"save: {'true' if body.get('save') else 'false'}"]
        lines.extend(f"{key}: {value}" for key, value in parsed.items())
        temporary = tuning_request_path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(temporary, tuning_request_path)
        return jsonify({"ok": True, "id": request_id, "save": bool(body.get("save"))}), 202

    return app


def main():
    parser = argparse.ArgumentParser(description="HNU vision shared-memory debug server")
    parser.add_argument("--shm-name", default=os.environ.get("DEBUG_WEB_SHM", "/hnu_vision_debug"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app(args.shm_name).run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
