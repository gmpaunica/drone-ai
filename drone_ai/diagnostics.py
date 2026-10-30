"""Run with python -m drone_ai.diagnostics. Ground is the default; never auto-fly."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import queue
import select
import socket
import subprocess
import sys
import threading
import time
import uuid

from drone_ai.diagnostic_protocol import (
    AXES, AXIS_NAMES, ZERO, decode_native, flight_fault, native_packet,
    native_state, native_sticks, sdk_state,
)

HOST = "192.168.10.1"
PERIOD = 0.05
PULSE_SECONDS = 0.20
HOVER_ACCEPTANCE_SECONDS = 15.0


class Evidence:
    """Disk IO never runs on the control thread. Overflow/errors fail flight closed."""

    def __init__(self, root: Path, protocol: str, mode: str):
        self.path = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                            + f"-{protocol}-{mode}-{uuid.uuid4().hex[:8]}")
        self.path.mkdir(parents=True, exist_ok=False)
        self.error: str | None = None
        self.items: queue.Queue = queue.Queue(maxsize=50000)
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def emit(self, kind: str, **data):
        try:
            self.items.put_nowait({"t": time.monotonic(), "kind": kind, **data})
        except queue.Full:
            self.error = "evidence queue overflow"

    def _write(self):
        try:
            with (self.path / "events.jsonl").open("x", encoding="utf-8") as out:
                sequence = 0
                while True:
                    item = self.items.get()
                    if item is None:
                        break
                    out.write(json.dumps({"sequence": sequence, **item}, allow_nan=False) + "\n")
                    sequence += 1
                    if sequence % 20 == 0:
                        out.flush()
        except Exception as exc:
            self.error = f"evidence writer: {exc}"

    def close(self):
        if self.thread.is_alive():
            try:
                self.items.put(None, timeout=2)
            except queue.Full:
                self.error = "evidence shutdown queue overflow"
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                self.error = "evidence writer did not stop"


def environment() -> dict:
    result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
    for name in ("tellopy", "djitellopy", "av", "opencv-python", "ultralytics"):
        try:
            result["packages"][name] = version(name)
        except Exception:
            result["packages"][name] = "missing"
    result["source_sha256"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ("diagnostics.py", "diagnostic_protocol.py")
    }
    # UDP connect chooses a route locally; it does not send a datagram.
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect((HOST, 8889))
            result["route_local_ip"] = route.getsockname()[0]
    except OSError as exc:
        result["route_error"] = str(exc)
    if sys.platform == "win32":
        for label, command in (("wifi", ["netsh", "wlan", "show", "interfaces"]),
                               ("routes", ["route", "print", HOST])):
            try:
                proc = subprocess.run(command, capture_output=True, text=True,
                                      errors="replace", timeout=10)
                result[label] = proc.stdout + proc.stderr
            except Exception as exc:
                result[label] = str(exc)
    return result


class Probe:
    def __init__(self, protocol: str, evidence: Evidence, *, flight: bool = False,
                 video: bool = True, bind_ip: str = "0.0.0.0"):
        self.protocol, self.evidence = protocol, evidence
        self.flight_enabled, self.video_enabled, self.bind_ip = flight, video, bind_ip
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.send_lock = threading.Lock()
        self.sockets: dict[socket.socket, str] = {}
        self.threads: list[threading.Thread] = []
        self.responses: queue.Queue = queue.Queue()
        self.video_queue: queue.Queue = queue.Queue(maxsize=512)
        self.connected = False
        self.state: dict | None = None
        self.state_at = 0.0
        self.fault: str | None = None
        self.axes = ZERO
        self.motion_until = 0.0
        self.takeoff_at: float | None = None
        self.airborne_at: float | None = None
        self.landing_at: float | None = None
        self.ground_since: float | None = None
        self.unexpected_ground_since: float | None = None
        self.landed_evidence = False
        self.sequence = 0
        self.next_sdk_heartbeat_at: float | None = None
        self.pending_sdk_heartbeat_at: float | None = None
        self.video_packets = self.video_bytes = self.decoded_frames = 0
        self.video_drops = self.decode_errors = 0
        self.last_video_at = self.last_frame_at = 0.0
        self.latest_frame = None
        self.resolution = None
        self.codec = None

    def _bind(self, port: int, channel: str):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.bind_ip, port))
            sock.setblocking(False)
        except Exception:
            sock.close()
            raise
        self.sockets[sock] = channel
        self.evidence.emit("socket.bound", channel=channel, address=sock.getsockname())
        return sock

    def _thread(self, target):
        def guarded():
            try:
                target()
            except Exception as exc:
                self.fault = f"{target.__name__}: {exc}"
                self.evidence.emit("worker.error", error=self.fault)
        thread = threading.Thread(target=guarded, daemon=True)
        self.threads.append(thread)
        thread.start()

    def open(self):
        if self.protocol == "native":
            if version("tellopy") != "0.6.0":
                raise RuntimeError("Install requirements-diagnostics.txt (requires tellopy 0.6.0)")
        if self.video_enabled:
            import av
            self.codec = av.CodecContext.create("h264", "r")
        self.control = self._bind(9000 if self.protocol == "native" else 8889, "control")
        if self.protocol == "sdk":
            self._bind(8890, "state")
        if self.video_enabled:
            self._bind(6038 if self.protocol == "native" else 11111, "video")
            self._thread(self._video_worker)
        self._thread(self._receive)
        for _ in range(3):
            if self.protocol == "native":
                # Same little-endian video port as TelloPy (0x1796 == 6038).
                self.send(b"conn_req:" + (6038).to_bytes(2, "little"), "connect")
                deadline = time.monotonic() + 2
                while not self.connected and time.monotonic() < deadline:
                    time.sleep(0.02)
            else:
                self.connected = self.request("command") == "ok"
            if self.connected:
                break
        if not self.connected:
            raise RuntimeError("No connection acknowledgement. Check Wi-Fi, firewall and other controllers.")
        if self.protocol == "native":
            self.send(native_packet(0x46, b"\x00", kind=0x50, timestamp=datetime.now()), "time")
        self._thread(self._pump)
        if self.protocol == "sdk":
            # Read-only capability probes. Unsupported query is recorded, not fatal.
            for query in ("sdk?", "battery?", "wifi?"):
                self.request(query)
            if self.video_enabled:
                self.request("streamon")
        deadline = time.monotonic() + 8
        while self.state is None and time.monotonic() < deadline and not self.fault:
            time.sleep(0.05)
        state, age = self.snapshot()
        if state is None or age > 1:
            raise RuntimeError("Connected, but no fresh valid flight telemetry arrived.")

    def send(self, data: bytes, label: str):
        with self.send_lock:
            self.control.sendto(data, (HOST, 8889))
            self.evidence.emit("tx", label=label, hex=data.hex())

    def native_command(self, command: int, payload: bytes = b"", label: str = "native"):
        with self.send_lock:
            self.sequence = (self.sequence + 1) & 0xffff
            data = native_packet(command, payload, sequence=self.sequence)
            self.control.sendto(data, (HOST, 8889))
            self.evidence.emit("tx", label=label, hex=data.hex())

    def request(self, command: str) -> str | None:
        # Only used on the ground. Preserve every reply, including late/orphan replies.
        while not self.responses.empty():
            self.responses.get_nowait()
        started = time.monotonic()
        self.send(command.encode("ascii"), command)
        try:
            reply = self.responses.get(timeout=2)
        except queue.Empty:
            reply = None
        self.evidence.emit("sdk.query", command=command, response=reply,
                           elapsed_s=time.monotonic() - started,
                           correlation="best_effort_text_protocol_has_no_sequence_id")
        time.sleep(0.15)
        return reply

    def snapshot(self):
        with self.lock:
            return self.state, time.monotonic() - self.state_at

    def _accept_state(self, state: dict):
        now = time.monotonic()
        with self.lock:
            self.state, self.state_at = state, now
            if self.takeoff_at is not None and state["height_m"] >= 0.3:
                if self.airborne_at is None:
                    self.airborne_at = now
                    self.evidence.emit("takeoff.observed", state=state)
            if self.landing_at is not None:
                grounded = state["height_m"] <= 0.1
                if self.protocol == "native":
                    grounded = grounded and state["ground_flag"] and not state["flying_flag"]
                if grounded:
                    if self.ground_since is None:
                        self.ground_since = now
                    elif now - self.ground_since >= 2:
                        self.landed_evidence = True
                else:
                    self.ground_since = None
        self.evidence.emit("telemetry", state=state)

    def _receive(self):
        while not self.stop.is_set():
            ready, _, _ = select.select(list(self.sockets), [], [], 0.1)
            for sock in ready:
                data, address = sock.recvfrom(65535)
                channel = self.sockets[sock]
                if address[0] != HOST:
                    self.evidence.emit("rx.foreign", channel=channel, address=address)
                    continue
                if channel == "video":
                    now = time.monotonic()
                    gap = now - self.last_video_at if self.last_video_at else None
                    self.last_video_at = now
                    self.video_packets += 1
                    self.video_bytes += len(data)
                    self.evidence.emit("video.packet", size=len(data), gap_s=gap,
                                       header=data[:2].hex())
                    payload = data[2:] if self.protocol == "native" else data
                    try:
                        self.video_queue.put_nowait(payload)
                    except queue.Full:
                        self.video_drops += 1
                    continue
                self.evidence.emit("rx", channel=channel, hex=data.hex())
                try:
                    if channel == "state":
                        self._accept_state(sdk_state(data))
                    elif self.protocol == "sdk":
                        reply = data.decode("ascii", errors="replace").strip()
                        if self.pending_sdk_heartbeat_at is not None:
                            try:
                                battery_reply = int(reply)
                            except ValueError:
                                pass
                            else:
                                if 0 <= battery_reply <= 100:
                                    self.evidence.emit(
                                        "sdk.heartbeat_response", value=battery_reply,
                                        elapsed_s=time.monotonic() - self.pending_sdk_heartbeat_at,
                                    )
                                    self.pending_sdk_heartbeat_at = None
                        self.responses.put(reply)
                    elif len(data) >= 11 and data.startswith(b"conn_ack:"):
                        self.connected = True
                        self.evidence.emit("native.connected", hex=data.hex())
                    else:
                        command, sequence, payload = decode_native(data)
                        self.evidence.emit("native.packet", command=command, sequence=sequence,
                                           payload_hex=payload.hex())
                        if command == 0x56:
                            self._accept_state(native_state(payload))
                except (ValueError, KeyError, UnicodeError) as exc:
                    self.evidence.emit("rx.invalid", error=str(exc))

    def _video_worker(self):
        import av

        with (self.evidence.path / "video.h264").open("xb") as stream:
            while not self.stop.is_set() or not self.video_queue.empty():
                try:
                    data = self.video_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                stream.write(data)
                try:
                    for packet in self.codec.parse(data):
                        for frame in self.codec.decode(packet):
                            now = time.monotonic()
                            gap = now - self.last_frame_at if self.last_frame_at else None
                            self.last_frame_at = now
                            self.decoded_frames += 1
                            self.resolution = (frame.width, frame.height)
                            self.latest_frame = frame.to_ndarray(format="bgr24")
                            self.evidence.emit("video.frame", width=frame.width,
                                               height=frame.height, gap_s=gap)
                except av.FFmpegError as exc:
                    self.decode_errors += 1
                    self.evidence.emit("video.decode_error", error=str(exc))

    def start_detector(self, detector):
        """Exercise the project's actual YOLO adapter, with no authority to move."""
        if self.flight_enabled:
            raise RuntimeError("detector workload diagnostics are ground-only")

        def inference():
            seen = 0.0
            while not self.stop.is_set():
                if self.latest_frame is not None and self.last_frame_at > seen:
                    frame, seen = self.latest_frame, self.last_frame_at
                    started = time.monotonic()
                    detections = detector.detect(frame)
                    self.evidence.emit("detector.inference", elapsed_s=time.monotonic() - started,
                                       input_age_s=started - seen,
                                       detections=[asdict(d) for d in detections])
                self.stop.wait(0.5)
        self._thread(inference)

    def _pump(self):
        next_tick = time.monotonic()
        last_tick = next_tick
        next_video = next_land = 0.0
        while not self.stop.is_set():
            now = time.monotonic()
            with self.lock:
                if self.takeoff_at is not None and not self.landed_evidence:
                    state, age = self.snapshot()
                    reason = self.fault or self.evidence.error or flight_fault(
                        state, age, now - self.takeoff_at, self.airborne_at is not None)
                    if (self.airborne_at is not None and self.landing_at is None
                            and state is not None and age <= 1):
                        grounded = state["height_m"] <= 0.1
                        if self.protocol == "native":
                            grounded = grounded and bool(state["ground_flag"])
                        if grounded:
                            if self.unexpected_ground_since is None:
                                self.unexpected_ground_since = now
                            elif now - self.unexpected_ground_since >= 0.3:
                                reason = "unexpected_ground_contact"
                        else:
                            self.unexpected_ground_since = None
                    if self.landing_at is None and now - last_tick > 0.25:
                        reason = "control_scheduler_stalled"
                    if reason and self.landing_at is None:
                        self.fault = reason
                        self.request_land(reason)
                if now >= self.motion_until or self.landing_at is not None:
                    if self.axes != ZERO:
                        self.evidence.emit("motion.zero", reason="deadline_or_landing")
                    self.axes = ZERO
                axes = self.axes
                try:
                    if self.protocol == "native":
                        self.send(native_sticks(axes), "sticks")
                    else:
                        self.send(("rc " + " ".join(map(str, axes))).encode("ascii"), "sticks")
                    self.evidence.emit("control.tick", axes=axes, gap_s=now - last_tick)
                    if (self.protocol == "sdk" and self.takeoff_at is not None
                            and self.landing_at is None and self.next_sdk_heartbeat_at is not None
                            and now >= self.next_sdk_heartbeat_at):
                        # This basic Tello accepts battery? but rejects the newer
                        # literal keepalive command. Two hardware runs showed that
                        # zero RC packets did not reset its inactivity auto-land.
                        self.send(b"battery?", "sdk_heartbeat_battery_query")
                        self.evidence.emit("sdk.heartbeat", command="battery?",
                                           response_correlation="raw_rx_only")
                        self.pending_sdk_heartbeat_at = now
                        self.next_sdk_heartbeat_at = now + 5.0
                    if self.landing_at is not None and not self.landed_evidence and now >= next_land:
                        self._send_land()
                        next_land = now + 1
                    if self.protocol == "native" and self.video_enabled and now >= next_video:
                        self.send(native_packet(0x25, kind=0x60), "video_start")
                        next_video = now + 2
                except OSError as exc:
                    self.fault = f"send_failed: {exc}"
                    self.evidence.emit("tx.error", error=self.fault)
                    self.axes = ZERO
                    if self.takeoff_at is not None:
                        self.request_land(self.fault)
            last_tick = now
            next_tick += PERIOD
            if next_tick < time.monotonic():
                next_tick = time.monotonic() + PERIOD
            self.stop.wait(max(0, next_tick - time.monotonic()))

    def takeoff(self):
        with self.lock:
            if not self.flight_enabled:
                raise RuntimeError("ground mode cannot take off")
            if self.takeoff_at is not None:
                raise RuntimeError("one takeoff per diagnostic session; restart for another")
            state, age = self.snapshot()
            if self.fault or self.evidence.error:
                raise RuntimeError(self.fault or self.evidence.error)
            if state is None or age > 1 or state["battery"] < 30 or state["height_m"] > 0.1:
                raise RuntimeError("takeoff requires fresh grounded telemetry and battery >=30%")
            reason = flight_fault(state, age, 0, False)
            if reason:
                raise RuntimeError(reason)
            if self.protocol == "native" and state["flying_flag"]:
                raise RuntimeError("native flight flag indicates already flying")
            self.takeoff_at = time.monotonic()  # Set BEFORE send; uncertain sends still require landing.
            if self.protocol == "sdk":
                self.next_sdk_heartbeat_at = self.takeoff_at + 4.0
            self.evidence.emit("takeoff.requested")
            if self.protocol == "native":
                # Deliberately do not call TelloPy.takeoff(), which changes altitude limit.
                self.native_command(0x54, label="takeoff")
            else:
                self.send(b"takeoff", "takeoff")

    def pulse(self, key: str):
        with self.lock:
            state, age = self.snapshot()
            now = time.monotonic()
            if not self.flight_enabled or self.takeoff_at is None or self.airborne_at is None:
                raise RuntimeError("takeoff must be observed before movement")
            reason = flight_fault(state, age, now - self.takeoff_at, True)
            if self.fault or reason or self.landing_at is not None or now - self.airborne_at < 3:
                raise RuntimeError(self.fault or reason or "wait for three seconds of hover / landing active")
            if self.evidence.error:
                raise RuntimeError(self.evidence.error)
            if state["height_m"] < 0.3 or (key == "f" and state["height_m"] < 0.5):
                raise RuntimeError("too close to the ground for this pulse")
            if key == "r" and state["height_m"] >= 1.2:
                raise RuntimeError("up pulse blocked near ceiling")
            self.axes = AXES[key]
            self.motion_until = now + PULSE_SECONDS
            self.evidence.emit("motion.pulse", axis=AXIS_NAMES[key], key=key,
                               duration_s=PULSE_SECONDS, sticks=self.axes, before=state)

    def hover(self):
        with self.lock:
            self.axes, self.motion_until = ZERO, 0
            self.evidence.emit("motion.zero", reason="operator")

    def request_land(self, reason: str):
        with self.lock:
            self.axes, self.motion_until = ZERO, 0
            if self.takeoff_at is not None and self.landing_at is None:
                self.landing_at = time.monotonic()
                self.evidence.emit("landing.requested", reason=reason)

    def _send_land(self):
        if self.protocol == "native":
            self.native_command(0x55, b"\x00", label="land")
        else:
            self.send(b"land", "land")

    def close(self):
        if self.takeoff_at is not None and not self.landed_evidence:
            self.request_land("shutdown")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not self.landed_evidence:
                # Independent fallback even if the pump itself failed.
                try:
                    self._send_land()
                except Exception as exc:
                    self.evidence.emit("landing.send_error", error=str(exc))
                time.sleep(0.5)
            if not self.landed_evidence:
                print("LANDING NOT CONFIRMED. Check the aircraft immediately; link may be unavailable.")
        if self.takeoff_at is not None:
            self.evidence.emit("landing.result", telemetry_grounded=self.landed_evidence,
                               physical_confirmation_required=True)
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=3)
        if self.protocol == "sdk" and self.connected and self.video_enabled:
            try:
                self.send(b"streamoff", "streamoff")
            except OSError:
                pass
        for sock in self.sockets:
            sock.close()
        if any(thread.is_alive() for thread in self.threads):
            self.fault = self.fault or "worker shutdown incomplete"


def keyboard():
    import msvcrt

    if msvcrt.kbhit():
        key = msvcrt.getwch()
        if key in ("\x00", "\xe0"):
            msvcrt.getwch()
            return None
        return key.lower()
    return None


def run_console(probe: Probe, seconds: int, preview: bool, *, hover_only: bool,
                allowed_axis: str | None):
    started = time.monotonic()
    probe.evidence.emit("observation.started", requested_seconds=seconds)
    last_status = last_input = 0.0
    pending: str | None = None
    pending_at = 0.0
    hover_prompted = False
    status_message = "Press T when the aircraft and area are ready"
    axis_key = next((key for key, name in AXIS_NAMES.items() if name == allowed_axis), None)
    if hover_only:
        print(f"HOVER-ONLY: T takes off; the tool lands after {HOVER_ACCEPTANCE_SECONDS:.0f}s stable hover.")
    else:
        print(f"SINGLE-AXIS: T takes off; after 10s stable hover press {axis_key.upper()} once for {allowed_axis}.")
    print("H or Space: neutral | L/Q/Esc: land and finish | Y/N/U: correct/wrong/unclear last pulse")
    print("A movement press is 0.20s at 10% stick. Confirm it with Y/N/U; then the tool lands.")
    print("Watch the aircraft. Native yaw is not decoded; visual confirmation is required.")
    try:
        while time.monotonic() - started < seconds:
            now = time.monotonic()
            state, age = probe.snapshot()
            if now - last_status >= 1:
                h = state["height_m"] if state else None
                b = state["battery"] if state else None
                print(f"height={h}m battery={b}% state_age={age:.2f}s "
                      f"frames={probe.decoded_frames} drops={probe.video_drops}")
                last_status = now
                probe.evidence.emit("health", telemetry_age_s=age,
                                    video_age_s=now - probe.last_video_at if probe.last_video_at else None,
                                    frame_age_s=now - probe.last_frame_at if probe.last_frame_at else None)
            if probe.fault or probe.evidence.error:
                raise RuntimeError(probe.fault or probe.evidence.error)
            if (hover_only and probe.airborne_at is not None
                    and now - probe.airborne_at >= HOVER_ACCEPTANCE_SECONDS
                    and not hover_prompted):
                pending, pending_at = "hover_stability", now
                hover_prompted = True
                status_message = "Hover complete: Y stable / N unstable / U unclear"
                print(status_message)
            if probe.landing_at is not None:
                break
            key = keyboard() if sys.platform == "win32" else None
            if preview:
                import cv2
                if probe.latest_frame is not None:
                    display = probe.latest_frame.copy()
                    cv2.putText(display, status_message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 255, 255), 2)
                    cv2.imshow("Tello diagnostic - L land / Q quit", display)
                    code = cv2.waitKey(1) & 0xff
                    if code != 255:
                        key = chr(code).lower()
                    if cv2.getWindowProperty("Tello diagnostic - L land / Q quit", cv2.WND_PROP_VISIBLE) < 1:
                        key = "q"
            if key is None:
                time.sleep(0.02)
                continue
            released = now - last_input >= 0.8
            last_input = now
            if key in ("q", "l", "\x1b"):
                probe.request_land("operator")
                break
            if key in ("h", " "):
                probe.hover()
                continue
            if key in ("y", "n", "u") and pending is not None and now - pending_at >= 1:
                result = {"y": "correct", "n": "wrong", "u": "unclear"}[key]
                if pending == "hover_stability":
                    probe.evidence.emit("hover.observation", result=result, after=state,
                                        source="operator_visual_confirmation")
                else:
                    probe.evidence.emit("motion.observation", axis=pending, result=result,
                                        after=state, source="operator_visual_confirmation")
                pending = None
                probe.request_land({"y": "observation_correct", "n": "operator_observed_wrong_motion",
                                    "u": "observation_unclear"}[key])
                break
                continue
            if not released:
                continue
            try:
                if key == "t":
                    probe.takeoff()
                    status_message = "Takeoff sent; wait for observed altitude"
                elif key in AXES:
                    if hover_only:
                        print("BLOCKED: this is a hover-only flight; press L to land if needed.")
                        probe.evidence.emit("operator.blocked", key=key, reason="hover_only")
                        continue
                    if key != axis_key:
                        print(f"BLOCKED: this run permits only {axis_key.upper()} ({allowed_axis}).")
                        probe.evidence.emit("operator.blocked", key=key, reason="wrong_axis_for_run")
                        continue
                    if pending:
                        print("Confirm the previous pulse with Y/N/U first.")
                    else:
                        if probe.airborne_at is None or now - probe.airborne_at < 10:
                            raise RuntimeError("wait for ten seconds of stable hover before the test pulse")
                        probe.pulse(key)
                        pending, pending_at = AXIS_NAMES[key], now
                        status_message = f"Testing {pending}; after neutral: Y correct / N wrong / U unclear"
                        print(status_message)
            except RuntimeError as exc:
                print(f"BLOCKED: {exc}")
                probe.evidence.emit("operator.blocked", key=key, reason=str(exc))
    finally:
        probe.evidence.emit("observation.finished", elapsed_s=time.monotonic() - started)
        if preview:
            import cv2
            cv2.destroyAllWindows()


def stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {"count": len(values), "median": ordered[len(values) // 2],
            "p95": ordered[min(len(values) - 1, int(len(values) * 0.95))], "max": max(values)}


def summarize(path: Path) -> dict:
    events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = json.loads((path / "session.json").read_text(encoding="utf-8"))
    telemetry = [e for e in events if e["kind"] == "telemetry"]
    ticks = [e for e in events if e["kind"] == "control.tick"]
    frames = [e for e in events if e["kind"] == "video.frame"]
    takeoff_requested = next((e["t"] for e in events if e["kind"] == "takeoff.requested"), None)
    landing_requested = next((e["t"] for e in events if e["kind"] == "landing.requested"), None)
    if meta["mode"] == "flight" and takeoff_requested is not None:
        evaluation_end = landing_requested if landing_requested is not None else float("inf")
        telemetry_eval = [e for e in telemetry if takeoff_requested <= e["t"] <= evaluation_end]
        health_eval = [e for e in events if e["kind"] == "health"
                       and takeoff_requested <= e["t"] <= evaluation_end]
    else:
        telemetry_eval = telemetry
        health_eval = [e for e in events if e["kind"] == "health"]
    gaps = [b["t"] - a["t"] for a, b in zip(telemetry_eval, telemetry_eval[1:])]
    tick_gaps = [e["gap_s"] for e in ticks[1:]]
    observations = {name: "untested" for name in AXIS_NAMES.values()}
    trials = []
    for pulse in (e for e in events if e["kind"] == "motion.pulse"):
        following = [e for e in ticks if e["t"] >= pulse["t"]]
        first_zero = next((e for e in following if not any(e["axes"])), None)
        end = first_zero["t"] if first_zero else pulse["t"] + 1
        active = [e for e in following if e["t"] <= end and any(e["axes"])]
        after = next((e["state"] for e in telemetry if e["t"] >= end + 0.5), None)
        before = pulse.get("before")
        yaw_delta = None
        if before and after and before["yaw_degrees"] is not None and after["yaw_degrees"] is not None:
            yaw_delta = (after["yaw_degrees"] - before["yaw_degrees"] + 180) % 360 - 180
        trials.append({"axis": pulse["axis"], "active_packets": len(active),
                       "neutral_after_request_s": end - pulse["t"] if first_zero else None,
                       "height_change_m": after["height_m"] - before["height_m"] if after and before else None,
                       "yaw_change_degrees": yaw_delta,
                       "interpretation": "telemetry change is evidence, not measured displacement or causation"})
    for event in events:
        if event["kind"] == "motion.observation":
            name, outcome = event["axis"], event["result"]
            # A later success does not erase a previously observed failure.
            if observations[name] != "wrong":
                observations[name] = outcome
    reasons = []
    if meta.get("error"):
        reasons.append(meta["error"])
    observed = [e["elapsed_s"] for e in events if e["kind"] == "observation.finished"]
    if meta["mode"] == "ground" and (not observed or max(observed) < meta.get("seconds", 30) - 1):
        reasons.append("ground observation did not complete its requested duration")
    if len(telemetry_eval) < 20 or (gaps and max(gaps) > 1):
        reasons.append("insufficient or interrupted telemetry")
    if len(ticks) < 20 or (tick_gaps and max(tick_gaps) > 0.25):
        reasons.append("insufficient or stalled control transmission")
    health = [e for e in events if e["kind"] == "health"]
    if any(e["telemetry_age_s"] > 1 for e in health_eval):
        reasons.append("stale telemetry during observation")
    if meta["video"] and (len(frames) < 20 or any(
        e.get("frame_age_s") is None or e["frame_age_s"] > 2 for e in health[5:]
    )):
        reasons.append("insufficient or stalled decoded video")
    if meta["video"] and frames and any(e["width"] < 640 or e["height"] < 480 for e in frames):
        reasons.append("decoded video resolution below project minimum 640x480")
    if meta.get("video_drops", 0):
        reasons.append("local video queue dropped packets")
    if any(t["active_packets"] == 0 or t["neutral_after_request_s"] is None
           or t["neutral_after_request_s"] > 0.5 for t in trials):
        reasons.append("motion transmission or neutral deadline not verified")
    inferences = [e for e in events if e["kind"] == "detector.inference"]
    inference_stats = stats([e["elapsed_s"] for e in inferences])
    if meta.get("model") and (len(inferences) < 10 or inference_stats["p95"] > 0.5):
        reasons.append("detector insufficient samples or p95 inference exceeds 0.5s project budget")
    if meta.get("model") and not any(e["detections"] for e in inferences):
        reasons.append("person detection not demonstrated; repeat with a person visible during ground test")
    took_off = any(e["kind"] == "takeoff.observed" for e in events)
    grounded = any(e["kind"] == "landing.result" and e["telemetry_grounded"] for e in events)
    if meta["mode"] == "flight":
        if not took_off:
            reasons.append("takeoff not observed")
        if not grounded:
            reasons.append("landing not confirmed by fresh telemetry")
        flight_test = meta.get("flight_test")
        if flight_test == "hover_only":
            hover_results = [e["result"] for e in events if e["kind"] == "hover.observation"]
            if hover_results != ["correct"]:
                reasons.append("15-second hover was not visually confirmed stable")
        elif flight_test in observations:
            if observations[flight_test] != "correct":
                reasons.append(f"{flight_test} was not visually confirmed correct")
        elif any(v != "correct" for v in observations.values()):
            reasons.append("not all eight directions visually confirmed correct")
        if meta["protocol"] == "sdk" and took_off:
            heartbeats = [e for e in events if e["kind"] == "sdk.heartbeat"]
            heartbeat_responses = [e for e in events if e["kind"] == "sdk.heartbeat_response"]
            if not heartbeats or len(heartbeat_responses) < len(heartbeats):
                reasons.append("supported SDK heartbeat query was not acknowledged")
    return {"session": str(path), "protocol": meta["protocol"], "mode": meta["mode"],
            "diagnostic_pass": not reasons, "issues": reasons,
            "telemetry_packets": len(telemetry), "telemetry_gap_s": stats(gaps),
            "control_gap_s": stats(tick_gaps), "decoded_frames": len(frames),
            "decode_errors": meta.get("decode_errors", 0),
            "max_height_m": max((e["state"]["height_m"] for e in telemetry), default=None),
            "takeoff_observed": took_off, "landing_telemetry_grounded": grounded,
            "flight_test": meta.get("flight_test"),
            "hover_observations": [e for e in events if e["kind"] == "hover.observation"],
            "axis_observations": observations,
            "motion_trials": trials, "detector_inference_s": inference_stats,
            "frames_with_person": sum(bool(e["detections"]) for e in inferences),
            "video_tested": meta["video"], "detector_tested": bool(meta.get("model")),
            "observation_duration_s": max(observed, default=0),
            "native_status_flags_observed": {
                key: sorted({e["state"]["raw"][key] for e in telemetry if key in e["state"]["raw"]})
                for key in ("fly_mode", "down_visual_state", "imu_state", "battery_state",
                            "power_state", "wind_state", "temperature_height", "camera_state")
            } if meta["protocol"] == "native" else None,
            "sdk_queries": [e for e in events if e["kind"] == "sdk.query"],
            "sdk_heartbeats": [e for e in events if e["kind"] == "sdk.heartbeat"],
            "sdk_heartbeat_responses": [e for e in events if e["kind"] == "sdk.heartbeat_response"],
            "invalid_packets": sum(e["kind"] == "rx.invalid" for e in events),
            "limitations": ["Not an autonomous-flight certification or measured displacement calibration.",
                            "Native yaw/position are not decoded; raw native packets are retained.",
                            "Packet send success does not prove reception or motion.",
                            "Link-loss and process-crash landing must not be induced in flight.",
                            "Verify physical landing; SDK has no authoritative flying flag here."]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("environment", help="print read-only host/dependency/network information")
    run = sub.add_parser("run", help="ground diagnostics by default; flight requires --flight and a T key")
    run.add_argument("--protocol", choices=("sdk", "native"), required=True)
    run.add_argument("--flight", action="store_true")
    flight_kind = run.add_mutually_exclusive_group()
    flight_kind.add_argument("--hover-only", action="store_true",
                             help="flight acceptance: take off, hold neutral 15s, land")
    flight_kind.add_argument("--axis", choices=tuple(AXIS_NAMES.values()),
                             help="flight acceptance: permit one named direction, then land")
    run.add_argument("--seconds", type=int, default=30)
    run.add_argument("--no-video", action="store_true", help="isolate control/telemetry; not a full video pass")
    run.add_argument("--preview", action="store_true")
    run.add_argument("--bind-ip", default="0.0.0.0")
    run.add_argument("--output", type=Path, default=Path("recordings/diagnostics"))
    run.add_argument("--note", default="", help="firmware/model, lighting, battery, app baseline and observations")
    run.add_argument("--model", type=Path, help="ground-only workload test using a local YOLO model file")
    report = sub.add_parser("report", help="offline summary of all completed sessions under a directory")
    report.add_argument("path", type=Path, nargs="?", default=Path("recordings/diagnostics"))
    args = parser.parse_args(argv)
    if args.command == "environment":
        print(json.dumps(environment(), indent=2))
        return 0
    if args.command == "report":
        paths = [args.path] if (args.path / "session.json").exists() else sorted(
            p.parent for p in args.path.glob("*/session.json"))
        if not paths:
            print("No diagnostic sessions found.")
            return 1
        results = []
        for path in paths:
            try:
                results.append(summarize(path))
            except Exception as exc:
                results.append({"session": str(path), "diagnostic_pass": False, "error": str(exc)})
        print(json.dumps(results, indent=2))
        return 0 if all(r["diagnostic_pass"] for r in results) else 1
    if not 10 <= args.seconds <= 180:
        parser.error("--seconds must be between 10 and 180")
    if args.flight and sys.platform != "win32":
        parser.error("supervised flight console currently requires Windows")
    if args.flight and not (args.hover_only or args.axis):
        parser.error("--flight requires either --hover-only or one --axis")
    if not args.flight and (args.hover_only or args.axis):
        parser.error("--hover-only and --axis require --flight")
    if args.preview and args.no_video:
        parser.error("--preview requires video")
    if args.model and (args.flight or args.no_video or not args.model.is_file()):
        parser.error("--model requires ground mode, video, and an existing local model file")
    evidence = Evidence(args.output, args.protocol, "flight" if args.flight else "ground")
    meta = {"schema_version": 1, "protocol": args.protocol,
            "mode": "flight" if args.flight else "ground", "video": not args.no_video,
            "flight_test": "hover_only" if args.hover_only else args.axis,
            "seconds": args.seconds, "note": args.note, "environment": environment(),
            "model": str(args.model) if args.model else None,
            "started_at": datetime.now(timezone.utc).isoformat(), "error": None}
    session_file = evidence.path / "session.json"
    session_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Evidence: {evidence.path.resolve()}")
    probe = Probe(args.protocol, evidence, flight=args.flight,
                  video=not args.no_video, bind_ip=args.bind_ip)
    try:
        detector = None
        if args.model:
            import numpy as np
            from drone_ai.vision.detector import YOLODetector
            meta["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()
            detector = YOLODetector(str(args.model.resolve()), target_classes=("person",))
            detector.validate_target_classes()
            detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
            evidence.emit("detector.warmed")
        probe.open()
        if detector is not None:
            probe.start_detector(detector)
        if args.flight:
            print("FLIGHT ENABLED. Props/guards fitted, clear lit area. Press T only when ready.")
        else:
            print("GROUND MODE. No takeoff, movement, landing or motor-cutoff commands are enabled.")
        run_console(probe, args.seconds, args.preview, hover_only=args.hover_only,
                    allowed_axis=args.axis)
    except KeyboardInterrupt:
        evidence.emit("operator.interrupt")
    except Exception as exc:
        meta["error"] = str(exc)
        print(f"DIAGNOSTIC ERROR: {exc}")
    finally:
        try:
            probe.close()
        except Exception as exc:
            meta["error"] = f"{meta['error'] or ''}; cleanup: {exc}"
        meta["error"] = meta["error"] or probe.fault or evidence.error
        meta.update(video_packets=probe.video_packets, video_bytes=probe.video_bytes,
                    video_drops=probe.video_drops, decode_errors=probe.decode_errors,
                    resolution=probe.resolution, finished_at=datetime.now(timezone.utc).isoformat())
        evidence.close()
        meta["error"] = meta["error"] or evidence.error
        session_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    result = summarize(evidence.path)
    (evidence.path / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["diagnostic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
