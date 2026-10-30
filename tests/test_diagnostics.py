from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import queue
import struct
import threading
import time

import pytest

from drone_ai import diagnostics as diag
from drone_ai.diagnostic_protocol import (
    AXES, ZERO, decode_native, flight_fault, native_packet, native_state,
    native_sticks, sdk_state, validate_axes,
)


class MemoryEvidence:
    error = None

    def __init__(self):
        self.events = []

    def emit(self, kind, **data):
        self.events.append({"kind": kind, **data})


def state(height=0, battery=85):
    return sdk_state(f"h:{height};bat:{battery};yaw:10;vgx:0;vgy:0;vgz:0;".encode())


def probe(protocol="sdk", flight=True):
    result = diag.Probe(protocol, MemoryEvidence(), flight=flight, video=False)
    result._accept_state(state())
    result.connected = True
    result.sent = []
    result.send = lambda data, label: result.sent.append((label, data))
    result.native_command = lambda cmd, payload=b"", label="native": result.sent.append(
        (label, native_packet(cmd, payload)))
    return result


@pytest.mark.parametrize("axes", list(AXES.values()) + [ZERO])
def test_native_sticks_match_reference_encoder_without_constructing_drone(axes):
    # Invoke only the upstream encoding method on an uninitialized shell: no sockets/threads.
    from tellopy._internal.tello import Tello
    from tellopy._internal.protocol import Packet
    from unittest.mock import patch

    shell = object.__new__(Tello)
    shell.right_x, shell.right_y, shell.left_y, shell.left_x = [x / 100 for x in axes]
    captured = []
    shell.send_packet = lambda packet: captured.append(bytes(packet.get_buffer()))
    stamp = datetime(2026, 9, 10, 14, 27, 39, 123000)
    original = Packet.add_time
    with patch.object(Packet, "add_time", lambda p: original(p, stamp)):
        shell._Tello__send_stick_command()
    assert native_sticks(axes, stamp) == captured[0]
    command, sequence, payload = decode_native(captured[0])
    assert command == 0x50
    packed = int.from_bytes(payload[:6], "little")
    assert packed >> 44 == 0  # fast-mode flag disabled
    assert [(packed >> (11 * i)) & 0x7ff for i in range(4)] == [
        int(1024 + 660 * x / 100) for x in axes]


def test_native_timestamp_changes_and_crc_rejects_corruption():
    a = native_sticks(ZERO, datetime(2026, 1, 1, 1, 1, 1))
    b = native_sticks(ZERO, datetime(2026, 1, 1, 1, 1, 2))
    assert a != b
    for index in (0, 1, 3, 5, 9, len(a) - 1):
        damaged = bytearray(a)
        damaged[index] ^= 1
        with pytest.raises(ValueError):
            decode_native(bytes(damaged))
    with pytest.raises(ValueError):
        decode_native(a[:8])


def test_telemetry_units_and_flags_are_distinct():
    payload = bytearray(24)
    struct.pack_into("<h", payload, 0, 7)
    payload[12], payload[17] = 85, 1
    decoded = native_state(payload)
    assert decoded["height_m"] == 0.7
    assert decoded["flying_flag"] is True
    assert decoded["ground_flag"] is False
    assert decoded["yaw_degrees"] is None
    assert state(70)["height_m"] == decoded["height_m"]
    payload[12] = 255
    with pytest.raises(ValueError):
        native_state(payload)
    with pytest.raises(ValueError):
        native_state(b"\x00" * 23)


@pytest.mark.parametrize("data", [b"h:nan;bat:80;", b"h:0;bat:101;",
                                      b"h:0;bat:85.5;", b"h:0;bat:80;yaw:inf;", b"bat:80;"])
def test_bad_sdk_telemetry_cannot_refresh_health(data):
    with pytest.raises((ValueError, KeyError)):
        sdk_state(data)


@pytest.mark.parametrize("axes", [(21, 0, 0, 0), (True, 0, 0, 0), (0, 0, 0), (20, 20, 0, 0)])
def test_stick_caps_and_single_axis(axes):
    with pytest.raises(ValueError):
        validate_axes(axes)


@pytest.mark.parametrize("protocol", ["sdk", "native"])
def test_ground_never_sends_flight_commands(protocol):
    drone = probe(protocol, flight=False)
    with pytest.raises(RuntimeError):
        drone.takeoff()
    with pytest.raises(RuntimeError):
        drone.pulse("w")
    drone.request_land("quit")
    drone.close()
    assert drone.sent == []


@pytest.mark.parametrize("change", ["stale", "low_battery", "high", "log_failed"])
def test_preflight_blocks_before_sending(change):
    drone = probe()
    if change == "stale":
        drone.state_at -= 2
    elif change == "low_battery":
        drone._accept_state(state(battery=29))
    elif change == "high":
        drone._accept_state(state(height=50))
    else:
        drone.evidence.error = "disk failure"
    with pytest.raises(RuntimeError):
        drone.takeoff()
    assert drone.takeoff_at is None
    assert not drone.sent


@pytest.mark.parametrize("protocol", ["sdk", "native"])
def test_takeoff_once_and_no_configuration_writes(protocol):
    drone = probe(protocol)
    drone.takeoff()
    assert drone.takeoff_at is not None
    assert [label for label, _ in drone.sent] == ["takeoff"]
    if protocol == "native":
        assert decode_native(drone.sent[0][1])[0] == 0x54
    with pytest.raises(RuntimeError):
        drone.takeoff()
    assert drone.airborne_at is None  # sent/acknowledged is not observed airborne


def test_uncertain_takeoff_send_still_requires_cleanup_landing(monkeypatch):
    drone = probe()
    def broken_send(*args):
        raise OSError("lost reply / failed send")
    drone.send = broken_send
    with pytest.raises(OSError):
        drone.takeoff()
    assert drone.takeoff_at is not None
    calls = []
    def fallback_land():
        calls.append("land")
        drone.landed_evidence = True
    drone._send_land = fallback_land
    monkeypatch.setattr(diag.time, "sleep", lambda seconds: None)
    drone.close()
    assert calls == ["land"]
    assert any(e["kind"] == "landing.result" for e in drone.evidence.events)


@pytest.mark.parametrize("values,expected", [
    ((None, 0, 0, False), "telemetry_stale"),
    ((state(60), 1.1, 3, True), "telemetry_stale"),
    ((state(60, 15), 0, 3, True), "critical_battery"),
    ((state(150), 0, 3, True), "height_ceiling"),
    ((state(60), 0, 120, True), "session_time_limit"),
    ((state(), 0, 11, False), "takeoff_not_observed"),
    ((state(60), 0, 5, True), None),
])
def test_watchdog_thresholds(values, expected):
    assert flight_fault(*values) == expected


def test_pump_expires_motion_without_ui_or_video():
    drone = probe()
    drone.takeoff_at = time.monotonic() - 5
    drone.airborne_at = time.monotonic() - 4
    drone._accept_state(state(60))
    drone.pulse("w")
    thread = threading.Thread(target=drone._pump)
    thread.start()
    time.sleep(0.48)
    drone.stop.set()
    thread.join(2)
    sticks = [data for label, data in drone.sent if label == "sticks"]
    assert b"rc 0 10 0 0" in sticks
    assert sticks[-1] == b"rc 0 0 0 0"
    assert drone.axes == ZERO
    assert not thread.is_alive()


def test_stale_state_neutralizes_and_lands_on_pump_thread():
    drone = probe()
    drone.takeoff_at = time.monotonic() - 5
    drone.airborne_at = time.monotonic() - 4
    drone.axes, drone.motion_until = AXES["w"], time.monotonic() + 10
    drone.state_at -= 2
    thread = threading.Thread(target=drone._pump)
    thread.start()
    time.sleep(0.12)
    drone.stop.set()
    thread.join(2)
    assert drone.fault == "telemetry_stale"
    assert ("land", b"land") in drone.sent
    assert all(data == b"rc 0 0 0 0" for label, data in drone.sent if label == "sticks")


def test_landing_requires_new_ground_packets_and_duration(monkeypatch):
    drone = probe()
    drone.takeoff_at = 10
    drone.landing_at = 11
    clock = [12.0]
    monkeypatch.setattr(diag.time, "monotonic", lambda: clock[0])
    drone._accept_state(state())
    clock[0] = 15
    assert not drone.landed_evidence  # merely waiting on cached state is not proof
    drone._accept_state(state())
    assert drone.landed_evidence


def test_native_landing_also_requires_ground_flag(monkeypatch):
    drone = probe("native")
    drone.takeoff_at, drone.landing_at = 10, 11
    clock = [12.0]
    monkeypatch.setattr(diag.time, "monotonic", lambda: clock[0])
    payload = bytearray(24)
    payload[12] = 80
    drone._accept_state(native_state(payload))
    clock[0] = 15
    drone._accept_state(native_state(payload))
    assert not drone.landed_evidence
    payload[17] = 2
    drone._accept_state(native_state(payload))
    clock[0] = 18
    drone._accept_state(native_state(payload))
    assert drone.landed_evidence


class FakeSocket:
    """No real sockets; exercise the actual receiver and handshake implementations."""
    def __init__(self, *args):
        self.port = None
        self.incoming = queue.Queue()
        self.sent = []
        self.closed = False

    def bind(self, address):
        self.port = address[1]

    def setblocking(self, value):
        pass

    def getsockname(self):
        return ("0.0.0.0", self.port)

    def sendto(self, data, address):
        self.sent.append(data)
        if data.startswith(b"conn_req:"):
            self.incoming.put(b"conn_ack:\x96\x17")
            payload = bytearray(24)
            payload[12], payload[17] = 85, 2
            self.incoming.put(native_packet(0x56, payload))
        elif data in (b"command", b"sdk?", b"battery?", b"wifi?"):
            self.incoming.put(b"ok" if data == b"command" else b"85")
        return len(data)

    def recvfrom(self, size):
        return self.incoming.get_nowait(), (diag.HOST, 8889)

    def close(self):
        self.closed = True


@pytest.mark.parametrize("protocol", ["native", "sdk"])
def test_ground_handshake_and_receive_end_to_end_without_drone(monkeypatch, protocol):
    sockets = []
    def factory(*args):
        sock = FakeSocket()
        sockets.append(sock)
        return sock
    def select_ready(read, write, errors, timeout):
        for sock in read:
            if sock.port == 8890:
                sock.incoming.put(b"h:0;bat:85;yaw:0;")
        time.sleep(0.005)
        return [s for s in read if not s.incoming.empty()], [], []
    monkeypatch.setattr(diag.socket, "socket", factory)
    monkeypatch.setattr(diag.select, "select", select_ready)
    drone = diag.Probe(protocol, MemoryEvidence(), video=False)
    try:
        drone.open()
        assert drone.connected
        assert drone.snapshot()[0]["battery"] == 85
    finally:
        drone.close()
    assert all(s.closed for s in sockets)
    all_data = [data for s in sockets for data in s.sent]
    assert b"takeoff" not in all_data and b"land" not in all_data
    assert all(decode_native(data)[0] not in (0x54, 0x55, 0x58)
               for data in all_data if data.startswith(b"\xcc"))


def test_video_decodes_and_records_offline(tmp_path):
    import av
    import numpy as np
    from fractions import Fraction

    encoder = av.CodecContext.create("libx264", "w")
    encoder.width, encoder.height, encoder.pix_fmt = 640, 480, "yuv420p"
    encoder.time_base = Fraction(1, 20)
    encoder.options = {"preset": "ultrafast", "tune": "zerolatency"}
    frame = av.VideoFrame.from_ndarray(np.zeros((480, 640, 3), dtype=np.uint8), format="bgr24")
    data = b"".join(bytes(p) for _ in range(4) for p in encoder.encode(frame))
    data += b"".join(bytes(p) for p in encoder.encode(None))
    evidence = MemoryEvidence()
    evidence.path = tmp_path
    drone = diag.Probe("native", evidence)
    drone.codec = av.CodecContext.create("h264", "r")
    drone.video_queue.put(data)
    drone.stop.set()
    drone._video_worker()
    assert drone.decoded_frames >= 2
    assert drone.resolution == (640, 480)
    assert (tmp_path / "video.h264").read_bytes() == data


def test_report_does_not_call_dispatch_success_a_movement_pass(tmp_path):
    evidence = diag.Evidence(tmp_path, "sdk", "flight")
    evidence.emit("motion.pulse", axis="forward", duration_s=0.35)
    evidence.emit("takeoff.requested")
    evidence.close()
    (evidence.path / "session.json").write_text(json.dumps({
        "mode": "flight", "protocol": "sdk", "video": False}), encoding="utf-8")
    result = diag.summarize(evidence.path)
    assert not result["diagnostic_pass"]
    assert not result["takeoff_observed"]
    assert result["axis_observations"]["forward"] == "untested"


def test_report_retains_wrong_observation_after_success(tmp_path):
    evidence = diag.Evidence(tmp_path, "sdk", "flight")
    evidence.emit("motion.observation", axis="forward", result="wrong")
    evidence.emit("motion.observation", axis="forward", result="correct")
    evidence.close()
    (evidence.path / "session.json").write_text(json.dumps({
        "mode": "flight", "protocol": "sdk", "video": False}), encoding="utf-8")
    assert diag.summarize(evidence.path)["axis_observations"]["forward"] == "wrong"


def test_detector_worker_serializes_actual_project_detection():
    from drone_ai.vision.models import Detection

    evidence = MemoryEvidence()
    drone = diag.Probe("sdk", evidence, flight=False)
    drone.latest_frame, drone.last_frame_at = object(), time.monotonic()

    class Detector:
        def detect(self, frame):
            drone.stop.set()
            return [Detection("person", 0.9, (1, 2, 30, 40))]

    drone.start_detector(Detector())
    for thread in drone.threads:
        thread.join(2)
    events = [e for e in evidence.events if e["kind"] == "detector.inference"]
    assert events[0]["detections"][0]["class_name"] == "person"
    assert drone.fault is None


def test_detector_has_no_flight_mode():
    with pytest.raises(RuntimeError):
        probe(flight=True).start_detector(object())


def test_hot_native_drone_is_blocked_before_takeoff():
    drone = probe("native")
    payload = bytearray(24)
    payload[12], payload[17], payload[23] = 85, 2, 1
    drone._accept_state(native_state(payload))
    with pytest.raises(RuntimeError, match="overtemperature"):
        drone.takeoff()
    assert not drone.sent


def test_excessive_sdk_attitude_is_a_flight_fault():
    unstable = state(60)
    unstable["raw"]["pitch"] = 15.0
    assert flight_fault(unstable, 0, 5, True) == "excessive_attitude"


def test_unexpected_ground_contact_lands(monkeypatch):
    drone = probe()
    now = time.monotonic()
    drone.takeoff_at, drone.airborne_at = now - 5, now - 4
    drone._accept_state(state())
    drone.unexpected_ground_since = now - 1
    thread = threading.Thread(target=drone._pump)
    thread.start()
    time.sleep(0.12)
    drone.stop.set()
    thread.join(2)
    assert drone.fault == "unexpected_ground_contact"
    assert ("land", b"land") in drone.sent


def test_sdk_query_heartbeat_is_independent_of_zero_rc_packets():
    drone = probe()
    now = time.monotonic()
    drone.takeoff_at, drone.airborne_at = now - 5, now - 4
    drone.next_sdk_heartbeat_at = now
    drone._accept_state(state(60))
    thread = threading.Thread(target=drone._pump)
    thread.start()
    time.sleep(0.12)
    drone.stop.set()
    thread.join(2)
    assert ("sdk_heartbeat_battery_query", b"battery?") in drone.sent
    assert sum(e["kind"] == "sdk.heartbeat" for e in drone.evidence.events) == 1


def test_sdk_receiver_correlates_numeric_heartbeat_response(monkeypatch):
    drone = probe()
    control = FakeSocket()
    drone.sockets = {control: "control"}
    drone.control = control
    drone.pending_sdk_heartbeat_at = time.monotonic()
    control.incoming.put(b"72")
    calls = [0]
    def select_once(read, write, errors, timeout):
        calls[0] += 1
        if calls[0] > 1:
            drone.stop.set()
            return [], [], []
        return [control], [], []
    monkeypatch.setattr(diag.select, "select", select_once)
    drone._receive()
    responses = [e for e in drone.evidence.events if e["kind"] == "sdk.heartbeat_response"]
    assert responses[0]["value"] == 72
    assert drone.pending_sdk_heartbeat_at is None


def test_early_ground_exit_cannot_pass(tmp_path):
    evidence = diag.Evidence(tmp_path, "sdk", "ground")
    evidence.emit("observation.finished", elapsed_s=2)
    evidence.close()
    (evidence.path / "session.json").write_text(json.dumps({
        "mode": "ground", "protocol": "sdk", "video": False, "seconds": 30}), encoding="utf-8")
    result = diag.summarize(evidence.path)
    assert "ground observation did not complete its requested duration" in result["issues"]
