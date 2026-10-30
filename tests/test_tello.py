from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from drone_ai.cameras.base import CameraError
from drone_ai.cameras.tello import TelloCamera
from drone_ai.controllers.base import DroneControllerError
from drone_ai.controllers.tello import TelloDroneController
from drone_ai.tello import TelloAdapterError, TelloSession


class FrameReader:
    frame = np.ones((720, 960, 3), dtype=np.uint8)
    stopped = False


class FakeTello:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.height = 0
        self.yaw = -10
        self.battery = 87
        self.is_flying = False
        self.reader = FrameReader()

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append((name, args))

    def connect(self) -> None:
        self._record("connect")

    def end(self) -> None:
        self._record("end")

    def set_speed(self, speed: int) -> None:
        self._record("set_speed", speed)

    def takeoff(self) -> None:
        self._record("takeoff")
        self.is_flying = True

    def land(self) -> None:
        self._record("land")
        self.is_flying = False

    def emergency(self) -> None:
        self._record("emergency")
        self.is_flying = False

    def send_control_command(self, command: str) -> None:
        self._record("send_control_command", command)

    def send_rc_control(
        self,
        left_right_velocity: int,
        forward_backward_velocity: int,
        up_down_velocity: int,
        yaw_velocity: int,
    ) -> None:
        self._record(
            "send_rc_control",
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity,
        )

    def move_forward(self, amount: int) -> None:
        self._record("move_forward", amount)

    def move_back(self, amount: int) -> None:
        self._record("move_back", amount)

    def move_left(self, amount: int) -> None:
        self._record("move_left", amount)

    def move_right(self, amount: int) -> None:
        self._record("move_right", amount)

    def move_up(self, amount: int) -> None:
        self._record("move_up", amount)

        self.height += amount

    def move_down(self, amount: int) -> None:
        self._record("move_down", amount)
        self.height -= amount

    def rotate_clockwise(self, amount: int) -> None:
        self._record("rotate_clockwise", amount)

    def rotate_counter_clockwise(self, amount: int) -> None:
        self._record("rotate_counter_clockwise", amount)

    def get_height(self) -> int:
        self._record("get_height")
        return self.height

    def get_yaw(self) -> int:
        self._record("get_yaw")
        return self.yaw

    def get_battery(self) -> int:
        self._record("get_battery")
        return self.battery

    def query_battery(self) -> int:
        self._record("query_battery")
        return self.battery

    def streamon(self) -> None:
        self._record("streamon")

    def streamoff(self) -> None:
        self._record("streamoff")

    def get_frame_read(self) -> FrameReader:
        self._record("get_frame_read")
        return self.reader


def connected_controller() -> tuple[FakeTello, TelloSession, TelloDroneController]:
    client = FakeTello()
    session = TelloSession(client)
    controller = TelloDroneController(session)
    controller.connect()
    return client, session, controller


def test_method_mapping_and_unit_conversion() -> None:
    client, _, controller = connected_controller()
    controller.takeoff()
    controller.hover()
    controller.move_forward(0.305)
    controller.move_backward(0.2)
    controller.move_left(0.4)
    controller.move_right(0.5)
    controller.move_up(0.2)
    controller.move_down(0.2)
    controller.rotate_clockwise(10.4)
    controller.rotate_counter_clockwise(20.6)
    assert ("set_speed", (20,)) in client.calls
    assert ("send_control_command", ("stop",)) in client.calls
    assert ("move_forward", (30,)) in client.calls
    assert ("move_back", (20,)) in client.calls
    assert ("move_left", (40,)) in client.calls
    assert ("move_right", (50,)) in client.calls
    assert ("move_up", (20,)) in client.calls
    assert ("move_down", (20,)) in client.calls
    assert ("rotate_clockwise", (10,)) in client.calls
    assert ("rotate_counter_clockwise", (21,)) in client.calls


def test_telemetry_conversion_and_normalization() -> None:
    client, _, controller = connected_controller()
    controller.takeoff()
    state = controller.get_state()
    assert state.x_m is None and state.y_m is None
    assert state.altitude_m == 0.0
    assert state.yaw_degrees == 350.0
    assert state.battery_percent == 87
    assert state.speed_mps == 0.2
    assert state.is_flying
    assert ("send_control_command", ("stop",)) in client.calls


def test_airborne_state_sends_immediate_then_rate_limited_hover_heartbeats() -> None:
    now = [0.0]
    client = FakeTello()
    session = TelloSession(client)
    controller = TelloDroneController(
        session,
        keepalive_interval_s=5.0,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()
    client.height = 100

    controller.get_state()
    assert [name for name, _ in client.calls].count("send_control_command") == 1

    now[0] = 4.9
    controller.get_state()
    assert [name for name, _ in client.calls].count("send_control_command") == 1

    now[0] = 5.0
    controller.get_state()
    assert [name for name, _ in client.calls].count("send_control_command") == 2


@pytest.mark.parametrize(
    ("method_name", "expected_velocity"),
    [
        ("move_forward", (0, 40, 0, 0)),
        ("move_backward", (0, -40, 0, 0)),
        ("move_left", (-40, 0, 0, 0)),
        ("move_right", (40, 0, 0, 0)),
        ("move_up", (0, 0, 40, 0)),
        ("move_down", (0, 0, -40, 0)),
    ],
)
def test_manual_rc_translation_uses_velocity_burst_then_hover(
    method_name: str,
    expected_velocity: tuple[int, int, int, int],
) -> None:
    now = [0.0]
    client = FakeTello()
    client.height = 100
    session = TelloSession(client)
    controller = TelloDroneController(
        session,
        rc_control_mode=True,
        keepalive_interval_s=0.5,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()
    client.calls.clear()

    getattr(controller, method_name)(0.2)

    # Scheduling is immediate and non-blocking; telemetry ticks maintain the
    # velocity and stop it at the requested duration.
    assert now[0] == 0.0
    now[0] = 0.375
    controller.get_state()
    now[0] = 0.75
    controller.get_state()

    rc_calls = [args for name, args in client.calls if name == "send_rc_control"]
    assert expected_velocity in rc_calls
    assert rc_calls[-1] == (0, 0, 0, 0)
    assert now[0] == pytest.approx(0.75)
    assert not any(name.startswith("move_") for name, _ in client.calls)


def test_manual_rc_motion_can_be_stopped_before_its_deadline() -> None:
    now = [0.0]
    client = FakeTello()
    client.height = 100
    controller = TelloDroneController(
        TelloSession(client),
        rc_control_mode=True,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()
    client.calls.clear()

    controller.move_forward(0.2)
    controller.hover()
    now[0] = 0.5
    controller.get_state()

    rc_calls = [args for name, args in client.calls if name == "send_rc_control"]
    assert rc_calls == [(0, 40, 0, 0), (0, 0, 0, 0)]


def test_manual_rc_uses_supported_acknowledged_query_as_keepalive() -> None:
    now = [0.0]
    client = FakeTello()
    client.height = 100
    controller = TelloDroneController(
        TelloSession(client),
        rc_control_mode=True,
        acknowledged_keepalive_interval_s=5.0,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()
    client.calls.clear()

    now[0] = 4.9
    controller.get_state()
    assert not any(name == "query_battery" for name, _ in client.calls)

    now[0] = 5.0
    state = controller.get_state()
    assert state.battery_percent == client.battery
    assert [name for name, _ in client.calls].count("query_battery") == 1

    now[0] = 9.9
    controller.get_state()
    assert [name for name, _ in client.calls].count("query_battery") == 1

    now[0] = 10.0
    controller.get_state()
    assert [name for name, _ in client.calls].count("query_battery") == 2


def test_manual_rc_acknowledged_keepalive_failure_is_explicit() -> None:
    class BrokenQuery(FakeTello):
        def query_battery(self) -> int:
            raise RuntimeError("no reply")

    now = [0.0]
    client = BrokenQuery()
    client.height = 100
    controller = TelloDroneController(
        TelloSession(client), rc_control_mode=True, monotonic_clock=lambda: now[0]
    )
    controller.connect()
    controller.takeoff()
    now[0] = 5.0
    with pytest.raises(TelloAdapterError, match="acknowledged keepalive failed"):
        controller.get_state()


def test_emergency_cancels_scheduled_manual_rc_motion() -> None:
    now = [0.0]
    client = FakeTello()
    client.height = 100
    controller = TelloDroneController(
        TelloSession(client),
        rc_control_mode=True,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()
    client.calls.clear()

    controller.move_right(0.2)
    controller.emergency_stop()
    now[0] = 0.5
    assert not controller.get_state().is_flying

    assert ("emergency", ()) in client.calls
    assert [args for name, args in client.calls if name == "send_rc_control"] == [
        (40, 0, 0, 0)
    ]


@pytest.mark.parametrize(
    ("method_name", "expected_velocity"),
    [
        ("rotate_clockwise", (0, 0, 0, 40)),
        ("rotate_counter_clockwise", (0, 0, 0, -40)),
    ],
)
def test_manual_rotation_uses_non_blocking_rc_yaw(
    method_name: str,
    expected_velocity: tuple[int, int, int, int],
) -> None:
    now = [0.0]
    client = FakeTello()
    client.height = 100
    controller = TelloDroneController(
        TelloSession(client),
        rc_control_mode=True,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()
    client.calls.clear()

    getattr(controller, method_name)(20.0)
    assert now[0] == 0.0
    assert ("send_rc_control", expected_velocity) in client.calls
    assert not any(name.startswith("rotate_") for name, _ in client.calls)


def test_spontaneous_landing_reconciles_stale_sdk_state_and_allows_takeoff() -> None:
    now = [0.0]
    client = FakeTello()
    session = TelloSession(client)
    controller = TelloDroneController(
        session,
        keepalive_interval_s=0.5,
        landed_confirmation_s=1.0,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()

    client.height = 100
    assert controller.get_state().is_flying

    # A Tello can land itself without DJITelloPy clearing its cached flag.
    client.height = 0
    now[0] = 1.0
    assert controller.get_state().is_flying
    now[0] = 2.0
    assert not controller.get_state().is_flying
    assert client.is_flying is False

    controller.takeoff()
    assert client.is_flying is True
    assert [name for name, _ in client.calls].count("takeoff") == 2


def test_unconfirmed_takeoff_has_grace_then_recovers_as_landed() -> None:
    now = [0.0]
    client = FakeTello()
    session = TelloSession(client)
    controller = TelloDroneController(
        session,
        landed_confirmation_s=0.1,
        takeoff_confirmation_timeout_s=5.0,
        monotonic_clock=lambda: now[0],
    )
    controller.connect()
    controller.takeoff()
    now[0] = 4.9
    assert controller.get_state().is_flying
    now[0] = 5.0
    assert not controller.get_state().is_flying


@pytest.mark.parametrize("interval", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_keepalive_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="keepalive interval"):
        TelloDroneController(keepalive_interval_s=interval)


@pytest.mark.parametrize("interval", [0.0, 0.01, 0.51, float("nan"), float("inf")])
def test_invalid_rc_update_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="RC update interval"):
        TelloDroneController(rc_update_interval_s=interval)


@pytest.mark.parametrize("interval", [0.0, 1.99, 10.01, float("nan"), float("inf")])
def test_invalid_acknowledged_keepalive_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match="acknowledged keepalive interval"):
        TelloDroneController(acknowledged_keepalive_interval_s=interval)


@pytest.mark.parametrize("velocity", [19, 51, 20.0, True])
def test_invalid_rc_velocity_is_rejected(velocity: object) -> None:
    with pytest.raises(ValueError, match="RC velocity"):
        TelloDroneController(rc_velocity=velocity)  # type: ignore[arg-type]


@pytest.mark.parametrize("duration", [0.1, 1.1, float("nan"), float("inf")])
def test_invalid_rc_step_duration_is_rejected(duration: float) -> None:
    with pytest.raises(ValueError, match="RC step duration"):
        TelloDroneController(rc_step_duration_s=duration)


def test_installed_djitellopy_exposes_every_used_flight_api() -> None:
    djitellopy = pytest.importorskip("djitellopy")
    required_methods = {
        "connect",
        "end",
        "set_speed",
        "takeoff",
        "land",
        "emergency",
        "send_rc_control",
        "move_forward",
        "move_back",
        "move_left",
        "move_right",
        "move_up",
        "move_down",
        "rotate_clockwise",
        "rotate_counter_clockwise",
        "get_height",
        "get_yaw",
        "get_battery",
        "streamon",
        "streamoff",
        "get_frame_read",
    }
    missing = sorted(name for name in required_methods if not hasattr(djitellopy.Tello, name))
    assert missing == []


def test_shared_video_session_and_cleanup() -> None:
    client, session, controller = connected_controller()
    camera = TelloCamera(session)
    camera.open()
    frame = camera.read()
    assert frame.shape == (720, 960, 3)
    assert frame is not client.reader.frame
    camera.close()
    controller.disconnect()
    assert [name for name, _ in client.calls].count("connect") == 1
    assert ("streamon", ()) in client.calls
    assert ("streamoff", ()) in client.calls
    assert client.calls[-1] == ("end", ())


def test_emergency_maps_directly() -> None:
    client, _, controller = connected_controller()
    controller.takeoff()
    controller.emergency_stop()
    assert ("emergency", ()) in client.calls
    assert not controller.get_state().is_flying


def test_vertical_motion_fails_closed_when_altitude_unknown() -> None:
    client, _, controller = connected_controller()
    controller.takeoff()
    client.height = None  # type: ignore[assignment]
    with pytest.raises(TelloAdapterError, match="altitude telemetry unavailable"):
        controller.move_up(0.2)


def test_missing_optional_dependency_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "djitellopy", None)
    with pytest.raises(TelloAdapterError, match="requirements-tello"):
        _ = TelloSession().client


def test_camera_waits_for_real_frame_instead_of_sdk_placeholder() -> None:
    class DelayedReader:
        stopped = False

        def __init__(self) -> None:
            self.reads = 0

        @property
        def frame(self) -> np.ndarray:
            self.reads += 1
            if self.reads < 3:
                return np.zeros((300, 400, 3), dtype=np.uint8)
            return np.ones((720, 960, 3), dtype=np.uint8)

    client = FakeTello()
    client.reader = DelayedReader()  # type: ignore[assignment]
    elapsed = [0.0]
    session = TelloSession(client)
    camera = TelloCamera(
        session,
        first_frame_timeout_s=1.0,
        monotonic_clock=lambda: elapsed[0],
        sleeper=lambda delay: elapsed.__setitem__(0, elapsed[0] + delay),
    )
    camera.open()
    try:
        frame = camera.read()
    finally:
        camera.close()
        session.close()
    assert frame.shape == (720, 960, 3)
    assert client.reader.reads == 3


def test_camera_times_out_on_placeholder_frames() -> None:
    client = FakeTello()
    client.reader.frame = np.zeros((300, 400, 3), dtype=np.uint8)
    elapsed = [0.0]
    session = TelloSession(client)
    camera = TelloCamera(
        session,
        first_frame_timeout_s=0.1,
        poll_interval_s=0.05,
        monotonic_clock=lambda: elapsed[0],
        sleeper=lambda delay: elapsed.__setitem__(0, elapsed[0] + delay),
    )
    camera.open()
    try:
        with pytest.raises(Exception, match="non-placeholder"):
            camera.read()
    finally:
        camera.close()
        session.close()


@pytest.mark.parametrize("battery", [-1, 101, 87.5])
def test_invalid_battery_telemetry_is_rejected(battery: float) -> None:
    client, _, controller = connected_controller()
    client.battery = battery  # type: ignore[assignment]
    with pytest.raises(TelloAdapterError, match="battery telemetry is invalid"):
        controller.get_state()


def test_session_operations_are_idempotent() -> None:
    client = FakeTello()
    session = TelloSession(client)
    session.connect()
    session.connect()
    assert session.start_stream() is client.reader
    assert session.start_stream() is client.reader
    session.stop_stream()
    session.stop_stream()
    session.close()
    session.close()
    assert [name for name, _ in client.calls].count("connect") == 1
    assert [name for name, _ in client.calls].count("streamon") == 1
    assert [name for name, _ in client.calls].count("streamoff") == 1
    assert [name for name, _ in client.calls].count("end") == 1


def test_real_sdk_style_stream_cleanup_does_not_wait_for_acknowledgement() -> None:
    class NonBlockingCleanup(FakeTello):
        stream_on = True
        background_frame_read = None

        def send_command_without_return(self, command: str) -> None:
            self._record("send_command_without_return", command)

    client = NonBlockingCleanup()
    session = TelloSession(client)
    session.connect()
    session._streaming = True
    session.stop_stream()

    assert ("send_command_without_return", ("streamoff",)) in client.calls
    assert ("streamoff", ()) not in client.calls
    assert client.stream_on is False
    assert session.streaming is False


def test_client_factory_and_connection_failures_are_explicit() -> None:
    with pytest.raises(TelloAdapterError, match="could not create"):
        _ = TelloSession(client_factory=lambda: (_ for _ in ()).throw(RuntimeError("factory"))).client

    class ConnectFailure(FakeTello):
        def connect(self) -> None:
            raise RuntimeError("no wifi")

    session = TelloSession(ConnectFailure())
    with pytest.raises(TelloAdapterError, match="verify its Wi-Fi"):
        session.connect()
    assert session.connected is False


def test_frame_reader_failure_rolls_back_a_new_stream() -> None:
    class ReaderFailure(FakeTello):
        def get_frame_read(self):  # type: ignore[no-untyped-def]
            self._record("get_frame_read")
            raise RuntimeError("decoder failed")

    client = ReaderFailure()
    session = TelloSession(client)
    with pytest.raises(TelloAdapterError, match="decoder failed"):
        session.start_stream()
    assert session.connected is True
    assert session.streaming is False
    assert [name for name, _ in client.calls][-2:] == ["get_frame_read", "streamoff"]
    session.close()


def test_stream_rollback_and_close_preserve_all_cleanup_errors() -> None:
    class BrokenCleanup(FakeTello):
        def get_frame_read(self):  # type: ignore[no-untyped-def]
            raise RuntimeError("reader failed")

        def streamoff(self) -> None:
            self._record("streamoff")
            raise RuntimeError("streamoff failed")

        def end(self) -> None:
            self._record("end")
            raise RuntimeError("end failed")

    client = BrokenCleanup()
    session = TelloSession(client)
    with pytest.raises(TelloAdapterError, match="rollback also failed"):
        session.start_stream()
    assert session.streaming is False

    client = BrokenCleanup()
    session = TelloSession(client)
    session.connect()
    session._streaming = True
    with pytest.raises(TelloAdapterError) as captured:
        session.close()
    assert "streamoff failed" in str(captured.value)
    assert "end failed" in str(captured.value)
    assert session.streaming is False
    assert session.connected is False


def test_speed_setup_failure_closes_partial_connection() -> None:
    class SpeedFailure(FakeTello):
        def set_speed(self, speed: int) -> None:
            self._record("set_speed", speed)
            raise RuntimeError("speed rejected")

    client = SpeedFailure()
    session = TelloSession(client)
    controller = TelloDroneController(session)
    with pytest.raises(TelloAdapterError, match="could not set Tello speed"):
        controller.connect()
    assert session.connected is False
    assert client.calls[-1] == ("end", ())


def test_disconnect_checks_sdk_flying_state_not_only_local_flag() -> None:
    client, session, controller = connected_controller()
    client.is_flying = True
    with pytest.raises(DroneControllerError, match="while Tello is flying"):
        controller.disconnect()
    assert session.connected is True
    assert ("end", ()) not in client.calls


def test_camera_reader_stop_regression_and_close_failure_are_explicit() -> None:
    client = FakeTello()
    session = TelloSession(client)
    camera = TelloCamera(session)
    with pytest.raises(CameraError, match="not open"):
        camera.read()
    camera.open()
    client.reader.stopped = True
    with pytest.raises(CameraError, match="stopped unexpectedly"):
        camera.read()
    client.reader.stopped = False
    camera.close()
    session.close()

    class StopFailure(FakeTello):
        def streamoff(self) -> None:
            self._record("streamoff")
            raise RuntimeError("cannot stop")

    broken_client = StopFailure()
    broken_session = TelloSession(broken_client)
    broken_camera = TelloCamera(broken_session)
    broken_camera.open()
    with pytest.raises(CameraError, match="cannot stop"):
        broken_camera.close()
    assert broken_session.streaming is False
    broken_camera.close()
    broken_session.close()


def test_camera_rejects_immediate_regression_after_first_valid_frame() -> None:
    client = FakeTello()
    session = TelloSession(client)
    camera = TelloCamera(session)
    camera.open()
    assert camera.read().shape == (720, 960, 3)
    client.reader.frame = np.zeros((300, 400, 3), dtype=np.uint8)
    with pytest.raises(CameraError, match="did not provide a valid"):
        camera.read()
    camera.close()
    session.close()


def test_vertical_limits_and_invalid_telemetry_fail_closed() -> None:
    client, _, controller = connected_controller()
    controller.takeoff()
    client.height = 290
    with pytest.raises(TelloAdapterError, match="altitude ceiling"):
        controller.move_up(0.2)
    client.height = 10
    with pytest.raises(TelloAdapterError, match="altitude floor"):
        controller.move_down(0.2)
    client.height = float("nan")  # type: ignore[assignment]
    with pytest.raises(TelloAdapterError, match="not finite"):
        controller.get_state()
