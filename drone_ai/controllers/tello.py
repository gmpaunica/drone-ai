"""DJI Tello implementation of the flight-controller interface."""

from __future__ import annotations

from collections.abc import Callable
import logging
import math
import time
from typing import Any

from drone_ai.config import FlightConfig
from drone_ai.controllers.base import DroneController, DroneControllerError, DroneState
from drone_ai.tello import TelloAdapterError, TelloSession


LOGGER = logging.getLogger(__name__)


class TelloDroneController(DroneController):
    def __init__(
        self,
        session: TelloSession | None = None,
        config: FlightConfig | None = None,
        *,
        rc_control_mode: bool = False,
        keepalive_interval_s: float = 5.0,
        rc_update_interval_s: float = 0.1,
        rc_velocity: int = 40,
        rc_step_duration_s: float = 0.75,
        acknowledged_keepalive_interval_s: float = 5.0,
        airborne_altitude_threshold_m: float = 0.3,
        landed_altitude_threshold_m: float = 0.1,
        landed_confirmation_s: float = 1.0,
        takeoff_confirmation_timeout_s: float = 5.0,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session = session or TelloSession()
        self.config = config or FlightConfig()
        if not 0 < self.config.cruise_speed_mps <= self.config.max_speed_mps:
            raise ValueError("cruise speed must be positive and within the hard maximum")
        if not math.isfinite(keepalive_interval_s) or keepalive_interval_s <= 0:
            raise ValueError("Tello keepalive interval must be finite and positive")
        if not math.isfinite(rc_update_interval_s) or not 0.02 <= rc_update_interval_s <= 0.5:
            raise ValueError("Tello RC update interval must be within 0.02-0.5 seconds")
        if isinstance(rc_velocity, bool) or not isinstance(rc_velocity, int) or not 20 <= rc_velocity <= 50:
            raise ValueError("Tello RC velocity must be an integer within 20-50")
        if not math.isfinite(rc_step_duration_s) or not 0.2 <= rc_step_duration_s <= 1.0:
            raise ValueError("Tello RC step duration must be within 0.2-1.0 seconds")
        if (not math.isfinite(acknowledged_keepalive_interval_s)
                or not 2.0 <= acknowledged_keepalive_interval_s <= 10.0):
            raise ValueError("Tello acknowledged keepalive interval must be within 2-10 seconds")
        if not (
            math.isfinite(airborne_altitude_threshold_m)
            and math.isfinite(landed_altitude_threshold_m)
            and 0.0 <= landed_altitude_threshold_m < airborne_altitude_threshold_m
        ):
            raise ValueError("Tello landing-detection altitude thresholds are invalid")
        if not math.isfinite(landed_confirmation_s) or landed_confirmation_s <= 0:
            raise ValueError("Tello landed confirmation time must be finite and positive")
        if (
            not math.isfinite(takeoff_confirmation_timeout_s)
            or takeoff_confirmation_timeout_s <= 0
        ):
            raise ValueError("Tello takeoff confirmation timeout must be finite and positive")
        self._keepalive_interval_s = keepalive_interval_s
        self._rc_control_mode = rc_control_mode
        self._rc_update_interval_s = rc_update_interval_s
        self._rc_velocity = rc_velocity
        self._rc_step_duration_s = rc_step_duration_s
        self._acknowledged_keepalive_interval_s = acknowledged_keepalive_interval_s
        self._airborne_altitude_threshold_m = airborne_altitude_threshold_m
        self._landed_altitude_threshold_m = landed_altitude_threshold_m
        self._landed_confirmation_s = landed_confirmation_s
        self._takeoff_confirmation_timeout_s = takeoff_confirmation_timeout_s
        self._clock = monotonic_clock
        self._last_command_at: float | None = None
        self._last_acknowledged_command_at: float | None = None
        self._rc_motion_velocity: tuple[int, int, int, int] | None = None
        self._rc_motion_deadline: float | None = None
        self._next_rc_update_at: float | None = None
        self._is_flying = False
        self._airborne_altitude_seen = False
        self._landed_altitude_since: float | None = None
        self._takeoff_started_at: float | None = None

    def connect(self) -> None:
        self.session.connect()
        speed_cm_s = int(round(self.config.cruise_speed_mps * 100))
        try:
            self.session.client.set_speed(speed_cm_s)
        except Exception as exc:
            cleanup_detail = ""
            try:
                self.session.close()
            except Exception as cleanup_exc:
                cleanup_detail = f"; session cleanup also failed: {cleanup_exc}"
            raise TelloAdapterError(
                f"connected but could not set Tello speed: {exc}{cleanup_detail}"
            ) from exc
        self._last_command_at = float(self._clock())
        self._last_acknowledged_command_at = self._last_command_at
        self._is_flying = bool(getattr(self.session.client, "is_flying", False))
        self._airborne_altitude_seen = False
        self._landed_altitude_since = None
        self._takeoff_started_at = None

    def disconnect(self) -> None:
        sdk_flying = False
        if self.session.connected:
            try:
                sdk_flying = bool(getattr(self.session.client, "is_flying", False))
            except Exception as exc:
                raise DroneControllerError(
                    f"cannot confirm Tello is landed before disconnect: {exc}"
                ) from exc
        if self._is_flying or sdk_flying:
            raise DroneControllerError("cannot disconnect while Tello is flying")
        self.session.close()
        self._cancel_rc_motion(send_zero=False)
        self._last_command_at = None
        self._last_acknowledged_command_at = None
        self._reset_flight_observation()

    def takeoff(self) -> None:
        self._cancel_rc_motion(send_zero=False)
        self._call("takeoff")
        self._is_flying = True
        self._airborne_altitude_seen = False
        self._landed_altitude_since = None
        self._takeoff_started_at = float(self._clock())
        # Force a hover heartbeat on the first telemetry tick after takeoff.
        self._last_command_at = None

    def land(self) -> None:
        if self._rc_control_mode:
            self._cancel_rc_motion(send_zero=True)
        self._call("land")
        self._mark_landed()

    def emergency_stop(self) -> None:
        # Access the client directly so the SDK emergency command is never
        # translated into a softer landing or hover operation.
        try:
            self.session.client.emergency()
        except Exception as exc:
            raise TelloAdapterError(f"Tello emergency command failed: {exc}") from exc
        self._cancel_rc_motion(send_zero=False)
        self._mark_landed()

    def hover(self) -> None:
        if self._rc_control_mode:
            self._cancel_rc_motion(send_zero=True)
        else:
            # DJITelloPy 2.5.0 has no convenience stop() method, but exposes
            # the official SDK command transport.
            self._call("send_control_command", "stop")

    def move_forward(self, distance_m: float) -> None:
        if self._rc_control_mode:
            self._move_with_rc(distance_m, forward_backward=1)
        else:
            self._call("move_forward", self._centimeters(distance_m))

    def move_backward(self, distance_m: float) -> None:
        if self._rc_control_mode:
            self._move_with_rc(distance_m, forward_backward=-1)
        else:
            self._call("move_back", self._centimeters(distance_m))

    def move_left(self, distance_m: float) -> None:
        if self._rc_control_mode:
            self._move_with_rc(distance_m, left_right=-1)
        else:
            self._call("move_left", self._centimeters(distance_m))

    def move_right(self, distance_m: float) -> None:
        if self._rc_control_mode:
            self._move_with_rc(distance_m, left_right=1)
        else:
            self._call("move_right", self._centimeters(distance_m))

    def move_up(self, distance_m: float) -> None:
        altitude = self._altitude_m()
        if altitude + distance_m > self.config.max_altitude_m:
            raise TelloAdapterError("vertical move would exceed configured altitude ceiling")
        if self._rc_control_mode:
            self._move_with_rc(distance_m, up_down=1)
        else:
            self._call("move_up", self._centimeters(distance_m))

    def move_down(self, distance_m: float) -> None:
        altitude = self._altitude_m()
        if altitude - distance_m < self.config.min_altitude_m:
            raise TelloAdapterError("vertical move would exceed configured altitude floor")
        if self._rc_control_mode:
            self._move_with_rc(distance_m, up_down=-1)
        else:
            self._call("move_down", self._centimeters(distance_m))

    def rotate_clockwise(self, degrees: float) -> None:
        if self._rc_control_mode:
            self._rotate_with_rc(degrees, direction=1)
        else:
            self._call("rotate_clockwise", self._whole_degrees(degrees))

    def rotate_counter_clockwise(self, degrees: float) -> None:
        if self._rc_control_mode:
            self._rotate_with_rc(degrees, direction=-1)
        else:
            self._call("rotate_counter_clockwise", self._whole_degrees(degrees))

    def get_state(self) -> DroneState:
        if not self.session.connected:
            return DroneState(
                x_m=None,
                y_m=None,
                altitude_m=None,
                yaw_degrees=0.0,
                is_flying=False,
                is_connected=False,
                battery_percent=None,
                speed_mps=self.config.cruise_speed_mps,
            )
        client = self.session.client
        altitude_m = self._telemetry(client.get_height, "altitude") / 100.0
        yaw = self._telemetry(client.get_yaw, "yaw") % 360.0
        battery_value = self._telemetry(client.get_battery, "battery")
        if not battery_value.is_integer() or not 0 <= battery_value <= 100:
            raise TelloAdapterError(f"Tello battery telemetry is invalid: {battery_value!r}")
        battery = int(battery_value)
        sdk_flying = bool(getattr(client, "is_flying", self._is_flying))
        is_flying = self._reconcile_flight_state(sdk_flying, altitude_m)
        if is_flying:
            if self._rc_control_mode and self._rc_motion_velocity is not None:
                self._service_rc_motion()
            else:
                self._send_hover_heartbeat_if_due()
            if self._rc_control_mode:
                queried_battery = self._send_acknowledged_keepalive_if_due()
                if queried_battery is not None:
                    battery = queried_battery
        return DroneState(
            x_m=None,
            y_m=None,
            altitude_m=altitude_m,
            yaw_degrees=yaw,
            is_flying=is_flying,
            is_connected=True,
            battery_percent=battery,
            speed_mps=self.config.cruise_speed_mps,
        )

    def _call(self, method_name: str, *args: Any) -> None:
        client = self.session.require_connected()
        try:
            getattr(client, method_name)(*args)
        except Exception as exc:
            raise TelloAdapterError(f"Tello {method_name} failed: {exc}") from exc
        self._last_command_at = float(self._clock())
        if method_name != "send_rc_control":
            self._last_acknowledged_command_at = self._last_command_at

    def _send_acknowledged_keepalive_if_due(self) -> int | None:
        """Prevent basic Tello firmware from auto-landing during neutral RC input.

        Hardware evidence showed that zero RC packets did not reset this Tello's
        inactivity timer and that its firmware rejects the newer ``keepalive``
        command. ``battery?`` is supported and its reply is consumed here, so it
        cannot be mistaken for a later blocking lifecycle-command response.
        """
        now = float(self._clock())
        if (self._last_acknowledged_command_at is not None
                and now - self._last_acknowledged_command_at
                < self._acknowledged_keepalive_interval_s):
            return None
        client = self.session.require_connected()
        try:
            value = client.query_battery()
        except Exception as exc:
            raise TelloAdapterError(f"Tello acknowledged keepalive failed: {exc}") from exc
        numeric = self._telemetry(lambda: value, "keepalive battery")
        if not numeric.is_integer() or not 0 <= numeric <= 100:
            raise TelloAdapterError(f"Tello keepalive battery response is invalid: {value!r}")
        self._last_acknowledged_command_at = now
        self._last_command_at = now
        return int(numeric)

    def _send_hover_heartbeat_if_due(self) -> None:
        now = float(self._clock())
        if (
            self._last_command_at is not None
            and now - self._last_command_at < self._keepalive_interval_s
        ):
            return
        self.hover()

    def _move_with_rc(
        self,
        distance_m: float,
        *,
        left_right: int = 0,
        forward_backward: int = 0,
        up_down: int = 0,
    ) -> None:
        """Translate using the same velocity-control mode as controller apps."""

        self._centimeters(distance_m)
        axes = (left_right, forward_backward, up_down)
        if sum(value != 0 for value in axes) != 1 or any(value not in {-1, 0, 1} for value in axes):
            raise DroneControllerError("exactly one RC translation axis is required")
        # RC values are unitless stick positions, not centimeters per second.
        # Scale duration from the schema's 0.2m manual reference step; actual
        # displacement remains open-loop and must be hardware calibrated.
        duration_s = self._rc_step_duration_s * (distance_m / 0.2)
        self._schedule_rc_motion(
            (
                left_right * self._rc_velocity,
                forward_backward * self._rc_velocity,
                up_down * self._rc_velocity,
                0,
            ),
            duration_s,
        )

    def _rotate_with_rc(self, degrees: float, *, direction: int) -> None:
        self._whole_degrees(degrees)
        if direction not in {-1, 1}:
            raise DroneControllerError("RC yaw direction must be -1 or 1")
        duration_s = self._rc_step_duration_s * (degrees / 20.0)
        self._schedule_rc_motion((0, 0, 0, direction * self._rc_velocity), duration_s)

    def _schedule_rc_motion(
        self,
        velocity: tuple[int, int, int, int],
        duration_s: float,
    ) -> None:
        now = float(self._clock())
        self._rc_motion_velocity = velocity
        self._rc_motion_deadline = now + duration_s
        self._next_rc_update_at = now
        self._service_rc_motion()

    def _service_rc_motion(self) -> None:
        velocity = self._rc_motion_velocity
        deadline = self._rc_motion_deadline
        if velocity is None or deadline is None:
            return
        now = float(self._clock())
        if now >= deadline:
            self._cancel_rc_motion(send_zero=True)
            return
        if self._next_rc_update_at is not None and now < self._next_rc_update_at:
            return
        try:
            self._call("send_rc_control", *velocity)
        except Exception:
            self._cancel_rc_motion(send_zero=False)
            raise
        self._next_rc_update_at = now + self._rc_update_interval_s

    def _cancel_rc_motion(self, *, send_zero: bool) -> None:
        had_motion = self._rc_motion_velocity is not None
        self._rc_motion_velocity = None
        self._rc_motion_deadline = None
        self._next_rc_update_at = None
        if send_zero and (had_motion or self._rc_control_mode):
            self._call("send_rc_control", 0, 0, 0, 0)

    def _reconcile_flight_state(self, sdk_flying: bool, altitude_m: float) -> bool:
        """Recover when DJITelloPy's local ``is_flying`` flag becomes stale.

        DJITelloPy changes that flag when its own takeoff/land methods return,
        but the aircraft can also land itself. After a real airborne altitude
        has been observed, a sustained ground-level reading is sufficient to
        clear the stale flag and permit a deliberate later takeoff.
        """

        if not (self._is_flying or sdk_flying):
            self._cancel_rc_motion(send_zero=False)
            self._reset_flight_observation()
            return False

        now = float(self._clock())
        if altitude_m >= self._airborne_altitude_threshold_m:
            self._airborne_altitude_seen = True
            self._landed_altitude_since = None
            self._takeoff_started_at = None
            return True

        if not self._airborne_altitude_seen:
            if (
                altitude_m <= self._landed_altitude_threshold_m
                and self._takeoff_started_at is not None
                and now - self._takeoff_started_at >= self._takeoff_confirmation_timeout_s
            ):
                LOGGER.warning(
                    "[TELLO] Takeoff was not confirmed within %.1fs and altitude is %.2fm; "
                    "synchronizing controller state to landed",
                    self._takeoff_confirmation_timeout_s,
                    altitude_m,
                )
                self._synchronize_sdk_landed()
                return False
            return True

        if altitude_m > self._landed_altitude_threshold_m:
            self._landed_altitude_since = None
            return True

        if self._landed_altitude_since is None:
            self._landed_altitude_since = now
            return True
        if now - self._landed_altitude_since < self._landed_confirmation_s:
            return True

        LOGGER.warning(
            "[TELLO] Aircraft remained at %.2fm after being airborne; "
            "synchronizing controller state to landed",
            altitude_m,
        )
        self._synchronize_sdk_landed()
        return False

    def _synchronize_sdk_landed(self) -> None:
        self._mark_landed()
        try:
            client = self.session.client
            if hasattr(client, "is_flying"):
                client.is_flying = False
        except Exception:
            # The adapter's own state is authoritative if the optional SDK
            # object does not allow its cached flag to be updated.
            pass

    def _mark_landed(self) -> None:
        self._cancel_rc_motion(send_zero=False)
        self._is_flying = False
        self._last_command_at = None
        self._last_acknowledged_command_at = None
        self._reset_flight_observation()

    def _reset_flight_observation(self) -> None:
        self._airborne_altitude_seen = False
        self._landed_altitude_since = None
        self._takeoff_started_at = None

    def _altitude_m(self) -> float:
        client = self.session.require_connected()
        return self._telemetry(client.get_height, "altitude") / 100.0

    @staticmethod
    def _telemetry(getter: Any, name: str) -> float:
        try:
            value = getter()
        except Exception as exc:
            raise TelloAdapterError(f"Tello {name} telemetry unavailable: {exc}") from exc
        if value is None:
            raise TelloAdapterError(f"Tello {name} telemetry unavailable")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise TelloAdapterError(f"Tello {name} telemetry is invalid: {value!r}") from exc
        if not math.isfinite(numeric):
            raise TelloAdapterError(f"Tello {name} telemetry is not finite")
        return numeric

    @staticmethod
    def _centimeters(distance_m: float) -> int:
        if not math.isfinite(distance_m) or distance_m <= 0:
            raise DroneControllerError("distance must be finite and positive")
        return int(round(distance_m * 100))

    @staticmethod
    def _whole_degrees(degrees: float) -> int:
        if not math.isfinite(degrees) or degrees <= 0:
            raise DroneControllerError("rotation must be finite and positive")
        return int(round(degrees))
