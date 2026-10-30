from __future__ import annotations

import math

import pytest

from drone_ai.controllers.base import DroneControllerError
from drone_ai.controllers.simulated import SimulatedDrone


def flying_drone() -> SimulatedDrone:
    drone = SimulatedDrone()
    drone.connect()
    drone.takeoff()
    return drone


def test_initial_state_and_connection_rules() -> None:
    drone = SimulatedDrone()
    assert drone.get_state().x_m == 0.0
    assert drone.get_state().altitude_m == 0.0
    assert not drone.get_state().is_connected
    with pytest.raises(DroneControllerError):
        drone.takeoff()
    with pytest.raises(DroneControllerError):
        drone.move_forward(1.0)


def test_configurable_initial_pose_is_finite_and_normalized() -> None:
    drone = SimulatedDrone(
        initial_x_m=1.25,
        initial_y_m=-0.5,
        initial_yaw_degrees=450.0,
    )
    state = drone.get_state()
    assert state.x_m == pytest.approx(1.25)
    assert state.y_m == pytest.approx(-0.5)
    assert state.yaw_degrees == pytest.approx(90.0)
    with pytest.raises(ValueError, match="position must be finite"):
        SimulatedDrone(initial_x_m=math.nan)


def test_takeoff_land_and_disconnect() -> None:
    drone = flying_drone()
    assert drone.get_state().is_flying
    assert drone.get_state().altitude_m == pytest.approx(1.0)
    with pytest.raises(DroneControllerError):
        drone.disconnect()
    drone.land()
    assert not drone.get_state().is_flying
    assert drone.get_state().altitude_m == 0.0
    drone.disconnect()
    assert not drone.get_state().is_connected


def test_all_movement_axes_at_zero_yaw() -> None:
    drone = flying_drone()
    drone.move_forward(1.0)
    drone.move_backward(0.25)
    drone.move_right(2.0)
    drone.move_left(0.5)
    drone.move_up(0.4)
    drone.move_down(0.2)
    state = drone.get_state()
    assert state.x_m == pytest.approx(1.5)
    assert state.y_m == pytest.approx(0.75)
    assert state.altitude_m == pytest.approx(1.2)


def test_cardinal_yaw_changes_world_direction() -> None:
    drone = flying_drone()
    drone.rotate_clockwise(90)
    drone.move_forward(1.0)
    drone.move_right(2.0)
    state = drone.get_state()
    assert state.x_m == pytest.approx(1.0)
    assert state.y_m == pytest.approx(-2.0)


def test_non_cardinal_yaw_uses_trigonometry() -> None:
    drone = flying_drone()
    drone.rotate_clockwise(45)
    drone.move_forward(math.sqrt(2))
    state = drone.get_state()
    assert state.x_m == pytest.approx(1.0)
    assert state.y_m == pytest.approx(1.0)


def test_yaw_wraps_in_both_directions() -> None:
    drone = flying_drone()
    drone.rotate_counter_clockwise(10)
    assert drone.get_state().yaw_degrees == pytest.approx(350)
    drone.rotate_clockwise(30)
    assert drone.get_state().yaw_degrees == pytest.approx(20)


def test_emergency_stops_even_when_disconnected_and_requires_reconnect() -> None:
    drone = SimulatedDrone()
    drone.emergency_stop()
    assert not drone.get_state().is_flying
    drone.connect()
    drone.takeoff()
    drone.emergency_stop()
    assert drone.get_state().altitude_m == 0.0
    assert not drone.get_state().is_flying
    with pytest.raises(DroneControllerError):
        drone.takeoff()
    drone.disconnect()
    drone.connect()
    drone.takeoff()
    assert drone.get_state().is_flying
