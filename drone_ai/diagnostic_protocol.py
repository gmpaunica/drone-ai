"""Small, inspectable protocol codec for supervised Tello diagnostics.

Native packet framing/CRC uses the pinned TelloPy 0.6.0 implementation.
Reference: https://github.com/hanyazou/TelloPy (MIT).
Native height is decimeters, SDK state height is centimeters:
https://pkg.go.dev/gobot.io/x/gobot/v2/platforms/dji/tello#FlightData
No sockets, threads, configuration changes, or flight are started here.
"""

from __future__ import annotations

import math
import struct
from datetime import datetime

ZERO = (0, 0, 0, 0)
AXES = {
    "w": (0, 10, 0, 0), "s": (0, -10, 0, 0),
    "j": (-10, 0, 0, 0), "k": (10, 0, 0, 0),
    "r": (0, 0, 10, 0), "f": (0, 0, -10, 0),
    "a": (0, 0, 0, -10), "d": (0, 0, 0, 10),
}
AXIS_NAMES = dict(zip(AXES, (
    "forward", "backward", "left", "right", "up", "down", "yaw_left", "yaw_right",
)))


def validate_axes(axes: tuple[int, ...]) -> None:
    if len(axes) != 4 or any(type(x) is not int or abs(x) > 10 for x in axes):
        raise ValueError("diagnostic sticks must be four integers within -10..10")
    if sum(x != 0 for x in axes) > 1:
        raise ValueError("diagnostics permit only one active axis")


def native_packet(command: int, payload: bytes = b"", *, kind: int = 0x68,
                  sequence: int = 0, timestamp: datetime | None = None) -> bytes:
    from tellopy._internal import crc
    from tellopy._internal.protocol import Packet

    packet = Packet(command, kind, payload)
    if timestamp is not None:
        # TelloPy's default timestamp is evaluated at import. Always supply it.
        packet.add_time(timestamp)
    # Upstream 0.6.0 fixup shifts only the low length byte and overflows for
    # packets >=32 bytes. Encode the entire 13-bit length before computing CRCs.
    buffer = packet.get_buffer()
    if len(buffer) + 2 > 8191:
        raise ValueError("native packet is too large")
    buffer[1:3] = ((len(buffer) + 2) << 3).to_bytes(2, "little")
    buffer[3] = crc.crc8(buffer[:3])
    buffer[7:9] = (sequence & 0xffff).to_bytes(2, "little")
    return bytes(buffer) + crc.crc16(buffer).to_bytes(2, "little")


def native_sticks(axes: tuple[int, ...], timestamp: datetime | None = None) -> bytes:
    validate_axes(axes)
    # roll, pitch, throttle, yaw: four 11-bit channels; fast-mode bit stays zero.
    packed = sum(int(1024 + 660 * (axis / 100)) << (11 * i)
                 for i, axis in enumerate(axes))
    return native_packet(0x50, packed.to_bytes(6, "little"), kind=0x60,
                         timestamp=timestamp or datetime.now())


def decode_native(packet: bytes) -> tuple[int, int, bytes]:
    from tellopy._internal import crc

    if len(packet) < 11 or packet[0] != 0xcc:
        raise ValueError("not a complete native packet")
    length = int.from_bytes(packet[1:3], "little") >> 3
    if length != len(packet) or crc.crc8(packet[:3]) != packet[3]:
        raise ValueError("invalid native length/header CRC")
    if crc.crc16(packet[:-2]) != int.from_bytes(packet[-2:], "little"):
        raise ValueError("invalid native packet CRC")
    return (*struct.unpack_from("<HH", packet, 5), packet[9:-2])


def native_state(payload: bytes) -> dict:
    from tellopy._internal.protocol import FlightData

    if len(payload) < 24:
        raise ValueError("short native flight telemetry")
    raw = vars(FlightData(payload))
    state = {"height_m": raw["height"] / 10.0,
             "battery": raw["battery_percentage"], "yaw_degrees": None,
             "flying_flag": bool(raw["em_sky"]),
             "ground_flag": bool(raw["em_ground"]),
             "raw": raw}
    validate_state(state)
    return state


def sdk_state(packet: bytes) -> dict:
    raw = {}
    for field in packet.decode("ascii").strip().split(";"):
        if ":" in field:
            key, value = field.split(":", 1)
            raw[key] = float(value)
    state = {"height_m": raw["h"] / 100.0, "battery": raw["bat"],
             "yaw_degrees": raw.get("yaw"), "flying_flag": None,
             "ground_flag": None, "raw": raw}
    validate_state(state)
    return state


def validate_state(state: dict) -> None:
    h, b = state["height_m"], state["battery"]
    if not math.isfinite(h) or not 0 <= h <= 100:
        raise ValueError("invalid height telemetry")
    if not math.isfinite(b) or not 0 <= b <= 100 or int(b) != b:
        raise ValueError("invalid battery telemetry")
    if any(isinstance(v, float) and not math.isfinite(v) for v in state["raw"].values()):
        raise ValueError("non-finite telemetry")


def flight_fault(state: dict | None, age: float, elapsed: float,
                 airborne_seen: bool) -> str | None:
    if state is None or age > 1.0:
        return "telemetry_stale"
    if state["battery"] <= 15:
        return "critical_battery"
    if state["height_m"] >= 1.5:
        return "height_ceiling"
    pitch = state["raw"].get("pitch")
    roll = state["raw"].get("roll")
    if ((pitch is not None and abs(pitch) >= 15)
            or (roll is not None and abs(roll) >= 15)):
        return "excessive_attitude"
    if elapsed >= 120:
        return "session_time_limit"
    if not airborne_seen and elapsed > 10:
        return "takeoff_not_observed"
    if state["raw"].get("temperature_height"):
        return "overtemperature"
    return None
