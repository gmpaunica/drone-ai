# Tello first-day commissioning

For the current unreliable-controls investigation, use [DIAGNOSTICS.md](DIAGNOSTICS.md)
first to compare minimal SDK and native binary control before returning to app flight.

This sequence separates non-flying checks from guarded flight acceptance. Do not skip directly to autonomous following.

## 1. Prepare the area and aircraft

- Fully charge the Tello and controller/device batteries. Software blocks takeoff below 30%, but start commissioning with a full battery.
- Inspect the airframe, motors, propellers, guards, and battery seating. Confirm each propeller is on its matching motor position.
- For telemetry/video commissioning, remove the propellers and keep the aircraft stationary on a clear surface.
- Close other software that could use UDP ports or the camera. Disable VPN routing if it prevents the Tello Wi-Fi connection.
- Keep the Tello's physical power button accessible. Know that `E` is the application's immediate motor emergency key during camera modes.

## 2. Verify the computer before joining Tello Wi-Fi

From the project directory:

```powershell
.\.venv\Scripts\Activate.ps1
python main.py doctor
python -m pytest
```

The shorter daily equivalent is:

```powershell
.\scripts\ready.ps1
```

If you created a local profile, put the global option before the mode:

```powershell
python main.py --config drone.local.toml doctor
```

Expected: Python, dependencies, YOLO, and Ollama pass. The laptop webcam may warn. The Tello check is skipped because it was not requested.

If Ollama is stopped, launch its Windows application or run this in a separate terminal:

```powershell
ollama serve
```

## 3. Commission the Tello without flight

1. Power on the Tello.
2. Join its `TELLO-...` Wi-Fi network and accept that the network has no internet.
3. With propellers still removed, run:

```powershell
.\scripts\ready.ps1 -Tello
```

The command connects once, sets the conservative speed, reads altitude/yaw/battery telemetry, starts video, verifies a BGR frame, stops video, and disconnects. It never calls takeoff, movement, landing, or emergency.

The first real video frame may take several seconds; the software ignores DJITelloPy's small black placeholder and waits up to five seconds. Commissioning also requires the landed altitude to be at most 0.3 m and battery telemetry to be an integer from 0–100%.

Do not proceed if any Tello check fails. Save the complete console output for diagnosis.

Startup is transactional: failure to acquire the video reader rolls back a newly started stream, and failure to set conservative speed closes the partial session. Cleanup reports both stream and SDK-end failures when both occur. Never work around a disconnect rejection reporting that the SDK still considers the aircraft airborne; establish its physical state and land safely first.

## 3a. Record camera-calibration footage while landed

Print `calibration\checkerboard-9x6-24mm.svg` in A4 landscape at 100%/actual size. Verify that a square measures exactly 24 mm. Keep the Tello stationary with propellers removed, start a recorded manual session, and do not press `T`:

```powershell
python main.py manual --drone-mode tello --record-dir recordings
```

Move and tilt the rigid, flat checkerboard through the center, edges, near, and far parts of the frame. Avoid blur and glare, then press `Q`. Calibrate from that session's `video.mp4`:

```powershell
python main.py calibrate recordings\<session>\video.mp4 calibration\tello-camera.json
```

Keep the accepted JSON with the test evidence. If calibration rejects view diversity or RMS quality, record better footage rather than weakening the thresholds.

Add the accepted path to `drone.local.toml` only after confirming its reported resolution matches Tello video:

```toml
[vision]
calibration_file = "calibration/tello-camera.json"
```

Then rerun `.\scripts\ready.ps1 -Config .\drone.local.toml -Tello`. Autonomous preflight will reject a resolution mismatch before takeoff.

You can evaluate textured, moving footage offline after calibration:

```powershell
python main.py odometry recordings\<session>\video.mp4 `
  --calibration calibration\tello-camera.json `
  --jsonl captures\odometry.jsonl
```

Treat this only as a camera/scene-quality diagnostic. It reports translation direction without metric scale and is intentionally disconnected from flight/navigation.

After calibration, the included 120 mm ArUco marker can be evaluated while the Tello remains stationary and propeller-free. Print `fiducials\marker-0-120mm.svg` on A4 portrait at 100%/actual size, verify the two 120 mm measurements, mount it flat, and record it from several positions. Then run:

```powershell
python main.py fiducials recordings\<session>\video.mp4 maps\example-fiducials.json `
  --calibration calibration\tello-camera.json `
  --jsonl captures\fiducial-poses.jsonl `
  --annotated-video captures\fiducial-annotated.mp4
```

This is an offline camera/localization diagnostic only. The example marker-to-world transform is not a surveyed room placement, and its output is intentionally disconnected from Tello flight and static route execution.

When testing a surveyed map with multiple visible markers, require `pose_consistent=true`. The analyzer withholds the world pose if otherwise valid marker estimates differ by more than 0.25 m or 15 degrees. Fix marker size, flatness, calibration, or surveyed transforms instead of loosening those caps.

Create additional marker sheets with `python main.py make-marker <output.svg> --id <n> --size-m 0.12`. Use a unique ID for each surveyed placement, keep dictionary/size identical to the map entry, print at actual size, and measure every sheet before mounting it.

## 4. Guarded basic-flight acceptance

Reinstall and verify the propellers. Use propeller guards, a large clear indoor area, a level launch surface, one observer, and no people or fragile objects near the aircraft.

Start with the manual console so each action requires a deliberate key press:

```powershell
python main.py manual --drone-mode tello --record-dir recordings
```

Verify live telemetry/video while landed. Press and release `T` once to take off; do not hold it or repeatedly press it. Hold `W/S` for forward/back, `A/D` to turn, and Space/Ctrl to rise/descend; releasing the key returns to hover. `J/K` strafe left/right and `H` commands hover immediately. Then press `L` to land. `Q`, Escape, Ctrl+C, and the window's X button perform orderly landing and cleanup. `E` cuts the motors immediately and will make an airborne drone fall, so reserve it for imminent harm when normal landing is not safe enough.

The manual controller sends zero-velocity RC packets while airborne. Hardware testing on this basic Tello showed that its firmware does not count neutral RC packets toward the inactivity timer and rejects the newer literal `keepalive` command, so manual RC mode also sends and consumes a supported `battery?` query every five seconds. If a takeoff is not confirmed and altitude remains at ground level for five seconds, or an aircraft that was airborne remains at ground level for one second, the local flying state is reset so a later deliberate takeoff is not permanently locked out.

In Windows Tello manual mode, flight keys are polled as held/released controls while the camera window has focus. RC strength is 40, release sends zero velocity, and a 0.75-second deadline prevents a missed key-up from causing indefinite movement. Scheduling is non-blocking so `H`, `L`, or `E` remains responsive. RC movement is not position-measured. The console should show repeated `rc` packets while held and `rc 0 0 0 0` on release.

Do not use `--config config.commissioning.toml` for general manual flying: that diagnostic profile intentionally lands after five seconds. It no longer requires a 90% charge; both normal and commissioning modes allow takeoff from 30% and retain the 15% critical landing protection.

As a separate automated-cleanup check, test takeoff followed by the application's automatic orderly landing:

```powershell
python main.py command "takeoff" --drone-mode tello
```

Then exercise one small primitive at a time:

```powershell
python main.py command "move forward 0.2 meters" --drone-mode tello --takeoff
python main.py command "rotate right 20 degrees" --drone-mode tello --takeoff
```

Confirm direction, yaw convention, altitude telemetry, cleanup, and battery reporting before testing larger distances or compound plans.

During manual/follow/search operation, expect the application to land at 15% battery, after eight minutes airborne, or after three consecutive failed telemetry checks. A watchdog landing appears in console output and, when recording is enabled, in `events.jsonl`. Do not use these software checks as a reason to fly until the battery is depleted; land with a comfortable reserve.

Recorded sessions include `telemetry.snapshot` evidence from each actual 2 Hz watchdog check without additional Tello queries. Confirm landed/flying transitions, altitude, yaw, battery, and any explicit telemetry error align with the command/video timeline.

For an orderly quit while airborne, confirm the session ends with an approved `shutdown.land` event. After an emergency motor stop, confirm no later `shutdown.land` is claimed.

Immediately audit each commissioning session:

```powershell
python main.py audit-session recordings\<session>
```

Do not rely on a session for diagnosis until manifest/events/video/lifecycle checks pass. A telemetry warning can accurately document a link failure; a failed integrity check means the evidence itself is incomplete or inconsistent.

Keep `manifest.json` and `provenance.json` with every session. The audit verifies effective-config, dependency-lock, model/calibration/profile fingerprints and the sidecar hash; note that these are reproducibility fingerprints rather than a signed chain of custody.

## 5. Vision without autonomous movement

Run follow/search without `--takeoff`. The video, YOLO, tracking, and decisions remain visible, while safety rejects movement because the aircraft is landed:

```powershell
python main.py follow --drone-mode tello
python main.py search person --drone-mode tello
```

Confirm low latency, correct left/right interpretation, boxes around the intended targets, `Q`/Escape cleanup, and the `E` emergency path before enabling flight.

## 6. Guarded autonomous acceptance

Only after the previous stages pass:

```powershell
python main.py follow --drone-mode tello --takeoff --record-dir recordings
python main.py search person --drone-mode tello --takeoff --record-dir recordings
```

Start with one person, good lighting, low speed, propeller guards, and an observer ready to press `E` or power down the aircraft. Test target loss, left/right correction, distance correction, orderly quit, and low-battery rejection separately.

Do not run static-room navigation on the aircraft. `python main.py navigate ...` is deliberately simulator-only: the current system has no metric localization, live map registration, or obstacle avoidance. Enabling it on Tello requires a future hardware-validated localization milestone, not a command-line override.

Static maps can still be reviewed safely with `--svg captures\route-preview.svg --route-json captures\route.json`. Verify raw obstacles, orange inflated clearance, start/goal, and the blue route visually, then run only with the simulator if desired.

A simulator execution is accepted only when `route audit: passed=True` confirms the final x/y/yaw matches the mapped goal. This audit deliberately does not apply to Tello: the physical aircraft does not yet expose a validated room pose to compare.

Recorded consensus-approved camera poses can be inspected against a surveyed map with `python main.py map-poses <fiducial.jsonl> <room.json> --svg <trajectory.svg>`. Treat green/red points as offline camera diagnostics only. They are not Tello body positions and cannot authorize static-route flight.

## Record before changing code

For any failure, record:

- the exact command and complete console output;
- battery percentage and whether props were installed;
- Wi-Fi adapter/network state;
- whether telemetry, video, inference, or motion failed first;
- a short video of aircraft behavior if flight occurred;
- the time of the test so logs can be matched.

Recorded MP4 footage can be reproduced later without Wi-Fi, motors, or a drone connection:

```powershell
python main.py replay recordings\problem.mp4 --target person `
  --annotated-video captures\problem-annotated.mp4 `
  --jsonl captures\problem-frames.jsonl
```
