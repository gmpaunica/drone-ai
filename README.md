# Drone AI

For the SDK-versus-TelloFPV/native control investigation, start with
[DIAGNOSTICS.md](DIAGNOSTICS.md). It includes separate ground, video/AI workload,
and supervised flight tests with packet-level evidence and exact PowerShell commands.

A safety-first Python 3.11 prototype that follows the largest visible person through this vertical slice:

`CameraSource -> YOLO detector -> person tracker -> follow decision -> SafetyController -> DroneController`

The simulator is the default and requires no flight hardware. DJI/Ryze Tello flight and video are optional, lazily imported adapters. The current scope intentionally excludes UI beyond the OpenCV preview, identity tracking, SLAM, and physical room navigation.

Milestone 2 adds optional local-LLM interpretation through Ollama. It translates natural language into the same strict command schema; it cannot call a drone or bypass `SafetyController`.

Milestone 3 adds bounded compound plans. Plans contain at most eight primitive steps, cannot queue takeoff or emergency/high-level abilities, and have aggregate budgets of 6 m movement and 360 degrees rotation. Every step receives a fresh timestamp and passes through `SafetyController`; execution aborts on the first rejection/failure and requests a system hover when the drone is still airborne.

Milestone 4 adds bounded visual search. `find_person` and `find_object` select a configured YOLO class, rotate clockwise in 20-degree safety-gated steps, stop as soon as the target is visible, and stop with `NOT_FOUND` after at most 360 degrees. Only approved rotations count toward the scan. Unsupported target names are rejected against the model's class list before connecting or taking off. This is a visual sweep, not identity tracking, mapping, or autonomous room navigation.

Milestone 5 adds fail-closed autonomous preflight and a non-flying system doctor. Follow/search will not take off until connection state, landed state, battery, altitude/yaw telemetry, target classes, a three-channel camera frame, and one detector warm-up inference all pass. The doctor verifies the installed environment and can commission Tello telemetry/video without ever sending takeoff or movement.

Milestone 6 adds a manual commissioning console. Simulator/manual commands use bounded steps; Tello manual mode uses held-key RC control with a bounded fallback timeout. Motion dispatch is rate-limited to 2 Hz, manual override is enabled, and every normal key action still routes through `SafetyController`. Emergency retains its dedicated immediate path. The simulator uses a generated status frame, so this mode can be exercised without any camera.

Milestone 7 adds hardware-free video replay. A finite video-file camera feeds the same YOLO conversion and target tracker without constructing any drone controller. Replay is headless by default, supports a frame bound, can write annotated MP4 plus strict per-frame JSONL evidence, and refuses to overwrite either its input or existing outputs unless explicitly authorized.

Milestone 8 adds opt-in session evidence for manual, follow, and search modes. Each run gets a collision-safe UTC/UUID directory with flushed JSONL control events, annotated MP4 frames, and a final manifest. Recording errors abort the active loop so orderly landing can run. Nothing is recorded unless `--record-dir` is supplied; recordings may contain identifiable people and remain git-ignored.

Milestone 9 adds an airborne watchdog. Manual/follow/search poll flight health at 2 Hz, request an orderly system landing at 15% battery or after eight minutes airborne, and attempt a dedicated fail-safe landing after three consecutive telemetry failures. Critical/unknown in-flight battery also blocks new movement and rotation in `SafetyController`. The watchdog never substitutes an automatic emergency motor cutoff for landing.

Milestone 10 adds validated TOML profiles. Every section can be partially overlaid without editing Python, while unknown keys, wrong types, non-finite values, remote Ollama URLs, and attempts to weaken hard safety caps are rejected before connecting. CLI-specific camera/model/Ollama overrides still take precedence.

Milestone 11 adds a known-good Windows/Python 3.11 dependency lock plus idempotent PowerShell setup/readiness scripts. Setup creates the virtual environment if necessary, installs exact verified packages, checks dependencies, runs the full test suite, prepares YOLO/Ollama models, and runs the non-flying doctor. Daily readiness starts the local Ollama service when needed and never invokes a flight command.

Milestone 12 adds offline camera calibration for later visual odometry/SLAM work. It detects a checkerboard across recorded video, ignores near-duplicate stationary views, requires diversity in position/scale/rotation, enforces a maximum reprojection RMS error, and writes a strict camera-intrinsics JSON artifact. Calibration never constructs a drone controller.

Milestone 13 adds opt-in lens-distortion correction. Replay/manual/follow/search can wrap any selected camera with a strict calibration artifact while preserving BGR resolution. A resolution mismatch is an explicit camera error; in autonomous modes it is caught by preflight before takeoff. The doctor validates a calibration configured in the TOML profile.

Milestone 14 adds offline-only monocular visual-odometry analysis. It uses calibrated/undistorted frames, ORB ratio-tested matches, RANSAC essential matrices, and pose recovery to report relative rotation plus translation direction. Every record hard-codes `metric_scale_available=false`; no odometry result is connected to planning, safety, or a drone controller.

Milestone 15 hardens the Tello video/telemetry handoff. The camera waits up to five seconds for a non-placeholder frame of at least 640×480, returns an owned copy rather than the SDK background buffer, and fails explicitly if the reader stops or regresses. Preflight rejects a supposedly landed aircraft above 0.3 m, while invalid fractional/out-of-range battery telemetry fails at the adapter boundary.

Milestone 16 adds deterministic static-room route planning for the simulator. Strict JSON maps describe a bounded grid, fixed obstacles, resolution, and an obstacle/wall clearance radius. A turn-aware A* search chooses a shortest path while preferring less rotation, then converts it into the same bounded `ActionPlan` primitives used elsewhere. Preview is the default, execution requires both `--execute` and `--takeoff`, and the CLI deliberately exposes no Tello mode. It is not localization, live obstacle avoidance, SLAM, or authorization to fly a physical route.

Milestone 17 adds calibrated ArUco fiducial analysis for recorded video. A strict marker map supplies measured marker sizes and rigid marker-to-world transforms; accepted detections report metric marker-to-camera and camera-to-world transforms with reprojection error. Unmapped, undersized, poor-quality, or resolution-mismatched observations fail closed. The command is finite, evidence-producing, and offline-only: it exposes no drone mode and its records cannot be relabeled as live-flight data.

Milestone 18 adds multi-marker pose consensus. When a frame sees more than one mapped marker, every individually accepted camera-to-world estimate must agree within 0.25 m and 15 degrees before the frame publishes a world pose. These caps can only be tightened in TOML. A disagreement preserves the individual observations for diagnosis, marks the frame inconsistent, and withholds `best_camera_to_world` rather than silently trusting the lowest reprojection error.

Milestone 19 adds deterministic printable-marker generation. OpenCV's installed dictionary bytes are converted directly into an A4 portrait SVG with a metric black square and matching verification line. Supported IDs and 30-160 mm page-fit bounds are validated, the output must be `.svg`, and existing files are preserved unless `--overwrite` is explicit. This utility has no camera, drone, or takeoff surface.

Milestone 20 adds auditable static-route artifacts. `navigate` can write an SVG that distinguishes raw obstacles, inflated clearance, the chosen path, and start/goal cells, plus a strict JSON snapshot of the immutable route and bounded action plan. Map +y is drawn upward. Preview remains the default, outputs are collision-safe, and both artifacts retain the simulator-only boundary.

Milestone 21 closes the simulator route loop. Execution initializes simulator x/y at the start-cell center and, after every safety-gated plan step, independently compares actual x/y/yaw with the mapped goal and final path heading. CLI success now requires both plan completion and sub-micrometer/sub-microdegree deterministic pose agreement. The immutable audit's pass flag is derived from its errors and cannot shed its simulator-only marker.

Milestone 22 adds offline static-map pose auditing. Room maps now define a world-space lower-left grid origin and clockwise grid yaw; route planning, simulator initialization, and route auditing all honor the same registration. Consensus-approved fiducial camera poses can be classified as safe, clearance-blocked, outside, or missing and exported as strict JSONL plus an SVG trajectory. The records are immutable/offline-only and are not consumed by planning or flight.

Milestone 23 adds periodic telemetry evidence to opt-in live recordings. Each actual 2 Hz watchdog read is cached as a strict `telemetry.snapshot` event containing connection/flying state, altitude/yaw/battery/speed, simulator x/y when available, or an explicit read/validation error. Rate-limited polls produce no duplicate and recording never performs an extra controller read. A watchdog landing remains ordered before its final cached snapshot.

Milestone 24 closes the recorded flight lifecycle. If orderly cleanup finds the aircraft connected and flying, its system landing result is persisted as `shutdown.land` before disconnection and manifest closure. An already-landed or emergency-stopped aircraft does not receive a fictitious landing event, preserving the distinction between controlled landing and immediate motor stop.

Milestone 25 adds an offline session-integrity auditor. It validates safe manifest child paths, strict contiguous/time-bounded JSONL events, cached telemetry schemas/sequences, decoded BGR video frame count/resolution, and the approved takeoff-to-landing/emergency lifecycle. Captured telemetry errors are warnings because they may be genuine evidence; corruption, path escape, frame-count mismatch, or an unterminated takeoff fails the audit.

Milestone 26 adds versioned session provenance. CLI-created manual/follow/search recordings include `provenance.json` with Python/platform, selected adapters/model/calibration/profile, a canonical effective-config SHA-256, and hashes of every available dependency lock/model/calibration/profile file. The manifest hashes the sidecar and the auditor verifies it. These are reproducibility/tamper-evidence fingerprints, not a cryptographic signature or legal chain of custody.

Milestone 27 hardens optional Tello lifecycle failures. Frame-reader acquisition rolls back a stream it just started, speed-setup failure closes the partial connection, repeated connect/stream/stop/close operations are idempotent, combined stream/end cleanup errors are preserved, and disconnect checks the SDK's cached flying state in addition to the controller flag. Fake-SDK coverage exercises these paths without Wi-Fi, motors, or hardware.

## Safety model

Every parsed or ability-generated flight action passes through `SafetyController`. It rejects stale or malformed commands, unsafe distances/rotations/altitudes, disconnected or landed movement, and autonomous sources while manual override is enabled. Emergency stop has a dedicated immediate path. The defaults are:

- Altitude: 0-3 m; takeoff: 1 m
- Movement: 0.2-3 m per command
- Rotation: at most 180 degrees
- Cruise speed: 0.2 m/s; hard configured maximum: 0.5 m/s
- Command freshness: 2 seconds
- Minimum takeoff battery: 30%; unknown battery telemetry blocks takeoff
- Critical battery: 15%; watchdog landing and movement lockout
- Maximum supervised airborne session: 8 minutes
- Repeated telemetry loss: fail-safe landing after 3 consecutive 2 Hz checks
- Landed preflight altitude: at most 0.3 m
- First Tello video frame: non-placeholder, at least 640×480, within 5 s
- Follow loop: 2 Hz, 10-degree turns, 0.3 m forward/back steps
- Horizontal dead zone: +/-0.12; good target-size ratio: 0.08-0.28
- Detector: `yolo26n.pt`, confidence 0.50

Simulator coordinates are `+x` right, `+y` forward, and `+z` up. Yaw is clockwise-positive and normalized to `[0, 360)`.

This is experimental software, not a certified flight-control system. Keep a clear guarded area, retain direct access to emergency controls, and follow local aviation/privacy rules. Test Tello connectivity and video with propellers removed before any flight test.

## Setup

Python 3.11 is required. On PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

That is the recommended reproducible path and uses `requirements-lock.txt`. Pass `-UseVersionRanges` only when intentionally refreshing dependencies, followed by full verification and an updated lock.

The equivalent manual setup is:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

For Tello support, additionally install:

```powershell
python -m pip install -r requirements-tello.txt
```

### Optional local LLM

Install the Ollama Windows application separately, then download the default model while normal internet access is available:

```powershell
ollama pull qwen3:4b-instruct
```

The default 4B quantized model is about 2.5 GB and is a practical starting point for this machine's GTX 1060 6 GB. Ollama serves its local API at `http://localhost:11434` by default. No Python Ollama package is required because this project uses that local HTTP API directly.

Named YOLO weights are downloaded by Ultralytics on the first live inference run. Tests use fakes and never need a network, weights, a camera, or a drone.

Run the non-flying readiness report at any time:

```powershell
python main.py doctor
python main.py doctor --json
```

To create a machine-specific profile without modifying code:

```powershell
Copy-Item config.example.toml drone.local.toml
python main.py --config drone.local.toml doctor
python main.py --config drone.local.toml manual
```

`--config` is a global option and therefore comes before `doctor`, `manual`, `follow`, or another mode. Local profiles matching `*.local.toml` are git-ignored. The complete documented template is [config.example.toml](config.example.toml); profiles may omit unchanged sections and keys. Safety values may be made more conservative but cannot raise the 3 m altitude, 3 m movement, 180-degree rotation, 0.5 m/s speed, eight-step plan, or eight-minute airborne caps.

If the Ollama check reports connection refused, launch the Ollama application or run `ollama serve` in another terminal. A missing webcam is a warning because Tello supplies its own camera. After joining the powered-on Tello's Wi-Fi, commission its telemetry and video—still without takeoff—with:

```powershell
python main.py doctor --tello
```

See [HARDWARE_CHECKLIST.md](HARDWARE_CHECKLIST.md) for the guarded first-day sequence.

For a daily non-flying check that also starts Ollama if needed:

```powershell
.\scripts\ready.ps1
```

After joining the Tello Wi-Fi, the same script can commission telemetry/video without takeoff:

```powershell
.\scripts\ready.ps1 -Tello
```

## Run

Open the manual simulator console:

```powershell
python main.py manual
```

Keys are `T` takeoff (press once), `L` land, hold `W/S` to move forward/back, hold `A/D` to turn, hold Space/Ctrl to rise/descend, `J/K` to strafe left/right, `H` to hover immediately, `E` for immediate motor cutoff, and `Q`/Escape for orderly landing and quit. Releasing a held flight key commands hover. Closing the camera window or pressing Ctrl+C also uses orderly cleanup. For the Tello camera and flight adapter:

```powershell
python main.py manual --drone-mode tello
```

There is intentionally no automatic-takeoff flag in manual mode; the operator must press `T` after seeing live telemetry/video.

While the Tello is airborne, the adapter sends a zero-velocity RC heartbeat at 2 Hz. This keeps the command link active without blocking keyboard handling. The optional `config.commissioning.toml` profile uses the normal 30% takeoff threshold but intentionally limits a diagnostic hover to five seconds; omit that profile for normal manual flight.

On Windows, Tello manual mode polls the actual held-key state only while the camera window has focus. It sends controller-style RC input at strength 40 while a key is held and zero velocity when it is released. The 0.75-second scheduler deadline is only a fail-safe if a release is missed. The camera/key loop remains active throughout, so `H`, `L`, and `E` remain responsive. RC input is open-loop and not a measured distance. Rule/ability commands outside manual mode retain SDK distance commands.

Opt in to a diagnostic recording when commissioning:

```powershell
python main.py manual --drone-mode tello --record-dir recordings
python main.py follow --drone-mode tello --takeoff --record-dir recordings
python main.py search person --drone-mode tello --takeoff --record-dir recordings
```

Each invocation creates a new directory; it never reuses an earlier session name. Treat captured video as sensitive personal data and delete or retain it according to the consent/privacy rules that apply where you operate.

CLI-created sessions also contain `provenance.json`. It records names/paths and SHA-256 digests, never copies model weights or calibration contents. Manual mode correctly leaves YOLO fields empty; follow/search fingerprint the local weight file when the configured model resolves to one. The manifest links the sidecar with its own digest.

`events.jsonl` also contains `telemetry.snapshot` records at the watchdog's 2 Hz check rate. The watchdog sequence is independent of the enclosing session-event sequence; `checked_at_monotonic_s` is process-relative while each enclosing event retains a UTC `recorded_at`. A snapshot with `error` instead of `state` proves the telemetry failure without a second SDK query.

When orderly cleanup actually lands an airborne aircraft, the final lifecycle evidence includes `shutdown.land` with the full `SafetyResult`. Emergency stop remains its own earlier command/event and intentionally produces no later landing claim.

Audit any completed session without connecting to a camera or drone:

```powershell
python main.py audit-session recordings\<session>
python main.py audit-session recordings\<session> --json
```

The audit decodes the recorded video rather than trusting container metadata, so large sessions can take a little time. `WARN` does not fail integrity (for example, an explicitly captured telemetry-link error); any `FAIL` makes the command exit nonzero. Keep the original directory unchanged when sharing evidence for diagnosis.

The provenance check detects accidental edits or sidecar tampering as long as the manifest is retained. It is not signed, so it cannot prove who created a session or resist an attacker who rewrites both files.

Analyze recorded footage without connecting to a drone:

```powershell
python main.py replay recording.mp4 --target person
python main.py replay recording.mp4 --target person `
  --annotated-video captures\person-annotated.mp4 `
  --jsonl captures\person-frames.jsonl
```

Add `--display` for an interactive preview, `--max-frames N` for a bounded test, or `--overwrite` only when replacing known replay outputs. `captures/` and `recordings/` are git-ignored.

Calibrate a camera from recorded checkerboard footage:

```powershell
python main.py calibrate recordings\calibration.mp4 calibration\tello-camera.json
```

The default target has 9×6 inner corners (10×7 physical squares), with each square exactly 24 mm. Print [checkerboard-9x6-24mm.svg](calibration/checkerboard-9x6-24mm.svg) in A4 landscape at 100%/actual size and verify a square with a ruler. Record many sharply focused views while moving and tilting the board across the entire image. The command samples every fifth frame, requires at least eight diverse accepted views, and rejects RMS error above 1 px. Override board geometry only when using a physically different measured target.

The resulting JSON is groundwork for future visual odometry; follow/search does not require or silently apply it unless a calibration path is explicitly selected.

Apply a verified calibration explicitly:

```powershell
python main.py replay recording.mp4 --calibration calibration\tello-camera.json
python main.py follow --drone-mode tello --calibration calibration\tello-camera.json --takeoff
```

For persistent use, set `vision.calibration_file = "calibration/tello-camera.json"` in a local TOML profile. The wrapper requires the live/recorded frame dimensions to match the calibrated dimensions exactly and precomputes OpenCV remap tables once per opened camera.

Assess whether recorded footage has enough visual texture and parallax for later odometry work:

```powershell
python main.py odometry recording.mp4 `
  --calibration calibration\tello-camera.json `
  --jsonl captures\odometry.jsonl
```

This command is deliberately offline and exits nonzero when no frame pair meets the configured match/inlier thresholds. Monocular translation has unknown scale, so the output is not a metric position, SLAM map, obstacle detector, or navigation authorization.

Test metric fiducial localization on recorded calibrated footage:

```powershell
python main.py fiducials recordings\marker-test.mp4 maps\example-fiducials.json `
  --calibration calibration\tello-camera.json `
  --jsonl captures\fiducial-poses.jsonl `
  --annotated-video captures\fiducial-annotated.mp4
```

The example map matches [marker-0-120mm.svg](fiducials/marker-0-120mm.svg), a `DICT_4X4_50` marker with ID 0 and a 120 mm black square. Print it on A4 portrait at 100%/actual size and verify both the black square and printed check line measure exactly 120 mm. Mount it flat and rigid with a clear white margin. The example identity transform is only a coordinate demonstration; replace `marker_to_world` with a surveyed, right-handed rigid transform before treating output as a room-relative pose.

Generate distinct sheets for a multi-marker map without manually copying bit patterns:

```powershell
python main.py make-marker fiducials\marker-1-120mm.svg --id 1 --size-m 0.12
python main.py make-marker fiducials\marker-2-120mm.svg --id 2 --size-m 0.12
```

Pass `--dictionary` when the marker map uses something other than `DICT_4X4_50`. The dictionary name, ID, and measured black-square size must match the JSON map exactly; printing with “fit to page” enabled invalidates the metric pose.

Transforms use column vectors and the `source_to_destination` naming convention. Marker axes are +x right, +y up, and +z out of the printed front face; OpenCV camera axes are +x right, +y down, and +z forward. Fiducial results remain disconnected from static-route execution and all flight controllers until physical-camera accuracy and room registration can be validated.

If multiple mapped markers appear together, inspect `pose_consistent`, `maximum_translation_disagreement_m`, and `maximum_rotation_disagreement_degrees` in each JSONL record. A detected marker can have an individually good pose while the frame correctly reports `metric_pose_available=false` because the surveyed marker placements disagree.

Project accepted fiducial camera poses into the registered static room map:

```powershell
python main.py map-poses captures\fiducial-poses.jsonl maps\example-room.json `
  --jsonl captures\mapped-camera-poses.jsonl `
  --svg captures\camera-trajectory.svg
```

`world_origin_x_m` and `world_origin_y_m` locate the grid's lower-left outer corner in the fiducial map's world frame. `grid_yaw_degrees` rotates grid axes clockwise in that frame. The supplied zero-origin/zero-yaw example is only a schema demonstration; survey the shared frame before interpreting cell labels. Green SVG points are clearance-safe camera positions, red points are in inflated clearance, and missing/outside records remain in JSONL even when they cannot be plotted.

This is still not physical navigation. The trajectory is the camera optical-center position, not a validated Tello body pose; camera-to-body extrinsics, timing, live confidence, and obstacle sensing remain unresolved. `map-poses` exposes no drone/takeoff options and nothing routes its output to `SafetyController`.

Preview a clearance-inflated route on the included example map:

```powershell
python main.py navigate maps\example-room.json --start 1,1 --goal 10,8 `
  --svg captures\route-preview.svg `
  --route-json captures\route.json
```

Execute the generated plan only in the simulator:

```powershell
python main.py navigate maps\example-room.json --start 1,1 --goal 10,8 --execute --takeoff
```

Successful execution prints `route audit: passed=True` before orderly landing. The simulator state uses the metric center of the start/goal grid cells, so its final x/y is directly checkable against the map resolution. An audit failure makes the command exit nonzero even if all plan steps reported completion.

Map coordinates are integer grid cells. The route planner enforces the existing eight-step, 6 m movement, and 360-degree rotation budgets; start/goal cells inside inflated walls or obstacles are rejected. This mode has no `--drone-mode` option and cannot control a Tello because a static map provides neither live pose nor obstacle sensing.

Open the SVG before simulation and confirm the orange clearance area matches the intended physical margin, the blue path never crosses orange/dark cells, and green/red are the intended start/goal. Existing artifacts are never replaced unless `--overwrite` is supplied.

Parse and execute one command in the simulator:

```powershell
python main.py command "move forward two meters" --takeoff
```

Use the local LLM for freer single-command wording:

```powershell
python main.py command "please move the drone forward by 0.5 meters" --interpreter ollama --takeoff
```

Or use deterministic rules first and only fall back to Ollama for unknown wording:

```powershell
python main.py command "please move the drone forward by 0.5 meters" --interpreter hybrid --takeoff
```

The rule interpreter remains the default. The LLM response is constrained with a JSON schema, revalidated by Pydantic, tagged with source `AI`, and then passed through `SafetyController`. Model-generated emergency stops are refused; use the explicit `emergency stop` rule command or the `E` key. Compound and parameterless commands are rejected before hybrid fallback.

Preview a deterministic compound plan (preview never connects to a drone):

```powershell
python main.py plan "move forward one meter then turn right ninety degrees then move forward 0.5 meters"
```

Execute a local-LLM plan in the simulator:

```powershell
python main.py plan "please move forward 0.5 meters, rotate right 90 degrees, and move forward 0.5 meters" --planner ollama --execute --takeoff
```

`--execute` is always required for plan execution. `--takeoff` remains a separate operator preflight; takeoff and emergency stop are never queued inside a plan.

Search directly for a COCO/YOLO class:

```powershell
python main.py search person --takeoff
python main.py search backpack --takeoff
```

The same abilities are routed from high-level commands:

```powershell
python main.py command "find person" --takeoff
python main.py command "look for a backpack" --takeoff
python main.py command "follow me" --takeoff
```

Search is limited to one full turn and returns to orderly landing on completion, cancellation, or Ctrl+C. `E` remains the immediate emergency path.

Start person following with a webcam:

```powershell
python main.py follow --takeoff
```

Preview without takeoff (decisions remain visible and movement is rejected as landed):

```powershell
python main.py follow
```

Use Tello flight and camera on the Tello Wi-Fi network:

```powershell
python main.py follow --drone-mode tello --takeoff
```

`--drone-mode tello` selects both Tello adapters unless `--camera webcam` is supplied. Press `E` for immediate emergency motor stop. Press `Q`, Escape, or Ctrl+C for an orderly landing and cleanup. Frames are not mirrored, so left/right follows the camera image directly.

## Commands

The conservative rule parser accepts one action at a time: takeoff, land, hover/stop, emergency stop, directional movement with a distance, left/right rotation with degrees, turn around, follow me, find person, and find/look for an object. It supports digits and common English number words. Unknown, ambiguous, and parameterless commands are rejected instead of guessed. Compound requests go through the separate bounded `plan` mode.

## Tests

```powershell
python -m pytest
```

The suite covers the locked environment, strict TOML overlays, synthetic camera calibration/undistortion/relative-pose geometry, the simulator, schema/parser, safety, vision conversion and tracking, follow policy, manual key mapping/rate limiting, finite video replay, collision-safe live session evidence, Tello mappings with a fake SDK client, fake-camera integration, local-LLM structured output, bounded plan parsing/execution, autonomous preflight ordering, and non-flying Tello commissioning.

The local development environment has the live dependencies installed and `yolo26n.pt` downloaded (weights remain git-ignored). The installed PyTorch build runs YOLO on CPU; Ollama runs `qwen3:4b-instruct` fully on the GTX 1060. No webcam is currently available at index 0, and physical Tello acceptance remains pending hardware.

## License

Copyright (C) 2026 Drone AI contributors. Licensed under the GNU Affero General Public License v3.0 or later. Ultralytics is distributed under AGPL-3.0; commercial or closed-source deployment requires a separate licensing review.
