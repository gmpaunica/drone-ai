# Tello protocol comparison and project readiness

This is a separate diagnostic harness, not a replacement for the main controller.
It compares the text SDK against the native binary protocol used by controller
apps. It never starts autonomous following. Ground is the default. Flight requires
both `--flight` and a deliberate `T` press after connecting.

Use PowerShell in the project directory. Run one command at a time. Do not paste
the entire flight sequence and leave it unattended.

## 1. Prepare the computer while normal internet is available

The project's `.venv` already exists, and TelloPy 0.6.0 has been installed in it.
The install command is included for reproducibility; rerunning it is harmless.

```powershell
Set-Location 'C:\path\to\Drone AI'
.\.venv\Scripts\python.exe -m pip install -r requirements-diagnostics.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe main.py doctor --skip-webcam --skip-ollama
```

The doctor loads the existing `yolo26n.pt` and performs an inference without
connecting to the drone. For the separate optional language-command feature:

```powershell
.\scripts\ready.ps1
```

That checks Ollama and the project environment without flying. Ollama is not
involved in these protocol tests; its failure does not explain manual stick failure.

## 2. Establish the working-app baseline

Use TelloFPV as you normally do and note whether takeoff, hovering, all directions,
and landing work. Record the drone model, firmware shown by the app, app version,
battery, floor/lighting, and any drift. Do not change firmware during this comparison.
An external video of the later flight trials is useful for aligning physical motion
with the recorded command times.

Land, exit TelloFPV, and disconnect the phone from Tello Wi-Fi. Close the main
project's manual/follow windows. Power-cycle the grounded drone and connect the
**computer** to its `TELLO-...` Wi-Fi. Keep only one controller connected.

Remove propellers for the ground tests. Use the same well-lit, textured floor for
both protocols. Cool the drone between ground runs; it does not get propeller airflow.

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics environment
```

This reports dependency versions, Wi-Fi, route, and selected local source IP. It
does not transmit commands. If the wrong interface is selected, `--bind-ip` can
select the computer's actual Tello Wi-Fi IPv4 address. Do not put the drone's
`192.168.10.1` address in that option.

## 3. Compare control and telemetry without video processing

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics run --protocol sdk --no-video --seconds 30
```

Power-cycle the grounded drone and reconnect computer Wi-Fi, then:

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics run --protocol native --no-video --seconds 30
```

These runs send only connection/clock/read-only queries and neutral sticks. They
never send takeoff, nonzero sticks, landing, motor cutoff, flips, Wi-Fi changes,
or altitude-limit changes. `--no-video` omits video requests/listening/decoding;
firmware behavior after the native handshake determines whether it still emits video.

This stage tests connection acknowledgement, received packet validity, fresh
telemetry, and an independent 20 Hz neutral-stick transmitter. A text query timing
out or returning `unknown command` is retained as a capability result; it is not
silently interpreted as a successful query. SDK query/response pairing is best
effort because the text protocol has no command sequence identifiers.

## 4. Compare video plus the real project detector

With propellers still removed, stand in view of the camera during each run. Move
left/right and nearer/farther while keeping the drone on the ground. The preview
shows the received camera image; detections and timing go to the evidence log.
Reboot/reconnect before switching protocols, and cool the drone between runs.

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics run --protocol sdk --seconds 30 --preview --model .\yolo26n.pt
```

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics run --protocol native --seconds 30 --preview --model .\yolo26n.pt
```

This exercises UDP video arrival, H.264 decoding, valid resolution, frame freshness,
the project's actual `YOLODetector`, and control timing while inference runs. The
model must already be a local file; no model download is attempted. A run with no
person detected is explicitly incomplete for person-follow readiness.

If this stage fails but stage 3 succeeds, isolate video from inference by repeating
the same command without `--model .\yolo26n.pt`. An input-video stall and a decoder
failure are separate events. The recorded frame gaps measure this receiver's output,
not true camera-to-screen latency. Native packet headers are retained, but this
version does not claim an exact over-the-air video packet-loss percentage.

If a socket cannot bind, another program may own the port. The tool does not reuse
occupied ports or change firewall rules. Permit this project's Python executable
on the Tello connection if Windows asks, then repeat. Do not disable the firewall
globally. Port details: SDK control 8889/state 8890/video 11111; native local control
9000/video 6038; both send control to drone UDP 8889.

## 5. Supervised flight comparison

Only test a protocol in flight after its ground checks work. Refit correct
propellers and guards. Use a clear indoor area, a textured floor and good lighting,
with an observer and an immediately accessible keyboard. Fully charge first;
takeoff is blocked below 30%. Reboot/reconnect between protocols.

Start with a hover-only SDK flight. It permits no movement keys. After 15 seconds,
press `Y` only if it remained physically stable, `N` if it drifted or behaved
unexpectedly, or `U` if uncertain. It then lands.

On this basic Tello, two SDK hover trials auto-landed about 12 seconds after takeoff
even though neutral `rc 0 0 0 0` packets arrived at 20 Hz. The firmware rejects the
newer literal `keepalive` command, but accepts `battery?`. The corrected SDK test
therefore sends and consumes an acknowledged `battery?` query every five seconds.
This specifically tests whether a compatible acknowledged command prevents the
inactivity landing; `sdk_heartbeats` in the summary records those transmissions.

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics run --protocol sdk --flight --hover-only --seconds 60 --preview
```

Do not continue if this fails. Inspect the aircraft/propellers and compare a
hover-only flight in TelloFPV first. When SDK hover passes, repeat hover-only using
the native protocol:

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics run --protocol native --flight --hover-only --seconds 60 --preview
```

Movement tests permit exactly one named direction per flight. Start with forward;
the permitted key is printed and displayed in the preview:

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics run --protocol sdk --flight --axis forward --seconds 60 --preview
```

After both hover tests pass, substitute `native` for `sdk` to compare the same
direction. Then repeat separate flights using `--axis backward`, `left`, `right`,
`yaw_left`, `yaw_right`, `up`, and `down`. Stop the comparison as soon as a result
is wrong or unclear; inspect that run before continuing.

For an axis run:

1. Press `T` once and wait for actual takeoff plus ten seconds of stable hover.
2. Tap the single displayed movement key once. It sends 10% stick for 0.20 seconds,
   followed by neutral packets.
3. Press `Y` only if direction and stopping were correct, `N` if wrong, or `U` if
   unclear. Any answer requests landing and ends that flight.
4. `H` or Space sends neutral immediately. `L`, `Q`, Escape, window close, and
   Ctrl+C request landing. Verify physical touchdown and motor stop.

Keys work in the terminal or preview. Tap and release; keyboard auto-repeat is
suppressed. A blocked or wrong key produces an evidence event and no nonzero packet.
If 10% for 0.20 seconds is too small to see, record `U`; do not increase it ad hoc.

There is one takeoff per run. The flight watchdog requests landing after 120 seconds
from the takeoff request, at 15% battery, at a 1.5m telemetry height, when telemetry
is older than one second, after a control scheduler stall above 0.25 seconds, after
sustained unexpected ground contact, or when SDK pitch/roll reaches 15 degrees.
Upward pulses are blocked at 1.2m; downward pulses are blocked below 0.5m.
These are software intervention thresholds, not guaranteed physical flight bounds.

Takeoff is sent once. Landing attempts continue during cleanup even if takeoff's
send/acknowledgement failed, because aircraft state is then uncertain. Neutral
sticks and landing use independent scheduling from the UI/decoder. The console
warns if fresh telemetry does not establish landing within the cleanup interval.
The native test also requires the aircraft's ground/flying flags to agree; SDK
landing evidence is limited to sustained low altitude and needs visual confirmation.

Do not deliberately disconnect Wi-Fi, crash the process, test low-battery exhaustion,
or test motor cutoff in flight. A process that is killed or a lost radio link cannot
guarantee software landing. Their software paths are exercised with fake transports.
There is no emergency motor-cutoff key in this diagnostic console.

## 6. Read and return the results

```powershell
.\.venv\Scripts\python.exe -m drone_ai.diagnostics report
```

Each run creates a unique directory under `recordings\diagnostics\` containing:

- `session.json`: protocol, mode, firmware/environment notes, source/model hashes,
  dependency and network information, terminal errors and video counters.
- `events.jsonl`: raw sent/received control packets, decoded telemetry, monotonic
  timestamps, transmission gaps, video packet/frame events, detector results,
  pulse requests, visual observations and landing evidence.
- `summary.json`: pass/incomplete reasons, median/p95/max gaps, video and detector
  results, eight-direction coverage, pulse-to-neutral timing and SDK yaw/height changes.
- `video.h264`: received compressed video, when enabled. Packet arrival order is
  retained; dropped local queue data and decoder errors are reported separately.

Keep the whole directory. Start by sharing the `summary.json` files and
`session.json` files from both protocols. Retain events/video for detailed analysis.
The report exits with code 1 when a run failed or coverage is incomplete; that is a
diagnostic result, not necessarily a Python/programming error. Earlier failed runs
remain in the combined report. To inspect one run, supply its directory to `report`.

Use `--note "..."` on any run to record firmware/app version, floor, lighting,
battery, baseline behavior, or why a stage was repeated.

## What conclusions these tests support

| Result | Next investigation |
| --- | --- |
| Native works; minimal SDK fails | Native transport/firmware compatibility; integrate a native adapter next |
| Both minimal protocols work; original app controls fail | Main app scheduling, input mapping, state handling and command semantics |
| Neither laptop protocol works; Android works | Laptop route, firewall, Wi-Fi adapter, connection ownership; preserve raw replies |
| Telemetry works; no video packets | Video request/port/routing/firewall path |
| Video packets arrive; decoder produces no fresh frames | H.264 framing, loss, decoder compatibility |
| Ground video works; inference causes timing gaps | Compute scheduling, inference budget and frame backpressure |
| A pulse is sent but the aircraft does not move as expected | Aircraft mode/status, control semantics, telemetry and external flight video |

A pass means the **requested stage's checks** passed. A no-video ground pass is
not a flight/video/AI pass. Control timing budgets are diagnostic acceptance limits,
not manufacturer-certified guarantees. Automatic summaries never treat a sent
command as measured motion. Native height/velocity/status flags are decoded;
native yaw, log-derived position, and exact displacement are not yet decoded or
calibrated. Raw native packets are saved to support that work.

This suite establishes transport, manual control and perception feasibility. Before
autonomous following, we still need to integrate the chosen transport into the
existing safety/controller interfaces, validate native yaw if chosen, measure real
movement and braking, and test target loss/manual override in that integrated app.
Room navigation additionally needs calibrated metric localization and obstacle
handling; the existing simulator navigation does not establish those capabilities.

Protocol references: [TelloFPV developer explanation](https://tellopilots.com/threads/new-app-tellofpv-for-android.2710/page-151),
[TelloPy source](https://github.com/hanyazou/TelloPy),
[native telemetry units](https://pkg.go.dev/gobot.io/x/gobot/v2/platforms/dji/tello#FlightData),
[Ryze SDK 1.3](https://dl-cdn.ryzerobotics.com/downloads/tello/20180910/Tello%20SDK%20Documentation%20EN_1.3.pdf).
