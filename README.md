# trainit_cell_io — simulation control of a cell's machines

TrainIt's machine side of a **machine service** in simulation: the station that a robot task asks to work (`request`)
and waits for (`wait for: done | ready`). It runs the station's devices and sequence, and serves the machine-service
contract to the robot through `trainit_interfaces/ChannelRequest` / `ChannelStatus`, on a deterministic **mock** plant
or on an **Isaac Sim** articulation. The same station logic runs on both; only the plant adapter differs.

**What it is not.** Not a PLC and not a PLC runtime (an external PLC is `control_mode: external_plc`, served by a PLC
adapter in a later milestone). Not a safety function: no signal of this package is a safety signal (ADR-0033). Not part
of the robot: it is a **cell-level process**, started apart from the robot's runner; a robot bundle never launches it.
The robot application names a machine service by its logical id only (`turner`); this package's `cell_io.yaml` says
what serves it.

**Status: version 0.1.0** (TrainIt P0.b, the simulated machine handshake): the pure core (devices, `python_logic`
sequences, the `ProcessDevice` per service, the scan clock in integer physics steps, the mock plant with fault
injection), the ROS side (`cell_io_node`, `cell_io_bridge`, `cell_io_cli`, two launch files), the Isaac plant (a pure
adapter, the Kit glue and transport) and the PUBLIC neutral fixture (`examples/neutral_station/`: a turn station and a
line indexer, mock and Isaac).

**Stability.** At 0.1.0 the public surface is: the format of `cell_io.yaml` (below), the `PlantAdapter` protocol
(`trainit_cell_io/core/plant.py`), the `Sequence` protocol and `Cell(..., sequences=...)` of a `logic: code`
service (`trainit_cell_io/core/logic.py`, `core/cell.py`) and the Isaac adapter's `JointAccess` protocol
(`trainit_cell_io/isaac/plant.py`),
the ROS topics (the machine-service contract and `/cell_io/control` / `/cell_io/state`) and the `cell_io_cli`
command line. Bound to Isaac Sim 5.1 and free to change without notice: `trainit_cell_io/isaac/kit.py`,
`isaac/kit_ros.py`, `isaac/loop.py` (and with them the `/cell_io/kit/...` boundary between the stage and the bridge),
the `isaac:` binding block of `cell_io.yaml` and everything under `examples/`. The Kit runner
(`examples/neutral_station/run_neutral_station.py`) is an example of how a stage serves the cell, not an API.

**Scope.** One cell per ROS domain: `/cell_io/control`, `/cell_io/state` and `/cell_io/kit/...` are cell-level topics,
not qualified by a robot slot, so two cells on one domain would collide (only the machine-service topics live in the
robot's slot). `/cell_io/control` is unauthenticated: anything on the domain may reset a station, inject a fault or
restart the controller. It is SIMULATION control only -- the operator's and the test's tool, never the robot's, and
never a way to command a real machine.

## How it fits

```text
 TSA (setup assistant)            TMR (motion runtime)                 trainit_cell_io (this repository)
 machine_services,           ->   RequestMachine / WaitForMachine  <->  one ProcessDevice per machine service
 request / wait steps             (liveness, bounded waits, the          its sequence (python_logic | code)
 (logical ids only)               MACHINE states the operator reads)     its devices -> a PlantAdapter:
                                                                           MockPlantAdapter  | IsaacPlantAdapter
```

- **Machine services.** The application author declares a machine service by a logical id (`turner`) and writes
  `request` / `wait for: done | ready` steps in the setup assistant (TSA); never a topic, a PLC address or an I/O. The
  generated bundle runs on TrainIt's motion runtime (TMR), whose `RequestMachine` / `WaitForMachine` speak the
  machine-service contract (`trainit_interfaces/ChannelRequest` / `ChannelStatus`). This package is the other side of
  that contract in simulation; the bundle never launches it (the cell is started apart, as a real machine would be).
- **`ProcessDevice`.** Each service of `cell_io.yaml` is served by one `ProcessDevice`: it consumes the robot's request
  image, runs the station's sequence on the station's devices (binary actuators, position axes, digital sensors) and
  publishes the status image -- READY / BUSY / DONE / FAULT / HOMING, a heartbeat minted by its own scan, the identity
  of the request its cycle serves.
- **`MockPlantAdapter`.** A deterministic plant in the process, with no simulator: device motions in integer physics
  steps and fault injection (a stuck actuator, a sensor fault, a held station). The plant of the tests and of a first
  run on any machine.
- **`IsaacPlantAdapter`.** The same devices bound to the joints of an Isaac Sim articulation: a command drives a joint,
  and DONE comes from the joint's read-back. The station's logic is identical on both plants; only the adapter
  changes.
- **Not in this repository.** No PLC runtime and no vendor transport: no CODESYS, OPC UA, Modbus, ADS or fieldbus
  code, and no PLC-side library. In a real cell the machines are served by their PLC through a PLC adapter -- a later
  TrainIt milestone, outside this repository. No safety function (above).

## Install

In a ROS 2 Humble colcon workspace, next to `trainit_interfaces` (>= 0.5.0):

```bash
cd ~/trainit_ws/src
git clone https://github.com/LearnRoboticsWROS-Lab/trainit_cell_io.git
cd ~/trainit_ws && rosdep install --from-paths src --ignore-src -y
colcon build --packages-select trainit_interfaces trainit_cell_io
source install/setup.bash && ros2 pkg prefix trainit_cell_io
```

The core alone, without ROS (a test environment, a simulator's own Python), needs only PyYAML -- e.g. in a venv:

```bash
cd trainit_cell_io                               # this repository
pip install .                                    # PyYAML; no ROS, no simulator
cd /tmp && python3 -c "import trainit_cell_io.core as c; print('trainit_cell_io core', c.Cell.__module__)"
```

## Run it on the mock plant

```bash
ros2 launch trainit_cell_io cell_io.launch.py \
    config:=$(ros2 pkg prefix trainit_cell_io)/share/trainit_cell_io/examples/neutral_station/cell_io_mock.yaml
```

`cell_io_node` serves every service of the file on wall time and logs, when ready,
`cell_io: serving 2 machine service(s) (turner, indexer) on the mock plant ... -- ready`. Start it before (or after)
the robot's task server; the robot's first `request` synchronises on the status heartbeat either way. Stop it with
Ctrl-C (an orderly stop: a restarted cell is a NEW source session that never executes a request image of the old one).

Exit codes: 0 after Ctrl-C / SIGTERM, or for a valid `control_mode: external_plc` file (nothing to simulate); 2 when
the file is refused (an unknown key, a wrong type, a non-finite number, a duration that is not a whole number of
physics steps, `external_plc` with a simulated service or a logic backend), missing, or names `plant: isaac` (that plant
runs inside Isaac Sim). A refused file is always one readable line, never a traceback (`cell_io_bridge` and the Kit
runner refuse it the same way). The node runs the cell on wall time in whole physics steps; after a stall (an
overloaded machine) it catches up at most 200 steps per scan timer -- it warns once, and never skips a step.

## Run it on the Isaac plant (Isaac Sim 5.1)

Two processes, the same cell file with `plant: isaac` (`cell_io_isaac.yaml` is `cell_io_mock.yaml` with only the
`plant:` line changed). The stage runner runs INSIDE Isaac Sim, from a clean environment (a ROS-sourced shell crashes
Kit's ROS bridge, and `env -i` drops `ROS_DOMAIN_ID`, so it is passed explicitly — the same domain as the robot):

```bash
cd $(ros2 pkg prefix trainit_cell_io)/share/trainit_cell_io/examples/neutral_station   # or this repository's copy
env -i HOME=$HOME PATH=/usr/bin:/bin ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0} ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0} \
    LD_LIBRARY_PATH=$HOME/isaacsim/exts/isaacsim.ros2.bridge/humble/lib \
    $HOME/isaacsim/python.sh run_neutral_station.py --config cell_io_isaac.yaml
```

(`--gui` opens the Isaac Sim window: pass `DISPLAY`, `XAUTHORITY` and `XDG_RUNTIME_DIR` through `env -i` too.) It
refuses a bad cell file before Kit starts (one line, `CELL_IO_STAGE_END rc=2`, exit 2), prints
`CELL_IO_STAGE_BUILDING` once its own stop handling is in, `CELL_IO_STAGE_READY ...` when serving, the measured
real-time factor every 5 s (`RTF 1.000 ...`: the loop is paced to at most real time) and `CELL_IO_STAGE_END rc=0` at
the end. Stop it with Ctrl-C, or SIGINT / SIGTERM to its PROCESS GROUP (`python.sh` runs Kit's Python as a child: a
signal to the wrapper alone leaves Kit running) -- from `CELL_IO_STAGE_BUILDING` on, also during the stage build (the
build stops between two of its phases, in order); `--duration <s>` stops it by itself. Then, in a ROS terminal on the
same domain:

```bash
ros2 launch trainit_cell_io cell_io_bridge.launch.py \
    config:=$(ros2 pkg prefix trainit_cell_io)/share/trainit_cell_io/examples/neutral_station/cell_io_isaac.yaml
```

The robot's runner then follows the stage's clock: `use_sim_time:=true` (the stage publishes `/clock` after every
physics step: the wall time at its start plus the physics time, so it never jumps back across a stage restart).

Without ROS, `run_neutral_station.py --selfcheck` builds the stage, runs one cycle of every service from the joints'
read-back, prints each step's timing and read-back error and ends with `CELL_IO_SELFCHECK PASS` (or `FAIL: <why>`).
`build_neutral_station.py --out <file>.usda` writes the stage as text to inspect it (any Python with `pxr`).

The neutral stage is authored from USD primitives only (no remote, binary or third-party asset): a fixed pedestal, a
turntable on `turntable_joint` (revolute, vertical axis, 0 / 180 deg), a jaw on `jaw_joint` (prismatic, on the
turntable) closing the part against a fixed jaw, a part resting in a nest, and a line-indexer carriage on
`carriage_joint` (prismatic, 0 / 300 mm), all with position drives.

## The operator's command line

```bash
ros2 run trainit_cell_io cell_io_cli state [--json]           # every service's state, counters, devices, injections
ros2 run trainit_cell_io cell_io_cli watch                    # print every change until Ctrl-C
ros2 run trainit_cell_io cell_io_cli reset turner             # RESET a faulted service: HOMING, then READY
ros2 run trainit_cell_io cell_io_cli inject stuck_actuator device=clamp
ros2 run trainit_cell_io cell_io_cli inject consume_delay service=turner ms=2000   # a late request (D12)
ros2 run trainit_cell_io cell_io_cli restart                  # simulation: the controller restarts in place (D16)
```

| Verb | What it does |
|---|---|
| `state [--json]` | every service's state and detail, its counters (cycles accepted / completed / aborted, rejections, other-session images ignored, resets), the request images received and the re-deliveries ignored, the devices, the injections, the last command's answer |
| `watch` | the same, printed on every change until Ctrl-C |
| `reset <service>` | RESET a service (the operator's or the machine master's action): HOMING, then READY |
| `inject <fault> [key=value ...]` | a simulation fault injection (the table below), e.g. `consume_delay service=turner ms=2000`: a NEW request image is acted on only 2000 ms after the source first saw it (a request reaching the source late, e.g. after a station fault) |
| `restart` | the cell's controller restarts in place (implementation decision D16; simulation only) |

`restart` simulates a controller restart (a power cycle, a program restart) while the cell's communication endpoint
stays up: every service gets a NEW source session (READY, no cycle, counters from zero), the actuators are
de-energised and the axes hold where they are. A request image of the old session is never executed; a robot waiting
on it reads STATION_UNAVAILABLE ("restarted"). Stopping and relaunching the process is the other restart: the robot
then sees the source gone before the new session arrives.

The same commands serve the mock cell and the Isaac stage (`/cell_io/control` and `/cell_io/state`, std_msgs JSON).
A RESET is the operator's or the machine master's action, never the robot's. Every verb takes `--timeout <s>` (default
5). Exit codes: 0 done, 1 no cell answered, 2 the cell refused (the reason is printed) or wrong arguments. A command
carries a unique id and is re-sent until answered; the cell executes an id at most once.

Fault injection (simulation only; the same words on both plants):

| Fault | Parameters | Effect |
|---|---|---|
| `stuck_actuator` | `device` | the device does not move (mock: its motion stops; Isaac: its drive target is held) — the step times out, exactly at its `timeout_ms`: FAULT `ACTUATOR_TIMEOUT` |
| `slow` | `device`, `factor` 1..100 | its motions take `factor` times longer (a late DONE) |
| `failed_sensor` | `device` [`signal` open\|closed] | a digital sensor's diagnosis fails; on an actuator a limit sensor stuck ON — FAULT `SENSOR_FAULT` |
| `sensor_value` | `device`, `value` true\|false | a digital sensor reads `value` (e.g. no part) |
| `station_fault` | `service` [`detail`] | the station reports a fault (level, until `clear`) — FAULT `STATION_FAULT` |
| `hold` | `service` | the station holds its cycle: BUSY, timers stopped, no DONE until `clear` |
| `consume_delay` | `service`, `ms` | a NEW request image is acted on only `ms` after it was first seen (a whole number of scans; 0 removes it) |
| `clear` | — | every injection removed (a FAULT stays until the station's RESET) |

## The machine-service contract (the source side)

Per service, in the robot's slot (`robot_slot` of the cell file; ADR-0023), RELIABLE, TRANSIENT_LOCAL, depth 1:

| Topic | Type | Direction |
|---|---|---|
| `<slot>/<service>/request` | `trainit_interfaces/ChannelRequest` | the robot's runner (exactly one) -> the cell |
| `<slot>/<service>/status` | `trainit_interfaces/ChannelStatus` | the cell (exactly one source) -> the robot, every status period (100 ms) and at once on a change |

A request has an IDENTITY `(requester_session, seq)` and names the source session it is for. The source's rules, in
order:

1. it starts with a new random non-zero `source_session`, READY, no cycle, no rejection;
2. it acts on a request only when the request names ITS session and the identity differs from the last one it
   consumed: a re-delivered image is a no-op, an image of another session (left from before a restart) never runs.
   So that a re-delivered image stays a no-op when ANOTHER requester's request was consumed in between, the source
   also remembers the last seq it consumed from each of its last 16 requesters: an image equal to its requester's last
   consumed identity is a no-op too, and so is one OLDER than it (a requester's seq only grows, wrapping: 32-bit
   serial-number order) -- a replayed history (a TRANSIENT_LOCAL history of depth > 1 re-sent on a re-match) never
   runs a request again, and such an image is never held (it would displace a newer image of its requester);
3. READY or DONE + a new request: ACCEPTED, `cycle_*` = its identity, BUSY, the cycle starts;
4. BUSY + a new request: REJECTED `BUSY` (`reject_*` = its identity); the running cycle continues untouched;
5. FAULT + a new request: REJECTED `FAULTED`; HOMING + a new request: REJECTED `BUSY` — consumed, never run later;
6. more than one publisher on `<service>/request`: every new request REJECTED `TWO_REQUESTERS`;
7. completion: `cycle_done` true, DONE (until the next accepted request);
8. a device fault: FAULT with its code (`ACTUATOR_TIMEOUT`, `SENSOR_FAULT`, `STATION_FAULT`); the cycle stops where it
   is; FAULT persists until a RESET, which runs the return-to-home sequence (HOMING) and then reports READY;
9. the status every period and at once on a change; `heartbeat` +1 every period, minted by the cell's own scan.

`detail` is diagnosis only. The robot side (the runner's `request` / `wait` steps, liveness on its steady clock, the
user-facing MACHINE states) is TrainIt's motion runtime; the application's view (`machine_services`, `request`,
`wait`) is the setup assistant's.

Every request image is consumed and named. The cell keeps the latest image of each requester (the last 16) and, at
each scan, the source consumes at most one new identity -- the earliest arrived -- so two requesters' images that
arrive together are two consumptions, each named by `cycle_*` or `reject_*` in its own published image; none is
displaced unseen and left to run late. `cell_io_node` and `cell_io_bridge` read `<service>/request` with a deeper queue
(the contract's RELIABLE + TRANSIENT_LOCAL; one image per writer, level-held at depth 1), and the bridge forwards every
image of every requester to the stage. The cell's state shows, per service, `request_images` (every image handed to the
source) and `ignored_redelivered` (those that re-delivered an identity already consumed, or an older one of the same
requester: no-ops); a re-delivery never changes the source's `counters`.

## `cell_io.yaml`

The cell's machine-simulation file, hand-written (strict: an unknown key, a wrong type, a non-finite number or a step
the station could not build is refused with a readable message, at load). The public example is
`examples/neutral_station/cell_io_mock.yaml`.

| Key | Meaning |
|---|---|
| `cell_io: 1` | the format marker |
| `control_mode` | `simulation` (this package runs the machine logic) or `external_plc` (the logic runs on a PLC: a simulated service or a logic backend is then refused) |
| `plant` | `mock` or `isaac` (simulation only) |
| `robot_slot` | the robot's namespace the services live in (`''`: the root) |
| `scan` | `physics_dt_ms` (or `physics_dt` in seconds, e.g. `"1/60"`), `scan_every_steps`, `status_period_ms` (a whole number of scans, at most 500 ms: the robot synchronises a request on two heartbeat changes within its `channel_status_max_age_ms`, default 1000 ms): every duration in the file must be a whole number of physics steps (never rounded) |
| `devices` | `binary_actuator` (`timeout_ms`; out `<name>.cmd`, in the `open` / `closed` limit sensors), `position_axis` (`unit` deg\|mm, named `positions`, in-position `band`, `timeout_ms`), `digital_sensor`. A motion's `timeout_ms` is a whole number of scans and EXACT: a device at its target when it has elapsed passes, one that is not faults `ACTUATOR_TIMEOUT` at that scan |
| `services.<id>` | `logic: python_logic`, `cycle` and `home`: chains of `{require: <sensor>}`, `{close: <actuator>}`, `{open: <actuator>}`, `{move: <axis>, to: <position>}`; one service per device. Or `logic: code` (no `cycle` / `home`): the station's sequences are CODE (the `Sequence` protocol of `trainit_cell_io/core/logic.py`: `start()`, `scan(inputs, held, steps)` -> RUNNING / DONE / a `Fault`, `describe()`), handed by the program that runs the cell -- `Cell(config, plant, sequences={id: factory})`, the factory returning `(cycle, home)` once per source session; `cell_io_node` and the neutral stage runner hand none and refuse such a file (exit 2). The service's rules are the same `ProcessDevice`; only the contract's fault codes reach the wire (a `Fault` with any other code is reported as `STATION_FAULT`, its value in `detail`), and a sequence that raises faults its own station (`STATION_FAULT`), never the cell. A controller restart builds every new station first: a factory that fails restarts nothing |
| `mock` | the mock time model: `seed`, `jitter` (0..0.5, drawn per command change: the same seed gives the same trace), per device `close_ms` / `open_ms`, `speed` + `start`, `value`; motions in whole physics steps (a stroke of N steps takes exactly N) |
| `isaac` | the Isaac binding: `stage`, `articulations` (name -> prim path), per device: an actuator `{articulation, joint, open_m, closed_m, band_m, stroke_ms}` (a prismatic joint; its limit sensors are in-band read-backs), an axis `{articulation, joint, speed}` (deg <-> rad, mm <-> m), a sensor `{part, frame, nest_m, tolerance_m}` (ON while the part's position in the frame is within the tolerance of the nest point) |

## The simulator boundary

Isaac Sim's bundled rclpy (Python 3.11) carries the standard message packages but no `trainit_interfaces`, so the
Kit side speaks JSON in `std_msgs/String` (`trainit_cell_io/core/kit_boundary.py`, boundary version 1) on cell-level
topics, and `cell_io_bridge` translates:

| Topic | Direction | Content |
|---|---|---|
| `/cell_io/kit/<service>/status` | Kit -> bridge | every status image (latched); the stamp is the stage's clock |
| `/cell_io/kit/<service>/request` | bridge -> Kit | every request image of every requester, one per message, + the number of request publishers |
| `/cell_io/control`, `/cell_io/state` | the CLI <-> the cell | the operator's commands and view |
| `/clock` | Kit -> ROS | `rosgraph_msgs/Clock` after every physics step |

The bridge forwards each image as it comes and never mints a heartbeat or a session, nor republishes on its own: when
the stage dies the heartbeat freezes and the robot sees the source as not live, although the bridge still runs. A
malformed image is refused with a log line and never half-applied. A restarted bridge forwards the robot's level-held
request again, and a change of the requester count re-forwards each requester's latest image: the source has consumed
those identities already, so nothing runs twice (the state's `ignored_redelivered` counts them).

## Tests

From this repository, in a sourced workspace with `trainit_interfaces` >= 0.5.0 (the pure suites also run in a plain
Python with PyYAML; the ROS process tests use ROS domain 57 and skip without ROS):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider test
```

The structural stage test needs `pxr` (e.g. `pip install usd-core`) and skips without it. The Isaac smoke test is
opt-in (Isaac Sim 5.1 at `$ISAAC_SIM_PATH`, default `~/isaacsim`; ROS domain 58; logs under
`$TRAINIT_CELL_IO_EVIDENCE/isaac` when set):

```bash
TRAINIT_ISAAC_TEST=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q -p no:cacheprovider -s \
    test/test_isaac_neutral_station.py
```

| Layout | |
|---|---|
| `trainit_cell_io/core/` | pure Python (no ROS, no simulator): contract, scan clock, devices, sequences, `ProcessDevice`, mock plant, cell, Kit boundary, operator desk, the refusals (`ConfigError`), the bounded log |
| `trainit_cell_io/ros/` | `cell_io_node` (mock), `cell_io_bridge` (Isaac), `cell_io_cli` |
| `trainit_cell_io/isaac/` | `plant.py` (the pure `IsaacPlantAdapter`), `loop.py` (the Kit side without Isaac), `kit_ros.py` (std_msgs transport), `kit.py` (the Kit glue, imported inside Isaac Sim) |
| `launch/` | `cell_io.launch.py`, `cell_io_bridge.launch.py` |
| `examples/neutral_station/` | the public fixture: the two cell files, the stage builder, the stage runner |

## Licence

Apache-2.0 (see `LICENSE`).
