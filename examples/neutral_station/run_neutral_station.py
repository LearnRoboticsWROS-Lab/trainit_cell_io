#!/usr/bin/env python3
"""The Kit-side runner of the PUBLIC neutral station (P0B_DESIGN.md §4/§5; P0.b D13). Runs INSIDE Isaac Sim 5.1.

It builds the stage from USD primitives (build_neutral_station.py), steps the physics at 200 Hz, and runs the cell's
station logic -- the core `Cell` with the `IsaacPlantAdapter` -- IN Kit, one scan every `scan_every_steps` physics
steps, on the joints' read-back. Only the handshake images cross to ROS, as std_msgs JSON; `cell_io_bridge` (ROS side,
same domain) publishes the canonical ChannelRequest / ChannelStatus in the robot's slot. The heartbeat and the session
are minted here: when this process dies, the bridge has nothing to forward and the robot sees the source as not live.

Start it from a CLEAN environment (a ROS-sourced shell crashes Kit's bridge; `env -i` drops ROS_DOMAIN_ID, so pass it):

    env -i HOME=$HOME PATH=/usr/bin:/bin ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp \\
        ROS_DOMAIN_ID=<the robot's domain> ROS_LOCALHOST_ONLY=<as on the ROS side, e.g. 1> \\
        LD_LIBRARY_PATH=$HOME/isaacsim/exts/isaacsim.ros2.bridge/humble/lib \\
        $HOME/isaacsim/python.sh run_neutral_station.py [--config cell_io_isaac.yaml] [--gui]

and, in a ROS terminal on the same domain:  ros2 launch trainit_cell_io cell_io_bridge.launch.py config:=<same file>
(+ `DISPLAY XAUTHORITY XDG_RUNTIME_DIR` through `env -i` for --gui). The robot's runner then uses this stage's /clock:
`use_sim_time:=true`.

Per physics step: world.step -> the cell's step (the read-back, the scan) -> the due status images (JSON) -> /clock
(rosgraph_msgs/Clock: the wall time at start + the physics time, monotonic across a restart, D13) -> spin (request
images, operator commands on /cell_io/control) -> /cell_io/state at 1 Hz and on a change. The loop is paced to at most
1x real time; the measured real-time factor is printed.

Lines a driver waits for: `CELL_IO_STAGE_BUILDING` (the runner's own SIGINT / SIGTERM handling is in: from here a stop
signal ends the run in order), `CELL_IO_STAGE_READY ...`, `CELL_IO_STAGE_END rc=<n>` (0 after an orderly stop --
during the build too: the stop flag is checked between the build phases, a Kit call is never interrupted; 2 when the
cell file is refused, before Kit starts; 1 on an error). With --selfcheck (no ROS at all): one cycle of every service
from the physical read-back, its timings and read-back errors, then `CELL_IO_SELFCHECK PASS` or
`CELL_IO_SELFCHECK FAIL: <why>` (or `CELL_IO_SELFCHECK STOPPED ...`: a stop signal, no verdict). Not a PLC, not a
safety function (ADR-0033).
"""

import argparse
import glob
import math
import os
import random
import signal
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
BUILDING = 'CELL_IO_STAGE_BUILDING'
READY = 'CELL_IO_STAGE_READY'
END = 'CELL_IO_STAGE_END'
SELFCHECK = 'CELL_IO_SELFCHECK'
EXIT_REFUSED = 2
SELFCHECK_REQUESTER = 0x5E1FC4EC
ENV_HINT = ('start Kit from a clean environment: env -i HOME=$HOME PATH=/usr/bin:/bin ROS_DISTRO=humble '
            'RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=<d> LD_LIBRARY_PATH=$HOME/isaacsim/exts/'
            'isaacsim.ros2.bridge/humble/lib $HOME/isaacsim/python.sh run_neutral_station.py')


def parse_args(argv):
    parser = argparse.ArgumentParser(description='The neutral station inside Isaac Sim (the Isaac plant of '
                                                 'trainit_cell_io).')
    parser.add_argument('--config', default=os.path.join(HERE, 'cell_io_isaac.yaml'),
                        help="the cell's machine-simulation file (plant: isaac); default: cell_io_isaac.yaml here")
    parser.add_argument('--gui', action='store_true', help='open the Isaac Sim window (rendered steps)')
    parser.add_argument('--selfcheck', action='store_true',
                        help='no ROS: run one cycle of every service, print timings and read-back errors, exit')
    parser.add_argument('--duration', type=float, default=0.0,
                        help='stop after this many seconds of wall time (0: until SIGINT / SIGTERM)')
    parser.add_argument('--max-rtf', type=float, default=1.0, help='pace to at most this real-time factor (1.0)')
    parser.add_argument('--rtf-every', type=float, default=5.0, help='print the real-time factor every N s (5)')
    return parser.parse_args(argv)


class StopFlag:
    """The runner's own SIGINT / SIGTERM handling: a flag that the build phases and the loops check, so a stop signal
    ends the run in order (`CELL_IO_STAGE_END rc=0`) wherever it lands -- never a KeyboardInterrupt inside a Kit call,
    never rclpy's handler shutting the context down mid-build. The first signal is kept (its name is reported)."""

    def __init__(self):
        self.signal_name = None

    @property
    def requested(self) -> bool:
        return self.signal_name is not None

    def __call__(self, signum, frame=None):   # noqa: ARG002 -- a signal handler's signature
        if self.signal_name is None:
            self.signal_name = signal.Signals(signum).name

    def install(self) -> 'StopFlag':
        """Ours for SIGINT and SIGTERM, replacing whatever was there (rclpy.init() installs its own)."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self)
        return self


def run_phases(stop: StopFlag, phases, log=print) -> bool:
    """Run the build phases `[(name, callable), ...]` in order, checking the stop flag BETWEEN them (a phase -- a Kit
    call -- is never interrupted). True when every phase ran; False, with one line naming where, when a stop signal
    ended the build."""
    for name, phase in phases:
        if stop.requested:
            log(f'stopped by {stop.signal_name} during the build, before {name} -- an orderly stop')
            return False
        phase()
    if stop.requested:
        log(f'stopped by {stop.signal_name} at the end of the build -- an orderly stop')
        return False
    return True


def core_path() -> str:
    """Make trainit_cell_io importable in Kit's Python, APPENDED to sys.path (never shadow Kit's own modules): this
    checkout when the script runs from one, else the installed package beside this share/ directory."""
    checkout = os.path.dirname(os.path.dirname(HERE))
    if os.path.isdir(os.path.join(checkout, 'trainit_cell_io', 'core')):
        sys.path.append(checkout)
        return checkout
    prefix = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))   # share/<pkg>/examples/<x>
    for site in sorted(glob.glob(os.path.join(prefix, 'lib', 'python3*', 'site-packages'))):
        if os.path.isdir(os.path.join(site, 'trainit_cell_io')):
            sys.path.append(site)
            return site
    raise SystemExit('run_neutral_station: cannot find the trainit_cell_io package (a checkout or an install)')


def load_config(path):
    """Validate the cell file and the Isaac binding BEFORE Kit starts (a refusal costs nothing): a refused file raises
    `ConfigError` (or the plant's `InjectionError`) with a readable message; `main()` exits 2 on it."""
    from trainit_cell_io.core import Cell, ConfigError, load
    from trainit_cell_io.isaac.plant import IsaacPlantAdapter
    cfg = load(path)
    if cfg.control_mode != 'simulation' or cfg.plant != 'isaac':
        raise ConfigError(f'{path}: needs control_mode simulation and plant: isaac (got {cfg.control_mode} / '
                          f'{cfg.plant})')
    if cfg.isaac.get('stage') != 'neutral_station':
        raise ConfigError(f"{path}: isaac.stage must be 'neutral_station' (this runner builds it)")
    Cell(cfg, IsaacPlantAdapter(cfg.isaac), rng=random.Random(0))       # every device bound, or refused here
    return cfg


def readback(cell, plant) -> str:
    """Every device's read-back against what it was commanded to, in its own unit."""
    parts = []
    for name, dev in cell.devices.items():
        if dev.kind == 'binary_actuator':
            b = plant.actuators[name]
            q = plant.joint_position(name)
            goal = b.closed_m if dev.cmd else b.open_m
            parts.append(f'{name} {"?" if q is None else f"{q * 1000:.2f}"} mm (commanded '
                         f'{"closed" if dev.cmd else "open"} {goal * 1000:g} mm'
                         + ('' if q is None else f', error {abs(q - goal) * 1000:.2f} mm') + ')')
        elif dev.kind == 'position_axis':
            if dev.target_name is None or dev.position is None:
                parts.append(f'{name} {dev.describe()}')
            else:
                goal = dev.positions[dev.target_name]
                parts.append(f'{name} {dev.position:.2f} {dev.unit} (commanded {dev.target_name} {goal:g}, error '
                             f'{abs(dev.position - goal):.2f} {dev.unit})')
        else:
            parts.append(dev.describe())
    return '; '.join(parts)


def part_offset_mm(plant, access):
    for name, s in plant.sensors.items():
        where = access.body_position_in(s.part, s.frame)
        if where is None:
            return name, None
        return name, math.dist(where, s.nest_m) * 1000.0
    return None, None


def selfcheck(world, cell, plant, access, render, stop=None) -> int:
    """One cycle of every service, from the physical read-back, WITHOUT ROS; timings and read-back errors. A stop
    signal ends it in order with `CELL_IO_SELFCHECK STOPPED ...` (no verdict) and rc 0."""
    from trainit_cell_io.core import RequestImage
    from trainit_cell_io.core.contract import STATE_DONE, STATE_FAULT, STATE_NAMES
    stop = (stop or StopFlag()).install()   # ours again after the build: nothing in the build can have replaced it
    clock = cell.clock
    for _ in range(40):                     # settle 0.2 s: the part rests in its nest
        world.step(render=render)
        cell.step(advance_plant=False)
    failures = []
    sensor, before = part_offset_mm(plant, access)
    if sensor:
        print(f'{SELFCHECK}: {sensor} reads {cell.devices[sensor].value} before; the part is '
              f'{"unknown" if before is None else f"{before:.2f} mm"} from its nest point')
    for n, (sid, pd) in enumerate(cell.services.items(), start=1):
        cell.set_request(sid, RequestImage(pd.session, SELFCHECK_REQUESTER, n))
        start, wall0 = cell.steps, time.monotonic()
        detail = None
        limit = start + clock.steps(30000)
        while pd.state not in (STATE_DONE, STATE_FAULT) and cell.steps < limit and not stop.requested:
            world.step(render=render)
            cell.step(advance_plant=False)
            if pd.image.detail != detail:
                detail = pd.image.detail
                print(f'{SELFCHECK}: {sid} t+{clock.seconds(cell.steps - start):6.3f} s {detail} | '
                      f'{readback(cell, plant)}')
        if stop.requested:
            print(f'{SELFCHECK} STOPPED by {stop.signal_name} during the {sid} cycle: no verdict', flush=True)
            return 0
        sim_s, wall_s = clock.seconds(cell.steps - start), time.monotonic() - wall0
        verdict = STATE_NAMES.get(pd.state, pd.state)
        print(f'{SELFCHECK}: {sid} request -> {verdict} in {sim_s:.3f} s of simulated time ({wall_s:.3f} s wall, '
              f'RTF {sim_s / wall_s if wall_s > 0 else 0:.2f}); cycle_done={pd.image.cycle_done}')
        if pd.state != STATE_DONE or not pd.image.cycle_done:
            failures.append(f'{sid} ended {verdict}: {pd.image.detail}')
        for _ in range(40):                 # settle 0.2 s, then the read-back at rest
            world.step(render=render)
            cell.step(advance_plant=False)
        print(f'{SELFCHECK}: {sid} at rest: {readback(cell, plant)}')
        for name, dev in cell.devices.items():
            if dev.kind == 'position_axis' and dev.target_name is not None and dev.position is not None:
                error = abs(dev.position - dev.positions[dev.target_name])
                if error > dev.band:
                    failures.append(f'{name} rests {error:.2f} {dev.unit} from {dev.target_name} (band {dev.band})')
    if sensor:
        _, after = part_offset_mm(plant, access)
        print(f'{SELFCHECK}: {sensor} reads {cell.devices[sensor].value} after; the part is '
              f'{"unknown" if after is None else f"{after:.2f} mm"} from its nest point')
        if after is None or after > plant.sensors[sensor].tolerance_m * 1000.0:
            failures.append(f'the part left its nest ({after} mm)')
    if failures:
        print(f'{SELFCHECK} FAIL: ' + '; '.join(failures))
        return 1
    print(f'{SELFCHECK} PASS')
    return 0


def serve(app, world, cell, args, stop) -> int:
    """The served stage: the cell's images over the bundled rclpy, /clock every step, paced to <= max_rtf; until the
    stop flag (SIGINT / SIGTERM, installed by run()), Kit's own end or --duration."""
    import rclpy
    from trainit_cell_io.isaac.kit_ros import KitRos
    from trainit_cell_io.isaac.loop import KitSide, Pacer, SimClock
    clock = SimClock(cell.clock.dt, time.time_ns())
    kit = KitSide(cell, clock, log=lambda text: print(text, flush=True))
    ros = KitRos(cell.services, kit.on_request, kit.on_command)
    kit.transport = ros
    stop.install()        # ours again after every build call and the Kit-side node / executor (a signal already
    #                       received stays recorded: the flag is the same object)
    pacer = Pacer(float(cell.clock.dt), max_rtf=args.max_rtf)
    sessions = ', '.join(f'{sid} {pd.session:#010x}' for sid, pd in cell.services.items())
    print(f'{READY} services={",".join(cell.services)} domain={os.environ.get("ROS_DOMAIN_ID", "0")} '
          f'localhost_only={os.environ.get("ROS_LOCALHOST_ONLY", "0")} physics_dt_ms={cell.clock.dt_ms_text()} '
          f'scan_every_steps={cell.clock.scan_every_steps} sessions: {sessions}', flush=True)
    end = time.monotonic() + args.duration if args.duration > 0 else math.inf
    next_report = time.monotonic() + args.rtf_every
    pacer.start(cell.steps)
    try:
        while not stop.requested and app.is_running() and time.monotonic() < end:
            world.step(render=args.gui)
            kit.after_physics_step()
            pacer.pace(cell.steps)
            if time.monotonic() >= next_report:
                next_report += args.rtf_every
                print(f'RTF {pacer.window_rtf(cell.steps):.3f} over the last {args.rtf_every:g} s (since start '
                      f'{pacer.rtf(cell.steps):.3f}); simulated {clock.seconds(cell.steps):.1f} s', flush=True)
    finally:
        print(f'stopping: simulated {clock.seconds(cell.steps):.2f} s, RTF since start {pacer.rtf(cell.steps):.3f}',
              flush=True)
        ros.close()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def run(app, args, cfg, stop: StopFlag) -> int:
    from trainit_cell_io.core import Cell
    from trainit_cell_io.isaac.kit import KitJointAccess
    from trainit_cell_io.isaac.plant import IsaacPlantAdapter
    rclpy = None
    if not args.selfcheck:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('isaacsim.ros2.bridge')            # its own rclpy (Python 3.11): std_msgs, rosgraph_msgs
        app.update()
        try:
            import rclpy
        except ImportError as exc:
            print(f'run_neutral_station: the ROS 2 bridge extension gave no rclpy ({exc}); {ENV_HINT}', flush=True)
            return 1
        try:                                                # A-B2: why the handshake crosses as std_msgs JSON
            import trainit_interfaces  # noqa: F401
            custom = 'importable'
        except ImportError:
            custom = 'not importable (expected: the images cross as std_msgs JSON, cell_io_bridge translates)'
        print(f'Kit rclpy: std_msgs + rosgraph_msgs in use; trainit_interfaces {custom}', flush=True)
        rclpy.init()
    # Right after rclpy.init() (which installs ITS handlers: a SIGINT would shut the context down and raise
    # KeyboardInterrupt in the middle of a Kit call): ours, a flag the build phases below and the loops check.
    stop.install()
    print(f'{BUILDING} (SIGINT / SIGTERM from here stop the run in order)', flush=True)
    built = {}

    def build_stage():
        import omni.usd
        import build_neutral_station
        build_neutral_station.build(omni.usd.get_context().get_stage())   # authored BEFORE the World and its reset

    def make_world():
        from isaacsim.core.api import World
        World.clear_instance()
        dt = float(cfg.clock.dt)
        built['world'] = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt, set_defaults=False)

    def make_cell():
        built['plant'] = plant = IsaacPlantAdapter(cfg.isaac)
        built['cell'] = Cell(cfg, plant)                    # new source sessions: a new start is a new session
        built['access'] = KitJointAccess(built['world'], plant.joints(), plant.bodies())

    def reset_world():
        built['world'].reset()

    def bind_joints():
        built['access'].resolve()
        built['plant'].attach(built['access'])

    phases = [('the stage', build_stage), ('the physics world', make_world), ('the cell', make_cell),
              ('the world reset', reset_world), ('the joint binding', bind_joints)]
    if not run_phases(stop, phases, lambda text: print(text, flush=True)):
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()
        return 0
    world, cell, plant, access = built['world'], built['cell'], built['plant'], built['access']
    print(f'stage built: articulations {access.dof_names()}; physics dt {cfg.clock.dt_ms_text()} ms', flush=True)
    if args.selfcheck:
        return selfcheck(world, cell, plant, access, args.gui, stop)
    return serve(app, world, cell, args, stop)


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)             # Kit's close does not flush a redirected stdout
    args = parse_args(sys.argv[1:])                         # BEFORE SimulationApp (it forwards argv to Kit)
    core_path()
    from trainit_cell_io.core import ConfigError, InjectionError
    try:
        cfg = load_config(args.config)
    except (ConfigError, InjectionError) as exc:            # refused before Kit starts: one line, never a traceback
        print(f'run_neutral_station: refused: {exc}', file=sys.stderr, flush=True)
        print(f'{END} rc={EXIT_REFUSED}', flush=True)
        return EXIT_REFUSED
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': not args.gui})
    rc = 1
    try:
        rc = run(app, args, cfg, StopFlag())
    except Exception:  # noqa: BLE001 -- reported, then Kit is closed
        traceback.print_exc()
        sys.stdout.flush()
        rc = 1
    finally:
        print(f'{END} rc={rc}', flush=True)
        app.close()
    return rc


if __name__ == '__main__':
    sys.exit(main())
