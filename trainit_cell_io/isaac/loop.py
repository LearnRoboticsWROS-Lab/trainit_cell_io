"""The Kit side of the Isaac plant, without Isaac (P0B_DESIGN.md §4; D13): the stage clock, the pacing, the per-step
sequence. PURE: no omni / isaacsim / pxr / rclpy import, so it is unit-tested in any Python and driven, in a test, by a
mock plant and the system rclpy; inside Kit the stage runner hands it Isaac's world and the bundled rclpy transport.

Per physics step (`KitSide.after_physics_step`, called right AFTER the simulator stepped):
    1. the cell's step (`Cell.step(advance_plant=False)`: the simulator moved the plant already) -> the status images
       due now (every period, and at once on a change), as std_msgs JSON on /cell_io/kit/<service>/status;
    2. /clock (rosgraph_msgs/Clock) after EVERY step: the stamp = the wall time at start + the physics time, so it is
       monotonic across a stage restart while the stage runs at most at real time (D13);
    3. the transport is spun: the latest request images (from cell_io_bridge) and the operator's commands;
    4. /cell_io/state at 1 Hz (wall) and at once after a change or a command.
The heartbeat and the session are the cell's own (minted by its scan, here, Kit-side): the ROS-side bridge never mints
them, so a dead stage freezes the heartbeat the robot sees (PB-P9).
"""

from __future__ import annotations

import math
import time
from fractions import Fraction
from typing import Callable, Optional, Protocol

from ..core.cell import Cell, ControlError
from ..core.kit_boundary import BoundaryError, decode_request, encode_status
from ..core.operator_desk import OperatorDesk

NS = 10 ** 9


class SimClock:
    """The stage's clock in integer nanoseconds: `start_ns` (the wall time at start) + `steps` x dt, exactly."""

    def __init__(self, dt: Fraction, start_ns: int):
        if not isinstance(dt, Fraction) or dt <= 0:
            raise ValueError(f'dt must be a positive Fraction of a second, got {dt!r}')
        self.dt = dt
        self.start_ns = int(start_ns)

    def stamp_ns(self, steps: int) -> int:
        return self.start_ns + math.floor(steps * self.dt * NS)

    def seconds(self, steps: int) -> float:
        return float(steps * self.dt)


class Pacer:
    """Paces a stepping loop to at most `max_rtf` x real time; measures the real-time factor (RTF).

    `pace(steps)` sleeps until the wall clock has caught up with `steps` physics steps (never ahead of real time); a
    slow simulator simply runs at its own RTF < 1, which `rtf()` reports (the design's liveness and timeout bounds are
    stated for RTF <= 1).
    """

    def __init__(self, dt_s: float, max_rtf: float = 1.0, now: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if not dt_s > 0 or not max_rtf > 0:
            raise ValueError('dt_s and max_rtf must be > 0')
        self.dt_s = dt_s
        self.max_rtf = max_rtf
        self._now = now
        self._sleep = sleep
        self._t0: Optional[float] = None
        self._window = (0.0, 0)

    def start(self, steps: int = 0) -> None:
        self._t0 = self._now() - steps * self.dt_s / self.max_rtf
        self._window = (self._now(), steps)

    def pace(self, steps: int) -> None:
        if self._t0 is None:
            self.start(steps)
        due = self._t0 + steps * self.dt_s / self.max_rtf
        ahead = due - self._now()
        if ahead > 0:
            self._sleep(ahead)

    def rtf(self, steps: int) -> float:
        """The RTF since start: simulated time / wall time."""
        if self._t0 is None:
            return 0.0
        wall = self._now() - self._t0
        return (steps * self.dt_s) / wall if wall > 0 else 0.0

    def window_rtf(self, steps: int) -> float:
        """The RTF since the previous call (a sliding report), then a new window starts."""
        t, s = self._window
        now = self._now()
        self._window = (now, steps)
        return ((steps - s) * self.dt_s) / (now - t) if now > t else 0.0


class Transport(Protocol):
    """The Kit side's ROS transport (the bundled rclpy in Kit, `KitRos`); it calls back `KitSide.on_request` /
    `KitSide.on_command` from `spin()`."""

    def publish_status(self, service: str, text: str) -> None: ...

    def publish_clock(self, stamp_ns: int) -> None: ...

    def publish_state(self, text: str) -> None: ...

    def spin(self) -> None: ...


class KitSide:
    def __init__(self, cell: Cell, clock: SimClock, transport: Optional[Transport] = None,
                 log: Callable[[str], None] = print, advance_plant: bool = False, state_period_s: float = 1.0,
                 wall: Callable[[], float] = time.monotonic):
        self.cell = cell
        self.clock = clock
        self.transport = transport
        self.desk = OperatorDesk(cell)
        self._log = log
        self._advance = advance_plant
        self._state_period_s = state_period_s
        self._wall = wall
        self._next_state = 0.0
        self._log_index = 0               # the running total of the cell's log lines already printed
        self.requests_received = {sid: 0 for sid in cell.services}
        self.refused_requests = 0

    # --- the transport's callbacks --------------------------------------------------------------------------------
    def on_request(self, topic_service: str, text: str) -> None:
        """A request image from cell_io_bridge (JSON, one image per message): handed to the cell as it comes -- the cell
        keeps each requester's latest and consumes one new identity per scan; never half-applied."""
        try:
            service, image, requesters = decode_request(text)
            if service != topic_service:
                raise BoundaryError(f"the image names service {service!r} on the topic of '{topic_service}'")
            self.cell.set_request(service, image, max(1, requesters))
        except (BoundaryError, ControlError) as exc:
            self.refused_requests += 1
            self._log(f"refused a request image for '{topic_service}' (nothing applied): {exc}")
            return
        self.requests_received[service] += 1

    def on_command(self, text: str) -> None:
        answer, repeated = self.desk.handle(text)
        if not repeated:
            self._log(f"cell command {'done' if answer['ok'] else 'REFUSED'}: {answer['text']}")
        self._publish_state()

    # --- one physics step ---------------------------------------------------------------------------------------------
    def after_physics_step(self) -> None:
        due = self.cell.step(advance_plant=self._advance)
        stamp = self.clock.stamp_ns(self.cell.steps)
        if self.transport is not None:
            for sid, image in due:
                self.transport.publish_status(sid, encode_status(sid, image, stamp))
            self.transport.publish_clock(stamp)
            self.transport.spin()
        lines, self._log_index, dropped = self.cell.log.since(self._log_index)   # every new line exactly once
        if dropped:
            self._log(f'{dropped} cell log line(s) were dropped before they could be printed (the log keeps the last '
                      f'{self.cell.log.maxlen})')
        for step, text in lines:
            self._log(f'[{self.clock.seconds(step):8.2f} s] {text}')
        now = self._wall()
        if lines or dropped or now >= self._next_state:
            self._publish_state()

    def _publish_state(self) -> None:
        self._next_state = self._wall() + self._state_period_s
        if self.transport is not None:
            self.transport.publish_state(self.desk.state_json())
