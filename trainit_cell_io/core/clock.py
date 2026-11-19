"""The scan clock: time in INTEGER physics steps (P0B_DESIGN.md §4; case C2).

A simulated machine never counts wall time: every duration is a whole number of physics steps of the plant (the mock's
time model, or the simulator's physics). A millisecond value that is not a whole number of steps fails LOUDLY at load
time -- never rounded, so a timeout is never silently shortened or stretched.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Tuple, Union

from .errors import ConfigError


class ClockError(ConfigError):
    """A duration that the scan clock cannot represent exactly (a configuration refusal: exit 2, never a traceback)."""


def parse_dt(physics_dt: Union[int, float, str, Fraction, None] = None,
             physics_dt_ms: Union[int, None] = None) -> Fraction:
    """The physics step in seconds, EXACTLY: `physics_dt_ms: 5` (an integer), or `physics_dt: "1/60"` / `0.005`."""
    if (physics_dt is None) == (physics_dt_ms is None):
        raise ClockError('give exactly one of physics_dt (seconds, e.g. "1/60") or physics_dt_ms (an integer)')
    if physics_dt_ms is not None:
        if isinstance(physics_dt_ms, bool) or not isinstance(physics_dt_ms, int) or physics_dt_ms <= 0:
            raise ClockError(f'physics_dt_ms must be a positive integer, got {physics_dt_ms!r}')
        return Fraction(physics_dt_ms, 1000)
    if isinstance(physics_dt, bool):
        raise ClockError(f'physics_dt must be a number of seconds, got {physics_dt!r}')
    if isinstance(physics_dt, Fraction):
        dt = physics_dt
    elif isinstance(physics_dt, str):
        try:
            dt = Fraction(physics_dt.strip())
        except (ValueError, ZeroDivisionError) as exc:
            raise ClockError(f'physics_dt {physics_dt!r} is not a number of seconds') from exc
    elif isinstance(physics_dt, int):
        dt = Fraction(physics_dt)
    elif isinstance(physics_dt, float):
        # A float is taken by its shortest decimal text (0.005 -> 1/200), never by its binary expansion; a non-finite
        # value (.nan, .inf) has no such text and is refused like any other non-number.
        try:
            dt = Fraction(repr(physics_dt))
        except (ValueError, OverflowError) as exc:
            raise ClockError(f'physics_dt must be a finite number of seconds, got {physics_dt!r}') from exc
    else:
        raise ClockError(f'physics_dt must be a number of seconds, got {physics_dt!r}')
    if dt <= 0:
        raise ClockError(f'physics_dt must be positive, got {physics_dt!r}')
    return dt


class ScanClock:
    """Physics steps, the scan (every `scan_every_steps` steps) and the status period, all exact."""

    def __init__(self, dt: Fraction, scan_every_steps: int = 1, status_period_ms: int = 100):
        if isinstance(scan_every_steps, bool) or not isinstance(scan_every_steps, int) or scan_every_steps < 1:
            raise ClockError(f'scan_every_steps must be an integer >= 1, got {scan_every_steps!r}')
        self.dt = dt
        self.scan_every_steps = scan_every_steps
        self.status_period_steps = self.steps(status_period_ms, 'status_period_ms')
        if self.status_period_steps == 0:
            raise ClockError('status_period_ms must be > 0')
        if self.status_period_steps % scan_every_steps:
            raise ClockError(
                f'status_period_ms = {status_period_ms} ms is {self.status_period_steps} physics steps, not a whole '
                f'number of scans (one scan = {scan_every_steps} steps)')

    @property
    def scan_period_steps(self) -> int:
        return self.scan_every_steps

    def dt_ms_text(self) -> str:
        ms = self.dt * 1000
        return str(ms.numerator) if ms.denominator == 1 else f'{ms.numerator}/{ms.denominator}'

    def steps(self, ms: int, what: str = 'duration') -> int:
        """`ms` as a whole number of physics steps; fails loudly otherwise (never rounds)."""
        if isinstance(ms, bool) or not isinstance(ms, int):
            raise ClockError(f'{what} must be an integer number of milliseconds, got {ms!r}')
        if ms < 0:
            raise ClockError(f'{what} must not be negative, got {ms} ms')
        q = Fraction(ms, 1000) / self.dt
        if q.denominator != 1:
            raise ClockError(
                f'{what} = {ms} ms is not a whole number of physics steps (dt = {self.dt_ms_text()} ms): '
                f'{float(q):.4f} steps')
        return q.numerator

    def ms(self, steps: int) -> float:
        """Physics steps as milliseconds (for reports only)."""
        return float(Fraction(steps) * self.dt * 1000)

    def seconds(self, steps: int) -> float:
        return float(Fraction(steps) * self.dt)


class CatchUp:
    """A cell on WALL time (the mock plant's `cell_io_node`): how many whole physics steps to run now.

    The wall clock says where the cell should be (`int((now - t0) / dt)` steps); one call never runs more than `cap`
    steps, so a stalled process (a debugger, a swapped-out machine) never runs an unbounded loop in one timer callback.
    The step count stays EXACT: a lag is caught up over the following calls (at most `cap` steps each), never skipped.
    """

    def __init__(self, dt_s: float, t0: float, cap: int = 200):
        if not dt_s > 0:
            raise ValueError(f'dt_s must be > 0, got {dt_s!r}')
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f'cap must be an integer >= 1, got {cap!r}')
        self.dt_s = dt_s
        self.t0 = t0
        self.cap = cap

    def due(self, now: float, steps: int) -> Tuple[int, int]:
        """(the steps to run now, the steps the cell is behind wall time before running them)."""
        behind = max(0, int((now - self.t0) / self.dt_s) - steps)
        return min(behind, self.cap), behind
