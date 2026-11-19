"""The `MockPlantAdapter`: a deterministic time model of the cell's devices, with fault injection (case C4).

Deterministic: the motion of every device is a function of the commands, the physics steps and a seeded random
generator (drawn only when a command CHANGES, for the `jitter` of the motion times), so the same seed and the same
commands give the same trace, step for step.

Exact in WHOLE physics steps: a motion counts its own steps `k` from where it started and arrives at step
n = ceil(travel x steps-per-unit) -- an actuator's stroke of N steps closes at its N-th step (s = k / N), an axis
covering D at R per step arrives at step ceil(D / R) -- never one step late because floats were accumulated. A motion
re-bases (from where the device is) when its command or its speed changes (a reversal, a `slow` injection).

`cell_io.yaml`, `plant: mock`:

    mock:
      seed: 1
      jitter: 0.0              # 0 .. 0.5: each motion time x (1 +- jitter), drawn per command change
      devices:
        clamp: {close_ms: 300, open_ms: 300}          # a binary actuator's stroke times
        table: {speed: 360.0, start: load}            # an axis' speed (its unit per second) and starting position
        part_present: {value: true}                   # a sensor's value

Fault injection (simulation control only; the operator's CLI or a test, never the robot):
    stuck_actuator  device            the device never moves (its limit / position is never reached)
    failed_sensor   device [signal]   a sensor's diagnosis fails; for an actuator, a limit sensor stuck ON
    sensor_value    device value      a digital sensor reads `value` (e.g. no part)
    station_fault   service [detail]  the station reports a fault (level, until `clear`)
    hold            service           the station holds its cycle (BUSY, no DONE, timers stopped) until `clear`
    slow            device factor     the device's motions take `factor` times longer (a late DONE)
    consume_delay   service ms        the source acts on a NEW request image only `ms` after it first saw it (a late
                                      request: e.g. one that reaches the source after a station fault); `ms` is a whole
                                      number of scans, 0 removes it
    clear                             every injection removed (a FAULT stays until the station's RESET)
"""

from __future__ import annotations

import math
import random
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .bounded_log import DEFAULT_MAXLEN, BoundedLog
from .clock import ClockError, ScanClock
from .devices import BinaryActuator, DigitalSensor, PositionAxis
from .plant import InjectionError, PlantAdapter

FAULTS = ('stuck_actuator', 'failed_sensor', 'sensor_value', 'station_fault', 'hold', 'slow', 'consume_delay',
          'clear')
MAX_CONSUME_DELAY_MS = 600000
STEP_EPS = 1e-9          # a motion's step count is ceil(x - 1e-9): a float rounding never costs a whole physics step


def _whole_steps(steps: float) -> int:
    """The whole physics steps a motion of `steps` (a float: a stroke time over dt, x jitter, x slow) needs: >= 1."""
    return max(1, math.ceil(steps - STEP_EPS))


def consume_delay_steps(clock: Optional[ScanClock], ms: object) -> int:
    """`ms` of the consume_delay injection as physics steps: an integer 0..600000 that is a whole number of SCANS (the
    source looks at its request once per scan), refused loudly otherwise (C2: never rounded)."""
    if clock is None:
        raise InjectionError('consume_delay: the plant is not bound to a cell yet')
    if isinstance(ms, bool) or not isinstance(ms, int) or not 0 <= ms <= MAX_CONSUME_DELAY_MS:
        raise InjectionError(f'consume_delay: ms must be an integer 0..{MAX_CONSUME_DELAY_MS} (got {ms!r})')
    try:
        steps = clock.steps(ms, 'consume_delay ms')
    except ClockError as exc:
        raise InjectionError(str(exc)) from exc
    if steps % clock.scan_every_steps:
        raise InjectionError(f'consume_delay: {ms} ms is not a whole number of scans (one scan = '
                             f'{clock.ms(clock.scan_every_steps):g} ms)')
    return steps


class _Actuator:
    def __init__(self, name: str, close_steps: float, open_steps: float):
        self.name = name
        self.close_steps = close_steps
        self.open_steps = open_steps
        self.s = 0.0                 # 0 open .. 1 closed
        self.cmd = False
        self.factor = 1.0            # jitter of the current motion
        self.stuck = False
        self.slow = 1.0
        self.failed_limit: Optional[str] = None   # 'open' | 'closed': that limit sensor stuck ON
        # the current motion, in whole steps: from `origin`, `k` steps so far, a full stroke of `stroke` steps, `n`
        # steps to the limit (None: re-base at the next step -- a new command or a new speed)
        self.origin = 0.0
        self.k = 0
        self.stroke: Optional[float] = None
        self.n = 0

    def step(self) -> bool:
        """One physics step of motion toward the commanded limit; True when it arrived at this step."""
        goal = 1.0 if self.cmd else 0.0
        stroke = (self.close_steps if self.cmd else self.open_steps) * self.factor * self.slow
        if stroke != self.stroke:
            self.origin, self.k, self.stroke = self.s, 0, stroke
            self.n = _whole_steps(abs(goal - self.s) * stroke)
        self.k += 1
        self.s = goal if self.k >= self.n else self.origin + math.copysign(self.k / stroke, goal - self.origin)
        return self.s == goal


class _Axis:
    def __init__(self, name: str, speed_per_step: float, start: float):
        self.name = name
        self.speed_per_step = speed_per_step
        self.pos = start
        self.target: Optional[float] = None
        self.factor = 1.0
        self.stuck = False
        self.slow = 1.0
        # the current motion, in whole steps: from `origin` at `rate` per step, `k` steps so far, `n` steps to the
        # target (rate None: re-base at the next step -- a new target or a new speed)
        self.origin = start
        self.k = 0
        self.rate: Optional[float] = None
        self.n = 0

    def step(self) -> bool:
        """One physics step toward the target at the axis' speed; True when it arrived at this step."""
        rate = self.speed_per_step / (self.factor * self.slow)
        if rate != self.rate:
            self.origin, self.k, self.rate = self.pos, 0, rate
            self.n = _whole_steps(abs(self.target - self.pos) / rate)
        self.k += 1
        travel = math.copysign(self.k * rate, self.target - self.origin)
        self.pos = self.target if self.k >= self.n else self.origin + travel
        return self.pos == self.target


class _Sensor:
    def __init__(self, name: str, value: bool):
        self.name = name
        self.nominal = value
        self.value = value
        self.ok = True


class MockPlantAdapter(PlantAdapter):
    """`trace`: a `BoundedLog` of `(plant step, text)` -- every command change, arrival and injection -- bounded to the
    last `trace_lines` (the determinism check, C4, compares it)."""

    name = 'mock'

    def __init__(self, model: Mapping[str, object], trace_lines: int = DEFAULT_MAXLEN):
        allowed = {'seed', 'jitter', 'devices'}
        unknown = sorted(str(k) for k in model if k not in allowed)
        if unknown:
            raise InjectionError(f'mock: unknown key(s) {", ".join(unknown)} (allowed: {", ".join(sorted(allowed))})')
        seed = model.get('seed', 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise InjectionError(f'mock.seed must be an integer (got {seed!r})')
        jitter = model.get('jitter', 0.0)
        if isinstance(jitter, bool) or not isinstance(jitter, (int, float)) or not 0.0 <= jitter <= 0.5:
            raise InjectionError(f'mock.jitter must be in 0 .. 0.5 (got {jitter!r})')
        devices = model.get('devices', {}) or {}
        if not isinstance(devices, Mapping):
            raise InjectionError('mock.devices is a mapping device -> its model')
        self.seed = seed
        self.jitter = float(jitter)
        self._model = devices
        self._rng = random.Random(seed)
        self._actuators: Dict[str, _Actuator] = {}
        self._axes: Dict[str, _Axis] = {}
        self._sensors: Dict[str, _Sensor] = {}
        self._services: Tuple[str, ...] = ()
        self._hold: Dict[str, bool] = {}
        self._fault: Dict[str, Optional[str]] = {}
        self._consume_delay: Dict[str, Tuple[int, int]] = {}   # service -> (ms, physics steps)
        self._clock: Optional[ScanClock] = None
        self._step = 0
        self.trace: BoundedLog[Tuple[int, str]] = BoundedLog(trace_lines)

    # --- binding ------------------------------------------------------------------------------------------------------
    def bind(self, devices: Mapping[str, object], services: Sequence[str], clock: ScanClock) -> None:
        dt_ms = clock.ms(1)
        for name, dev in devices.items():
            spec = self._model.get(name, {}) or {}
            if not isinstance(spec, Mapping):
                raise InjectionError(f'mock.devices.{name}: a mapping')
            if isinstance(dev, BinaryActuator):
                _only(spec, {'close_ms', 'open_ms'}, name)
                close_ms = _positive(spec.get('close_ms', 300), f'mock.devices.{name}.close_ms')
                open_ms = _positive(spec.get('open_ms', 300), f'mock.devices.{name}.open_ms')
                self._actuators[name] = _Actuator(name, close_ms / dt_ms, open_ms / dt_ms)
            elif isinstance(dev, PositionAxis):
                _only(spec, {'speed', 'start'}, name)
                speed = _positive(spec.get('speed', 100.0), f'mock.devices.{name}.speed')
                start_name = spec.get('start', next(iter(dev.positions)))
                if not isinstance(start_name, str) or start_name not in dev.positions:
                    raise InjectionError(f'mock.devices.{name}.start: no position {start_name!r} '
                                         f'(positions: {", ".join(dev.positions)})')
                self._axes[name] = _Axis(name, speed * dt_ms / 1000.0, dev.positions[start_name])
            elif isinstance(dev, DigitalSensor):
                _only(spec, {'value'}, name)
                value = spec.get('value', True)
                if not isinstance(value, bool):
                    raise InjectionError(f'mock.devices.{name}.value must be true or false')
                self._sensors[name] = _Sensor(name, value)
        unknown = sorted(str(k) for k in set(self._model) - set(devices))
        if unknown:
            raise InjectionError(f'mock.devices: no such device(s) {", ".join(unknown)}')
        self._services = tuple(services)
        self._hold = {s: False for s in services}
        self._fault = {s: None for s in services}
        self._consume_delay = {}
        self._clock = clock

    def consume_delay_steps(self, service: str) -> int:
        return self._consume_delay.get(service, (0, 0))[1]

    # --- the time model -----------------------------------------------------------------------------------------------
    def advance(self, steps: int) -> None:
        for _ in range(steps):
            self._step += 1
            for act in self._actuators.values():
                if act.stuck or act.s == (1.0 if act.cmd else 0.0):
                    continue                  # stuck (its motion paused, not lost) or at its commanded limit
                if act.step():
                    self.trace.append((self._step, f'{act.name} {"closed" if act.cmd else "open"}'))
            for axis in self._axes.values():
                if axis.stuck or axis.target is None or axis.pos == axis.target:
                    continue
                if axis.step():
                    self.trace.append((self._step, f'{axis.name} at {axis.target:g}'))

    def read(self) -> Dict[str, object]:
        image: Dict[str, object] = {}
        for act in self._actuators.values():
            image[f'{act.name}.open'] = act.s == 0.0 or act.failed_limit == 'open'
            image[f'{act.name}.closed'] = act.s == 1.0 or act.failed_limit == 'closed'
        for axis in self._axes.values():
            image[f'{axis.name}.position'] = axis.pos
        for sensor in self._sensors.values():
            image[sensor.name] = sensor.value
            image[f'{sensor.name}.ok'] = sensor.ok
        for service in self._services:
            image[f'{service}.hold'] = self._hold[service]
            image[f'{service}.fault'] = self._fault[service] is not None
            image[f'{service}.fault_detail'] = self._fault[service] or ''
        return image

    def apply(self, outputs: Mapping[str, object]) -> None:
        for act in self._actuators.values():
            cmd = outputs.get(f'{act.name}.cmd')
            if isinstance(cmd, bool) and cmd != act.cmd:
                act.cmd = cmd
                act.factor = self._draw()
                act.stroke = None             # a new motion: it re-bases from where the actuator is
                self.trace.append((self._step, f'{act.name} cmd {"close" if cmd else "open"}'))
        for axis in self._axes.values():
            target = outputs.get(f'{axis.name}.target')
            if isinstance(target, (int, float)) and not isinstance(target, bool) and target != axis.target:
                axis.target = float(target)
                axis.factor = self._draw()
                axis.rate = None              # a new motion: it re-bases from where the axis is
                self.trace.append((self._step, f'{axis.name} cmd {target:g}'))

    def _draw(self) -> float:
        if self.jitter <= 0.0:
            return 1.0
        return 1.0 + self._rng.uniform(-self.jitter, self.jitter)

    # --- fault injection ----------------------------------------------------------------------------------------------
    def inject(self, fault: Mapping[str, object]) -> str:
        kind = fault.get('fault')
        if kind not in FAULTS:
            raise InjectionError(f'unknown fault {kind!r} (known: {", ".join(FAULTS)})')
        if kind == 'clear':
            for act in self._actuators.values():
                act.stuck, act.slow, act.failed_limit = False, 1.0, None
            for axis in self._axes.values():
                axis.stuck, axis.slow = False, 1.0
            for sensor in self._sensors.values():
                sensor.ok, sensor.value = True, sensor.nominal
            for service in self._services:
                self._hold[service] = False
                self._fault[service] = None
            self._consume_delay.clear()
            text = 'every injection cleared'
        elif kind == 'consume_delay':
            service = self._service(fault)
            ms = fault.get('ms')
            steps = consume_delay_steps(self._clock, ms)
            if steps:
                self._consume_delay[service] = (int(ms), steps)
                text = f'{service}: a new request is acted on {ms} ms after it is first seen'
            else:
                self._consume_delay.pop(service, None)
                text = f'{service}: a new request is acted on at once'
        elif kind == 'stuck_actuator':
            dev = self._movable(fault)
            dev.stuck = True
            text = f'{dev.name} stuck: it will not move'
        elif kind == 'slow':
            dev = self._movable(fault)
            factor = fault.get('factor')
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 1.0 <= factor <= 100.0:
                raise InjectionError('slow: factor in 1 .. 100')
            dev.slow = float(factor)
            text = f'{dev.name} {factor:g}x slower'
        elif kind == 'failed_sensor':
            name = _name(fault.get('device'))
            if name in self._sensors:
                self._sensors[name].ok = False
                text = f'{name}: sensor failed'
            elif name in self._actuators:
                signal = fault.get('signal', 'closed')
                if signal not in ('open', 'closed'):
                    raise InjectionError('failed_sensor on an actuator: signal open | closed')
                self._actuators[name].failed_limit = signal
                text = f'{name}: {signal} limit sensor stuck on'
            else:
                raise InjectionError(f'failed_sensor: no sensor or actuator {name!r}')
        elif kind == 'sensor_value':
            name = _name(fault.get('device'))
            value = fault.get('value')
            if name not in self._sensors or not isinstance(value, bool):
                raise InjectionError('sensor_value: device <a digital sensor>, value true|false')
            self._sensors[name].value = value
            text = f'{name} reads {"on" if value else "off"}'
        elif kind == 'station_fault':
            service = self._service(fault)
            detail = fault.get('detail') or 'station fault (injected)'
            self._fault[service] = str(detail)
            text = f'{service}: station fault'
        else:   # hold
            service = self._service(fault)
            self._hold[service] = True
            text = f'{service}: held (no DONE until cleared)'
        self.trace.append((self._step, f'inject {text}'))
        return text

    def _movable(self, fault: Mapping[str, object]):
        name = _name(fault.get('device'))
        if name in self._actuators:
            return self._actuators[name]
        if name in self._axes:
            return self._axes[name]
        raise InjectionError(f'{fault.get("fault")}: no actuator or axis {name!r} '
                             f'(devices: {", ".join(sorted(list(self._actuators) + list(self._axes)))})')

    def _service(self, fault: Mapping[str, object]) -> str:
        service = fault.get('service')
        if not isinstance(service, str) or service not in self._services:
            raise InjectionError(f'{fault.get("fault")}: no service {service!r} '
                                 f'(services: {", ".join(self._services)})')
        return str(service)

    def describe(self) -> Dict[str, object]:
        return {
            'plant': self.name,
            'seed': self.seed,
            'jitter': self.jitter,
            'step': self._step,
            'injections': {
                'stuck': sorted([a.name for a in self._actuators.values() if a.stuck]
                                + [x.name for x in self._axes.values() if x.stuck]),
                'slow': {d.name: d.slow for d in list(self._actuators.values()) + list(self._axes.values())
                         if d.slow != 1.0},
                'failed_sensors': sorted([s.name for s in self._sensors.values() if not s.ok]
                                         + [f'{a.name}.{a.failed_limit}' for a in self._actuators.values()
                                            if a.failed_limit]),
                'held': sorted(s for s, v in self._hold.items() if v),
                'station_faults': sorted(s for s, v in self._fault.items() if v),
                'consume_delay_ms': {s: ms for s, (ms, _) in sorted(self._consume_delay.items())},
            },
        }


def _name(value: object) -> str:
    """A device name from an operator command: a string, or '' (never found) -- an unhashable value never crashes."""
    return value if isinstance(value, str) else ''


def _only(spec: Mapping[str, object], allowed: set, name: str) -> None:
    unknown = sorted(str(k) for k in spec if k not in allowed)
    if unknown:
        raise InjectionError(f'mock.devices.{name}: unknown key(s) {", ".join(unknown)} '
                             f'(allowed: {", ".join(sorted(allowed))})')


def _positive(value: object, at: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not value > 0:
        raise InjectionError(f'{at} must be a finite number > 0 (got {value!r})')
    return float(value)
