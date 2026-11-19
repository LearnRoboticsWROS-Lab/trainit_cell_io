"""The `python_logic` backend: a service's cycle (and its return-to-home) as a chain of device steps.

A station sequence is Python, never the robot's: the robot only requests the service and waits (P0B_DESIGN.md §3).
The chain is declarative (`cell_io.yaml`):

    cycle:
      - {require: part_present}        # the sensor must be healthy and ON
      - {close: clamp}                 # a BinaryActuator to its closed limit, within its timeout
      - {move: table, to: turned}      # a PositionAxis to a named position, within its motion timeout
      - {open: clamp}
      - {move: table, to: load}

Any other sequence (a station whose logic is code, e.g. a private twin's own cycle class) implements the same
`Sequence` protocol: `start()`, `scan(inputs, held, steps) -> StepResult`, `describe()`.

Timers count physics steps and stop while the station is HELD (its own pause input), so a held station is BUSY for as
long as it is held -- never a fault, never a completion.

A motion step's timeout is EXACT: the step's timer counts the physics steps its device has had since its command
reached the plant (the command goes out at the end of the scan that entered the step), and each scan first adds the
scan's steps, then checks. A device that reads at its target when `timeout_ms` has elapsed passes; one that does not
faults ACTUATOR_TIMEOUT at that very scan -- never a scan later (`cell_io.yaml` makes a device timeout a whole number
of scans, so T always falls on a scan).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Protocol, Sequence as Seq, Union

from .contract import FAULT_ACTUATOR_TIMEOUT, FAULT_SENSOR_FAULT, FAULT_STATION_FAULT
from .devices import BinaryActuator, DigitalSensor, PositionAxis
from .errors import ConfigError

RUNNING = 'running'
DONE = 'done'


@dataclass(frozen=True)
class Fault:
    code: int
    detail: str


StepResult = Union[str, Fault]   # RUNNING, DONE or a Fault


class Sequence(Protocol):
    def start(self) -> None: ...

    def scan(self, inputs: Mapping[str, object], held: bool, steps: int) -> StepResult: ...

    def describe(self) -> str: ...


class LogicError(ConfigError):
    """A chain the cell refuses (an unknown step, device or position): a configuration refusal."""


class _ActuatorStep:
    def __init__(self, device: BinaryActuator, close: bool):
        self.device = device
        self.close = close
        self.elapsed = 0

    def enter(self) -> None:
        self.elapsed = 0
        self.device.command(self.close)

    def scan(self, steps: int) -> StepResult:
        self.elapsed += steps            # the physics steps the device has had since its command (this scan's included)
        problem = self.device.implausible()
        if problem is not None:
            return Fault(FAULT_SENSOR_FAULT, problem)
        if self.device.at(self.close):
            return DONE                  # at its limit when the timeout elapses: still a pass
        if self.elapsed >= self.device.timeout_steps:
            return Fault(FAULT_ACTUATOR_TIMEOUT,
                         f'{self.device.name} did not reach {self.target()} within {self.device.timeout_ms} ms')
        return RUNNING

    def target(self) -> str:
        return 'closed' if self.close else 'open'

    def describe(self) -> str:
        return f'{self.device.name} {"closing" if self.close else "opening"}'


class _AxisStep:
    def __init__(self, device: PositionAxis, position: str):
        self.device = device
        self.position = position
        self.elapsed = 0

    def enter(self) -> None:
        self.elapsed = 0
        self.device.command(self.position)

    def scan(self, steps: int) -> StepResult:
        self.elapsed += steps            # the physics steps the axis has had since its command (this scan's included)
        problem = self.device.implausible()
        if problem is not None:
            return Fault(FAULT_SENSOR_FAULT, problem)
        if self.device.at(self.position):
            return DONE                  # in position when the timeout elapses: still a pass
        if self.elapsed >= self.device.timeout_steps:
            where = self.device.describe()
            return Fault(FAULT_ACTUATOR_TIMEOUT,
                         f'{self.device.name} did not reach {self.position!r} '
                         f'({self.device.positions[self.position]:g} {self.device.unit}) within '
                         f'{self.device.timeout_ms} ms: {where}')
        return RUNNING

    def describe(self) -> str:
        return f'{self.device.name} moving to {self.position}'


class _RequireStep:
    def __init__(self, device: DigitalSensor):
        self.device = device

    def enter(self) -> None:
        pass

    def scan(self, steps: int) -> StepResult:
        if not self.device.ok:
            return Fault(FAULT_SENSOR_FAULT, f'{self.device.name}: sensor fault')
        if self.device.value is not True:
            return Fault(FAULT_STATION_FAULT, f'{self.device.name} is off')
        return DONE

    def describe(self) -> str:
        return f'checking {self.device.name}'


class DeviceChain:
    """A cycle (or a return-to-home) as an ordered chain of device steps."""

    def __init__(self, name: str, steps: List[object]):
        self.name = name
        self._steps = steps
        self._index = 0
        self._entered = False

    def start(self) -> None:
        self._index = 0
        self._entered = False

    def scan(self, inputs: Mapping[str, object], held: bool, steps: int) -> StepResult:
        # A chain step may complete in the very scan it starts (a device already there): advance through every step
        # that is complete in this scan, never more than the chain's length.
        for _ in range(len(self._steps) + 1):
            if self._index >= len(self._steps):
                return DONE
            if held:
                return RUNNING              # a held station keeps its outputs and its timers stop
            step = self._steps[self._index]
            if not self._entered:
                step.enter()
                self._entered = True
                return RUNNING              # the command reaches the plant after this scan
            result = step.scan(steps)
            if result == DONE:
                self._index += 1
                self._entered = False
                continue
            return result
        return RUNNING

    def describe(self) -> str:
        if self._index >= len(self._steps):
            return f'{self.name} complete'
        return self._steps[self._index].describe()


def build_chain(name: str, spec: Seq[Mapping[str, object]], devices: Mapping[str, object]) -> DeviceChain:
    """A `DeviceChain` from its `cell_io.yaml` list; refuses an unknown step, device or position (readable)."""
    if not isinstance(spec, (list, tuple)) or not spec:
        raise LogicError(f'{name}: a non-empty list of steps')
    steps: List[object] = []
    for i, item in enumerate(spec):
        where = f'{name} step {i + 1}'
        if not isinstance(item, Mapping):
            raise LogicError(f'{where}: a mapping such as {{close: clamp}}, got {item!r}')
        keys = set(item)
        if keys == {'close'} or keys == {'open'}:
            verb = 'close' if 'close' in keys else 'open'
            dev = _device(devices, item[verb], BinaryActuator, where, 'binary_actuator')
            steps.append(_ActuatorStep(dev, close=(verb == 'close')))
        elif keys == {'move', 'to'}:
            dev = _device(devices, item['move'], PositionAxis, where, 'position_axis')
            if not isinstance(item['to'], str) or item['to'] not in dev.positions:
                raise LogicError(f'{where}: axis {dev.name!r} has no position {item["to"]!r} '
                                 f'(its positions: {", ".join(sorted(dev.positions))})')
            steps.append(_AxisStep(dev, str(item['to'])))
        elif keys == {'require'}:
            dev = _device(devices, item['require'], DigitalSensor, where, 'digital_sensor')
            steps.append(_RequireStep(dev))
        else:
            raise LogicError(f'{where}: unknown step {dict(item)!r} (use close / open / move+to / require)')
    return DeviceChain(name, steps)


def _device(devices: Mapping[str, object], name: object, cls: type, where: str, kind: str):
    if not isinstance(name, str) or name not in devices:
        known = ', '.join(sorted(devices)) or 'none'
        raise LogicError(f'{where}: unknown device {name!r} (declared: {known})')
    dev = devices[name]
    if not isinstance(dev, cls):
        raise LogicError(f'{where}: {name!r} is a {getattr(dev, "kind", type(dev).__name__)}, not a {kind}')
    return dev


def chain_devices(chain: DeviceChain) -> Dict[str, object]:
    """The devices a chain drives or reads (for the one-service-per-device check)."""
    return {s.device.name: s.device for s in chain._steps}   # noqa: SLF001 -- same module family
