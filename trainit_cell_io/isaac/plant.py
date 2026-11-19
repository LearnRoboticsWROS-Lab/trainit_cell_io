"""The `IsaacPlantAdapter` (P0B_DESIGN.md §4): the cell's devices bound to the joints of Isaac Sim articulations.

PURE: it imports no omni / isaacsim / pxr / carb / rclpy. It sees the simulator only through a small `JointAccess`
(joint read-backs, position-drive targets, a body position in another body's frame), which the Kit runner implements
over Isaac's articulation API (`trainit_cell_io.isaac.kit`) and a unit test fakes. The station LOGIC is the core's,
unchanged: the same devices, chains and `ProcessDevice`s as on the mock plant; only this binding differs (the parity
the P0.b gate measures).

`cell_io.yaml`, `plant: isaac` (values in SI at the joint; axis speeds in the axis unit per second):

    isaac:
      stage: neutral_station                  # the stage the runner builds
      articulations: {station: /World/neutral_station/station, indexer: /World/neutral_station/indexer}
      devices:
        clamp: {articulation: station, joint: jaw_joint, open_m: 0.0, closed_m: 0.018, band_m: 0.003, stroke_ms: 300}
        table: {articulation: station, joint: turntable_joint, speed: 180.0}        # deg/s (a revolute joint)
        index: {articulation: indexer, joint: carriage_joint, speed: 200.0}         # mm/s (a prismatic joint)
        part_present: {part: /World/.../part, frame: /World/.../turntable, nest_m: [0.1, 0.0, 0.04], tolerance_m: 0.01}

- a BinaryActuator is a PRISMATIC joint with an open and a closed position; its two limit sensors are in-band
  read-backs (|q - open| <= band, |q - closed| <= band): a jaw stopped by the part it clamps, within the band of
  `closed`, reads closed; a jaw that never gets there never does (ACTUATOR_TIMEOUT);
- a PositionAxis maps its unit to the joint's: deg <-> rad (revolute), mm <-> m (prismatic);
- a DigitalSensor is computed from the position of a body (the part) in the frame of another (the turntable): ON when
  within `tolerance_m` of the nest point;
- the drive target is never stepped at once: it RAMPS toward the commanded position at the device's speed (an axis'
  `speed`; an actuator's stroke over `stroke_ms`), one increment per scan, as a servo's trajectory generator or a
  throttled cylinder would -- the joint follows under physics, and only its read-back completes a step.

Fault injection (simulation control only; the same words as the mock):
    stuck_actuator  device            the device does not move: its drive target is HELD where it is (for an actuator:
                                      the valve does not switch)
    slow            device factor     the device's ramp is `factor` times slower
    failed_sensor   device [signal]   a digital sensor's diagnosis fails; for an actuator a limit sensor stuck ON
    sensor_value    device value      a digital sensor reads `value` whatever the part does
    station_fault   service [detail]  the station reports a fault (level, until `clear`)
    hold            service           the station holds its cycle (BUSY, timers stopped) until `clear`
    consume_delay   service ms        a NEW request image is acted on only `ms` after it was first seen
    clear                             every injection removed (a FAULT stays until the station's RESET)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Sequence, Set, Tuple

from ..core.clock import ScanClock
from ..core.config import ConfigError
from ..core.devices import BinaryActuator, DigitalSensor, PositionAxis
from ..core.mock_plant import consume_delay_steps
from ..core.plant import InjectionError, PlantAdapter

FAULTS = ('stuck_actuator', 'failed_sensor', 'sensor_value', 'station_fault', 'hold', 'slow', 'consume_delay',
          'clear')
Vec3 = Tuple[float, float, float]


class JointAccess(Protocol):
    """The adapter's only view of the simulator."""

    def joint_positions(self) -> Mapping[str, float]:
        """Every bound joint's measured position by key `<articulation>/<joint>`: rad (revolute) or m (prismatic)."""

    def set_joint_targets(self, targets: Mapping[str, float]) -> None:
        """Position-drive targets by key, rad or m, for the next physics steps."""

    def body_position_in(self, body: str, frame: str) -> Optional[Vec3]:
        """The position (m) of the rigid body `body` in the frame of the rigid body `frame`, or None if unknown."""


@dataclass(frozen=True)
class ActuatorBinding:
    device: str
    articulation: str
    joint: str
    open_m: float
    closed_m: float
    band_m: float
    stroke_ms: int

    @property
    def key(self) -> str:
        return f'{self.articulation}/{self.joint}'

    @property
    def speed(self) -> float:          # m/s
        return abs(self.closed_m - self.open_m) / (self.stroke_ms / 1000.0)


@dataclass(frozen=True)
class AxisBinding:
    device: str
    articulation: str
    joint: str
    speed: float                       # the axis unit per second

    @property
    def key(self) -> str:
        return f'{self.articulation}/{self.joint}'


@dataclass(frozen=True)
class SensorBinding:
    device: str
    part: str
    frame: str
    nest_m: Vec3
    tolerance_m: float


def _bad(at: str, text: str) -> ConfigError:
    return ConfigError(f'isaac{"." + at if at else ""}: {text}')


def _keys(spec: Mapping, required: Set[str], optional: Set[str], at: str) -> None:
    if not isinstance(spec, Mapping):
        raise _bad(at, 'a mapping')
    unknown = sorted(str(k) for k in spec if k not in required | optional)
    if unknown:
        raise _bad(at, f'unknown key(s) {", ".join(unknown)} (allowed: {", ".join(sorted(required | optional))})')
    missing = sorted(required - set(spec))
    if missing:
        raise _bad(at, f'missing {", ".join(missing)}')


def _number(value: object, at: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _bad(at, f'a finite number, got {value!r}')
    if positive and not value > 0:
        raise _bad(at, f'must be > 0, got {value!r}')
    return float(value)


def _name(value: object, at: str) -> str:
    if not isinstance(value, str) or not value:
        raise _bad(at, f'a non-empty name, got {value!r}')
    return value


def _prim(value: object, at: str) -> str:
    if not isinstance(value, str) or not value.startswith('/') or value.endswith('/') or '//' in value:
        raise _bad(at, f'an absolute prim path such as /World/station, got {value!r}')
    return value


def point_in_frame(point: Sequence[float], frame_position: Sequence[float],
                   frame_orientation_wxyz: Sequence[float]) -> Vec3:
    """A world point expressed in a frame given by its world pose (orientation a unit quaternion w, x, y, z)."""
    w, x, y, z = (float(v) for v in frame_orientation_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not norm > 0:
        raise ValueError('a frame orientation needs a non-zero quaternion')
    w, x, y, z = w / norm, -x / norm, -y / norm, -z / norm          # the inverse rotation (the conjugate)
    vx, vy, vz = (float(p) - float(f) for p, f in zip(point, frame_position))
    # v' = v + 2 w (u x v) + 2 u x (u x v), u = (x, y, z)
    cx, cy, cz = y * vz - z * vy, z * vx - x * vz, x * vy - y * vx
    return (vx + 2 * (w * cx + y * cz - z * cy), vy + 2 * (w * cy + z * cx - x * cz),
            vz + 2 * (w * cz + x * cy - y * cx))


def _finite(value: object) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


class IsaacPlantAdapter(PlantAdapter):
    name = 'isaac'

    def __init__(self, binding: Mapping[str, object], access: Optional[JointAccess] = None):
        _keys(binding, {'articulations', 'devices'}, {'stage'}, '')
        self.stage = binding.get('stage')
        if self.stage is not None:
            _name(self.stage, 'stage')
        arts = binding['articulations']
        if not isinstance(arts, Mapping) or not arts:
            raise _bad('articulations', 'a non-empty mapping name -> prim path')
        self.articulations: Dict[str, str] = {}
        for art, path in arts.items():
            self.articulations[_name(art, 'articulations')] = _prim(path, f'articulations.{art}')
        devices = binding['devices']
        if not isinstance(devices, Mapping) or not devices:
            raise _bad('devices', 'a non-empty mapping device -> its binding')
        self._raw_devices = dict(devices)
        self.actuators: Dict[str, ActuatorBinding] = {}
        self.axes: Dict[str, AxisBinding] = {}
        self.sensors: Dict[str, SensorBinding] = {}
        self._access = access
        self._clock: Optional[ScanClock] = None
        self._scan_s = 0.0
        self._services: Tuple[str, ...] = ()
        self._units: Dict[str, str] = {}
        # run-time state
        self._last: Dict[str, Optional[float]] = {}       # the latest read-back per joint key (joint units)
        self._desired: Dict[str, float] = {}              # where each drive is commanded to go (joint units)
        self._ramp: Dict[str, float] = {}                 # the drive target sent now (joint units)
        self._stuck: Set[str] = set()                     # device names
        self._slow: Dict[str, float] = {}
        self._failed_limit: Dict[str, str] = {}           # actuator -> 'open' | 'closed'
        self._failed_sensor: Set[str] = set()
        self._forced: Dict[str, bool] = {}
        self._hold: Dict[str, bool] = {}
        self._fault: Dict[str, Optional[str]] = {}
        self._consume_delay: Dict[str, Tuple[int, int]] = {}

    # --- what the Kit glue must provide -------------------------------------------------------------------------------
    def attach(self, access: JointAccess) -> None:
        self._access = access

    def joints(self) -> Dict[str, Tuple[str, str]]:
        """The joints to read and drive: key -> (articulation prim path, joint name). Valid after `bind`."""
        out = {}
        for b in list(self.actuators.values()) + list(self.axes.values()):
            out[b.key] = (self.articulations[b.articulation], b.joint)
        return out

    def bodies(self) -> Set[str]:
        """The rigid bodies whose poses the sensors need (prim paths). Valid after `bind`."""
        return {p for s in self.sensors.values() for p in (s.part, s.frame)}

    # --- binding ------------------------------------------------------------------------------------------------------
    def bind(self, devices: Mapping[str, object], services: Sequence[str], clock: ScanClock) -> None:
        unknown = sorted(set(self._raw_devices) - set(devices))
        if unknown:
            raise _bad('devices', f'no such device(s) in the cell: {", ".join(map(str, unknown))}')
        used: Dict[str, str] = {}
        for name, dev in devices.items():
            spec = self._raw_devices.get(name)
            at = f'devices.{name}'
            if spec is None:
                raise _bad(at, f'the device has no Isaac binding (a {getattr(dev, "kind", "device")})')
            if isinstance(dev, BinaryActuator):
                _keys(spec, {'articulation', 'joint', 'open_m', 'closed_m', 'band_m', 'stroke_ms'}, set(), at)
                b = ActuatorBinding(name, self._articulation(spec['articulation'], at), _name(spec['joint'], at),
                                    _number(spec['open_m'], f'{at}.open_m'), _number(spec['closed_m'],
                                                                                     f'{at}.closed_m'),
                                    _number(spec['band_m'], f'{at}.band_m', positive=True),
                                    self._stroke_ms(spec['stroke_ms'], f'{at}.stroke_ms'))
                if not b.band_m < abs(b.closed_m - b.open_m) / 2:
                    raise _bad(at, f'band_m {b.band_m:g} must be under half the stroke ({abs(b.closed_m - b.open_m):g} '
                                   'm): the open and closed limit windows must not overlap')
                self.actuators[name] = b
                key = b.key
            elif isinstance(dev, PositionAxis):
                _keys(spec, {'articulation', 'joint', 'speed'}, set(), at)
                b2 = AxisBinding(name, self._articulation(spec['articulation'], at), _name(spec['joint'], at),
                                 _number(spec['speed'], f'{at}.speed', positive=True))
                if dev.unit not in ('deg', 'mm'):
                    raise _bad(at, f'axis unit {dev.unit!r}: deg (a revolute joint) or mm (a prismatic joint)')
                self.axes[name] = b2
                self._units[name] = dev.unit
                key = b2.key
            elif isinstance(dev, DigitalSensor):
                _keys(spec, {'part', 'frame', 'nest_m', 'tolerance_m'}, set(), at)
                nest = spec['nest_m']
                if not isinstance(nest, (list, tuple)) or len(nest) != 3:
                    raise _bad(f'{at}.nest_m', f'three numbers (x, y, z in metres), got {nest!r}')
                self.sensors[name] = SensorBinding(
                    name, _prim(spec['part'], f'{at}.part'), _prim(spec['frame'], f'{at}.frame'),
                    tuple(_number(v, f'{at}.nest_m') for v in nest),
                    _number(spec['tolerance_m'], f'{at}.tolerance_m', positive=True))
                continue
            else:
                raise _bad(at, f'unsupported device type {type(dev).__name__}')
            if key in used:
                raise _bad(at, f'joint {key} already drives device {used[key]!r} (one device per joint)')
            used[key] = name
        self._clock = clock
        self._scan_s = clock.seconds(clock.scan_every_steps)
        self._services = tuple(services)
        self._hold = {s: False for s in services}
        self._fault = {s: None for s in services}

    def _articulation(self, value: object, at: str) -> str:
        name = _name(value, f'{at}.articulation')
        if name not in self.articulations:
            raise _bad(f'{at}.articulation',
                       f'no articulation {name!r} (declared: {", ".join(sorted(self.articulations))})')
        return name

    @staticmethod
    def _stroke_ms(value: object, at: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 600000:
            raise _bad(at, f'an integer number of milliseconds in 1..600000, got {value!r}')
        return value

    # --- unit conversions ---------------------------------------------------------------------------------------------
    def _to_joint(self, axis: str, value: float) -> float:
        return math.radians(value) if self._units[axis] == 'deg' else value / 1000.0

    def _from_joint(self, axis: str, value: float) -> float:
        return math.degrees(value) if self._units[axis] == 'deg' else value * 1000.0

    def _joint_speed(self, device: str) -> float:
        if device in self.actuators:
            return self.actuators[device].speed
        return self._to_joint(device, self.axes[device].speed)

    # --- the scan: inputs ---------------------------------------------------------------------------------------------
    def _need_access(self) -> JointAccess:
        if self._access is None:
            raise RuntimeError('the Isaac plant has no joint access attached (attach() before the first scan)')
        return self._access

    def read(self) -> Dict[str, object]:
        access = self._need_access()
        positions = access.joint_positions()
        image: Dict[str, object] = {}
        for name, b in self.actuators.items():
            q = _finite(positions.get(b.key))
            self._last[b.key] = q
            failed = self._failed_limit.get(name)
            if q is None:
                image[f'{name}.open'] = True if failed == 'open' else None
                image[f'{name}.closed'] = True if failed == 'closed' else None
                continue
            image[f'{name}.open'] = abs(q - b.open_m) <= b.band_m or failed == 'open'
            image[f'{name}.closed'] = abs(q - b.closed_m) <= b.band_m or failed == 'closed'
        for name, b in self.axes.items():
            q = _finite(positions.get(b.key))
            self._last[b.key] = q
            image[f'{name}.position'] = None if q is None else self._from_joint(name, q)
        for name, s in self.sensors.items():
            if name in self._forced:
                value: Optional[bool] = self._forced[name]
            else:
                where = access.body_position_in(s.part, s.frame)
                value = where is not None and all(_finite(v) is not None for v in where) and math.dist(
                    tuple(float(v) for v in where), s.nest_m) <= s.tolerance_m
            image[name] = bool(value)
            image[f'{name}.ok'] = name not in self._failed_sensor
        for service in self._services:
            image[f'{service}.hold'] = self._hold[service]
            image[f'{service}.fault'] = self._fault[service] is not None
            image[f'{service}.fault_detail'] = self._fault[service] or ''
        return image

    # --- the scan: outputs (one ramp increment per scan) --------------------------------------------------------------
    def apply(self, outputs: Mapping[str, object]) -> None:
        access = self._need_access()
        for name, b in self.actuators.items():
            cmd = outputs.get(f'{name}.cmd')
            if isinstance(cmd, bool) and name not in self._stuck:
                self._desired[b.key] = b.closed_m if cmd else b.open_m
        for name, b in self.axes.items():
            target = _finite(outputs.get(f'{name}.target'))
            if target is not None and name not in self._stuck:
                self._desired[b.key] = self._to_joint(name, target)
        targets: Dict[str, float] = {}
        for name, b in list(self.actuators.items()) + list(self.axes.items()):
            key = b.key
            if key not in self._desired:
                continue                      # never commanded: the drive keeps the stage's own target
            current = self._ramp.get(key)
            if current is None:
                last = self._last.get(key)
                current = last if last is not None else self._desired[key]
            if name not in self._stuck:
                step = self._joint_speed(name) * self._scan_s / self._slow.get(name, 1.0)
                delta = self._desired[key] - current
                current = self._desired[key] if abs(delta) <= step else current + math.copysign(step, delta)
            self._ramp[key] = current
            targets[key] = current
        if targets:
            access.set_joint_targets(targets)

    # --- fault injection ----------------------------------------------------------------------------------------------
    def inject(self, fault: Mapping[str, object]) -> str:
        kind = fault.get('fault')
        if kind not in FAULTS:
            raise InjectionError(f'unknown fault {kind!r} (known: {", ".join(FAULTS)})')
        if kind == 'clear':
            self._stuck.clear()
            self._slow.clear()
            self._failed_limit.clear()
            self._failed_sensor.clear()
            self._forced.clear()
            self._consume_delay.clear()
            for service in self._services:
                self._hold[service] = False
                self._fault[service] = None
            return 'every injection cleared'
        if kind == 'stuck_actuator':
            name = self._movable(fault)
            self._stuck.add(name)
            return f'{name} stuck: its drive target is held where it is'
        if kind == 'slow':
            name = self._movable(fault)
            factor = fault.get('factor')
            if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 1.0 <= factor <= 100.0:
                raise InjectionError('slow: factor in 1 .. 100')
            self._slow[name] = float(factor)
            return f'{name} {factor:g}x slower'
        if kind == 'failed_sensor':
            name = fault.get('device')
            if isinstance(name, str) and name in self.sensors:
                self._failed_sensor.add(name)
                return f'{name}: sensor failed'
            if isinstance(name, str) and name in self.actuators:
                signal = fault.get('signal', 'closed')
                if signal not in ('open', 'closed'):
                    raise InjectionError('failed_sensor on an actuator: signal open | closed')
                self._failed_limit[name] = signal
                return f'{name}: {signal} limit sensor stuck on'
            raise InjectionError(f'failed_sensor: no sensor or actuator {name!r}')
        if kind == 'sensor_value':
            name, value = fault.get('device'), fault.get('value')
            if not isinstance(name, str) or name not in self.sensors or not isinstance(value, bool):
                raise InjectionError('sensor_value: device <a digital sensor>, value true|false')
            self._forced[name] = value
            return f'{name} reads {"on" if value else "off"}'
        if kind == 'consume_delay':
            service = self._service(fault)
            ms = fault.get('ms')
            steps = consume_delay_steps(self._clock, ms)
            if steps:
                self._consume_delay[service] = (int(ms), steps)
                return f'{service}: a new request is acted on {ms} ms after it is first seen'
            self._consume_delay.pop(service, None)
            return f'{service}: a new request is acted on at once'
        service = self._service(fault)
        if kind == 'station_fault':
            self._fault[service] = str(fault.get('detail') or 'station fault (injected)')
            return f'{service}: station fault'
        self._hold[service] = True
        return f'{service}: held (no DONE until cleared)'

    def consume_delay_steps(self, service: str) -> int:
        return self._consume_delay.get(service, (0, 0))[1]

    def _movable(self, fault: Mapping[str, object]) -> str:
        name = fault.get('device')
        if isinstance(name, str) and (name in self.actuators or name in self.axes):
            return name
        raise InjectionError(f'{fault.get("fault")}: no actuator or axis {name!r} '
                             f'(devices: {", ".join(sorted(list(self.actuators) + list(self.axes)))})')

    def _service(self, fault: Mapping[str, object]) -> str:
        service = fault.get('service')
        if not isinstance(service, str) or service not in self._services:
            raise InjectionError(f'{fault.get("fault")}: no service {service!r} '
                                 f'(services: {", ".join(self._services)})')
        return service

    # --- the operator's view ------------------------------------------------------------------------------------------
    def joint_position(self, device: str) -> Optional[float]:
        """The latest read-back of a device's joint (joint units), for reports."""
        b = self.actuators.get(device) or self.axes.get(device)
        return None if b is None else self._last.get(b.key)

    def describe(self) -> Dict[str, object]:
        return {
            'plant': self.name,
            'stage': self.stage,
            'injections': {
                'stuck': sorted(self._stuck),
                'slow': dict(sorted(self._slow.items())),
                'failed_sensors': sorted(list(self._failed_sensor)
                                         + [f'{a}.{s}' for a, s in self._failed_limit.items()]),
                'forced_sensors': dict(sorted(self._forced.items())),
                'held': sorted(s for s, v in self._hold.items() if v),
                'station_faults': sorted(s for s, v in self._fault.items() if v),
                'consume_delay_ms': {s: ms for s, (ms, _) in sorted(self._consume_delay.items())},
            },
            'joints': {key: {'position': self._last.get(key), 'target': self._ramp.get(key)}
                       for key in sorted(self._last)},
        }
