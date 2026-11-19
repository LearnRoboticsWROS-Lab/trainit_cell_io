"""`cell_io.yaml` -- the CELL's machine-simulation file (P0B_DESIGN.md §4), hand-written until M5a / CellSpec.

It belongs to the cell, never to the robot project: the robot application names a machine service by its logical id
only (`turner`), and this file says what serves it. Strict: an unknown key, a wrong type or a duration that is not a
whole number of physics steps is refused with a readable message.

    cell_io: 1                        # the format marker
    control_mode: simulation          # simulation | external_plc (external_plc: the logic runs on a PLC, P0.c)
    plant: mock                       # mock | isaac (simulation only)
    robot_slot: ''                    # the robot slot the services serve (ADR-0023); '' = the root namespace
    scan: {physics_dt_ms: 5, scan_every_steps: 2, status_period_ms: 100}
    devices:
      clamp: {type: binary_actuator, timeout_ms: 2000}
      table: {type: position_axis, unit: deg, positions: {load: 0.0, turned: 180.0}, band: 1.0, timeout_ms: 4000}
      part_present: {type: digital_sensor}
    services:
      turner:
        description: turn the part over
        logic: python_logic
        cycle: [{require: part_present}, {close: clamp}, {move: table, to: turned}, {open: clamp},
                {move: table, to: load}]
        home: [{open: clamp}, {move: table, to: load}]
    mock: {...}                       # the mock plant's time model (read when plant: mock)
    isaac: {...}                      # the Isaac plant's binding (read when plant: isaac)

`control_mode: external_plc` refuses any service (a `ProcessDevice`) and any logic backend: in that mode the machine
logic runs on the PLC and the cell simulates nothing (ADR-0013 Amendment P0 item 4; case C3).

`logic: code` -- a station whose sequence is CODE (e.g. a twin's own cycle class, `logic.Sequence`): the service lists
no `cycle` / `home`; the program that runs the cell hands the sequences to `Cell(..., sequences={id: factory})`, and a
cell built without them refuses the file (so `cell_io_node` and the neutral stage runner refuse it, exit 2). The
service's rules are the same `ProcessDevice` (P0B_DESIGN.md §3.3); only where its cycle comes from differs.

`parse()` validates EVERYTHING the cell would refuse later -- every step of every chain (its shape, its device and the
device's type, an axis position), finite numbers, the status period's bound -- so a refused file is always a
`ConfigError` at load: the executables exit 2 with its message, never a traceback (`ClockError`, `DeviceError` and
`LogicError` are `ConfigError`s too).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .clock import ClockError, ScanClock, parse_dt
from .errors import ConfigError

FORMAT_VERSION = 1
CONTROL_MODES = ('simulation', 'external_plc')
PLANTS = ('mock', 'isaac')
LOGIC_BACKENDS = ('python_logic', 'code')
DEVICE_TYPES = ('binary_actuator', 'position_axis', 'digital_sensor')
# The robot's request synchronises on TWO heartbeat changes of the source within its `channel_status_max_age_ms`
# (default 1000 ms, P0B_DESIGN.md §3.4): a status period above half of it can never give two changes in time.
MAX_STATUS_PERIOD_MS = 500
# Which device type each chain verb drives or reads (the same rule as logic.build_chain).
STEP_DEVICE_TYPES = {'close': 'binary_actuator', 'open': 'binary_actuator', 'move': 'position_axis',
                     'require': 'digital_sensor'}
_ID = re.compile(r'^[a-z][a-z0-9_]{0,62}$')
_SLOT = re.compile(r'^([a-z][a-z0-9_]*(/[a-z][a-z0-9_]*)*)?$')

__all__ = ['FORMAT_VERSION', 'CONTROL_MODES', 'PLANTS', 'LOGIC_BACKENDS', 'DEVICE_TYPES', 'MAX_STATUS_PERIOD_MS',
           'ConfigError', 'DeviceSpec', 'ServiceSpec', 'CellConfig', 'load', 'parse']


@dataclass
class DeviceSpec:
    name: str
    type: str
    timeout_ms: Optional[int] = None
    unit: str = ''
    positions: Dict[str, float] = field(default_factory=dict)
    band: Optional[float] = None


@dataclass
class ServiceSpec:
    id: str
    description: str
    logic: str
    cycle: List[Mapping[str, Any]]
    home: List[Mapping[str, Any]]


@dataclass
class CellConfig:
    control_mode: str
    plant: Optional[str]
    robot_slot: str
    clock: Optional[ScanClock]
    devices: Dict[str, DeviceSpec]
    services: Dict[str, ServiceSpec]
    mock: Dict[str, Any]
    isaac: Dict[str, Any]
    source_path: Optional[str] = None

    def topic(self, service: str, leaf: str) -> str:
        """The canonical topic of a service in the robot's slot: `/<slot>/<service>/<leaf>` (or `/<service>/<leaf>`)."""
        prefix = f'/{self.robot_slot}' if self.robot_slot else ''
        return f'{prefix}/{service}/{leaf}'


def load(path: str) -> CellConfig:
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f'cannot read {path}: {exc}') from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f'{path} is not valid YAML: {exc}') from exc
    cfg = parse(data, where=path)
    cfg.source_path = path
    return cfg


def parse(data: Any, where: str = 'cell_io.yaml') -> CellConfig:
    if not isinstance(data, Mapping):
        raise ConfigError(f'{where}: a mapping with `cell_io: {FORMAT_VERSION}` at the top')
    _keys(data, {'cell_io', 'control_mode', 'plant', 'robot_slot', 'scan', 'devices', 'services', 'mock', 'isaac'},
          where)
    if data.get('cell_io') != FORMAT_VERSION or isinstance(data.get('cell_io'), bool):
        raise ConfigError(f'{where}: `cell_io: {FORMAT_VERSION}` is required (got {data.get("cell_io")!r}); is this '
                          'the cell\'s machine-simulation file?')
    mode = data.get('control_mode')
    if mode not in CONTROL_MODES:
        raise ConfigError(f'{where}: control_mode must be one of {", ".join(CONTROL_MODES)} (got {mode!r})')
    services_raw = data.get('services') or {}
    if mode == 'external_plc':
        if services_raw:
            raise ConfigError(
                f'{where}: control_mode external_plc refuses a simulated machine service (a ProcessDevice: '
                f'{", ".join(sorted(map(str, services_raw)))}) -- in that mode the machine logic runs on the PLC '
                '(ADR-0013 Amendment P0)')
        if any(isinstance(s, Mapping) and 'logic' in s for s in (services_raw.values()
                                                                 if isinstance(services_raw, Mapping) else [])):
            raise ConfigError(f'{where}: control_mode external_plc refuses a logic backend')
        if data.get('plant') is not None:
            raise ConfigError(f'{where}: control_mode external_plc has no simulated plant (remove `plant`)')
    slot = data.get('robot_slot', '')
    if not isinstance(slot, str) or not _SLOT.match(slot):
        raise ConfigError(f'{where}: robot_slot must be \'\' or a relative namespace such as r1 (got {slot!r})')
    devices = _devices(data.get('devices') or {}, where)
    clock = None
    plant = data.get('plant')
    if mode == 'simulation':
        if plant not in PLANTS:
            raise ConfigError(f'{where}: plant must be one of {", ".join(PLANTS)} (got {plant!r})')
        clock = _clock(data.get('scan'), where)
        for dev in devices.values():
            if dev.timeout_ms is not None:
                try:
                    steps = clock.steps(dev.timeout_ms, f'devices.{dev.name}.timeout_ms')
                except ClockError as exc:
                    raise ConfigError(f'{where}: {exc}') from exc
                if steps % clock.scan_every_steps:
                    # the station looks at its devices once per scan: a timeout between two scans is not exact
                    raise ConfigError(
                        f'{where}: devices.{dev.name}.timeout_ms = {dev.timeout_ms} ms is {steps} physics steps, not '
                        f'a whole number of scans (one scan = {clock.scan_every_steps} steps): a motion timeout ends '
                        'on a scan, exactly')
    services = _services(services_raw, devices, where)
    if mode == 'simulation' and not services:
        raise ConfigError(f'{where}: control_mode simulation needs at least one service')
    mock = data.get('mock') or {}
    isaac = data.get('isaac') or {}
    if not isinstance(mock, Mapping) or not isinstance(isaac, Mapping):
        raise ConfigError(f'{where}: `mock` and `isaac` are mappings')
    return CellConfig(control_mode=mode, plant=plant, robot_slot=slot, clock=clock, devices=devices,
                      services=services, mock=dict(mock), isaac=dict(isaac))


def _keys(data: Mapping, allowed: set, where: str) -> None:
    unknown = sorted(str(k) for k in data if k not in allowed)
    if unknown:
        raise ConfigError(f'{where}: unknown key(s) {", ".join(unknown)} (allowed: {", ".join(sorted(allowed))})')


def _clock(scan: Any, where: str) -> ScanClock:
    if not isinstance(scan, Mapping):
        raise ConfigError(f'{where}: `scan: {{physics_dt_ms: 5, scan_every_steps: 2, status_period_ms: 100}}` is '
                          'required in simulation')
    _keys(scan, {'physics_dt_ms', 'physics_dt', 'scan_every_steps', 'status_period_ms'}, f'{where}: scan')
    period = scan.get('status_period_ms', 100)
    try:
        dt = parse_dt(scan.get('physics_dt'), scan.get('physics_dt_ms'))
        clock = ScanClock(dt, scan.get('scan_every_steps', 1), period)
    except ClockError as exc:
        raise ConfigError(f'{where}: scan: {exc}') from exc
    if period > MAX_STATUS_PERIOD_MS:          # an integer here: ScanClock refused anything else
        raise ConfigError(
            f'{where}: scan.status_period_ms = {period} ms is above {MAX_STATUS_PERIOD_MS} ms: the robot synchronises '
            'a request on two heartbeat changes of the source within its channel_status_max_age_ms (default 1000 ms), '
            'so the status period must be at most half of it')
    return clock


def _devices(raw: Any, where: str) -> Dict[str, DeviceSpec]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f'{where}: devices is a mapping name -> {{type: ...}}')
    out: Dict[str, DeviceSpec] = {}
    for name, spec in raw.items():
        at = f'{where}: devices.{name}'
        if not isinstance(name, str) or not _ID.match(name):
            raise ConfigError(f'{at}: a device name is a lowercase identifier')
        if not isinstance(spec, Mapping):
            raise ConfigError(f'{at}: a mapping with a type')
        kind = spec.get('type')
        if kind not in DEVICE_TYPES:
            raise ConfigError(f'{at}: type must be one of {", ".join(DEVICE_TYPES)} (got {kind!r})')
        if kind == 'binary_actuator':
            _keys(spec, {'type', 'timeout_ms', 'description'}, at)
            out[name] = DeviceSpec(name, kind, timeout_ms=_ms(spec.get('timeout_ms'), f'{at}.timeout_ms'))
        elif kind == 'position_axis':
            _keys(spec, {'type', 'timeout_ms', 'unit', 'positions', 'band', 'description'}, at)
            positions = spec.get('positions')
            if not isinstance(positions, Mapping) or not positions:
                raise ConfigError(f'{at}: positions is a non-empty mapping name -> value')
            for pos_name, value in positions.items():
                if not isinstance(pos_name, str) or not _ID.match(pos_name):
                    raise ConfigError(f'{at}: a position name is a lowercase identifier (got {pos_name!r})')
                if not _finite(value):
                    raise ConfigError(f'{at}: position {pos_name!r} must be a finite number (got {value!r})')
            band = spec.get('band')
            if not _finite(band) or not band > 0:
                raise ConfigError(f'{at}: band (the in-position tolerance) must be a finite number > 0 (got {band!r})')
            unit = spec.get('unit', '')
            if unit not in ('deg', 'mm'):
                raise ConfigError(f'{at}: unit must be deg or mm (got {unit!r})')
            out[name] = DeviceSpec(name, kind, timeout_ms=_ms(spec.get('timeout_ms'), f'{at}.timeout_ms'),
                                   unit=unit, positions={k: float(v) for k, v in positions.items()},
                                   band=float(band))
        else:
            _keys(spec, {'type', 'description'}, at)
            out[name] = DeviceSpec(name, kind)
    return out


def _ms(value: Any, at: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= 600000:
        raise ConfigError(f'{at} must be an integer number of milliseconds in 1..600000 (got {value!r})')
    return value


def _finite(value: Any) -> bool:
    """A real number (never a bool), and finite: .nan and .inf are refused."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_step(step: Any, devices: Mapping[str, DeviceSpec], at: str) -> str:
    """One chain step, checked as `logic.build_chain` will build it: `{close|open: <binary_actuator>}`,
    `{move: <position_axis>, to: <one of its positions>}` or `{require: <digital_sensor>}`. Returns the device's name.
    Any other value -- a list where a name belongs, an unknown verb, a device of the wrong type -- is a ConfigError."""
    if not isinstance(step, Mapping):
        raise ConfigError(f'{at}: a step is a mapping such as {{close: clamp}} (got {step!r})')
    verb = next((v for v in STEP_DEVICE_TYPES if v in step), None)
    expected = {'move', 'to'} if verb == 'move' else {verb}
    if verb is None or set(step) != expected:
        raise ConfigError(f'{at}: unknown step {dict(step)!r} (use close / open / move + to / require)')
    name = step[verb]
    if not isinstance(name, str) or name not in devices:
        raise ConfigError(f'{at}: unknown device {name!r} (declared: {", ".join(sorted(devices)) or "none"})')
    kind = devices[name].type
    if kind != STEP_DEVICE_TYPES[verb]:
        raise ConfigError(f'{at}: {verb} needs a {STEP_DEVICE_TYPES[verb]}; {name!r} is a {kind}')
    if verb == 'move':
        to = step['to']
        positions = devices[name].positions
        if not isinstance(to, str) or to not in positions:
            raise ConfigError(f'{at}: axis {name!r} has no position {to!r} (its positions: '
                              f'{", ".join(sorted(positions))})')
    return name


def _services(raw: Any, devices: Mapping[str, DeviceSpec], where: str) -> Dict[str, ServiceSpec]:
    if not isinstance(raw, Mapping):
        raise ConfigError(f'{where}: services is a mapping id -> {{cycle, home}}')
    out: Dict[str, ServiceSpec] = {}
    owner: Dict[str, str] = {}
    for sid, spec in raw.items():
        at = f'{where}: services.{sid}'
        if not isinstance(sid, str) or not _ID.match(sid):
            raise ConfigError(f'{at}: a service id is a lowercase identifier (the robot application\'s logical id)')
        if not isinstance(spec, Mapping):
            raise ConfigError(f'{at}: a mapping with logic, cycle and home')
        _keys(spec, {'description', 'logic', 'cycle', 'home'}, at)
        logic = spec.get('logic', 'python_logic')
        if logic not in LOGIC_BACKENDS:
            raise ConfigError(f'{at}: logic must be one of {", ".join(LOGIC_BACKENDS)} (got {logic!r})')
        if logic == 'code':
            listed = [part for part in ('cycle', 'home') if part in spec]
            if listed:
                raise ConfigError(f'{at}: logic code lists no {" / ".join(listed)} (its sequences are the program\'s '
                                  'code, handed to the Cell)')
            description = spec.get('description', '')
            if not isinstance(description, str):
                raise ConfigError(f'{at}.description: a string')
            out[sid] = ServiceSpec(sid, description, logic, [], [])
            continue
        for part in ('cycle', 'home'):
            steps = spec.get(part)
            if not isinstance(steps, list) or not steps:
                raise ConfigError(f'{at}.{part}: a non-empty list of steps')
            for number, step in enumerate(steps, start=1):
                dev = _check_step(step, devices, f'{at}.{part} step {number}')
                prev = owner.setdefault(dev, sid)
                if prev != sid:
                    raise ConfigError(f'{at}.{part}: device {dev!r} already belongs to service {prev!r} '
                                      '(one service per device)')
        description = spec.get('description', '')
        if not isinstance(description, str):
            raise ConfigError(f'{at}.description: a string')
        out[sid] = ServiceSpec(sid, description, logic, list(spec['cycle']), list(spec['home']))
    return out
