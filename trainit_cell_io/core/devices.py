"""The machine devices of simulation control (P0B_DESIGN.md §4): BinaryActuator, PositionAxis, DigitalSensor.

A device is the controller's VIEW of one piece of the plant: what it commands (its outputs) and what it reads back
(its inputs) through the I/O image, never how the plant moves. The plant adapter (the mock's time model, or a
simulator's articulation) owns the motion. Tag names are `<device>` / `<device>.<signal>` -- cell-level names, never a
robot topic, a PLC address or a physical I/O point.

  BinaryActuator  out `<name>.cmd` (bool: True = closed / extended, False = open / retracted -- monostable: the plant
                  returns it to open when the command drops); in `<name>.open`, `<name>.closed` (the two limit sensors)
  PositionAxis    out `<name>.target` (float, the axis unit); in `<name>.position` (float, the read-back)
  DigitalSensor   in `<name>` (bool) and `<name>.ok` (bool, the sensor's own diagnosis; True if the plant says nothing)
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional

from .errors import ConfigError


class DeviceError(ConfigError):
    """A device definition the cell refuses (a configuration refusal: exit 2, never a traceback)."""


class BinaryActuator:
    kind = 'binary_actuator'

    def __init__(self, name: str, timeout_steps: int, timeout_ms: int):
        self.name = name
        self.timeout_steps = timeout_steps
        self.timeout_ms = timeout_ms
        self.cmd = False                 # de-energised: open
        self.open_limit: Optional[bool] = None
        self.closed_limit: Optional[bool] = None

    def command(self, close: bool) -> None:
        self.cmd = bool(close)

    def read(self, inputs: Mapping[str, object]) -> None:
        self.open_limit = _bool_or_none(inputs.get(f'{self.name}.open'))
        self.closed_limit = _bool_or_none(inputs.get(f'{self.name}.closed'))

    def at(self, close: bool) -> bool:
        if close:
            return self.closed_limit is True and self.open_limit is False
        return self.open_limit is True and self.closed_limit is False

    def implausible(self) -> Optional[str]:
        """A limit-sensor reading that cannot be true of a healthy actuator (a failed sensor), else None."""
        if self.open_limit is None or self.closed_limit is None:
            return f'{self.name}: no limit-sensor reading'
        if self.open_limit and self.closed_limit:
            return f'{self.name}: open and closed limit sensors both on'
        return None

    def outputs(self) -> Dict[str, object]:
        return {f'{self.name}.cmd': self.cmd}

    def describe(self) -> str:
        where = 'closed' if self.at(True) else 'open' if self.at(False) else 'between limits'
        return f'{self.name} {where}'


class PositionAxis:
    kind = 'position_axis'

    def __init__(self, name: str, positions: Mapping[str, float], band: float, timeout_steps: int,
                 timeout_ms: int, unit: str):
        if not positions:
            raise DeviceError(f'axis {name!r}: at least one named position')
        for pos_name, value in positions.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise DeviceError(f'axis {name!r}: position {pos_name!r} must be a finite number, got {value!r}')
        if not isinstance(band, (int, float)) or isinstance(band, bool) or not math.isfinite(band) or not band > 0:
            raise DeviceError(f'axis {name!r}: band must be a finite number > 0, got {band!r}')
        self.name = name
        self.positions = {k: float(v) for k, v in positions.items()}
        self.band = float(band)
        self.timeout_steps = timeout_steps
        self.timeout_ms = timeout_ms
        self.unit = unit
        self.target: Optional[float] = None    # nothing commanded yet: the plant holds where it is
        self.target_name: Optional[str] = None
        self.position: Optional[float] = None

    def command(self, position_name: str) -> None:
        self.target = self.positions[position_name]
        self.target_name = position_name

    def read(self, inputs: Mapping[str, object]) -> None:
        value = inputs.get(f'{self.name}.position')
        self.position = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def at(self, position_name: str) -> bool:
        if self.position is None or not math.isfinite(self.position):
            return False
        return abs(self.position - self.positions[position_name]) <= self.band

    def implausible(self) -> Optional[str]:
        if self.position is None or not math.isfinite(self.position):
            return f'{self.name}: no position read-back'
        return None

    def outputs(self) -> Dict[str, object]:
        return {} if self.target is None else {f'{self.name}.target': self.target}

    def describe(self) -> str:
        if self.position is None:
            return f'{self.name} (no read-back)'
        return f'{self.name} at {self.position:.1f} {self.unit}'


class DigitalSensor:
    kind = 'digital_sensor'

    def __init__(self, name: str):
        self.name = name
        self.value: Optional[bool] = None
        self.ok = True

    def read(self, inputs: Mapping[str, object]) -> None:
        self.value = _bool_or_none(inputs.get(self.name))
        ok = inputs.get(f'{self.name}.ok', True)
        self.ok = bool(ok) if isinstance(ok, bool) else False

    def outputs(self) -> Dict[str, object]:
        return {}

    def describe(self) -> str:
        if not self.ok:
            return f'{self.name} FAILED'
        return f'{self.name} {"on" if self.value else "off" if self.value is False else "unknown"}'


def _bool_or_none(value: object) -> Optional[bool]:
    return value if isinstance(value, bool) else None
