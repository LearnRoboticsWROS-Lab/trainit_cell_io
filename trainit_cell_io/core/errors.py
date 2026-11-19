"""The cell file's refusals: ONE exception family (P0.b, Python review MINOR 1).

Every way a `cell_io.yaml` can be refused -- a key, a type, a duration the scan clock cannot represent (`ClockError`),
a device definition (`DeviceError`), a station chain (`LogicError`) -- is a `ConfigError`, so the executables refuse
the file with one readable line and exit 2, never with a traceback. It lives in its own module so that `clock`,
`devices` and `logic` can subclass it without importing `config` (which imports them).
"""

from __future__ import annotations


class ConfigError(ValueError):
    """A `cell_io.yaml` (or a part of it) the cell refuses."""


__all__ = ['ConfigError']
