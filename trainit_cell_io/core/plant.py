"""The plant adapter: where the devices' commands meet a plant (P0B_DESIGN.md §4).

The SAME station logic (the devices, the `ProcessDevice`s, the chains) runs against any plant; only the adapter
differs -- the deterministic `MockPlantAdapter` (a time model with fault injection) or the `IsaacPlantAdapter` (an
articulation in Isaac Sim, read back every scan). That is the parity the P0.b gate measures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Mapping, Sequence

from .clock import ScanClock


class InjectionError(ValueError):
    """A fault injection the plant refuses (unknown fault, device or service)."""


class PlantAdapter(ABC):
    name = 'plant'

    @abstractmethod
    def bind(self, devices: Mapping[str, object], services: Sequence[str], clock: ScanClock) -> None:
        """Called once by the cell: the devices it simulates, the service ids, the scan clock."""

    def advance(self, steps: int) -> None:
        """Advance the plant by `steps` physics steps (the mock's time model; a simulator advances itself)."""

    @abstractmethod
    def read(self) -> Dict[str, object]:
        """The input image: every device's read-back (limit sensors, positions, sensor values) and the station inputs
        (`<service>.hold`, `<service>.fault`, `<service>.fault_detail`)."""

    @abstractmethod
    def apply(self, outputs: Mapping[str, object]) -> None:
        """The output image: every device's command (`<actuator>.cmd`, `<axis>.target`)."""

    def inject(self, fault: Mapping[str, object]) -> str:
        """A fault injection (simulation control only); returns what it did, in plain words."""
        raise InjectionError(f'the {self.name} plant does not inject faults')

    def consume_delay_steps(self, service: str) -> int:
        """Physics steps the source of `service` waits, after it first sees a NEW request image, before it acts on it
        (the `consume_delay` injection: a late request, e.g. a request that reaches the source after a station fault).
        0 = at once, the normal case."""
        return 0

    def describe(self) -> Dict[str, object]:
        return {'plant': self.name}
