"""The machine-service contract, source side (P0B_DESIGN.md §3.2 / §3.3).

The constants mirror `trainit_interfaces/msg/ChannelStatus` (pinned there by `test_machine_channel`); the images are
plain dataclasses so the core runs without ROS (case C1) and inside a simulator's own Python (the Kit side of the
Isaac plant adapter, which carries std_msgs only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

STATE_NONE = 0
STATE_READY = 1
STATE_BUSY = 2
STATE_DONE = 3
STATE_FAULT = 4
STATE_HOMING = 5
STATE_NAMES = {
    STATE_NONE: 'NONE',
    STATE_READY: 'READY',
    STATE_BUSY: 'BUSY',
    STATE_DONE: 'DONE',
    STATE_FAULT: 'FAULT',
    STATE_HOMING: 'HOMING',
}

REJECT_NONE = 0
REJECT_BUSY = 1
REJECT_FAULTED = 2
REJECT_TWO_REQUESTERS = 3
REJECT_NAMES = {
    REJECT_NONE: 'NONE',
    REJECT_BUSY: 'BUSY',
    REJECT_FAULTED: 'FAULTED',
    REJECT_TWO_REQUESTERS: 'TWO_REQUESTERS',
}

FAULT_NONE = 0
FAULT_ACTUATOR_TIMEOUT = 1
FAULT_SENSOR_FAULT = 2
FAULT_STATION_FAULT = 3
FAULT_NAMES = {
    FAULT_NONE: 'NONE',
    FAULT_ACTUATOR_TIMEOUT: 'ACTUATOR_TIMEOUT',
    FAULT_SENSOR_FAULT: 'SENSOR_FAULT',
    FAULT_STATION_FAULT: 'STATION_FAULT',
}

UINT16_MASK = 0xFFFF
UINT32_MAX = 0xFFFFFFFF

Identity = Tuple[int, int]   # (requester_session, seq)


def _is_uint32(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= UINT32_MAX


@dataclass(frozen=True)
class RequestImage:
    """One `<service>/request` image as the source receives it (level-held, re-delivered at will)."""

    source_session: int
    requester_session: int
    seq: int

    @property
    def identity(self) -> Identity:
        return (self.requester_session, self.seq)

    def well_formed(self) -> bool:
        """A request a source may act on at all: every field a non-zero uint32 (a default image says nothing)."""
        return (_is_uint32(self.source_session) and _is_uint32(self.requester_session) and _is_uint32(self.seq)
                and self.source_session != 0 and self.requester_session != 0 and self.seq != 0)


@dataclass
class StatusImage:
    """One `<service>/status` image (the source's current truth)."""

    source_session: int = 0
    heartbeat: int = 0
    state: int = STATE_NONE
    cycle_requester: int = 0
    cycle_seq: int = 0
    cycle_done: bool = False
    reject_requester: int = 0
    reject_seq: int = 0
    reject_code: int = REJECT_NONE
    fault_code: int = FAULT_NONE
    detail: str = ''

    def without_heartbeat(self) -> tuple:
        """Every field but the heartbeat: a change of any of them is published AT ONCE (§3.3 rule 9)."""
        return (self.source_session, self.state, self.cycle_requester, self.cycle_seq, self.cycle_done,
                self.reject_requester, self.reject_seq, self.reject_code, self.fault_code, self.detail)

    def copy(self) -> 'StatusImage':
        return StatusImage(**self.as_dict())

    def as_dict(self) -> dict:
        return {
            'source_session': self.source_session,
            'heartbeat': self.heartbeat,
            'state': self.state,
            'cycle_requester': self.cycle_requester,
            'cycle_seq': self.cycle_seq,
            'cycle_done': self.cycle_done,
            'reject_requester': self.reject_requester,
            'reject_seq': self.reject_seq,
            'reject_code': self.reject_code,
            'fault_code': self.fault_code,
            'detail': self.detail,
        }


@dataclass
class DeviceCounters:
    """What the source did, for tests and the operator (never on the wire)."""

    accepted: int = 0            # requests accepted as a cycle (= cycles started)
    completed: int = 0           # cycles completed (DONE)
    aborted: int = 0             # cycles stopped by a fault or a RESET before completing
    rejected_busy: int = 0
    rejected_faulted: int = 0
    rejected_two_requesters: int = 0
    ignored_other_session: int = 0   # images naming another source session (never executed)
    resets: int = 0
    homings_completed: int = 0
    last_event: Optional[str] = field(default=None)

    def as_dict(self) -> dict:
        return {
            'accepted': self.accepted,
            'completed': self.completed,
            'aborted': self.aborted,
            'rejected_busy': self.rejected_busy,
            'rejected_faulted': self.rejected_faulted,
            'rejected_two_requesters': self.rejected_two_requesters,
            'ignored_other_session': self.ignored_other_session,
            'resets': self.resets,
            'homings_completed': self.homings_completed,
        }
