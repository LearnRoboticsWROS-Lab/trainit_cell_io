"""The source side of ONE machine service: the `ProcessDevice` (P0B_DESIGN.md §3.3).

The rules, in the order the pre-registration states them:
 1. it starts with a new `source_session` (random, non-zero), READY, no cycle, no rejection;
 2. it acts on a request only when the request names ITS session AND the request's identity (requester_session, seq)
    differs from the LAST CONSUMED identity (accepted or rejected): a re-delivered image is a no-op, and a request
    naming another session -- an image left from before a restart -- is NEVER executed;
 3. READY or DONE + a new request -> ACCEPTED: `cycle_*` := its identity, `cycle_done` := false, BUSY, the cycle starts;
 4. BUSY + a new request -> REJECTED `BUSY`: not executed, the RUNNING cycle continues untouched;
 5. FAULT + a new request -> REJECTED `FAULTED`; HOMING + a new request -> REJECTED `BUSY` -- consumed either way, so it
    is never executed after a RESET or after the homing;
 6. more than one publisher on `<service>/request` -> every new request REJECTED `TWO_REQUESTERS`;
 7. completion -> `cycle_done` := true, DONE (level: until the next request is accepted);
 8. a device fault -> FAULT + its code, the cycle stops where it is (`cycle_done` stays false); FAULT persists until a
    RESET (an operator / machine-master action, never the robot); a RESET runs the return-to-home sequence (HOMING)
    and reports READY only once it completed; the session stays;
 9. the status every period and at once on every change; `heartbeat` +1 every period (the cell's scan mints it).

A RESET is accepted in any state: from BUSY it ABORTS the cycle (`cycle_done` stays false), so a waiter that looks
later sees an aborted cycle (the robot's STATION_FAULT, never a timeout). The cycle and rejection fields are KEPT
across a RESET: they are the history a waiter matches its request against.

Rule 2 is an "only when" condition whose purpose is that a re-delivered image is a no-op. One last consumed identity
is not enough for that purpose once a SECOND requester's request was consumed in between (a two-requester episode,
rule 6): the first requester's latched image, re-delivered by a bridge restart or a DDS re-match, would differ from
the last consumed identity and run a second cycle -- or a request REJECTED FAULTED would run after the RESET. So the
source ALSO remembers, per requester, the last seq it consumed (the last REMEMBERED_REQUESTERS requesters, the least
recently used forgotten first): an image equal to its requester's last consumed identity is a no-op too, and so is
one OLDER than it (a requester's seq only grows, wrapping: 32-bit serial order, `seq_newer`) -- a replayed history never
runs a request again. With one requester sending growing seqs the memories agree and the behaviour is unchanged.

A station's sequence is the cell's own code (`python_logic`) or the program's (`logic: code`). Whatever a sequence
returns or raises, the wire carries only the contract's fault codes (P0B_DESIGN.md §3.2): a code other than
ACTUATOR_TIMEOUT / SENSOR_FAULT / STATION_FAULT is reported as STATION_FAULT (its own value kept in `detail`), and a
sequence that raises faults ITS station (STATION_FAULT, the exception in `detail`) -- never the whole cell.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from typing import Mapping, Optional, Tuple

from .contract import (FAULT_ACTUATOR_TIMEOUT, FAULT_NAMES, FAULT_NONE, FAULT_SENSOR_FAULT, FAULT_STATION_FAULT,
                       REJECT_BUSY, REJECT_FAULTED, REJECT_NAMES,
                       REJECT_TWO_REQUESTERS, STATE_BUSY, STATE_DONE, STATE_FAULT, STATE_HOMING, STATE_NAMES,
                       STATE_READY, UINT16_MASK, UINT32_MAX, DeviceCounters, Identity, RequestImage, StatusImage)
from .logic import DONE, RUNNING, Fault, Sequence

ACCEPTED = 'accepted'
REJECTED = 'rejected'

# What rule 2 makes of a request image (`ProcessDevice.classify`)
NEW = 'new'                        # a request the source acts on: it will be ACCEPTED or REJECTED (consumed)
CONSUMED = 'consumed'              # its identity was consumed already: a re-delivery, a no-op
OTHER_SESSION = 'other_session'    # it names another source session (from before a restart): never executed
MALFORMED = 'malformed'            # a zero / out-of-range field (a default image): it says nothing

# Requesters whose last consumed seq the source remembers (rule 2, per requester; least recently used forgotten). One
# robot per service is the contract (rule 6): 16 covers every restart of a requester the source can meet in a run.
REMEMBERED_REQUESTERS = 16


def new_session(rng: Optional[random.Random] = None) -> int:
    """A random, non-zero uint32 (a collision with the previous session: 2^-32 per restart, documented)."""
    r = rng or random.SystemRandom()
    while True:
        value = r.randint(1, UINT32_MAX)
        if value != 0:
            return value


# the fault codes the wire carries (P0B_DESIGN.md §3.2): any other station-reported fault is STATION_FAULT
CONTRACT_FAULTS = (FAULT_ACTUATOR_TIMEOUT, FAULT_SENSOR_FAULT, FAULT_STATION_FAULT)


def seq_newer(seq: int, last: int) -> bool:
    """`seq` is NEWER than `last` in 32-bit serial-number order (RFC 1982): ahead by 1 .. 2^31 - 1, so a seq that
    wrapped (... 0xFFFFFFFF, 1, ...: 0 is skipped) is still newer. Equal, or behind, is not newer: a re-delivered or
    a replayed (older) image of the same requester -- e.g. a TRANSIENT_LOCAL history of depth > 1 re-sent to a reader
    that matches again -- is never a new request."""
    ahead = (seq - last) & 0xFFFFFFFF
    return 0 < ahead < 0x80000000


class ProcessDevice:
    def __init__(self, service: str, cycle: Sequence, home: Sequence, session: int):
        if not isinstance(session, int) or isinstance(session, bool) or not 0 < session <= UINT32_MAX:
            raise ValueError(f'{service}: the source session must be a non-zero uint32, got {session!r}')
        self.service = service
        self.cycle = cycle
        self.home = home
        self.image = StatusImage(source_session=session, state=STATE_READY, detail='ready')
        self.counters = DeviceCounters()
        self._last_consumed: Optional[Identity] = None
        # rule 2 per requester: requester_session -> the last seq consumed from it (bounded, least recently used out)
        self._consumed_by: 'OrderedDict[int, int]' = OrderedDict()
        # images of another session already counted (each counted once; bounded like the requesters)
        self._ignored: 'OrderedDict[Tuple[int, int, int], None]' = OrderedDict()
        self._active: Optional[Sequence] = None

    # --- the request (rules 2-6) ------------------------------------------------------------------------------------
    @property
    def session(self) -> int:
        return self.image.source_session

    @property
    def state(self) -> int:
        return self.image.state

    def classify(self, request: Optional[RequestImage]) -> str:
        """What rule 2 makes of `request`, without acting on it: NEW, CONSUMED (its identity is the last consumed one,
        or its requester's last consumed one -- a re-delivery -- or OLDER than that one: a replayed history; a
        requester's seq only grows, wrapping, P0B_DESIGN §3.2), OTHER_SESSION or MALFORMED."""
        if request is None or not request.well_formed():
            return MALFORMED
        if request.source_session != self.image.source_session:
            return OTHER_SESSION
        identity = request.identity
        if identity == self._last_consumed:
            return CONSUMED
        last = self._consumed_by.get(identity[0])
        if last is not None and not seq_newer(identity[1], last):
            return CONSUMED
        return NEW

    def consume(self, request: Optional[RequestImage], requesters: int = 1) -> Optional[str]:
        """Act on one `<service>/request` image: ACCEPTED / REJECTED (it was NEW: now consumed), or None (a no-op)."""
        kind = self.classify(request)
        if kind == OTHER_SESSION:
            self._ignore_other_session(request)
            return None
        if kind != NEW:
            return None
        identity = request.identity
        self._last_consumed = identity
        self._consumed_by[identity[0]] = identity[1]
        self._consumed_by.move_to_end(identity[0])
        while len(self._consumed_by) > REMEMBERED_REQUESTERS:
            self._consumed_by.popitem(last=False)
        if requesters > 1:
            return self._reject(identity, REJECT_TWO_REQUESTERS)
        state = self.image.state
        if state == STATE_FAULT:
            return self._reject(identity, REJECT_FAULTED)
        if state in (STATE_BUSY, STATE_HOMING):
            return self._reject(identity, REJECT_BUSY)
        # READY or DONE: a new cycle
        img = self.image
        img.cycle_requester, img.cycle_seq = identity
        img.cycle_done = False
        img.state = STATE_BUSY
        self.counters.accepted += 1
        self._active = self.cycle
        img.detail = f'busy: cycle {self.counters.accepted} for request {identity[1]}'
        started = self._call(self.cycle.start)
        if isinstance(started, Fault):
            self._fault(started.code, started.detail)
        return ACCEPTED

    def _ignore_other_session(self, request: RequestImage) -> None:
        """Rule 2: an image of another source session is never executed; each distinct one is counted once."""
        key = (request.source_session, request.requester_session, request.seq)
        if key in self._ignored:
            self._ignored.move_to_end(key)
            return
        self._ignored[key] = None
        while len(self._ignored) > REMEMBERED_REQUESTERS:
            self._ignored.popitem(last=False)
        self.counters.ignored_other_session += 1

    def _reject(self, identity: Identity, code: int) -> str:
        img = self.image
        img.reject_requester, img.reject_seq = identity
        img.reject_code = code
        if code == REJECT_BUSY:
            self.counters.rejected_busy += 1
        elif code == REJECT_FAULTED:
            self.counters.rejected_faulted += 1
        else:
            self.counters.rejected_two_requesters += 1
        # The detail keeps the device's own activity; the rejection is a separate, level field.
        return REJECTED

    # --- the scan (rules 7, 8) ---------------------------------------------------------------------------------------
    def scan(self, inputs: Mapping[str, object], steps: int) -> None:
        """One scan of the station's logic: `steps` physics steps since the previous scan."""
        held = inputs.get(f'{self.service}.hold') is True
        if inputs.get(f'{self.service}.fault') is True and self.image.state != STATE_FAULT:
            detail = inputs.get(f'{self.service}.fault_detail')
            self._fault(FAULT_STATION_FAULT, detail if isinstance(detail, str) and detail else
                        'the station reported a fault')
            return
        state = self.image.state
        if state == STATE_BUSY:
            result = self._call(self.cycle.scan, inputs, held, steps)
            if result == DONE:
                self.image.cycle_done = True
                self.image.state = STATE_DONE
                self.counters.completed += 1
                self._active = None
                self.image.detail = f'done: cycle {self.counters.accepted} for request {self.image.cycle_seq}'
            elif isinstance(result, Fault):
                self._fault(result.code, result.detail)
            else:
                self.image.detail = f'busy: {self.cycle.describe()}' + (' (held)' if held else '')
        elif state == STATE_HOMING:
            result = self._call(self.home.scan, inputs, held, steps)
            if result == DONE:
                self.image.state = STATE_READY
                self.counters.homings_completed += 1
                self._active = None
                self.image.detail = 'ready (homed)'
            elif isinstance(result, Fault):
                self._fault(result.code, result.detail)
            else:
                self.image.detail = f'homing: {self.home.describe()}' + (' (held)' if held else '')

    def _call(self, fn, *args):
        """A sequence's start() / scan(): an exception is ITS station's fault (STATION_FAULT), never the cell's stop."""
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001 -- a station sequence's error becomes that station's FAULT
            return Fault(FAULT_STATION_FAULT, f'the station sequence raised {type(exc).__name__}: {exc}')

    def _fault(self, code: int, detail: str) -> None:
        img = self.image
        if img.state == STATE_BUSY:
            self.counters.aborted += 1     # cycle_done stays false: the cycle stopped where it is
        if code not in CONTRACT_FAULTS:    # only the contract's codes reach the wire (§3.2); the rest is STATION_FAULT
            detail = f'{detail} (station fault code {code!r} reported as STATION_FAULT)'
            code = FAULT_STATION_FAULT
        img.state = STATE_FAULT
        img.fault_code = code
        img.detail = f'FAULT {FAULT_NAMES.get(code, code)}: {detail}'
        self._active = None

    # --- the RESET (rule 8): an operator / machine-master action, never the robot -----------------------------------
    def reset(self) -> None:
        img = self.image
        self.counters.resets += 1
        if img.state == STATE_BUSY:
            self.counters.aborted += 1
        img.state = STATE_HOMING
        img.fault_code = FAULT_NONE
        img.detail = 'homing'
        self._active = self.home
        started = self._call(self.home.start)
        if isinstance(started, Fault):
            self._fault(started.code, started.detail)

    # --- the period (rule 9) -----------------------------------------------------------------------------------------
    def beat(self) -> None:
        self.image.heartbeat = (self.image.heartbeat + 1) & UINT16_MASK

    def status(self) -> StatusImage:
        return self.image.copy()

    def describe(self) -> str:
        img = self.image
        text = f'{self.service}: {STATE_NAMES.get(img.state, img.state)}'
        if img.cycle_seq:
            text += f', cycle of request {img.cycle_seq}{" done" if img.cycle_done else ""}'
        if img.reject_code:
            text += f', last rejected request {img.reject_seq} ({REJECT_NAMES.get(img.reject_code)})'
        return text


__all__ = ['ACCEPTED', 'REJECTED', 'NEW', 'CONSUMED', 'OTHER_SESSION', 'MALFORMED', 'REMEMBERED_REQUESTERS',
           'ProcessDevice', 'new_session', 'seq_newer', 'RUNNING']
