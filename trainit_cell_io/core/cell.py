"""The cell: the devices, one `ProcessDevice` per machine service, the scan, and a plant adapter.

Transport-free: a ROS node (`cell_io_node`, the mock plant), the Kit side of the Isaac plant, or a unit test drives it
with `step()` (one physics step) and hands it EVERY request image it receives with `set_request()`. Every `step()`
returns the status images to publish NOW: every service's image each status period (its heartbeat +1) and at once on
any change (P0B_DESIGN.md §3.3 rule 9).

Request images (implementation review fix C2): the cell keeps the latest image of EACH requester, per service
(bounded: the last REMEMBERED_REQUESTERS requesters), in the order each identity arrived. At each scan the source
consumes at most ONE new identity -- the earliest-arrived one it has not consumed (rule 2) -- so two requesters' images
within one scan are two consumptions in two successive published images, each named by `cycle_*` or `reject_*`: none
is displaced unconsumed, none is left to run late. A consumed image stays held as its requester's latest (a no-op at
every scan); a re-delivered or OLDER image of a requester (its identity consumed already, or behind its last consumed
seq) is counted and never held -- it would displace a newer, not yet consumed image of that requester; an image of
another source session is counted once by the source (never executed) and dropped. The
`consume_delay` injection holds back each NEW image exactly its delay from the scan that first saw THAT image.

`request_images` / `ignored_redelivered` (per service, in the snapshot): every request image the transport handed in,
and how many of them were re-deliveries of an identity the source had consumed already, or older images of a
requester (no-ops, rule 2) -- the evidence
that a restarted bridge re-forwarded the robot's latched image and that nothing ran twice (PB-P21). They count the
transport, so a controller restart does not reset them; they stay out of the source's `counters` (cycles, rejections,
other-session images), which a re-delivery never changes.

`log` is a `BoundedLog` of `(step, text)` lines -- every verdict, every change of a service's image, every operator
command -- bounded to the last `log_lines`; a reader keeps the running total it has read (`log.since(index)`).
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from .bounded_log import DEFAULT_MAXLEN, BoundedLog
from .config import CellConfig, ConfigError
from .contract import REJECT_NAMES, STATE_NAMES, RequestImage, StatusImage
from .devices import BinaryActuator, DigitalSensor, PositionAxis
from .logic import build_chain, LogicError, Sequence
from .plant import InjectionError, PlantAdapter
from .process_device import (ACCEPTED, CONSUMED, MALFORMED, NEW, OTHER_SESSION, REMEMBERED_REQUESTERS, ProcessDevice,
                             new_session)


class ControlError(ValueError):
    """A cell control command the cell refuses (unknown command or service)."""


# a `logic: code` service's sequences: () -> (cycle, home), called once per source session
SequenceFactory = Callable[[], Tuple[Sequence, Sequence]]


@dataclass
class _Held:
    """A requester's latest request image for one service."""

    image: RequestImage
    arrival: int                           # when THIS identity arrived: the order in which the source consumes
    first_seen: Optional[int] = None       # the physics step of the scan that first saw it (consume_delay)


class Cell:
    def __init__(self, config: CellConfig, plant: PlantAdapter, rng: Optional[random.Random] = None,
                 sessions: Optional[Mapping[str, int]] = None, log_lines: int = DEFAULT_MAXLEN,
                 sequences: Optional[Mapping[str, SequenceFactory]] = None):
        """`sequences`: for each `logic: code` service, a factory returning its (cycle, home) `Sequence`s -- called
        once per source session (at the start and at a controller restart: fresh sequences each time)."""
        if config.control_mode != 'simulation' or config.clock is None:
            raise ConfigError('a simulated cell needs control_mode: simulation')
        self._sequences = dict(sequences or {})
        for sid in self._sequences:
            if sid not in config.services or config.services[sid].logic != 'code':
                raise ConfigError(f"sequences were handed for service {sid!r}, which is not a `logic: code` service of "
                                  'this cell')
        for sid, spec in config.services.items():
            if spec.logic == 'code' and sid not in self._sequences:
                raise ConfigError(f"service {sid!r} is `logic: code`: its sequences are code that only the program "
                                  'running it supplies (e.g. a twin\'s own stage runner) -- none was handed here')
        self.config = config
        self.clock = config.clock
        self.plant = plant
        self.devices: Dict[str, object] = {}
        for spec in config.devices.values():
            if spec.type == 'binary_actuator':
                self.devices[spec.name] = BinaryActuator(spec.name, self.clock.steps(spec.timeout_ms), spec.timeout_ms)
            elif spec.type == 'position_axis':
                self.devices[spec.name] = PositionAxis(spec.name, spec.positions, spec.band,
                                                       self.clock.steps(spec.timeout_ms), spec.timeout_ms, spec.unit)
            else:
                self.devices[spec.name] = DigitalSensor(spec.name)
        self.services: Dict[str, ProcessDevice] = {}
        self._session_rng = rng or random.SystemRandom()
        for sid in config.services:
            self.services[sid] = self._new_device(sid, (sessions or {}).get(sid) or new_session(self._session_rng))
        plant.bind(self.devices, list(self.services), self.clock)
        # per service: requester_session -> its latest image (least recently handed first; bounded)
        self._held: Dict[str, 'OrderedDict[int, _Held]'] = {sid: OrderedDict() for sid in self.services}
        self._arrivals = 0
        self._requesters: Dict[str, int] = {sid: 1 for sid in self.services}
        self._published: Dict[str, Optional[tuple]] = {sid: None for sid in self.services}
        self.request_images: Dict[str, int] = {sid: 0 for sid in self.services}
        self.ignored_redelivered: Dict[str, int] = {sid: 0 for sid in self.services}
        self.steps = 0
        self.log: BoundedLog[Tuple[int, str]] = BoundedLog(log_lines)

    def _new_device(self, sid: str, session: int) -> ProcessDevice:
        spec = self.config.services[sid]
        if spec.logic == 'code':
            try:
                made = self._sequences[sid]()
            except Exception as exc:  # noqa: BLE001 -- the program's factory: its error is a refusal, never a crash
                raise ConfigError(f"service {sid!r}: its sequences' factory raised {type(exc).__name__}: {exc}") \
                    from exc
            if not (isinstance(made, tuple) and len(made) == 2 and made[0] is not made[1] and
                    all(callable(getattr(q, m, None)) for q in made for m in ('start', 'scan', 'describe'))):
                raise ConfigError(f"service {sid!r}: its sequences' factory must return (cycle, home), two distinct "
                                  'Sequences (start, scan, describe)')
            return ProcessDevice(sid, made[0], made[1], session)
        try:
            cycle = build_chain(f'{sid} cycle', spec.cycle, self.devices)
            home = build_chain(f'{sid} home', spec.home, self.devices)
        except LogicError as exc:
            raise ConfigError(str(exc)) from exc
        return ProcessDevice(sid, cycle, home, session)

    # --- the transport hands in every request image it receives --------------------------------------------------
    def set_request(self, service: str, image: Optional[RequestImage], requesters: int = 1) -> None:
        """One request image received for `service` (each one, as it comes) and the number of publishers on
        `<service>/request` (rule 6). It becomes its requester's latest image; the scans act on it (`_next_request`).
        A re-delivery of an identity the source consumed already (or an older image of the same requester: a replay)
        is counted (`ignored_redelivered`): a no-op."""
        if service not in self.services:
            raise ControlError(f'no service {service!r}')
        self._requesters[service] = requesters
        if image is None:
            return
        self.request_images[service] += 1
        kind = self.services[service].classify(image)
        if kind == CONSUMED:
            self.ignored_redelivered[service] += 1
            return                                 # never held: it would displace a NEWER image of its requester
        if kind == MALFORMED:
            return                                 # a default image says nothing: never held, never acted on
        held = self._held[service]
        entry = held.get(image.requester_session)
        if entry is None or entry.image != image:  # a new identity of this requester: a new arrival
            self._arrivals += 1
            held[image.requester_session] = _Held(image, self._arrivals)
        held.move_to_end(image.requester_session)
        while len(held) > REMEMBERED_REQUESTERS:
            held.popitem(last=False)               # the least recently handed requester is forgotten

    def set_requesters(self, service: str, requesters: int) -> None:
        if service in self._requesters:
            self._requesters[service] = requesters

    # --- the clock ----------------------------------------------------------------------------------------------------
    def step(self, advance_plant: bool = True) -> List[Tuple[str, StatusImage]]:
        """One physics step (advance the mock plant first; a simulator has already stepped its physics)."""
        if advance_plant:
            self.plant.advance(1)
        self.steps += 1
        if self.steps % self.clock.scan_every_steps:
            return []
        return self._scan()

    def _scan(self) -> List[Tuple[str, StatusImage]]:
        inputs = self.plant.read()
        for dev in self.devices.values():
            dev.read(inputs)
        for sid, pd in self.services.items():
            before = pd.state
            request = self._next_request(sid, pd)
            verdict = pd.consume(request, self._requesters[sid]) if request is not None else None
            if verdict is not None:
                text = f'{sid}: request {request.seq} of requester {request.requester_session:#010x} {verdict}'
                if verdict != ACCEPTED:
                    text += f' {REJECT_NAMES.get(pd.image.reject_code)} (the source was {STATE_NAMES[before]})'
                self.log.append((self.steps, text))
            pd.scan(inputs, self.clock.scan_every_steps)
        outputs: Dict[str, object] = {}
        for dev in self.devices.values():
            outputs.update(dev.outputs())
        self.plant.apply(outputs)
        period = self.steps % self.clock.status_period_steps == 0
        due: List[Tuple[str, StatusImage]] = []
        for sid, pd in self.services.items():
            if period:
                pd.beat()
            key = pd.image.without_heartbeat()
            if period or key != self._published[sid]:
                if key != self._published[sid]:
                    self.log.append((self.steps, f'{pd.describe()} -- {pd.image.detail}'))
                self._published[sid] = key
                due.append((sid, pd.status()))
        return due

    def _next_request(self, sid: str, pd: ProcessDevice) -> Optional[RequestImage]:
        """The ONE request image the source acts on in this scan, or None.

        The earliest-arrived held image the source has not consumed (rule 2) -- at most one per scan, so every
        consumption gets its own published image. Under the `consume_delay` fault injection each NEW image is held back
        exactly its delay from the scan that first saw THAT image (nothing behind it overtakes it). An image of another
        source session is handed to the source once (it counts it, never executes it) and dropped; a consumed one stays
        held, a no-op. Counted in physics steps: the same seed and the same stimuli give the same trace (C4).
        """
        held = self._held[sid]
        if not held:
            return None
        delay = self.plant.consume_delay_steps(sid)
        head: Optional[_Held] = None
        for key, entry in sorted(held.items(), key=lambda item: item[1].arrival):
            kind = pd.classify(entry.image)
            if kind == OTHER_SESSION:
                pd.consume(entry.image)            # counted once; never executed
                del held[key]
                continue
            if kind != NEW:
                continue                           # consumed already: its requester's latest image, a no-op
            if entry.first_seen is None:
                entry.first_seen = self.steps
                if delay:
                    image = entry.image
                    self.log.append((self.steps, f'{sid}: request {image.seq} of requester '
                                                 f'{image.requester_session:#010x} seen; held back '
                                                 f'{self.clock.ms(delay):g} ms (consume_delay injected)'))
            if head is None:
                head = entry
        if head is None or (delay and self.steps - head.first_seen < delay):
            return None
        return head.image

    # --- cell control: the operator / machine master, never the robot ------------------------------------------------
    def control(self, command: Mapping[str, object]) -> str:
        cmd = command.get('cmd')
        if cmd == 'reset':
            sid = command.get('service')
            if not isinstance(sid, str) or sid not in self.services:
                raise ControlError(f'reset: no service {sid!r} (services: {", ".join(self.services)})')
            self.services[sid].reset()
            text = f'{sid}: RESET (homing)'
        elif cmd == 'inject':
            try:
                text = self.plant.inject({k: v for k, v in command.items() if k != 'cmd'})
            except InjectionError as exc:
                raise ControlError(str(exc)) from exc
        elif cmd == 'restart':
            text = self._restart()
        else:
            raise ControlError(f'unknown cell command {cmd!r} (reset | inject | restart)')
        self.log.append((self.steps, text))
        return text

    def _restart(self) -> str:
        """The cell's controller restarts IN PLACE (simulation control, P0B_DESIGN.md §3.3 rule 1; implementation
        decision D16): what a controller power cycle or program restart looks like to the robot while the cell's
        communication endpoint stays up -- every service a NEW source session, READY, no cycle and no rejection, its
        counters from zero, its station chains fresh; the actuators de-energised (open) and the axes holding where they
        are. The plant does not move by itself. The request images the cell holds are of the OLD session: at the next
        scan the new source counts each once as an image of another session and never executes it (rule 2)."""
        fresh: Dict[str, ProcessDevice] = {}
        sessions = []
        for sid, old in self.services.items():         # every new device first: a failing factory changes nothing
            session = new_session(self._session_rng)
            while session == old.session:              # a restart never reuses the session it replaces
                session = new_session(self._session_rng)
            try:
                fresh[sid] = self._new_device(sid, session)
            except ConfigError as exc:
                raise ControlError(f'the restart was not done (nothing changed): {exc}') from exc
            sessions.append(f'{sid} {old.session:#010x} -> {session:#010x}')
        for dev in self.devices.values():
            if isinstance(dev, BinaryActuator):
                dev.command(False)
            elif isinstance(dev, PositionAxis) and dev.position is not None:
                dev.target, dev.target_name = dev.position, None
        for sid, device in fresh.items():
            self.services[sid] = device
            self._published[sid] = None                # the new session's image goes out at the next scan
        return 'the controller RESTARTED (new source sessions: ' + '; '.join(sessions) + ')'

    def snapshot(self) -> Dict[str, object]:
        """The operator's view (the `/cell_io/state` summary and the CLI): never a contract value.

        Per service: the source's image fields, its `counters` (cycles, rejections, other-session images, resets: a
        re-delivery never changes them; a controller restart starts them from zero) and the transport's delivery counts
        `request_images` (every request image handed in) and `ignored_redelivered` (those that re-delivered an identity
        already consumed, or an older image of its requester: no-ops) -- kept across a controller restart."""
        return {
            'plant': self.plant.describe(),
            'step': self.steps,
            'time_s': self.clock.seconds(self.steps),
            'services': {
                sid: {
                    'state': STATE_NAMES.get(pd.image.state, str(pd.image.state)),
                    'detail': pd.image.detail,
                    'source_session': pd.image.source_session,
                    'heartbeat': pd.image.heartbeat,
                    'cycle_seq': pd.image.cycle_seq,
                    'cycle_done': pd.image.cycle_done,
                    'counters': pd.counters.as_dict(),
                    'request_images': self.request_images[sid],
                    'ignored_redelivered': self.ignored_redelivered[sid],
                } for sid, pd in self.services.items()
            },
            'devices': {name: dev.describe() for name, dev in self.devices.items()},
        }
