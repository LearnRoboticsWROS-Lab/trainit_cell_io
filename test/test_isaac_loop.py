"""U level: the Kit side of the Isaac plant without Isaac (trainit_cell_io.isaac.loop) -- the stage clock (D13), the
pacing, the per-step sequence, the transport callbacks. A recording transport stands in for the bundled rclpy.
"""

import copy
import json
import os
import random
import sys
from fractions import Fraction

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from trainit_cell_io.core import Cell, MockPlantAdapter, RequestImage, parse  # noqa: E402
from trainit_cell_io.core import contract as C  # noqa: E402
from trainit_cell_io.core import kit_boundary as KB  # noqa: E402
from trainit_cell_io.isaac.loop import KitSide, Pacer, SimClock  # noqa: E402

EXAMPLE = os.path.join(ROOT, 'examples', 'neutral_station', 'cell_io_mock.yaml')


class Recorder:
    def __init__(self):
        self.calls = []
        self.pending = []           # callbacks to deliver at the next spin

    def publish_status(self, service, text):
        self.calls.append(('status', service, json.loads(text)))

    def publish_clock(self, stamp_ns):
        self.calls.append(('clock', stamp_ns))

    def publish_state(self, text):
        self.calls.append(('state', json.loads(text)))

    def spin(self):
        self.calls.append(('spin',))
        pending, self.pending = self.pending, []
        for fn in pending:
            fn()


class FakeWall:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _kit(wall=None, start_ns=1_000_000_000_000):
    with open(EXAMPLE, encoding='utf-8') as handle:
        cfg = parse(copy.deepcopy(yaml.safe_load(handle)))
    cell = Cell(cfg, MockPlantAdapter(cfg.mock), rng=random.Random(9), sessions={'turner': 0xAAAA, 'indexer': 0xBBBB})
    rec = Recorder()
    lines = []
    kit = KitSide(cell, SimClock(cfg.clock.dt, start_ns), rec, log=lines.append, advance_plant=True,
                  wall=wall or FakeWall())
    return kit, rec, lines


# --- the stage clock (D13) ---------------------------------------------------------------------------------------

def test_the_stage_clock_is_wall_at_start_plus_exact_physics_time():
    clock = SimClock(Fraction(1, 200), 1_000)
    assert clock.stamp_ns(0) == 1_000 and clock.stamp_ns(1) == 5_001_000 and clock.stamp_ns(200) == 1_000_001_000
    assert clock.seconds(400) == 2.0
    sixtieth = SimClock(Fraction(1, 60), 0)
    assert [sixtieth.stamp_ns(n) for n in (1, 2, 3)] == [16_666_666, 33_333_333, 50_000_000]   # exact, floored
    with pytest.raises(ValueError):
        SimClock(0.005, 0)


def test_a_restarted_stage_never_stamps_backwards_at_rtf_at_most_one():
    # the first stage ran 10 s of wall time and (RTF <= 1) at most 10 s of physics; the restart starts later
    first = SimClock(Fraction(1, 200), 50_000_000_000)
    last_first = first.stamp_ns(2000)                         # 10 s of physics
    restarted = SimClock(Fraction(1, 200), 50_000_000_000 + 10_000_000_000 + 1)
    assert restarted.stamp_ns(0) > last_first


# --- pacing ------------------------------------------------------------------------------------------------------

def test_the_pacer_never_runs_ahead_of_real_time_and_measures_the_rtf():
    now = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s
    pacer = Pacer(0.005, now=lambda: now[0], sleep=sleep)
    pacer.start()
    for step in range(1, 201):      # an instant simulator: the pacer holds it at real time
        pacer.pace(step)
    assert now[0] == pytest.approx(1.0) and pacer.rtf(200) == pytest.approx(1.0)
    slept.clear()
    pacer.window_rtf(200)           # a new report window
    now[0] += 1.0                   # a slow stretch: 1 s of wall for 100 steps (RTF 0.5) -> no sleep, measured
    pacer.pace(300)
    assert slept == [] and pacer.window_rtf(300) == pytest.approx(0.5)


# --- one physics step: the order, the clock every step, the statuses, the state ----------------------------------

def test_each_step_cell_then_statuses_then_clock_then_spin_and_the_clock_every_step():
    kit, rec, _ = _kit()
    for _ in range(40):
        kit.after_physics_step()
    clocks = [c for c in rec.calls if c[0] == 'clock']
    spins = [c for c in rec.calls if c[0] == 'spin']
    assert len(clocks) == 40 and len(spins) == 40               # /clock after EVERY physics step
    assert [c[1] for c in clocks] == [kit.clock.stamp_ns(n) for n in range(1, 41)]
    order = [c[0] for c in rec.calls if c[0] in ('status', 'clock', 'spin')]
    first_status = order.index('status')
    assert order[first_status:first_status + 4] == ['status', 'status', 'clock', 'spin']   # both services, then
    statuses = [c for c in rec.calls if c[0] == 'status' and c[1] == 'turner']
    beats = [s[2]['heartbeat'] for s in statuses]
    assert beats == [0, 1, 2]       # the start image, then one per 100 ms period (steps 20, 40): minted by the scan
    assert all(s[2]['stamp_ns'] == kit.clock.stamp_ns(2) for s in statuses[:1])


def test_a_request_from_the_transport_reaches_the_cell_and_a_change_is_published_at_once():
    kit, rec, lines = _kit()
    kit.after_physics_step()
    session = kit.cell.services['turner'].session
    text = KB.encode_request('turner', RequestImage(session, 0xF00D, 1), 1)
    rec.pending.append(lambda: kit.on_request('turner', text))
    kit.after_physics_step()        # delivered at this step's spin, consumed at the next scan
    kit.after_physics_step()
    kit.after_physics_step()
    busy = [c for c in rec.calls if c[0] == 'status' and c[1] == 'turner' and c[2]['state'] == C.STATE_BUSY]
    assert busy and (busy[0][2]['cycle_requester'], busy[0][2]['cycle_seq']) == (0xF00D, 1)
    states = [c for c in rec.calls if c[0] == 'state']
    assert states and states[-1][1]['services']['turner']['state'] == 'BUSY'   # the operator view, at once
    assert kit.requests_received['turner'] == 1 and any('accepted' in line for line in lines)


def test_a_malformed_or_misrouted_request_image_is_refused_never_half_applied():
    kit, rec, lines = _kit()
    session = kit.cell.services['turner'].session
    kit.on_request('turner', 'nope')
    kit.on_request('turner', KB.encode_request('indexer', RequestImage(session, 1, 1), 1))
    bad_seq = KB.encode_request('turner', RequestImage(session, 1, 1), 1).replace('"seq": 1', '"seq": -1')
    kit.on_request('turner', bad_seq)
    assert kit.refused_requests == 3 and kit.requests_received == {'turner': 0, 'indexer': 0}
    assert sum('refused a request image' in line for line in lines) == 3
    for _ in range(10):
        kit.after_physics_step()
    assert kit.cell.services['turner'].state == C.STATE_READY


def test_the_requester_count_reaches_the_source_two_requesters_reject():
    kit, rec, _ = _kit()
    session = kit.cell.services['turner'].session
    kit.on_request('turner', KB.encode_request('turner', RequestImage(session, 5, 1), 2))
    for _ in range(4):
        kit.after_physics_step()
    img = kit.cell.services['turner'].image
    assert (img.reject_requester, img.reject_seq, img.reject_code) == (5, 1, C.REJECT_TWO_REQUESTERS)


def test_two_requesters_images_in_one_spin_are_both_consumed_in_two_successive_status_images():
    # review fix C2 on the Kit side: each boundary message (one image) is handed to the cell as it comes
    kit, rec, lines = _kit()
    kit.after_physics_step()
    session = kit.cell.services['turner'].session
    for requester in (0xA1, 0xB2):
        text = KB.encode_request('turner', RequestImage(session, requester, 1), 2)
        rec.pending.append(lambda text=text: kit.on_request('turner', text))
    for _ in range(6):
        kit.after_physics_step()
    rejects = [(c[2]['reject_requester'], c[2]['reject_code']) for c in rec.calls
               if c[0] == 'status' and c[1] == 'turner' and c[2]['reject_code']]
    assert rejects[:2] == [(0xA1, C.REJECT_TWO_REQUESTERS), (0xB2, C.REJECT_TWO_REQUESTERS)]
    state = [c[1] for c in rec.calls if c[0] == 'state'][-1]['services']['turner']    # review fix C3
    assert (state['request_images'], state['counters']['rejected_two_requesters']) == (2, 2)
    assert kit.requests_received['turner'] == 2


def test_the_kit_side_prints_every_new_log_line_exactly_once_from_a_bounded_log():
    # review fix C7: the cell's log is bounded; the Kit loop reads it by running total
    kit, rec, lines = _kit()
    for n in range(30):
        kit.on_command(json.dumps({'cmd': 'inject', 'fault': 'clear', 'id': f'c{n}'}))   # one log line each
        kit.after_physics_step()

    def cell_lines():                                    # the cell's own log lines ("[  t s] ..."), not the answers
        return [line for line in lines if line.startswith('[') and line.endswith('every injection cleared')]
    assert len(cell_lines()) == 30 and kit.cell.log.total >= 30
    kit.after_physics_step()
    assert len(cell_lines()) == 30                       # nothing twice
    small, _, small_lines = _kit()
    small.cell.log = type(small.cell.log)(4)             # a tiny log: a reader that falls behind is told
    for n in range(10):
        small.cell.control({'cmd': 'inject', 'fault': 'clear'})
    small.after_physics_step()
    assert any('6 cell log line(s) were dropped' in line for line in small_lines), small_lines
    assert sum(line.endswith('every injection cleared') for line in small_lines) == 4


def test_operator_commands_are_answered_and_the_state_published_at_one_hertz():
    wall = FakeWall()
    kit, rec, lines = _kit(wall=wall)
    for _ in range(4):              # past the first scan (its start images are a change: published at once)
        kit.after_physics_step()
    rec.calls.clear()
    for _ in range(10):             # 50 ms of physics, no wall time: no periodic state
        kit.after_physics_step()
    assert not [c for c in rec.calls if c[0] == 'state']
    wall.t += 1.01
    kit.after_physics_step()
    assert len([c for c in rec.calls if c[0] == 'state']) == 1
    kit.on_command(json.dumps({'cmd': 'inject', 'fault': 'hold', 'service': 'turner', 'id': 'c1'}))
    last = [c for c in rec.calls if c[0] == 'state'][-1][1]
    assert last['last_command'] == {'id': 'c1', 'cmd': 'inject', 'ok': True,
                                    'text': 'turner: held (no DONE until cleared)'}
    kit.on_command(json.dumps({'cmd': 'inject', 'fault': 'hold', 'service': 'turner', 'id': 'c1'}))  # a repeat
    assert sum('cell command done' in line for line in lines) == 1
