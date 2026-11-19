"""U level: trainit_cell_io's core, without ROS (P0_CHECKPOINT.md, P0.b frozen matrix).

Cases: C1 (no rclpy), C2 (the scan clock), C3 (external_plc refuses a ProcessDevice / logic), C4 (the mock plant is
deterministic); the SOURCE halves of PB-P2 (b), PB-P3, PB-P4, PB-P11 (rule 6), PB-P13, PB-P23; the §3.3 rules one by
one; the Kit boundary (C5's pure half).
"""

import ast
import copy
import json
import os
import random
import subprocess
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from trainit_cell_io.core import (Cell, ClockError, ConfigError, ControlError, MockPlantAdapter,  # noqa: E402
                                  RequestImage, ScanClock, load, parse, parse_dt)
from trainit_cell_io.core import contract as C  # noqa: E402
from trainit_cell_io.core import kit_boundary as KB  # noqa: E402
from trainit_cell_io.core.operator_desk import OperatorDesk  # noqa: E402
from trainit_cell_io.core.plant import PlantAdapter  # noqa: E402

EXAMPLE = os.path.join(ROOT, 'examples', 'neutral_station', 'cell_io_mock.yaml')


def _example():
    with open(EXAMPLE, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _cell(data=None, seed_sessions=(0x1111, 0x2222), rng_seed=7):
    cfg = parse(copy.deepcopy(data or _example()))
    plant = MockPlantAdapter(cfg.mock)
    sessions = dict(zip(cfg.services, seed_sessions))
    return Cell(cfg, plant, rng=random.Random(rng_seed), sessions=sessions), plant


def _run(cell, steps):
    out = []
    for _ in range(steps):
        out.extend(cell.step())
    return out


def _until(cell, service, state, limit_steps=20000):
    pd = cell.services[service]
    for _ in range(limit_steps):
        if pd.state == state:
            return
        cell.step()
    raise AssertionError(f'{service} never reached {C.STATE_NAMES[state]}: {pd.image}')


# --- C1: the core imports no rclpy ----------------------------------------------------------------------------------

def test_c1_the_core_imports_no_rclpy_nor_a_simulator():
    core_dir = os.path.join(ROOT, 'trainit_cell_io', 'core')
    forbidden = {'rclpy', 'rclcpp', 'std_msgs', 'trainit_interfaces', 'omni', 'isaacsim', 'pxr', 'carb'}
    for name in sorted(os.listdir(core_dir)):
        if not name.endswith('.py'):
            continue
        tree = ast.parse(open(os.path.join(core_dir, name), encoding='utf-8').read())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            for mod in mods:
                assert mod.split('.')[0] not in forbidden, f'{name} imports {mod}'
    # and at run time: importing the core in a fresh interpreter loads no rclpy
    code = ('import sys; sys.path.insert(0, %r); import trainit_cell_io.core; '
            'print("rclpy" in sys.modules)' % ROOT)
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


# --- C2: integer physics steps; a ms value not a multiple of dt fails loudly ----------------------------------------

def test_c2_the_scan_clock_counts_integer_physics_steps():
    clock = ScanClock(parse_dt(physics_dt_ms=5), scan_every_steps=2, status_period_ms=100)
    assert clock.steps(2000, 'timeout') == 400
    assert clock.status_period_steps == 20
    assert clock.ms(400) == 2000.0


@pytest.mark.parametrize('ms', [1, 7, 2001, 3])
def test_c2_a_ms_value_not_a_multiple_of_dt_fails_loudly(ms):
    clock = ScanClock(parse_dt(physics_dt_ms=5), scan_every_steps=2, status_period_ms=100)
    with pytest.raises(ClockError, match='not a whole number of physics steps'):
        clock.steps(ms, 'timeout')


def test_c2_an_isaac_sixtieth_dt_is_exact_and_refuses_10_ms():
    clock = ScanClock(parse_dt(physics_dt='1/60'), scan_every_steps=1, status_period_ms=100)
    assert clock.steps(100) == 6 and clock.steps(5000) == 300
    with pytest.raises(ClockError):
        clock.steps(10)


def test_c2_a_config_whose_timeout_is_not_whole_steps_is_refused_at_load():
    data = _example()
    data['devices']['clamp']['timeout_ms'] = 2003
    with pytest.raises(ConfigError, match='not a whole number of physics steps'):
        parse(data)


def test_c2_a_status_period_that_is_not_whole_scans_is_refused():
    data = _example()
    data['scan'] = {'physics_dt_ms': 5, 'scan_every_steps': 3, 'status_period_ms': 100}
    with pytest.raises(ConfigError, match='not a whole number of scans'):
        parse(data)


def test_c2_non_integer_or_float_ms_never_rounded():
    clock = ScanClock(parse_dt(physics_dt_ms=5))
    for bad in (100.0, '100', True):
        with pytest.raises(ClockError):
            clock.steps(bad)


# --- C3: control_mode external_plc refuses a ProcessDevice and a logic backend --------------------------------------

def test_c3_external_plc_refuses_a_process_device():
    data = _example()
    data['control_mode'] = 'external_plc'
    del data['plant']
    with pytest.raises(ConfigError, match='external_plc refuses a simulated machine service'):
        parse(data)


def test_c3_external_plc_refuses_a_logic_backend_even_alone():
    data = {'cell_io': 1, 'control_mode': 'external_plc',
            'services': {'turner': {'logic': 'python_logic'}}}
    with pytest.raises(ConfigError, match='external_plc'):
        parse(data)


def test_c3_external_plc_refuses_a_plant():
    data = {'cell_io': 1, 'control_mode': 'external_plc', 'plant': 'mock'}
    with pytest.raises(ConfigError, match='no simulated plant'):
        parse(data)


# --- a refused cell file is ALWAYS a ConfigError at load (implementation review fix C4: Python review MINOR 1) -------

def _set_step(part, index, step):
    def mutate(data):
        data['services']['turner'][part][index] = step
    return mutate


def _set_device(device, key, value):
    def mutate(data):
        data['devices'][device][key] = value
    return mutate


def _set_position(name, value):
    def mutate(data):
        data['devices']['table']['positions'][name] = value
    return mutate


def _nan_physics_dt(data):
    del data['scan']['physics_dt_ms']
    data['scan']['physics_dt'] = float('nan')


@pytest.mark.parametrize('mutate, why', [
    (_set_step('cycle', 1, {'close': ['clamp']}), r"unknown device \['clamp'\]"),     # was a TypeError (unhashable)
    (_set_step('cycle', 2, {'move': 'table', 'to': ['turned']}), "has no position"),  # was a TypeError in Cell()
    (_set_step('cycle', 1, {'close': 'table'}), 'close needs a binary_actuator'),     # was a ConfigError only in Cell()
    (_set_step('cycle', 0, {'require': 'clamp'}), 'require needs a digital_sensor'),
    (_set_step('cycle', 1, 'close clamp'), 'a step is a mapping'),
    (_set_step('cycle', 1, {'close': 'clamp', 'open': 'clamp'}), 'unknown step'),
    (_set_step('home', 1, {'move': 'table'}), 'unknown step'),
    (_set_step('home', 1, {'move': 'table', 'to': 'nowhere'}), "has no position 'nowhere'"),
    (_set_position('turned', float('nan')), 'must be a finite number'),              # was a DeviceError in Cell()
    (_set_position('turned', float('inf')), 'must be a finite number'),
    (_set_device('table', 'band', float('inf')), 'finite number > 0'),               # was accepted everywhere
    (_set_device('table', 'band', float('nan')), 'finite number > 0'),
    (_nan_physics_dt, 'finite number of seconds'),                                   # was a ValueError (Fraction)
])
def test_a_malformed_cell_file_is_a_config_error_at_load_never_a_type_error(mutate, why):
    data = _example()
    mutate(data)
    with pytest.raises(ConfigError, match=why):
        parse(data)


def test_non_finite_yaml_values_are_refused_by_load(tmp_path):
    text = open(EXAMPLE, encoding='utf-8').read()
    for old, new, why in (('positions: {load: 0.0, turned: 180.0}', 'positions: {load: 0.0, turned: .nan}',
                           'finite number'),
                          ('positions: {home: 0.0, forward: 300.0}', 'positions: {home: 0.0, forward: .inf}',
                           'finite number'),
                          ('    band: 1.0\n    timeout_ms: 4000\n    description: the turntable',
                           '    band: .inf\n    timeout_ms: 4000\n    description: the turntable', 'finite number'),
                          ('- {close: clamp}', '- {close: [clamp]}', 'unknown device')):
        assert old in text, old
        path = tmp_path / 'cell_io.yaml'
        path.write_text(text.replace(old, new, 1), encoding='utf-8')
        with pytest.raises(ConfigError, match=why):
            load(str(path))


def test_device_clock_and_logic_errors_are_config_errors():
    """Whatever a later layer refuses is a ConfigError too: an executable that catches ConfigError never prints a
    traceback for a refused file."""
    from trainit_cell_io.core.devices import DeviceError, PositionAxis
    from trainit_cell_io.core.logic import LogicError, build_chain
    assert issubclass(DeviceError, ConfigError) and issubclass(ClockError, ConfigError)
    assert issubclass(LogicError, ConfigError)
    with pytest.raises(ConfigError, match='finite number > 0'):
        PositionAxis('table', {'load': 0.0}, float('inf'), 10, 50, 'deg')
    with pytest.raises(ConfigError, match='finite number'):
        PositionAxis('table', {'load': float('nan')}, 1.0, 10, 50, 'deg')
    table = PositionAxis('table', {'load': 0.0}, 1.0, 10, 50, 'deg')
    with pytest.raises(ConfigError, match='has no position'):
        build_chain('x', [{'move': 'table', 'to': ['load']}], {'table': table})     # was a TypeError (unhashable)
    with pytest.raises(ConfigError, match='finite number of seconds'):
        parse_dt(physics_dt=float('inf'))


@pytest.mark.parametrize('model, why', [
    ({'table': {'speed': 180.0, 'start': ['load']}}, 'no position'),     # was a TypeError (unhashable)
    ({'clamp': {'close_ms': float('inf'), 'open_ms': 300}}, 'finite number > 0'),
    ({'table': {'speed': float('nan'), 'start': 'load'}}, 'finite number > 0'),
])
def test_the_mock_model_refuses_unhashable_or_non_finite_values_readably(model, why):
    from trainit_cell_io.core import InjectionError
    data = _example()
    data['mock']['devices'].update(model)
    cfg = parse(data)
    with pytest.raises(InjectionError, match=why):
        Cell(cfg, MockPlantAdapter(cfg.mock), rng=random.Random(1))


# --- the status period's bound (implementation review fix C9: protocol N2) ------------------------------------------

@pytest.mark.parametrize('period', [510, 600, 1000, 2000])
def test_a_status_period_above_half_the_robots_default_max_age_is_refused(period):
    data = _example()
    data['scan']['status_period_ms'] = period
    with pytest.raises(ConfigError, match=r'status_period_ms = %d ms is above 500 ms: the robot synchronises a request '
                                          r'on two heartbeat changes' % period):
        parse(data)


def test_a_status_period_of_at_most_500_ms_is_accepted():
    for period in (20, 100, 250, 500):
        data = _example()
        data['scan']['status_period_ms'] = period
        assert parse(data).clock.status_period_steps == period // 5


def test_c3_the_node_exits_non_zero_on_external_plc_with_a_service(tmp_path):
    data = _example()
    data['control_mode'] = 'external_plc'
    del data['plant']
    path = tmp_path / 'cell_io.yaml'
    path.write_text(yaml.safe_dump(data))
    code = ('import sys; sys.path.insert(0, %r); from trainit_cell_io.core.config import load, ConfigError\n'
            'try:\n    load(%r)\nexcept ConfigError as e:\n    print(e); sys.exit(2)\n' % (ROOT, str(path)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert out.returncode == 2 and 'external_plc' in out.stdout


# --- C4: the mock plant is deterministic ------------------------------------------------------------------------------

def _scenario_trace(seed, jitter=0.2):
    data = _example()
    data['mock']['seed'] = seed
    data['mock']['jitter'] = jitter
    cell, plant = _cell(data)
    turner = cell.services['turner']
    for n in range(1, 6):
        cell.set_request('turner', RequestImage(turner.session, 0xABC, n))
        _run(cell, 4)
        _until(cell, 'turner', C.STATE_DONE)
    cell.control({'cmd': 'inject', 'fault': 'stuck_actuator', 'device': 'clamp'})
    cell.set_request('turner', RequestImage(turner.session, 0xABC, 6))
    _until(cell, 'turner', C.STATE_FAULT)
    return list(plant.trace), list(cell.log)


def test_c4_same_seed_same_trace():
    assert _scenario_trace(42) == _scenario_trace(42)


def test_c4_another_seed_another_trace_with_jitter():
    assert _scenario_trace(42)[0] != _scenario_trace(43)[0]


def test_c4_fault_injections_are_refused_when_unknown():
    cell, _ = _cell()
    with pytest.raises(ControlError, match='unknown fault'):
        cell.control({'cmd': 'inject', 'fault': 'melt'})
    with pytest.raises(ControlError, match='no actuator or axis'):
        cell.control({'cmd': 'inject', 'fault': 'stuck_actuator', 'device': 'robot'})
    with pytest.raises(ControlError, match='unknown cell command'):
        cell.control({'cmd': 'start_cycle'})   # the cell has no robot-facing command


# --- §3.3 rules, source side ----------------------------------------------------------------------------------------

def test_rule1_a_source_starts_ready_with_no_cycle_and_no_rejection():
    cell, _ = _cell()
    img = cell.services['turner'].image
    assert img.state == C.STATE_READY and img.source_session == 0x1111
    assert (img.cycle_requester, img.cycle_seq, img.cycle_done) == (0, 0, False)
    assert (img.reject_requester, img.reject_seq, img.reject_code) == (0, 0, C.REJECT_NONE)


def test_rule3_rule7_ready_then_accepted_busy_then_done_level():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _run(cell, 2)
    assert t.state == C.STATE_BUSY and (t.image.cycle_requester, t.image.cycle_seq) == (0xA, 1)
    assert not t.image.cycle_done
    _until(cell, 'turner', C.STATE_DONE)
    assert t.image.cycle_done
    _run(cell, 400)
    assert t.state == C.STATE_DONE and t.counters.accepted == 1   # level: DONE until the next request


def test_pb_p4_a_re_delivered_image_runs_one_cycle():
    cell, _ = _cell()
    t = cell.services['turner']
    req = RequestImage(t.session, 0xA, 1)
    for _ in range(50):              # the level-held image, re-delivered at every scan and more
        cell.set_request('turner', req)
        _run(cell, 3)
    _until(cell, 'turner', C.STATE_DONE)
    for _ in range(50):
        cell.set_request('turner', req)
        _run(cell, 3)
    assert t.counters.accepted == 1 and t.counters.completed == 1 and t.state == C.STATE_DONE


def test_pb_p2b_a_new_request_reaching_a_busy_source_is_rejected_and_the_cycle_continues():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _run(cell, 20)
    assert t.state == C.STATE_BUSY
    cell.set_request('turner', RequestImage(t.session, 0xA, 2))
    _run(cell, 2)
    assert (t.image.reject_requester, t.image.reject_seq, t.image.reject_code) == (0xA, 2, C.REJECT_BUSY)
    assert t.state == C.STATE_BUSY and t.image.cycle_seq == 1        # untouched
    _until(cell, 'turner', C.STATE_DONE)
    assert t.image.cycle_seq == 1 and t.image.cycle_done              # its own waiter succeeds
    assert t.counters.accepted == 1 == t.counters.completed           # the cycle counter = the accepted requests
    _run(cell, 400)
    assert t.counters.accepted == 1                                   # the rejected one never runs


def test_pb_p13_a_request_reaching_a_fault_source_is_consumed_and_never_runs_after_reset():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.control({'cmd': 'inject', 'fault': 'stuck_actuator', 'device': 'clamp'})
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _until(cell, 'turner', C.STATE_FAULT)
    assert t.image.fault_code == C.FAULT_ACTUATOR_TIMEOUT and not t.image.cycle_done
    cell.set_request('turner', RequestImage(t.session, 0xA, 2))
    _run(cell, 2)
    assert (t.image.reject_seq, t.image.reject_code) == (2, C.REJECT_FAULTED)
    cell.control({'cmd': 'inject', 'fault': 'clear'})
    cell.control({'cmd': 'reset', 'service': 'turner'})
    assert t.state == C.STATE_HOMING
    _until(cell, 'turner', C.STATE_READY)
    _run(cell, 1000)                                                  # the image stays level-held
    assert t.counters.accepted == 1 and t.state == C.STATE_READY      # 0 cycles ran after the RESET
    assert t.image.cycle_seq == 1 and not t.image.cycle_done          # the aborted cycle stays visible


def test_pb_p23_a_request_during_homing_is_rejected_busy_and_never_runs_after_the_homing():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.control({'cmd': 'reset', 'service': 'turner'})
    cell.control({'cmd': 'inject', 'fault': 'slow', 'device': 'clamp', 'factor': 20})
    _run(cell, 2)
    assert t.state == C.STATE_HOMING
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _run(cell, 2)
    assert (t.image.reject_seq, t.image.reject_code) == (1, C.REJECT_BUSY)
    cell.control({'cmd': 'inject', 'fault': 'clear'})
    _until(cell, 'turner', C.STATE_READY)
    _run(cell, 1000)
    assert t.counters.accepted == 0


def test_pb_p3_a_restarted_source_never_executes_the_old_sessions_request():
    data = _example()
    cell, _ = _cell(data, seed_sessions=(0x1111, 0x2222))
    t = cell.services['turner']
    old = RequestImage(t.session, 0xA, 1)
    cell.set_request('turner', old)
    _run(cell, 20)
    assert t.state == C.STATE_BUSY
    # the source restarts (a new process): a new session; the level-held image of the old session is still there
    cell2, _ = _cell(data, seed_sessions=(0x3333, 0x4444))
    t2 = cell2.services['turner']
    assert t2.session != t.session
    cell2.set_request('turner', old)
    _run(cell2, 2000)
    assert t2.counters.accepted == 0 and t2.state == C.STATE_READY and t2.counters.ignored_other_session == 1
    # the next request, for the NEW session, runs
    cell2.set_request('turner', RequestImage(t2.session, 0xA, 2))
    _until(cell2, 'turner', C.STATE_DONE)
    assert t2.counters.accepted == 1


def test_d16_an_in_place_controller_restart_is_a_new_session_that_never_runs_the_old_request():
    """D16: `restart` -- every service a NEW session, READY, counters from zero, chains fresh; the actuators
    de-energised, the axes holding; the transport's request image of the old session is never executed (rule 2) and is
    counted once; the next request for the new session runs a full cycle."""
    cell, plant = _cell()
    t = cell.services['turner']
    old_session, idx_session = t.session, cell.services['indexer'].session
    old = RequestImage(old_session, 0xA, 1)
    cell.set_request('turner', old)
    _until(cell, 'turner', C.STATE_BUSY)
    _run(cell, 200)                                         # mid-cycle: the clamp closed, the table moving
    assert cell.devices['clamp'].cmd is True
    text = cell.control({'cmd': 'restart'})
    assert 'RESTARTED' in text and f'{old_session:#010x}' in text
    t2 = cell.services['turner']
    assert t2 is not t and t2.session not in (0, old_session)
    assert cell.services['indexer'].session != idx_session
    assert t2.state == C.STATE_READY and t2.image.cycle_seq == 0 and t2.image.reject_seq == 0
    assert t2.counters.as_dict() == C.DeviceCounters().as_dict()
    assert cell.devices['clamp'].cmd is False                # de-energised
    table = cell.devices['table']
    held_at = table.target                                  # the last read-back: the axis holds there
    assert held_at is not None and 0.0 < held_at < 180.0 and table.target_name is None
    published = _run(cell, 2)                               # the new session goes out at the next scan
    assert any(sid == 'turner' and img.source_session == t2.session for sid, img in published)
    _run(cell, 2000)                                        # the old image is still the transport's latest
    assert t2.counters.accepted == 0 and t2.counters.ignored_other_session == 1 and t2.state == C.STATE_READY
    assert table.target == held_at and abs(table.position - held_at) < 1e-9   # stopped at the hold, not at a position
    cell.set_request('turner', RequestImage(t2.session, 0xA, 2))
    _until(cell, 'turner', C.STATE_DONE)
    assert t2.counters.accepted == t2.counters.completed == 1 and t2.image.cycle_seq == 2


def test_d16_the_restart_command_is_answered_by_the_operator_desk_and_refuses_nothing_else():
    cell, _ = _cell()
    desk = OperatorDesk(cell)
    answer, repeated = desk.handle('{"cmd": "restart", "id": "r1"}')
    assert answer['ok'] and not repeated and 'RESTARTED' in answer['text']
    again, repeated = desk.handle('{"cmd": "restart", "id": "r1"}')        # at most once per id
    assert repeated and again == answer
    bad, _ = desk.handle('{"cmd": "reboot", "id": "r2"}')
    assert not bad['ok'] and 'restart' in bad['text']


# --- rule 2 per requester: a re-delivery stays a no-op across a two-requester episode (review fix C1: protocol MAJOR-1)

R_SESSION, X_SESSION = 0x0000A11C, 0x0000B0B0


def test_rule2_a_re_delivered_image_after_a_two_requester_episode_never_runs_a_second_cycle():
    """The protocol review's probe: R (r,5) accepted and DONE; X (x,1) rejected TWO_REQUESTERS; X leaves; R's latched
    (r,5) re-delivered (a bridge restart, a DDS re-match) -> before the fix: ACCEPTED again, a second cycle."""
    cell, _ = _cell()
    t = cell.services['turner']
    r5 = RequestImage(t.session, R_SESSION, 5)
    cell.set_request('turner', r5)
    _run(cell, 4)
    _until(cell, 'turner', C.STATE_DONE)
    cell.set_request('turner', RequestImage(t.session, X_SESSION, 1), requesters=2)
    _run(cell, 4)
    rejected = (t.image.reject_requester, t.image.reject_seq, t.image.reject_code)
    assert rejected == (X_SESSION, 1, C.REJECT_TWO_REQUESTERS)
    cell.set_requesters('turner', 1)                         # X leaves
    for _ in range(5):                                       # R's latched image re-delivered, again and again
        cell.set_request('turner', r5)
        _run(cell, 40)
    assert t.counters.accepted == 1 and t.counters.completed == 1 and t.state == C.STATE_DONE   # one cycle, ever
    assert (t.image.cycle_requester, t.image.cycle_seq, t.image.cycle_done) == (R_SESSION, 5, True)
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 6))       # R's NEXT request still runs
    _run(cell, 4)
    _until(cell, 'turner', C.STATE_DONE)
    assert t.counters.accepted == 2 and t.image.cycle_seq == 6


def test_rule2_a_request_rejected_faulted_never_runs_after_the_reset_even_after_another_requester():
    """The second probe: R (r,1) reaches a FAULT source -> REJECTED FAULTED (consumed); X (x,1) is consumed in between
    (TWO_REQUESTERS); the RESET homes the source; R's image re-delivered -> before the fix: ACCEPTED, a cycle nobody
    waits for."""
    cell, _ = _cell()
    t = cell.services['turner']
    _inject(cell, fault='station_fault', service='turner')
    _run(cell, 2)
    assert t.state == C.STATE_FAULT
    r1 = RequestImage(t.session, R_SESSION, 1)
    cell.set_request('turner', r1)
    _run(cell, 2)
    assert (t.image.reject_requester, t.image.reject_seq, t.image.reject_code) == (R_SESSION, 1, C.REJECT_FAULTED)
    cell.set_request('turner', RequestImage(t.session, X_SESSION, 1), requesters=2)
    _run(cell, 2)
    assert t.image.reject_requester == X_SESSION and t.counters.rejected_two_requesters == 1
    cell.set_requesters('turner', 1)
    _inject(cell, fault='clear')
    cell.control({'cmd': 'reset', 'service': 'turner'})
    _until(cell, 'turner', C.STATE_READY)
    for _ in range(5):
        cell.set_request('turner', r1)
        _run(cell, 40)
    assert t.counters.accepted == 0 and t.state == C.STATE_READY             # 0 cycles after the RESET


def test_rule2_classify_keeps_the_last_consumed_identity_and_each_requesters_last_seq():
    from trainit_cell_io.core.process_device import CONSUMED, MALFORMED, NEW, OTHER_SESSION, REMEMBERED_REQUESTERS
    cell, _ = _cell()
    t = cell.services['turner']
    assert t.classify(None) == MALFORMED and t.classify(RequestImage(t.session, 0, 1)) == MALFORMED
    assert t.classify(RequestImage(t.session ^ 1, R_SESSION, 1)) == OTHER_SESSION
    assert t.classify(RequestImage(t.session, R_SESSION, 1)) == NEW
    assert t.consume(RequestImage(t.session, R_SESSION, 1)) == 'accepted'
    assert t.classify(RequestImage(t.session, R_SESSION, 1)) == CONSUMED           # the last consumed identity
    assert t.consume(RequestImage(t.session, X_SESSION, 7)) == 'rejected'           # BUSY
    assert t.classify(RequestImage(t.session, R_SESSION, 1)) == CONSUMED           # R's last seq (per requester)
    assert t.classify(RequestImage(t.session, R_SESSION, 2)) == NEW                # another seq: a new request
    assert t.consume(RequestImage(t.session, R_SESSION, 1)) is None                # a no-op: nothing counted
    assert (t.counters.accepted, t.counters.rejected_busy) == (1, 1)
    # bounded: the last REMEMBERED_REQUESTERS requesters (least recently used forgotten first)
    for n in range(1, REMEMBERED_REQUESTERS + 3):
        t.consume(RequestImage(t.session, 0x100 + n, 1))
    assert len(t._consumed_by) == REMEMBERED_REQUESTERS                             # noqa: SLF001
    last = range(3 + 0x100, REMEMBERED_REQUESTERS + 3 + 0x100)
    assert all(t.classify(RequestImage(t.session, rs, 1)) == CONSUMED for rs in last)


def test_rule2_an_older_image_of_a_requester_is_a_replay_never_a_new_request():
    """The verify round's replay probe: the bridge -> Kit request link keeps a depth-20 TRANSIENT_LOCAL history (every
    requester's image reaches Kit); a reader that matches that writer again could be re-sent the history in order --
    R's seq 1..5 after R's 5 was consumed. Before this rule seq 1 ran a SECOND cycle and the newest (5) was rejected
    BUSY; a requester's seq only grows (wrapping), so an older one is a replay: CONSUMED (a no-op, counted as such)."""
    from trainit_cell_io.core.process_device import CONSUMED, NEW
    cell, _ = _cell()
    t = cell.services['turner']
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 5))
    _run(cell, 4)
    _until(cell, 'turner', C.STATE_DONE)
    before = cell.ignored_redelivered['turner']
    for seq in (1, 2, 3, 4, 5):                              # the ordered replay, one image per scan
        assert t.classify(RequestImage(t.session, R_SESSION, seq)) == CONSUMED, seq
        cell.set_request('turner', RequestImage(t.session, R_SESSION, seq))
        _run(cell, 2)
    _run(cell, 40)
    assert t.counters.accepted == 1 and t.counters.completed == 1 and t.counters.rejected_busy == 0
    assert (t.image.cycle_requester, t.image.cycle_seq, t.image.reject_seq) == (R_SESSION, 5, 0)
    assert cell.ignored_redelivered['turner'] - before == 5
    assert t.classify(RequestImage(t.session, R_SESSION, 6)) == NEW                 # R's next request still runs
    assert t.classify(RequestImage(t.session, X_SESSION, 1)) == NEW                 # another requester: its own order


def test_rule2_the_seq_order_wraps_like_the_robots_seq():
    """32-bit serial-number order: the robot's seq wraps (... 0xFFFFFFFF, 1: 0 skipped, PB-P6) and stays newer."""
    from trainit_cell_io.core.process_device import CONSUMED, NEW, seq_newer
    assert seq_newer(2, 1) and seq_newer(1, 0xFFFFFFFF) and seq_newer(0x7FFFFFFF, 0)
    assert not seq_newer(1, 1) and not seq_newer(1, 2) and not seq_newer(0xFFFFFFFF, 1) and not seq_newer(0x80000000, 0)
    cell, _ = _cell()
    t = cell.services['turner']
    assert t.consume(RequestImage(t.session, R_SESSION, 0xFFFFFFFE)) == 'accepted'
    _until(cell, 'turner', C.STATE_DONE)
    assert t.classify(RequestImage(t.session, R_SESSION, 0xFFFFFFFF)) == NEW
    assert t.consume(RequestImage(t.session, R_SESSION, 0xFFFFFFFF)) == 'accepted'
    _until(cell, 'turner', C.STATE_DONE)
    assert t.classify(RequestImage(t.session, R_SESSION, 1)) == NEW                 # wrapped: still newer
    assert t.classify(RequestImage(t.session, R_SESSION, 0xFFFFFFFE)) == CONSUMED   # behind: a replay


def test_rule2_an_image_of_another_session_is_counted_once_however_often_it_comes():
    cell, _ = _cell()
    t = cell.services['turner']
    a, b = RequestImage(0x9999, R_SESSION, 1), RequestImage(0x9999, X_SESSION, 1)
    for image in (a, b, a, b, a):
        assert t.consume(image) is None
    assert t.counters.ignored_other_session == 2 and t.counters.accepted == 0


# --- every requester's latest image, one consumption per scan (review fix C2: protocol MAJOR-1, fix 2) ---------------

def _published(cell, service, steps):
    out = []
    for _ in range(steps):
        out.extend((cell.steps, image) for sid, image in cell.step() if sid == service)
    return out


def _first_named(images):
    """(the step of the first published image naming each rejection, its requester), in order."""
    seen, out = set(), []
    for step, image in images:
        key = (image.reject_requester, image.reject_seq)
        if image.reject_code and key not in seen:
            seen.add(key)
            out.append((step, image.reject_requester))
    return out


def test_two_requesters_images_within_one_scan_are_two_consumptions_in_two_successive_images():
    """Before the fix the cell kept ONE latest image per service: the first image was displaced, never consumed and
    never named (its waiter ended 305; a later re-delivery could execute it late)."""
    cell, _ = _cell()
    t = cell.services['turner']
    _run(cell, 3)                                            # between two scans (the next one at step 4)
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 1), requesters=2)
    cell.set_request('turner', RequestImage(t.session, X_SESSION, 1), requesters=2)
    images = _published(cell, 'turner', 7)
    named = [(step, (i.reject_requester, i.reject_seq, i.reject_code)) for step, i in images]
    assert named == [(4, (R_SESSION, 1, C.REJECT_TWO_REQUESTERS)), (6, (X_SESSION, 1, C.REJECT_TWO_REQUESTERS))]
    assert t.counters.rejected_two_requesters == 2 and t.counters.accepted == 0 and t.state == C.STATE_READY
    verdicts = [text for _, text in cell.log if ' of requester ' in text]
    assert verdicts == [f'turner: request 1 of requester {rs:#010x} rejected TWO_REQUESTERS (the source was READY)'
                        for rs in (R_SESSION, X_SESSION)]


def test_two_images_with_a_lagging_requester_count_are_accepted_then_rejected_busy_both_named():
    cell, _ = _cell()
    t = cell.services['turner']
    _run(cell, 3)
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 1))
    cell.set_request('turner', RequestImage(t.session, X_SESSION, 1))
    images = _published(cell, 'turner', 3)
    assert [(step, i.state, (i.cycle_requester, i.cycle_seq), (i.reject_requester, i.reject_code))
            for step, i in images] == [(4, C.STATE_BUSY, (R_SESSION, 1), (0, C.REJECT_NONE)),
                                       (6, C.STATE_BUSY, (R_SESSION, 1), (X_SESSION, C.REJECT_BUSY))]
    _until(cell, 'turner', C.STATE_DONE)
    _run(cell, 400)
    assert (t.counters.accepted, t.counters.completed, t.counters.rejected_busy) == (1, 1, 1)   # X's never runs


def test_consume_delay_holds_each_image_exactly_from_the_scan_that_first_saw_that_image():
    cell, _ = _cell()
    t = cell.services['turner']
    _inject(cell, fault='consume_delay', service='turner', ms=100)       # 20 physics steps
    _run(cell, 3)
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 1), requesters=2)     # seen at scan 4
    _run(cell, 10)
    cell.set_request('turner', RequestImage(t.session, X_SESSION, 1), requesters=2)     # seen at scan 14
    assert _first_named(_published(cell, 'turner', 40)) == [(24, R_SESSION), (34, X_SESSION)]
    cell2, _ = _cell()
    t2 = cell2.services['turner']
    _inject(cell2, fault='consume_delay', service='turner', ms=100)
    _run(cell2, 3)
    cell2.set_request('turner', RequestImage(t2.session, R_SESSION, 1), requesters=2)   # both seen at scan 4:
    cell2.set_request('turner', RequestImage(t2.session, X_SESSION, 1), requesters=2)   # released at 24, one a scan
    assert _first_named(_published(cell2, 'turner', 40)) == [(24, R_SESSION), (26, X_SESSION)]
    assert sum('held back 100 ms' in text for _, text in cell2.log) == 2


def test_the_held_images_are_bounded_to_the_last_requesters_of_a_service():
    from trainit_cell_io.core.process_device import REMEMBERED_REQUESTERS
    cell, _ = _cell()
    t = cell.services['turner']
    for n in range(REMEMBERED_REQUESTERS + 4):
        cell.set_request('turner', RequestImage(t.session, 0x200 + n, 1))
    assert len(cell._held['turner']) == REMEMBERED_REQUESTERS                      # noqa: SLF001
    _run(cell, 2 * (REMEMBERED_REQUESTERS + 4))
    c = t.counters
    assert c.accepted == 1 and c.rejected_busy == REMEMBERED_REQUESTERS - 1        # one consumption per scan
    assert t.image.cycle_requester == 0x200 + 4                                    # the 4 least recent: forgotten


# --- the delivery counts in the snapshot (review fix C3: the PB-P21 evidence) -------------------------------------

def test_the_snapshot_counts_the_request_images_and_the_redeliveries_per_service():
    cell, _ = _cell()
    t = cell.services['turner']
    desk = OperatorDesk(cell)
    r1 = RequestImage(t.session, R_SESSION, 1)
    cell.set_request('turner', r1)
    _run(cell, 4)
    _until(cell, 'turner', C.STATE_DONE)
    before = json.loads(desk.state_json())['services']['turner']
    for _ in range(3):                                       # a restarted bridge re-forwards the latched image
        cell.set_request('turner', r1)
        _run(cell, 2)
    cell.set_request('turner', RequestImage(t.session, 0, 0))                     # a default image: handed, ignored
    state = json.loads(desk.state_json())['services']
    turner = state['turner']
    assert (turner['request_images'], turner['ignored_redelivered']) == (5, 3)
    assert turner['counters'] == before['counters'] and turner['counters']['accepted'] == 1   # nothing ran twice
    assert not {'request_images', 'ignored_redelivered'} & set(turner['counters'])         # the counters stay stable
    assert (state['indexer']['request_images'], state['indexer']['ignored_redelivered']) == (0, 0)
    cell.control({'cmd': 'restart'})                         # the transport's counts survive a controller restart
    after = cell.snapshot()['services']['turner']
    assert (after['request_images'], after['ignored_redelivered'], after['counters']['accepted']) == (5, 3, 0)
    _run(cell, 2)                                            # the held image: the new source counts it once
    assert cell.snapshot()['services']['turner']['counters']['ignored_other_session'] == 1


def test_rule6_two_requesters_every_new_request_rejected():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.set_request('turner', RequestImage(t.session, 0xA, 1), requesters=2)
    _run(cell, 400)
    assert (t.image.reject_seq, t.image.reject_code) == (1, C.REJECT_TWO_REQUESTERS)
    assert t.counters.accepted == 0 and t.state == C.STATE_READY      # no cycle runs
    cell.set_request('turner', RequestImage(t.session, 0xB, 1), requesters=1)
    _until(cell, 'turner', C.STATE_DONE)                              # back to one requester: it runs


def test_rule2_a_malformed_or_default_image_is_never_acted_on():
    cell, _ = _cell()
    t = cell.services['turner']
    for bad in (RequestImage(0, 0, 0), RequestImage(t.session, 0, 1), RequestImage(t.session, 0xA, 0)):
        cell.set_request('turner', bad)
        _run(cell, 10)
    assert t.counters.accepted == 0 and t.state == C.STATE_READY


def test_rule8_a_station_fault_aborts_the_cycle_and_persists_until_reset():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _run(cell, 40)
    cell.control({'cmd': 'inject', 'fault': 'station_fault', 'service': 'turner', 'detail': 'guard door open'})
    _run(cell, 2)
    assert t.state == C.STATE_FAULT and t.image.fault_code == C.FAULT_STATION_FAULT
    assert 'guard door open' in t.image.detail and not t.image.cycle_done and t.counters.aborted == 1
    _run(cell, 1000)
    assert t.state == C.STATE_FAULT
    cell.control({'cmd': 'reset', 'service': 'turner'})               # the cause still there: faults again
    _run(cell, 4)
    assert t.state == C.STATE_FAULT
    cell.control({'cmd': 'inject', 'fault': 'clear'})
    cell.control({'cmd': 'reset', 'service': 'turner'})
    _until(cell, 'turner', C.STATE_READY)
    assert t.image.fault_code == C.FAULT_NONE


def test_rule8_a_failed_sensor_is_a_sensor_fault():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.control({'cmd': 'inject', 'fault': 'failed_sensor', 'device': 'part_present'})
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _until(cell, 'turner', C.STATE_FAULT)
    assert t.image.fault_code == C.FAULT_SENSOR_FAULT


def test_rule8_both_limits_on_is_a_sensor_fault():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.control({'cmd': 'inject', 'fault': 'failed_sensor', 'device': 'clamp', 'signal': 'closed'})
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _until(cell, 'turner', C.STATE_FAULT)
    assert t.image.fault_code == C.FAULT_SENSOR_FAULT and 'both on' in t.image.detail


def test_a_held_station_stays_busy_without_timing_out():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.control({'cmd': 'inject', 'fault': 'hold', 'service': 'turner'})
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _run(cell, 4000)                                                   # 20 s >> every device timeout
    assert t.state == C.STATE_BUSY and t.counters.completed == 0
    cell.control({'cmd': 'inject', 'fault': 'clear'})
    _until(cell, 'turner', C.STATE_DONE)


def test_a_reset_from_busy_aborts_the_cycle_and_keeps_its_identity():
    cell, _ = _cell()
    t = cell.services['turner']
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    _run(cell, 40)
    cell.control({'cmd': 'reset', 'service': 'turner'})
    _until(cell, 'turner', C.STATE_READY)
    assert (t.image.cycle_seq, t.image.cycle_done) == (1, False) and t.counters.aborted == 1


def test_rule9_status_every_period_with_heartbeat_and_at_once_on_change():
    cell, _ = _cell()
    t = cell.services['turner']
    published = []
    for _ in range(200):                                               # 1 s: 10 periods
        for sid, img in cell.step():
            if sid == 'turner':
                published.append((cell.steps, img.heartbeat, img.state))
    # the start image (heartbeat 0) at the first scan, then one per period, each +1
    assert published[0][:2] == (2, 0)
    assert [p[1] for p in published if p[0] % 20 == 0] == list(range(1, 11))
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    step_of_request = cell.steps
    changes = []
    for _ in range(4):
        for sid, img in cell.step():
            if sid == 'turner' and img.state == C.STATE_BUSY:
                changes.append(cell.steps)
    assert changes and changes[0] - step_of_request <= 2                # within one scan, not at the next period


def test_one_service_per_device_and_unknown_keys_are_refused():
    data = _example()
    data['services']['indexer']['cycle'].append({'close': 'clamp'})
    with pytest.raises(ConfigError, match='already belongs to service'):
        parse(data)
    data = _example()
    data['services']['turner']['topic'] = '/turner'
    with pytest.raises(ConfigError, match='unknown key'):
        parse(data)
    data = _example()
    data['cell_io'] = 2
    with pytest.raises(ConfigError, match='cell_io: 1'):
        parse(data)


# --- the mock plant moves in WHOLE physics steps (implementation review fix C6: Python review MINOR 3) ---------------

def _bound_mock(model, devices):
    plant = MockPlantAdapter({'seed': 1, 'jitter': 0.0, 'devices': model})
    plant.bind(devices, [], ScanClock(parse_dt(physics_dt_ms=5), scan_every_steps=1, status_period_ms=100))
    return plant


def _steps_until(plant, predicate, limit=100000):
    for n in range(1, limit + 1):
        plant.advance(1)
        if predicate(plant.read()):
            return n
    raise AssertionError('never')


def _clamp_plant(close_steps, open_steps=None):
    from trainit_cell_io.core.devices import BinaryActuator
    model = {'clamp': {'close_ms': close_steps * 5, 'open_ms': (open_steps or close_steps) * 5}}
    return _bound_mock(model, {'clamp': BinaryActuator('clamp', 400, 2000)})


@pytest.mark.parametrize('stroke', [1, 6, 7, 10, 30, 60, 300, 400])
def test_an_actuator_stroke_of_n_steps_closes_and_opens_in_exactly_n_steps(stroke):
    # before the fix (s = min(1, s + 1/n), floats accumulated) 6 / 7 / 10 / 30 / 300 / 400 took one step more
    plant = _clamp_plant(stroke)
    plant.apply({'clamp.cmd': True})
    assert _steps_until(plant, lambda r: r['clamp.closed']) == stroke
    plant.apply({'clamp.cmd': False})
    assert _steps_until(plant, lambda r: r['clamp.open']) == stroke


@pytest.mark.parametrize('speed, unit, positions, steps', [
    (90.0, 'deg', {'a': 0.0, 'b': 180.0}, 400),      # 0.45 deg per step: 401 steps before the fix
    (180.0, 'deg', {'a': 0.0, 'b': 180.0}, 200),
    (60.0, 'deg', {'a': 0.0, 'b': 180.0}, 600),      # 0.3 deg per step
    (70.0, 'deg', {'a': 0.0, 'b': 180.0}, 515),      # 0.35 deg per step: 514.3 -> the 515th step arrives
    (200.0, 'mm', {'a': 0.0, 'b': 300.0}, 300),
    (200.0, 'mm', {'a': 300.0, 'b': 0.0}, 300),      # the other direction
])
def test_an_axis_move_takes_exactly_ceil_distance_over_speed_steps(speed, unit, positions, steps):
    from trainit_cell_io.core.devices import PositionAxis
    axis = PositionAxis('table', positions, 1.0, 400, 2000, unit)
    plant = _bound_mock({'table': {'speed': speed, 'start': 'a'}}, {'table': axis})
    plant.apply({'table.target': positions['b']})
    assert _steps_until(plant, lambda r: r['table.position'] == positions['b']) == steps


def test_a_motion_re_bases_exactly_on_a_reversal_a_slow_injection_and_a_stuck_pause():
    plant = _clamp_plant(60)
    plant.apply({'clamp.cmd': True})
    plant.advance(20)                                            # a third of the stroke
    plant.apply({'clamp.cmd': False})                            # reversal: a third of the open stroke = 20 steps
    assert _steps_until(plant, lambda r: r['clamp.open']) == 20
    plant.apply({'clamp.cmd': True})
    plant.advance(30)
    plant.inject({'fault': 'slow', 'device': 'clamp', 'factor': 2})   # the other half at 120 steps per stroke
    assert _steps_until(plant, lambda r: r['clamp.closed']) == 60
    plant.inject({'fault': 'clear'})
    plant.apply({'clamp.cmd': False})
    plant.advance(10)
    plant.inject({'fault': 'stuck_actuator', 'device': 'clamp'})
    plant.advance(500)                                           # stuck: the motion pauses, it is not lost
    assert not plant.read()['clamp.open'] and not plant.read()['clamp.closed']
    plant.inject({'fault': 'clear'})
    assert _steps_until(plant, lambda r: r['clamp.open']) == 50


def test_c4_whole_step_motions_stay_deterministic_with_jitter():
    first, second = _scenario_trace(7, jitter=0.5), _scenario_trace(7, jitter=0.5)
    assert first == second and _scenario_trace(8, jitter=0.5)[0] != first[0]


# --- bounded storage, readers by running total, a capped catch-up (implementation review fix C7: Python MINOR 4) -----

def test_a_bounded_log_keeps_the_last_entries_and_a_reader_gets_each_new_one_exactly_once():
    from trainit_cell_io.core.bounded_log import BoundedLog
    log = BoundedLog(5)
    index, seen = 0, []
    for n in range(23):
        log.append(n)
        if n % 3 == 2:                                   # a reader that keeps up (reads every 3 entries)
            lines, index, dropped = log.since(index)
            assert dropped == 0
            seen.extend(lines)
    lines, index, dropped = log.since(index)
    seen.extend(lines)
    assert seen == list(range(23)) and index == log.total == 23 and len(log) == 5 and list(log) == list(range(18, 23))
    assert log.since(index) == ([], 23, 0)                      # nothing new: nothing twice
    for n in range(23, 31):
        log.append(n)                                    # 8 new entries, 5 held: a late reader is told what it lost
    assert log.since(index) == ([26, 27, 28, 29, 30], 31, 3)
    with pytest.raises(ValueError):
        BoundedLog(0)


def test_the_cell_log_and_the_mock_trace_are_bounded():
    data = _example()
    cfg = parse(data)
    plant = MockPlantAdapter(cfg.mock, trace_lines=6)
    cell = Cell(cfg, plant, rng=random.Random(2), sessions={'turner': 0x1111, 'indexer': 0x2222}, log_lines=8)
    for n in range(1, 4):
        cell.set_request('turner', RequestImage(0x1111, 0xA, n))
        _run(cell, 4)                                    # accepted: BUSY (the previous DONE is not this cycle's)
        _until(cell, 'turner', C.STATE_DONE)
    assert cell.services['turner'].counters.completed == 3
    assert len(cell.log) == 8 < cell.log.total and len(plant.trace) == 6 < plant.trace.total
    assert list(cell.log)[-1] == cell.log.since(cell.log.total - 1)[0][0]       # the newest line is held


def test_the_catch_up_runs_at_most_its_cap_per_call_and_never_skips_a_step():
    from trainit_cell_io.core.clock import CatchUp
    catch = CatchUp(0.005, t0=100.0, cap=200)
    assert catch.due(99.0, 0) == (0, 0)                  # before the start: nothing
    assert catch.due(100.0101, 0) == (2, 2)              # in step: the wall's whole steps
    steps, calls = 0, 0
    while True:                                          # a 10 s stall: 2000 steps owed, at most 200 per call
        run, behind = catch.due(110.0, steps)
        if not run:
            break
        assert run <= 200 and behind == 2000 - steps
        steps, calls = steps + run, calls + 1
    assert (steps, calls) == (2000, 10)                  # every step run exactly once, in 10 bounded calls
    with pytest.raises(ValueError):
        CatchUp(0.005, 0.0, cap=0)


# --- the motion timeout is EXACT (implementation review fix C5: Python review MINOR 2) -------------------------------

class _StepExactPlant(PlantAdapter):
    """A plant whose actuator reaches the commanded limit EXACTLY `close_steps` physics steps after its command changed,
    and whose axis jumps onto its new target EXACTLY `move_steps` steps after the target changed (no in-band approach):
    the boundary of a motion timeout, to the physics step."""

    name = 'step_exact'

    def __init__(self, close_steps, move_steps):
        self.close_steps, self.move_steps = close_steps, move_steps
        self.step = 0
        self.cmd, self.cmd_step, self.closed = False, None, False
        self.target, self.target_step, self.position = None, None, 0.0

    def bind(self, devices, services, clock):
        self.services = list(services)

    def advance(self, steps):
        for _ in range(steps):
            self.step += 1
            if self.cmd_step is not None and self.step - self.cmd_step == self.close_steps:
                self.closed = self.cmd
            if self.target_step is not None and self.step - self.target_step == self.move_steps:
                self.position = self.target

    def read(self):
        image = {'clamp.open': not self.closed, 'clamp.closed': self.closed, 'table.position': self.position}
        for s in self.services:
            image.update({f'{s}.hold': False, f'{s}.fault': False, f'{s}.fault_detail': ''})
        return image

    def apply(self, outputs):
        if outputs.get('clamp.cmd') is not None and outputs['clamp.cmd'] != self.cmd:
            self.cmd, self.cmd_step = outputs['clamp.cmd'], self.step
        if outputs.get('table.target') is not None and outputs['table.target'] != self.target:
            self.target, self.target_step = outputs['table.target'], self.step


def _timeout_cell(close_steps=1, move_steps=1, timeout_ms=2000):
    data = {'cell_io': 1, 'control_mode': 'simulation', 'plant': 'mock',
            'scan': {'physics_dt_ms': 5, 'scan_every_steps': 2, 'status_period_ms': 100},
            'devices': {'clamp': {'type': 'binary_actuator', 'timeout_ms': timeout_ms},
                        'table': {'type': 'position_axis', 'unit': 'deg', 'positions': {'a': 0.0, 'b': 90.0},
                                  'band': 0.5, 'timeout_ms': timeout_ms}},
            'services': {'grip': {'cycle': [{'close': 'clamp'}], 'home': [{'open': 'clamp'}]},
                         'turn': {'cycle': [{'move': 'table', 'to': 'b'}], 'home': [{'move': 'table', 'to': 'a'}]}}}
    plant = _StepExactPlant(close_steps, move_steps)
    cell = Cell(parse(data), plant, rng=random.Random(1), sessions={'grip': 0x51, 'turn': 0x52})
    return cell, plant


def _command_to_verdict(service, close_steps=1, move_steps=1):
    """(physics steps from the device's command to the scan that decided, the final state, the fault code)."""
    cell, plant = _timeout_cell(close_steps, move_steps)
    pd = cell.services[service]
    cell.set_request(service, RequestImage(pd.session, 0xC5, 1))
    for _ in range(2000):
        cell.step()
        if pd.state in (C.STATE_DONE, C.STATE_FAULT):
            break
    commanded = plant.cmd_step if service == 'grip' else plant.target_step
    return cell.steps - commanded, pd.state, pd.image.fault_code


@pytest.mark.parametrize('service', ['grip', 'turn'])
def test_a_device_at_its_target_exactly_at_the_timeout_passes(service):
    # 2000 ms = 400 physics steps: the device arrives at the 400th step after its command
    elapsed, state, _ = _command_to_verdict(service, close_steps=400, move_steps=400)
    assert (elapsed, state) == (400, C.STATE_DONE)


@pytest.mark.parametrize('service', ['grip', 'turn'])
def test_a_device_not_at_its_target_at_the_timeout_faults_at_the_timeout_not_a_scan_later(service):
    # one scan late (402 steps): the fault fires at 400 steps = 2000 ms exactly (before the fix: DONE at 402, a pass)
    elapsed, state, code = _command_to_verdict(service, close_steps=402, move_steps=402)
    assert (elapsed, state, code) == (400, C.STATE_FAULT, C.FAULT_ACTUATOR_TIMEOUT)
    elapsed, state, code = _command_to_verdict(service, close_steps=10 ** 6, move_steps=10 ** 6)   # never arrives
    assert (elapsed, state, code) == (400, C.STATE_FAULT, C.FAULT_ACTUATOR_TIMEOUT)


def test_a_device_timeout_must_be_a_whole_number_of_scans():
    data = _example()
    data['devices']['clamp']['timeout_ms'] = 2005             # 401 physics steps of 5 ms; a scan is 2 steps
    with pytest.raises(ConfigError, match='not a whole number of scans'):
        parse(data)


# --- the Kit boundary (C5's pure half) -------------------------------------------------------------------------------

def test_the_kit_boundary_round_trips_and_refuses_malformed_images():
    cell, _ = _cell()
    img = cell.services['turner'].status()
    service, back, stamp = KB.decode_status(KB.encode_status('turner', img, 123))
    assert service == 'turner' and back == img and stamp == 123
    req = RequestImage(0x1111, 0xA, 7)
    service, rback, n = KB.decode_request(KB.encode_request('turner', req, 2))
    assert (service, rback, n) == ('turner', req, 2)
    for bad in ('{}', 'nope', '{"v": 2}', KB.encode_status('turner', img, 1).replace('"state": 1', '"state": 9')):
        with pytest.raises(KB.BoundaryError):
            KB.decode_status(bad)
    zero = img.copy()
    zero.source_session = 0
    with pytest.raises(KB.BoundaryError, match='source_session 0'):
        KB.decode_status(KB.encode_status('turner', zero, 1))


# --- the consume_delay injection (D12: a request that reaches the source late, e.g. after a station fault) --------

def _inject(cell, **kw):
    return cell.control(dict({'cmd': 'inject'}, **kw))


def test_consume_delay_holds_a_new_request_back_exactly_its_ms_from_the_scan_that_first_saw_it():
    cell, _ = _cell()
    t = cell.services['turner']
    _inject(cell, fault='consume_delay', service='turner', ms=300)
    _run(cell, 3)                                    # step 3: the request arrives between two scans
    cell.set_request('turner', RequestImage(t.session, 0xA, 1))
    busy_at = None
    for _ in range(200):
        cell.step()
        if busy_at is None and t.state == C.STATE_BUSY:
            busy_at = cell.steps
    first_seen = 4                                   # the next scan (scan_every_steps 2)
    assert busy_at == first_seen + 60                # 300 ms = 60 physics steps of 5 ms, exactly
    assert any('held back 300 ms (consume_delay injected)' in text for _, text in cell.log)
    assert t.counters.accepted == 1


def test_pb_p13_a_request_written_to_a_ready_source_that_faults_before_consuming_it_is_rejected_faulted():
    # D12: PB-P13 at M -- the robot writes its request to a READY source (it never writes to a FAULT one); the request
    # reaches the source only after a station fault; consumed as REJECTED FAULTED; after the RESET 0 cycles run.
    cell, _ = _cell()
    t = cell.services['turner']
    _inject(cell, fault='consume_delay', service='turner', ms=1000)
    req = RequestImage(t.session, 0xA, 1)
    cell.set_request('turner', req)
    _run(cell, 20)                                   # 100 ms: seen, held back, the source still READY
    assert t.state == C.STATE_READY and t.image.reject_code == C.REJECT_NONE
    _inject(cell, fault='station_fault', service='turner', detail='guard open')
    _run(cell, 2)
    assert t.state == C.STATE_FAULT and t.image.fault_code == C.FAULT_STATION_FAULT
    _run(cell, 200)                                  # the delay expires: the source consumes it
    assert (t.image.reject_requester, t.image.reject_seq, t.image.reject_code) == (0xA, 1, C.REJECT_FAULTED)
    assert t.counters.rejected_faulted == 1 and t.counters.accepted == 0
    _inject(cell, fault='clear')
    cell.control({'cmd': 'reset', 'service': 'turner'})
    _until(cell, 'turner', C.STATE_READY)
    for _ in range(50):                              # the image stays level-held (re-delivered)
        cell.set_request('turner', req)
        _run(cell, 20)
    assert t.counters.accepted == 0 and t.state == C.STATE_READY      # 0 cycles after the RESET


def test_consume_delay_never_delays_a_re_delivered_image_and_clear_releases_a_held_one():
    cell, _ = _cell()
    t = cell.services['turner']
    req = RequestImage(t.session, 0xA, 1)
    cell.set_request('turner', req)
    _until(cell, 'turner', C.STATE_DONE)
    _inject(cell, fault='consume_delay', service='turner', ms=2000)
    for _ in range(10):
        cell.set_request('turner', req)              # the consumed identity: a no-op, delayed or not
        _run(cell, 4)
    assert t.counters.accepted == 1 and not any('held back' in text for _, text in cell.log)
    cell.set_request('turner', RequestImage(t.session, 0xA, 2))
    _run(cell, 40)
    assert t.state == C.STATE_DONE                   # held back
    _inject(cell, fault='clear')
    _run(cell, 2)
    assert t.state == C.STATE_BUSY and t.image.cycle_seq == 2         # released at the next scan
    _inject(cell, fault='consume_delay', service='turner', ms=500)
    _inject(cell, fault='consume_delay', service='turner', ms=0)      # 0 removes it for that service
    assert cell.plant.consume_delay_steps('turner') == 0


@pytest.mark.parametrize('bad, why', [
    ({'service': 'turner', 'ms': 7}, 'not a whole number of physics steps'),
    ({'service': 'turner', 'ms': 15}, 'not a whole number of scans'),
    ({'service': 'turner', 'ms': -10}, 'integer 0..600000'),
    ({'service': 'turner', 'ms': 1.5}, 'integer 0..600000'),
    ({'service': 'turner', 'ms': True}, 'integer 0..600000'),
    ({'service': 'turner'}, 'integer 0..600000'),
    ({'service': 'nope', 'ms': 100}, 'no service'),
    ({'service': ['turner'], 'ms': 100}, 'no service'),
])
def test_consume_delay_refuses_a_value_it_cannot_honour_exactly(bad, why):
    cell, _ = _cell()
    with pytest.raises(ControlError, match=why):
        _inject(cell, fault='consume_delay', **bad)


def test_c4_consume_delay_keeps_the_trace_deterministic_and_is_described():
    def trace():
        data = _example()
        data['mock']['jitter'] = 0.2
        cell, plant = _cell(data)
        t = cell.services['turner']
        _inject(cell, fault='consume_delay', service='turner', ms=250)
        for n in range(1, 4):
            cell.set_request('turner', RequestImage(t.session, 0xB, n))
            _until(cell, 'turner', C.STATE_DONE)
            _run(cell, 7)
        return list(plant.trace), list(cell.log), plant.describe()['injections']['consume_delay_ms']
    first, second = trace(), trace()
    assert first == second and first[2] == {'turner': 250}


def test_an_unhashable_operator_value_is_refused_never_raised_as_a_crash():
    cell, _ = _cell()
    for command in ({'cmd': 'reset', 'service': ['turner']},
                    {'cmd': 'inject', 'fault': 'stuck_actuator', 'device': ['clamp']},
                    {'cmd': 'inject', 'fault': 'failed_sensor', 'device': {'x': 1}},
                    {'cmd': 'inject', 'fault': 'sensor_value', 'device': ['part_present'], 'value': True},
                    {'cmd': 'inject', 'fault': 'hold', 'service': {'turner': 1}}):
        with pytest.raises(ControlError):
            cell.control(command)


# --- logic: code -- a station whose sequence is code, handed to the Cell (e.g. a twin's own cycle class) -------------

class _CodeSequence:
    """A code `Sequence`: DONE after `scans` scans, or the Fault it is told to return."""

    def __init__(self, scans):
        self.scans, self.n, self.fault, self.starts = scans, 0, None, 0

    def start(self):
        self.n = 0
        self.starts += 1

    def scan(self, inputs, held, steps):
        from trainit_cell_io.core import DONE, RUNNING
        if self.fault is not None:
            return self.fault
        self.n += 1
        return DONE if self.n >= self.scans else RUNNING

    def describe(self):
        return f'a code sequence ({self.n} of {self.scans} scans)'


def _code_config():
    data = _example()
    data['services']['coded'] = {'description': 'a station whose cycle is code', 'logic': 'code'}
    return data


def test_logic_code_a_service_whose_sequences_are_code_follows_the_same_rules():
    from trainit_cell_io.core import Fault
    made = []

    def factory():
        made.append((_CodeSequence(5), _CodeSequence(3)))
        return made[-1]
    cfg = parse(_code_config())
    assert cfg.services['coded'].logic == 'code' and cfg.services['coded'].cycle == []
    cell = Cell(cfg, MockPlantAdapter(cfg.mock), rng=random.Random(3), sequences={'coded': factory})
    f = cell.services['coded']
    cell.set_request('coded', RequestImage(f.session, R_SESSION, 1))
    _run(cell, 2)
    assert f.state == C.STATE_BUSY and (f.image.cycle_requester, f.image.cycle_seq) == (R_SESSION, 1)
    _until(cell, 'coded', C.STATE_DONE)
    assert f.image.cycle_done and f.counters.accepted == f.counters.completed == 1
    cycle, home = made[-1]
    assert cycle.starts == 1 and home.starts == 0
    # the code cycle faults: FAULT with its code, held until a RESET, which runs the code home (HOMING -> READY)
    cycle.fault = Fault(C.FAULT_ACTUATOR_TIMEOUT, 'the axis did not arrive within its timeout')
    cell.set_request('coded', RequestImage(f.session, R_SESSION, 2))
    _run(cell, 4)
    assert f.state == C.STATE_FAULT and f.image.fault_code == C.FAULT_ACTUATOR_TIMEOUT
    _run(cell, 400)
    assert f.state == C.STATE_FAULT                                   # never cleared by itself
    cycle.fault = None
    cell.control({'cmd': 'reset', 'service': 'coded'})
    _run(cell, 2)
    assert f.state == C.STATE_HOMING and home.starts == 1
    _until(cell, 'coded', C.STATE_READY)
    # a controller restart: fresh sequences from the factory, a new session; the other services unchanged in kind
    old = f.session
    cell.control({'cmd': 'restart'})
    assert len(made) == 2 and cell.services['coded'].session != old
    assert cell.services['turner'].counters.accepted == 0


def test_logic_code_is_refused_without_its_sequences_with_a_chain_listed_or_for_another_service():
    cfg = parse(_code_config())
    with pytest.raises(ConfigError, match='logic: code'):
        Cell(cfg, MockPlantAdapter(cfg.mock))               # what cell_io_node and the neutral runner do: exit 2
    with pytest.raises(ConfigError, match='not a `logic: code` service'):
        Cell(cfg, MockPlantAdapter(cfg.mock),
             sequences={'coded': lambda: (_CodeSequence(1), _CodeSequence(1)),
                        'turner': lambda: (_CodeSequence(1), _CodeSequence(1))})
    data = _code_config()
    data['services']['coded']['cycle'] = [{'close': 'clamp'}]
    with pytest.raises(ConfigError, match='lists no cycle'):
        parse(data)


def _code_cell(cycle_scans=5, home_scans=3, made=None):
    made = [] if made is None else made

    def factory():
        made.append((_CodeSequence(cycle_scans), _CodeSequence(home_scans)))
        return made[-1]
    cfg = parse(_code_config())
    return Cell(cfg, MockPlantAdapter(cfg.mock), rng=random.Random(5), sequences={'coded': factory}), made


def test_logic_code_only_the_contracts_fault_codes_reach_the_wire():
    """Delta review R1: a code sequence's Fault with a code outside ACTUATOR_TIMEOUT / SENSOR_FAULT / STATION_FAULT is
    reported as STATION_FAULT (its own value in the detail) -- the bridge's boundary accepts the image (a code it
    refuses would silence the station: the robot's 310 instead of 306)."""
    from trainit_cell_io.core import Fault
    for code in (7, 0, 70000):
        cell, made = _code_cell()
        f = cell.services['coded']
        made[-1][0].fault = Fault(code, 'a station-specific fault')
        cell.set_request('coded', RequestImage(f.session, R_SESSION, 1))
        _run(cell, 4)
        assert f.state == C.STATE_FAULT and f.image.fault_code == C.FAULT_STATION_FAULT, code
        assert f'station fault code {code!r} reported as STATION_FAULT' in f.image.detail
        service, image, _stamp = KB.decode_status(KB.encode_status('coded', f.status(), 1))
        assert service == 'coded' and image.fault_code == C.FAULT_STATION_FAULT
    cell, made = _code_cell()
    f = cell.services['coded']
    made[-1][0].fault = Fault(C.FAULT_ACTUATOR_TIMEOUT, 'the axis did not arrive within its timeout')
    cell.set_request('coded', RequestImage(f.session, R_SESSION, 1))
    _run(cell, 4)
    assert f.image.fault_code == C.FAULT_ACTUATOR_TIMEOUT                           # a contract code: unchanged


def test_logic_code_a_sequence_that_raises_faults_its_station_never_the_cell():
    cell, made = _code_cell()
    f, t = cell.services['coded'], cell.services['turner']

    def boom(*_args):
        raise RuntimeError('a bug in the station code')
    made[-1][0].scan = boom
    cell.set_request('coded', RequestImage(f.session, R_SESSION, 1))
    cell.set_request('turner', RequestImage(t.session, R_SESSION + 1, 1))
    _run(cell, 4)                                            # never raises out of the cell
    assert f.state == C.STATE_FAULT and f.image.fault_code == C.FAULT_STATION_FAULT and f.counters.aborted == 1
    assert 'the station sequence raised RuntimeError: a bug in the station code' in f.image.detail
    assert t.state == C.STATE_BUSY                           # the other station runs on
    cell2, made2 = _code_cell()
    g = cell2.services['coded']
    made2[-1][0].start = boom                                # start() raising on the accept
    cell2.set_request('coded', RequestImage(g.session, R_SESSION, 1))
    _run(cell2, 2)
    assert g.state == C.STATE_FAULT and g.counters.accepted == 1 and g.counters.aborted == 1
    made2[-1][1].start = boom                                # a RESET whose home start() raises: FAULT again
    cell2.control({'cmd': 'reset', 'service': 'coded'})
    assert g.state == C.STATE_FAULT and 'raised RuntimeError' in g.image.detail


def test_logic_code_a_home_sequence_fault_and_a_restart_runs_the_new_sequences():
    from trainit_cell_io.core import Fault
    made = []
    cell, _ = _code_cell(made=made)
    f = cell.services['coded']
    cell.set_request('coded', RequestImage(f.session, R_SESSION, 1))
    _until(cell, 'coded', C.STATE_DONE)
    made[-1][1].fault = Fault(C.FAULT_ACTUATOR_TIMEOUT, 'the home stroke did not arrive')
    cell.control({'cmd': 'reset', 'service': 'coded'})
    _run(cell, 2)
    assert f.state == C.STATE_FAULT and f.image.fault_code == C.FAULT_ACTUATOR_TIMEOUT
    old_cycle = made[-1][0]
    cell.control({'cmd': 'restart'})
    g = cell.services['coded']
    assert len(made) == 2 and g is not f and g.state == C.STATE_READY
    cell.set_request('coded', RequestImage(g.session, R_SESSION, 1))
    _until(cell, 'coded', C.STATE_DONE)
    assert made[-1][0].starts == 1 and old_cycle.starts == 1      # the NEW cycle ran, never the old one again


def test_logic_code_a_restart_whose_factory_fails_changes_nothing():
    calls = []

    def factory():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError('the station code cannot start')
        return _CodeSequence(3), _CodeSequence(3)
    cfg = parse(_code_config())
    cell = Cell(cfg, MockPlantAdapter(cfg.mock), rng=random.Random(5), sequences={'coded': factory})
    before = {sid: pd.session for sid, pd in cell.services.items()}
    desk = OperatorDesk(cell)
    answer, _ = desk.handle('{"cmd": "restart", "id": "r1"}')
    assert not answer['ok'] and 'the restart was not done (nothing changed)' in answer['text'], answer
    assert 'RuntimeError: the station code cannot start' in answer['text']
    assert {sid: pd.session for sid, pd in cell.services.items()} == before    # no service restarted


def test_logic_code_a_factory_that_raises_or_returns_no_pair_is_refused_at_construction():
    cfg = parse(_code_config())
    with pytest.raises(ConfigError, match="factory raised RuntimeError"):
        Cell(cfg, MockPlantAdapter(cfg.mock), sequences={'coded': lambda: (_ for _ in ()).throw(RuntimeError('x'))})
    one = _CodeSequence(1)
    for bad in (lambda: one, lambda: (one, one), lambda: (one, object()), lambda: [one, _CodeSequence(1)]):
        with pytest.raises(ConfigError, match=r'must return \(cycle, home\)'):
            Cell(cfg, MockPlantAdapter(cfg.mock), sequences={'coded': bad})


def test_rule2_an_older_image_never_displaces_a_newer_held_one():
    """Delta review R5: R's 5 consumed; R's 6 arrives and is held back (consume_delay); a replayed OLDER image (4)
    arrives meanwhile: it is counted and never held, so 6 is still consumed -- before the fix it replaced 6 and 6 never
    ran."""
    cell, _ = _cell()
    t = cell.services['turner']
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 5))
    _until(cell, 'turner', C.STATE_DONE)
    _inject(cell, fault='consume_delay', service='turner', ms=300)
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 6))
    _run(cell, 2)
    before = cell.ignored_redelivered['turner']
    cell.set_request('turner', RequestImage(t.session, R_SESSION, 4))
    assert cell.ignored_redelivered['turner'] == before + 1
    _run(cell, 200)
    assert t.counters.accepted == 2 and t.image.cycle_seq == 6
