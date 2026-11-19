"""The mock cell's ROS side at the process level (P0.b WP2): `cell_io_node` and `cell_io_cli` run from this source tree
against a scripted rclpy requester, on a private ROS domain (57; ros_harness.py).

Exercised (P0_CHECKPOINT.md, P0.b frozen matrix; P0B_IMPLEMENTATION_PLAN.md D11, D12):
- the canonical topics and their QoS in the robot's slot (§3.2): RELIABLE + TRANSIENT_LOCAL, the contract types;
- the status every period with the heartbeat +1, and at once on a change (§3.3 rule 9);
- a request -> BUSY -> DONE, the counters on /cell_io/state; PB-P4 (a re-delivered image runs one cycle);
- PB-P2 (b) as D12 runs it at M: a single scripted requester against the real cell_io_node (REJECTED BUSY, the
  running cycle continues, the source's cycle counter = the accepted requests);
- PB-P11's source half: two publishers on <service>/request -> every new request REJECTED TWO_REQUESTERS;
- PB-P13's mechanics as D12 runs it at M: the `consume_delay` injection through the CLI, a station fault before the
  request is consumed -> REJECTED FAULTED; after the RESET (HOMING -> READY) 0 cycles run;
- PB-P3's source half with D11's orderly restart (SIGINT): a new session that never executes the old image;
- C3 at the process level (control_mode external_plc), the refusal of `plant: isaac`, the CLI's exit codes.
"""

import json
import signal
import statistics
import time

import pytest

import ros_harness as H

pytestmark = pytest.mark.skipif(H.ros_missing() is not None, reason=H.ros_missing() or '')

try:
    from trainit_interfaces.msg import ChannelStatus as CS
except ImportError:          # the module is skipped
    CS = None

REQ = 0x5C21A7ED             # the scripted requester's session


@pytest.fixture(scope='module')
def ctx():
    context = H.init_context()
    yield context
    context.try_shutdown()


@pytest.fixture
def procs():
    started = []
    yield started
    for proc in started:
        proc.stop()


@pytest.fixture
def requester(ctx):
    req = H.Requester(ctx, ['turner', 'indexer'])
    yield req
    req.close()


def _start_cell(tmp_path, procs, mutate=None, name='cell', fast=True):
    cfg = H.write_config(tmp_path / f'{name}.yaml', mutate, fast)
    proc = H.cell_node(cfg, tmp_path / f'{name}.log')
    procs.append(proc)
    assert proc.wait_for(r'cell_io: serving 2 machine service\(s\)', 30), proc.text()
    return proc


def _ready(requester, service='turner', timeout=15.0):
    hit = requester.wait_status(service, lambda m: m.state in (CS.STATE_READY, CS.STATE_DONE), timeout)
    assert hit, f'no READY status of {service}'
    assert requester.wait_matched(service)
    requester.spin_for(0.3)
    return requester.latest(service).source_session


def _is(state, cycle=None, done=None):
    def check(msg):
        return (msg.state == state and (cycle is None or (msg.cycle_requester, msg.cycle_seq) == cycle)
                and (done is None or msg.cycle_done == done))
    return check


# --- the canonical topics, the QoS, the cadence ----------------------------------------------------------------------

def test_the_canonical_topics_types_and_qos(tmp_path, procs, requester):
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    _start_cell(tmp_path, procs)
    node = requester.node

    def mine(infos):            # the cell's endpoints (a stale endpoint of an earlier test may linger in the graph)
        return [i for i in infos if i.node_name == 'cell_io']
    for s in ('turner', 'indexer'):
        assert requester.spin_until(lambda: len(mine(node.get_publishers_info_by_topic(f'/{s}/status'))) == 1, 10)
        (pub,) = mine(node.get_publishers_info_by_topic(f'/{s}/status'))
        assert pub.topic_type == 'trainit_interfaces/msg/ChannelStatus'
        assert pub.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
        assert requester.spin_until(lambda: mine(node.get_subscriptions_info_by_topic(f'/{s}/request')), 10)
        (sub,) = mine(node.get_subscriptions_info_by_topic(f'/{s}/request'))
        assert sub.topic_type == 'trainit_interfaces/msg/ChannelRequest'
        assert sub.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert sub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    assert requester.spin_until(lambda: mine(node.get_publishers_info_by_topic('/cell_io/state'))
                                and mine(node.get_subscriptions_info_by_topic('/cell_io/control')), 10)
    (state,) = mine(node.get_publishers_info_by_topic('/cell_io/state'))
    assert state.topic_type == 'std_msgs/msg/String'
    assert state.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    (control,) = mine(node.get_subscriptions_info_by_topic('/cell_io/control'))
    assert control.qos_profile.reliability == ReliabilityPolicy.RELIABLE
    assert control.qos_profile.durability == DurabilityPolicy.VOLATILE


def test_the_services_live_in_the_robot_slot(tmp_path, procs, ctx):
    def slot(data):
        data['robot_slot'] = 'r1'
    _start_cell(tmp_path, procs, slot)
    req = H.Requester(ctx, ['turner'], slot='r1', name='slot_requester')
    try:
        session = _ready(req)
        assert req.node.get_publishers_info_by_topic('/turner/status') == []
        req.request('turner', session, REQ, 1)
        assert req.wait_status('turner', _is(CS.STATE_DONE, (REQ, 1), True), 15)
    finally:
        req.close()


def test_status_every_period_with_the_heartbeat_and_at_once_on_a_change(tmp_path, procs, requester):
    _start_cell(tmp_path, procs)
    session = _ready(requester)
    requester.spin_for(1.6)
    tail = requester.status['turner'][-14:]
    beats = [m.heartbeat for _, m in tail]
    assert all((b - a) % 65536 == 1 for a, b in zip(beats, beats[1:])), beats   # +1 every period, never minted twice
    gaps = [b - a for (a, _), (b, _) in zip(tail, tail[1:])]
    period = statistics.median(gaps)
    H.timing('status period, median inter-arrival over 13 periods', period, f'min {min(gaps)*1000:.1f} ms, '
             f'max {max(gaps)*1000:.1f} ms')
    assert 0.08 <= period <= 0.12, gaps
    t0 = requester.request('turner', session, REQ, 1)
    hit = requester.wait_status('turner', _is(CS.STATE_BUSY, (REQ, 1), False), 5, after=t0)
    assert hit, 'no BUSY image'
    H.timing('request published -> BUSY image received (at once on a change)', hit[0] - t0)
    assert hit[0] - t0 < 0.06                        # at once: within a scan (10 ms) + delivery, not the next period


def test_a_stalled_cell_catches_up_in_bounded_steps_and_its_clock_stays_exact(tmp_path, procs, requester):
    # review fix C7 (Python review MINOR 4): the node's catch-up runs at most 200 physics steps per scan timer callback;
    # a 2.5 s stall (SIGSTOP) is caught up over several callbacks, no step skipped: the cell's time keeps its offset to
    # the wall clock (the same CLOCK_MONOTONIC in both processes)
    import os
    from std_msgs.msg import String
    cell = _start_cell(tmp_path, procs)
    _ready(requester)
    states = []                                          # (arrival, the cell's simulated time) of every state image

    def on_state(msg):
        states.append((time.monotonic(), json.loads(msg.data)['time_s']))
    requester.node.create_subscription(String, '/cell_io/state', on_state, H.reader_qos())
    requester.spin_for(2.5)
    before = sorted(t - s for t, s in states if t > time.monotonic() - 2.0)
    assert len(before) >= 2, states
    os.kill(cell.p.pid, signal.SIGSTOP)
    time.sleep(2.5)
    t_resume = time.monotonic()
    os.kill(cell.p.pid, signal.SIGCONT)
    assert cell.wait_for(r'behind wall time \(machine overloaded\?\) -- catching up, at most 200 physics steps per '
                         r'scan timer \(no step is skipped\)', 10), cell.text()[-2000:]
    requester.spin_for(3.0)
    after = sorted(t - s for t, s in states if t > t_resume + 1.0)
    assert len(after) >= 2, states
    drift = after[len(after) // 2] - before[len(before) // 2]
    H.timing('a 2.5 s stall caught up: offset change of the cell clock to the wall clock', drift)
    assert abs(drift) < 0.1, (before, after)
    assert 'Traceback' not in cell.text()


# --- a request, the counters, a re-delivered image ---------------------------------------------------------------

def test_a_request_runs_one_cycle_busy_then_done_and_the_counters_count_it(tmp_path, procs, requester):
    _start_cell(tmp_path, procs)
    session = _ready(requester)
    t0 = requester.request('turner', session, REQ, 1)
    busy = requester.wait_status('turner', _is(CS.STATE_BUSY, (REQ, 1), False), 5, after=t0)
    done = requester.wait_status('turner', _is(CS.STATE_DONE, (REQ, 1), True), 15, after=t0)
    assert busy and done
    H.timing('turner request -> BUSY', busy[0] - t0)
    H.timing('turner request -> DONE (the fast mock test model)', done[0] - t0)
    assert requester.wait_state(lambda s: requester.counters('turner').get('accepted') == 1
                                and requester.counters('turner').get('completed') == 1, 5), requester.state
    for _ in range(20):                              # PB-P4: the level-held image re-delivered many times
        requester.request('turner', session, REQ, 1)
        requester.spin_for(0.03)
    requester.spin_for(1.2)
    assert requester.latest('turner').state == CS.STATE_DONE
    assert requester.counters('turner')['accepted'] == 1
    t1 = requester.request('turner', session, REQ, 2)
    assert requester.wait_status('turner', _is(CS.STATE_DONE, (REQ, 2), True), 15, after=t1)
    assert requester.wait_state(lambda s: requester.counters('turner').get('accepted') == 2
                                and requester.counters('turner').get('completed') == 2, 5)
    session_i = _ready(requester, 'indexer')
    t2 = requester.request('indexer', session_i, REQ, 1)
    done_i = requester.wait_status('indexer', _is(CS.STATE_DONE, (REQ, 1), True), 15, after=t2)
    assert done_i
    H.timing('indexer request -> DONE (the fast mock test model)', done_i[0] - t2)


def test_pb_p2b_a_request_reaching_a_busy_source_is_rejected_and_the_running_cycle_continues(tmp_path, procs,
                                                                                             requester):
    # D12: PB-P2 (b) at M -- one scripted requester against the real cell_io_node (no robot server running)
    def slow_table(data):
        data['mock']['devices']['table']['speed'] = 90.0           # 2 s per half turn: a long BUSY
        data['devices']['table']['timeout_ms'] = 4000
    _start_cell(tmp_path, procs, slow_table)
    session = _ready(requester)
    t0 = requester.request('turner', session, REQ, 1)
    assert requester.wait_status('turner', _is(CS.STATE_BUSY, (REQ, 1)), 5, after=t0)
    requester.spin_for(0.5)
    t1 = requester.request('turner', session, REQ, 2)
    rejected = requester.wait_status('turner', lambda m: (m.reject_requester, m.reject_seq, m.reject_code)
                                     == (REQ, 2, CS.REJECT_BUSY), 5, after=t1)
    assert rejected, 'the second request was not rejected BUSY'
    H.timing('request to a BUSY source -> REJECTED BUSY image', rejected[0] - t1)
    assert rejected[1].state == CS.STATE_BUSY and (rejected[1].cycle_requester, rejected[1].cycle_seq) == (REQ, 1)
    done = requester.wait_status('turner', _is(CS.STATE_DONE, (REQ, 1), True), 15, after=t1)
    assert done, 'the running cycle did not continue to its DONE'
    requester.spin_for(1.2)
    assert requester.latest('turner').state == CS.STATE_DONE
    c = requester.counters('turner')
    assert (c['accepted'], c['completed'], c['rejected_busy']) == (1, 1, 1), c   # cycles = the accepted requests


# --- two requesters ------------------------------------------------------------------------------------------------

def test_two_request_publishers_every_new_request_is_rejected_two_requesters(tmp_path, procs, requester, ctx):
    cell = _start_cell(tmp_path, procs)
    session = _ready(requester)
    other = H.Requester(ctx, ['turner'], name='second_requester')
    try:
        node = requester.node
        assert requester.spin_until(lambda: node.count_publishers('/turner/request') == 2, 10)
        requester.spin_for(0.6)                     # the cell counts its publishers every period
        t0 = requester.request('turner', session, REQ, 1)
        hit = requester.wait_status('turner', lambda m: (m.reject_requester, m.reject_seq, m.reject_code)
                                    == (REQ, 1, CS.REJECT_TWO_REQUESTERS), 5, after=t0)
        assert hit, cell.text()
        requester.spin_for(0.6)
        assert requester.latest('turner').state == CS.STATE_READY and requester.counters('turner')['accepted'] == 0
        assert 'every new request is REJECTED TWO_REQUESTERS' in cell.text()
    finally:
        other.close()
    assert requester.spin_until(lambda: requester.node.count_publishers('/turner/request') == 1, 10)
    requester.spin_for(0.6)
    t1 = requester.request('turner', session, REQ, 2)
    assert requester.wait_status('turner', _is(CS.STATE_DONE, (REQ, 2), True), 15, after=t1)   # one requester again


OTHER = 0x0E1BA5ED           # a second requester's session


def _second_requester(ctx, requester):
    """A second publisher on /turner/request, matched with the cell and counted by it (rule 6)."""
    other = H.Requester(ctx, ['turner'], name='second_requester')
    assert requester.spin_until(lambda: requester.node.count_publishers('/turner/request') == 2, 10)
    assert other.wait_matched('turner')
    requester.spin_for(0.6)                              # the cell counts its publishers every period
    return other


def test_c2_two_requesters_images_back_to_back_are_both_consumed_and_named(tmp_path, procs, requester, ctx):
    # review fix C2 at the process level: before it the cell kept ONE latest image per service and its request reader
    # was depth 1, so the first of two images arriving together was displaced: never consumed, never named
    cell = _start_cell(tmp_path, procs)
    session = _ready(requester)
    other = _second_requester(ctx, requester)
    try:
        requester.request('turner', session, REQ, 1)
        other.request('turner', session, OTHER, 1)
        assert requester.wait_state(lambda s: requester.counters('turner').get('rejected_two_requesters') == 2, 5), \
            (requester.state, cell.text()[-2000:])
        svc = requester.state['services']['turner']
        assert svc['counters']['accepted'] == 0 and svc['request_images'] == 2, svc
        text = cell.text()
        for rs in (REQ, OTHER):
            assert f'request 1 of requester {rs:#010x} rejected TWO_REQUESTERS' in text, text[-2000:]
        named = {(m.reject_requester, m.reject_code) for _, m in requester.status['turner']}
        assert (OTHER, CS.REJECT_TWO_REQUESTERS) in named or (REQ, CS.REJECT_TWO_REQUESTERS) in named
    finally:
        other.close()


def test_c1_a_re_delivered_image_after_a_two_requester_episode_is_a_no_op(tmp_path, procs, requester, ctx):
    # review fix C1 at the process level, the protocol review's probe: R (REQ,1) accepted and DONE; a second requester
    # X (OTHER,1) rejected TWO_REQUESTERS; X leaves; R's image re-delivered -> before the fix a SECOND cycle
    cell = _start_cell(tmp_path, procs)
    session = _ready(requester)
    t0 = requester.request('turner', session, REQ, 1)
    assert requester.wait_status('turner', _is(CS.STATE_DONE, (REQ, 1), True), 15, after=t0)
    other = _second_requester(ctx, requester)
    try:
        t1 = other.request('turner', session, OTHER, 1)
        assert requester.wait_status('turner', lambda m: (m.reject_requester, m.reject_seq, m.reject_code)
                                     == (OTHER, 1, CS.REJECT_TWO_REQUESTERS), 5, after=t1), cell.text()[-2000:]
    finally:
        other.close()
    assert requester.spin_until(lambda: requester.node.count_publishers('/turner/request') == 1, 10)
    requester.spin_for(0.6)
    t2 = time.monotonic()
    for _ in range(5):                                   # R's latched image re-delivered (a re-match, a bridge restart)
        requester.request('turner', session, REQ, 1)
        requester.spin_for(0.05)
    requester.spin_for(1.2)
    assert requester.wait_state(lambda s: s['services']['turner']['state'] == 'DONE', 5), requester.state
    c = requester.counters('turner')
    assert (c['accepted'], c['completed'], c['rejected_two_requesters']) == (1, 1, 1), c   # one cycle, ever
    assert not requester.first_after('turner', t2, lambda m: m.state == CS.STATE_BUSY)
    assert requester.latest('turner').state == CS.STATE_DONE
    assert requester.wait_state(lambda s: s['services']['turner'].get('ignored_redelivered', 0) >= 5, 5), \
        requester.state                                  # each re-delivery reached the source and was a no-op


# --- the operator CLI: injections, RESET, the state ---------------------------------------------------------------

def test_the_cli_injects_a_stuck_actuator_resets_and_reports(tmp_path, procs, requester):
    _start_cell(tmp_path, procs)
    session = _ready(requester)
    out = H.cli('inject', 'stuck_actuator', 'device=clamp')
    assert out.returncode == 0 and out.stdout.startswith('OK: clamp stuck'), out
    t0 = requester.request('turner', session, REQ, 1)
    fault = requester.wait_status('turner', lambda m: m.state == CS.STATE_FAULT, 10, after=t0)
    assert fault and fault[1].fault_code == CS.FAULT_ACTUATOR_TIMEOUT and not fault[1].cycle_done
    H.timing('stuck clamp: request -> FAULT ACTUATOR_TIMEOUT (clamp timeout 400 ms)', fault[0] - t0)
    state = H.cli('state')
    assert state.returncode == 0 and 'turner: FAULT -- FAULT ACTUATOR_TIMEOUT' in state.stdout, state
    assert 'injections: {' in state.stdout and '"stuck": ["clamp"]' in state.stdout
    raw = H.cli('state', '--json')
    data = json.loads(raw.stdout)
    assert data['services']['turner']['counters']['aborted'] == 1 and data['services']['turner']['state'] == 'FAULT'
    assert H.cli('inject', 'clear').returncode == 0
    t1 = time.monotonic()
    reset = H.cli('reset', 'turner')
    assert reset.returncode == 0 and 'RESET' in reset.stdout, reset
    assert requester.wait_status('turner', lambda m: m.state == CS.STATE_HOMING, 5, after=t1)
    ready = requester.wait_status('turner', lambda m: m.state == CS.STATE_READY, 10, after=t1)
    assert ready and ready[1].fault_code == CS.FAULT_NONE
    assert (ready[1].cycle_seq, ready[1].cycle_done) == (1, False)        # the aborted cycle stays visible
    refused = H.cli('inject', 'melt')
    assert refused.returncode == 2 and refused.stdout.startswith('REFUSED: unknown fault'), refused
    assert H.cli('reset', 'nope').returncode == 2
    assert H.cli('inject', 'stuck_actuator', 'device').returncode == 2     # not key=value


def test_pb_p13_mechanics_consume_delay_then_a_station_fault_rejects_faulted_and_nothing_runs_after_reset(
        tmp_path, procs, requester):
    # D12: the request is WRITTEN to a READY source (the robot never writes to a FAULT one) and reaches it only after
    # a station fault; it is consumed REJECTED FAULTED; after the RESET 0 cycles run.
    _start_cell(tmp_path, procs)
    session = _ready(requester)
    assert H.cli('inject', 'consume_delay', 'service=turner', 'ms=2000').returncode == 0
    t0 = requester.request('turner', session, REQ, 7)
    requester.spin_for(0.3)
    assert requester.latest('turner').state == CS.STATE_READY             # held back, not consumed
    assert H.cli('inject', 'station_fault', 'service=turner', 'detail=guard_open').returncode == 0
    assert requester.wait_status('turner', lambda m: m.state == CS.STATE_FAULT
                                 and m.fault_code == CS.FAULT_STATION_FAULT, 5, after=t0)
    rejected = requester.wait_status('turner', lambda m: (m.reject_requester, m.reject_seq, m.reject_code)
                                     == (REQ, 7, CS.REJECT_FAULTED), 6, after=t0)
    assert rejected, 'the late request was not consumed REJECTED FAULTED'
    H.timing('consume_delay 2000 ms: request -> REJECTED FAULTED', rejected[0] - t0)
    assert rejected[0] - t0 >= 1.95
    assert H.cli('inject', 'clear').returncode == 0
    t1 = time.monotonic()
    assert H.cli('reset', 'turner').returncode == 0
    assert requester.wait_status('turner', lambda m: m.state == CS.STATE_READY, 10, after=t1)
    for _ in range(10):                                                    # the image stays level-held
        requester.request('turner', session, REQ, 7)
        requester.spin_for(0.1)
    assert requester.wait_state(lambda s: s['services']['turner']['state'] == 'READY', 5)
    c = requester.counters('turner')
    assert c['accepted'] == 0 and c['rejected_faulted'] == 1, c           # 0 cycles ran


# --- an orderly restart (D11): a new session that never executes the old image -----------------------------------

@pytest.mark.parametrize('stop_signal', [signal.SIGINT, signal.SIGTERM])
def test_an_orderly_restart_is_a_new_session_that_never_executes_the_old_image(
        tmp_path, procs, requester, stop_signal):
    first = _start_cell(tmp_path, procs, name='first')
    old = _ready(requester)
    assert H.cli('inject', 'hold', 'service=turner').returncode == 0
    t0 = requester.request('turner', old, REQ, 1)
    assert requester.wait_status('turner', _is(CS.STATE_BUSY, (REQ, 1)), 5, after=t0)
    assert first.stop(stop_signal) == 'clean' and first.p.returncode == 0, first.text()
    assert 'Traceback' not in first.text()
    t1 = time.monotonic()
    _start_cell(tmp_path, procs, name='second')
    hit = requester.wait_status('turner', lambda m: m.source_session != old, 15, after=t1)
    assert hit, 'no status of the restarted source'
    new = hit[1].source_session
    assert new != 0 and hit[1].state == CS.STATE_READY and hit[1].cycle_seq == 0
    requester.spin_for(1.0)                          # the old image (latched by the requester) reaches the new source
    assert requester.wait_state(lambda s: requester.counters('turner').get('ignored_other_session', 0) >= 1, 5)
    assert requester.latest('turner').state == CS.STATE_READY and requester.counters('turner')['accepted'] == 0
    t2 = requester.request('turner', new, REQ, 2)
    assert requester.wait_status('turner', _is(CS.STATE_DONE, (REQ, 2), True), 15, after=t2)


# --- the process exit codes (C3 at the process level; plant isaac refused) ------------------------------------------

def _node_exit(tmp_path, data_or_text, name):
    path = tmp_path / f'{name}.yaml'
    if isinstance(data_or_text, str):
        path.write_text(data_or_text, encoding='utf-8')
    else:
        import yaml
        path.write_text(yaml.safe_dump(data_or_text), encoding='utf-8')
    proc = H.cell_node(path, tmp_path / f'{name}.log')
    rc = proc.wait_exit(60)
    proc.stop()
    return rc, proc.text()


def test_c3_external_plc_with_a_process_device_exits_non_zero(tmp_path):
    import yaml
    data = yaml.safe_load(H.MOCK_EXAMPLE.read_text())
    data['control_mode'] = 'external_plc'
    del data['plant']
    rc, text = _node_exit(tmp_path, data, 'plc_with_service')
    assert rc == 2 and 'external_plc refuses a simulated machine service' in text, text
    rc, text = _node_exit(tmp_path, {'cell_io': 1, 'control_mode': 'external_plc',
                                     'services': {'turner': {'logic': 'python_logic'}}}, 'plc_with_logic')
    assert rc == 2 and 'external_plc' in text, text
    rc, text = _node_exit(tmp_path, {'cell_io': 1, 'control_mode': 'external_plc'}, 'plc_alone')
    assert rc == 0 and 'simulates nothing' in text, text


def test_cell_io_node_refuses_the_isaac_plant_and_a_bad_file(tmp_path):
    text = H.MOCK_EXAMPLE.read_text().replace('\nplant: mock\n', '\nplant: isaac\n')
    rc, out = _node_exit(tmp_path, text, 'isaac')
    assert rc == 2 and "plant 'isaac' is not served by this node" in out and 'cell_io_bridge' in out, out
    rc, out = _node_exit(tmp_path, 'cell_io: [1\n', 'not_yaml')
    assert rc == 2 and 'not valid YAML' in out, out
    proc = H.Proc(H.python_module('trainit_cell_io.ros.cell_io_node'), tmp_path / 'no_param.log')
    assert proc.wait_exit(60) == 2 and 'config:=<cell_io.yaml> is required' in proc.text()
    proc.stop()
    missing = H.cell_node(tmp_path / 'absent.yaml', tmp_path / 'absent.log')
    assert missing.wait_exit(60) == 2 and 'cannot read' in missing.text()
    missing.stop()


@pytest.mark.parametrize('old, new, why', [
    ('- {close: clamp}', '- {close: [clamp]}', "unknown device ['clamp']"),               # was a TypeError traceback
    ('positions: {load: 0.0, turned: 180.0}', 'positions: {load: 0.0, turned: .nan}', 'must be a finite number'),
    ('    band: 1.0\n    timeout_ms: 4000\n    description: the turntable',
     '    band: .inf\n    timeout_ms: 4000\n    description: the turntable', 'finite number > 0'),
    ('status_period_ms: 100}', 'status_period_ms: 1000}', 'is above 500 ms'),
    ('table: {speed: 180.0, start: load}', 'table: {speed: 180.0, start: [load]}', 'no position'),
])
def test_cell_io_node_refuses_a_malformed_file_with_exit_2_never_a_traceback(tmp_path, old, new, why):
    # implementation review fix C4 (Python review MINOR 1) and C9 (protocol N2), at the process level
    text = H.MOCK_EXAMPLE.read_text()
    assert old in text, old
    rc, out = _node_exit(tmp_path, text.replace(old, new, 1), 'malformed')
    assert rc == 2 and 'refused' in out and why in out, out
    assert 'Traceback' not in out, out


def test_the_cli_says_when_no_cell_answers(tmp_path):
    out = H.cli('state', '--timeout', '1.5')
    assert out.returncode == 1 and 'no cell is publishing /cell_io/state' in out.stdout, out
    out = H.cli('reset', 'turner', '--timeout', '1.5')
    assert out.returncode == 1 and 'no answer from the cell' in out.stdout, out


@pytest.mark.parametrize('delay', [0.02, 0.05, 0.08])
def test_a_second_stop_signal_during_the_cleanup_is_ignored(tmp_path, procs, delay):
    # Ctrl-C reaches the whole process group and `ros2 launch` forwards its own SIGINT: a launched node gets two.
    # Before the fix, a second SIGINT 20-80 ms after the first (during destroy_node) crashed the node: a traceback
    # (exit -2) or a SIGSEGV (exit 139), measured.
    import os
    proc = _start_cell(tmp_path, procs, name=f'cell_{int(delay * 1000)}')
    os.killpg(proc.pgid, signal.SIGINT)
    time.sleep(delay)
    try:
        os.killpg(proc.pgid, signal.SIGINT)
    except ProcessLookupError:
        pass
    assert proc.wait_exit(20) == 0, proc.text()[-2000:]
    assert proc.stop() == 'clean' and 'Traceback' not in proc.text(), proc.text()[-2000:]
