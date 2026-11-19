"""`cell_io_bridge` at the process level (P0.b WP2), with a FAKE Kit, on a private ROS domain (57; ros_harness.py).

Two fakes stand in for the Isaac stage:
- a raw publisher of Kit status JSON (field-level forwarding, malformed images);
- the REAL Kit-side loop (`trainit_cell_io.isaac.loop.KitSide` + `kit_ros.KitRos`, the code the stage runner runs in
  Kit) stepping the core cell over the MOCK plant in a thread of this process: the Isaac pipeline minus Isaac physics.

Exercised: C5's ROS half (the bridge publishes the canonical types from std_msgs JSON), the bridge never mints a
heartbeat nor republishes on its own (a dead Kit freezes the status: PB-P9's mechanics), the request forwarding with
the requester count and its re-forwarding when the count changes (rule 6), malformed Kit images refused with a log
line and never half-applied, a restarted bridge re-forwarding the level-held request without a second cycle (PB-P21's
mechanics), the operator CLI served by the Kit side, the bridge's exit codes.
"""

import json
import threading
import time

import pytest

import ros_harness as H

pytestmark = pytest.mark.skipif(H.ros_missing() is not None, reason=H.ros_missing() or '')

try:
    from trainit_interfaces.msg import ChannelStatus as CS
except ImportError:          # the module is skipped
    CS = None

from trainit_cell_io.core import Cell, MockPlantAdapter, load  # noqa: E402
from trainit_cell_io.core import kit_boundary as KB  # noqa: E402
from trainit_cell_io.core.contract import StatusImage  # noqa: E402
from trainit_cell_io.isaac.loop import KitSide, Pacer, SimClock  # noqa: E402

REQ = 0x0B21D6E5


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


def _isaac_config(tmp_path, mutate=None):
    def isaac(data):
        data['plant'] = 'isaac'
        if mutate:
            mutate(data)
    return H.write_config(tmp_path / 'cell_io_isaac.yaml', isaac)


def _start_bridge(tmp_path, procs, cfg, name='bridge'):
    proc = H.bridge_node(cfg, tmp_path / f'{name}.log')
    procs.append(proc)
    assert proc.wait_for(r'cell_io_bridge: bridging 2 machine service\(s\)', 30), proc.text()
    return proc


class FakeKit:
    """Raw Kit status JSON out, the forwarded request JSON in (recorded with arrival times)."""

    def __init__(self, ctx, services=('turner', 'indexer')):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String
        self._String = String
        self.node = rclpy.create_node('fake_kit', context=ctx)
        self.executor = SingleThreadedExecutor(context=ctx)
        self.executor.add_node(self.node)
        self.pubs = {s: self.node.create_publisher(String, KB.kit_topic(s, 'status'), H.latched_qos())
                     for s in services}
        self.requests = {s: [] for s in services}
        for s in services:
            self.node.create_subscription(String, KB.kit_topic(s, 'request'),
                                          lambda msg, s=s: self.requests[s].append((time.monotonic(),
                                                                                   json.loads(msg.data))),
                                          H.reader_qos())

    def publish(self, service, text):
        self.pubs[service].publish(self._String(data=text))

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=0.01)

    def close(self):
        self.executor.remove_node(self.node)
        self.node.destroy_node()
        self.executor.shutdown(timeout_sec=1.0)


class KitThread:
    """The real Kit-side loop over the mock plant, paced to real time in a thread (the Isaac pipeline minus Isaac)."""

    def __init__(self, ctx, cfg_path, name='fake_stage'):
        from trainit_cell_io.isaac.kit_ros import KitRos
        cfg = load(str(cfg_path))
        cfg.plant = 'mock'           # the same cell file, the mock time model standing in for Isaac's physics
        self.cell = Cell(cfg, MockPlantAdapter(cfg.mock))
        self.lines = []
        self.kit = KitSide(self.cell, SimClock(cfg.clock.dt, time.time_ns()), log=self.lines.append,
                           advance_plant=True)
        self.ros = KitRos(self.cell.services, self.kit.on_request, self.kit.on_command, context=ctx, node_name=name)
        self.kit.transport = self.ros
        self.pacer = Pacer(float(cfg.clock.dt))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        self.pacer.start()
        while not self._stop.is_set():
            self.kit.after_physics_step()
            self.pacer.pace(self.cell.steps)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        self.ros.close()


def _image(**kw):
    base = dict(source_session=0x1234ABCD, heartbeat=1, state=CS.STATE_READY, detail='ready')
    base.update(kw)
    return StatusImage(**base)


def _warm_up(kit, requester, service='turner'):
    """Republish a warm-up image until the bridge has forwarded it. DDS matching is asymmetric: the fake Kit's writer
    may see the bridge's reader matched before the reader has matched the writer, and a depth-1 (level-held) writer
    history then replaces an unacknowledged first sample -- the live stage republishes every period, a test must not
    lose its first stimulus. Returns how many images the requester has received so far."""
    image = KB.encode_status(service, _image(heartbeat=0, detail='warm-up'), 1)
    end = time.monotonic() + 15.0
    while time.monotonic() < end:
        if any(m.detail == 'warm-up' for _, m in requester.status[service]):
            requester.spin_for(0.2)
            return len(requester.status[service])
        kit.publish(service, image)
        requester.spin_for(0.05)
    raise AssertionError('the bridge never forwarded the warm-up image')


# --- the raw fake Kit: forwarding, never minting, refusing --------------------------------------------------------

def test_c5_each_kit_image_is_forwarded_field_for_field_as_the_canonical_status(tmp_path, procs, requester, ctx):
    from rclpy.qos import DurabilityPolicy, ReliabilityPolicy
    _start_bridge(tmp_path, procs, _isaac_config(tmp_path))
    kit = FakeKit(ctx)
    try:
        base = _warm_up(kit, requester)
        images = [_image(heartbeat=n) for n in (1, 2)] + [
            _image(heartbeat=3, state=CS.STATE_BUSY, cycle_requester=REQ, cycle_seq=9, detail='busy: jaw closing'),
            _image(heartbeat=3, state=CS.STATE_DONE, cycle_requester=REQ, cycle_seq=9, cycle_done=True,
                   reject_requester=REQ, reject_seq=10, reject_code=CS.REJECT_BUSY, detail='done'),
            _image(heartbeat=65535, state=CS.STATE_FAULT, fault_code=CS.FAULT_ACTUATOR_TIMEOUT, detail='FAULT x')]
        stamps = [1_795_000_000_000_000_000 + n * 5_000_000 for n in range(len(images))]
        for image, stamp in zip(images, stamps):
            kit.publish('turner', KB.encode_status('turner', image, stamp))
            kit.spin_for(0.05)
        assert requester.spin_until(lambda: len(requester.status['turner']) >= base + len(images), 5)
        got = [m for _, m in requester.status['turner'][base:]]
        assert len(got) == len(images)
        for image, stamp, msg in zip(images, stamps, got):
            assert (msg.source_session, msg.heartbeat, msg.state, msg.cycle_requester, msg.cycle_seq, msg.cycle_done,
                    msg.reject_requester, msg.reject_seq, msg.reject_code, msg.fault_code, msg.detail) == (
                image.source_session, image.heartbeat, image.state, image.cycle_requester, image.cycle_seq,
                image.cycle_done, image.reject_requester, image.reject_seq, image.reject_code, image.fault_code,
                image.detail)
            assert msg.header.stamp.sec * 10 ** 9 + msg.header.stamp.nanosec == stamp   # the Kit's clock
        (pub,) = [i for i in requester.node.get_publishers_info_by_topic('/turner/status')
                  if i.node_name == 'cell_io_bridge']
        assert pub.topic_type == 'trainit_interfaces/msg/ChannelStatus'
        assert pub.qos_profile.reliability == ReliabilityPolicy.RELIABLE
        assert pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    finally:
        kit.close()


def test_the_bridge_never_mints_a_heartbeat_a_silent_kit_freezes_the_status(tmp_path, procs, requester, ctx):
    bridge = _start_bridge(tmp_path, procs, _isaac_config(tmp_path))
    kit = FakeKit(ctx)
    try:
        base = _warm_up(kit, requester)
        for n in range(1, 6):
            kit.publish('turner', KB.encode_status('turner', _image(heartbeat=n), n))
            kit.spin_for(0.1)
        assert requester.spin_until(lambda: len(requester.status['turner']) >= base + 5, 5)
        before = len(requester.status['turner'])
        requester.spin_for(2.5)                       # the Kit is silent: nothing new, ever
        assert len(requester.status['turner']) == before and requester.latest('turner').heartbeat == 5
        assert bridge.alive() and 'no status image from the Isaac stage for' in bridge.text()
        assert requester.status['indexer'] == []      # a service the Kit never served: nothing invented
    finally:
        kit.close()


def test_malformed_kit_images_are_refused_with_a_log_line_and_never_half_applied(tmp_path, procs, requester, ctx):
    bridge = _start_bridge(tmp_path, procs, _isaac_config(tmp_path))
    kit = FakeKit(ctx)
    try:
        base = _warm_up(kit, requester)
        good = KB.encode_status('turner', _image(heartbeat=7), 1)
        bad = ['nope', '{"v": 2}', good.replace('"state": 1', '"state": 9'),
               good.replace('"heartbeat": 7', '"heartbeat": 70000'),
               good.replace('"source_session": 305441741', '"source_session": 0'),
               good.replace('"cycle_done": false', '"cycle_done": 0'),
               good.replace('"stamp_ns": 1,', '"stamp_ns": 9223372036854775807,'),   # beyond a ROS time
               KB.encode_status('indexer', _image(heartbeat=7), 1)]       # another service's image on this topic
        for text in bad:
            kit.publish('turner', text)
            kit.spin_for(0.08)
        requester.spin_for(0.8)
        assert len(requester.status['turner']) == base, 'a malformed image was forwarded'
        assert bridge.text().count('refused a malformed status image from the Isaac stage') == len(bad), bridge.text()
        kit.publish('turner', good)
        assert requester.spin_until(lambda: len(requester.status['turner']) == base + 1, 5)
        msg = requester.latest('turner')
        assert (msg.heartbeat, msg.state, msg.source_session) == (7, CS.STATE_READY, 0x1234ABCD)
    finally:
        kit.close()


def test_a_request_is_forwarded_with_the_requester_count_and_re_forwarded_when_it_changes(tmp_path, procs,
                                                                                          requester, ctx):
    _start_bridge(tmp_path, procs, _isaac_config(tmp_path))
    kit = FakeKit(ctx)
    try:
        assert requester.wait_matched('turner')
        requester.spin_for(0.3)
        requester.request('turner', 0x77, REQ, 1)
        assert requester.spin_until(lambda: (kit.spin_for(0.02) or True) and kit.requests['turner'], 5)
        first = kit.requests['turner'][-1][1]
        assert {k: first[k] for k in ('service', 'source_session', 'requester_session', 'seq', 'requesters')} == {
            'service': 'turner', 'source_session': 0x77, 'requester_session': REQ, 'seq': 1, 'requesters': 1}
        other = H.Requester(ctx, ['turner'], name='second_requester')
        try:
            def saw(n):
                kit.spin_for(0.02)
                return kit.requests['turner'][-1][1]['requesters'] == n
            assert requester.spin_until(lambda: saw(2), 5), kit.requests['turner']
            last = kit.requests['turner'][-1][1]
            assert (last['requester_session'], last['seq']) == (REQ, 1)        # the SAME image, the new count
        finally:
            other.close()
        assert requester.spin_until(lambda: saw(1), 5)
    finally:
        kit.close()


def test_the_bridge_refuses_the_mock_plant_and_a_missing_config(tmp_path):
    proc = H.bridge_node(H.MOCK_EXAMPLE, tmp_path / 'mock.log')
    assert proc.wait_exit(60) == 2 and 'the mock plant is served by cell_io_node' in proc.text(), proc.text()
    proc.stop()
    proc = H.Proc(H.python_module('trainit_cell_io.ros.cell_io_bridge'), tmp_path / 'none.log')
    assert proc.wait_exit(60) == 2 and 'config:=<cell_io.yaml> is required' in proc.text()
    proc.stop()


@pytest.mark.parametrize('old, new, why', [
    ('- {close: clamp}', '- {close: [clamp]}', "unknown device ['clamp']"),               # was a TypeError traceback
    ('positions: {home: 0.0, forward: 300.0}', 'positions: {home: 0.0, forward: .inf}', 'must be a finite number'),
    ('status_period_ms: 100}', 'status_period_ms: 600}', 'is above 500 ms'),
])
def test_the_bridge_refuses_a_malformed_file_with_exit_2_never_a_traceback(tmp_path, old, new, why):
    # implementation review fix C4 (Python review MINOR 1) and C9 (protocol N2), at the process level: the bridge never
    # builds a Cell, so the file must be refused by load() itself
    text = (H.EXAMPLE_DIR / 'cell_io_isaac.yaml').read_text()
    assert old in text, old
    path = tmp_path / 'malformed.yaml'
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    proc = H.bridge_node(path, tmp_path / 'malformed.log')
    rc = proc.wait_exit(60)
    proc.stop()
    out = proc.text()
    assert rc == 2 and 'refused' in out and why in out, out
    assert 'Traceback' not in out, out


# --- the real Kit-side loop (mock plant) behind the bridge --------------------------------------------------------

def test_the_kit_side_through_the_bridge_runs_a_cycle_and_serves_the_operator_cli(tmp_path, procs, requester, ctx):
    cfg = _isaac_config(tmp_path)
    _start_bridge(tmp_path, procs, cfg)
    stage = KitThread(ctx, cfg)
    try:
        hit = requester.wait_status('turner', lambda m: m.state == CS.STATE_READY, 15)
        assert hit and hit[1].source_session == stage.cell.services['turner'].session
        assert requester.wait_matched('turner')
        requester.spin_for(0.3)
        t0 = requester.request('turner', hit[1].source_session, REQ, 1)
        busy = requester.wait_status('turner', lambda m: m.state == CS.STATE_BUSY
                                     and (m.cycle_requester, m.cycle_seq) == (REQ, 1), 5, after=t0)
        done = requester.wait_status('turner', lambda m: m.state == CS.STATE_DONE and m.cycle_done
                                     and (m.cycle_requester, m.cycle_seq) == (REQ, 1), 15, after=t0)
        assert busy and done
        H.timing('Kit side (mock plant) + bridge: request -> BUSY', busy[0] - t0)
        H.timing('Kit side (mock plant) + bridge: request -> DONE (fast mock test model)', done[0] - t0)
        stamp = done[1].header.stamp.sec + done[1].header.stamp.nanosec * 1e-9
        assert abs(stamp - time.time()) < 5.0         # the Kit's clock: wall at start + physics time (RTF <= 1)
        beats = [m.heartbeat for _, m in requester.status['turner'][-6:]]
        assert len(set(beats)) >= 4                   # the heartbeat runs (minted by the Kit-side scan)
        state = H.cli('state')
        assert state.returncode == 0 and 'turner: DONE' in state.stdout, state
        assert H.cli('inject', 'hold', 'service=turner').returncode == 0
        assert stage.cell.plant.describe()['injections']['held'] == ['turner']
    finally:
        stage.stop()


def test_pb_p9_mechanics_a_dead_kit_side_freezes_the_heartbeat_behind_a_living_bridge(tmp_path, procs, requester,
                                                                                      ctx):
    cfg = _isaac_config(tmp_path)
    bridge = _start_bridge(tmp_path, procs, cfg)
    stage = KitThread(ctx, cfg)
    clock_seen = []
    from rosgraph_msgs.msg import Clock
    requester.node.create_subscription(Clock, '/clock', lambda m: clock_seen.append(time.monotonic()), 10)
    try:
        assert requester.wait_status('turner', lambda m: m.heartbeat >= 3, 15)
        assert clock_seen
    finally:
        stage.stop()                                  # the stage dies: its scan, its heartbeat, its /clock stop
    t_dead = time.monotonic()
    requester.spin_for(0.3)
    last = requester.latest('turner').heartbeat
    count = len(requester.status['turner'])
    requester.spin_for(2.5)
    assert len(requester.status['turner']) == count and requester.latest('turner').heartbeat == last
    assert not [t for t in clock_seen if t > t_dead + 0.3]
    assert bridge.alive() and 'no status image from the Isaac stage for' in bridge.text()


def test_pb_p21_mechanics_a_restarted_bridge_re_forwards_the_request_and_no_second_cycle_runs(
        tmp_path, procs, requester, ctx):
    cfg = _isaac_config(tmp_path)
    first = _start_bridge(tmp_path, procs, cfg, name='bridge_a')
    stage = KitThread(ctx, cfg)
    try:
        hit = requester.wait_status('turner', lambda m: m.state == CS.STATE_READY, 15)
        assert hit and requester.wait_matched('turner')
        requester.spin_for(0.3)
        t0 = requester.request('turner', hit[1].source_session, REQ, 1)
        assert requester.wait_status('turner', lambda m: m.state == CS.STATE_DONE and m.cycle_done, 15, after=t0)
        received = stage.kit.requests_received['turner']
        assert requester.wait_state(lambda s: s['services']['turner']['request_images'] == received
                                    and s['services']['turner']['state'] == 'DONE', 5)
        before = dict(requester.state['services']['turner'])
        assert first.stop() == 'clean'
        t1 = time.monotonic()
        _start_bridge(tmp_path, procs, cfg, name='bridge_b')
        assert requester.spin_until(lambda: stage.kit.requests_received['turner'] > received, 10), \
            'the restarted bridge did not re-forward the level-held request'
        assert requester.wait_status('turner', lambda m: m.state == CS.STATE_DONE, 10, after=t1)
        requester.spin_for(1.0)
        pd = stage.cell.services['turner']
        assert pd.counters.accepted == 1 and pd.state == CS.STATE_DONE      # a no-op: no second cycle
        # review fix C3: the evidence on /cell_io/state -- the image reached the source again, and it was a no-op
        assert requester.wait_state(lambda s: s['services']['turner']['request_images'] > before['request_images'], 5)
        after = requester.state['services']['turner']
        assert after['ignored_redelivered'] - before['ignored_redelivered'] == \
            after['request_images'] - before['request_images'] >= 1, (before, after)
        assert after['counters'] == before['counters'], (before, after)
    finally:
        stage.stop()


OTHER = 0x0E1BA5ED           # a second requester's session


def _second_requester(ctx, requester, timeout=10.0):
    other = H.Requester(ctx, ['turner'], name='second_requester')
    assert requester.spin_until(lambda: requester.node.count_publishers('/turner/request') == 2, timeout)
    assert other.wait_matched('turner')
    requester.spin_for(0.6)                              # the bridge counts its publishers every 100 ms
    return other


def test_c1_the_bridges_re_forward_after_a_two_requester_episode_runs_no_second_cycle(
        tmp_path, procs, requester, ctx):
    """The protocol review's probe, exactly as the bridge produces it: R (REQ,1) accepted and DONE; X appears (the
    bridge re-forwards R's image with the new count) and requests (OTHER,1): REJECTED TWO_REQUESTERS; X leaves -> the
    bridge re-forwards R's and X's latest images with the count 1. Before review fix C1 R's image ran a SECOND cycle."""
    cfg = _isaac_config(tmp_path)
    _start_bridge(tmp_path, procs, cfg)
    stage = KitThread(ctx, cfg)
    try:
        hit = requester.wait_status('turner', lambda m: m.state == CS.STATE_READY, 15)
        assert hit and requester.wait_matched('turner')
        requester.spin_for(0.3)
        session = hit[1].source_session
        t0 = requester.request('turner', session, REQ, 1)
        assert requester.wait_status('turner', lambda m: m.state == CS.STATE_DONE and m.cycle_done, 15, after=t0)
        other = _second_requester(ctx, requester)
        try:
            t1 = other.request('turner', session, OTHER, 1)
            assert requester.wait_status('turner', lambda m: (m.reject_requester, m.reject_seq, m.reject_code)
                                         == (OTHER, 1, CS.REJECT_TWO_REQUESTERS), 5, after=t1)
        finally:
            other.close()
        handed = stage.cell.request_images['turner']
        assert requester.spin_until(lambda: requester.node.count_publishers('/turner/request') == 1, 10)
        assert requester.spin_until(lambda: stage.cell.request_images['turner'] >= handed + 2, 5), \
            'the bridge did not re-forward each requester\'s latest image on the count change'
        requester.spin_for(1.5)
        pd = stage.cell.services['turner']
        c = pd.counters
        assert (c.accepted, c.completed, c.rejected_two_requesters) == (1, 1, 1), c.as_dict()   # one cycle, ever
        assert pd.state == CS.STATE_DONE and stage.cell.ignored_redelivered['turner'] >= 2
    finally:
        stage.stop()


def test_c2_two_requesters_images_through_the_bridge_are_both_consumed(tmp_path, procs, requester, ctx):
    cfg = _isaac_config(tmp_path)
    _start_bridge(tmp_path, procs, cfg)
    stage = KitThread(ctx, cfg)
    try:
        hit = requester.wait_status('turner', lambda m: m.state == CS.STATE_READY, 15)
        assert hit and requester.wait_matched('turner')
        session = hit[1].source_session
        other = _second_requester(ctx, requester)
        try:
            requester.request('turner', session, REQ, 1)
            other.request('turner', session, OTHER, 1)
            pd = stage.cell.services['turner']
            assert requester.spin_until(lambda: pd.counters.rejected_two_requesters == 2, 5), \
                (pd.counters.as_dict(), stage.lines[-10:])
            assert pd.counters.accepted == 0
            for rs in (REQ, OTHER):
                assert any(f'request 1 of requester {rs:#010x} rejected TWO_REQUESTERS' in line
                           for line in stage.lines), stage.lines[-10:]
        finally:
            other.close()
    finally:
        stage.stop()


def test_two_requesters_through_the_bridge_reject_a_new_request(tmp_path, procs, requester, ctx):
    cfg = _isaac_config(tmp_path)
    _start_bridge(tmp_path, procs, cfg)
    stage = KitThread(ctx, cfg)
    other = None
    try:
        hit = requester.wait_status('turner', lambda m: m.state == CS.STATE_READY, 15)
        assert hit and requester.wait_matched('turner')
        other = H.Requester(ctx, ['turner'], name='second_requester')
        assert requester.spin_until(lambda: requester.node.count_publishers('/turner/request') == 2, 10)
        requester.spin_for(0.6)
        t0 = requester.request('turner', hit[1].source_session, REQ, 1)
        rejected = requester.wait_status('turner', lambda m: (m.reject_requester, m.reject_seq, m.reject_code)
                                         == (REQ, 1, CS.REJECT_TWO_REQUESTERS), 5, after=t0)
        assert rejected and stage.cell.services['turner'].counters.accepted == 0
    finally:
        if other is not None:
            other.close()
        stage.stop()


@pytest.mark.parametrize('delay', [0.02, 0.05, 0.08])
def test_a_second_stop_signal_during_the_bridge_cleanup_is_ignored(tmp_path, procs, delay):
    import os
    import signal
    proc = _start_bridge(tmp_path, procs, _isaac_config(tmp_path), name=f'bridge_{int(delay * 1000)}')
    os.killpg(proc.pgid, signal.SIGINT)
    time.sleep(delay)
    try:
        os.killpg(proc.pgid, signal.SIGINT)
    except ProcessLookupError:
        pass
    assert proc.wait_exit(20) == 0, proc.text()[-2000:]
    assert proc.stop() == 'clean' and 'Traceback' not in proc.text(), proc.text()[-2000:]
