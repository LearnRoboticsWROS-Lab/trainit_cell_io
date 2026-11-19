"""The neutral station in Isaac Sim, headless -- opt-in: TRAINIT_ISAAC_TEST=1, Isaac Sim 5.1 at $ISAAC_SIM_PATH (default
~/isaacsim), a sourced ROS 2 Humble workspace with trainit_interfaces >= 0.5.0. Otherwise every test SKIPS.

The PLANT only, on its own private ROS domain (58 by default, TRAINIT_ISAAC_TEST_DOMAIN): no robot and no runner here
(the gate's I level runs the robot on mock against this stage, with use_sim_time). Logs go to
$TRAINIT_CELL_IO_EVIDENCE/isaac when set (else pytest's temp dir).

1. `run_neutral_station.py --selfcheck` (no ROS): one turner and one indexer cycle from the physical read-back -> PASS;
2. the served stage + `cell_io_bridge` (ROS side) + a scripted requester: turner and indexer REQUEST -> BUSY -> DONE
   through the canonical topics, from the joints' read-back; the cycle times in wall and simulated time (the status
   stamps are the stage's clock) and the stage's measured real-time factor are recorded;
3. the Kit process group KILLED (SIGKILL) while the bridge lives: no status image, no heartbeat progress and no /clock
   after the kill -- the bridge never mints one (C5, PB-P9's mechanics at the I level);
4. orderly stops: a SIGINT / SIGTERM sent as soon as the runner prints CELL_IO_STAGE_BUILDING (its own stop flag is in:
   a stop during the build), and after READY or at --duration.
"""

import os
import re
import signal
import time
from pathlib import Path

import pytest

import ros_harness as H

ISAAC = Path(os.environ.get('ISAAC_SIM_PATH', str(Path.home() / 'isaacsim')))
DOMAIN = os.environ.get('TRAINIT_ISAAC_TEST_DOMAIN', '58')
EX = H.EXAMPLE_DIR
RUNNER = EX / 'run_neutral_station.py'
REQ = 0x15AAC5E1

pytestmark = [
    pytest.mark.skipif(os.environ.get('TRAINIT_ISAAC_TEST') != '1', reason='opt-in: set TRAINIT_ISAAC_TEST=1 '
                       '(runs Isaac Sim headless)'),
    pytest.mark.skipif(not (ISAAC / 'python.sh').exists(), reason=f'no Isaac Sim at {ISAAC} (ISAAC_SIM_PATH)'),
    pytest.mark.skipif(H.ros_missing() is not None, reason=H.ros_missing() or ''),
]

try:
    from trainit_interfaces.msg import ChannelStatus as CS
except ImportError:          # skipped
    CS = None


def kit_env():
    """A CLEAN environment for Kit (a ROS-sourced one crashes its bridge); the domain passed explicitly."""
    return {'HOME': str(Path.home()), 'PATH': '/usr/bin:/bin', 'ROS_DISTRO': 'humble',
            'RMW_IMPLEMENTATION': 'rmw_fastrtps_cpp', 'ROS_DOMAIN_ID': DOMAIN, 'ROS_LOCALHOST_ONLY': '1',
            'LD_LIBRARY_PATH': str(ISAAC / 'exts' / 'isaacsim.ros2.bridge' / 'humble' / 'lib')}


@pytest.fixture(scope='module')
def evidence(tmp_path_factory):
    out = os.environ.get('TRAINIT_CELL_IO_EVIDENCE')
    path = Path(out) / 'isaac' if out else tmp_path_factory.mktemp('isaac')
    path.mkdir(parents=True, exist_ok=True)
    return path


def _note(evidence, line):
    print(line)
    with open(evidence / 'summary.txt', 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def test_the_selfcheck_runs_both_cycles_from_the_physical_read_back(evidence):
    proc = H.Proc([str(ISAAC / 'python.sh'), str(RUNNER), '--selfcheck'], evidence / 'selfcheck.log', env=kit_env(),
                  cwd=EX)
    rc = proc.wait_exit(900)
    proc.stop()
    text = proc.text()
    for line in text.splitlines():
        if line.startswith('CELL_IO_SELFCHECK') and any(word in line for word in ('->', 'PASS', 'FAIL', 'nest')):
            _note(evidence, f'selfcheck: {line}')
    assert 'CELL_IO_SELFCHECK PASS' in text and 'CELL_IO_STAGE_END rc=0' in text, text[-3000:]
    assert rc == 0 or rc is not None


def test_the_stage_through_the_bridge_then_a_killed_kit_freezes_the_heartbeat(evidence):
    stage = H.Proc([str(ISAAC / 'python.sh'), str(RUNNER)], evidence / 'stage.log', env=kit_env(), cwd=EX)
    bridge = None
    ctx = None
    req = None
    try:
        assert stage.wait_for(r'^CELL_IO_STAGE_READY', 600), stage.text()[-3000:]
        _note(evidence, 'stage: ' + re.search(r'^CELL_IO_STAGE_READY.*$', stage.text(), re.M).group(0))
        bridge = H.bridge_node(EX / 'cell_io_isaac.yaml', evidence / 'bridge.log', domain=DOMAIN)
        assert bridge.wait_for(r'cell_io_bridge: bridging 2', 60), bridge.text()
        ctx = H.init_context(DOMAIN)
        req = H.Requester(ctx, ['turner', 'indexer'], name='isaac_smoke_requester')
        clock_seen = []
        from rosgraph_msgs.msg import Clock
        req.node.create_subscription(Clock, '/clock', lambda m: clock_seen.append((time.monotonic(), m)), 50)
        for n, service in enumerate(('turner', 'indexer'), start=1):
            ready = req.wait_status(service, lambda m: m.state == CS.STATE_READY, 60)
            assert ready, f'no READY status of {service} through the bridge: {bridge.text()[-2000:]}'
            assert req.wait_matched(service)
            req.spin_for(0.5)
            t0 = req.request(service, ready[1].source_session, REQ, n)
            busy = req.wait_status(service, lambda m: m.state == CS.STATE_BUSY
                                   and (m.cycle_requester, m.cycle_seq) == (REQ, n), 10, after=t0)
            done = req.wait_status(service, lambda m: m.state in (CS.STATE_DONE, CS.STATE_FAULT)
                                   and (m.cycle_requester, m.cycle_seq) == (REQ, n), 60, after=t0)
            assert busy and done, f'{service}: {req.latest(service)}'
            assert done[1].state == CS.STATE_DONE and done[1].cycle_done, done[1]
            _note(evidence, f'{service}: request -> BUSY {1000 * (busy[0] - t0):.1f} ms wall; request -> DONE '
                            f'{done[0] - t0:.3f} s wall; BUSY -> DONE {_stamp(done[1]) - _stamp(busy[1]):.3f} s '
                            f'simulated (status stamps = the stage clock); detail "{done[1].detail}"')
        state = H.cli('state', domain=DOMAIN)
        assert state.returncode == 0 and 'turner: DONE' in state.stdout and 'indexer: DONE' in state.stdout, state
        _note(evidence, 'cell_io_cli state (served by the Kit side):\n' + state.stdout.rstrip())
        assert stage.wait_for(r'^RTF ', 30)
        for line in re.findall(r'^RTF .*$', stage.text(), re.M)[-3:]:
            _note(evidence, f'stage: {line}')
        assert clock_seen and req.status['turner']
        # --- PB-P9's mechanics: Kit killed (SIGKILL, its process group only), the bridge alive -------------------
        req.spin_for(0.5)
        before = (len(req.status['turner']), req.latest('turner').heartbeat)
        stage.kill()
        t_kill = time.monotonic()
        req.spin_for(0.3)                                     # in flight before the kill
        after_kill = (len(req.status['turner']), req.latest('turner').heartbeat)
        req.spin_for(3.0)
        frozen = (len(req.status['turner']), req.latest('turner').heartbeat)
        late_clock = [t for t, _ in clock_seen if t > t_kill + 0.3]
        _note(evidence, f'Kit SIGKILLed: turner images {before[0]} (heartbeat {before[1]}) before; {after_kill[0]} '
                        f'(heartbeat {after_kill[1]}) 0.3 s after; {frozen[0]} (heartbeat {frozen[1]}) 3.3 s after; '
                        f'/clock messages after the kill (+0.3 s): {len(late_clock)}; the bridge alive: '
                        f'{bridge.alive()}')
        assert frozen == after_kill, 'a status image arrived after the stage died'
        assert late_clock == [] and bridge.alive()
        assert bridge.wait_for(r'no status image from the Isaac stage for', 5)
        assert bridge.stop() == 'clean'
        _note(evidence, 'bridge stopped cleanly (SIGINT)')
    finally:
        if req is not None:
            req.close()
        if ctx is not None:
            ctx.try_shutdown()
        if bridge is not None:
            bridge.stop()
        if stage.alive():
            stage.stop(signal.SIGINT, timeout=60)


@pytest.mark.parametrize('how', ['SIGINT', 'SIGTERM'])
def test_a_stop_signal_from_the_build_on_ends_the_run_in_order(evidence, how):
    """Review fix C8: the runner installs its own SIGINT / SIGTERM flag right after rclpy.init() and prints
    CELL_IO_STAGE_BUILDING; a stop signal sent at once ends the run in order (CELL_IO_STAGE_END rc=0, no traceback):
    between two build phases ("stopped by ... during the build") or, if the build was already done, in the serve
    loop ("stopping: ...")."""
    stage = H.Proc([str(ISAAC / 'python.sh'), str(RUNNER)], evidence / f'stage_stop_build_{how}.log', env=kit_env(),
                   cwd=EX)
    try:
        assert stage.wait_for(r'^CELL_IO_STAGE_BUILDING', 600), stage.text()[-3000:]
        os.killpg(stage.pgid, getattr(signal, how))
        assert stage.wait_for(r'^CELL_IO_STAGE_END rc=0', 120, group=True), stage.text()[-3000:]
        gone = stage.wait_group_exit(120)
        text = stage.text()
        where = re.search(rf'^(stopped by {how} .*|stopping: .*)$', text, re.M)
        _note(evidence, f'{how} at CELL_IO_STAGE_BUILDING: "{where.group(0) if where else "?"}", '
                        f'CELL_IO_STAGE_END rc=0, every process of its group gone: {gone}')
        assert where and gone and 'Traceback' not in text, text[-3000:]
    finally:
        if stage.alive():
            stage.kill()


@pytest.mark.parametrize('how', ['SIGINT', 'SIGTERM', 'duration'])
def test_the_stage_stops_in_order_on_a_signal_or_its_duration(evidence, how):
    args = [str(ISAAC / 'python.sh'), str(RUNNER)] + (['--duration', '3'] if how == 'duration' else [])
    stage = H.Proc(args, evidence / f'stage_stop_{how}.log', env=kit_env(), cwd=EX)
    try:
        assert stage.wait_for(r'^CELL_IO_STAGE_READY', 600), stage.text()[-3000:]
        if how != 'duration':
            time.sleep(2.0)
            os.killpg(stage.pgid, getattr(signal, how))
        # python.sh runs Kit's Python as a child: on SIGTERM the wrapper dies at once, the child stops in order
        assert stage.wait_for(r'^CELL_IO_STAGE_END rc=0', 120, group=True), stage.text()[-3000:]
        gone = stage.wait_group_exit(120)
        stopping = re.search(r'^stopping: .*$', stage.text(), re.M)
        _note(evidence, f'stage stopped by {how}: "{stopping.group(0) if stopping else "?"}", CELL_IO_STAGE_END rc=0, '
                        f'every process of its group gone: {gone}')
        assert stopping and gone
    finally:
        if stage.alive():
            stage.kill()
