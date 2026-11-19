"""Shared helpers of trainit_cell_io's ROS-level tests (a helper module: pytest collects nothing here).

The cell's executables run as CHILD processes from this source tree (`python3 -m trainit_cell_io.ros.<node>` with
this repository first on PYTHONPATH), each in its own process group, on a private ROS domain (57 by default,
`TRAINIT_CELL_IO_TEST_DOMAIN` overrides it) with ROS_LOCALHOST_ONLY=1. The test process plays the robot's runner with
a scripted rclpy requester. Needs ROS 2 Humble and trainit_interfaces >= 0.5.0 on the path (a sourced workspace);
without them every ROS test SKIPS, naming why.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))      # this source tree first (the child processes get the same on PYTHONPATH)
EXAMPLE_DIR = ROOT / 'examples' / 'neutral_station'
MOCK_EXAMPLE = EXAMPLE_DIR / 'cell_io_mock.yaml'
DOMAIN = os.environ.get('TRAINIT_CELL_IO_TEST_DOMAIN', '57')


def ros_missing() -> Optional[str]:
    """None when this interpreter can run the ROS tests, else why not."""
    try:
        import rclpy  # noqa: F401
        from trainit_interfaces.msg import ChannelStatus
    except ImportError as exc:
        return f'needs ROS 2 (rclpy) and trainit_interfaces >= 0.5.0 in a sourced workspace: {exc}'
    if not hasattr(ChannelStatus, 'STATE_HOMING'):
        return 'needs trainit_interfaces >= 0.5.0 (ChannelStatus)'
    return None


def configure_this_process() -> None:
    """The private domain for THIS process; call before rclpy.init()."""
    os.environ['ROS_DOMAIN_ID'] = DOMAIN
    os.environ['ROS_LOCALHOST_ONLY'] = '1'


def child_env(extra: Optional[Dict[str, str]] = None, domain: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(ROS_DOMAIN_ID=domain or DOMAIN, ROS_LOCALHOST_ONLY='1', PYTHONUNBUFFERED='1',
               RCUTILS_COLORIZED_OUTPUT='0', PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if extra:
        env.update(extra)
    return env


def write_config(path: Path, mutate: Optional[Callable[[dict], None]] = None, fast: bool = True) -> Path:
    """The public example, optionally sped up (fast station motions) and mutated; written to `path`."""
    data = yaml.safe_load(MOCK_EXAMPLE.read_text(encoding='utf-8'))
    if fast:
        data['devices']['clamp']['timeout_ms'] = 400
        data['mock']['devices'] = {'clamp': {'close_ms': 100, 'open_ms': 100},
                                   'table': {'speed': 900.0, 'start': 'load'},
                                   'part_present': {'value': True}, 'index': {'speed': 1500.0, 'start': 'home'}}
    if mutate:
        mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    return path


class Proc:
    """A child process in its own process group, its output in a log file."""

    def __init__(self, args: Sequence[str], log: Path, env: Optional[Dict[str, str]] = None,
                 cwd: Optional[Path] = None):
        self.args = list(args)
        self.log = log
        self._fh = open(log, 'w', encoding='utf-8')
        self.p = subprocess.Popen(self.args, stdout=self._fh, stderr=subprocess.STDOUT, env=env or child_env(),
                                  cwd=str(cwd) if cwd else None, start_new_session=True, stdin=subprocess.DEVNULL)
        self.pgid = self.p.pid
        self.outcome: Optional[str] = None

    def text(self) -> str:
        if not self._fh.closed:
            self._fh.flush()
        return self.log.read_text(encoding='utf-8', errors='replace')

    def alive(self) -> bool:
        return self.p.poll() is None

    def wait_for(self, pattern: str, timeout: float, group: bool = False) -> bool:
        """The pattern appears in the log while the process (`group`: any process of its group) runs."""
        end = time.monotonic() + timeout
        rx = re.compile(pattern, re.M)
        while time.monotonic() < end:
            if rx.search(self.text()):
                return True
            if not (self.group_alive() if group else self.alive()):
                return bool(rx.search(self.text()))
            time.sleep(0.05)
        return False

    def wait_exit(self, timeout: float) -> Optional[int]:
        try:
            return self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def wait_group_exit(self, timeout: float) -> bool:
        """Every process of the group gone within `timeout` (a wrapper script may die before its child)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.p.poll()
            if not self.group_alive():
                return True
            time.sleep(0.05)
        return False

    def group_alive(self) -> bool:
        try:
            os.killpg(self.pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def stop(self, sig: int = signal.SIGINT, timeout: float = 20.0) -> str:
        """'clean' = exited within `timeout` after `sig` and nothing of its group is left; else what it needed."""
        if self.outcome is not None:
            return self.outcome
        if self.alive():
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                pass
        rc = self.wait_exit(timeout)
        end = time.monotonic() + 5.0
        while self.group_alive() and time.monotonic() < end:
            time.sleep(0.05)
        if rc is None or self.group_alive():
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.p.wait(timeout=10)
            self.outcome = 'needed SIGKILL'
        else:
            self.outcome = 'clean'
        self._fh.close()
        return self.outcome

    def kill(self) -> None:
        """SIGKILL the group (an abrupt death)."""
        try:
            os.killpg(self.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.p.wait(timeout=10)
        self.outcome = 'killed'
        self._fh.close()


def python_module(module: str, *ros_args: str) -> List[str]:
    return [sys.executable, '-m', module, *ros_args]


def cell_node(config: Path, log: Path) -> Proc:
    return Proc(python_module('trainit_cell_io.ros.cell_io_node', '--ros-args', '-p', f'config:={config}'), log)


def bridge_node(config: Path, log: Path, domain: Optional[str] = None) -> Proc:
    return Proc(python_module('trainit_cell_io.ros.cell_io_bridge', '--ros-args', '-p', f'config:={config}'), log,
                env=child_env(domain=domain))


def cli(*args: str, timeout: float = 60.0, domain: Optional[str] = None) -> subprocess.CompletedProcess:
    return subprocess.run(python_module('trainit_cell_io.ros.cli', *args), env=child_env(domain=domain),
                          capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def init_context(domain: Optional[str] = None):
    """A private rclpy context for this test process (no signal handlers: pytest owns the signals)."""
    import rclpy
    from rclpy.context import Context
    from rclpy.signals import SignalHandlerOptions
    configure_this_process()
    ctx = Context()
    rclpy.init(context=ctx, domain_id=int(domain or DOMAIN), signal_handler_options=SignalHandlerOptions.NO)
    return ctx


def reader_qos(depth: int = 100):
    """RELIABLE + TRANSIENT_LOCAL like the contract, deeper on the reader side so no image is missed."""
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(depth=depth, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def latched_qos():
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Requester:
    """The robot's runner in miniature: a status subscription and a request publisher per service, plus the cell's
    operator view. Every status image is recorded with its arrival time (time.monotonic)."""

    def __init__(self, context, services: Sequence[str], slot: str = '', name: str = 'scripted_requester',
                 publish_requests: bool = True):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from std_msgs.msg import String
        from trainit_interfaces.msg import ChannelRequest, ChannelStatus
        self._rclpy = rclpy
        self._ChannelRequest = ChannelRequest
        self.node = rclpy.create_node(name, context=context)
        self.executor = SingleThreadedExecutor(context=context)
        self.executor.add_node(self.node)
        prefix = f'/{slot}' if slot else ''
        self.topics = {s: (f'{prefix}/{s}/status', f'{prefix}/{s}/request') for s in services}
        self.status: Dict[str, List[Tuple[float, object]]] = {s: [] for s in services}
        self.pubs = {}
        for s, (status_topic, request_topic) in self.topics.items():
            self.node.create_subscription(ChannelStatus, status_topic,
                                          lambda msg, s=s: self.status[s].append((time.monotonic(), msg)),
                                          reader_qos())
            if publish_requests:
                self.pubs[s] = self.node.create_publisher(ChannelRequest, request_topic, latched_qos())
        self.state: Optional[dict] = None
        self.node.create_subscription(String, '/cell_io/state', self._on_state, latched_qos())

    def _on_state(self, msg) -> None:
        try:
            self.state = json.loads(msg.data)
        except ValueError:
            pass

    def spin_for(self, seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.executor.spin_once(timeout_sec=min(0.02, max(0.0, end - time.monotonic())))

    def spin_until(self, predicate: Callable[[], bool], timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            self.executor.spin_once(timeout_sec=0.02)
        return predicate()

    def latest(self, service: str):
        return self.status[service][-1][1] if self.status[service] else None

    def first_after(self, service: str, t0: float, predicate: Callable[[object], bool]):
        for t, msg in self.status[service]:
            if t >= t0 and predicate(msg):
                return t, msg
        return None

    def wait_status(self, service: str, predicate: Callable[[object], bool], timeout: float, after: float = 0.0):
        """The first status image (arrived at or after `after`) that satisfies `predicate`, or None."""
        found = []

        def check() -> bool:
            hit = self.first_after(service, after, predicate)
            if hit:
                found.append(hit)
            return bool(hit)
        self.spin_until(check, timeout)
        return found[0] if found else None

    def wait_matched(self, service: str, timeout: float = 10.0) -> bool:
        """Our request publisher matched the source's subscription (so a request is never a late-joiner sample)."""
        return self.spin_until(lambda: self.pubs[service].get_subscription_count() >= 1, timeout)

    def request(self, service: str, source_session: int, requester_session: int, seq: int) -> float:
        msg = self._ChannelRequest()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.source_session, msg.requester_session, msg.seq = source_session, requester_session, seq
        self.pubs[service].publish(msg)
        return time.monotonic()

    def counters(self, service: str) -> dict:
        return ((self.state or {}).get('services', {}).get(service) or {}).get('counters', {})

    def wait_state(self, predicate: Callable[[dict], bool], timeout: float) -> bool:
        return self.spin_until(lambda: self.state is not None and predicate(self.state), timeout)

    def close(self) -> None:
        try:
            self.executor.remove_node(self.node)
            self.node.destroy_node()
            self.executor.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: BLE001 -- teardown
            pass


def timing(name: str, value_s: float, note: str = '') -> str:
    """Record a measured timing (printed; appended to $TRAINIT_CELL_IO_EVIDENCE/timings.txt when set)."""
    line = f'TIMING {name}: {value_s * 1000.0:.1f} ms{" -- " + note if note else ""}'
    print(line)
    out = os.environ.get('TRAINIT_CELL_IO_EVIDENCE')
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / 'timings.txt', 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')
    return line
