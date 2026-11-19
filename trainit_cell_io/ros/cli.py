"""`cell_io_cli` -- the cell operator's command line (P0B_DESIGN.md §4): never a robot command.

    ros2 run trainit_cell_io cell_io_cli state [--json]                every service's state, in plain words (or JSON)
    ros2 run trainit_cell_io cell_io_cli watch                         print every change until Ctrl-C
    ros2 run trainit_cell_io cell_io_cli reset <service>               RESET a faulted service (the operator / master)
    ros2 run trainit_cell_io cell_io_cli restart                       simulation: the cell's controller restarts in
                                                                       place (every service a NEW source session)
    ros2 run trainit_cell_io cell_io_cli inject <fault> [key=value]    simulation fault injection, e.g.
        inject stuck_actuator device=clamp | inject station_fault service=turner | inject hold service=turner
        inject slow device=table factor=3 | inject failed_sensor device=part_present
        inject consume_delay service=turner ms=3000 | inject clear

Every verb takes `--timeout <s>` (default 5). Exit codes: 0 done; 1 no cell answered (is `cell_io_node`, or the
Isaac stage runner, running in this ROS domain?); 2 the cell REFUSED the command (the reason is printed) or the
arguments are wrong. The same commands work against the mock cell and the Isaac stage (the same topics).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from typing import Callable, Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from std_msgs.msg import String

from .qos import COMMANDS, CONTROL_TOPIC, LATCHED, STATE_TOPIC
from .runtime import shutdown

EXIT_OK = 0
EXIT_NO_CELL = 1
EXIT_REFUSED = 2
RESEND_S = 1.0


def _value(text: str):
    low = text.lower()
    if low in ('true', 'false'):
        return low == 'true'
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def describe(snapshot: dict) -> str:
    """The cell's state in plain words (the operator's view; never a contract value)."""
    lines = []
    plant = snapshot.get('plant', {})
    lines.append(f"cell: plant {plant.get('plant', '?')}, t = {snapshot.get('time_s', 0.0):.1f} s")
    for sid, svc in sorted(snapshot.get('services', {}).items()):
        c = svc.get('counters', {})
        lines.append(f"  {sid}: {svc.get('state')} -- {svc.get('detail')}")
        lines.append(f"      cycles accepted {c.get('accepted', 0)}, completed {c.get('completed', 0)}, "
                     f"aborted {c.get('aborted', 0)}; rejected busy {c.get('rejected_busy', 0)}, "
                     f"faulted {c.get('rejected_faulted', 0)}, two requesters {c.get('rejected_two_requesters', 0)}; "
                     f"other-session images ignored {c.get('ignored_other_session', 0)}")
        lines.append(f"      request images received {svc.get('request_images', 0)}, of which re-deliveries ignored "
                     f"{svc.get('ignored_redelivered', 0)}")
    devices = snapshot.get('devices', {})
    if devices:
        lines.append('  devices: ' + '; '.join(str(v) for _, v in sorted(devices.items())))
    injections = plant.get('injections')
    if injections and any(injections.values()):
        lines.append(f'  injections: {json.dumps(injections, sort_keys=True)}')
    last = snapshot.get('last_command')
    if last:
        lines.append(f"  last command: {'OK' if last.get('ok') else 'REFUSED'} -- {last.get('text')}")
    return '\n'.join(lines)


class _Cli(Node):
    def __init__(self):
        super().__init__(f'cell_io_cli_{uuid.uuid4().hex[:8]}')
        self.snapshot: Optional[dict] = None
        self.create_subscription(String, STATE_TOPIC, self._on_state, LATCHED)
        self.pub = self.create_publisher(String, CONTROL_TOPIC, COMMANDS)

    def _on_state(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(data, dict):
            self.snapshot = data

    def wait_state(self, timeout_s: float, predicate: Callable[[dict], bool] = lambda s: True) -> Optional[dict]:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.snapshot is not None and predicate(self.snapshot):
                return self.snapshot
        return None

    def send(self, command: dict, timeout_s: float) -> Optional[dict]:
        """Publish the command until the cell answers it (the same id: the cell executes it at most once)."""
        command = dict(command, id=uuid.uuid4().hex)
        end = time.monotonic() + timeout_s
        while self.count_subscribers(CONTROL_TOPIC) == 0 and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        text = json.dumps(command, sort_keys=True)
        answered = lambda s: (s.get('last_command') or {}).get('id') == command['id']  # noqa: E731
        while time.monotonic() < end:
            self.pub.publish(String(data=text))
            snap = self.wait_state(min(RESEND_S, max(0.0, end - time.monotonic())), answered)
            if snap is not None:
                return snap
        return None


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--timeout', type=float, default=5.0, help='seconds to wait for the cell (default 5)')
    parser = argparse.ArgumentParser(prog='cell_io_cli', description="The cell operator's commands (never a robot "
                                     'command).')
    sub = parser.add_subparsers(dest='verb', required=True)
    p_state = sub.add_parser('state', parents=[common], help="every service's state")
    p_state.add_argument('--json', action='store_true', help='print the raw state (JSON)')
    sub.add_parser('watch', parents=[common], help='print every change until Ctrl-C')
    p_reset = sub.add_parser('reset', parents=[common], help='RESET a faulted service')
    p_reset.add_argument('service')
    sub.add_parser('restart', parents=[common], help="simulation: the cell's controller restarts in place (a new "
                   'source session per service; a request of the old session never runs)')
    p_inject = sub.add_parser('inject', parents=[common], help='a simulation fault injection')
    p_inject.add_argument('fault')
    p_inject.add_argument('params', nargs='*', help='key=value, e.g. device=clamp')
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(remove_ros_args(sys.argv)[1:] if argv is None else argv)
    command = None
    if args.verb == 'reset':
        command = {'cmd': 'reset', 'service': args.service}
    elif args.verb == 'restart':
        command = {'cmd': 'restart'}
    elif args.verb == 'inject':
        command = {'cmd': 'inject', 'fault': args.fault}
        for item in args.params:
            key, sep, value = item.partition('=')
            if not sep or not key:
                print(f'inject: parameters are key=value, got {item!r}', file=sys.stderr)
                return EXIT_REFUSED
            command[key] = _value(value)
    rclpy.init()
    node = _Cli()
    try:
        if args.verb == 'state':
            snap = node.wait_state(args.timeout)
            if snap is None:
                print('no cell is publishing /cell_io/state (is cell_io_node, or the Isaac stage runner, running in '
                      'this ROS domain?)')
                return EXIT_NO_CELL
            print(json.dumps(snap, sort_keys=True, indent=2) if args.json else describe(snap))
            return EXIT_OK
        if args.verb == 'watch':
            last = None
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                if node.snapshot is not None:
                    text = describe(node.snapshot)
                    if text != last:
                        print(text, flush=True)
                        last = text
            return EXIT_OK
        snap = node.send(command, args.timeout)
        if snap is None:
            print('no answer from the cell (is cell_io_node, or the Isaac stage runner, running in this ROS domain?)')
            return EXIT_NO_CELL
        last = snap.get('last_command') or {}
        print(f"{'OK' if last.get('ok') else 'REFUSED'}: {last.get('text')}")
        return EXIT_OK if last.get('ok') else EXIT_REFUSED
    except (KeyboardInterrupt, ExternalShutdownException):
        return EXIT_OK
    except Exception:  # noqa: BLE001 -- a stop signal can invalidate the context mid-wait
        if rclpy.ok():
            raise
        return EXIT_OK
    finally:
        shutdown(node)


if __name__ == '__main__':
    sys.exit(main())
