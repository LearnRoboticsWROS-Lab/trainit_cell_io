"""`cell_io_node` -- the MOCK plant's cell process (P0B_DESIGN.md §4).

    ros2 launch trainit_cell_io cell_io.launch.py config:=<cell_io.yaml>
    ros2 run trainit_cell_io cell_io_node --ros-args -p config:=<cell_io.yaml>

A cell-level process, started apart from the robot: the robot bundle never launches it (ADR-0020). It runs the cell's
station logic against the deterministic `MockPlantAdapter` on wall time and serves every machine service of
`cell_io.yaml` on the canonical topics of the robot's slot:

    <slot>/<service>/request   trainit_interfaces/ChannelRequest   (subscribed; RELIABLE, TRANSIENT_LOCAL; every
                                                                     image of every requester handed to the cell)
    <slot>/<service>/status    trainit_interfaces/ChannelStatus    (published every period and at once on a change)

and the cell's own operator surface (never a robot command):

    /cell_io/control   std_msgs/String JSON   {"cmd": "reset", "service": ...} | {"cmd": "inject", "fault": ...}
    /cell_io/state     std_msgs/String JSON   every service's state and counters, the devices, the last answer

The topics are absolute: the cell is a CELL-level process that serves a robot's slot (ADR-0023 governs robot-scoped
runtime code; the slot comes from `robot_slot` in `cell_io.yaml`). The robot's runner, in its slot, uses the
relative names.

Exit codes: 0 after an orderly stop (SIGINT / SIGTERM) or for a valid `control_mode: external_plc` file (nothing to
simulate); 2 when the file is refused (C3: `external_plc` with a ProcessDevice or a logic backend), missing, or names
a plant this node does not serve (`plant: isaac` runs inside Isaac Sim, with `cell_io_bridge`). Not a PLC, not a
safety function (ADR-0033).
"""

from __future__ import annotations

import sys
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from trainit_interfaces.msg import ChannelRequest, ChannelStatus

from ..core.cell import Cell
from ..core.clock import CatchUp
from ..core.config import CellConfig, ConfigError, load
from ..core.mock_plant import MockPlantAdapter
from ..core.operator_desk import OperatorDesk
from ..core.plant import InjectionError
from .convert import request_from_msg, status_to_msg
from .qos import COMMANDS, CONTROL_TOPIC, LATCHED, REQUEST_READER, STATE_TOPIC
from .runtime import shutdown, spin

EXIT_REFUSED = 2
MAX_CATCH_UP_S = 1.0              # warn (once) when the cell is this far behind wall time
MAX_STEPS_PER_CALLBACK = 200      # one scan timer callback runs at most this many physics steps (1 s at dt 5 ms)
STATE_PERIOD_S = 1.0
READY_TEXT = 'cell_io: serving'   # the ready line starts with this (a launch tier waits for it)


class CellIoNode(Node):
    def __init__(self, cell: Cell):
        super().__init__('cell_io')
        self.cell = cell
        self.cfg: CellConfig = cell.config
        self.desk = OperatorDesk(cell)
        self._status_pub: Dict[str, object] = {}
        self._request_topic: Dict[str, str] = {}
        self._requesters: Dict[str, int] = {}
        for sid in cell.services:
            status_topic = self.cfg.topic(sid, 'status')
            request_topic = self.cfg.topic(sid, 'request')
            self._status_pub[sid] = self.create_publisher(ChannelStatus, status_topic, LATCHED)
            self._request_topic[sid] = request_topic
            self._requesters[sid] = 0
            self.create_subscription(ChannelRequest, request_topic,
                                     lambda msg, s=sid: self._on_request(s, msg), REQUEST_READER)
            self.get_logger().info(f"machine service '{sid}': status on {status_topic}, requests on {request_topic} "
                                   f"(source session {cell.services[sid].session:#010x})")
        self._state_pub = self.create_publisher(String, STATE_TOPIC, LATCHED)
        self.create_subscription(String, CONTROL_TOPIC, self._on_control, COMMANDS)
        self._log_index = 0               # the running total of the cell's log lines already printed
        clock = self.cfg.clock
        self._dt_s = float(clock.dt)
        self._period_s = clock.seconds(clock.status_period_steps)
        self._t0 = time.monotonic()
        self._catch_up = CatchUp(self._dt_s, self._t0, MAX_STEPS_PER_CALLBACK)
        self._next_count = self._t0
        self._behind_warned = False
        self.create_timer(clock.seconds(clock.scan_every_steps), self._on_scan_timer)
        self.create_timer(STATE_PERIOD_S, self._publish_state)
        self.get_logger().info(
            f'{READY_TEXT} {len(cell.services)} machine service(s) ({", ".join(cell.services)}) on the mock plant '
            f'(seed {cell.plant.seed}): physics dt {clock.dt_ms_text()} ms, scan every {clock.scan_every_steps} '
            f'step(s), status period {clock.ms(clock.status_period_steps):g} ms -- ready')

    # --- inputs -------------------------------------------------------------------------------------------------------
    def _count(self, sid: str) -> int:
        """Publishers on `<service>/request` (rule 6), counted on the graph; the change is logged once."""
        count = self.count_publishers(self._request_topic[sid])
        if count != self._requesters[sid]:
            self.get_logger().info(f"machine service '{sid}': {count} requester(s) publish on "
                                   f'{self._request_topic[sid]}'
                                   + (' -- every new request is REJECTED TWO_REQUESTERS' if count > 1 else ''))
            self._requesters[sid] = count
        return max(1, count)

    def _on_request(self, sid: str, msg: ChannelRequest) -> None:
        """EVERY image, as it comes: the cell keeps each requester's latest and consumes one new identity per scan."""
        self.cell.set_request(sid, request_from_msg(msg), self._count(sid))

    def _on_control(self, msg: String) -> None:
        answer, repeated = self.desk.handle(msg.data)
        if not repeated:
            # one call site per severity: rclpy refuses a severity change at the same call site
            if answer['ok']:
                self.get_logger().info(f"cell command done: {answer['text']}")
            else:
                self.get_logger().warning(f"cell command REFUSED: {answer['text']}")
        self._publish_state()

    # --- the clock: whole physics steps, caught up to wall time (at most MAX_STEPS_PER_CALLBACK per callback) ---------
    def _on_scan_timer(self) -> None:
        now = time.monotonic()
        if now >= self._next_count:
            self._next_count = now + self._period_s
            for sid in self.cell.services:
                self.cell.set_requesters(sid, self._count(sid))
        run, behind = self._catch_up.due(now, self.cell.steps)
        if behind * self._dt_s > MAX_CATCH_UP_S and not self._behind_warned:
            self.get_logger().warning(f'the mock plant fell {behind * self._dt_s:.2f} s behind wall time '
                                      f'(machine overloaded?) -- catching up, at most {MAX_STEPS_PER_CALLBACK} '
                                      'physics steps per scan timer (no step is skipped)')
            self._behind_warned = True
        for _ in range(run):
            for sid, image in self.cell.step():
                self._status_pub[sid].publish(status_to_msg(image, self.get_clock().now().to_msg()))
        lines, self._log_index, dropped = self.cell.log.since(self._log_index)     # each new line exactly once
        if dropped:
            self.get_logger().warning(f'{dropped} cell log line(s) were dropped before they could be printed (the '
                                      f'log keeps the last {self.cell.log.maxlen})')
        for step, text in lines:
            self.get_logger().info(f'[{self.cfg.clock.seconds(step):8.2f} s] {text}')
        if lines or dropped:
            self._publish_state()                         # the operator's view is current, not up to 1 s old

    def _publish_state(self) -> None:
        self._state_pub.publish(String(data=self.desk.state_json()))


def _config_path() -> str:
    probe = rclpy.create_node('cell_io_config')
    try:
        probe.declare_parameter('config', '')
        return probe.get_parameter('config').get_parameter_value().string_value
    finally:
        probe.destroy_node()


def _refuse(text: str) -> int:
    print(f'cell_io_node: {text}', file=sys.stderr, flush=True)
    if rclpy.ok():
        rclpy.shutdown()
    return EXIT_REFUSED


def main(argv=None) -> int:
    rclpy.init(args=argv)          # its signal handlers turn SIGINT and SIGTERM into an orderly stop (D11)
    path = _config_path()
    if not path:
        return _refuse('the parameter config:=<cell_io.yaml> is required')
    try:
        cfg = load(path)
    except ConfigError as exc:
        return _refuse(f'refused: {exc}')
    if cfg.control_mode == 'external_plc':
        print('cell_io_node: control_mode external_plc -- the machine logic runs on the PLC; trainit_cell_io '
              'simulates nothing (the PLC adapter is P0.c)', file=sys.stderr, flush=True)
        rclpy.shutdown()
        return 0
    if cfg.plant != 'mock':
        return _refuse(f'plant {cfg.plant!r} is not served by this node: the Isaac plant runs inside Isaac Sim (the '
                       'stage runner) with cell_io_bridge (cell_io_bridge.launch.py)')
    try:
        cell = Cell(cfg, MockPlantAdapter(cfg.mock))
    except (ConfigError, InjectionError) as exc:
        return _refuse(f'refused: {exc}')
    node: Optional[CellIoNode] = None
    try:
        node = CellIoNode(cell)
        spin(node)
    except KeyboardInterrupt:        # a stop signal before spinning started: still an orderly stop
        pass
    finally:
        shutdown(node)
    return 0


if __name__ == '__main__':
    sys.exit(main())
