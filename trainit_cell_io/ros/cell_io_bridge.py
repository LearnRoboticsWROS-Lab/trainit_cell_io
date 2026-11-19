"""`cell_io_bridge` -- the ROS side of the Isaac plant (P0B_DESIGN.md §4; cases C5, PB-P9, PB-P21).

    ros2 launch trainit_cell_io cell_io_bridge.launch.py config:=<cell_io.yaml with plant: isaac>
    ros2 run trainit_cell_io cell_io_bridge --ros-args -p config:=<cell_io.yaml>

Isaac Sim's bundled rclpy carries the standard message packages only (no `trainit_interfaces`), so the Kit side of the
Isaac plant -- the station logic, its scan, its `ProcessDevice`s -- speaks JSON in `std_msgs/String` on cell-level
topics, and this bridge translates, per machine service of the cell file:

    /cell_io/kit/<service>/status (String JSON, from Kit) -> <slot>/<service>/status (trainit_interfaces/ChannelStatus)
    <slot>/<service>/request (trainit_interfaces/ChannelRequest) -> /cell_io/kit/<service>/request (String JSON, to Kit)

What it NEVER does: mint a heartbeat or a source session, or republish a status on its own timer. It forwards each Kit
image as it arrives, field for field (the header stamp is the Kit's clock), so when Kit dies the heartbeat freezes and
the robot sees the source as not live (PB-P9) although this bridge is still up. It refuses a malformed Kit image with a
log line and publishes nothing for it (never half-applied).

Requests (implementation review fix C2): it forwards EVERY request image it receives, of every requester, one image
per boundary message, and keeps the latest image of each requester (the last 16 requesters) -- never one image per
service that a second requester's image would displace unforwarded. It counts the publishers of
`<slot>/<service>/request` (the source's rule 6: more than one requester -> every new request REJECTED TWO_REQUESTERS)
every status period and on each request, and when the count changes it re-forwards the latest image of EACH requester
with the new count. A re-forwarded image whose identity the source consumed already is a no-op there (rule 2, per
requester): a restarted bridge forwards the robot's level-held image again and nothing runs twice (PB-P21).

Exit codes: 0 after an orderly stop (SIGINT / SIGTERM) or for a valid `control_mode: external_plc` file (nothing to
bridge); 2 when the file is refused, missing, or names the mock plant (served by `cell_io_node`). Not a safety function
(ADR-0033).
"""

from __future__ import annotations

import sys
import time
from collections import OrderedDict
from typing import Dict, Optional

import rclpy
from builtin_interfaces.msg import Time
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from trainit_interfaces.msg import ChannelRequest, ChannelStatus

from ..core.config import CellConfig, ConfigError, load
from ..core.contract import STATE_NAMES, RequestImage
from ..core.kit_boundary import BoundaryError, decode_status, encode_request, kit_topic
from ..core.process_device import REMEMBERED_REQUESTERS
from .convert import request_from_msg, status_to_msg
from .qos import LATCHED, REQUEST_READER
from .runtime import shutdown, spin

EXIT_REFUSED = 2
COUNT_PERIOD_S = 0.1
KIT_SILENT_S = 1.0
READY_TEXT = 'cell_io_bridge: bridging'   # the ready line starts with this

# Kit -> bridge: the writer is latched (depth 1); a deeper reader queue so the bridge forwards EVERY image in order.
KIT_STATUS_READER = QoSProfile(depth=20, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
# bridge -> Kit: a burst of forwarded images (two requesters, a re-forward of each requester's latest) must not replace
# one another in the writer's history before Kit took them; a late-joining (restarted) Kit receives the last ones too.
KIT_REQUEST_WRITER = QoSProfile(depth=20, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)


INT32_MAX = 2 ** 31 - 1


def stamp_from_ns(ns: int) -> Time:
    """A ROS stamp from the stage's integer nanoseconds; refused (BoundaryError) beyond the int32 seconds of Time."""
    if not 0 <= ns // 1_000_000_000 <= INT32_MAX:
        raise BoundaryError(f'stamp_ns {ns} is beyond the range of a ROS time')
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class CellIoBridge(Node):
    def __init__(self, cfg: CellConfig):
        super().__init__('cell_io_bridge')
        self.cfg = cfg
        self._status_pub: Dict[str, object] = {}
        self._kit_request_pub: Dict[str, object] = {}
        self._request_topic: Dict[str, str] = {}
        # per service: requester_session -> its latest request image (least recently received first; bounded)
        self._latest: Dict[str, 'OrderedDict[int, RequestImage]'] = {}
        self._requesters: Dict[str, int] = {}
        self._forwarded_with: Dict[str, int] = {}
        self._last_kit: Dict[str, Optional[float]] = {}
        self._silent: Dict[str, bool] = {}
        self._last_state: Dict[str, Optional[int]] = {}
        self.forwarded: Dict[str, int] = {}
        self.refused: Dict[str, int] = {}
        for sid in cfg.services:
            status_topic, request_topic = cfg.topic(sid, 'status'), cfg.topic(sid, 'request')
            self._status_pub[sid] = self.create_publisher(ChannelStatus, status_topic, LATCHED)
            self._kit_request_pub[sid] = self.create_publisher(String, kit_topic(sid, 'request'), KIT_REQUEST_WRITER)
            self._request_topic[sid] = request_topic
            self._latest[sid] = OrderedDict()
            self._requesters[sid] = 0
            self._forwarded_with[sid] = 0
            self._last_kit[sid] = None
            self._silent[sid] = False
            self._last_state[sid] = None
            self.forwarded[sid] = 0
            self.refused[sid] = 0
            self.create_subscription(String, kit_topic(sid, 'status'),
                                     lambda msg, s=sid: self._on_kit_status(s, msg), KIT_STATUS_READER)
            self.create_subscription(ChannelRequest, request_topic, lambda msg, s=sid: self._on_request(s, msg),
                                     REQUEST_READER)
            self.get_logger().info(f"machine service '{sid}': {kit_topic(sid, 'status')} -> {status_topic}; "
                                   f"{request_topic} -> {kit_topic(sid, 'request')}")
        self._t0 = time.monotonic()
        self.create_timer(COUNT_PERIOD_S, self._on_count_timer)
        self.create_timer(0.5, self._on_silence_timer)
        self.get_logger().info(
            f'{READY_TEXT} {len(cfg.services)} machine service(s) ({", ".join(cfg.services)}) between the Isaac '
            'stage and the robot slot ' + (f"'{cfg.robot_slot}'" if cfg.robot_slot else '(root namespace)')
            + ' -- ready; it never mints a heartbeat: the status comes from the stage or not at all')

    # --- Kit -> the robot: forward each image, never mint, never republish ------------------------------------------
    def _on_kit_status(self, sid: str, msg: String) -> None:
        try:
            service, image, stamp_ns = decode_status(msg.data)
            if service != sid:
                raise BoundaryError(f"the image names service {service!r} on the topic of '{sid}'")
            out = status_to_msg(image, stamp_from_ns(stamp_ns))
        except BoundaryError as exc:
            self.refused[sid] += 1
            self.get_logger().warning(f"refused a malformed status image from the Isaac stage for '{sid}' (nothing "
                                      f'forwarded): {exc}')
            return
        self._status_pub[sid].publish(out)
        self.forwarded[sid] += 1
        self._last_kit[sid] = time.monotonic()
        if self._last_state[sid] != image.state:
            self._last_state[sid] = image.state
            self.get_logger().info(f"'{sid}': {STATE_NAMES.get(image.state, image.state)} -- {image.detail} "
                                   f'(source session {image.source_session:#010x})')

    def _on_silence_timer(self) -> None:
        now = time.monotonic()
        for sid in self.cfg.services:
            last = self._last_kit[sid]
            silent = (now - (last if last is not None else self._t0)) > KIT_SILENT_S
            if silent and not self._silent[sid]:
                self.get_logger().warning(
                    f"no status image from the Isaac stage for '{sid}' for more than {KIT_SILENT_S:g} s (is the "
                    'stage running, on this ROS domain?) -- nothing is published: the robot sees the source as '
                    'not live')
            elif not silent and self._silent[sid]:
                self.get_logger().info(f"'{sid}': status images from the Isaac stage again")
            self._silent[sid] = silent

    # --- the robot -> Kit: every request image of every requester, with the requester count -------------------------
    def _count(self, sid: str) -> int:
        count = self.count_publishers(self._request_topic[sid])
        if count != self._requesters[sid]:
            self.get_logger().info(f"'{sid}': {count} requester(s) publish on {self._request_topic[sid]}"
                                   + (' -- the source rejects every new request TWO_REQUESTERS' if count > 1 else ''))
            self._requesters[sid] = count
        return count

    def _on_request(self, sid: str, msg: ChannelRequest) -> None:
        image = request_from_msg(msg)
        latest = self._latest[sid]
        latest[image.requester_session] = image
        latest.move_to_end(image.requester_session)
        while len(latest) > REMEMBERED_REQUESTERS:
            latest.popitem(last=False)
        self._count(sid)
        if max(1, self._requesters[sid]) != self._forwarded_with[sid]:
            self._forward_all(sid)        # the count changed: every requester's latest image (this one last)
        else:
            self._forward(sid, image)

    def _forward(self, sid: str, image: RequestImage) -> None:
        requesters = max(1, self._requesters[sid])
        self._kit_request_pub[sid].publish(String(data=encode_request(sid, image, requesters)))
        self._forwarded_with[sid] = requesters

    def _forward_all(self, sid: str) -> None:
        for image in list(self._latest[sid].values()):
            self._forward(sid, image)

    def _on_count_timer(self) -> None:
        for sid in self.cfg.services:
            self._count(sid)
            if self._latest[sid] and max(1, self._requesters[sid]) != self._forwarded_with[sid]:
                self._forward_all(sid)    # the count changed: the source must know before the next new request


def _config_path() -> str:
    probe = rclpy.create_node('cell_io_bridge_config')
    try:
        probe.declare_parameter('config', '')
        return probe.get_parameter('config').get_parameter_value().string_value
    finally:
        probe.destroy_node()


def _refuse(text: str) -> int:
    print(f'cell_io_bridge: {text}', file=sys.stderr, flush=True)
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
        print('cell_io_bridge: control_mode external_plc -- no simulated plant, nothing to bridge', file=sys.stderr,
              flush=True)
        rclpy.shutdown()
        return 0
    if cfg.plant != 'isaac':
        return _refuse(f"plant {cfg.plant!r}: this bridge serves the Isaac plant; the mock plant is served by "
                       'cell_io_node (cell_io.launch.py)')
    node: Optional[CellIoBridge] = None
    try:
        node = CellIoBridge(cfg)
        spin(node)
    except KeyboardInterrupt:        # a stop signal before spinning started: still an orderly stop
        pass
    finally:
        shutdown(node)
    return 0


if __name__ == '__main__':
    sys.exit(main())
