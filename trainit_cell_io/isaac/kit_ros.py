"""The Kit side's ROS transport (P0B_DESIGN.md §4; F-A, A-B2, C5): std_msgs and rosgraph_msgs ONLY.

Inside Isaac Sim this runs on the bridge extension's bundled rclpy (Python 3.11), which carries the standard message
packages but no `trainit_interfaces`: the handshake images cross as `std_msgs/String` JSON (`core/kit_boundary.py`)
and `cell_io_bridge`, on the ROS side, publishes the canonical types. The same code runs on the system rclpy, which
is how it is tested outside Isaac.

    /cell_io/kit/<service>/status   String JSON  published (RELIABLE, TRANSIENT_LOCAL, depth 1)
    /cell_io/kit/<service>/request  String JSON  subscribed (every request image of every requester, from
                                                 cell_io_bridge; a deeper reader queue: none replaces another unseen)
    /cell_io/control                String JSON  subscribed (the operator's commands, VOLATILE)
    /cell_io/state                  String JSON  published (the operator's view, latched)
    /clock                          rosgraph_msgs/Clock, after every physics step (the robot's runner uses it with
                                    use_sim_time:=true)

rclpy is imported when a `KitRos` is built, never at module level (the Kit runner enables the bridge extension first).
"""

from __future__ import annotations

from typing import Callable, Iterable

from ..core.kit_boundary import kit_topic

CONTROL_TOPIC = '/cell_io/control'
STATE_TOPIC = '/cell_io/state'
CLOCK_TOPIC = '/clock'
NS = 10 ** 9


class KitRos:
    def __init__(self, services: Iterable[str], on_request: Callable[[str, str], None],
                 on_command: Callable[[str], None], context=None, node_name: str = 'cell_io_stage',
                 max_callbacks_per_spin: int = 20):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from rosgraph_msgs.msg import Clock
        from std_msgs.msg import String
        self._String = String
        self._Clock = Clock
        latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        commands = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        # every forwarded request image reaches the cell, in order (implementation review fix C2): the bridge's writer
        # and this reader keep up to 20, so a burst (two requesters, a re-forward) never replaces an image unseen
        requests = QoSProfile(depth=20, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node = rclpy.create_node(node_name, context=context)
        self._status = {}
        self._calls = 0
        self._max = max_callbacks_per_spin
        for sid in services:
            self._status[sid] = self.node.create_publisher(String, kit_topic(sid, 'status'), latched)
            self.node.create_subscription(String, kit_topic(sid, 'request'),
                                          lambda msg, s=sid: self._call(on_request, s, msg.data), requests)
        self.node.create_subscription(String, CONTROL_TOPIC, lambda msg: self._call(on_command, msg.data), commands)
        self._state = self.node.create_publisher(String, STATE_TOPIC, latched)
        self._clock = self.node.create_publisher(Clock, CLOCK_TOPIC, 10)
        self._executor = SingleThreadedExecutor(context=context)
        self._executor.add_node(self.node)

    def _call(self, fn: Callable, *args) -> None:
        self._calls += 1
        fn(*args)

    def publish_status(self, service: str, text: str) -> None:
        self._status[service].publish(self._String(data=text))

    def publish_clock(self, stamp_ns: int) -> None:
        msg = self._Clock()
        msg.clock.sec, msg.clock.nanosec = stamp_ns // NS, stamp_ns % NS
        self._clock.publish(msg)

    def publish_state(self, text: str) -> None:
        self._state.publish(self._String(data=text))

    def spin(self) -> None:
        """Deliver every callback ready now (bounded), without waiting."""
        for _ in range(self._max):
            before = self._calls
            self._executor.spin_once(timeout_sec=0.0)
            if self._calls == before:
                break

    def close(self) -> None:
        try:
            self._executor.remove_node(self.node)
            self.node.destroy_node()
            self._executor.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: BLE001 -- teardown after a signal may find the context gone
            pass
