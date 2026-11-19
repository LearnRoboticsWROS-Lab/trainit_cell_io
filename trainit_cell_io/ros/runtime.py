"""Process plumbing shared by the cell's ROS executables: an orderly stop on SIGINT / SIGTERM (D11)."""

from __future__ import annotations

import signal

import rclpy
from rclpy.executors import ExternalShutdownException


def ignore_further_stop_signals() -> None:
    """The stop has begun: a SECOND SIGINT / SIGTERM must not interrupt the cleanup. Ctrl-C reaches the whole
    foreground process group and `ros2 launch` forwards its own SIGINT too, so a launched node gets two."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, signal.SIG_IGN)
        except (ValueError, OSError):      # not the main thread: nothing to do
            pass


def spin(node) -> None:
    """`rclpy.spin(node)` until SIGINT or SIGTERM; returns normally on that orderly stop.

    rclpy's signal handlers shut the context down; the executor then raises `ExternalShutdownException`, or -- when the
    signal lands while it builds a wait set -- an `RCLError` on the invalid context. Both are the orderly stop. Any
    error while the context is still valid is a real failure and propagates. From here on, stop signals are ignored.
    """
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001 -- re-raised unless the context was shut down by the stop signal
        if rclpy.ok():
            raise
    finally:
        ignore_further_stop_signals()


def shutdown(node=None) -> None:
    """Destroy the node and shut rclpy down; never raises (a late stop signal included)."""
    ignore_further_stop_signals()
    try:
        if node is not None:
            node.destroy_node()
    except BaseException:  # noqa: BLE001 -- the context may be gone after a signal; the process is stopping anyway
        pass
    try:
        if rclpy.ok():
            rclpy.shutdown()
    except BaseException:  # noqa: BLE001
        pass
