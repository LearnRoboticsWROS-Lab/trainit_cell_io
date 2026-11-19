"""The simulator boundary: the handshake images as std_msgs/String JSON (P0B_DESIGN.md §4; F-A, A-B2, case C5).

Isaac Sim 5.1's bridge rclpy is Python 3.11 and carries the standard message packages only: a `trainit_interfaces`
type is not importable inside Kit. So the Kit side of the Isaac plant speaks JSON in `std_msgs/String`, and the
ROS-side `cell_io_bridge` translates to and from the canonical `ChannelRequest` / `ChannelStatus`. The heartbeat and
the session are minted by the scan INSIDE Kit and carried through this boundary -- the bridge never mints them, so a
dead Kit behind a living bridge freezes the heartbeat and the robot sees the source as not live (PB-P9).

Kit topics (cell level, never in a robot slot):
    /cell_io/kit/<service>/status    Kit -> bridge, every status period and on change
    /cell_io/kit/<service>/request   bridge -> Kit, every request image of every requester, one image per message
                                     (TRANSIENT_LOCAL; the requester count with each)
    /cell_io/control                 the operator's commands (the CLI), JSON (also served by the mock cell)
    /cell_io/state                   the operator's view, JSON (also published by the mock cell)
"""

from __future__ import annotations

import json
from typing import Mapping, Optional, Tuple

from .contract import (FAULT_NAMES, REJECT_NAMES, STATE_NAMES, UINT16_MASK, UINT32_MAX, RequestImage,
                       StatusImage)

BOUNDARY_VERSION = 1


class BoundaryError(ValueError):
    """A boundary message the other side refuses (never half-applied)."""


def kit_topic(service: str, leaf: str) -> str:
    return f'/cell_io/kit/{service}/{leaf}'


def encode_status(service: str, image: StatusImage, stamp_ns: int) -> str:
    return json.dumps({'v': BOUNDARY_VERSION, 'service': service, 'stamp_ns': int(stamp_ns), **image.as_dict()},
                      sort_keys=True)


def decode_status(text: str) -> Tuple[str, StatusImage, int]:
    data = _load(text)
    service = _str(data, 'service')
    image = StatusImage(
        source_session=_uint(data, 'source_session', UINT32_MAX),
        heartbeat=_uint(data, 'heartbeat', UINT16_MASK),
        state=_enum(data, 'state', STATE_NAMES),
        cycle_requester=_uint(data, 'cycle_requester', UINT32_MAX),
        cycle_seq=_uint(data, 'cycle_seq', UINT32_MAX),
        cycle_done=_bool(data, 'cycle_done'),
        reject_requester=_uint(data, 'reject_requester', UINT32_MAX),
        reject_seq=_uint(data, 'reject_seq', UINT32_MAX),
        reject_code=_enum(data, 'reject_code', REJECT_NAMES),
        fault_code=_enum(data, 'fault_code', FAULT_NAMES),
        detail=_str(data, 'detail', allow_empty=True),
    )
    if image.source_session == 0:
        raise BoundaryError('status: source_session 0 (a source always has a session)')
    return service, image, _uint(data, 'stamp_ns', 2 ** 63 - 1)


def encode_request(service: str, image: RequestImage, requesters: int) -> str:
    return json.dumps({'v': BOUNDARY_VERSION, 'service': service, 'source_session': image.source_session,
                       'requester_session': image.requester_session, 'seq': image.seq,
                       'requesters': int(requesters)}, sort_keys=True)


def decode_request(text: str) -> Tuple[str, RequestImage, int]:
    data = _load(text)
    service = _str(data, 'service')
    image = RequestImage(source_session=_uint(data, 'source_session', UINT32_MAX),
                         requester_session=_uint(data, 'requester_session', UINT32_MAX),
                         seq=_uint(data, 'seq', UINT32_MAX))
    return service, image, _uint(data, 'requesters', 1000)


def encode_command(command: Mapping[str, object]) -> str:
    return json.dumps(dict(command), sort_keys=True)


def decode_command(text: str) -> dict:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise BoundaryError(f'control: not JSON: {exc}') from exc
    if not isinstance(data, dict) or not isinstance(data.get('cmd'), str):
        raise BoundaryError('control: a JSON object with "cmd"')
    return data


def _load(text: str) -> dict:
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise BoundaryError(f'not JSON: {exc}') from exc
    if not isinstance(data, dict):
        raise BoundaryError('a JSON object')
    if data.get('v') != BOUNDARY_VERSION:
        raise BoundaryError(f'boundary version {data.get("v")!r}, expected {BOUNDARY_VERSION}')
    return data


def _uint(data: Mapping, key: str, maximum: int) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise BoundaryError(f'{key}: an integer in 0..{maximum}, got {value!r}')
    return value


def _enum(data: Mapping, key: str, names: Mapping[int, str]) -> int:
    value = _uint(data, key, 0xFFFF)
    if value not in names:
        raise BoundaryError(f'{key}: {value} is not one of {sorted(names)}')
    return value


def _bool(data: Mapping, key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise BoundaryError(f'{key}: true or false, got {value!r}')
    return value


def _str(data: Mapping, key: str, allow_empty: bool = False) -> str:
    value = data.get(key)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise BoundaryError(f'{key}: a {"" if allow_empty else "non-empty "}string, got {value!r}')
    return value


def optional_request(text: Optional[str]) -> Optional[Tuple[str, RequestImage, int]]:
    return None if text is None else decode_request(text)
