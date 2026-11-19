"""The cell's operator surface, transport-free (P0B_DESIGN.md §4): commands in, answers and the state out.

The operator (or a machine master in simulation control) talks to the cell with JSON commands -- never the robot:

    {"cmd": "reset", "service": "turner", "id": "<unique>"}
    {"cmd": "inject", "fault": "stuck_actuator", "device": "clamp", "id": "<unique>"}
    {"cmd": "restart", "id": "<unique>"}      (simulation: the controller restarts in place -- new source sessions)

and reads the cell's state (every service's state, counters, devices, injections, the last command's answer). The mock
cell (`cell_io_node`) and the Kit side of the Isaac plant serve the SAME surface on `/cell_io/control` and
`/cell_io/state`, so `cell_io_cli` works against either.

A command is executed AT MOST ONCE per `id`: a client that got no answer may re-send the same command (a VOLATILE
command topic can lose a sample while the endpoints are still matching), and a repeated id is answered again, never
executed twice. A malformed or refused command is answered `ok: false` with the reason; nothing raises out of here.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from .cell import Cell, ControlError
from .kit_boundary import BoundaryError, decode_command

REMEMBERED_IDS = 64


class OperatorDesk:
    def __init__(self, cell: Cell, remember: int = REMEMBERED_IDS):
        self.cell = cell
        self.last_command: Optional[Dict[str, object]] = None
        self._answers: 'OrderedDict[str, Dict[str, object]]' = OrderedDict()
        self._remember = remember

    def handle(self, text: str) -> Tuple[Dict[str, object], bool]:
        """Execute one JSON command (at most once per id); returns (the answer, whether it was a repeat)."""
        try:
            command = decode_command(text)
        except BoundaryError as exc:
            answer = {'id': None, 'cmd': None, 'ok': False, 'text': str(exc)}
            self.last_command = answer
            return answer, False
        cid = command.get('id')
        cid = cid if isinstance(cid, str) and cid else None
        if cid is not None and cid in self._answers:
            answer = self._answers[cid]
            self.last_command = answer
            return answer, True
        body = {k: v for k, v in command.items() if k != 'id'}
        try:
            reply = self.cell.control(body)
            ok = True
        except ControlError as exc:
            reply, ok = str(exc), False
        except Exception as exc:  # noqa: BLE001 -- an operator command is refused, it never stops the cell
            reply, ok = f'malformed command ({type(exc).__name__}: {exc})', False
        answer = {'id': cid, 'cmd': command.get('cmd'), 'ok': ok, 'text': reply}
        self.last_command = answer
        if cid is not None:
            self._answers[cid] = answer
            while len(self._answers) > self._remember:
                self._answers.popitem(last=False)
        return answer, False

    def state(self) -> Dict[str, object]:
        snapshot = self.cell.snapshot()
        snapshot['last_command'] = self.last_command
        return snapshot

    def state_json(self) -> str:
        return json.dumps(self.state(), sort_keys=True)
