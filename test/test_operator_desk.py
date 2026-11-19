"""U level: the cell's operator surface (trainit_cell_io.core.operator_desk), without ROS.

A command is executed at most once per id (a client may re-send a command it got no answer for); a malformed or
refused command is answered `ok: false` with the reason and never raises; the state carries the last answer.
"""

import copy
import json
import os
import random
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from trainit_cell_io.core import Cell, MockPlantAdapter, parse  # noqa: E402
from trainit_cell_io.core.operator_desk import OperatorDesk  # noqa: E402

EXAMPLE = os.path.join(ROOT, 'examples', 'neutral_station', 'cell_io_mock.yaml')


def _desk():
    with open(EXAMPLE, encoding='utf-8') as handle:
        cfg = parse(copy.deepcopy(yaml.safe_load(handle)))
    cell = Cell(cfg, MockPlantAdapter(cfg.mock), rng=random.Random(3))
    return OperatorDesk(cell), cell


def test_a_command_is_executed_once_per_id_and_a_repeat_gets_the_same_answer():
    desk, cell = _desk()
    text = json.dumps({'cmd': 'reset', 'service': 'turner', 'id': 'a1'})
    answer, repeated = desk.handle(text)
    assert answer == {'id': 'a1', 'cmd': 'reset', 'ok': True, 'text': 'turner: RESET (homing)'} and not repeated
    again, repeated = desk.handle(text)
    assert again == answer and repeated
    assert cell.services['turner'].counters.resets == 1            # executed once
    desk.handle(json.dumps({'cmd': 'reset', 'service': 'turner', 'id': 'a2'}))
    assert cell.services['turner'].counters.resets == 2            # another id: executed


def test_malformed_and_refused_commands_are_answered_never_raised():
    desk, _ = _desk()
    for text, why in (('nope', 'not JSON'), ('[1]', 'a JSON object with "cmd"'), ('{"cmd": 3}', '"cmd"'),
                      ('{"cmd": "start_cycle", "id": "x"}', 'unknown cell command'),
                      ('{"cmd": "inject", "fault": "melt", "id": "y"}', 'unknown fault'),
                      ('{"cmd": "reset", "service": ["turner"], "id": "z"}', 'no service')):
        answer, _ = desk.handle(text)
        assert answer['ok'] is False and why in answer['text'], (text, answer)
        assert desk.state()['last_command'] == answer


def test_an_unexpected_error_in_a_command_is_refused_with_its_type():
    desk, cell = _desk()

    def broken(_fault):
        raise RuntimeError('boom')
    cell.plant.inject = broken
    answer, _ = desk.handle(json.dumps({'cmd': 'inject', 'fault': 'clear', 'id': 'q'}))
    assert answer['ok'] is False and 'RuntimeError: boom' in answer['text']


def test_the_state_is_the_cell_snapshot_plus_the_last_answer_and_ids_are_bounded():
    desk, _ = _desk()
    for n in range(80):
        desk.handle(json.dumps({'cmd': 'inject', 'fault': 'clear', 'id': f'id{n}'}))
    assert len(desk._answers) == 64 and 'id0' not in desk._answers and 'id79' in desk._answers   # noqa: SLF001
    state = json.loads(desk.state_json())
    assert set(state) >= {'plant', 'services', 'devices', 'last_command'}
    assert state['last_command']['id'] == 'id79' and state['services']['turner']['state'] == 'READY'
