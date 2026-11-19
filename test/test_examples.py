"""U level: the PUBLIC neutral fixture's files (P0B_DESIGN.md §5; D7; P1's cell-side half).

cell_io_mock.yaml and cell_io_isaac.yaml differ in exactly ONE line, `plant:` -- the same station logic, devices,
positions and timeouts on both plants; both load and bind (the mock time model, the Isaac joints).
"""

import difflib
import os
import random
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from trainit_cell_io.core import Cell, MockPlantAdapter, load  # noqa: E402
from trainit_cell_io.isaac.plant import IsaacPlantAdapter  # noqa: E402

EX = os.path.join(ROOT, 'examples', 'neutral_station')
RUNNER = os.path.join(EX, 'run_neutral_station.py')


def test_the_isaac_cell_file_differs_from_the_mock_one_only_in_the_plant_line():
    with open(os.path.join(EX, 'cell_io_mock.yaml'), encoding='utf-8') as handle:
        mock = handle.read().splitlines()
    with open(os.path.join(EX, 'cell_io_isaac.yaml'), encoding='utf-8') as handle:
        isaac = handle.read().splitlines()
    changed = [line for line in difflib.unified_diff(mock, isaac, lineterm='', n=0)
               if line[:1] in '+-' and not line.startswith(('+++', '---'))]
    assert changed == ['-plant: mock', '+plant: isaac']


def test_both_cell_files_load_and_bind_their_plant():
    mock = load(os.path.join(EX, 'cell_io_mock.yaml'))
    isaac = load(os.path.join(EX, 'cell_io_isaac.yaml'))
    assert (mock.plant, isaac.plant) == ('mock', 'isaac')
    assert list(mock.services) == list(isaac.services) == ['turner', 'indexer']
    Cell(mock, MockPlantAdapter(mock.mock), rng=random.Random(1))
    plant = IsaacPlantAdapter(isaac.isaac)
    Cell(isaac, plant, rng=random.Random(1))                    # every device has its joint / body binding
    assert sorted(j for _, j in plant.joints().values()) == ['carriage_joint', 'jaw_joint', 'turntable_joint']
    assert isaac.devices['index'].positions == {'home': 0.0, 'forward': 300.0}      # D7: the 300 mm stroke


# --- the Kit runner refuses a bad cell file BEFORE Kit starts: exit 2, one line, never a traceback (review fix C4) ---

@pytest.mark.parametrize('old, new, why', [
    ('positions: {load: 0.0, turned: 180.0}', 'positions: {load: 0.0, turned: .nan}', 'must be a finite number'),
    ('    band: 1.0\n    timeout_ms: 4000\n    description: the turntable',
     '    band: .inf\n    timeout_ms: 4000\n    description: the turntable', 'finite number > 0'),
    ('- {close: clamp}', '- {close: [clamp]}', "unknown device ['clamp']"),
    ('status_period_ms: 100}', 'status_period_ms: 1000}', 'is above 500 ms'),
    ('\nplant: isaac\n', '\nplant: mock\n', 'needs control_mode simulation and plant: isaac'),
    ('stage: neutral_station', 'stage: another_stage', "isaac.stage must be 'neutral_station'"),
    ('band_m: 0.003', 'band_m: .inf', 'a finite number'),
])
def test_the_kit_runner_refuses_a_bad_cell_file_with_exit_2_before_kit(tmp_path, old, new, why):
    with open(os.path.join(EX, 'cell_io_isaac.yaml'), encoding='utf-8') as handle:
        text = handle.read()
    assert old in text, old
    path = tmp_path / 'cell_io.yaml'
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    env = {'HOME': os.environ.get('HOME', '/tmp'), 'PATH': '/usr/bin:/bin'}     # no ROS, no Isaac: refused first
    out = subprocess.run([sys.executable, RUNNER, '--config', str(path)], capture_output=True, text=True, env=env,
                         cwd=str(tmp_path), timeout=60)
    assert out.returncode == 2, (out.stdout, out.stderr)
    assert 'Traceback' not in out.stdout + out.stderr, out.stderr
    assert out.stderr.startswith('run_neutral_station: refused: ') and why in out.stderr, out.stderr
    assert out.stdout.strip().splitlines()[-1] == 'CELL_IO_STAGE_END rc=2', out.stdout


# --- the Kit runner's own stop handling (review fix C8: Python review MINOR 6) -- the pure helpers, without Kit -----

def _runner_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_neutral_station_under_test', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)                  # stdlib imports only at module level: no Kit, no ROS
    return module


def test_the_runners_stop_flag_turns_sigint_and_sigterm_into_a_flag_never_an_exception():
    import signal
    R = _runner_module()
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        stop = R.StopFlag().install()
        assert signal.getsignal(signal.SIGINT) is stop and signal.getsignal(signal.SIGTERM) is stop
        assert not stop.requested
        os.kill(os.getpid(), signal.SIGTERM)         # handled in this (the main) thread: a flag, the process lives on
        for _ in range(100):
            if stop.requested:
                break
        assert stop.requested and stop.signal_name == 'SIGTERM'
        os.kill(os.getpid(), signal.SIGINT)          # no KeyboardInterrupt; the first signal stays the reported one
        for _ in range(100):
            pass
        assert stop.signal_name == 'SIGTERM'
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


def test_the_build_phases_stop_in_order_between_two_phases():
    import signal
    R = _runner_module()
    stop, ran, lines = R.StopFlag(), [], []
    phases = [('the stage', lambda: ran.append('stage')),
              ('the physics world', lambda: (ran.append('world'), stop(signal.SIGINT))),   # a signal lands here
              ('the world reset', lambda: ran.append('reset'))]
    assert R.run_phases(stop, phases, lines.append) is False
    assert ran == ['stage', 'world']                 # the phase in progress finished; the next one never started
    assert lines == ['stopped by SIGINT during the build, before the world reset -- an orderly stop']
    stop, ran, lines = R.StopFlag(), [], []
    assert R.run_phases(stop, phases[:1], lines.append) is True and ran == ['stage'] and lines == []
    stop, lines = R.StopFlag(), []
    assert R.run_phases(stop, [('the stage', lambda: stop(signal.SIGTERM))], lines.append) is False
    assert lines == ['stopped by SIGTERM at the end of the build -- an orderly stop']
