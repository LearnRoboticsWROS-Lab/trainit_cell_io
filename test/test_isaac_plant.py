"""U level: the IsaacPlantAdapter (trainit_cell_io.isaac.plant) against a FAKE articulation -- no Isaac, no ROS.

The fake follows each drive target at a capped speed (a stiff position drive), can be stopped short of a target (a
part in the jaw's way), and carries the part in the turntable's nest. The station logic is the core's, unchanged (the
neutral fixture's cell file with `plant: isaac`): the adapter's job -- units, ramps, in-band limit read-back, the part
sensor, the injections with the mock's words, the binding's validation -- is what is tested here; the physics is
Isaac's (the I level and the stage runner's --selfcheck).
"""

import ast
import copy
import math
import os
import random
import subprocess
import sys

import pytest
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from trainit_cell_io.core import Cell, ConfigError, ControlError, RequestImage, parse  # noqa: E402
from trainit_cell_io.core import contract as C  # noqa: E402
from trainit_cell_io.isaac.plant import IsaacPlantAdapter  # noqa: E402

EXAMPLE = os.path.join(ROOT, 'examples', 'neutral_station', 'cell_io_isaac.yaml')
DT = 0.005
JAW, TABLE, CARRIAGE = 'station/jaw_joint', 'station/turntable_joint', 'indexer/carriage_joint'
PART, FRAME = '/World/neutral_station/part', '/World/neutral_station/station/turntable'


class FakeArticulation:
    def __init__(self):
        self.q = {JAW: 0.0, TABLE: 0.0, CARRIAGE: 0.0}
        self.target = dict(self.q)
        self.vmax = {JAW: 0.5, TABLE: 20.0, CARRIAGE: 3.0}     # a stiff drive: it keeps up with every ramp
        self.stop = {}                                        # key -> the highest position the joint can reach
        self.nan = set()
        self.part_offset = (0.0, 0.0, 0.0)
        self.sent = []

    def step(self, dt=DT):
        for key, q in self.q.items():
            delta = self.target[key] - q
            move = max(-self.vmax[key] * dt, min(self.vmax[key] * dt, delta))
            q += move
            if key in self.stop:
                q = min(q, self.stop[key])
            self.q[key] = q

    def joint_positions(self):
        return {k: (float('nan') if k in self.nan else v) for k, v in self.q.items()}

    def set_joint_targets(self, targets):
        self.sent.append(dict(targets))
        self.target.update(targets)

    def body_position_in(self, body, frame):
        if (body, frame) != (PART, FRAME):
            return None
        if self.part_offset is None:
            return None
        return tuple(a + b for a, b in zip((0.1, 0.0, 0.04), self.part_offset))


def _data():
    with open(EXAMPLE, encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _cell(data=None, fake=None):
    cfg = parse(copy.deepcopy(data or _data()))
    fake = fake or FakeArticulation()
    plant = IsaacPlantAdapter(cfg.isaac, fake)
    cell = Cell(cfg, plant, rng=random.Random(5), sessions={'turner': 0x1111, 'indexer': 0x2222})
    return cell, plant, fake


def _step(cell, fake, n=1):
    for _ in range(n):
        fake.step()                      # the simulator steps its physics first (Kit: world.step) ...
        cell.step(advance_plant=False)   # ... then the cell scans the read-back


def _until(cell, fake, service, states, limit_s=30.0):
    pd = cell.services[service]
    for _ in range(int(limit_s / DT)):
        if pd.state in states:
            return cell.steps
        _step(cell, fake)
    raise AssertionError(f'{service} never reached {states}: {pd.image}')


def _request(cell, service, seq):
    cell.set_request(service, RequestImage(cell.services[service].session, 0xAB, seq))


# --- a whole cycle, units, ramps ---------------------------------------------------------------------------------

def test_a_turner_cycle_and_an_indexer_cycle_complete_from_the_read_back():
    cell, plant, fake = _cell()
    _request(cell, 'turner', 1)
    start = cell.steps
    done = _until(cell, fake, 'turner', (C.STATE_DONE, C.STATE_FAULT))
    t = cell.services['turner']
    assert t.state == C.STATE_DONE and t.image.cycle_done, t.image
    seconds = (done - start) * DT
    # the jaw reads closed / open 3 mm (of 18) before its ramp ends: 0.25 s each; a half turn reads in position 1 deg
    # (of 180) early: 0.994 s each; plus a scan or two per step
    assert 2.45 <= seconds <= 2.75, seconds
    assert abs(fake.q[TABLE]) < 1e-9 and abs(fake.q[JAW]) < 1e-9
    assert max(s.get(TABLE, 0.0) for s in fake.sent) == pytest.approx(math.pi)   # the target in RADIANS
    _request(cell, 'indexer', 1)
    _until(cell, fake, 'indexer', (C.STATE_DONE, C.STATE_FAULT))
    assert cell.services['indexer'].state == C.STATE_DONE
    assert fake.q[CARRIAGE] == pytest.approx(0.3)                                   # 300 mm in METRES
    assert cell.devices['index'].position == pytest.approx(300.0)                  # read back in mm
    assert cell.devices['table'].position == pytest.approx(0.0, abs=1e-6)


def test_the_drive_target_ramps_at_the_device_speed_one_increment_per_scan():
    cell, plant, fake = _cell()
    _request(cell, 'indexer', 1)
    _until(cell, fake, 'indexer', (C.STATE_DONE,))
    ramps = [s[CARRIAGE] for s in fake.sent if CARRIAGE in s]
    steps = [b - a for a, b in zip(ramps, ramps[1:]) if b != a]
    assert steps and max(steps) == pytest.approx(0.2 * 0.010)       # 200 mm/s x one 10 ms scan, never a jump
    assert JAW not in fake.sent[0] or fake.sent[0][JAW] == 0.0       # the jaw is only commanded open (where it is)


def test_a_never_commanded_joint_keeps_the_stage_target():
    cell, plant, fake = _cell()
    _step(cell, fake, 40)
    assert all(TABLE not in s and CARRIAGE not in s for s in fake.sent)


def test_the_jaw_stopped_by_the_part_inside_the_band_reads_closed():
    fake = FakeArticulation()
    fake.stop[JAW] = 0.016                          # the part stops the jaw 2 mm short of its 18 mm target
    cell, plant, _ = _cell(fake=fake)
    _request(cell, 'turner', 1)
    _until(cell, fake, 'turner', (C.STATE_DONE, C.STATE_FAULT))
    assert cell.services['turner'].state == C.STATE_DONE


def test_the_limit_sensors_are_in_band_read_backs():
    cell, plant, fake = _cell()
    for q, expect in ((0.0, (True, False)), (0.002, (True, False)), (0.0035, (False, False)), (0.009, (False, False)),
                      (0.0155, (False, True)), (0.018, (False, True)), (0.0215, (False, False))):
        fake.q[JAW] = q
        image = plant.read()
        assert (image['clamp.open'], image['clamp.closed']) == expect, q


def test_the_part_sensor_is_the_part_position_in_the_turntable_frame():
    cell, plant, fake = _cell()
    assert plant.read()['part_present'] is True
    fake.part_offset = (0.009, 0.0, 0.0)
    assert plant.read()['part_present'] is True       # inside the 10 mm tolerance
    fake.part_offset = (0.0, 0.008, 0.008)
    assert plant.read()['part_present'] is False      # 11.3 mm away
    fake.part_offset = None
    assert plant.read()['part_present'] is False      # no pose: no part


def test_no_part_in_the_nest_is_a_station_fault_before_anything_moves():
    fake = FakeArticulation()
    fake.part_offset = (0.05, 0.0, 0.0)
    cell, plant, _ = _cell(fake=fake)
    _request(cell, 'turner', 1)
    _until(cell, fake, 'turner', (C.STATE_FAULT, C.STATE_DONE))
    t = cell.services['turner']
    assert t.state == C.STATE_FAULT and t.image.fault_code == C.FAULT_STATION_FAULT and 'part_present is off' in \
        t.image.detail
    assert fake.q[TABLE] == 0.0


# --- injections (the mock's words) -------------------------------------------------------------------------------

def _inject(cell, **kw):
    return cell.control(dict({'cmd': 'inject'}, **kw))


def test_f1_a_stuck_actuator_holds_its_target_and_times_out():
    cell, plant, fake = _cell()
    assert 'held' in _inject(cell, fault='stuck_actuator', device='clamp')
    _request(cell, 'turner', 1)
    busy = _until(cell, fake, 'turner', (C.STATE_BUSY,))
    fault = _until(cell, fake, 'turner', (C.STATE_FAULT, C.STATE_DONE))
    t = cell.services['turner']
    assert t.state == C.STATE_FAULT and t.image.fault_code == C.FAULT_ACTUATOR_TIMEOUT and not t.image.cycle_done
    # EXACT (review fix C5): one scan (2 steps) for the `require` step, then the clamp's 2000 ms = 400 steps from its
    # command to the fault -- before the fix a timeout allowed one scan more (404 steps)
    assert fault - busy == 2 + 400
    assert fake.q[JAW] == 0.0 and all(s.get(JAW, 0.0) == 0.0 for s in fake.sent)   # the valve never switched
    _inject(cell, fault='clear')
    cell.control({'cmd': 'reset', 'service': 'turner'})
    _until(cell, fake, 'turner', (C.STATE_READY,))
    assert fake.q[JAW] == pytest.approx(0.0)                      # homed: the jaw open again


def test_a_stuck_axis_holds_its_target_and_times_out():
    cell, plant, fake = _cell()
    _inject(cell, fault='stuck_actuator', device='table')
    _request(cell, 'turner', 1)
    _until(cell, fake, 'turner', (C.STATE_FAULT, C.STATE_DONE))
    t = cell.services['turner']
    assert t.image.fault_code == C.FAULT_ACTUATOR_TIMEOUT and 'table did not reach' in t.image.detail
    assert fake.q[TABLE] == 0.0


def test_slow_stretches_the_motion_and_still_completes():
    cell, plant, fake = _cell()
    _inject(cell, fault='slow', device='table', factor=3)
    _request(cell, 'turner', 1)
    start = cell.steps
    done = _until(cell, fake, 'turner', (C.STATE_DONE, C.STATE_FAULT))
    assert cell.services['turner'].state == C.STATE_DONE
    assert 6.4 <= (done - start) * DT <= 6.75                      # two half turns at 60 deg/s (2.98 s) + the jaw


def test_failed_sensors_are_sensor_faults():
    cell, plant, fake = _cell()
    _inject(cell, fault='failed_sensor', device='part_present')
    _request(cell, 'turner', 1)
    _until(cell, fake, 'turner', (C.STATE_FAULT,))
    assert cell.services['turner'].image.fault_code == C.FAULT_SENSOR_FAULT
    cell, plant, fake = _cell()
    _inject(cell, fault='failed_sensor', device='clamp', signal='open')
    _request(cell, 'turner', 1)
    _until(cell, fake, 'turner', (C.STATE_FAULT,))
    assert cell.services['turner'].image.fault_code == C.FAULT_SENSOR_FAULT and 'both on' in \
        cell.services['turner'].image.detail


def test_sensor_value_station_fault_hold_consume_delay_and_clear():
    cell, plant, fake = _cell()
    _inject(cell, fault='sensor_value', device='part_present', value=False)
    _request(cell, 'turner', 1)
    _until(cell, fake, 'turner', (C.STATE_FAULT,))
    assert 'part_present is off' in cell.services['turner'].image.detail
    _inject(cell, fault='clear')
    cell.control({'cmd': 'reset', 'service': 'turner'})
    _until(cell, fake, 'turner', (C.STATE_READY,))
    _inject(cell, fault='hold', service='turner')
    _request(cell, 'turner', 2)
    _step(cell, fake, 2000)                                        # 10 s, beyond every timeout
    assert cell.services['turner'].state == C.STATE_BUSY
    _inject(cell, fault='clear')
    _until(cell, fake, 'turner', (C.STATE_DONE,))
    _inject(cell, fault='station_fault', service='turner', detail='door open')
    _step(cell, fake, 2)
    assert cell.services['turner'].image.fault_code == C.FAULT_STATION_FAULT
    _inject(cell, fault='clear')
    cell.control({'cmd': 'reset', 'service': 'turner'})
    _until(cell, fake, 'turner', (C.STATE_READY,))
    _inject(cell, fault='consume_delay', service='turner', ms=500)
    _request(cell, 'turner', 3)
    seen = cell.steps
    busy = _until(cell, fake, 'turner', (C.STATE_BUSY,))
    assert 0.5 <= (busy - seen) * DT <= 0.52
    described = plant.describe()['injections']
    assert described['consume_delay_ms'] == {'turner': 500}
    for bad in ({'fault': 'melt'}, {'fault': 'stuck_actuator', 'device': 'part_present'},
                {'fault': 'slow', 'device': 'table', 'factor': 0.5}, {'fault': 'hold', 'service': 'nope'},
                {'fault': 'consume_delay', 'service': 'turner', 'ms': 15},
                {'fault': 'failed_sensor', 'device': ['x']}):
        with pytest.raises(ControlError):
            _inject(cell, **bad)


def test_a_missing_read_back_is_a_sensor_fault():
    fake = FakeArticulation()
    fake.nan.add(TABLE)
    cell, plant, _ = _cell(fake=fake)
    _request(cell, 'turner', 1)
    _until(cell, fake, 'turner', (C.STATE_FAULT,))
    assert 'no position read-back' in cell.services['turner'].image.detail
    fake = FakeArticulation()
    fake.nan.add(JAW)
    cell, plant, _ = _cell(fake=fake)
    image = plant.read()
    assert image['clamp.open'] is None and image['clamp.closed'] is None


# --- the binding -----------------------------------------------------------------------------------------------

def test_the_glue_gets_the_joints_and_bodies_it_must_provide():
    cell, plant, fake = _cell()
    assert plant.joints() == {JAW: ('/World/neutral_station/station', 'jaw_joint'),
                              TABLE: ('/World/neutral_station/station', 'turntable_joint'),
                              CARRIAGE: ('/World/neutral_station/indexer', 'carriage_joint')}
    assert plant.bodies() == {PART, FRAME}
    assert plant.describe()['plant'] == 'isaac' and plant.describe()['stage'] == 'neutral_station'


def _mutated(path, value):
    data = _data()
    node = data['isaac']
    for key in path[:-1]:
        node = node[key]
    if value is KeyError:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return data


@pytest.mark.parametrize('path, value, why', [
    (('devices', 'clamp'), KeyError, 'devices.clamp: the device has no Isaac binding'),
    (('devices', 'gripper'), {'articulation': 'station', 'joint': 'x', 'speed': 1.0}, 'no such device'),
    (('devices', 'clamp', 'speed'), 1.0, 'unknown key'),
    (('devices', 'clamp', 'band_m'), 0.01, 'must be under half the stroke'),
    (('devices', 'clamp', 'stroke_ms'), 0, 'stroke_ms'),
    (('devices', 'table', 'articulation'), 'robot', "no articulation 'robot'"),
    (('devices', 'index', 'joint'), 'turntable_joint', None),       # another articulation: a distinct joint key
    (('devices', 'index', 'articulation'), 'station', None),
    (('devices', 'table', 'speed'), -5.0, 'must be > 0'),
    (('devices', 'part_present', 'nest_m'), [0.1, 0.0], 'three numbers'),
    (('devices', 'part_present', 'part'), 'World/part', 'absolute prim path'),
    (('articulations',), {}, 'non-empty mapping'),
    (('articulations', 'station'), 'station', 'absolute prim path'),
])
def test_the_binding_is_validated_with_readable_refusals(path, value, why):
    data = _mutated(path, value)
    if why is None:
        if path[-1] == 'articulation':
            data['isaac']['devices']['index']['joint'] = 'turntable_joint'
            with pytest.raises(ConfigError, match='one device per joint'):
                _cell(data)
        else:
            _cell(data)                  # indexer/turntable_joint differs from station/turntable_joint: fine
        return
    with pytest.raises(ConfigError, match=why):
        _cell(data)


def test_the_isaac_plant_modules_are_pure():
    """C1 for the Isaac side: no simulator or ROS import at module level (the Kit glue imports lazily)."""
    forbidden = {'omni', 'isaacsim', 'pxr', 'carb', 'rclpy', 'std_msgs', 'rosgraph_msgs', 'trainit_interfaces'}
    for name in ('plant.py', 'loop.py', 'kit_ros.py', 'kit.py'):
        path = os.path.join(ROOT, 'trainit_cell_io', 'isaac', name)
        if not os.path.exists(path):
            continue
        tree = ast.parse(open(path, encoding='utf-8').read())
        for node in tree.body:                                   # module level only
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            for mod in mods:
                assert mod.split('.')[0] not in forbidden, f'{name} imports {mod} at module level'
    code = ('import sys; sys.path.insert(0, %r); import trainit_cell_io.isaac.plant, trainit_cell_io.isaac.loop, '
            'trainit_cell_io.isaac.kit_ros; '
            'print(sorted(m for m in %r if m in sys.modules))' % (ROOT, sorted(forbidden)))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == '[]', out.stdout


def test_a_world_point_in_a_rotated_frame():
    from trainit_cell_io.isaac.plant import point_in_frame
    half = math.sqrt(0.5)
    # a frame at (1, 2, 0.4) turned +90 deg about Z: the world point 1 m along +Y from it lies on the frame's +X
    p = point_in_frame((1.0, 3.0, 0.4), (1.0, 2.0, 0.4), (half, 0.0, 0.0, half))
    assert p == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)
    # turned 180 deg about Z (the turntable at `turned`): the nest at +0.1 m X in the frame is at -0.1 m X in the world
    p = point_in_frame((-0.1, 0.0, 0.441), (0.0, 0.0, 0.401), (0.0, 0.0, 0.0, 1.0))
    assert p == pytest.approx((0.1, 0.0, 0.04), abs=1e-12)
    assert point_in_frame((1, 2, 3), (0, 0, 0), (2.0, 0, 0, 0)) == pytest.approx((1, 2, 3))   # normalised
    with pytest.raises(ValueError):
        point_in_frame((0, 0, 0), (0, 0, 0), (0, 0, 0, 0))
