"""The INSTALLED package (P0.b WP2): a colcon build of this source tree into a TEMPORARY workspace (never a developer's
workspace), then `ros2 launch trainit_cell_io cell_io.launch.py` / `cell_io_bridge.launch.py` with the INSTALLED example
files, the three console entry points, the installed stage runner finding its package, and an orderly SIGINT that
leaves nothing in the process group. Needs colcon and a sourced ROS 2 workspace with trainit_interfaces >= 0.5.0
(else SKIPS); ROS domain 57 (ros_harness.py).
"""

import os
import re
import shutil
import subprocess
import sys
import time

import pytest

import ros_harness as H

pytestmark = [pytest.mark.skipif(H.ros_missing() is not None, reason=H.ros_missing() or ''),
              pytest.mark.skipif(shutil.which('colcon') is None, reason='needs colcon')]

try:
    from trainit_interfaces.msg import ChannelStatus as CS
except ImportError:          # skipped
    CS = None


def _installed_env():
    """The test's ROS environment WITHOUT this checkout on PYTHONPATH: the installed package must stand alone."""
    env = H.child_env()
    env['PYTHONPATH'] = os.pathsep.join(p for p in env.get('PYTHONPATH', '').split(os.pathsep)
                                        if p and os.path.realpath(p) != os.path.realpath(str(H.ROOT)))
    return env


@pytest.fixture(scope='module')
def install(tmp_path_factory):
    ws = tmp_path_factory.mktemp('cell_io_ws')
    cmd = ['colcon', '--log-base', str(ws / 'log'), 'build', '--base-paths', str(H.ROOT), '--build-base',
           str(ws / 'build'), '--install-base', str(ws / 'install'), '--packages-select', 'trainit_cell_io',
           '--event-handlers', 'console_cohesion-']
    out = subprocess.run(cmd, cwd=str(ws), env=_installed_env(), capture_output=True, text=True, timeout=900,
                         stdin=subprocess.DEVNULL)
    assert out.returncode == 0 and '1 package finished' in out.stdout, out.stdout[-3000:] + out.stderr[-3000:]
    yield ws / 'install'
    shutil.rmtree(ws, ignore_errors=True)


@pytest.fixture(scope='module')
def ctx():
    context = H.init_context()
    yield context
    context.try_shutdown()


def _sourced(install, command):
    return ['bash', '-c', f'source {install}/setup.bash && exec {command}']


def _share(install):
    return install / 'trainit_cell_io' / 'share' / 'trainit_cell_io'


def _prefix(install):
    return install / 'trainit_cell_io'


def test_the_install_carries_the_entry_points_launch_files_and_examples(install):
    prefix = _prefix(install)
    for exe in ('cell_io_node', 'cell_io_bridge', 'cell_io_cli'):
        assert (prefix / 'lib' / 'trainit_cell_io' / exe).is_file(), exe
    share = _share(install)
    for name in ('launch/cell_io.launch.py', 'launch/cell_io_bridge.launch.py', 'package.xml', 'README.md',
                 'examples/neutral_station/cell_io_mock.yaml', 'examples/neutral_station/cell_io_isaac.yaml',
                 'examples/neutral_station/build_neutral_station.py',
                 'examples/neutral_station/run_neutral_station.py'):
        assert (share / name).is_file(), name
    assert re.search(r'<version>0\.1\.0</version>', (share / 'package.xml').read_text())
    sites = list(prefix.glob('lib/python3*/site-packages/trainit_cell_io'))
    assert sites and all((sites[0] / sub / '__init__.py').is_file() for sub in ('core', 'isaac', 'ros'))


def test_the_installed_stage_runner_finds_its_package_and_validates_its_cell_file(install, tmp_path):
    examples = _share(install) / 'examples' / 'neutral_station'
    code = ('import sys; sys.path.insert(0, %r); import run_neutral_station as R; where = R.core_path(); '
            'cfg = R.load_config(R.os.path.join(R.HERE, "cell_io_isaac.yaml")); '
            'import trainit_cell_io; print(where); print(trainit_cell_io.__file__); print(cfg.plant)' % str(examples))
    env = {'HOME': os.environ.get('HOME', '/tmp'), 'PATH': '/usr/bin:/bin'}     # no ROS, no checkout: as in Kit
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=str(tmp_path),
                         timeout=60)
    assert out.returncode == 0, out.stderr
    where, module, plant = out.stdout.split()
    assert where.endswith('site-packages') and str(_prefix(install)) in module and plant == 'isaac'


def test_ros2_launch_serves_the_installed_mock_example_and_stops_cleanly(install, ctx, tmp_path):
    config = _share(install) / 'examples' / 'neutral_station' / 'cell_io_mock.yaml'
    launch = H.Proc(_sourced(install, f'ros2 launch trainit_cell_io cell_io.launch.py config:={config}'),
                    tmp_path / 'launch.log', env=_installed_env(), cwd=tmp_path)
    req = H.Requester(ctx, ['turner', 'indexer'])
    try:
        assert launch.wait_for(r'cell_io: serving 2 machine service\(s\)', 60), launch.text()
        assert str(_prefix(install)) in subprocess.run(                              # the INSTALLED package runs
            _sourced(install, 'python3 -c "import trainit_cell_io; print(trainit_cell_io.__file__)"'),
            env=_installed_env(), capture_output=True, text=True, timeout=60, cwd=str(tmp_path)).stdout
        hit = req.wait_status('turner', lambda m: m.state == CS.STATE_READY, 20)
        assert hit and req.wait_matched('turner')
        req.spin_for(0.3)
        t0 = req.request('turner', hit[1].source_session, 0x1A0C4, 1)
        done = req.wait_status('turner', lambda m: m.state == CS.STATE_DONE and m.cycle_done, 20, after=t0)
        assert done, req.latest('turner')
        H.timing('installed launch, the example mock model: turner request -> DONE', done[0] - t0)
        state = subprocess.run(_sourced(install, 'ros2 run trainit_cell_io cell_io_cli state'), env=_installed_env(),
                               capture_output=True, text=True, timeout=60, cwd=str(tmp_path))
        assert state.returncode == 0 and 'turner: DONE' in state.stdout, state
    finally:
        req.close()
        outcome = launch.stop()
    assert outcome == 'clean', launch.text()[-3000:]
    assert 'Traceback' not in launch.text()


def test_ros2_launch_bridges_the_installed_isaac_example_and_stops_cleanly(install, tmp_path):
    config = _share(install) / 'examples' / 'neutral_station' / 'cell_io_isaac.yaml'
    launch = H.Proc(_sourced(install, f'ros2 launch trainit_cell_io cell_io_bridge.launch.py config:={config}'),
                    tmp_path / 'bridge_launch.log', env=_installed_env(), cwd=tmp_path)
    try:
        assert launch.wait_for(r'cell_io_bridge: bridging 2 machine service\(s\)', 60), launch.text()
        time.sleep(1.5)
        assert 'no status image from the Isaac stage' in launch.text()     # no stage running: it says so, mints nothing
    finally:
        outcome = launch.stop()
    assert outcome == 'clean' and 'Traceback' not in launch.text(), launch.text()[-3000:]


def test_a_launch_without_a_config_is_refused(install, tmp_path):
    out = subprocess.run(_sourced(install, 'ros2 launch trainit_cell_io cell_io.launch.py'), env=_installed_env(),
                         capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL, cwd=str(tmp_path))
    assert out.returncode != 0 and 'config' in (out.stdout + out.stderr)
