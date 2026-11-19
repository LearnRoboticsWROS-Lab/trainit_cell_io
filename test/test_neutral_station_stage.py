"""U level (needs pxr, e.g. usd-core; SKIPS without it): the PUBLIC neutral station's stage as the builder authors it.

What a USD read can prove before Isaac runs it (the physics itself is the stage runner's --selfcheck and the I level):
the joint names (D7), types, axes, limits (the 0 / 300 mm stroke), position drives whose target IS the authored joint
state (no lurch at the first step), two fixed-base articulation roots, a free part resting in the nest, a static
procedural ground, nothing referenced from anywhere (no remote, binary or third-party asset), and the cell file's
binding (prim paths, nest point, jaw positions) consistent with the stage.
"""

import importlib.util
import os
import sys

import pytest

pxr = pytest.importorskip('pxr')
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
EX = os.path.join(ROOT, 'examples', 'neutral_station')

from trainit_cell_io.core import load  # noqa: E402


def _builder():
    spec = importlib.util.spec_from_file_location('build_neutral_station', os.path.join(EX, 'build_neutral_station.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def built():
    B = _builder()
    stage = Usd.Stage.CreateInMemory()
    B.build(stage)
    return B, stage


def _authored_schemas(prim):
    """The apiSchemas tokens as authored (usd-core's GetAppliedSchemas() omits the PhysX ones its registry lacks)."""
    op = prim.GetMetadata('apiSchemas')
    return list(op.GetAddedOrExplicitItems()) if op is not None else []


def _joint(stage, path):
    prim = stage.GetPrimAtPath(path)
    assert prim.IsValid(), path
    return prim


def test_the_three_joints_names_types_axes_and_limits(built):
    B, stage = built
    turn = _joint(stage, '/World/neutral_station/station/joints/turntable_joint')
    jaw = _joint(stage, '/World/neutral_station/station/joints/jaw_joint')
    carriage = _joint(stage, '/World/neutral_station/indexer/joints/carriage_joint')
    assert turn.IsA(UsdPhysics.RevoluteJoint) and UsdPhysics.RevoluteJoint(turn).GetAxisAttr().Get() == 'Z'
    assert jaw.IsA(UsdPhysics.PrismaticJoint) and UsdPhysics.PrismaticJoint(jaw).GetAxisAttr().Get() == 'X'
    assert carriage.IsA(UsdPhysics.PrismaticJoint) and UsdPhysics.PrismaticJoint(carriage).GetAxisAttr().Get() == 'X'
    revolute = UsdPhysics.RevoluteJoint(turn)
    lo, hi = revolute.GetLowerLimitAttr().Get(), revolute.GetUpperLimitAttr().Get()
    assert lo < 0.0 and hi > 180.0                                     # degrees: load 0 and turned 180 inside
    lo, hi = (UsdPhysics.PrismaticJoint(carriage).GetLowerLimitAttr().Get(),
              UsdPhysics.PrismaticJoint(carriage).GetUpperLimitAttr().Get())
    assert lo < 0.0 and 0.300 < hi < 0.35                              # metres: the 0 / 300 mm stroke (D7)
    names = sorted(p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.Joint) and not p.IsA(
        UsdPhysics.FixedJoint))
    assert names == ['carriage_joint', 'jaw_joint', 'turntable_joint']


@pytest.mark.parametrize('path, kind', [('/World/neutral_station/station/joints/turntable_joint', 'angular'),
                                        ('/World/neutral_station/station/joints/jaw_joint', 'linear'),
                                        ('/World/neutral_station/indexer/joints/carriage_joint', 'linear')])
def test_position_drives_whose_target_is_the_authored_initial_state(built, path, kind):
    _, stage = built
    prim = _joint(stage, path)
    drive = UsdPhysics.DriveAPI.Get(prim, kind)
    assert drive.GetTypeAttr().Get() == 'force' and drive.GetStiffnessAttr().Get() > 0
    assert drive.GetDampingAttr().Get() > 0 and drive.GetMaxForceAttr().Get() > 0
    assert drive.GetTargetPositionAttr().Get() == 0.0
    assert prim.GetAttribute(f'state:{kind}:physics:position').Get() == 0.0      # initial state = target
    assert {f'PhysicsJointStateAPI:{kind}', f'PhysicsDriveAPI:{kind}', 'PhysxJointAPI'} <= set(_authored_schemas(prim))
    assert prim.GetAttribute('physxJoint:maxJointVelocity').Get() > 0


def test_two_fixed_base_articulations_a_free_part_and_a_static_ground(built):
    _, stage = built
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert sorted(str(p.GetPath()) for p in roots) == ['/World/neutral_station/indexer/root_joint',
                                                       '/World/neutral_station/station/root_joint']
    for root in roots:
        joint = UsdPhysics.FixedJoint(root)
        assert joint and joint.GetBody0Rel().GetTargets() == [] and len(joint.GetBody1Rel().GetTargets()) == 1
        assert root.GetAttribute('physxArticulation:enabledSelfCollisions').Get() is False
    part = stage.GetPrimAtPath('/World/neutral_station/part')
    assert part.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.MassAPI(part).GetMassAttr().Get() > 0
    assert stage.GetPrimAtPath('/World/neutral_station/part/shape').HasAPI(UsdPhysics.CollisionAPI)
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    assert not any('/World/neutral_station/part' in str(t) for j in joints
                   for t in UsdPhysics.Joint(j).GetBody0Rel().GetTargets() + UsdPhysics.Joint(j).GetBody1Rel()
                   .GetTargets())                                      # the part is free, held only by contact
    ground = stage.GetPrimAtPath('/World/ground')
    assert ground.HasAPI(UsdPhysics.CollisionAPI) and not ground.HasAPI(UsdPhysics.RigidBodyAPI)
    scene = UsdPhysics.Scene(stage.GetPrimAtPath('/physicsScene'))
    assert scene and stage.GetPrimAtPath('/physicsScene').GetAttribute('physxScene:timeStepsPerSecond').Get() == 200
    assert UsdGeom.GetStageUpAxis(stage) == 'Z' and UsdGeom.GetStageMetersPerUnit(stage) == 1.0


def test_nothing_is_referenced_from_anywhere(built):
    _, stage = built
    assert list(stage.GetRootLayer().subLayerPaths) == []
    for prim in stage.Traverse():
        assert not prim.HasAuthoredReferences() and not prim.HasAuthoredPayloads(), prim.GetPath()
        for attr in prim.GetAttributes():         # no asset path authored (a schema's empty default is fine)
            assert attr.GetTypeName() != pxr.Sdf.ValueTypeNames.Asset or not attr.HasAuthoredValue(), attr.GetPath()
    assert [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)] == []   # primitives only, no mesh


def test_the_cell_file_binding_matches_the_stage(built):
    B, stage = built
    cfg = load(os.path.join(EX, 'cell_io_isaac.yaml'))
    isaac = cfg.isaac
    for path in isaac['articulations'].values():
        assert stage.GetPrimAtPath(path).IsValid(), path
    joints = {p.GetName(): str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)}
    for device in ('clamp', 'table', 'index'):
        spec = isaac['devices'][device]
        assert joints[spec['joint']].startswith(isaac['articulations'][spec['articulation']] + '/'), device
    sensor = isaac['devices']['part_present']
    assert stage.GetPrimAtPath(sensor['part']).IsValid() and stage.GetPrimAtPath(sensor['frame']).IsValid()
    assert tuple(sensor['nest_m']) == pytest.approx(B.NEST)            # the seated part's centre, turntable frame
    cache = UsdGeom.XformCache()
    part = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(sensor['part'])).ExtractTranslation()
    frame = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(sensor['frame'])).ExtractTranslation()
    offset = [a - b for a, b in zip(part, frame)]
    assert max(abs(a - b) for a, b in zip(offset, sensor['nest_m'])) < sensor['tolerance_m'] / 10
    # the jaw: open at 0; the part stops it before its closed TARGET, inside the closed window (in-band read-back)
    clamp = isaac['devices']['clamp']
    jaw_face_open = B.JAW_AT[0] + B.JAW_SIZE[0] / 2
    part_face = B.NEST[0] - B.PART_SIZE / 2
    contact = part_face - jaw_face_open                                # the jaw meets the centred part
    pushed = (B.FIXED_JAW[0][0] - B.FIXED_JAW[1][0] / 2) - B.PART_SIZE - jaw_face_open   # part against the fixed jaw
    assert clamp['open_m'] == 0.0 and contact < pushed < clamp['closed_m']
    assert clamp['closed_m'] - pushed <= clamp['band_m']               # squeezed, it still reads closed
    lo, hi = B.JAW_LIMITS
    assert lo < clamp['open_m'] and clamp['closed_m'] < hi
