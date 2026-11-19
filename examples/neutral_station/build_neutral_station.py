#!/usr/bin/env python3
"""Build the PUBLIC neutral station stage from USD primitives only (P0B_DESIGN.md §5; P0.b D7, D13).

A customer-neutral fixture for the machine-service handshake: a turn station and a line indexer, as two articulations
with position drives. Every shape is a USD primitive (boxes), every number is this file's own, every name is generic;
nothing is referenced or downloaded (no remote, binary or third-party asset).

Used by run_neutral_station.py, which calls `build(stage)` inside Isaac Sim BEFORE the World is created and reset.
Standalone, with any Python that has pxr (Isaac's python.sh, or usd-core), it writes the stage as .usda TEXT to look
at -- never commit that output:

    ~/isaacsim/python.sh build_neutral_station.py --out /tmp/neutral_station.usda

What it authors (metres, Z up):
  /physicsScene                      gravity -Z 9.81, 200 Hz (dt 5 ms), TGS solver, CPU dynamics
  /World/ground                      a static box collider (procedural ground)
  /World/neutral_station/station     ARTICULATION, fixed base:
      base        a pedestal, fixed to the world by root_joint (the articulation root)
      turntable   revolute `turntable_joint` about Z (0 deg load .. 180 deg turned; limits -10 .. 190 deg), position
                  drive; carries a fixed jaw and two nest guides
      jaw         prismatic `jaw_joint` along the turntable's X (0 open; the closed target 18 mm squeezes the part
                  against the fixed jaw), position drive
  /World/neutral_station/part        a dynamic 40 mm box resting in the nest
  /World/neutral_station/indexer     ARTICULATION, fixed base: rail (root_joint) + carriage on prismatic
                                     `carriage_joint` along X, a 0 / 300 mm stroke (limits -5 .. 310 mm), position drive
The initial joint state is the drive target (0) BEFORE any reset: nothing lurches when the physics starts.

USD units: a revolute joint's limits, drive target and joint state are in DEGREES and its drive gains per degree; a
prismatic joint's are in metres. (Isaac's articulation API reads and commands radians / metres.) PhysX-specific
attributes are authored by name, so the builder also runs where PhysxSchema is not importable.
"""

from __future__ import annotations

import argparse
from typing import Sequence, Tuple

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

ROOT = '/World/neutral_station'
STATION = f'{ROOT}/station'
INDEXER = f'{ROOT}/indexer'
PART = f'{ROOT}/part'
TURNTABLE = f'{STATION}/turntable'
PHYSICS_HZ = 200

Vec3 = Tuple[float, float, float]

# --- the layout (this fixture's own numbers) ---------------------------------------------------------------------
STATION_AT = (0.0, 0.0, 0.0)
PEDESTAL = (0.30, 0.30, 0.40)              # the base's box; the turntable's frame sits 1 mm above its top
TURNTABLE_Z = 0.401
PLATE = (0.36, 0.36, 0.02)                 # the turntable plate (its top 20 mm above the turntable frame)
PART_SIZE = 0.04
NEST = (0.10, 0.0, 0.04)                   # the part's centre in the turntable frame when seated (cell_io.yaml)
FIXED_JAW = ((0.131, 0.0, 0.04), (0.02, 0.06, 0.04))           # centre, size: its inner face 1 mm from the part
GUIDES = [((0.10, s * 0.028, 0.035), (0.06, 0.01, 0.03)) for s in (1, -1)]
JAW_AT = (0.055, 0.0, 0.036)               # the moving jaw's centre, open (its face 15 mm from the part)
JAW_SIZE = (0.02, 0.04, 0.03)
JAW_LIMITS = (-0.002, 0.022)               # m
TURN_LIMITS = (-10.0, 190.0)               # deg
INDEXER_AT = (0.0, -0.60, 0.0)
RAIL = ((0.15, 0.0, 0.02), (0.50, 0.06, 0.04))
CARRIAGE_Z = 0.061
CARRIAGE = (0.10, 0.10, 0.04)
CARRIAGE_LIMITS = (-0.005, 0.31)           # m: the 0 / 300 mm stroke and a margin

# drives: (stiffness, damping, max force, max joint velocity) -- per degree for the revolute joint
TURN_DRIVE = (5.0, 0.15, 50.0, 360.0)      # N m/deg, N m s/deg, N m, deg/s
JAW_DRIVE = (1000.0, 15.0, 20.0, 0.5)      # N/m, N s/m, N, m/s
CARRIAGE_DRIVE = (2000.0, 80.0, 200.0, 1.0)

MASS = {'base': 10.0, 'turntable': 2.0, 'jaw': 0.1, 'part': 0.1, 'rail': 5.0, 'carriage': 1.0}
COLOR = {'base': (0.25, 0.25, 0.28), 'turntable': (0.30, 0.45, 0.65), 'jaw': (0.90, 0.55, 0.15),
         'part': (0.20, 0.70, 0.30), 'rail': (0.40, 0.40, 0.42), 'carriage': (0.95, 0.80, 0.20),
         'ground': (0.80, 0.80, 0.78)}


def _add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _attr(prim: Usd.Prim, name: str, type_name, value) -> None:
    prim.CreateAttribute(name, type_name).Set(value)


def _box(stage: Usd.Stage, path: str, center: Vec3, size: Vec3, color: Vec3, collide: bool = True) -> UsdGeom.Cube:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, -0.5), Gf.Vec3f(0.5, 0.5, 0.5)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _body(stage: Usd.Stage, path: str, at: Vec3, mass: float) -> Usd.Prim:
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*at))
    prim = xform.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    return prim


def _bind_material(prim: Usd.Prim, material: UsdShade.Material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')


def _root_joint(stage: Usd.Stage, path: str, body: str, world_at: Vec3) -> Usd.Prim:
    """A FixedJoint body0 = the world, body1 = the base, carrying the articulation root (a fixed-base articulation)."""
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*world_at))      # body0 is the world: its frame is the world's
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    prim = joint.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    prim.AddAppliedSchema('PhysxArticulationAPI')
    _attr(prim, 'physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool, False)
    _attr(prim, 'physxArticulation:solverPositionIterationCount', Sdf.ValueTypeNames.Int, 32)
    _attr(prim, 'physxArticulation:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 1)
    return prim


def _drive(prim: Usd.Prim, kind: str, gains: Tuple[float, float, float, float]) -> None:
    """A position drive whose target AND initial joint state are 0 (authored before any reset)."""
    stiffness, damping, max_force, max_velocity = gains
    drive = UsdPhysics.DriveAPI.Apply(prim, kind)
    drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(max_force)
    drive.CreateTargetPositionAttr(0.0)
    prim.AddAppliedSchema('PhysxJointAPI')
    _attr(prim, 'physxJoint:maxJointVelocity', Sdf.ValueTypeNames.Float, max_velocity)
    prim.AddAppliedSchema(f'PhysicsJointStateAPI:{kind}')
    _attr(prim, f'state:{kind}:physics:position', Sdf.ValueTypeNames.Float, 0.0)
    _attr(prim, f'state:{kind}:physics:velocity', Sdf.ValueTypeNames.Float, 0.0)


def _joint(stage: Usd.Stage, cls, path: str, body0: str, body1: str, at0: Vec3, limits: Tuple[float, float]):
    joint = cls.Define(stage, path)
    joint.CreateAxisAttr('Z' if cls is UsdPhysics.RevoluteJoint else 'X')
    joint.CreateLowerLimitAttr(limits[0])
    joint.CreateUpperLimitAttr(limits[1])
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*at0))           # in body0's frame; body1's frame sits at the joint
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0))
    return joint.GetPrim()


def build(stage: Usd.Stage) -> None:
    """Author the neutral station into `stage` (an empty or Isaac's default stage)."""
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, '/physicsScene')
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    sp = scene.GetPrim()
    sp.AddAppliedSchema('PhysxSceneAPI')
    _attr(sp, 'physxScene:timeStepsPerSecond', Sdf.ValueTypeNames.UInt, PHYSICS_HZ)
    _attr(sp, 'physxScene:solverType', Sdf.ValueTypeNames.Token, 'TGS')
    _attr(sp, 'physxScene:broadphaseType', Sdf.ValueTypeNames.Token, 'MBP')
    _attr(sp, 'physxScene:enableGPUDynamics', Sdf.ValueTypeNames.Bool, False)
    _attr(sp, 'physxScene:enableCCD', Sdf.ValueTypeNames.Bool, False)
    _attr(sp, 'physxScene:enableStabilization', Sdf.ValueTypeNames.Bool, True)

    light = UsdLux.DistantLight.Define(stage, '/World/sun')
    light.CreateIntensityAttr(2500.0)
    light.AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 0.0))
    UsdLux.DomeLight.Define(stage, '/World/sky').CreateIntensityAttr(600.0)

    _box(stage, '/World/ground', (0.0, 0.0, -0.01), (3.0, 3.0, 0.02), COLOR['ground'])     # static: no rigid body
    UsdGeom.Xform.Define(stage, ROOT)

    grip = UsdShade.Material.Define(stage, f'{ROOT}/materials/grip')
    material = UsdPhysics.MaterialAPI.Apply(grip.GetPrim())
    material.CreateStaticFrictionAttr(0.8)
    material.CreateDynamicFrictionAttr(0.6)
    material.CreateRestitutionAttr(0.0)

    # --- the turn station ------------------------------------------------------------------------------------------
    UsdGeom.Xform.Define(stage, STATION).AddTranslateOp().Set(Gf.Vec3d(*STATION_AT))
    _body(stage, f'{STATION}/base', (0.0, 0.0, 0.0), MASS['base'])
    _box(stage, f'{STATION}/base/pedestal', (0.0, 0.0, PEDESTAL[2] / 2), PEDESTAL, COLOR['base'])
    _root_joint(stage, f'{STATION}/root_joint', f'{STATION}/base', STATION_AT)

    _body(stage, TURNTABLE, (0.0, 0.0, TURNTABLE_Z), MASS['turntable'])
    plate = _box(stage, f'{TURNTABLE}/plate', (0.0, 0.0, PLATE[2] / 2), PLATE, COLOR['turntable'])
    _bind_material(plate.GetPrim(), grip)
    _box(stage, f'{TURNTABLE}/fixed_jaw', FIXED_JAW[0], FIXED_JAW[1], COLOR['jaw'])
    for n, (center, size) in enumerate(GUIDES):
        _box(stage, f'{TURNTABLE}/guide_{n}', center, size, COLOR['turntable'])
    turn = _joint(stage, UsdPhysics.RevoluteJoint, f'{STATION}/joints/turntable_joint', f'{STATION}/base', TURNTABLE,
                  (0.0, 0.0, TURNTABLE_Z), TURN_LIMITS)
    _drive(turn, 'angular', TURN_DRIVE)

    _body(stage, f'{STATION}/jaw', _add((0.0, 0.0, TURNTABLE_Z), JAW_AT), MASS['jaw'])
    jaw_box = _box(stage, f'{STATION}/jaw/block', (0.0, 0.0, 0.0), JAW_SIZE, COLOR['jaw'])
    _bind_material(jaw_box.GetPrim(), grip)
    jaw = _joint(stage, UsdPhysics.PrismaticJoint, f'{STATION}/joints/jaw_joint', TURNTABLE, f'{STATION}/jaw',
                 JAW_AT, JAW_LIMITS)
    _drive(jaw, 'linear', JAW_DRIVE)

    # --- the part, resting in the nest (a free rigid body; 0.5 mm above the plate) ---------------------------------
    part_at = _add(_add(STATION_AT, (0.0, 0.0, TURNTABLE_Z)), (NEST[0], NEST[1], NEST[2] + 0.0005))
    part = _body(stage, PART, part_at, MASS['part'])
    part.AddAppliedSchema('PhysxRigidBodyAPI')
    _attr(part, 'physxRigidBody:maxDepenetrationVelocity', Sdf.ValueTypeNames.Float, 0.5)
    _attr(part, 'physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, 16)
    part_box = _box(stage, f'{PART}/shape', (0.0, 0.0, 0.0), (PART_SIZE,) * 3, COLOR['part'])
    _bind_material(part_box.GetPrim(), grip)

    # --- the line indexer ------------------------------------------------------------------------------------------
    indexer_at = _add(STATION_AT, INDEXER_AT)
    UsdGeom.Xform.Define(stage, INDEXER).AddTranslateOp().Set(Gf.Vec3d(*INDEXER_AT))
    _body(stage, f'{INDEXER}/rail', (0.0, 0.0, 0.0), MASS['rail'])
    _box(stage, f'{INDEXER}/rail/beam', RAIL[0], RAIL[1], COLOR['rail'])
    _root_joint(stage, f'{INDEXER}/root_joint', f'{INDEXER}/rail', indexer_at)
    _body(stage, f'{INDEXER}/carriage', (0.0, 0.0, CARRIAGE_Z), MASS['carriage'])
    _box(stage, f'{INDEXER}/carriage/block', (0.0, 0.0, 0.0), CARRIAGE, COLOR['carriage'])
    carriage = _joint(stage, UsdPhysics.PrismaticJoint, f'{INDEXER}/joints/carriage_joint', f'{INDEXER}/rail',
                      f'{INDEXER}/carriage', (0.0, 0.0, CARRIAGE_Z), CARRIAGE_LIMITS)
    _drive(carriage, 'linear', CARRIAGE_DRIVE)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Write the neutral station stage as .usda text (to inspect it).')
    parser.add_argument('--out', required=True, help='the .usda file to write (never commit it)')
    args = parser.parse_args(argv)
    stage = Usd.Stage.CreateNew(args.out)
    build(stage)
    stage.GetRootLayer().Save()
    print(f'wrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
