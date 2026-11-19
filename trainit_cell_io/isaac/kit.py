"""The Kit glue of the Isaac plant (P0B_DESIGN.md §4): `JointAccess` over Isaac Sim 5.1's articulation API.

Runs INSIDE Isaac Sim only (`isaacsim.core.*` is imported when a `KitJointAccess` is built, never at module level, so
this module imports anywhere). The adapter it serves (`IsaacPlantAdapter`) is pure and unit-tested with a fake; what
this glue adds -- the articulation handles, the DOF indices, the rigid-body poses -- is exercised by the stage
runner's --selfcheck and the I-level runs.

    access = KitJointAccess(world, plant.joints(), plant.bodies())   # BEFORE world.reset(): added to the scene
    world.reset()
    access.resolve()                                                 # AFTER: the DOF indices, checked by name
    plant.attach(access)
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Tuple

from .plant import Vec3, point_in_frame


def _scene_name(prim_path: str) -> str:
    return 'cell_io' + prim_path.replace('/', '_')


class KitJointAccess:
    def __init__(self, world, joints: Mapping[str, Tuple[str, str]], bodies: Iterable[str]):
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        self._joints = dict(joints)
        self._articulations: Dict[str, object] = {}
        for path, _ in self._joints.values():
            if path not in self._articulations:
                self._articulations[path] = world.scene.add(SingleArticulation(prim_path=path,
                                                                               name=_scene_name(path)))
        # articulation links and free rigid bodies alike: their world poses come from the physics view. NOT added to
        # the scene (its post-reset would try to set an articulation link's pose): initialized in resolve().
        self._bodies = {path: SingleRigidPrim(prim_path=path, name=_scene_name(path), reset_xform_properties=False)
                        for path in sorted(set(bodies))}
        self._index: Dict[str, Tuple[object, int]] = {}

    def resolve(self) -> None:
        """After world.reset(): the rigid-body views, and each bound joint's DOF index (a joint the articulation lacks
        is refused)."""
        for body in self._bodies.values():
            body.initialize()
        for key, (path, joint) in self._joints.items():
            art = self._articulations[path]
            names = list(art.dof_names or [])
            if joint not in names:
                raise RuntimeError(f'articulation {path} has no joint {joint!r} (its joints: {", ".join(names)})')
            self._index[key] = (art, names.index(joint))

    def dof_names(self) -> Dict[str, list]:
        return {path: list(art.dof_names or []) for path, art in self._articulations.items()}

    def joint_positions(self) -> Dict[str, float]:
        cache: Dict[int, object] = {}
        out: Dict[str, float] = {}
        for key, (art, index) in self._index.items():
            q = cache.get(id(art))
            if q is None:
                q = cache[id(art)] = art.get_joint_positions()
            out[key] = float(q[index])
        return out

    def set_joint_targets(self, targets: Mapping[str, float]) -> None:
        import numpy as np
        from isaacsim.core.utils.types import ArticulationAction
        grouped: Dict[int, Tuple[object, list, list]] = {}
        for key, value in targets.items():
            art, index = self._index[key]
            entry = grouped.setdefault(id(art), (art, [], []))
            entry[1].append(index)
            entry[2].append(float(value))
        for art, indices, values in grouped.values():
            art.apply_action(ArticulationAction(joint_positions=np.array(values, dtype=np.float32),
                                                joint_indices=np.array(indices, dtype=np.int32)))

    def world_pose(self, body: str):
        position, orientation = self._bodies[body].get_world_pose()
        return [float(v) for v in position], [float(v) for v in orientation]

    def body_position_in(self, body: str, frame: str) -> Optional[Vec3]:
        try:
            position, _ = self.world_pose(body)
            frame_position, frame_orientation = self.world_pose(frame)
        except Exception:  # noqa: BLE001 -- no pose (a body gone, a view not ready): the sensor reads off
            return None
        return point_in_frame(position, frame_position, frame_orientation)
