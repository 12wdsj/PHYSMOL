"""PHYSMOL Simulation Environment - MuJoCo wrapper with multimodal perception."""

import numpy as np
from typing import Optional, Dict, Any, List

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

from .perception import MultiModalPerception


# Simple XML scene for basic physics experiments
BALL_ON_RAMP_XML = """
<mujoco model="ball_on_ramp">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="5 5 0.1" rgba="0.9 0.9 0.9 1"/>

    <!-- Ramp -->
    <body name="ramp" pos="0 0 0.5">
      <geom name="ramp_geom" type="box" size="1 0.5 0.05"
            euler="0 0.3 0" rgba="0.5 0.5 0.5 1"/>
    </body>

    <!-- Ball -->
    <body name="ball" pos="-0.8 0 1.0">
      <freejoint name="ball_joint"/>
      <geom name="ball_geom" type="sphere" size="0.05"
            mass="0.1" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""


# Multi-object scene with different materials for multimodal learning
MULTIOBJECT_SCENE_XML = """
<mujoco model="multiobject_scene">
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".9 .9 .9" rgb2=".7 .7 .7" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4"/>
  </asset>

  <worldbody>
    <!-- Ground -->
    <geom name="ground" type="plane" size="5 5 0.1" material="grid_mat"/>

    <!-- Red rubber ball -->
    <body name="red_ball" pos="0 0 1.0">
      <freejoint name="red_ball_joint"/>
      <geom name="red_ball_geom" type="sphere" size="0.08"
            mass="0.2" rgba="1 0 0 1" solref="0.01 1" friction="1.0 0.005 0.0001"/>
    </body>

    <!-- Blue metal cube -->
    <body name="blue_cube" pos="0.5 0 1.0">
      <freejoint name="blue_cube_joint"/>
      <geom name="blue_cube_geom" type="box" size="0.06 0.06 0.06"
            mass="0.5" rgba="0.2 0.3 0.9 1" solref="0.005 1" friction="0.5 0.005 0.0001"/>
    </body>

    <!-- Green wooden cylinder -->
    <body name="green_cyl" pos="-0.5 0 1.0">
      <freejoint name="green_cyl_joint"/>
      <geom name="green_cyl_geom" type="cylinder" size="0.05 0.1"
            mass="0.15" rgba="0.1 0.8 0.2 1" solref="0.02 1" friction="0.8 0.005 0.0001"/>
    </body>

    <!-- Invisible occluder (for M1 testing) -->
    <body name="occluder" pos="0 0 0.3">
      <geom name="occluder_geom" type="box" size="0.3 0.3 0.3"
            rgba="0.8 0.8 0.8 0.5" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class PhysicsExperiment:
    """A single physics experiment in MuJoCo with multimodal perception."""

    def __init__(self, xml: str = BALL_ON_RAMP_XML, vsa_dim: int = 4096):
        if not HAS_MUJOCO:
            raise ImportError("MuJoCo not installed. pip install mujoco")

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.perception = MultiModalPerception(vsa_dim)
        self._collision_log = []  # store collision events for audio encoding

    def _log_collisions(self):
        """Log collision events from this step."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            force = np.zeros(6)
            # Get contact force
            mujoco.mj_contactForce(self.model, self.data, i, force)
            self._collision_log.append({
                'time': self.data.time,
                'geom1': contact.geom1,
                'geom2': contact.geom2,
                'pos': contact.pos.copy(),
                'force': force[:3].copy(),
                'normal': contact.frame[:3].copy(),
            })

    def get_object_perception(self, body_name: str) -> np.ndarray:
        """Get multimodal perception vector for a specific object."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.perception.encode_from_mujoco(self.model, self.data, body_id)

    def get_all_object_perceptions(self) -> Dict[str, np.ndarray]:
        """Get perception vectors for all named bodies."""
        result = {}
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name != "worldbody":
                result[name] = self.perception.encode_from_mujoco(
                    self.model, self.data, i)
        return result

    def get_material_label(self, geom_name: str) -> str:
        """Infer material type from geom properties (for olfactory encoding)."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            return "none"

        rgba = self.model.geom_rgba[geom_id]
        # Simple heuristic: red=rubber, blue=metal, green=wood
        r, g, b = rgba[0], rgba[1], rgba[2]
        if r > 0.7 and g < 0.3:
            return "rubber"
        elif b > 0.7:
            return "metal"
        elif g > 0.7:
            return "wood"
        else:
            return "stone"

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self, action: Optional[np.ndarray] = None):
        if action is not None:
            self.data.ctrl[:len(action)] = action
        mujoco.mj_step(self.model, self.data)
        self._log_collisions()

    def get_state(self) -> Dict[str, np.ndarray]:
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'time': self.data.time,
        }

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel

    def get_ball_position(self) -> np.ndarray:
        ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        return self.data.xpos[ball_id].copy()

    def run_episode(self, steps: int = 500) -> Dict[str, np.ndarray]:
        self.reset()
        positions = []
        velocities = []

        for _ in range(steps):
            self.step()
            state = self.get_state()
            positions.append(state['qpos'].copy())
            velocities.append(state['qvel'].copy())

        return {
            'positions': np.array(positions),
            'velocities': np.array(velocities),
        }


class RampExperiment:
    """Ball-on-ramp experiment for M1 (object permanence) testing."""

    def __init__(self):
        self.env = PhysicsExperiment()

    def run_with_occlusion(self, occlude_at_step: int = 200,
                           total_steps: int = 500) -> Dict[str, Any]:
        """Run experiment with an occluder at a specific timestep."""
        self.env.reset()
        pre_positions = []
        post_positions = []
        all_positions = []

        for t in range(total_steps):
            self.env.step()
            pos = self.env.get_ball_position()
            all_positions.append(pos)

            if t < occlude_at_step:
                pre_positions.append(pos)
            else:
                post_positions.append(pos)

        return {
            'pre_occlusion': np.array(pre_positions),
            'post_occlusion': np.array(post_positions),
            'all_positions': np.array(all_positions),
            'occlude_at': occlude_at_step,
        }


def create_parallel_envs(xml: str, num_envs: int = 4):
    """Create multiple independent environments for parallel exploration."""
    envs = [PhysicsExperiment(xml) for _ in range(num_envs)]
    return envs
