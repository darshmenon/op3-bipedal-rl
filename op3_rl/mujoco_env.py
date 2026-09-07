"""Standalone MuJoCo training environment for OP3, implementing rsl_rl's
`VecEnv` interface directly (no Isaac Sim/Isaac Lab involved at all).

Why this exists: fourteen experiments in Isaac Lab (six reward-shaping
variants, four residual-CPG action-space variants, a solver-fidelity fix)
all converged to the same "falls almost every episode, never lifts a
foot" collapse -- see STAGE4_PPO_TRAINING.md. Training the *same* task
spec (same PD gains, same torque limit, same reward weights, same
tiny-forward command range) against a completely different physics
engine is a direct way to tell apart two possibilities: if MuJoCo
training converges to the same collapse, that's strong evidence the
problem is in the task/reward/actuator *specification* itself (shared
between engines); if it doesn't, that points at something Isaac
Lab/PhysX-specific (contact solver, numerical integration, etc.).

The OP3 MuJoCo model (`~/mujoco_menagerie/robotis_op3/`) is the same
model this project's PD gains were originally sourced from -- see
`op3_rl/op3_robot_cfg.py`'s "Sourcing PD gains from real Dynamixel specs"
history. `<joint damping="1.084" armature="0.045" frictionloss="0.03"/>`
and `<position kp="21.1" forcerange="-5 5"/>` in `op3.xml` are exactly
what `OP3_CFG`'s `ImplicitActuatorCfg` copied.

No GPU parallelism (MuJoCo doesn't need it for a model this small) --
each of `num_envs` environments gets its own `mujoco.MjData` sharing one
`mujoco.MjModel`, stepped in a plain Python loop. Slower per-iteration
than Isaac Sim's 512-env GPU batch, but MuJoCo's own step cost is low
enough for this to still train at a reasonable pace on CPU.

Reward terms match `OP3WalkEasyRebalancedEnvCfg` (the original, most-
tested Isaac Lab baseline) as closely as practical: `track_lin_vel_xy_exp`,
`track_ang_vel_z_exp`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `dof_torques_l2`,
`dof_acc_l2`, `action_rate_l2`, `feet_air_time`, `undesired_contacts`,
`flat_orientation_l2`, `termination_penalty`, `alive`,
`joint_deviation_arms_head`. Not a line-for-line reimplementation of
Isaac Lab's manager-based reward functions -- reimplemented directly
against MuJoCo's own state (`qpos`/`qvel`/contacts), matching the same
physical quantities and weights.
"""

from __future__ import annotations

import concurrent.futures
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

_VENDORED_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie" / "robotis_op3" / "scene.xml"
)
_ABSOLUTE_DEV_MODEL_PATH = Path("/home/asimov/mujoco_menagerie/robotis_op3/scene.xml")
# Prefer the repo-relative vendored copy (portable -- works in a fresh clone
# or a Docker build) over the absolute path this project originally
# hardcoded to this one dev machine's sibling directory (kept as a fallback
# so an existing checkout without the vendored copy still works).
_MODEL_PATH = _VENDORED_MODEL_PATH if _VENDORED_MODEL_PATH.exists() else _ABSOLUTE_DEV_MODEL_PATH

# Same 12 leg + 8 arm/head joints, same nominal crouch pose, as
# op3_rl/op3_robot_cfg.py -- kept identical for a fair comparison, not
# re-derived from scratch.
_LEG_JOINTS = [
    "l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ank_pitch", "l_ank_roll",
    "r_hip_yaw", "r_hip_roll", "r_hip_pitch", "r_knee", "r_ank_pitch", "r_ank_roll",
]
_ARM_HEAD_JOINTS = ["l_sho_pitch", "l_sho_roll", "l_el", "r_sho_pitch", "r_sho_roll", "r_el", "head_pan", "head_tilt"]
_ALL_JOINTS = _LEG_JOINTS + _ARM_HEAD_JOINTS  # canonical 20-joint order for this env's own obs/action vectors

_DEFAULT_JOINT_POS = {
    "l_hip_pitch": -0.15, "l_knee": 0.3, "l_ank_pitch": -0.15,
    "r_hip_pitch": 0.15, "r_knee": -0.3, "r_ank_pitch": 0.15,
}  # everything else defaults to 0.0, matching op3_robot_cfg.py exactly

_BASE_HEIGHT = 0.32  # matches OP3_CFG.init_state.pos, not the mujoco_menagerie file's own default 0.3
_FEET_BODIES = ["l_ank_roll_link", "r_ank_roll_link"]
_UNDESIRED_CONTACT_BODIES = ["l_hip_pitch_link", "l_knee_link", "r_hip_pitch_link", "r_knee_link"]
_TORSO_BODY = "body_link"

_ACTION_SCALE = 0.3
_EFFORT_LIMIT = 5.0  # matches forcerange in op3.xml / effort_limit_sim in op3_robot_cfg.py -- not raised
_EPISODE_LENGTH_S = 20.0
# Control-frequency-matched to Isaac Lab's sim.dt=0.005s * decimation=4 =
# 0.02s (50Hz). MuJoCo's own default physics timestep is 0.002s (kept
# unchanged, not coarsened -- that's MuJoCo's own well-tested default
# integration step), so decimation=10 here (0.002*10=0.02s) reaches the
# same *control* rate via more, finer physics substeps per control step,
# rather than fewer/coarser ones. First launch of this env used
# decimation=4 (0.008s = 125Hz, 2.5x Isaac's rate) and showed a striking
# early advantage over every Isaac Lab result tonight -- caught before
# drawing any conclusion from it, since more frequent policy corrections
# alone would trivially make balance easier, independent of which physics
# engine is "better". See STAGE4_PPO_TRAINING.md.
_DECIMATION = 10

# `0.25` (i.e. std=0.5 m/s) is Isaac Lab's own stock default for
# track_lin_vel_xy_exp/track_ang_vel_z_exp (isaaclab_tasks' base
# `RewardsCfg`, unchanged even in its own G1/H1 humanoid reference
# configs) -- but that default was tuned for robots tracking ~0.5-1.5 m/s
# commands. OP3's task here commands only 0.02-0.2 m/s, 5-25x smaller, so
# the same absolute tolerance makes near-zero velocity score well almost
# everywhere in the command range: replay-validating a converged
# checkpoint found net x-displacement of exactly 0.0m over a full episode
# (standing still, not walking) at both the original tiny range and a
# widened (-0.2, 0.2) range -- widening the *command range* didn't help
# because the *tolerance* was the actual bug. std=0.05 (variance 0.0025)
# scales the tolerance down to match OP3's own command magnitude instead
# of inheriting a value sized for a much faster-walking robot: at the
# hardest command (0.2 m/s), standing still now scores exp(-0.04/0.0025)
# ~= 0, versus 0.85 under the old default. See STAGE4_PPO_TRAINING.md.
# Isaac Lab side ported to match (OP3RewardsCfg in op3_env_cfg.py) so both
# engines keep training against the identical task spec.
_VEL_TRACKING_VARIANCE = 0.0025


class OP3MujocoVecEnv:
    """Minimal vectorized OP3 walking env over raw MuJoCo, implementing
    `rsl_rl.env.VecEnv`'s duck-typed interface (get_observations/step/
    reset, `num_envs`/`num_actions`/`episode_length_buf`/`device`/`cfg`
    attributes) without subclassing it (avoids an unnecessary import-time
    dependency on rsl_rl's exact ABC version)."""

    def __init__(
        self,
        num_envs: int,
        device: str = "cpu",
        # Default (0.02, 0.08) matched Isaac's own tiny-forward stage for a
        # fair engine comparison -- but replay-validating the converged
        # `op3_mujoco_walk_easy_rebalanced` checkpoint (reward ~98.4, 100%
        # episode survival) found net x-displacement of exactly 0.0m over a
        # full 1000-step/20s episode on every one of 8 replayed envs, with
        # only 4-8 foot contact-state switches total (real walking would be
        # dozens+). The policy found a reward-hacking equilibrium: standing
        # (near-)still scores almost as well as tracking because the
        # commanded speed itself is so small the tracking reward's
        # exp(-error/0.25) shape gives large partial credit for zero
        # velocity, while standing carries none of stepping's fall risk.
        # See STAGE4_PPO_TRAINING.md. Callers escalating past this stage
        # should raise the floor well above the tracking tolerance so
        # standing still is no longer competitive.
        lin_vel_x_range: tuple[float, float] = (0.02, 0.08),
        ang_vel_z_range: tuple[float, float] = (0.0, 0.0),
        max_episode_length_s: float = _EPISODE_LENGTH_S,
        dof_torques_weight: float = -1.0e-4,
        feet_air_time_weight: float = 0.125,
        foot_load_asymmetry_weight: float = 0.0,
        reset_noise_scale: float = 0.02,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.cfg = {
            "dof_torques_weight": dof_torques_weight,
            "feet_air_time_weight": feet_air_time_weight,
            "foot_load_asymmetry_weight": foot_load_asymmetry_weight,
        }
        self.lin_vel_x_range = lin_vel_x_range
        self.ang_vel_z_range = ang_vel_z_range
        self.dof_torques_weight = dof_torques_weight
        self.feet_air_time_weight = feet_air_time_weight
        self.foot_load_asymmetry_weight = foot_load_asymmetry_weight
        self.reset_noise_scale = reset_noise_scale

        self.model = mujoco.MjModel.from_xml_path(str(_MODEL_PATH))
        self.dt = self.model.opt.timestep * _DECIMATION
        self.max_episode_length = int(round(max_episode_length_s / self.dt))

        self.num_actions = len(_ALL_JOINTS)
        self.num_obs = 3 + 3 + 3 + 3 + self.num_actions + self.num_actions + self.num_actions  # 72, matches Isaac

        self._joint_qpos_adr = np.array([self.model.joint(name).qposadr[0] for name in _ALL_JOINTS])
        self._joint_qvel_adr = np.array([self.model.joint(name).dofadr[0] for name in _ALL_JOINTS])
        self._joint_dof_ids = np.array([self.model.joint(name).id for name in _ALL_JOINTS])
        self._actuator_ids = np.array([self.model.actuator(f"{name}_act").id for name in _ALL_JOINTS])
        self._default_pos = np.array([_DEFAULT_JOINT_POS.get(name, 0.0) for name in _ALL_JOINTS])
        self._arm_head_mask = np.array([name in _ARM_HEAD_JOINTS for name in _ALL_JOINTS])

        self._torso_body_id = self.model.body(_TORSO_BODY).id
        self._floor_geom_id = self.model.geom("floor").id
        self._feet_body_ids = [self.model.body(n).id for n in _FEET_BODIES]
        self._undesired_body_ids = [self.model.body(n).id for n in _UNDESIRED_CONTACT_BODIES]

        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]

        # mj_step releases the GIL for its C computation, and MuJoCo
        # explicitly supports one MjModel (read-only) shared across threads
        # each stepping their own MjData -- so threading the per-env physics
        # loop below lets it use idle cores instead of running serially.
        self._step_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(num_envs, os.cpu_count() or 1)
        )

        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.commands = torch.zeros(num_envs, 3, device=self.device)  # lin_vel_x, lin_vel_y(=0), ang_vel_z(=0)
        self.last_actions = torch.zeros(num_envs, self.num_actions, device=self.device)
        self.actions = torch.zeros(num_envs, self.num_actions, device=self.device)
        self._air_time = np.zeros((num_envs, len(_FEET_BODIES)), dtype=np.float64)
        self._last_qacc = np.zeros((num_envs, self.num_actions), dtype=np.float64)

        for i in range(num_envs):
            self._reset_idx(i)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _reset_idx(self, i: int) -> None:
        d = self.datas[i]
        mujoco.mj_resetData(self.model, d)
        d.qpos[:3] = (0.0, 0.0, _BASE_HEIGHT)
        d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        d.qvel[:] = 0.0
        noise = np.random.uniform(-self.reset_noise_scale, self.reset_noise_scale, size=self.num_actions)
        d.qpos[self._joint_qpos_adr] = self._default_pos + noise
        d.ctrl[self._actuator_ids] = self._default_pos
        mujoco.mj_forward(self.model, d)
        self.episode_length_buf[i] = 0
        self.last_actions[i] = 0.0
        self._air_time[i] = 0.0
        cmd = np.random.uniform(self.lin_vel_x_range[0], self.lin_vel_x_range[1])
        self.commands[i, 0] = cmd
        self.commands[i, 1] = 0.0
        self.commands[i, 2] = np.random.uniform(self.ang_vel_z_range[0], self.ang_vel_z_range[1])

    def reset(self) -> tuple[TensorDict, dict]:
        for i in range(self.num_envs):
            self._reset_idx(i)
        return self.get_observations(), {}

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """World-frame vec -> body-frame vec, using MuJoCo's (w,x,y,z) quat."""
        w, x, y, z = quat_wxyz
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
                [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        return R @ vec

    def _obs_for(self, i: int) -> np.ndarray:
        d = self.datas[i]
        quat = d.qpos[3:7]
        lin_vel_w = d.qvel[0:3]
        ang_vel_w = d.qvel[3:6]
        base_lin_vel = self._quat_rotate_inverse(quat, lin_vel_w)
        base_ang_vel = self._quat_rotate_inverse(quat, ang_vel_w)
        gravity_w = np.array([0.0, 0.0, -1.0])
        projected_gravity = self._quat_rotate_inverse(quat, gravity_w)
        joint_pos_rel = d.qpos[self._joint_qpos_adr] - self._default_pos
        joint_vel = d.qvel[self._joint_qvel_adr]
        cmd = self.commands[i].cpu().numpy()
        last_action = self.last_actions[i].cpu().numpy()
        return np.concatenate(
            [base_lin_vel, base_ang_vel, projected_gravity, cmd, joint_pos_rel, joint_vel, last_action]
        ).astype(np.float32)

    def get_observations(self) -> TensorDict:
        obs = np.stack([self._obs_for(i) for i in range(self.num_envs)])
        obs_t = torch.from_numpy(obs).to(self.device)
        return TensorDict({"policy": obs_t}, batch_size=[self.num_envs])

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor):
        self.actions = actions.detach().to(self.device)
        actions_np = self.actions.cpu().numpy()
        targets = self._default_pos[None, :] + _ACTION_SCALE * actions_np

        rewards = np.zeros(self.num_envs, dtype=np.float64)
        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = np.zeros(self.num_envs, dtype=bool)
        applied_torque_abs_max = np.zeros(self.num_envs, dtype=np.float64)

        def _physics_step(i: int) -> None:
            d = self.datas[i]
            d.ctrl[self._actuator_ids] = targets[i]
            for _ in range(_DECIMATION):
                mujoco.mj_step(self.model, d)

        list(self._step_pool.map(_physics_step, range(self.num_envs)))

        for i in range(self.num_envs):
            d = self.datas[i]
            rewards[i], dones[i], time_outs[i] = self._reward_and_done(i, actions_np[i])
            applied_torque_abs_max[i] = np.max(np.abs(d.actuator_force[self._actuator_ids]))

        self.episode_length_buf += 1
        self.last_actions = self.actions.clone()

        for i in range(self.num_envs):
            if dones[i] or self.episode_length_buf[i] >= self.max_episode_length:
                dones[i] = self.episode_length_buf[i] >= self.max_episode_length or dones[i]
                self._reset_idx(i)

        obs = self.get_observations()
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        extras = {
            "time_outs": torch.from_numpy(time_outs).to(self.device),
            "log": {
                "mean_applied_torque_abs_max": float(np.mean(applied_torque_abs_max)),
            },
        }
        return obs, rewards_t, dones_t, extras

    # ------------------------------------------------------------------ #
    # Reward + termination (matches OP3WalkEasyRebalancedEnvCfg's weights)
    # ------------------------------------------------------------------ #
    def _body_in_contact_with_floor(self, d, body_id: int, threshold: float = 1.0) -> bool:
        for c in range(d.ncon):
            con = d.contact[c]
            g1_body = self.model.geom_bodyid[con.geom1]
            g2_body = self.model.geom_bodyid[con.geom2]
            floor_involved = con.geom1 == self._floor_geom_id or con.geom2 == self._floor_geom_id
            other_body = g2_body if con.geom1 == self._floor_geom_id else g1_body
            if floor_involved and other_body == body_id:
                return True
        return False

    def _foot_contact_force(self, d, body_id: int) -> float:
        """Sum normal contact-force magnitude between `body_id` and the floor.

        Port of Isaac Lab's `op3_mdp.foot_load_asymmetry_bonus`'s contact-
        force read (`ContactSensor.data.net_forces_w`), but from MuJoCo's own
        contact list via `mj_contactForce` since there's no equivalent
        sensor object here.
        """
        total = 0.0
        result = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            g1_body = self.model.geom_bodyid[con.geom1]
            g2_body = self.model.geom_bodyid[con.geom2]
            floor_involved = con.geom1 == self._floor_geom_id or con.geom2 == self._floor_geom_id
            other_body = g2_body if con.geom1 == self._floor_geom_id else g1_body
            if floor_involved and other_body == body_id:
                mujoco.mj_contactForce(self.model, d, c, result)
                total += abs(float(result[0]))
        return total

    def _reward_and_done(self, i: int, action: np.ndarray) -> tuple[float, bool, bool]:
        d = self.datas[i]
        quat = d.qpos[3:7]
        lin_vel_w = d.qvel[0:3]
        ang_vel_w = d.qvel[3:6]
        base_lin_vel = self._quat_rotate_inverse(quat, lin_vel_w)
        base_ang_vel = self._quat_rotate_inverse(quat, ang_vel_w)
        gravity_w = np.array([0.0, 0.0, -1.0])
        projected_gravity = self._quat_rotate_inverse(quat, gravity_w)

        cmd = self.commands[i].cpu().numpy()
        lin_vel_err = np.sum(np.square(cmd[:2] - base_lin_vel[:2]))
        ang_vel_err = np.square(cmd[2] - base_ang_vel[2])
        track_lin_vel_xy_exp = math.exp(-lin_vel_err / _VEL_TRACKING_VARIANCE)
        track_ang_vel_z_exp = math.exp(-ang_vel_err / _VEL_TRACKING_VARIANCE)

        lin_vel_z_l2 = base_lin_vel[2] ** 2
        ang_vel_xy_l2 = float(np.sum(np.square(base_ang_vel[:2])))
        flat_orientation_l2 = float(np.sum(np.square(projected_gravity[:2])))

        torque = d.actuator_force[self._actuator_ids]
        dof_torques_l2 = float(np.sum(np.square(torque)))

        qacc = d.qacc[self._joint_qvel_adr]
        dof_acc_l2 = float(np.sum(np.square(qacc)))

        last_action = self.last_actions[i].cpu().numpy()
        action_rate_l2 = float(np.sum(np.square(action - last_action)))

        joint_pos = d.qpos[self._joint_qpos_adr]
        joint_deviation_arms_head = float(
            np.sum(np.abs(joint_pos[self._arm_head_mask] - self._default_pos[self._arm_head_mask]))
        )

        undesired_contacts = 0.0
        for bid in self._undesired_body_ids:
            if self._body_in_contact_with_floor(d, bid):
                undesired_contacts = 1.0
                break

        feet_air_time_reward = 0.0
        cmd_moving = np.linalg.norm(cmd[:2]) > 0.1
        for k, bid in enumerate(self._feet_body_ids):
            in_contact = self._body_in_contact_with_floor(d, bid)
            if in_contact:
                if self._air_time[i, k] > 0.0 and cmd_moving:
                    feet_air_time_reward += self._air_time[i, k] - 0.5
                self._air_time[i, k] = 0.0
            else:
                self._air_time[i, k] += self.dt

        foot_load_asymmetry = 0.0
        if self.foot_load_asymmetry_weight != 0.0 and cmd_moving:
            foot_forces = [self._foot_contact_force(d, bid) for bid in self._feet_body_ids]
            total_force = max(sum(foot_forces), 1e-6)
            foot_load_asymmetry = abs(foot_forces[0] - foot_forces[1]) / total_force

        terminated = self._body_in_contact_with_floor(d, self._torso_body_id) or bool(np.any(np.isnan(d.qpos)))
        time_out = self.episode_length_buf[i].item() + 1 >= self.max_episode_length

        reward = (
            3.0 * track_lin_vel_xy_exp
            + 1.5 * track_ang_vel_z_exp
            + -2.0 * lin_vel_z_l2
            + -0.05 * ang_vel_xy_l2
            + self.dof_torques_weight * dof_torques_l2
            + -2.5e-7 * dof_acc_l2
            + -0.01 * action_rate_l2
            + self.feet_air_time_weight * feet_air_time_reward
            + self.foot_load_asymmetry_weight * foot_load_asymmetry
            + -1.0 * undesired_contacts
            + -2.5 * flat_orientation_l2
            + -50.0 * float(terminated)
            + 0.5 * 1.0
            + -0.1 * joint_deviation_arms_head
        ) * self.dt  # dt-scaled, matching Isaac Lab's RewardManager convention

        return reward, terminated, time_out
