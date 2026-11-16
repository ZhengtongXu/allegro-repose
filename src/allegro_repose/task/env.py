"""mjlab environment: fixed-wrist Allegro hand cube repose (Isaac Lab Allegro Direct task).

Every MDP signal (observation, reward, termination) comes from the pure-torch functions in
``allegro_repose.task.core``; this module wires them into mjlab managers.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from allegro_repose import math3d as m3
from allegro_repose.assets import FINGERTIP_BODIES, make_cube_spec_fn, make_hand_spec_fn, palm_pose
from allegro_repose.task import core
from allegro_repose.task.kinematics import HandKinematics

ROBOT = "robot"
CUBE = "cube"
REWARD_TERMS = ("dist", "rot", "action_penalty", "action_rate", "reach_goal", "fall")


def hand_model(hand_cfg: dict) -> mujoco.MjModel:
    """The hand alone (palm at the origin), compiled on the CPU."""
    return make_hand_spec_fn(hand_cfg)().compile()


def joint_limits(m: mujoco.MjModel) -> tuple[list[str], list[float], list[float]]:
    names = [m.joint(i).name for i in range(m.njnt)]
    return names, m.jnt_range[:, 0].tolist(), m.jnt_range[:, 1].tolist()


# ----------------------------------------------------------------------------------------
# State accessors (shared by the observation, reward and termination terms).
# ----------------------------------------------------------------------------------------


class Handles:
    """Cached indices into the batched MuJoCo-Warp state."""

    def __init__(self, env: ManagerBasedRlEnv):
        robot, cube = env.scene[ROBOT], env.scene[CUBE]
        self.robot, self.cube = robot, cube
        self.joint_ids, self.joint_names = robot.find_joints(env.joint_names, preserve_order=True)
        self.joint_ids = torch.tensor(self.joint_ids, device=env.device)
        self.q_adr = robot.indexing.joint_q_adr[self.joint_ids]
        self.v_adr = robot.indexing.joint_v_adr[self.joint_ids]
        self.cube_q_adr = cube.indexing.free_joint_q_adr
        self.cube_v_adr = cube.indexing.free_joint_v_adr
        mjm = env.sim.mj_model
        jnt = [mjm.joint(f"{ROBOT}/{n}").id for n in self.joint_names]
        act_joint = [int(mjm.actuator_trnid[robot.indexing.ctrl_ids[i].item(), 0]) for i in range(core.NJ)]
        assert act_joint == jnt, "actuator order must follow joint order"


def handles(env: ManagerBasedRlEnv) -> Handles:
    h = getattr(env, "_handles", None)
    if h is None:
        h = env._handles = Handles(env)
    return h


def get_state(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Physical state [N, 45] read straight from qpos/qvel."""
    h = handles(env)
    qpos, qvel = env.sim.data.qpos, env.sim.data.qvel
    cq, cv = qpos[:, h.cube_q_adr], qvel[:, h.cube_v_adr]
    quat = cq[:, 3:7]
    w_world = m3.quat_rotate(quat, cv[:, 3:6])  # free-joint angular velocity is body-local
    return torch.cat((qpos[:, h.q_adr], qvel[:, h.v_adr], cq[:, :3] - env.scene.env_origins, quat,
                      cv[:, :3], w_world), dim=-1)


# ----------------------------------------------------------------------------------------
# Goal command: goal orientation and success counting.
# ----------------------------------------------------------------------------------------


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
    resampling_time_range: tuple[float, float] = (1e9, 1e9)  # only on success / reset

    def build(self, env):
        return GoalCommand(self, env)


class GoalCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, d = self.num_envs, self.device
        self.goal = torch.zeros(n, 4, device=d)
        self.goal[:, 0] = 1
        self.metrics["successes"] = torch.zeros(n, device=d)
        self.metrics["rot_dist"] = torch.zeros(n, device=d)
        self.metrics["on_goal_frac"] = torch.zeros(n, device=d)
        self._steps = torch.zeros(n, device=d)
        self._err_sum = torch.zeros(n, device=d)
        self._on_goal = torch.zeros(n, device=d)
        # goal.dwell_steps: a goal counts and is replaced only after the cube stayed within tolerance
        # for this many consecutive steps (1 = Isaac Lab: replaced on the first step it is reached).
        self.dwell = int(env.exp_cfg["task"]["goal"].get("dwell_steps", 1))
        self.hold_steps = torch.zeros(n, dtype=torch.long, device=d)

    @property
    def command(self):
        return self.goal

    def reset(self, env_ids):
        self._steps[env_ids] = 0
        self._err_sum[env_ids] = 0
        self._on_goal[env_ids] = 0
        self.hold_steps[env_ids] = 0
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        self.goal[env_ids] = core.random_rotation(len(env_ids), self.device)

    def _update_metrics(self):
        # After the step's rewards: count the success and move to a new goal (Isaac Lab
        # _reset_target_pose on goal_env_ids).
        rot = core.rotation_distance(get_state(self._env)[:, core.SL_QUAT], self.goal)
        self._steps += 1
        self._err_sum += rot
        self.metrics["rot_dist"][:] = self._err_sum / self._steps
        hit = rot <= self._env.spec.success_tol
        self._on_goal += hit.float()
        self.metrics["on_goal_frac"][:] = self._on_goal / self._steps
        self.hold_steps = torch.where(hit, self.hold_steps + 1, torch.zeros_like(self.hold_steps))
        done = self.hold_steps >= self.dwell
        self.metrics["successes"] += done.float()
        ids = done.nonzero().flatten()
        if len(ids):
            self.hold_steps[ids] = 0
            self._resample_command(ids)

    def _update_command(self):
        pass


def _goal(env) -> GoalCommand:
    return env.command_manager.get_term("goal")


# ----------------------------------------------------------------------------------------
# Observation / reward / termination / reset terms.
# ----------------------------------------------------------------------------------------


# compiled: the fingertip kinematics are many small kernels when run eagerly
_observation = torch.compile(core.observation)
_actor_rot6d = torch.compile(core.actor_obs_rot6d)
_critic_privileged_rot6d = torch.compile(core.critic_privileged_rot6d)


def obs_actor(env):
    fn = _observation if env.spec.obs_layout == "isaaclab" else _actor_rot6d
    env._actor_obs = fn(get_state(env), _goal(env).goal, env.action_manager.action, env.spec)
    return env._actor_obs


def obs_critic(env):
    # The observation manager computes the actor group right before this one on the same state:
    # reuse it (one fingertip-kinematics pass per step) and add the critic's privileged part.
    actor, env._actor_obs = env._actor_obs, None
    if actor is None:
        actor = obs_actor(env)
        env._actor_obs = None
    if env.spec.obs_layout == "isaaclab":
        return actor  # symmetric
    return torch.cat((actor, _critic_privileged_rot6d(get_state(env), env.spec)), dim=-1)


def reward_term(env, term: str):
    # All terms are computed together once per step, by the first term (the reward manager runs
    # them in REWARD_TERMS order on the same state).
    if term == REWARD_TERMS[0]:
        am = env.action_manager
        env._rewards = core.reward_terms(get_state(env), _goal(env).goal, am.action, env.spec,
                                         prev_action=am.prev_action)
    return env._rewards[term]


def termination_fall(env):
    return core.termination(get_state(env), env.spec)


def termination_time_out(env):
    return env.episode_length_buf >= env.max_episode_length - 1  # Isaac Lab: >= max_episode_length - 1


def reset_hand_and_cube(env, env_ids, task_cfg: dict):
    """Isaac Lab ``_reset_idx``: cube at its default position + uniform noise with a random
    orientation and zero velocity; hand at the default joint positions + Isaac Lab's noise."""
    h, n, d, r = handles(env), len(env_ids), env.device, task_cfg["reset"]
    u = lambda *s: torch.rand(*s, device=d) * 2 - 1
    p = torch.as_tensor(task_cfg["cube"]["init_pos"], device=d).float() + r["position_noise"] * u(n, 3)
    root = torch.cat((p + env.scene.env_origins[env_ids], core.random_rotation(n, d),
                      torch.zeros(n, 6, device=d)), dim=-1)
    h.cube.write_root_state_to_sim(root, env_ids=env_ids)
    lo, hi, q0 = env.spec.joint_lo, env.spec.joint_hi, env.default_joint_pos
    delta_max, delta_min = hi - q0, lo - q0
    # Isaac Lab (from IsaacGymEnvs): rand_delta = delta_min + (delta_max - delta_min) * 0.5 * U(-1, 1)
    rand_delta = delta_min + (delta_max - delta_min) * 0.5 * u(n, core.NJ)
    q = torch.clamp(q0 + r["dof_pos_noise"] * rand_delta, lo, hi)
    qd = r["dof_vel_noise"] * u(n, core.NJ)
    h.robot.write_joint_state_to_sim(q, qd, joint_ids=h.joint_ids, env_ids=env_ids)


# ----------------------------------------------------------------------------------------
# Config + construction.
# ----------------------------------------------------------------------------------------


def make_env_cfg(task_cfg: dict, joints: tuple[list[str], list[float], list[float]], num_envs: int,
                 seed: int | None = None, goal_marker: bool = False) -> ManagerBasedRlEnvCfg:
    """``joints``: names and lower / upper limits of the hand joints (``joint_limits``)."""
    hand_cfg, sim_cfg = task_cfg["hand"], task_cfg["sim"]
    palm_pos, palm_quat = palm_pose(hand_cfg)
    hand_entity = EntityCfg(
        spec_fn=make_hand_spec_fn(hand_cfg),
        init_state=EntityCfg.InitialStateCfg(pos=palm_pos, rot=palm_quat),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)),
    )
    cube_entity = EntityCfg(
        spec_fn=make_cube_spec_fn(task_cfg["cube"], goal_marker=goal_marker),
        init_state=EntityCfg.InitialStateCfg(pos=tuple(task_cfg["cube"]["init_pos"])),
    )
    names, lo, hi = joints
    actions = {
        "joint_pos": JointPositionActionCfg(
            entity_name=ROBOT, actuator_names=(".*",), preserve_order=True, use_default_offset=False,
            scale={k: (b - a) / 2 for k, a, b in zip(names, lo, hi)},
            offset={k: (b + a) / 2 for k, a, b in zip(names, lo, hi)},
            clip={k: (a, b) for k, a, b in zip(names, lo, hi)},
        )
    }
    events = {
        "reset_state": EventTermCfg(func=reset_hand_and_cube, mode="reset", params={"task_cfg": task_cfg}),
    }
    rewards = {k: RewardTermCfg(func=reward_term, weight=1.0, params={"term": k}) for k in REWARD_TERMS}
    terminations = {
        "time_out": TerminationTermCfg(func=termination_time_out, time_out=True),
        "fall": TerminationTermCfg(func=termination_fall),
    }
    observations = {
        "actor": ObservationGroupCfg({"obs": ObservationTermCfg(func=obs_actor)}, enable_corruption=False),
        "critic": ObservationGroupCfg({"obs": ObservationTermCfg(func=obs_critic)}, enable_corruption=False),
    }
    return ManagerBasedRlEnvCfg(
        decimation=int(sim_cfg["decimation"]),
        scene=SceneCfg(num_envs=num_envs, env_spacing=0.75, entities={ROBOT: hand_entity, CUBE: cube_entity}),
        observations=observations,
        actions=actions,
        events=events,
        commands={"goal": GoalCommandCfg()},
        rewards=rewards,
        terminations=terminations,
        episode_length_s=float(task_cfg["episode_length_s"]),
        scale_rewards_by_dt=False,
        seed=seed,
        sim=SimulationCfg(
            njmax=int(sim_cfg["njmax"]),
            nconmax=int(sim_cfg["nconmax"]),
            mujoco=MujocoCfg(
                timestep=float(sim_cfg["physics_dt"]),
                integrator="implicitfast",
                iterations=int(sim_cfg["solver_iterations"]),
                ls_iterations=int(sim_cfg["ls_iterations"]),
                ccd_iterations=int(sim_cfg["ccd_iterations"]),
                impratio=float(sim_cfg["impratio"]),
                cone=sim_cfg["cone"],
            ),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            # looking down onto the palm, with the goal marker beside the hand
            lookat=(0.10, -0.19, 0.55), distance=0.5, elevation=-50.0, azimuth=110.0, width=960, height=720,
            max_extra_envs=0,
        ),
    )


class ReposeEnv(ManagerBasedRlEnv):
    """ManagerBasedRlEnv + task constants and helpers for the trainers."""

    def __init__(self, cfg: dict, num_envs: int, device: str, seed: int | None = None,
                 render_mode: str | None = None):
        self.exp_cfg = cfg
        task_cfg = cfg["task"]
        hand = hand_model(task_cfg["hand"])
        joints = joint_limits(hand)
        self.joint_names, lo, hi = joints
        palm_pos, palm_quat = palm_pose(task_cfg["hand"])
        kin = HandKinematics(hand, self.joint_names, list(FINGERTIP_BODIES), palm_pos, palm_quat, device)
        # The spec must exist before the managers are built (the command and resets use it).
        self._spec = core.TaskSpec.from_cfg(task_cfg, lo, hi, kin, device)
        self._actor_obs, self._rewards = None, {}
        dflt = task_cfg["hand"].get("default_joint_pos", {})
        self.default_joint_pos = torch.tensor([float(dflt.get(n, 0.0)) for n in self.joint_names], device=device)
        env_cfg = make_env_cfg(task_cfg, joints, num_envs, seed, goal_marker=render_mode is not None)
        super().__init__(env_cfg, device=device, render_mode=render_mode)

    def load_managers(self):
        super().load_managers()
        assert handles(self).joint_names == self.joint_names

    @property
    def spec(self) -> core.TaskSpec:
        return self._spec

    def state(self) -> torch.Tensor:
        return get_state(self)

    @property
    def goal(self) -> torch.Tensor:
        return _goal(self).goal
