"""G1 fall-recovery (get-up) task for mjlab: task id ``G1-Getup``."""

from __future__ import annotations

import os

import mjlab.envs.mdp as mdp
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import getup_mdp as gmdp

POOL = os.environ.get(
  "G1GETUP_POOL", "/path/to/workspace/data/floor_pool.npz"
)
HISTORY = 5
STEPS_PER_ENV = 24
ASSIST_DECAY_ITERS = 4000

NON_FOOT_BODIES = (
  r"^(pelvis|torso_link|.*_hip_.*_link|.*_knee_link|.*_shoulder_.*_link|.*_elbow_link|"
  r".*_wrist_.*_link|.*_rubber_hand)$"
)


def _actor_terms(noise: bool) -> dict[str, ObservationTermCfg]:
  def n(w):
    return Unoise(n_min=-w, n_max=w) if noise else None

  return {
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=n(0.05), history_length=HISTORY),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=n(0.2), history_length=HISTORY
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"biased": noise}, noise=n(0.01), history_length=HISTORY
    ),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=n(1.5), history_length=HISTORY),
    "actions": ObservationTermCfg(func=mdp.last_action, history_length=HISTORY),
  }


def dc_motor_robot_cfg(stall_ratio: float = 1.5):
  """G1 with torque-speed-limited motors: rated (mjlab) torque up to the corner speed,
  falling linearly to zero at the no-load speed (mjlab's velocity limits)."""
  from dataclasses import replace

  from mjlab.actuator.dc_actuator import DcMotorActuatorCfg
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as G

  vlim = {
    id(G.G1_ACTUATOR_5020): G.ACTUATOR_5020.velocity_limit,
    id(G.G1_ACTUATOR_7520_14): G.ACTUATOR_7520_14.velocity_limit,
    id(G.G1_ACTUATOR_7520_22): G.ACTUATOR_7520_22.velocity_limit,
    id(G.G1_ACTUATOR_4010): G.ACTUATOR_4010.velocity_limit,
    id(G.G1_ACTUATOR_WAIST): G.ACTUATOR_5020.velocity_limit,
    id(G.G1_ACTUATOR_ANKLE): G.ACTUATOR_5020.velocity_limit,
  }
  acts = []
  for a in G.G1_ARTICULATION.actuators:
    acts.append(DcMotorActuatorCfg(
      target_names_expr=a.target_names_expr, stiffness=a.stiffness, damping=a.damping,
      effort_limit=a.effort_limit, saturation_effort=stall_ratio * a.effort_limit,
      velocity_limit=vlim[id(a)], armature=a.armature,
    ))
  cfg = get_g1_robot_cfg()
  cfg.articulation = replace(cfg.articulation, actuators=tuple(acts))
  return cfg


REF = os.environ.get(
  "G1GETUP_REF", "/path/to/workspace/data/getup_ref.npz"
)


def getup_env_cfg(play: bool = False, dc_motor: bool = False, gentle: bool = False,
                  compact: bool = False) -> ManagerBasedRlEnvCfg:
  robot = dc_motor_robot_cfg() if dc_motor else get_g1_robot_cfg()
  feet = ContactSensorCfg(
    name="feet_ground",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
  )
  body = ContactSensorCfg(
    name="body_ground",
    primary=ContactMatch(mode="body", pattern=NON_FOOT_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"), reduce="netforce", num_slots=1,
  )

  critic = {
    **_actor_terms(noise=False),
    "heights": ObservationTermCfg(func=gmdp.body_heights),
    "root_lin_vel_b": ObservationTermCfg(func=gmdp.root_lin_vel_b),
    "assist": ObservationTermCfg(func=gmdp.assist_obs),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=_actor_terms(noise=not play), concatenate_terms=True, enable_corruption=not play),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }
  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=G1_ACTION_SCALE, use_default_offset=True
    )
  }
  probs = {"pool": 0.45, "drop": 0.25, "trip": 0.15, "stand": 0.15}
  events = {
    "reset": EventTermCfg(func=gmdp.reset_getup, mode="reset", params={"pool_path": POOL, "probs": probs}),
    "friction": EventTermCfg(
      mode="startup", func=dr.geom_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)), "operation": "abs",
              "ranges": (0.3, 1.2), "shared_random": True},
    ),
    "base_com": EventTermCfg(
      mode="startup", func=dr.body_com_offset,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add",
              "ranges": {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}},
    ),
    "encoder_bias": EventTermCfg(
      mode="startup", func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.015, 0.015)},
    ),
    "push": EventTermCfg(
      func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(4.0, 8.0),
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)}},
    ),
  }
  if not play and not dc_motor:  # v2 (dc_motor) fine-tunes from v1 without assist
    events["assist"] = EventTermCfg(
      func=gmdp.assist_force, mode="step",
      params={"max_force": 250.0, "decay_steps": ASSIST_DECAY_ITERS * STEPS_PER_ENV},
    )
  rewards = {
    "pelvis_height": RewardTermCfg(func=gmdp.pelvis_height, weight=1.0),
    "torso_height": RewardTermCfg(func=gmdp.torso_height, weight=1.0),
    "upright": RewardTermCfg(func=gmdp.torso_upright, weight=1.0),
    "stand_pose": RewardTermCfg(func=gmdp.stand_pose, weight=3.0, params={"std": 0.6}),
    "stand_still": RewardTermCfg(func=gmdp.stand_still, weight=1.0),
    "stand_feet": RewardTermCfg(func=gmdp.stand_feet_contact, weight=0.5, params={"sensor_name": "feet_ground"}),
    "stand_body_contact": RewardTermCfg(
      func=gmdp.stand_body_contact, weight=-1.0, params={"sensor_name": "body_ground"}
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-5e-4),
    "joint_vel_excess": RewardTermCfg(func=gmdp.joint_vel_excess, weight=-0.2),
    "torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-5),
    "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "impact": RewardTermCfg(func=gmdp.impact, weight=-0.5, params={"sensor_name": "body_ground"}),
  }
  if dc_motor:  # gentler, hardware-friendly motion
    rewards["action_rate"].weight = -0.1
    rewards["torques"].weight = -2e-5
    rewards["joint_vel_excess"].params = {"limit": 12.0}
    rewards["joint_vel_excess"].weight = -0.5
    rewards["impact"].weight = -1.0
  if gentle:  # v3: explicit hardware margins
    rewards["action_rate"].weight = -0.2
    rewards["speed_margin"] = RewardTermCfg(func=gmdp.joint_speed_margin, weight=-2.0, params={"frac": 0.65})
    rewards["torque_margin"] = RewardTermCfg(func=gmdp.torque_saturation, weight=-1.0, params={"frac": 0.85})
  if compact:  # v4: compact, human-like get-up (BONES-SEED human get-up prior + footprint limits)
    rewards["style_prior"] = RewardTermCfg(
      func=gmdp.style_prior, weight=float(os.environ.get("G1GETUP_STYLE_W", "2.0")),
      params={"ref_path": REF, "sigma": float(os.environ.get("G1GETUP_STYLE_SIGMA", "1.0")),
              "gate_standing": os.environ.get("G1GETUP_STYLE_GATE", "1") == "1"})
    rewards["limb_splay"] = RewardTermCfg(
      func=gmdp.limb_splay, weight=float(os.environ.get("G1GETUP_LIMB_W", "-2.0")),
      params={"limits": {"hip_roll": 0.6, "hip_yaw": 0.8, "shoulder_roll": 0.8}})
    rewards["hand_reach"] = RewardTermCfg(func=gmdp.hand_reach, weight=float(os.environ.get("G1GETUP_LIMB_W", "-2.0")),
                                          params={"limit": 0.5})
    rewards["footprint"] = RewardTermCfg(func=gmdp.footprint, weight=float(os.environ.get("G1GETUP_FP_W", "-4.0")),
                                         params={"margin": 0.1})
  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "nan": TerminationTermCfg(func=mdp.nan_detection),
  }
  if compact and os.environ.get("G1GETUP_FP_TERM"):  # v6: virtual wall around the lying footprint
    terminations["footprint"] = TerminationTermCfg(
      func=gmdp.footprint_violation, params={"limit": float(os.environ["G1GETUP_FP_TERM"])})
  metrics = {"stood_up": MetricsTermCfg(func=gmdp.stood_up)}
  if compact:
    metrics["style_d2"] = MetricsTermCfg(func=gmdp.style_dist)
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": robot},
      sensors=(feet, body), num_envs=1, extent=2.0,
    ),
    observations=observations, actions=actions, events=events, rewards=rewards,
    terminations=terminations, metrics=metrics,
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                        body_name="torso_link", distance=3.0, elevation=-10.0, azimuth=90.0),
    sim=SimulationCfg(nconmax=128, njmax=800, contact_sensor_maxmatch=500,
                      mujoco=MujocoCfg(timestep=0.005, iterations=20, ls_iterations=30)),
    decimation=4,
    episode_length_s=10.0,
  )


def floor_terrain_cfg():
  """45% flat, 55% uneven paving (<= 1.5-2.5 cm bumps); copy of tools/robust_task.floor_terrain_cfg."""
  from mjlab.terrains import BoxFlatTerrainCfg, HfRandomUniformTerrainCfg, HfWaveTerrainCfg
  from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
  return TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=10.0, num_rows=10, num_cols=20, curriculum=False,
    sub_terrains={
      "flat": BoxFlatTerrainCfg(proportion=0.45),
      "tiles": HfRandomUniformTerrainCfg(proportion=0.20, noise_range=(0.0, 0.015), noise_step=0.005,
                                         downsampled_scale=0.15, horizontal_scale=0.05, border_width=0.25),
      "undulate": HfRandomUniformTerrainCfg(proportion=0.20, noise_range=(0.0, 0.01), noise_step=0.005,
                                            downsampled_scale=0.4, horizontal_scale=0.1, border_width=0.25),
      "waves": HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.005, 0.025), num_waves=3,
                                horizontal_scale=0.1, border_width=0.25),
    },
    add_lights=True,
  )


SETTLE_CLIP = os.environ.get(
  "G1SETTLE_CLIP",
  "/path/to/workspace/exports/refslow/stand_up_lying_R_002__A473_x1.6.npz",
)


def settle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stage 1 of the human-style get-up: from any fallen pose, move gently to the lying pose
  that starts the human get-up clip (then the SuperMimic tracker plays the clip)."""
  cfg = getup_env_cfg(play=play, dc_motor=True)
  cfg.events["reset"].params["probs"] = {"pool": 0.5, "drop": 0.3, "trip": 0.2, "stand": 0.0}
  cfg.events.pop("assist", None)
  c = {"clip_path": SETTLE_CLIP}
  cfg.rewards = {
    "orient": RewardTermCfg(func=gmdp.settle_orient, weight=2.0, params=c),
    "pose": RewardTermCfg(func=gmdp.settle_pose, weight=3.0, params=c),
    "still": RewardTermCfg(func=gmdp.settle_still, weight=2.0, params=c),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=float(os.environ.get("G1SETTLE_AR", "-0.05"))),
    "joint_vel": RewardTermCfg(func=mdp.joint_vel_l2, weight=-5e-4),
    "speed_margin": RewardTermCfg(func=gmdp.joint_speed_margin, weight=-3.0, params={"frac": 0.5}),
    "torque_margin": RewardTermCfg(func=gmdp.torque_saturation, weight=-1.0, params={"frac": 0.8}),
    "torques": RewardTermCfg(func=mdp.joint_torques_l2, weight=-2e-5),
    "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
    "impact": RewardTermCfg(func=gmdp.impact, weight=-1.0, params={"sensor_name": "body_ground"}),
    "limb_splay": RewardTermCfg(func=gmdp.limb_splay, weight=-1.0,
                                params={"limits": {"hip_roll": 0.6, "hip_yaw": 0.8, "shoulder_roll": 1.0}}),
    "footprint": RewardTermCfg(func=gmdp.footprint, weight=-4.0, params={"margin": 0.25}),
  }
  cfg.metrics = {"settled": MetricsTermCfg(func=gmdp.settled, params=c)}
  if not play and os.environ.get("G1SETTLE_SOFT"):  # hard floors and soft mats
    cfg.events["contact_softness"] = EventTermCfg(mode="startup", func=gmdp.contact_softness)
  if os.environ.get("G1SETTLE_TERRAIN"):  # flat floor + gently uneven paving (same mix as the tracker)
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=floor_terrain_cfg())
    cfg.sim.nconmax = 160
    cfg.sim.njmax = 1200
  cfg.episode_length_s = 8.0
  return cfg


def getup_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
      distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "scalar"},
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0,
    ),
    experiment_name="g1_getup", save_interval=500, num_steps_per_env=STEPS_PER_ENV, max_iterations=15000,
  )


register_mjlab_task(
  task_id="G1-Getup",
  env_cfg=getup_env_cfg(),
  play_env_cfg=getup_env_cfg(play=True),
  rl_cfg=getup_ppo_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="G1-Getup-DC",
  env_cfg=getup_env_cfg(dc_motor=True),
  play_env_cfg=getup_env_cfg(play=True, dc_motor=True),
  rl_cfg=getup_ppo_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="G1-Getup-Compact",
  env_cfg=getup_env_cfg(dc_motor=True, gentle=True, compact=True),
  play_env_cfg=getup_env_cfg(play=True, dc_motor=True, gentle=True, compact=True),
  rl_cfg=getup_ppo_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="G1-Getup-DC-Gentle",
  env_cfg=getup_env_cfg(dc_motor=True, gentle=True),
  play_env_cfg=getup_env_cfg(play=True, dc_motor=True, gentle=True),
  rl_cfg=getup_ppo_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="G1-Settle",
  env_cfg=settle_env_cfg(),
  play_env_cfg=settle_env_cfg(play=True),
  rl_cfg=getup_ppo_cfg(),
  runner_cls=None,
)
