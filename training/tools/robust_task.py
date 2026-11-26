"""SuperMimic-GenCoord-G1-Robust: the SuperMimic tracking task plus randomization for real floors.

On top of SuperMimic-GenCoord-G1 (same observations, rewards, actor):
  * soft / hard ground: contact softness of every robot collision geom per env (35% hard floor,
    else up to a thick foam mat the lying robot sinks ~3 cm into)
  * friction of all robot collision geoms (hands, knees, back on a mat), not only feet
  * motor strength: effort limits x0.8-1.05, PD gains x0.85-1.15
  * G1 motors with torque-speed curves (DC motor model, as in the deploy sim2sim)
  * larger joint noise at reset (+-0.2 rad) so the get-up clip can start from an imperfect settle
"""
import os
import sys

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task, load_rl_cfg  # noqa: F401
from mjlab.envs.mdp.dr._core import _randomize_model_field

import supermimic.tasks  # noqa: F401
from supermimic.tasks.env_cfgs import gencoord_env_cfg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "getup"))
from g1getup import dc_motor_robot_cfg  # noqa: E402

_ALL_GEOMS = SceneEntityCfg("robot", geom_names=(".*_collision",))


HARD_PROB = float(os.environ.get("G1R_HARD_PROB", "0.35"))
TERRAIN = os.environ.get("G1R_TERRAIN", "0") == "1"


def floor_terrain_cfg():
  """Mostly flat floor plus gently uneven paving: 45% flat, 20% small random bumps (<=1.5 cm, 10-20 cm
  cells, like loose stone tiles), 20% smooth undulation (<=1 cm over ~40 cm), 15% long waves (<=2.5 cm).
  The policy is blind, so the bumps only teach it to tolerate uneven contact."""
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
U_MAX = float(os.environ.get("G1R_UMAX", "1.0"))


@requires_model_fields("geom_solref", "geom_solimp")
def contact_softness(env, env_ids, hard_prob=None, asset_cfg: SceneEntityCfg = _ALL_GEOMS):
  """Per env floor softness u: hard floor (u=0) with prob hard_prob, else u ~ U(0, 1).
  solref time constant 0.02 -> 0.20 s, solimp dmin 0.9 -> 0.5, width 1 mm -> 40 mm.
  Calibrated in CPU MuJoCo: u=1 lets the lying robot sink ~3 cm (a thick foam mat)."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)
  hard_prob = HARD_PROB if hard_prob is None else hard_prob
  u = U_MAX * torch.rand(n, device=env.device) * (torch.rand(n, device=env.device) > hard_prob)
  from mjlab.envs.mdp.dr._core import _get_entity_indices

  asset = env.scene[asset_cfg.name]
  gids = _get_entity_indices(asset.indexing, asset_cfg, "geom", False)
  eg, gg = torch.meshgrid(env_ids.to(env.device, dtype=torch.long), gids.long(), indexing="ij")
  m = env.sim.model
  uu = u[:, None].expand_as(eg)
  m.geom_solref[eg, gg, 0] = (0.02 + 0.18 * uu).to(m.geom_solref.dtype)
  m.geom_solimp[eg, gg, 0] = (0.9 - 0.4 * uu).clamp(min=0.3).to(m.geom_solimp.dtype)
  m.geom_solimp[eg, gg, 2] = (0.001 + 0.039 * uu).to(m.geom_solimp.dtype)


def still_joint_vel(env, command_name: str = "motion", gate_std: float = 1.0) -> torch.Tensor:
  """Joint velocity error to the reference, gated to near-static reference frames (standing still,
  the end of a get-up): sum_j (qd - qd_ref)^2 * exp(-|qd_ref|^2 / gate_std^2). Dancing frames ~0 weight.
  Targets the ankle / waist chatter seen in sim2sim while the tracker holds a standing pose."""
  cmd = env.command_manager.get_term(command_name)
  ref = cmd.joint_vel
  gate = torch.exp(-torch.sum(torch.square(ref), dim=-1) / gate_std**2)
  return gate * torch.sum(torch.square(cmd.robot.data.joint_vel - ref), dim=-1)


def robust_env_cfg(play: bool = False):
  cfg = gencoord_env_cfg(play=play)
  cfg.scene.entities["robot"] = dc_motor_robot_cfg()
  cfg.commands["motion"].joint_position_range = (-0.2, 0.2)
  if os.environ.get("G1R_ACTION_RATE"):  # stronger smoothness penalty (default -0.1): less standing jitter
    cfg.rewards["action_rate_l2"].weight = float(os.environ["G1R_ACTION_RATE"])
  if os.environ.get("G1R_DELAY"):  # command latency 0..N physics steps (5 ms each; sim2sim uses 20 ms)
    from dataclasses import replace as _rep
    n = int(os.environ["G1R_DELAY"])
    rob = cfg.scene.entities["robot"]
    acts = tuple(_rep(a, delay_min_lag=0, delay_max_lag=n, delay_update_period=20, delay_hold_prob=0.5)
                 for a in rob.articulation.actuators)
    cfg.scene.entities["robot"] = _rep(rob, articulation=_rep(rob.articulation, actuators=acts))
  if os.environ.get("G1R_STILL"):  # penalize joint chatter while the reference stands still
    from mjlab.managers.reward_manager import RewardTermCfg
    cfg.rewards["still_joint_vel"] = RewardTermCfg(func=still_joint_vel, weight=float(os.environ["G1R_STILL"]),
                                                   params={"command_name": "motion"})
  if TERRAIN:
    from mjlab.terrains.terrain_entity import TerrainEntityCfg
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=floor_terrain_cfg())
    # a lying robot on a heightfield touches many cells: bigger contact / constraint buffers
    cfg.sim.nconmax = 160
    cfg.sim.njmax = 1200
  if not play:
    cfg.events["contact_softness"] = EventTermCfg(mode="startup", func=contact_softness)
    cfg.events["body_friction"] = EventTermCfg(
      mode="startup", func=dr.geom_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=(".*_collision",)), "operation": "abs",
              "ranges": (0.4, 1.2), "shared_random": True})
    cfg.events["pd_gains"] = EventTermCfg(
      mode="startup", func=dr.pd_gains,
      params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.85, 1.15), "kd_range": (0.85, 1.15)})
    cfg.events["effort"] = EventTermCfg(
      mode="startup", func=dr.effort_limits,
      params={"asset_cfg": SceneEntityCfg("robot"), "effort_limit_range": (0.8, 1.05)})
  return cfg


from mjlab.tasks.registry import list_tasks  # noqa: E402

if "SuperMimic-GenCoord-G1-Robust" not in list_tasks():
  from mjlab.rl import RslRlOnPolicyRunnerCfg

  register_mjlab_task(task_id="SuperMimic-GenCoord-G1-Robust", env_cfg=robust_env_cfg(),
                      play_env_cfg=robust_env_cfg(play=True), rl_cfg=RslRlOnPolicyRunnerCfg(), runner_cls=None)
