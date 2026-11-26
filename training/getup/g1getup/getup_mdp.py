"""MDP terms for the G1 fall-recovery (get-up) task."""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul

_ROBOT = SceneEntityCfg("robot")
GRAVITY = (0.0, 0.0, -1.0)


# ---------------------------------------------------------------------------
# Standing reference (heights of the default standing pose, computed once by FK)
# ---------------------------------------------------------------------------


@functools.cache
def standing_heights() -> dict[str, float]:
  import mujoco
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME, get_spec
  import re

  m = get_spec().compile()
  d = mujoco.MjData(m)
  d.qpos[:3] = KNEES_BENT_KEYFRAME.pos
  d.qpos[3:7] = (1, 0, 0, 0)
  for j in range(1, m.njnt):
    name = m.joint(j).name
    for pat, v in KNEES_BENT_KEYFRAME.joint_pos.items():
      if re.fullmatch(pat, name):
        d.qpos[m.jnt_qposadr[j]] = v
  mujoco.mj_kinematics(m, d)
  return {"pelvis": float(d.xpos[m.body("pelvis").id, 2]),
          "torso": float(d.xpos[m.body("torso_link").id, 2])}


def _body_id(env: ManagerBasedRlEnv, name: str) -> int:
  cache = env.__dict__.setdefault("_getup_body_ids", {})
  if name not in cache:
    cache[name] = env.scene["robot"].find_bodies(name)[0][0]
  return cache[name]


def _heights(env):
  a = env.scene["robot"]
  pz = a.data.body_link_pos_w[:, _body_id(env, "pelvis"), 2]
  tz = a.data.body_link_pos_w[:, _body_id(env, "torso_link"), 2]
  return pz, tz


def _torso_up(env) -> torch.Tensor:
  """cos of the torso tilt: 1 upright, -1 upside down."""
  a = env.scene["robot"]
  q = a.data.body_link_quat_w[:, _body_id(env, "torso_link")]
  g = torch.tensor(GRAVITY, device=env.device).expand(q.shape[0], 3)
  return -quat_apply_inverse(q, g)[:, 2]


def _stand_gate(env) -> torch.Tensor:
  h = standing_heights()
  pz, _ = _heights(env)
  return ((pz > 0.88 * h["pelvis"]) & (_torso_up(env) > 0.9)).float()


# ---------------------------------------------------------------------------
# Reset: mixture of floor poses (mocap), random drops, trips and standing
# ---------------------------------------------------------------------------


def _random_quat(n, device):
  u = torch.rand(n, 3, device=device)
  q = torch.stack([
    torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
    torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
    torch.sqrt(u[:, 0]) * torch.sin(2 * math.pi * u[:, 2]),
    torch.sqrt(u[:, 0]) * torch.cos(2 * math.pi * u[:, 2]),
  ], dim=1)
  return q[:, [3, 0, 1, 2]]  # wxyz


def _yaw_quat(yaw):
  z = torch.zeros_like(yaw)
  return quat_from_euler_xyz(z, z, yaw)


def reset_getup(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pool_path: str,
  probs: dict[str, float],
  drop_height: float = 1.0,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  a = env.scene[asset_cfg.name]
  dev = env.device
  pool = env.__dict__.get("_getup_pool")
  if pool is None:
    z = np.load(pool_path)
    assert list(z["joint_names"]) == list(a.joint_names), "pool joint order differs from robot"
    pool = {k: torch.tensor(z[k], device=dev) for k in
            ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")}
    env.__dict__["_getup_pool"] = pool
  n = len(env_ids)
  names = list(probs.keys())
  p = torch.tensor([probs[k] for k in names], device=dev)
  mode = torch.multinomial(p / p.sum(), n, replacement=True)

  dflt_root = a.data.default_root_state[env_ids].clone()
  dflt_q = a.data.default_joint_pos[env_ids].clone()
  lim = a.data.soft_joint_pos_limits[env_ids]
  root = dflt_root.clone()
  jp = dflt_q.clone()
  jv = torch.zeros_like(jp)
  yaw = (torch.rand(n, device=dev) * 2 - 1) * math.pi
  qy = _yaw_quat(yaw)

  # floor poses from mocap
  m = mode == names.index("pool")
  if m.any():
    k = int(m.sum())
    idx = torch.randint(0, pool["joint_pos"].shape[0], (k,), device=dev)
    root[m, 0:3] = pool["root_pos_w"][idx] + torch.tensor([0, 0, 0.01], device=dev)
    root[m, 3:7] = quat_mul(qy[m], pool["root_quat_w"][idx])
    root[m, 7:10] = quat_apply(qy[m], pool["root_lin_vel_w"][idx])
    root[m, 10:13] = quat_apply(qy[m], pool["root_ang_vel_w"][idx])
    jp[m] = pool["joint_pos"][idx]
    jv[m] = pool["joint_vel"][idx]

  # random drop: any orientation, random joints, from drop_height
  m = mode == names.index("drop")
  if m.any():
    k = int(m.sum())
    root[m, 0:2] = 0.0
    root[m, 2] = drop_height
    root[m, 3:7] = _random_quat(k, dev)
    root[m, 7:13] = (torch.rand(k, 6, device=dev) * 2 - 1) * 0.5
    r = torch.rand_like(jp[m])
    jp[m] = lim[m, :, 0] + r * (lim[m, :, 1] - lim[m, :, 0])
    jp[m] = dflt_q[m] + 0.6 * (jp[m] - dflt_q[m])

  # trip: standing with a strong push -> falls during the episode (mid-fall recovery)
  m = mode == names.index("trip")
  if m.any():
    k = int(m.sum())
    root[m, 3:7] = qy[m]
    ang = torch.rand(k, device=dev) * 2 * math.pi
    spd = 1.0 + 2.0 * torch.rand(k, device=dev)
    root[m, 7] = spd * torch.cos(ang)
    root[m, 8] = spd * torch.sin(ang)
    root[m, 10:13] = (torch.rand(k, 3, device=dev) * 2 - 1) * 2.0
    jp[m] = dflt_q[m] + 0.15 * torch.randn_like(jp[m])

  # standing (power on while standing / hand-over from the tracker)
  m = mode == names.index("stand")
  if m.any():
    k = int(m.sum())
    root[m, 3:7] = qy[m]
    root[m, 7:9] = (torch.rand(k, 2, device=dev) * 2 - 1) * 0.3
    jp[m] = dflt_q[m] + 0.05 * torch.randn_like(jp[m])

  jp = torch.minimum(torch.maximum(jp, lim[..., 0]), lim[..., 1])
  root[:, 0:3] += env.scene.env_origins[env_ids]
  a.write_root_state_to_sim(root, env_ids=env_ids)
  a.write_joint_state_to_sim(jp, jv, env_ids=env_ids)
  st = env.__dict__.setdefault("_getup_reset_mode", torch.zeros(env.num_envs, dtype=torch.long, device=dev))
  st[env_ids] = mode


# ---------------------------------------------------------------------------
# Assist force curriculum (vertical lift on the torso, decays to zero)
# ---------------------------------------------------------------------------


def assist_level(env: ManagerBasedRlEnv, decay_steps: int) -> float:
  return max(0.0, 1.0 - env.common_step_counter / max(1, decay_steps))


def assist_force(env: ManagerBasedRlEnv, env_ids, max_force: float, decay_steps: int) -> None:
  a = env.scene["robot"]
  lvl = assist_level(env, decay_steps)
  env.__dict__["_getup_assist"] = lvl
  bid = _body_id(env, "torso_link")
  n = env.num_envs
  f = torch.zeros(n, 1, 3, device=env.device)
  if lvl > 0:
    h = standing_heights()
    pz, _ = _heights(env)
    need = (pz < 0.9 * h["pelvis"]).float()
    f[:, 0, 2] = lvl * max_force * need
  a.write_external_wrench_to_sim(f, torch.zeros_like(f), body_ids=[bid])


# ---------------------------------------------------------------------------
# Observations (critic-only privileged terms)
# ---------------------------------------------------------------------------


def body_heights(env: ManagerBasedRlEnv) -> torch.Tensor:
  pz, tz = _heights(env)
  return torch.stack([pz, tz], dim=1)


def assist_obs(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.full((env.num_envs, 1), env.__dict__.get("_getup_assist", 0.0), device=env.device)


def root_lin_vel_b(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_b


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


def pelvis_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  pz, _ = _heights(env)
  return (pz / standing_heights()["pelvis"]).clamp(0, 1)


def torso_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  _, tz = _heights(env)
  return (tz / standing_heights()["torso"]).clamp(0, 1)


def torso_upright(env: ManagerBasedRlEnv) -> torch.Tensor:
  return (_torso_up(env) + 1) * 0.5


def stand_pose(env: ManagerBasedRlEnv, std: float = 0.5) -> torch.Tensor:
  a = env.scene["robot"]
  err = torch.sum(torch.square(a.data.joint_pos - a.data.default_joint_pos), dim=1)
  return _stand_gate(env) * torch.exp(-err / std**2)


def stand_still(env: ManagerBasedRlEnv, std: float = 0.5) -> torch.Tensor:
  a = env.scene["robot"]
  v = torch.sum(torch.square(a.data.root_link_lin_vel_w[:, :2]), dim=1)
  w = torch.sum(torch.square(a.data.root_link_ang_vel_w), dim=1)
  return _stand_gate(env) * torch.exp(-(v + 0.2 * w) / std**2)


def stand_feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  found = env.scene[sensor_name].data.found  # [B, 2]
  return _stand_gate(env) * (found > 0).all(dim=1).float()


def stand_body_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  found = env.scene[sensor_name].data.found
  return _stand_gate(env) * (found > 0).any(dim=1).float()


def joint_vel_excess(env: ManagerBasedRlEnv, limit: float = 15.0) -> torch.Tensor:
  qd = env.scene["robot"].data.joint_vel
  return torch.sum((qd.abs() - limit).clamp(min=0), dim=1)


@functools.cache
def _motor_limits() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
  """(joint names, rated torque, no-load speed) per joint from mjlab's G1 actuator groups."""
  import re

  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as G

  groups = [(G.G1_ACTUATOR_5020, G.ACTUATOR_5020.velocity_limit), (G.G1_ACTUATOR_7520_14, G.ACTUATOR_7520_14.velocity_limit),
            (G.G1_ACTUATOR_7520_22, G.ACTUATOR_7520_22.velocity_limit), (G.G1_ACTUATOR_4010, G.ACTUATOR_4010.velocity_limit),
            (G.G1_ACTUATOR_WAIST, G.ACTUATOR_5020.velocity_limit), (G.G1_ACTUATOR_ANKLE, G.ACTUATOR_5020.velocity_limit)]
  m = G.get_spec().compile()
  names = tuple(m.joint(i).name for i in range(1, m.njnt))
  eff, vel = [], []
  for n in names:
    a, v = next((a, v) for a, v in groups if any(re.fullmatch(e, n) for e in a.target_names_expr))
    eff.append(a.effort_limit)
    vel.append(v)
  return names, tuple(eff), tuple(vel)


def _limits_t(env):
  c = env.__dict__.get("_getup_motor_lim")
  if c is None:
    names, eff, vel = _motor_limits()
    assert list(names) == list(env.scene["robot"].joint_names)
    c = (torch.tensor(eff, device=env.device), torch.tensor(vel, device=env.device))
    env.__dict__["_getup_motor_lim"] = c
  return c


def joint_speed_margin(env: ManagerBasedRlEnv, frac: float = 0.65) -> torch.Tensor:
  """Sum over joints of how far |qd| exceeds frac x that motor's no-load speed (normalised)."""
  _, vel = _limits_t(env)
  qd = env.scene["robot"].data.joint_vel
  return torch.sum(((qd.abs() - frac * vel) / vel).clamp(min=0), dim=1)


def torque_saturation(env: ManagerBasedRlEnv, frac: float = 0.85) -> torch.Tensor:
  """Sum over joints of how far the applied torque exceeds frac x rated torque (normalised)."""
  eff, _ = _limits_t(env)
  tau = env.scene["robot"].data.qfrc_actuator
  if tau.shape[1] == eff.shape[0] + 6:  # drop the free-joint dofs
    tau = tau[:, 6:]
  return torch.sum(((tau.abs() - frac * eff) / eff).clamp(min=0), dim=1)


def impact(env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 600.0) -> torch.Tensor:
  f = env.scene[sensor_name].data.force  # [B, P, 3] (netforce)
  return torch.sum((torch.linalg.norm(f, dim=-1) - threshold).clamp(min=0), dim=1) / threshold


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def stood_up(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _stand_gate(env)


# ---------------------------------------------------------------------------
# v4: compact, human-like get-up
# ---------------------------------------------------------------------------


def _ref_feats(env, ref_path: str, grav_w: float, stride: int):
  c = env.__dict__.get("_getup_ref")
  if c is None:
    z = np.load(ref_path)
    a = env.scene["robot"]
    assert list(z["joint_names"]) == list(a.joint_names)
    q0 = a.data.default_joint_pos[0]
    q = torch.tensor(z["joint_pos"][::stride], device=env.device) - q0
    g = torch.tensor(z["proj_gravity"][::stride], device=env.device) * grav_w
    c = torch.cat([q, g], dim=1)
    env.__dict__["_getup_ref"] = c
  return c


def style_prior(env: ManagerBasedRlEnv, ref_path: str, sigma: float = 1.0, grav_w: float = 2.0,
                stride: int = 2, gate_standing: bool = True) -> torch.Tensor:
  """exp(-d^2/sigma^2), d = distance of (joint pose, body tilt) to the nearest human get-up frame.
  Off once standing (stand_pose takes over there)."""
  ref = _ref_feats(env, ref_path, grav_w, stride)
  a = env.scene["robot"]
  f = torch.cat([a.data.joint_pos - a.data.default_joint_pos, a.data.projected_gravity_b * grav_w], dim=1)
  d2 = torch.full((f.shape[0],), float("inf"), device=env.device)
  for i in range(0, ref.shape[0], 8192):
    d2 = torch.minimum(d2, torch.cdist(f, ref[i:i + 8192]).square().min(dim=1).values)
  env.__dict__["_getup_style_d2"] = d2
  r = torch.exp(-d2 / sigma**2)
  # v4 gated this off once standing -> staying down in a human-like crouch paid as much as
  # standing (policy collapse). v5 keeps it on everywhere (the references end standing).
  return (1 - _stand_gate(env)) * r if gate_standing else r


def _group_ids(env, pats):
  key = "_gid_" + "|".join(pats)
  c = env.__dict__.get(key)
  if c is None:
    import re
    names = env.scene["robot"].joint_names
    c = torch.tensor([i for i, n in enumerate(names) if any(re.search(p, n) for p in pats)], device=env.device)
    env.__dict__[key] = c
  return c


def limb_splay(env: ManagerBasedRlEnv, limits: dict[str, float]) -> torch.Tensor:
  """Sum over joint groups of |q - q_default| beyond a per-group limit (legs/arms swinging out)."""
  a = env.scene["robot"]
  dq = (a.data.joint_pos - a.data.default_joint_pos).abs()
  out = torch.zeros(env.num_envs, device=env.device)
  for pat, lim in limits.items():
    ids = _group_ids(env, (pat,))
    out += (dq[:, ids] - lim).clamp(min=0).sum(dim=1)
  return out


def hand_reach(env: ManagerBasedRlEnv, limit: float = 0.5) -> torch.Tensor:
  """Horizontal hand distance from the pelvis beyond `limit` (arms flung out)."""
  a = env.scene["robot"]
  p = a.data.body_link_pos_w
  pel = p[:, _body_id(env, "pelvis"), :2]
  out = torch.zeros(env.num_envs, device=env.device)
  for h in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
    out += (torch.linalg.norm(p[:, _body_id(env, h), :2] - pel, dim=1) - limit).clamp(min=0)
  return out


def footprint(env: ManagerBasedRlEnv, margin: float = 0.1) -> torch.Tensor:
  """Once the robot has come to rest on the floor (or at reset when standing), remember the
  circle that holds its body; penalise any body point leaving that circle (+margin)."""
  a = env.scene["robot"]
  dev = env.device
  st = env.__dict__.get("_getup_fp")
  if st is None:
    st = {"set": torch.zeros(env.num_envs, dtype=torch.bool, device=dev),
          "c": torch.zeros(env.num_envs, 2, device=dev), "r": torch.zeros(env.num_envs, device=dev)}
    env.__dict__["_getup_fp"] = st
  # episode restarted -> forget the anchor
  st["set"] &= env.episode_length_buf > 1
  p = a.data.body_link_pos_w[..., :2]
  pel = p[:, _body_id(env, "pelvis")]
  pz, _ = _heights(env)
  speed = torch.linalg.norm(a.data.root_link_lin_vel_w, dim=1)
  standing = _stand_gate(env).bool()
  late = int(env.__dict__.get("_getup_fp_late", 100))
  ready = ~st["set"] & (((pz < 0.6) & (speed < 0.4)) | (standing & (env.episode_length_buf <= 2))
                        | (env.episode_length_buf >= late))  # anchor no later than `late` steps
  if ready.any():
    st["c"][ready] = pel[ready]
    st["r"][ready] = torch.linalg.norm(p[ready] - pel[ready, None], dim=-1).max(dim=1).values + margin
    st["set"] |= ready
  dist = torch.linalg.norm(p - st["c"][:, None], dim=-1).max(dim=1).values
  ex = st["set"].float() * (dist - st["r"]).clamp(min=0)
  st["excess"] = ex
  return ex


def footprint_violation(env: ManagerBasedRlEnv, limit: float = 0.35) -> torch.Tensor:
  """Termination ("virtual wall"): a body point left the anchored footprint by more than `limit`."""
  st = env.__dict__.get("_getup_fp")
  if st is None or "excess" not in st:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return st["excess"] > limit


def style_dist(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Metric: squared distance to the nearest human get-up frame (before standing)."""
  d2 = env.__dict__.get("_getup_style_d2")
  if d2 is None:
    return torch.zeros(env.num_envs, device=env.device)
  return d2 * (1 - _stand_gate(env))


# ---------------------------------------------------------------------------
# Settle: move from any fallen pose to the canonical lying pose that starts a human get-up clip
# ---------------------------------------------------------------------------


def _settle_target(env, clip_path: str):
  t = env.__dict__.get("_settle_tgt")
  if t is None:
    z = np.load(clip_path)
    a = env.scene["robot"]
    assert list(z["joint_names"]) == list(a.joint_names)
    q = torch.tensor(z["joint_pos"][0], device=env.device, dtype=torch.float32)
    qw = torch.tensor(z["root_quat_w"][0], device=env.device, dtype=torch.float32)[None]
    g = quat_apply_inverse(qw, torch.tensor([GRAVITY], device=env.device))[0]
    t = (q, g)
    env.__dict__["_settle_tgt"] = t
  return t


def _settle_err(env, clip_path: str):
  q_t, g_t = _settle_target(env, clip_path)
  a = env.scene["robot"]
  g = a.data.projected_gravity_b
  eg = torch.sum((g - g_t) ** 2, dim=1)
  eq = torch.mean((a.data.joint_pos - q_t) ** 2, dim=1)
  return eg, eq


def settle_orient(env: ManagerBasedRlEnv, clip_path: str, std: float = 0.5) -> torch.Tensor:
  eg, _ = _settle_err(env, clip_path)
  return torch.exp(-eg / std**2)


def settle_pose(env: ManagerBasedRlEnv, clip_path: str, std: float = 0.3) -> torch.Tensor:
  """Joint-pose match, weighted by how well the body orientation already matches."""
  eg, eq = _settle_err(env, clip_path)
  return torch.exp(-eq / std**2) * (0.3 + 0.7 * torch.exp(-eg / 0.25))


def settle_still(env: ManagerBasedRlEnv, clip_path: str) -> torch.Tensor:
  """Being at rest once in the target pose."""
  eg, eq = _settle_err(env, clip_path)
  a = env.scene["robot"]
  near = torch.exp(-eg / 0.05) * torch.exp(-eq / 0.02)
  v = torch.sum(a.data.joint_vel**2, dim=1) + 10 * torch.sum(a.data.root_link_ang_vel_w**2, dim=1)
  return near * torch.exp(-v / 4.0)


def settled(env: ManagerBasedRlEnv, clip_path: str) -> torch.Tensor:
  """Metric: same check the deploy supervisor uses to hand over to the get-up tracker."""
  eg, eq = _settle_err(env, clip_path)
  a = env.scene["robot"]
  slow = a.data.joint_vel.abs().max(dim=1).values < 1.0
  return ((eg < 0.05) & (eq.sqrt() < 0.15) & slow).float()



# ---------------------------------------------------------------------------
# Soft floors (same mapping as tools/robust_task.py for the tracker)
# ---------------------------------------------------------------------------
from mjlab.managers.event_manager import requires_model_fields  # noqa: E402


@requires_model_fields("geom_solref", "geom_solimp")
def contact_softness(env, env_ids, hard_prob: float = 0.5, u_max: float = 1.3,
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", geom_names=(".*_collision",))):
  """Per env floor softness u (0 = hard floor with prob hard_prob, else U(0, u_max)):
  solref time constant 0.02 + 0.18u, solimp dmin 0.9 - 0.4u, width 0.001 + 0.039u."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)
  u = u_max * torch.rand(n, device=env.device) * (torch.rand(n, device=env.device) > hard_prob)
  from mjlab.envs.mdp.dr._core import _get_entity_indices

  asset = env.scene[asset_cfg.name]
  gids = _get_entity_indices(asset.indexing, asset_cfg, "geom", False)
  eg, gg = torch.meshgrid(env_ids.to(env.device, dtype=torch.long), gids.long(), indexing="ij")
  m = env.sim.model
  uu = u[:, None].expand_as(eg)
  m.geom_solref[eg, gg, 0] = (0.02 + 0.18 * uu).to(m.geom_solref.dtype)
  m.geom_solimp[eg, gg, 0] = (0.9 - 0.4 * uu).clamp(min=0.3).to(m.geom_solimp.dtype)
  m.geom_solimp[eg, gg, 2] = (0.001 + 0.039 * uu).to(m.geom_solimp.dtype)
