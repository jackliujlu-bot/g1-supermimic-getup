"""Sim2sim acceptance tests in plain (CPU) MuJoCo using the deploy stack unchanged.

Scenarios (each run with randomised physics: PD gain x[0.9,1.1], friction [0.5,1.1],
1-tick action latency, IMU/encoder noise):
  boot_supine / boot_prone / boot_side / boot_sitting : power on lying -> must stand
  boot_standing                                       : power on standing -> must keep standing
  dance                                               : stand -> dance clip -> stand
  dance_push                                          : dance, hard shove mid-dance -> fall ->
                                                        auto get-up -> stand
Pass criterion for get-up: stable stand (upright > 0.95, pelvis > 0.65 m, pose error
< 0.35 rad, held 2 s) reached within the time limit, no joint over torque limit.

Usage: python sim2sim.py --getup getup.onnx [--dance dance.onnx] [--trials 10] [--video out.mp4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_stack import (  # noqa: E402
  DEFAULT_Q, KD, KP, NJ, DancePolicy, GetupPolicy, RobotState, Supervisor, projected_gravity,
)

SIM_DT = 0.005
DECIM = 4
from g1_stack import P as _P  # noqa: E402

VLIM = np.array(_P["vel_limit"])
STALL_RATIO = 1.5
DC_MOTOR = os.environ.get("SIM2SIM_MOTOR", "dc") == "dc"  # torque-speed curve (conservative)


TERRAIN = None  # max bump height [m] of the uneven-paving floor, None = flat plane
TERRAIN_SEED = 0
SEED_BASE = int(os.environ.get("S2S_SEED_BASE", "1000"))  # trial seeds; another base = held-out trials
VARY_TERRAIN = os.environ.get("S2S_VARY_TERRAIN") == "1"  # new uneven-floor layout per trial


def build_model():
  from mjlab.asset_zoo.robots import get_g1_robot_cfg
  from mjlab.entity import Entity

  ent = Entity(get_g1_robot_cfg())
  spec = ent.spec
  if TERRAIN:  # uneven paving: 6 x 6 m heightfield, 5 cm cells, random bumps on a 15 cm grid
    n, cell = 121, 0.05
    rng = np.random.default_rng(TERRAIN_SEED)
    coarse = rng.uniform(0, TERRAIN, (n // 3 + 2, n // 3 + 2))
    from scipy.ndimage import zoom
    h = zoom(coarse, 3, order=1)[:n, :n]
    spec.add_hfield(name="paving", nrow=n, ncol=n, size=[n * cell / 2, n * cell / 2, max(TERRAIN, 1e-3), 0.05],
                    userdata=(h / max(TERRAIN, 1e-3)).ravel().tolist())
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="paving",
                            friction=[0.8, 0.005, 0.0001])
    spec.worldbody.add_geom(name="floor_far", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1],
                            pos=[0, 0, -0.001], friction=[0.8, 0.005, 0.0001])
  else:
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1],
                            friction=[0.8, 0.005, 0.0001])
  # walls for the narrow-space tests, parked far away until placed. Static world geoms cannot be
  # re-oriented at runtime, so wall0/1 run along y and wall2/3 along x; the robot is turned instead.
  for i in range(4):
    size = [0.05, 3.0, 1.0] if i < 2 else [3.0, 0.05, 1.0]
    spec.worldbody.add_geom(name=f"wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                            pos=[100 + 10 * i, 0, 1.0], rgba=[0.6, 0.3, 0.3, 0.5])
  spec.option.timestep = SIM_DT
  spec.option.iterations = 50
  spec.option.ls_iterations = 50
  m = spec.compile()
  jn = [m.joint(i).name for i in range(1, m.njnt)]
  an = [m.actuator(i).name for i in range(m.nu)]
  return m, jn, an


class Sim:
  def __init__(self, rng, randomize=True):
    self.m, jn, an = build_model()
    from g1_stack import JOINT_NAMES

    assert jn == JOINT_NAMES, "joint order mismatch"
    # map actuators to joints
    self.act_j = np.array([JOINT_NAMES.index(self.m.joint(self.m.actuator_trnid[i, 0]).name) for i in range(self.m.nu)])
    self.d = mujoco.MjData(self.m)
    self.rng = rng
    self.kp_scale = rng.uniform(0.9, 1.1, NJ) if randomize else np.ones(NJ)
    self.kd_scale = rng.uniform(0.9, 1.1, NJ) if randomize else np.ones(NJ)
    fr = rng.uniform(0.5, 1.1) if randomize else 0.8
    self.m.geom_friction[:, 0] = fr
    if EXTRA_MASS:  # payload on the upper body (head sensors etc.): kg added to torso_link, placed high
      b = self.m.body("torso_link").id
      m0, c0 = self.m.body_mass[b], self.m.body_ipos[b].copy()
      c_pay = np.array([0.05, 0.0, 0.40])  # roughly the head, in torso_link frame
      self.m.body_mass[b] = m0 + EXTRA_MASS
      self.m.body_ipos[b] = (m0 * c0 + EXTRA_MASS * c_pay) / (m0 + EXTRA_MASS)
      self.m.body_inertia[b] *= (m0 + EXTRA_MASS) / m0
    if SOFT:  # soft mat, softness u in (0, 1] (same mapping as the training randomization)
      # on the uneven floor only the robot's geoms get soft (as in training): a soft heightfield
      # lets bodies tunnel through its cells in CPU MuJoCo
      skip = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("floor", "floor_far")} if TERRAIN else set()
      for g in range(self.m.ngeom):
        if (self.m.geom_contype[g] or self.m.geom_conaffinity[g]) and g not in skip:
          self.m.geom_solref[g] = [0.02 + 0.18 * SOFT, 1.0]
          self.m.geom_solimp[g, :3] = [0.9 - 0.4 * SOFT, 0.95, 0.001 + 0.039 * SOFT]
    if os.environ.get("S2S_MATCH_TRAIN") == "1":  # mjlab training solver settings
      self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
      self.m.opt.iterations, self.m.opt.ls_iterations = 10, 20
    self.noise = randomize
    self.pending = None
    self.effort = np.array([self.m.actuator_forcerange[np.where(self.act_j == j)[0][0], 1] for j in range(NJ)])
    self.max_torque_ratio = 0.0
    # explicit PD in python: zero the builtin actuator gains, use motor-style control
    self.m.actuator_gainprm[:, :] = 0
    self.m.actuator_biasprm[:, :] = 0
    self.m.actuator_gainprm[:, 0] = 1.0
    self.m.actuator_ctrlrange[:, :] = 0
    self.m.actuator_ctrllimited[:] = 0
    self.pid = self.m.body("pelvis").id
    self.floor = self.m.geom("floor").id
    self.mass = float(self.m.body_subtreemass[1])
    self.n_sub = self.n_sat = 0
    self.max_qd_ratio = 0.0
    self.peak_contact = 0.0
    self.walls = [self.m.geom(f"wall{i}").id for i in range(4)]
    self.wall_hits = 0
    self.wall_force = 0.0
    self.fp_center = None
    self.fp_r0 = 0.0
    self.fp_max = 0.0
    self._sweep = []
    self._stood = False
    self.limb_reach = 0.0
    self._limb_ids = [self.m.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link",
                                                   "left_ankle_roll_link", "right_ankle_roll_link")]
    self.imu_gyro = self.m.sensor("imu_ang_vel").adr[0] if self._has_sensor("imu_ang_vel") else None

  def _has_sensor(self, name):
    try:
      self.m.sensor(name)
      return True
    except KeyError:
      return False

  def set_state(self, qpos, qvel=None):
    self.d.qpos[:] = qpos
    self.d.qvel[:] = 0 if qvel is None else qvel
    mujoco.mj_forward(self.m, self.d)

  def settle_passive(self, seconds=1.5, kp=5.0, kd=1.0, target=None):
    """Let the robot fall / settle with weak damping (powered-off-like)."""
    tgt = self.d.qpos[7:].copy() if target is None else target
    for _ in range(int(seconds / SIM_DT)):
      tau = kp * (tgt - self.d.qpos[7:]) - kd * self.d.qvel[6:]
      self.d.ctrl[:] = tau[self.act_j]
      mujoco.mj_step(self.m, self.d)

  def state(self) -> RobotState:
    q = self.d.qpos[7:].copy()
    qd = self.d.qvel[6:].copy()
    quat = self.d.qpos[3:7].copy()
    gyro = self.d.sensordata[self.imu_gyro:self.imu_gyro + 3].copy() if self.imu_gyro is not None else self.d.qvel[3:6].copy()
    if self.noise:
      q += self.rng.normal(0, 0.005, NJ)
      qd += self.rng.normal(0, QD_NOISE, NJ)
      gyro += self.rng.normal(0, 0.05, 3)
      quat = quat + self.rng.normal(0, 0.005, 4)
      quat /= np.linalg.norm(quat)
    return RobotState(quat=quat, gyro=gyro, q=q, qd=qd)

  def clip_torque(self, tau, qd):
    if not DC_MOTOR:
      return np.clip(tau, -self.effort, self.effort)
    S = STALL_RATIO * self.effort
    top = np.minimum(S * (1.0 - qd / VLIM), self.effort)
    bot = np.maximum(S * (-1.0 - qd / VLIM), -self.effort)
    return np.clip(tau, bot, top)

  def apply(self, q_target, steps=DECIM, push=None):
    # 1-tick latency: the target computed now is applied next tick
    tgt = self.pending if (self.pending is not None and self.noise and LATENCY) else q_target
    self.pending = q_target
    for _ in range(steps):
      q = self.d.qpos[7:]
      qd = self.d.qvel[6:]
      tau = KP * self.kp_scale * (tgt - q) - KD * self.kd_scale * qd
      ratio = np.abs(tau) / self.effort
      self.max_torque_ratio = max(self.max_torque_ratio, float(ratio.max()))
      tau_c = self.clip_torque(tau, qd)
      self.n_sub += 1
      self.n_sat += int(np.any(np.abs(tau_c - tau) > 1e-6))
      self.max_qd_ratio = max(self.max_qd_ratio, float((np.abs(qd) / VLIM).max()))
      tau = tau_c
      self.d.ctrl[:] = tau[self.act_j]
      if push is not None:
        self.d.xfrc_applied[self.m.body("torso_link").id, :3] = push
      mujoco.mj_step(self.m, self.d)
      self.peak_contact = max(self.peak_contact, self.ground_force())
    self.d.xfrc_applied[:] = 0
    if self.fp_center is not None:
      self.fp_max = max(self.fp_max, self.extent())
      if not self._stood:  # only the get-up itself, not standing afterwards
        pts = self.body_points_xy()
        self._sweep.append(pts)
        pel = self.d.xpos[self.pid, :2]
        lim = self.d.xpos[self._limb_ids, :2] - pel
        self.limb_reach = max(self.limb_reach, float(np.linalg.norm(lim, axis=1).max()))

  def sweep_area(self):
    """Area [m^2] of the convex hull of every body point over the whole get-up."""
    from scipy.spatial import ConvexHull

    if not self._sweep:
      return 0.0
    pts = np.concatenate(self._sweep)
    return float(ConvexHull(pts).volume)

  # ---- footprint / narrow-space test
  def body_points_xy(self):
    """xy of every robot body origin plus every collision-geom centre."""
    pts = [self.d.xpos[1:, :2]]
    g = np.array([i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] > 0])
    pts.append(self.d.geom_xpos[g, :2])
    return np.concatenate(pts)

  def extent(self):
    return float(np.linalg.norm(self.body_points_xy() - self.fp_center, axis=1).max())

  def start_footprint(self):
    self.fp_center = self.d.xpos[self.pid, :2].copy()
    self.fp_r0 = self.extent()
    self.fp_max = self.fp_r0

  def place_walls(self, margin):
    """Box of walls around the lying robot: each wall `margin` beyond its surface extent."""
    c = self.fp_center
    xl, xh = self.surface_extent(c, np.array([1.0, 0.0]))
    yl, yh = self.surface_extent(c, np.array([0.0, 1.0]))
    poses = [(c[0] + xh + margin + 0.05, c[1]), (c[0] + xl - margin - 0.05, c[1]),
             (c[0], c[1] + yh + margin + 0.05), (c[0], c[1] + yl - margin - 0.05)]
    for gid, (x, y) in zip(self.walls, poses):
      self.m.geom_pos[gid] = [x, y, 1.0]
    mujoco.mj_forward(self.m, self.d)

  def turn_body_to_x(self):
    """Rotate the whole robot about the vertical through its pelvis so pelvis->torso points +x."""
    c = self.d.xpos[self.pid, :2].copy()
    ax = self.d.xpos[self.m.body("torso_link").id, :2] - c
    phi = -np.arctan2(ax[1], ax[0])
    cz, sz = np.cos(phi), np.sin(phi)
    R = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    q = self.d.qpos
    q[:3] = np.r_[c, 0] + R @ (q[:3] - np.r_[c, 0])
    qz = np.array([np.cos(phi / 2), 0, 0, np.sin(phi / 2)])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = q[3:7].copy()
    q[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
              w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
    self.d.qvel[:3] = R @ self.d.qvel[:3]  # world-frame linear velocity; angular part is body-frame
    mujoco.mj_forward(self.m, self.d)

  def surface_extent(self, c, n):
    """(min, max) of the collision-geom surfaces along horizontal unit direction n from point c."""
    lo, hi = np.inf, -np.inf
    n3 = np.array([n[0], n[1], 0.0])
    for g in range(self.m.ngeom):
      if self.m.geom_bodyid[g] == 0 or not (self.m.geom_contype[g] or self.m.geom_conaffinity[g]):
        continue
      t = self.m.geom_type[g]
      ctr = (self.d.geom_xpos[g, :2] - c) @ n
      r = self.m.geom_size[g, 0]
      if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = self.d.geom_xmat[g].reshape(3, 3)[:, 2]
        ext = abs(axis @ n3) * self.m.geom_size[g, 1] + r
      elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        ext = r
      else:
        ext = float(np.abs(self.d.geom_xmat[g].reshape(3, 3).T @ n3) @ self.m.geom_size[g])
      lo, hi = min(lo, ctr - ext), max(hi, ctr + ext)
    return lo, hi

  def place_corridor(self, margin):
    """Narrow aisle: two walls parallel to the lying body's long axis (pelvis -> torso), each
    `margin` beyond the body's widest point on that side (just wide enough to lie in)."""
    self.turn_body_to_x()  # body now lies along +x; the aisle walls (wall2/3) run along x
    self.start_footprint()
    c = self.fp_center
    lo, hi = self.surface_extent(c, np.array([0.0, 1.0]))
    self.corridor_width = float(hi - lo + 2 * margin)
    for gid, off in zip(self.walls[2:], (hi + margin + 0.05, lo - margin - 0.05)):
      self.m.geom_pos[gid] = [c[0], c[1] + off, 1.0]
    mujoco.mj_forward(self.m, self.d)

  def wall_contact(self):
    f6 = np.zeros(6)
    hit, tot = False, 0.0
    for i in range(self.d.ncon):
      c = self.d.contact[i]
      if c.geom1 in self.walls or c.geom2 in self.walls:
        hit = True
        mujoco.mj_contactForce(self.m, self.d, i, f6)
        tot += abs(f6[0])
    return hit, tot

  def fp_metrics(self):
    return dict(fp_r0=round(self.fp_r0, 3), fp_max=round(self.fp_max, 3),
                aisle_width=round(getattr(self, "corridor_width", 0.0), 3),
                sweep_area_m2=round(self.sweep_area(), 3), limb_reach_m=round(self.limb_reach, 3),
                fp_excess=round(self.fp_max - self.fp_r0, 3),
                wall_hit_ticks=self.wall_hits, wall_peak_force_bw=round(self.wall_force / (self.mass * 9.81), 2))

  def ground_force(self):
    f6 = np.zeros(6)
    tot = 0.0
    for i in range(self.d.ncon):
      c = self.d.contact[i]
      if c.geom1 == self.floor or c.geom2 == self.floor:
        mujoco.mj_contactForce(self.m, self.d, i, f6)
        tot += abs(f6[0])
    return tot

  def hw_metrics(self):
    return dict(max_torque_ratio=round(self.max_torque_ratio, 2),
                sat_frac=round(self.n_sat / max(1, self.n_sub), 3),
                max_qd_ratio=round(self.max_qd_ratio, 2),
                peak_ground_force_bw=round(self.peak_contact / (self.mass * 9.81), 2))

  @property
  def pelvis_z(self):
    return float(self.d.xpos[self.pid, 2])


def rand_quat(rng):
  u = rng.random(3)
  q = np.array([np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]), np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
                np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]), np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])])
  return q[[3, 0, 1, 2]]


def initial_pose(kind, rng):
  qpos = np.zeros(7 + NJ)
  qpos[7:] = DEFAULT_Q
  yaw = rng.uniform(-np.pi, np.pi)
  def eul(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r/2), np.sin(r/2), np.cos(p/2), np.sin(p/2), np.cos(y/2), np.sin(y/2)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])
  if kind == "standing":
    qpos[2] = 0.79
    qpos[3:7] = eul(0, 0, yaw)
    return qpos, False
  if kind == "supine":
    qpos[3:7] = eul(0, -np.pi / 2, yaw)
  elif kind == "prone":
    qpos[3:7] = eul(0, np.pi / 2, yaw)
  elif kind == "side":
    qpos[3:7] = eul(np.pi / 2 * rng.choice([-1, 1]), 0, yaw)
  elif kind == "sitting":
    qpos[3:7] = eul(0, -0.3, yaw)
    qpos[7:] += 0  # handled by settle with bent targets below
  elif kind == "random":
    qpos[3:7] = rand_quat(rng)
  qpos[7:] = DEFAULT_Q + rng.uniform(-0.4, 0.4, NJ) * 0.5
  qpos[2] = 0.9
  return qpos, True


STAND_POSES = [DEFAULT_Q]  # unified mode adds the tracker's standing pose


def stable(sim, s):
  near = min(np.abs(s.q - p).max() for p in STAND_POSES) < 0.35
  return (-projected_gravity(s.quat)[2] > 0.95 and sim.pelvis_z > 0.65 and near and np.abs(s.qd).max() < 1.5)


def _group_idx():
  from g1_stack import JOINT_NAMES

  arms = [i for i, n in enumerate(JOINT_NAMES) if any(k in n for k in ("shoulder", "elbow"))]
  splay = [i for i, n in enumerate(JOINT_NAMES) if any(k in n for k in ("hip_roll", "hip_yaw"))]
  return np.array(arms), np.array(splay)


LATENCY = True  # 1-tick (20 ms) action latency
QD_NOISE = 0.3  # joint velocity sensor noise std (rad/s)
EXTRA_MASS = 0.0  # kg of extra payload on the torso (head sensors), for sim-to-real checks
SOFT = None  # soft mat softness u in (0, 1]; 1 = the lying robot sinks ~3 cm. None = hard floor
HUMAN = None  # (settle.onnx, track.onnx, clip.npz) -> human-style get-up path in the supervisor
HUMAN_TILT = 45.0  # human path only within this angle of the routine's lying orientation (180 = always)


UNIFIED = None  # {"tracker": onnx, "routines": [(settle_onnx, getup_clip_npz)], "dance_clip": npz}


def make_supervisor(getup_path, dances):
  if UNIFIED:
    from unified import build_unified
    sup = build_unified(getup_path if UNIFIED.get("v3") else None, UNIFIED["tracker"], UNIFIED["routines"],
                        {"d": UNIFIED["dance_clip"]} if dances else {}, human_tilt_deg=HUMAN_TILT)
    if not any(np.allclose(sup.stand_q, p) for p in STAND_POSES):
      STAND_POSES.append(sup.stand_q.astype(np.float32))
    return sup
  from g1_stack import HumanGetup
  human = HumanGetup(*HUMAN, max_tilt_deg=HUMAN_TILT) if HUMAN else None
  return Supervisor(GetupPolicy(getup_path), dances, human=human)


def run_boot(kind, getup_path, seed, limit_s=15.0, frames=None, walls=None, corridor=None):
  global TERRAIN_SEED
  if VARY_TERRAIN:
    TERRAIN_SEED = seed
  if corridor and walls is None:
    walls = 0.0
  rng = np.random.default_rng(seed)
  sim = Sim(rng)
  qpos, settle = initial_pose(kind, rng)
  sim.set_state(qpos)
  if settle:
    tgt = None
    if kind == "sitting":
      tgt = DEFAULT_Q.copy()
      for n, v in (("hip_pitch", -1.6), ("knee", 1.8)):
        from g1_stack import JOINT_NAMES
        for i, jn in enumerate(JOINT_NAMES):
          if n in jn:
            tgt[i] = v
      sim.settle_passive(2.0, kp=40, kd=2, target=tgt)
    sim.settle_passive(1.5)
  z0 = sim.pelvis_z
  sim.start_footprint()
  if walls is not None:
    if corridor:
      sim.place_corridor(corridor)
    else:
      sim.place_walls(walls)
    assert not sim.wall_contact()[0], "test setup error: wall placed inside the robot"
  arms, splay = _group_idx()
  arm_span = splay_span = 0.0
  sup = make_supervisor(getup_path, {})
  hits_by_mode, reach_by_mode = {}, {}
  t_stand, held = None, 0
  tail, qtail = [], []
  for k in range(int(limit_s / 0.02)):
    s = sim.state()
    sim.apply(sup.step(s))
    dq = np.abs(sim.d.qpos[7:] - DEFAULT_Q)
    arm_span = max(arm_span, float(dq[arms].max()))
    splay_span = max(splay_span, float(dq[splay].max()))
    if walls is not None:
      hit, f = sim.wall_contact()
      sim.wall_hits += int(hit)
      sim.wall_force = max(sim.wall_force, f)
      if hit:
        hits_by_mode[sup.mode] = hits_by_mode.get(sup.mode, 0) + 1
    if not sim._stood:
      ext = sim.extent()
      reach_by_mode[sup.mode] = max(reach_by_mode.get(sup.mode, 0.0), round(ext - sim.fp_r0, 3))
    if frames is not None and k % 2 == 0:
      frames.append(sim.d.qpos.copy())
    held = held + 1 if stable(sim, s) else 0
    tail = (tail + [(float(-projected_gravity(s.quat)[2]), float(sim.pelvis_z), float(min(np.abs(s.q - p).max() for p in STAND_POSES)), float(np.abs(s.qd).max()), float(np.abs(sim.d.qvel[6:]).max()))])[-50:]
    qtail = (qtail + [sim.d.qpos[7:].copy()])[-50:]
    if held:
      sim._stood = True
    if held >= 100 and t_stand is None:
      t_stand = (k - 99) * 0.02
      if kind != "standing" and frames is None:
        break
  tl_ = np.array(tail)
  upright_end = bool(len(tl_) and (tl_[:, 0] > 0.95).all() and (tl_[:, 1] > 0.65).all())
  return dict(scenario=f"boot_{kind}", seed=seed, ok=bool(t_stand is not None), t_stand=t_stand,
              upright_end=upright_end, end_pose_err=round(float(tl_[:, 2].max()), 3) if len(tl_) else None,
              end_qd=round(float(tl_[:, 3].max()), 2) if len(tl_) else None,
              end_qd_true=round(float(tl_[:, 4].max()), 2) if len(tl_) else None,
              end_swing_deg=round(float(np.degrees(np.ptp(np.array(qtail), 0).max())), 2) if qtail else None,
              end_jitter_joint=int(np.argmax(np.abs(sim.d.qvel[6:]))),
              hits_by_mode=hits_by_mode, excess_by_mode=reach_by_mode,
              path=" ".join(e.split("->")[0] + ">" + e.split("->")[1] if "->" in e else e.split(".")[0]
                            for _, e in sup.events),
              start_pelvis_z=round(z0, 3), final_pelvis_z=round(sim.pelvis_z, 3),
              arm_span=round(arm_span, 2), splay_span=round(splay_span, 2),
              **sim.hw_metrics(), **sim.fp_metrics())


def run_dance(getup_path, dance_path, seed, push=True, frames=None):
  global TERRAIN_SEED
  if VARY_TERRAIN:
    TERRAIN_SEED = seed
  rng = np.random.default_rng(seed)
  sim = Sim(rng)
  qpos, _ = initial_pose("standing", rng)
  sim.set_state(qpos)
  if UNIFIED:
    UNIFIED["dance_clip"] = dance_path
    nf = len(np.load(dance_path)["joint_pos"])
    sup = make_supervisor(getup_path, {"d": None})
  else:
    dance = DancePolicy(dance_path)
    nf = dance.num_frames
    sup = make_supervisor(getup_path, {"d": dance})
  push_at = None
  t_fall = t_recovered = None
  started = False
  held = 0
  max_k = nf + int(30 / 0.02)
  for k in range(max_k):
    s = sim.state()
    if not started and sup.request_dance("d", s):
      started = True
      t0 = k
      push_at = t0 + int(UNIFIED is not None) * 50 + int(rng.uniform(0.25, 0.6) * nf) if push else None
    f = None
    if push_at is not None and push_at <= k < push_at + 10:  # 0.2 s shove
      ang = rng.uniform(0, 2 * np.pi) if k == push_at else ang  # noqa: F821
      f = np.array([np.cos(ang), np.sin(ang), 0.0]) * 900.0
    sim.apply(sup.step(s), push=f)
    if frames is not None and k % 2 == 0:
      frames.append(sim.d.qpos.copy())
    if t_fall is None and any(e[1].startswith("fall_detected") for e in sup.events):
      t_fall = k * 0.02
    if started and (sup.idle if UNIFIED else sup.mode == "GETUP") and k > (push_at or 0):
      ok_now = sup.stable_ticks > 0 if UNIFIED else stable(sim, s)
      held = held + 1 if ok_now else 0
      if held >= 100:
        t_recovered = k * 0.02
        break
  return dict(scenario="dance_push" if push else "dance", seed=seed, ok=t_recovered is not None,
              fell=t_fall is not None, t_fall=t_fall, t_final_stand=t_recovered,
              path=" ".join(e for _, e in sup.events if not e.startswith("dance:")),
              events=sup.events, **sim.hw_metrics())


def render(frames, path, fps=25):
  import imageio
  m, _, _ = build_model()
  d = mujoco.MjData(m)
  r = mujoco.Renderer(m, 480, 640)
  cam = mujoco.MjvCamera()
  cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 120
  with imageio.get_writer(path, fps=fps) as w:
    for q in frames:
      d.qpos[:] = q
      mujoco.mj_forward(m, d)
      cam.lookat[:] = [q[0], q[1], 0.5]
      r.update_scene(d, cam)
      w.append_data(r.render())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--getup", required=True)
  ap.add_argument("--dance", default="")
  ap.add_argument("--trials", type=int, default=10)
  ap.add_argument("--out", default="sim2sim_report.json")
  ap.add_argument("--video", default="")
  ap.add_argument("--scenarios", default="supine,prone,side,sitting,random,standing")
  ap.add_argument("--walls", type=float, default=None,
                  help="narrow-space test: walls at (lying footprint radius + this margin) [m]")
  ap.add_argument("--corridor", type=float, default=None,
                  help="aisle test: two walls along the lying body, this margin [m] beyond its widest point")
  ap.add_argument("--human", nargs="+", default=None, help="triples: SETTLE_ONNX TRACK_ONNX CLIP_NPZ")
  ap.add_argument("--limit", type=float, default=15.0)
  ap.add_argument("--human-tilt", type=float, default=45.0, help="180 = human path for every fallen pose")
  ap.add_argument("--soft", type=float, default=None, help="soft mat softness in (0, 1]; 1 = ~3 cm sink")
  ap.add_argument("--unified", nargs="+", default=None, metavar="TRACKER_ONNX SETTLE_ONNX GETUP_CLIP",
                  help="one tracker for get-up/stand/dance, then pairs SETTLE_ONNX GETUP_CLIP; --dance takes a clip .npz")
  ap.add_argument("--terrain", type=float, default=None,
                  help="uneven paving: random bumps up to this height [m] (e.g. 0.015) instead of a flat floor")
  a = ap.parse_args()
  global HUMAN, HUMAN_TILT, UNIFIED, SOFT, TERRAIN
  HUMAN, HUMAN_TILT, SOFT, TERRAIN = a.human, a.human_tilt, a.soft, a.terrain
  if a.unified:
    from unified import unified_pairs
    UNIFIED = {"tracker": a.unified[0], "routines": unified_pairs(a.unified), "dance_clip": None}
  res = []
  for kind in a.scenarios.split(","):
    for t in range(a.trials):
      r = run_boot(kind, a.getup, seed=SEED_BASE + t, walls=a.walls, corridor=a.corridor, limit_s=a.limit)
      res.append(r)
      print(json.dumps(r), flush=True)
  if a.dance:
    for t in range(a.trials):
      for push in (False, True):
        r = run_dance(a.getup, a.dance, seed=SEED_BASE + 1000 + t, push=push)
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "events"}), flush=True)
  summ = {}
  for r in res:
    s = summ.setdefault(r["scenario"], [0, 0])
    s[0] += int(r["ok"])
    s[1] += 1
  print("SUMMARY", json.dumps({k: f"{v[0]}/{v[1]}" for k, v in summ.items()}))
  json.dump({"results": res, "summary": summ}, open(a.out, "w"), indent=1, default=float)
  if a.video:
    os.environ.setdefault("MUJOCO_GL", "egl")
    fr = []
    for kind in ("supine", "prone"):
      run_boot(kind, a.getup, seed=7, limit_s=12, frames=fr)
    if a.dance:
      run_dance(a.getup, a.dance, seed=7, push=True, frames=fr)
    render(fr, a.video)
    print("video ->", a.video)


if __name__ == "__main__":
  main()
