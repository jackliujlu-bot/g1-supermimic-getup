"""sim2sim: get up by tracking a human get-up clip with the SuperMimic tracker.

The robot starts in the clip's first-frame pose (lifted out of the floor, random yaw,
perturbed joints), the tracker plays the clip, then the get-up/stand policy takes over to
hold the stand. Reports success and the same footprint / joint-span / wall metrics as sim2sim.

Usage: python test_ref_getup.py --clip CLIP.npz --dance CLIP.onnx --getup getup.onnx
         [--trials 5] [--perturb 0.1] [--walls 0.15]
"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402
from g1_stack import DEFAULT_Q, NJ, DancePolicy, GetupPolicy  # noqa: E402


def lift_out_of_floor(sim):
  mujoco.mj_forward(sim.m, sim.d)
  low = np.inf
  for g in range(sim.m.ngeom):
    if sim.m.geom_bodyid[g] == 0 or not (sim.m.geom_contype[g] or sim.m.geom_conaffinity[g]):
      continue
    r = sim.m.geom_size[g, 0]
    if sim.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      az = abs(sim.d.geom_xmat[g].reshape(3, 3)[2, 2])
      low = min(low, sim.d.geom_xpos[g, 2] - az * sim.m.geom_size[g, 1] - r)
    else:
      low = min(low, sim.d.geom_xpos[g, 2] - r)
  sim.d.qpos[2] += 0.005 - low
  mujoco.mj_forward(sim.m, sim.d)


def run(clip, dance_path, getup_path, seed, perturb, walls, limit_extra=4.0):
  rng = np.random.default_rng(seed)
  sim = S.Sim(rng)
  z = np.load(clip)
  q = np.zeros(7 + NJ)
  yaw = rng.uniform(-np.pi, np.pi)
  qy = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
  w1, x1, y1, z1 = qy
  w2, x2, y2, z2 = z["root_quat_w"][0]
  q[3:7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
  q[2] = z["root_pos_w"][0, 2]
  q[7:] = z["joint_pos"][0] + rng.uniform(-perturb, perturb, NJ)
  sim.set_state(q)
  lift_out_of_floor(sim)
  sim.settle_passive(1.0, kp=20, kd=2, target=q[7:].copy())
  sim.start_footprint()
  if walls is not None:
    sim.place_walls(walls)
    assert not sim.wall_contact()[0]
  dp = DancePolicy(dance_path)
  gp = GetupPolicy(getup_path)
  dp.start(sim.state())
  arms, splay = S._group_idx()
  arm_span = splay_span = 0.0
  mode, t_stand, held = "track", None, 0
  n = dp.num_frames + int(limit_extra / 0.02)
  for k in range(n):
    s = sim.state()
    if mode == "track" and dp.done:
      mode = "hold"
      gp.reset(last_action=dp.last_action)
    tgt = dp.act(s) if mode == "track" else gp.act(s)
    sim.apply(tgt)
    dq = np.abs(sim.d.qpos[7:] - DEFAULT_Q)
    arm_span = max(arm_span, float(dq[arms].max()))
    splay_span = max(splay_span, float(dq[splay].max()))
    if walls is not None:
      hit, f = sim.wall_contact()
      sim.wall_hits += int(hit)
      sim.wall_force = max(sim.wall_force, f)
    held = held + 1 if S.stable(sim, s) else 0
    if held:
      sim._stood = True
    if held >= 100 and t_stand is None:
      t_stand = (k - 99) * 0.02
  return dict(clip=os.path.basename(clip), seed=seed, ok=t_stand is not None, t_stand=t_stand,
              final_pelvis_z=round(sim.pelvis_z, 3), arm_span=round(arm_span, 2), splay_span=round(splay_span, 2),
              **sim.hw_metrics(), **sim.fp_metrics())


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--clip", required=True)
  ap.add_argument("--dance", required=True)
  ap.add_argument("--getup", required=True)
  ap.add_argument("--trials", type=int, default=5)
  ap.add_argument("--perturb", type=float, default=0.1)
  ap.add_argument("--walls", type=float, default=None)
  a = ap.parse_args()
  for t in range(a.trials):
    print(json.dumps(run(a.clip, a.dance, a.getup, 3000 + t, a.perturb, a.walls)), flush=True)
