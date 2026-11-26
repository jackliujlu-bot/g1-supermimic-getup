"""Remove floor penetration from converted clips and pack them for mjlab's MotionLoader.

For each clip listed in --list (csv with column `rel`), compute the exact lowest point of
the G1 collision geoms (capsules / spheres) per frame, lift the root by the penetration
(window max over +-0.2 s, then smoothed, so the lift is continuous), recompute the root
z velocity, and pack all clips into <out>/ (<field>.npy, motion_ends.npy, fps.npy,
joint_names.json, motion_names.json, plus lift_stats.csv).

Usage: python -I fix_and_pack.py --clips DIR/clips --list keep.csv --out DIR [--workers N]
"""

from __future__ import annotations

import argparse
import functools
import json
import multiprocessing as mp
import os

import mujoco
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, maximum_filter1d

FIELDS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")
FPS = 50.0
TOL = 0.005  # allowed penetration (m)


@functools.cache
def model():
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
  m = get_spec().compile()
  coll = np.array([g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]])
  types = m.geom_type[coll]
  assert set(types.tolist()) <= {int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE)}
  return m, coll


def lowest_points(m, d, coll, pos, quat, jp):
  r = m.geom_size[coll, 0]
  half = np.where(m.geom_type[coll] == mujoco.mjtGeom.mjGEOM_CAPSULE, m.geom_size[coll, 1], 0.0)
  out = np.empty(len(pos))
  for k in range(len(pos)):
    d.qpos[:3] = pos[k]
    d.qpos[3:7] = quat[k]
    d.qpos[7:] = jp[k]
    mujoco.mj_kinematics(m, d)
    cz = d.geom_xpos[coll, 2]
    az = d.geom_xmat[coll].reshape(-1, 3, 3)[:, 2, 2]  # z component of capsule axis
    out[k] = (cz - np.abs(az) * half - r).min()
  return out


def fix_one(rel: str, clips: str):
  m, coll = model()
  d = mujoco.MjData(m)
  with np.load(os.path.join(clips, rel[:-4] + ".npz")) as z:
    c = {k: z[k] for k in (*FIELDS, "joint_names")}
  low = lowest_points(m, d, coll, c["root_pos_w"], c["root_quat_w"], c["joint_pos"])
  raw = np.maximum(0.0, -(low + TOL))
  lift = gaussian_filter1d(maximum_filter1d(raw, size=21, mode="nearest"), sigma=3, mode="nearest")
  lift = np.maximum(lift, raw)  # never leave residual penetration
  c["root_pos_w"] = c["root_pos_w"].copy()
  c["root_pos_w"][:, 2] += lift.astype(np.float32)
  c["root_lin_vel_w"] = c["root_lin_vel_w"].copy()
  c["root_lin_vel_w"][:, 2] = np.gradient(c["root_pos_w"][:, 2].astype(np.float64), 1 / FPS).astype(np.float32)
  st = dict(rel=rel, low_min=float(low.min()), lift_max=float(lift.max()),
            low_after_min=float((low + lift).min()), frames=len(low))
  return c, st


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--clips", required=True)
  ap.add_argument("--list", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--workers", type=int, default=32)
  a = ap.parse_args()
  rels = pd.read_csv(a.list)["rel"].tolist()
  os.makedirs(a.out, exist_ok=True)
  parts = {k: [] for k in FIELDS}
  lengths, names, stats = [], [], []
  joint_names = None
  with mp.get_context("spawn").Pool(a.workers) as pool:
    for i, (c, st) in enumerate(pool.imap(functools.partial(fix_one, clips=a.clips), rels, chunksize=16)):
      jn = c.pop("joint_names").tolist()
      if joint_names is None:
        joint_names = jn
      assert jn == joint_names
      for k in FIELDS:
        parts[k].append(c[k])
      lengths.append(len(c["joint_pos"]))
      names.append(os.path.splitext(os.path.basename(st["rel"]))[0])
      stats.append(st)
      if (i + 1) % 5000 == 0:
        print(f"{i+1}/{len(rels)}", flush=True)
  for k in FIELDS:
    np.save(os.path.join(a.out, f"{k}.npy"), np.concatenate(parts[k]))
  np.save(os.path.join(a.out, "motion_ends.npy"), np.cumsum(lengths))
  np.save(os.path.join(a.out, "fps.npy"), np.array([FPS]))
  json.dump(joint_names, open(os.path.join(a.out, "joint_names.json"), "w"))
  json.dump(names, open(os.path.join(a.out, "motion_names.json"), "w"))
  pd.DataFrame(stats).to_csv(os.path.join(a.out, "lift_stats.csv"), index=False)
  print(f"packed {len(names)} clips, {sum(lengths)} frames ({sum(lengths)/FPS/3600:.2f} h) -> {a.out}", flush=True)


if __name__ == "__main__":
  main()
