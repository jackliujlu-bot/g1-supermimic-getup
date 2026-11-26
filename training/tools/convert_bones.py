"""BONES-SEED G1 CSV (120 Hz, cm, extrinsic-xyz Euler deg, joints deg) -> 50 Hz mjlab clips.

Writes one .npz per clip (mjlab MotionLoader keys) into <out>/clips/<date>/<name>.npz and a
per-clip stats table <out>/stats.csv used for filtering:
  frames, dur_s, zmin_min / zmin_p50 (lowest collision-geom bottom over the clip),
  support_s (longest run with lowest point > 0.12 m, i.e. body held up by a prop),
  pen (most negative lowest point), max_jvel, max_root_speed.

Usage: python -I convert_bones.py --csv-root .../g1/csv --out DIR [--limit N] [--workers W]
"""

from __future__ import annotations

import argparse
import csv
import functools
import multiprocessing as mp
import os
import sys
import traceback

import mujoco
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

IN_FPS = 120.0
OUT_FPS = 50.0
ROOT_COLS = ["root_translateX", "root_translateY", "root_translateZ",
             "root_rotateX", "root_rotateY", "root_rotateZ"]


@functools.cache
def model_and_joints(euler: str = "xyz"):
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
  spec = get_spec()
  m = spec.compile()
  joints = [m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
  assert m.nq == 7 + len(joints)
  coll = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
  if not coll:
    coll = list(range(m.ngeom))
  return m, joints, np.array(coll)


def parse(path: str, joints: list[str], euler: str = "xyz"):
  df = pd.read_csv(path)
  cols = {c.removesuffix("_dof"): c for c in df.columns[7:]}
  missing = [j for j in joints if j not in cols]
  if missing:
    raise ValueError(f"missing joints {missing}")
  root = df[ROOT_COLS].to_numpy(np.float64)
  pos = root[:, :3] / 100.0
  rot = Rotation.from_euler(euler, root[:, 3:6], degrees=True)
  jp = np.deg2rad(df[[cols[j] for j in joints]].to_numpy(np.float64))
  return pos, rot, jp


def resample(pos, rot: Rotation, jp):
  n = len(pos)
  dur = (n - 1) / IN_FPS
  n_out = int(np.floor(dur * OUT_FPS + 1e-9)) + 1
  t = np.minimum(np.arange(n_out) / OUT_FPS, dur)
  x = t * IN_FPS
  i0 = np.clip(np.floor(x).astype(int), 0, n - 2)
  w = (x - i0)[:, None]
  lerp = lambda a: a[i0] * (1 - w) + a[i0 + 1] * w
  q = Slerp(np.arange(n) / IN_FPS, rot)(t)
  return lerp(pos), q, lerp(jp)


def velocities(pos, rot: Rotation, jp):
  dt = 1.0 / OUT_FPS
  lin = np.gradient(pos, dt, axis=0)
  jv = np.gradient(jp, dt, axis=0)
  # world-frame angular velocity from consecutive orientations (central where possible)
  rv = (rot[1:] * rot[:-1].inv()).as_rotvec() / dt  # [T-1,3] world frame
  ang = np.zeros_like(pos)
  ang[0] = rv[0]
  ang[-1] = rv[-1]
  ang[1:-1] = 0.5 * (rv[:-1] + rv[1:])
  return lin, ang, jv


def geom_bottoms(m, d, pos, quat_wxyz, jp, coll, stride=2):
  """Lowest collision-geom AABB bottom per (strided) frame."""
  out = []
  for k in range(0, len(pos), stride):
    d.qpos[:3] = pos[k]
    d.qpos[3:7] = quat_wxyz[k]
    d.qpos[7:] = jp[k]
    mujoco.mj_kinematics(m, d)
    xm = d.geom_xmat[coll].reshape(-1, 3, 3)
    aabb = m.geom_aabb[coll]  # center(3), half(3) in geom frame
    cz = d.geom_xpos[coll, 2] + np.einsum("gj,gj->g", xm[:, 2, :], aabb[:, :3])
    hz = np.einsum("gj,gj->g", np.abs(xm[:, 2, :]), aabb[:, 3:])
    out.append((cz - hz).min())
  return np.array(out), stride / OUT_FPS


def longest_run(mask: np.ndarray) -> int:
  best = cur = 0
  for v in mask:
    cur = cur + 1 if v else 0
    best = max(best, cur)
  return best


def convert_one(rel: str, csv_root: str, out: str, euler: str = "xyz"):
  try:
    m, joints, coll = model_and_joints()
    d = mujoco.MjData(m)
    pos, rot, jp = parse(os.path.join(csv_root, rel), joints, euler)
    if len(pos) < 3:
      return rel, None, "too short"
    pos, rot, jp = resample(pos, rot, jp)
    lin, ang, jv = velocities(pos, rot, jp)
    quat = rot.as_quat()[:, [3, 0, 1, 2]]
    quat *= np.sign(quat[:, :1] + 1e-12)  # w >= 0 not required, but keep continuity below
    for k in range(1, len(quat)):  # hemisphere continuity
      if np.dot(quat[k], quat[k - 1]) < 0:
        quat[k] = -quat[k]
    zb, dts = geom_bottoms(m, d, pos, quat, jp, coll)
    stats = dict(
      frames=len(pos), dur_s=(len(pos) - 1) / OUT_FPS,
      zmin_min=float(zb.min()), zmin_p50=float(np.median(zb)), zmin_p95=float(np.percentile(zb, 95)),
      support_s=longest_run(zb > 0.12) * dts, pen=float(zb.min()),
      max_jvel=float(np.abs(jv).max()), max_root_speed=float(np.linalg.norm(lin, axis=1).max()),
      start_pelvis_z=float(pos[0, 2]), end_pelvis_z=float(pos[-1, 2]),
    )
    dst = os.path.join(out, "clips", rel[:-4] + ".npz")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez(dst, root_pos_w=pos.astype(np.float32), root_quat_w=quat.astype(np.float32),
             root_lin_vel_w=lin.astype(np.float32), root_ang_vel_w=ang.astype(np.float32),
             joint_pos=jp.astype(np.float32), joint_vel=jv.astype(np.float32),
             joint_names=np.array(joints), fps=np.array(OUT_FPS))
    return rel, stats, None
  except Exception as e:  # noqa: BLE001
    return rel, None, f"{type(e).__name__}: {e}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--csv-root", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--workers", type=int, default=48)
  ap.add_argument("--euler", default="xyz")
  ap.add_argument("--only", default="", help="comma list of relative csv paths")
  a = ap.parse_args()
  if a.only:
    rels = a.only.split(",")
  else:
    rels = sorted(os.path.relpath(os.path.join(r, f), a.csv_root)
                  for r, _, fs in os.walk(a.csv_root) for f in fs if f.endswith(".csv"))
  if a.limit:
    rels = rels[: a.limit]
  os.makedirs(a.out, exist_ok=True)
  fn = functools.partial(convert_one, csv_root=a.csv_root, out=a.out, euler=a.euler)
  keys = None
  ok = bad = 0
  with open(os.path.join(a.out, "stats.csv"), "w", newline="") as fs, \
       open(os.path.join(a.out, "errors.txt"), "w") as fe, \
       mp.get_context("spawn").Pool(a.workers) as pool:
    w = None
    for i, (rel, st, err) in enumerate(pool.imap_unordered(fn, rels, chunksize=16)):
      if st is None:
        bad += 1
        fe.write(f"{rel}\t{err}\n")
      else:
        ok += 1
        if w is None:
          w = csv.DictWriter(fs, fieldnames=["rel", *st.keys()])
          w.writeheader()
        w.writerow({"rel": rel, **st})
      if (i + 1) % 2000 == 0:
        print(f"{i+1}/{len(rels)} ok={ok} bad={bad}", flush=True)
  print(f"done ok={ok} bad={bad}", flush=True)


if __name__ == "__main__":
  main()
