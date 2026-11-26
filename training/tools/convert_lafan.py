"""LAFAN1 G1 retarget CSV (30 Hz: xyz, quat xyzw, 29 joints rad) -> 50 Hz mjlab clips, and
cut out every get-up segment (pelvis rises from the floor to standing).

For each segment it reports how it starts (lying on back / stomach / side, from the pelvis
orientation), duration, how long a hand is pressed on the floor while the pelvis is still low
(= hand-supported get-up), peak joint speed and floor penetration.

Usage: python convert_lafan.py CSV_DIR OUT_DIR
Writes OUT_DIR/full/<name>.npz, OUT_DIR/seg/<name>_s<k>.npz and OUT_DIR/segments.csv
"""
import csv
import os
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_bones import geom_bottoms, model_and_joints, velocities  # noqa: E402

IN_FPS, OUT_FPS = 30.0, 50.0


def load(path):
  a = np.loadtxt(path, delimiter=",")
  pos, rot, jp = a[:, :3], Rotation.from_quat(a[:, 3:7]), a[:, 7:36]
  n = len(a)
  dur = (n - 1) / IN_FPS
  t = np.arange(int(dur * OUT_FPS) + 1) / OUT_FPS
  ti = np.arange(n) / IN_FPS
  pos = np.stack([np.interp(t, ti, pos[:, j]) for j in range(3)], 1)
  jp = np.stack([np.interp(t, ti, jp[:, j]) for j in range(29)], 1)
  rot = Slerp(ti, rot)(t)
  return pos, rot, jp


def save(dst, pos, rot, jp, joints):
  lin, ang, jv = velocities(pos, rot, jp)
  quat = rot.as_quat()[:, [3, 0, 1, 2]]
  for k in range(1, len(quat)):
    if np.dot(quat[k], quat[k - 1]) < 0:
      quat[k] = -quat[k]
  np.savez(dst, root_pos_w=pos.astype(np.float32), root_quat_w=quat.astype(np.float32),
           root_lin_vel_w=lin.astype(np.float32), root_ang_vel_w=ang.astype(np.float32),
           joint_pos=jp.astype(np.float32), joint_vel=jv.astype(np.float32),
           joint_names=np.array(joints), fps=np.array(OUT_FPS))
  return jv


def body_heights(m, d, pos, quat, jp, names):
  ids = [m.body(n).id for n in names]
  out = np.zeros((len(pos), len(ids)))
  for k in range(len(pos)):
    d.qpos[:3], d.qpos[3:7], d.qpos[7:] = pos[k], quat[k], jp[k]
    mujoco.mj_kinematics(m, d)
    out[k] = d.xpos[ids, 2]
  return out


def posture(rot):
  g = rot.inv().apply([0, 0, -1.0])  # gravity in pelvis frame
  if g[2] > -0.5 and abs(g[0]) > abs(g[1]):
    return "supine" if g[0] > 0 else "prone"
  if g[2] > -0.5:
    return "side"
  return "upright"


def main():
  src, out = sys.argv[1], sys.argv[2]
  os.makedirs(f"{out}/full", exist_ok=True)
  os.makedirs(f"{out}/seg", exist_ok=True)
  m, joints, coll = model_and_joints()
  d = mujoco.MjData(m)
  hands = [n for n in ("left_wrist_yaw_link", "right_wrist_yaw_link") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
  rows = []
  for f in sorted(os.listdir(src)):
    if not f.endswith(".csv"):
      continue
    name = f[:-4]
    pos, rot, jp = load(os.path.join(src, f))
    save(f"{out}/full/{name}.npz", pos, rot, jp, joints)
    quat = rot.as_quat()[:, [3, 0, 1, 2]]
    z = pos[:, 2]
    # get-up segments: pelvis passes 0.4 -> 0.7 going up; start = last time it was at its low
    # plateau (< 0.3) before that, end = 1.5 s after it reaches 0.7
    up = np.where((z[1:] >= 0.7) & (z[:-1] < 0.7))[0] + 1
    k_seg = 0
    prev_e = 0
    for e in up:
      lowk = np.where(z[prev_e:e] < 0.3)[0] + prev_e
      if not len(lowk):
        continue  # pelvis did not reach the floor since the last get-up (z jitter around 0.7)
      prev_e = e
      lo = lowk[-1]
      kmin = int(lowk[0] + np.argmin(z[lowk[0]:lo + 1]))
      # walk back over the lying plateau to where the body came to rest (max 3 s)
      s = lo
      while s > 0 and z[s - 1] < 0.35 and lo - s < 150:
        s -= 1
      s = max(s, lo - 50, kmin)  # start at most 1 s before the push-up begins, not mid-fall
      e2 = min(len(z) - 1, e + 75)
      seg = slice(s, e2 + 1)
      dst = f"{out}/seg/{name}_s{k_seg}.npz"
      jv = save(dst, pos[seg], rot[seg], jp[seg], joints)
      zb, _ = geom_bottoms(m, d, pos[seg], quat[seg], jp[seg], coll, stride=1)
      hz = body_heights(m, d, pos[seg], quat[seg], jp[seg], hands) if hands else np.ones((e2 + 1 - s, 1))
      low = z[seg] < 0.55
      hand_s = float(((hz.min(1) < 0.12) & low).sum() / OUT_FPS)
      rows.append(dict(clip=os.path.basename(dst)[:-4], start=posture(rot[kmin]), t0=round(s / OUT_FPS, 2),
                       dur=round((e2 - s) / OUT_FPS, 2), hand_support_s=round(hand_s, 2),
                       max_jvel=round(float(np.abs(jv).max()), 1), min_bottom=round(float(zb.min()), 3),
                       end_z=round(float(z[e2]), 2)))
      k_seg += 1
  with open(f"{out}/segments.csv", "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
  for r in rows:
    print(r)


if __name__ == "__main__":
  main()
