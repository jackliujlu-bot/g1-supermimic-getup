"""Cut the two chosen LAFAN1 get-ups (hand-supported, slow) from the 50 Hz full clips, remove floor
penetration (same method as the training pack), add a 1 s standing hold at the end, and also save
a x1.25 slower copy. Start x,y moved to the origin.

  prone : fallAndGetUp2_subject2 28.0-35.0 s  lying face down -> push up on hands -> kneel -> stand
  supine: fallAndGetUp2_subject2 12.0-18.6 s  lying on back -> roll to side -> hands push -> squat -> stand

Usage: python make_lafan_getups.py FULL_DIR OUT_DIR
"""
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fix_and_pack import FIELDS, fix_one  # noqa: E402

PICKS = {"lafan_prone": ("fallAndGetUp2_subject2", 28.0, 35.0),
         "lafan_supine": ("fallAndGetUp2_subject2", 12.0, 18.6)}
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
for name, (clip, t0, t1) in PICKS.items():
  z = np.load(f"{src}/{clip}.npz")
  c = {k: z[k][int(t0 * 50):int(t1 * 50) + 1].copy() for k in FIELDS}
  c["root_pos_w"][:, :2] -= c["root_pos_w"][0, :2]
  for k in FIELDS:  # 1 s hold of the last frame
    tail = np.repeat(c[k][-1:], 50, 0)
    if k in ("root_lin_vel_w", "root_ang_vel_w", "joint_vel"):
      tail[:] = 0
    c[k] = np.concatenate([c[k], tail])
  tmp = f"{out}/_raw"
  os.makedirs(tmp, exist_ok=True)
  np.savez(f"{tmp}/{name}.npz", joint_names=z["joint_names"], fps=np.array(50.0), **c)
  fixed, st = fix_one(f"{name}.csv", tmp)
  print(st)
  np.savez(f"{out}/{name}.npz", fps=np.array(50.0), **fixed)
  subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stretch_clip.py"),
                  f"{out}/{name}.npz", f"{out}/{name}_x1.25.npz", "1.25"], check=True)
