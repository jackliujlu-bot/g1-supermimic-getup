"""Cut the first N motions out of a packed motion set (pack_v3 = pack_main + appended get-up copies).

Usage: python slice_pack.py data/pack_v3 data/pack_main [N=134603]
pack_main is exactly the first 134,603 motions of pack_v3 (names and segment ends checked 2026-11-26).
"""
import json
import os
import shutil
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 134603
os.makedirs(dst, exist_ok=True)
ends = np.load(os.path.join(src, "motion_ends.npy"))
names = json.load(open(os.path.join(src, "motion_names.json")))
frames = int(ends[n - 1])
for f in ("joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w"):
  a = np.load(os.path.join(src, f + ".npy"), mmap_mode="r")
  np.save(os.path.join(dst, f + ".npy"), np.ascontiguousarray(a[:frames]))
np.save(os.path.join(dst, "motion_ends.npy"), ends[:n])
json.dump(names[:n], open(os.path.join(dst, "motion_names.json"), "w"))
for f in ("fps.npy", "joint_names.json"):
  shutil.copy(os.path.join(src, f), dst)
print(f"{dst}: {n} motions, {frames} frames")
