"""Build the fine-tune motion pack: the full training pack + many copies of the get-up clip(s).

Usage: python -I pack_finetune.py --base data/pack_main --out data/pack_ft \
         --clip exports/refslow/getup_s88.npz:2000 [--clip other.npz:N ...]
"""
import argparse
import json
import os

import numpy as np

FIELDS = ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel")
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--clip", action="append", required=True, help="PATH.npz:COPIES")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
names = json.load(open(os.path.join(a.base, "motion_names.json")))
ends = np.load(os.path.join(a.base, "motion_ends.npy"))
jn = json.load(open(os.path.join(a.base, "joint_names.json")))
extra, extra_names = [], []
for spec in a.clip:
  path, n = spec.rsplit(":", 1)
  z = np.load(path)
  assert [str(x) for x in z["joint_names"]] == jn, "joint order differs"
  base = os.path.splitext(os.path.basename(path))[0]
  for i in range(int(n)):
    extra.append({k: z[k].astype(np.float32) for k in FIELDS})
    extra_names.append(f"{base}__ft{i:04d}")
lens = [e["joint_pos"].shape[0] for e in extra]
new_ends = np.concatenate([ends, ends[-1] + np.cumsum(lens)])
for k in FIELDS:
  b = np.load(os.path.join(a.base, f"{k}.npy"), mmap_mode="r")
  out = np.lib.format.open_memmap(os.path.join(a.out, f"{k}.npy"), mode="w+", dtype=b.dtype,
                                  shape=(b.shape[0] + sum(lens),) + b.shape[1:])
  out[: b.shape[0]] = b
  out[b.shape[0]:] = np.concatenate([e[k] for e in extra]).astype(b.dtype)
  out.flush()
  del out
  print(k, "done")
np.save(os.path.join(a.out, "motion_ends.npy"), new_ends)
json.dump(names + extra_names, open(os.path.join(a.out, "motion_names.json"), "w"))
json.dump(jn, open(os.path.join(a.out, "joint_names.json"), "w"))
np.save(os.path.join(a.out, "fps.npy"), np.load(os.path.join(a.base, "fps.npy")))
print(f"pack: {len(names)} + {len(extra_names)} clips, {int(new_ends[-1])} frames "
      f"(added {sum(lens)} = {100 * sum(lens) / new_ends[-1]:.2f}%)")
