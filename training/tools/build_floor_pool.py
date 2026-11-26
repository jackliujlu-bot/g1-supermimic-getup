"""Build a pool of on-the-floor start states for the get-up policy from the packed BONES data.

Takes frames whose pelvis is below --max-z from clips with floor body positions (lying,
sitting on floor, kneeling, crawling, all fours, ...). Root xy is zeroed; yaw is kept
(the env re-randomizes yaw). Velocities are scaled down (the clip may be moving).

Usage: python -I build_floor_pool.py --pack DIR --meta seed_metadata.csv --out pool.npz
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--meta", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max-z", type=float, default=0.55)
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

names = json.load(open(os.path.join(a.pack, "motion_names.json")))
ends = np.load(os.path.join(a.pack, "motion_ends.npy"))
starts = np.concatenate([[0], ends[:-1]])
m = pd.read_csv(a.meta, usecols=["move_name", "content_body_position"])
pos_by = dict(zip(m.move_name, m.content_body_position.astype(str)))
floor_kw = ("lying", "floor", "kneel", "crawl", "all fours", "sitting on heels", "crouch")
sel = [i for i, n in enumerate(names) if any(k in pos_by.get(n, "") for k in floor_kw)]
print(f"{len(sel)} floor clips of {len(names)}")

rp = np.load(os.path.join(a.pack, "root_pos_w.npy"), mmap_mode="r")
idx = np.concatenate([np.arange(starts[i], ends[i]) for i in sel])
z = np.asarray(rp[idx, 2])
idx = idx[z < a.max_z]
print(f"{len(idx)} candidate frames with pelvis z < {a.max_z}")
rng = np.random.default_rng(a.seed)
idx = np.sort(rng.choice(idx, size=min(a.n, len(idx)), replace=False))
out = {}
for k in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel"):
  out[k] = np.asarray(np.load(os.path.join(a.pack, f"{k}.npy"), mmap_mode="r")[idx]).astype(np.float32)
out["root_pos_w"][:, :2] = 0.0
for k in ("root_lin_vel_w", "root_ang_vel_w", "joint_vel"):
  out[k] *= 0.3
out["joint_names"] = np.array(json.load(open(os.path.join(a.pack, "joint_names.json"))))
np.savez(a.out, **out)
zz = out["root_pos_w"][:, 2]
print(f"saved {len(idx)} states -> {a.out}; pelvis z p5/p50/p95 = {np.percentile(zz,[5,50,95]).round(3)}")
