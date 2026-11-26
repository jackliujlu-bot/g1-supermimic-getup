"""Collect human get-up reference frames (BONES-SEED, retargeted to G1) for the style prior.

Takes the floor-to-standing clips (lying back/side/stomach, faint & stand up, sit/kneel stops),
excluding prop-assisted (crutch, chair) and pure-fall clips. Saves per frame:
joint_pos [29], projected gravity in the pelvis frame [3], joint_vel [29].

Usage: python -I build_getup_ref.py --pack DIR --list getup_candidates.csv --out getup_ref.npz
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--pack", required=True)
ap.add_argument("--list", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

c = pd.read_csv(a.list)
bad = c.move_name.str.contains(r"crutch|chair|fall_spin|run_fall|icecream_fall|puke_walk", regex=True)
c = c[~bad]
names = json.load(open(os.path.join(a.pack, "motion_names.json")))
idx_of = {n: i for i, n in enumerate(names)}
ends = np.load(os.path.join(a.pack, "motion_ends.npy"))
starts = np.concatenate([[0], ends[:-1]])
sel = [idx_of[n] for n in c.move_name if n in idx_of]
rows = np.concatenate([np.arange(starts[i], ends[i]) for i in sel])
L = lambda k: np.asarray(np.load(os.path.join(a.pack, f"{k}.npy"), mmap_mode="r")[rows])
q = L("joint_pos").astype(np.float32)
qd = L("joint_vel").astype(np.float32)
quat = L("root_quat_w")  # wxyz
g = Rotation.from_quat(quat[:, [1, 2, 3, 0]]).inv().apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)
pz = L("root_pos_w")[:, 2]
np.savez(a.out, joint_pos=q, proj_gravity=g, joint_vel=qd, pelvis_z=pz.astype(np.float32),
         joint_names=np.array(json.load(open(os.path.join(a.pack, "joint_names.json")))))
print(f"{len(sel)} clips, {len(rows)} frames ({len(rows)/50/60:.1f} min); pelvis z range {pz.min():.2f}-{pz.max():.2f}")
print(sorted(set(c.move_name.str.replace(r'_(R_)?\d{3}.*', '', regex=True)))[:40])
