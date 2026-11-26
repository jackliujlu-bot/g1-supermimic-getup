"""Time-stretch a 50 Hz motion clip by factor s (s>1 = slower). usage: stretch_clip.py in.npz out.npz s"""
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
src, dst, s = sys.argv[1], sys.argv[2], float(sys.argv[3])
z = dict(np.load(src))
n = len(z["joint_pos"]); t = np.arange(n); tn = np.arange(0, n - 1 + 1e-9, 1 / s)
out = dict(z)
out["joint_pos"] = np.stack([np.interp(tn, t, z["joint_pos"][:, j]) for j in range(29)], 1)
out["root_pos_w"] = np.stack([np.interp(tn, t, z["root_pos_w"][:, j]) for j in range(3)], 1)
qw = z["root_quat_w"]; rot = R.from_quat(qw[:, [1, 2, 3, 0]])
q = Slerp(t, rot)(tn).as_quat()[:, [3, 0, 1, 2]]
out["root_quat_w"] = q
dt = 0.02
out["joint_vel"] = np.gradient(out["joint_pos"], dt, axis=0)
out["root_lin_vel_w"] = np.gradient(out["root_pos_w"], dt, axis=0)
rv = Slerp(t, rot)(tn)
w = (rv[1:] * rv[:-1].inv()).as_rotvec() / dt
out["root_ang_vel_w"] = np.concatenate([w, w[-1:]], 0)
np.savez(dst, **out)
print(dst, n, "->", len(tn))
