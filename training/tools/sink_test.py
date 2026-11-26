"""Does a lying robot sink through the training floor? (mjlab / MuJoCo Warp, robust task + terrain)

Resets every env to the first (lying) frame of a get-up clip, holds the joints at that pose for
3 s and reports pelvis height above the env origin, grouped by floor softness u and by flat vs
heightfield tile. A floor that works keeps the lying pelvis at ~0.1 m; tunnelling shows as z < 0.

Usage: G1R_TERRAIN=1 G1R_HARD_PROB=0 python sink_test.py CLIP.npz [--envs 512] [--seconds 3]
"""
import argparse
import os
import sys

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robust_task  # noqa: F401,E402

ap = argparse.ArgumentParser()
ap.add_argument("clip")
ap.add_argument("--envs", type=int, default=512)
ap.add_argument("--seconds", type=float, default=3.0)
a = ap.parse_args()
dev = "cuda:0"
cfg = load_env_cfg("SuperMimic-GenCoord-G1-Robust", play=False)
cfg.scene.num_envs = a.envs
cfg.commands["motion"].motion_file = [a.clip]
cfg.commands["motion"].sampling_mode = "start"
cfg.terminations = {k: v for k, v in cfg.terminations.items() if k == "time_out"}
env = ManagerBasedRlEnv(cfg=cfg, device=dev)
env.reset()
robot = env.scene["robot"]
cmd = env.command_manager.get_term("motion")
m = env.sim.model
pelvis = robot.indexing.body_ids[0] if hasattr(robot.indexing, "body_ids") else None
u = (m.geom_solref[:, :, 0].max(dim=1).values - 0.02) / 0.18  # per env softness (robot geoms)
origins = env.scene.env_origins.clone()
# hold the clip's first pose: action = 0 is the default pose, so command the offset instead
am = env.action_manager
term = am.get_term(am.active_terms[0])
scale = term._scale if hasattr(term, "_scale") else 1.0
offset = term._offset if hasattr(term, "_offset") else 0.0
q0 = cmd.joint_pos.clone()
act = ((q0 - offset) / scale).clamp(-10, 10)
zs = []
for k in range(int(a.seconds / env.step_dt)):
  env.step(act)
  zs.append((robot.data.root_link_pos_w[:, 2] - origins[:, 2]).clone())
z = torch.stack(zs).cpu().numpy()  # T x N
u = u.cpu().numpy()
# terrain tile type per env from the generator layout (cols are tile types in proportion order)
tl = env.scene.terrain
types = getattr(tl, "terrain_types", None)
flat = (types.cpu().numpy() == 0) if types is not None else np.zeros(a.envs, bool)
print(f"start pelvis z {np.median(z[0]):.3f}; after {a.seconds}s: median {np.median(z[-1]):.3f}")
for name, sel in (("flat", flat), ("uneven", ~flat)):
  for lo, hi in ((0, 0.01), (0.01, 0.4), (0.4, 0.7), (0.7, 1.01), (1.01, 1.5), (1.5, 2.01)):
    s = sel & (u >= lo) & (u < hi)
    if s.sum():
      zf = z[-1, s]
      print(f"{name:6s} u {lo:.2f}-{hi:.2f}: n={s.sum():4d} final pelvis z median {np.median(zf):.3f} "
            f"min {zf.min():.3f}  sunk below floor (z<0.03): {(zf < 0.03).mean() * 100:.0f}%")
