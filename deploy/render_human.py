"""Videos of the BONES-SEED human get-up motions (retargeted to G1, kinematic replay) and
side-by-side comparison against get-up policies started from the same first-frame pose.

Usage:
  MUJOCO_GL=egl python render_human.py gallery CLIPS_ROOT out.mp4
  MUJOCO_GL=egl python render_human.py compare CLIPS_ROOT out.mp4 --policies v3.onnx v4A.onnx
"""
import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402
from g1_stack import GetupPolicy  # noqa: E402
from test_ref_getup import lift_out_of_floor  # noqa: E402

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
GALLERY = [
  ("231010/stand_up_lying_R_002__A473", "仰卧起身"),
  ("231010/stand_up_lying_stomach_R_002__A472", "俯卧起身"),
  ("231010/stand_up_lying_side_R_002__A472", "侧卧起身"),
  ("231024/faint_stand_up_lying_R_001__A492", "晕倒后仰卧起身"),
  ("231024/faint_stand_up_lying_stomach_R_001__A492", "晕倒后俯卧起身"),
  ("231024/faint_stand_up_lying_side_R_001__A492", "晕倒后侧卧起身"),
]
COMPARE = GALLERY[:3]
H = W = 420
FPS = 25

m, _, _ = S.build_model()
d = mujoco.MjData(m)
ren = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 3.4, -35, 135
font = ImageFont.truetype(FONT, 22)
small = ImageFont.truetype(FONT, 16)


def load_clip(root, rel):
  z = np.load(os.path.join(root, rel + ".npz"))
  q = np.concatenate([z["root_pos_w"], z["root_quat_w"], z["joint_pos"]], 1)
  q[:, :2] -= q[0, :2]
  return q[::2]  # 50 Hz -> 25 fps


def draw(q, center, label, t):
  d.qpos[:] = q
  mujoco.mj_forward(m, d)
  cam.lookat[:] = [center[0], center[1], 0.35]
  ren.update_scene(d, cam)
  img = Image.fromarray(ren.render())
  g = ImageDraw.Draw(img)
  g.rectangle([0, 0, W, 34], fill=(0, 0, 0))
  g.text((8, 4), label, font=font, fill=(255, 255, 255))
  g.text((W - 62, H - 24), f"{t:4.1f} s", font=small, fill=(20, 20, 20))
  return np.asarray(img)


def run_policy(q0, path, seed=0, seconds=6.0):
  sim = S.Sim(np.random.default_rng(seed))
  sim.set_state(q0.copy())
  lift_out_of_floor(sim)
  sim.settle_passive(1.0, kp=20, kd=2, target=q0[7:].copy())
  sim.start_footprint()
  gp = GetupPolicy(path)
  gp.reset()
  fr = []
  held, t_stand = 0, None
  for k in range(int(seconds / 0.02)):
    st = sim.state()
    sim.apply(gp.act(st))
    held = held + 1 if S.stable(sim, st) else 0
    if held:
      sim._stood = True
    if held >= 25 and t_stand is None:
      t_stand = (k - 24) * 0.02
    if k % 2 == 0:
      fr.append(sim.d.qpos.copy())
  return fr, sim.sweep_area(), t_stand


def human_sweep(clip):
  """Convex-hull area of the kinematic replay until the pelvis is up and the motion settles."""
  sim = S.Sim(np.random.default_rng(0), randomize=False)
  pts, t_up = [], None
  for i, q in enumerate(clip):
    sim.set_state(q)
    mujoco.mj_forward(sim.m, sim.d)
    pts.append(sim.body_points_xy())
    if t_up is None and q[2] > 0.70 and i > 5:
      t_up = i / FPS
      break
  from scipy.spatial import ConvexHull
  return float(ConvexHull(np.concatenate(pts)).volume), t_up


def grid(tiles, cols):
  sep = 6
  rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
  out = []
  for r in rows:
    r = r + [np.full_like(tiles[0], 255)] * (cols - len(r))
    line = []
    for j, t in enumerate(r):
      if j:
        line.append(np.full((H, sep, 3), 255, np.uint8))
      line.append(t)
    out.append(np.concatenate(line, 1))
  res = []
  for i, o in enumerate(out):
    if i:
      res.append(np.full((sep, o.shape[1], 3), 255, np.uint8))
    res.append(o)
  return np.concatenate(res, 0)


def title_card(text, sub, shape, n=50):
  img = Image.new("RGB", (shape[1], shape[0]), (20, 20, 20))
  g = ImageDraw.Draw(img)
  big = ImageFont.truetype(FONT, 40)
  g.text((40, shape[0] // 2 - 50), text, font=big, fill=(255, 255, 255))
  g.text((40, shape[0] // 2 + 10), sub, font=font, fill=(200, 200, 200))
  return [np.asarray(img)] * n


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mode", choices=["gallery", "compare"])
  ap.add_argument("root")
  ap.add_argument("out")
  ap.add_argument("--policies", nargs="*", default=[])
  ap.add_argument("--names", nargs="*", default=["v3部署版", "v4A紧凑版"])
  ap.add_argument("--clips", nargs="*", default=None, help="gallery clips as rel=label (default: BONES-SEED set)")
  ap.add_argument("--cols", type=int, default=3)
  a = ap.parse_args()
  import imageio

  with imageio.get_writer(a.out, fps=FPS, macro_block_size=2) as w:
    if a.mode == "gallery":
      gal = [tuple(c.split("=", 1)) for c in a.clips] if a.clips else GALLERY
      clips = [(load_clip(a.root, rel), name) for rel, name in gal]
      n = max(len(c) for c, _ in clips) + 15
      for i in range(n):
        tiles = [draw(c[min(i, len(c) - 1)], c[0][:2], "真人 · " + nm, min(i, len(c) - 1) / FPS) for c, nm in clips]
        w.append_data(grid(tiles, a.cols))
    else:
      for rel, name in COMPARE:
        hc = load_clip(a.root, rel)
        hs, ht = human_sweep(hc)
        cols = [(hc, f"真人 扫过{hs:.2f}m² 站起{ht:.1f}s")]
        for p, nm in zip(a.policies, a.names):
          fr, sw, ts = run_policy(hc[0], p)
          print(name, nm, "sweep", round(sw, 2), "t_stand", ts, "| human", round(hs, 2), ht, flush=True)
          cols.append((fr, f"{nm} 扫过{sw:.2f}m² 站起{ts:.1f}s" if ts else f"{nm} 扫过{sw:.2f}m² 未站稳"))
        n = max(len(c) for c, _ in cols) + 15
        shape = (H, W * len(cols) + 6 * (len(cols) - 1), 3)
        for f in title_card(name, "左：真人动捕（BONES-SEED，已映射到 G1）  其余：训练的策略，从同一个起始姿态开始", shape):
          w.append_data(f)
        for i in range(n):
          tiles = [draw(c[min(i, len(c) - 1)], c[0][:2], lb, min(i, len(c) - 1) / FPS) for c, lb in cols]
          w.append_data(grid(tiles, len(cols)))
  print("video ->", a.out)


if __name__ == "__main__":
  main()
