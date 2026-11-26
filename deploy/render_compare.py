"""Side-by-side video of two get-up policies from identical starts (high oblique camera).

Usage: MUJOCO_GL=egl python render_compare.py LEFT.onnx RIGHT.onnx out.mp4 [--scen supine,prone,side]
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("out")
ap.add_argument("--scen", default="supine,prone,side")
ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--human-b", nargs="+", default=None,
                help="right panel: human-style path (settle -> tracker on the human clip) instead of B alone")
ap.add_argument("--labels", nargs=2, default=None)
ap.add_argument("--limit", type=float, default=6.0)
ap.add_argument("--walls", type=float, default=None)
a = ap.parse_args()

import imageio  # noqa: E402

m, _, _ = S.build_model()
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 480, 480)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 3.2, -55, 135


def frames_for(policy, kind, human=None):
  fr = []
  S.HUMAN = human
  r = S.run_boot(kind, policy, seed=a.seed, limit_s=a.limit, frames=fr, walls=a.walls)
  S.HUMAN = None
  print(kind, "human" if human else "ppo", {k: r.get(k) for k in ("ok", "t_stand", "sweep_area_m2", "wall_hit_ticks", "path")})
  return fr


from PIL import Image, ImageDraw, ImageFont  # noqa: E402

FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 22)


def draw(q, center, label=None, t=None):
  d.qpos[:] = q
  mujoco.mj_forward(m, d)
  cam.lookat[:] = [center[0], center[1], 0.3]
  r.update_scene(d, cam)
  img = r.render().copy()
  if label is None:
    return img
  im = Image.fromarray(img)
  g = ImageDraw.Draw(im)
  g.rectangle([0, 0, 480, 34], fill=(0, 0, 0))
  g.text((8, 4), label, font=FONT, fill=(255, 255, 255))
  g.text((400, 450), f"{t:4.1f} s", font=FONT, fill=(20, 20, 20))
  return np.asarray(im)


with imageio.get_writer(a.out, fps=25) as w:
  for kind in a.scen.split(","):
    fa, fb = frames_for(a.a, kind), frames_for(a.b, kind, a.human_b)
    n = max(len(fa), len(fb))
    ca, cb = fa[0][:2].copy(), fb[0][:2].copy()
    la, lb = a.labels or (None, None)
    for i in range(n):
      ia = draw(fa[min(i, len(fa) - 1)], ca, la, i / 25)
      ib = draw(fb[min(i, len(fb) - 1)], cb, lb, i / 25)
      sep = np.full((480, 8, 3), 255, np.uint8)
      w.append_data(np.concatenate([ia, sep, ib], axis=1))
print("video ->", a.out)
