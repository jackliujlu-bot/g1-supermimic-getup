"""Side-by-side video of the last seconds of standing after the get-up (is it visibly calm?).
usage: MUJOCO_GL=egl python render_stand.py OUT.mp4 SETTLE_P CLIP_P SETTLE_S CLIP_S --cells LABEL:TRACKER:FLOOR ... [--seconds 8]
FLOOR: hard | soft | roughsoft"""
import argparse, os, sys
import mujoco, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("r", nargs=4)
ap.add_argument("--cells", nargs="+", required=True)
ap.add_argument("--scenario", default="prone"); ap.add_argument("--seed", type=int, default=1000)
ap.add_argument("--seconds", type=float, default=8.0)
a = ap.parse_args()
FL = {"hard": (None, None), "soft": (1.0, None), "rough": (None, 0.015), "roughsoft": (0.6, 0.015)}
H, W = 360, 360
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
cells = []
for c in a.cells:
  label, tracker, floor = c.split(":")
  S.SOFT, S.TERRAIN = FL[floor]
  S.STAND_POSES[:] = [S.DEFAULT_Q]
  S.UNIFIED = {"tracker": tracker, "routines": [(a.r[0], a.r[1]), (a.r[2], a.r[3])], "dance_clip": None}
  fr = []
  res = S.run_boot(a.scenario, None, a.seed, limit_s=25.0, frames=fr)
  fr = fr[-int(a.seconds * 25):]
  S.TERRAIN_SEED = a.seed
  m, _, _ = S.build_model(); d = mujoco.MjData(m); r = mujoco.Renderer(m, H, W)
  cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 2.2, -5, 90
  imgs = []
  for q in fr:
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    cam.lookat[:] = [q[0], q[1], 0.45]; r.update_scene(d, cam)
    im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
    dr.text((8, 6), f"{label}  swing {res['end_swing_deg']}°", fill=(255, 255, 0), font=font)
    imgs.append(np.array(im))
  cells.append(imgs)
  print(label, res["ok"], res["end_swing_deg"], res["path"][-60:], flush=True)
n = min(len(c) for c in cells); cols = 2
with imageio.get_writer(a.out, fps=25) as w:
  for k in range(n):
    rows = [np.concatenate([cells[i][k] for i in range(j, min(j + cols, len(cells)))] +
                           [np.zeros_like(cells[0][k])] * (j + cols - min(j + cols, len(cells))), 1)
            for j in range(0, len(cells), cols)]
    w.append_data(np.concatenate(rows, 0))
