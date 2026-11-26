"""Demo grid video of the full deploy stack in sim2sim (unified tracker + settle routines).
usage: MUJOCO_GL=egl python render_demo.py OUT.mp4 TRACKER SETTLE_P CLIP_P SETTLE_S CLIP_S
         --cells LABEL:SCENARIO:FLOOR:SEED ...   (SCENARIO = supine|prone|side|sitting|random|dance=CLIP.npz)
FLOOR: hard | soft | rough | roughsoft. Cells play in real time, padded with their last frame."""
import argparse, os, sys
import mujoco, numpy as np, imageio
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("u", nargs=5)
ap.add_argument("--cells", nargs="+", required=True)
ap.add_argument("--limit", type=float, default=40.0)
ap.add_argument("--cols", type=int, default=3)
a = ap.parse_args()
FL = {"hard": (None, None), "soft": (1.0, None), "rough": (None, 0.015), "roughsoft": (0.6, 0.015)}
H, W = 360, 400
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 20)
S.HUMAN_TILT = 180.0
cells = []
for c in a.cells:
  label, scen, floor, seed = c.split(":")
  seed = int(seed)
  S.SOFT, S.TERRAIN = FL[floor]
  S.STAND_POSES[:] = [S.DEFAULT_Q]
  S.UNIFIED = {"tracker": a.u[0], "routines": [(a.u[1], a.u[2]), (a.u[3], a.u[4])], "dance_clip": None}
  S.TERRAIN_SEED = seed
  fr = []
  if scen.startswith("dance="):
    res = S.run_dance(None, scen[6:], seed, push=True, frames=fr)
  else:
    res = S.run_boot(scen, None, seed, limit_s=a.limit, frames=fr)
  S.TERRAIN_SEED = seed
  m, _, _ = S.build_model(); d = mujoco.MjData(m); r = mujoco.Renderer(m, H, W)
  cam = mujoco.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 3.0, -15, 120
  imgs = []
  for i, q in enumerate(fr):
    d.qpos[:] = q; mujoco.mj_forward(m, d)
    cam.lookat[:] = [q[0], q[1], 0.45]; r.update_scene(d, cam)
    im = Image.fromarray(r.render()); dr = ImageDraw.Draw(im)
    dr.text((8, 6), label, fill=(255, 255, 0), font=font)
    dr.text((8, H - 30), f"{i / 25:4.1f} s", fill=(255, 255, 255), font=font)
    imgs.append(np.array(im))
  cells.append(imgs)
  print(label, res.get("ok"), res.get("t_stand"), res.get("path", "")[-80:], flush=True)
n = max(len(c) for c in cells)
blank = np.zeros_like(cells[0][0])
with imageio.get_writer(a.out, fps=25, quality=7) as w:
  for k in range(n):
    tiles = [c[min(k, len(c) - 1)] for c in cells]
    tiles += [blank] * (-len(tiles) % a.cols)
    rows = [np.concatenate(tiles[j:j + a.cols], 1) for j in range(0, len(tiles), a.cols)]
    w.append_data(np.concatenate(rows, 0))
