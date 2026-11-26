"""Grid video: the human-style get-up from many fallen poses (plus a fall during a dance).

Usage: MUJOCO_GL=egl python render_grid.py GETUP.onnx out.mp4 --human SETTLE TRACK CLIP
         [--dance dance.onnx] [--tilt 180] [--cases supine:1000,prone:1000,...]
"""
import argparse
import os
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402

NAMES = {"supine": "仰卧", "prone": "俯卧", "side": "侧卧", "sitting": "坐地", "random": "随机摔落",
         "dance": "跳舞时被推倒"}
STAGE = {"SETTLE": "归位", "HTRACK": "照真人起身", "GETUP": "v3 兜底", "DANCE": "跳舞",
         "getup": "照真人起身", "stand": "站立", "blend": "过渡", "dance": "跳舞"}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
H = W = 400

ap = argparse.ArgumentParser()
ap.add_argument("getup")
ap.add_argument("out")
ap.add_argument("--human", nargs="+", default=None)
ap.add_argument("--unified", nargs=3, default=None, metavar=("TRACKER", "SETTLE", "GETUP_CLIP"))
ap.add_argument("--dance", default="")
ap.add_argument("--tilt", type=float, default=180.0)
ap.add_argument("--cases", default="supine:1001,prone:1001,side:1001,sitting:1002,random:1003,random:1005")
ap.add_argument("--limit", type=float, default=16.0)
a = ap.parse_args()
S.HUMAN, S.HUMAN_TILT = a.human, a.tilt
if a.unified:
  S.UNIFIED = {"tracker": a.unified[0], "routines": [(a.unified[1], a.unified[2])], "dance_clip": None}

import imageio  # noqa: E402

m, _, _ = S.build_model()
d = mujoco.MjData(m)
ren = mujoco.Renderer(m, H, W)
cam = mujoco.MjvCamera()
cam.distance, cam.elevation, cam.azimuth = 3.6, -40, 135
font = ImageFont.truetype(FONT, 20)
small = ImageFont.truetype(FONT, 17)

# capture frames + the supervisor mode for every frame by wrapping Supervisor.step
from g1_stack import Supervisor  # noqa: E402

_modes = []
_orig = Supervisor.step


def _step(self, s):
  out = _orig(self, s)
  m = self.mode
  if m == "TRACK":
    m = self.tl.label[max(self.tl.t - 1, 0)].split(":")[0].split("_")[0]
  elif m == "GETUP" and not hasattr(self, "tl"):
    m = "GETUP_OLD"
  _modes.append(m)
  return out


Supervisor.step = _step
try:
  import unified as _U

  _orig_u = _U.UnifiedSupervisor.step

  def _step_u(self, s):
    out = _orig_u(self, s)
    m = self.mode
    if m == "TRACK":
      m = self.tl.label[self.tl.t].split(":")[0].split("_")[0]
    _modes.append(m)
    return out

  _U.UnifiedSupervisor.step = _step_u
except ImportError:
  pass

clips = []
for case in a.cases.split(","):
  kind, seed = case.split(":")
  fr = []
  _modes.clear()
  r = S.run_boot(kind, a.getup, seed=int(seed), limit_s=a.limit, frames=fr)
  modes = _modes[::2][:len(fr)]
  ok = "站起来了 %.1f s" % r["t_stand"] if r["ok"] else "没站起来"
  human = "HTRACK" in r["path"] or "TRACK:getup" in r["path"]
  print(case, r["ok"], r["t_stand"], r["path"], flush=True)
  clips.append((fr, modes, f"{NAMES[kind]}", ok + ("" if human else "（v3 接手）")))
if a.dance:
  fr = []
  _modes.clear()
  r = S.run_dance(a.getup, a.dance, seed=7, push=True, frames=fr)
  _modes[:] = _modes[:]  # one mode per control tick
  modes = _modes[::2][:len(fr)]
  print("dance", r["ok"], r["path"], flush=True)
  ok = "站起来了" if r["ok"] else "没站起来"
  hum = "HTRACK" in r["path"] or "TRACK:getup" in r["path"]
  clips.append((fr, modes, NAMES["dance"], ok + ("" if hum else "（v3 接手）")))


def tile(fr, modes, i, title, result):
  j = min(i, len(fr) - 1)
  q = fr[j]
  d.qpos[:] = q
  mujoco.mj_forward(m, d)
  cam.lookat[:] = [fr[0][0], fr[0][1], 0.35] if "跳舞" not in title else [q[0], q[1], 0.5]
  ren.update_scene(d, cam)
  im = Image.fromarray(ren.render())
  g = ImageDraw.Draw(im)
  g.rectangle([0, 0, W, 30], fill=(0, 0, 0))
  g.text((8, 3), title, font=font, fill=(255, 255, 255))
  st = STAGE.get(modes[min(j, len(modes) - 1)] if modes else "", "")
  g.text((W - 8 - 20 * len(st), 4), st, font=small, fill=(255, 220, 120))
  g.text((8, H - 26), f"{j / 25:4.1f} s", font=small, fill=(20, 20, 20))
  if i >= len(fr) - 1:
    g.rectangle([0, H - 60, W, H - 30], fill=(30, 110, 60) if "站起来" in result else (150, 40, 40))
    g.text((8, H - 58), result, font=small, fill=(255, 255, 255))
  return np.asarray(im)


cols = 3
n = max(len(c[0]) for c in clips) + 40
with imageio.get_writer(a.out, fps=25, macro_block_size=2) as w:
  for i in range(n):
    tiles = [tile(fr, md, i, t, res) for fr, md, t, res in clips]
    while len(tiles) % cols:
      tiles.append(np.full_like(tiles[0], 255))
    rows = []
    for k in range(0, len(tiles), cols):
      row = []
      for jj, t in enumerate(tiles[k:k + cols]):
        if jj:
          row.append(np.full((H, 6, 3), 255, np.uint8))
        row.append(t)
      rows.append(np.concatenate(row, 1))
    out = []
    for k, r_ in enumerate(rows):
      if k:
        out.append(np.full((6, r_.shape[1], 3), 255, np.uint8))
      out.append(r_)
    w.append_data(np.concatenate(out, 0))
print("video ->", a.out)
