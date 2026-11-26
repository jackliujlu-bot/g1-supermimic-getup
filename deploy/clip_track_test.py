"""Can the tracker follow a get-up clip from its own first pose to standing? (no settle stage)

Starts the CPU MuJoCo robot in the clip's first frame (+ joint noise, lowered onto the floor),
then the deploy TrackerPolicy follows [1 s blend from the actual pose -> clip]. UP = upright (>0.95)
with pelvis > 0.65 m for the whole last 1 s of the standing hold; CALM = also every joint < 1.0 rad/s (true joint speed)
over that second (no visible shaking).

Usage: python clip_track_test.py TRACKER.onnx CLIP.npz [--trials 8] [--soft U] [--terrain H] [--noise 0.1]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402
from g1_stack import projected_gravity  # noqa: E402
from test_ref_getup import lift_out_of_floor  # noqa: E402
from unified import RefTimeline, TrackerPolicy, load_clip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tracker")
ap.add_argument("clip")
ap.add_argument("--trials", type=int, default=8)
ap.add_argument("--soft", type=float, default=None)
ap.add_argument("--terrain", type=float, default=None)
ap.add_argument("--noise", type=float, default=0.1)
ap.add_argument("--hold", type=float, default=1.0, help="seconds of standing hold after the clip")
ap.add_argument("--clean", action="store_true", help="no sensor noise / latency")
ap.add_argument("--dump", default=None, help="save the hold-phase true joint angles (npz)")
ap.add_argument("--nolat", action="store_true")
ap.add_argument("--qdnoise", type=float, default=None)
ap.add_argument("--ema", type=float, default=1.0, help="low-pass the joint targets during the hold (1 = off)")
a = ap.parse_args()
S.SOFT, S.TERRAIN = a.soft, a.terrain
S.LATENCY = not a.nolat
if a.qdnoise is not None:
  S.QD_NOISE = a.qdnoise
clip = load_clip(a.clip)
ok = 0
calm_n = 0
jit = []
for seed in range(a.trials):
  rng = np.random.default_rng(seed)
  S.TERRAIN_SEED = seed
  sim = S.Sim(rng)
  if a.clean:
    sim.noise = False
  qp = np.zeros(36)
  qp[3:7] = clip["quat"][0]
  qp[7:] = clip["q"][0] + rng.uniform(-a.noise, a.noise, 29)
  qp[:2] = rng.uniform(-1, 1, 2)
  sim.set_state(qp)
  lift_out_of_floor(sim)
  sim.settle_passive(1.0, kp=20, kd=2, target=qp[7:].copy())
  s = sim.state()
  tl = RefTimeline()
  tl.reset_pose(s.q.astype(float), s.quat.astype(float), "blend")
  tl.append_blend(clip["q"][0], clip["quat"][0], 1.0)
  tl.append_clip(clip)
  tp = TrackerPolicy(a.tracker)
  tp.start(s, tl)
  n = len(tl.q) + int(a.hold * 50)
  qd_tail = []
  q_hold = []
  held, up_held, up_max, qdmax = 0, 0, 0.0, 0.0
  for k in range(n):
    s = sim.state()
    tgt = tp.act(s, tl)
    if k >= len(tl.q) and a.ema < 1.0:
      tgt = a.ema * tgt + (1 - a.ema) * prev
    prev = tgt
    sim.apply(tgt)
    tl.advance()
    up = -projected_gravity(s.quat)[2]
    up_max = max(up_max, up)
    qdmax = max(qdmax, float(np.abs(s.qd).max()))
    qd_true = np.abs(sim.d.qvel[6:])  # true joint speed (s.qd carries simulated sensor noise)
    if k >= len(tl.q):
      q_hold.append(sim.d.qpos[7:].copy())
    if k >= n - 50:
      qd_tail.append(qd_true)
    is_up = up > 0.95 and sim.pelvis_z > 0.65
    up_held = up_held + 1 if is_up else 0
    held = held + 1 if (is_up and qd_true.max() < 1.0) else 0
  stood = up_held >= 50
  calm = held >= 50
  ok += stood
  if a.dump:
    np.savez(f"{a.dump}_{seed}.npz", q=np.array(q_hold))
  calm_n += calm
  if stood:
    jit.append(float(np.sqrt(np.mean(np.square(qd_tail)))))
  print(f"seed {seed}: {'CALM' if calm else ('UP' if stood else 'FELL')}  max uprightness {up_max:.2f}  final pelvis z {sim.pelvis_z:.2f}  peak qd {qdmax:.1f}  last-1s max qd {np.max(qd_tail):.2f} (joint {np.argmax(np.max(qd_tail, 0))})  mean {np.mean(qd_tail):.2f}", flush=True)
print(f"{os.path.basename(a.clip)} soft={a.soft} terrain={a.terrain}: up {ok}/{a.trials} calm {calm_n}/{a.trials} jitter_rms {np.mean(jit):.3f}")
