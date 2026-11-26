"""Standing-hold test: boot standing, stand for N s (optionally with random shoves), then dance once.

Reports falls, pelvis drift, heading drift, and the joint-target jump at every switch of network
or reference segment (blend / dance / stand), vs the 99th percentile of normal per-tick change.

Usage: python hold_test.py --getup v3.onnx [--unified TRACKER SETTLE CLIP --dance clip.npz]
                           [--old-dance dance.onnx] [--seconds 60] [--push 300] [--trials 5]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402
from g1_stack import DancePolicy, projected_gravity, yaw_of  # noqa: E402


def tilt(q):
  return float(np.degrees(np.arccos(np.clip(-projected_gravity(q)[2], -1, 1))))


def run(a, seed):
  rng = np.random.default_rng(seed)
  sim = S.Sim(rng)
  qpos, _ = S.initial_pose("standing", rng)
  sim.set_state(qpos)
  if a.unified:
    S.UNIFIED = {"tracker": a.unified[0], "routines": [(a.unified[1], a.unified[2])], "dance_clip": a.dance}
    sup = S.make_supervisor(a.getup, {"d": None} if a.dance else {})
  else:
    sup = S.make_supervisor(a.getup, {"d": DancePolicy(a.old_dance)} if a.old_dance else {})
  n_hold = int(a.seconds / 0.02)
  pushes, next_push = 0, int(rng.uniform(4, 6) / 0.02) + 150
  tg, seg, tilts, pel = [], [], [], []
  fell_at = None
  danced = False
  k_end = n_hold + 150 + (int(60 / 0.02) if (a.dance or a.old_dance) else 0)
  yaw0 = None
  for k in range(k_end):
    s = sim.state()
    f = None
    if a.push and k >= next_push and k < n_hold:
      if k == next_push:
        ang = rng.uniform(0, 2 * np.pi)
        pushes += 1
      f = np.array([np.cos(ang), np.sin(ang), 0]) * a.push  # noqa: F821
      if k >= next_push + 5:  # 0.1 s shove
        next_push = k + int(rng.uniform(4, 6) / 0.02)
    if k >= n_hold and not danced and (a.dance or a.old_dance):
      danced = sup.request_dance("d", s)
    tgt = sup.step(s)
    sim.apply(tgt, push=f)
    tg.append(tgt.copy())
    lab = sup.mode + (":" + sup.tl.label[max(sup.tl.t - 1, 0)].split(":")[0] if sup.mode == "TRACK" else "")
    seg.append(lab)
    tilts.append(tilt(sim.state().quat))
    pel.append(sim.d.xpos[sim.pid, :2].copy())
    if yaw0 is None and k == 200:
      yaw0 = yaw_of(sim.state().quat)
    if fell_at is None and tilts[-1] > 50:
      fell_at = round(k * 0.02, 2)
    if danced and getattr(sup, "idle", sup.mode == "GETUP") and k > n_hold + 100:
      break
  tg = np.array(tg)
  dq = np.abs(np.diff(tg, axis=0)).max(1)
  base = float(np.percentile(dq, 99))
  sw = []
  for k in range(1, len(seg)):
    if seg[k] != seg[k - 1]:
      sw.append({"switch": f"{seg[k - 1]}->{seg[k]}", "jump": round(float(dq[k - 1]), 3),
                 "tilt_1s": round(max(tilts[k:k + 50]), 1)})
  hold = slice(250, min(n_hold, len(pel)))
  drift = float(np.linalg.norm(np.array(pel[hold.stop - 1]) - np.array(pel[hold.start]))) if hold.stop > 250 else 0.0
  return dict(seed=seed, fell_at=fell_at, pushes=pushes, danced=danced, base_p99=round(base, 3),
              drift_m=round(drift, 3), yaw_drift_deg=round(float(np.degrees(yaw_of(sim.state().quat) - (yaw0 or 0))), 1),
              switches=sw)


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--getup", required=True)
  ap.add_argument("--unified", nargs=3, default=None)
  ap.add_argument("--dance", default="")
  ap.add_argument("--old-dance", default="")
  ap.add_argument("--seconds", type=float, default=60)
  ap.add_argument("--push", type=float, default=0)
  ap.add_argument("--trials", type=int, default=5)
  a = ap.parse_args()
  S.HUMAN_TILT = 180.0
  for t in range(a.trials):
    print(json.dumps(run(a, 5000 + t)), flush=True)
