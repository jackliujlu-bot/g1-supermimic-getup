"""Measure what happens at every model switch (sim2sim, deploy code).

For each switch (e.g. GETUP->DANCE) records:
  jump      max |joint target change| on the switch tick [rad]
  base      99th percentile of per-tick target change inside the same run (normal motion) [rad]
  tilt      max torso tilt in the 1 s after the switch [deg]
  fell      a fall was detected within 2 s after the switch (only for switches between upright modes)
Usage: python switch_test.py --getup v3.onnx --dance a.onnx b.onnx ... [--human S T C] [--trials 10]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402
from g1_stack import DancePolicy, projected_gravity  # noqa: E402


def tilt_deg(s):
  return float(np.degrees(np.arccos(np.clip(-projected_gravity(s.quat)[2], -1, 1))))


def run(getup, dance_path, seed, human, kind="standing"):
  rng = np.random.default_rng(seed)
  sim = S.Sim(rng)
  qpos, settle = S.initial_pose(kind, rng)
  sim.set_state(qpos)
  if settle:
    sim.settle_passive(1.5)
  dances = {"d": DancePolicy(dance_path)} if dance_path else {}
  sup = S.make_supervisor(getup, dances)
  tg, modes, tilts = [], [], []
  started = False
  n = int(((dances["d"].num_frames * 0.02) if dances else 0) / 0.02) + int(16 / 0.02)
  for k in range(n):
    s = sim.state()
    if dances and not started and sup.request_dance("d", s):
      started = True
    tgt = sup.step(s)
    sim.apply(tgt)
    tg.append(tgt.copy())
    modes.append(sup.mode)
    tilts.append(tilt_deg(sim.state()))
  tg = np.array(tg)
  dq = np.abs(np.diff(tg, axis=0)).max(1)
  base = float(np.percentile(dq, 99))
  out = []
  for k in range(1, len(modes)):
    if modes[k] != modes[k - 1]:
      w = tilts[k:k + 50]
      upright_before = tilts[k - 1] < 30
      fell = upright_before and max(tilts[k:k + 100]) > 50
      out.append(dict(switch=f"{modes[k - 1]}->{modes[k]}", t=round(k * 0.02, 2), jump=round(float(dq[k - 1]), 3),
                      base=round(base, 3), tilt_before=round(tilts[k - 1], 1), tilt_max_1s=round(max(w), 1),
                      fell=bool(fell)))
  return out


if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--getup", required=True)
  ap.add_argument("--dance", nargs="*", default=[])
  ap.add_argument("--human", nargs="+", default=None)
  ap.add_argument("--human-tilt", type=float, default=180.0)
  ap.add_argument("--kind", default="standing")
  ap.add_argument("--trials", type=int, default=10)
  a = ap.parse_args()
  S.HUMAN, S.HUMAN_TILT = a.human, a.human_tilt
  for dp in (a.dance or [""]):
    for t in range(a.trials):
      for r in run(a.getup, dp, 4000 + t, a.human, a.kind):
        r["dance"] = os.path.basename(dp)
        r["kind"] = a.kind
        print(json.dumps(r), flush=True)
