"""Sim check from REAL robot poses: start the unified stack (fallback=damp) from a logged real
quat+q and report how it went. Usage: python real_replay_test.py POSE.npy [POSE.npy ...] --trials 10"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402
from unified import build_unified  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("poses", nargs="+")
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--fallback", default="damp")
ap.add_argument("--tracker", default="../final/policies/tracker.onnx")
ap.add_argument("--soft", type=float, default=None)
ap.add_argument("--settle", default="../exports/settle_final/settle.onnx")
a = ap.parse_args()
U = (a.tracker, a.settle, "motions/getup_supine.npz")
S.SOFT = a.soft
for p in a.poses:
  x = np.load(p)
  res = []
  for seed in range(a.trials):
    sim = S.Sim(np.random.default_rng(seed))
    qp = np.zeros(36); qp[3:7] = x[:4]; qp[7:] = x[4:]; qp[2] = 0.15
    sim.set_state(qp); sim.settle_passive(1.0, kp=20, kd=2, target=x[4:].copy())
    sup = build_unified(None, U[0], [(U[1], U[2])], {}, fallback=a.fallback)
    S.STAND_POSES.append(sup.stand_q.astype(np.float32))
    held, t_stand, qdmax = 0, None, 0.0
    for k in range(int(25 / 0.02)):
      s = sim.state()
      tgt = sup.step(s)
      if sup.wants_damp:
        break
      sim.apply(tgt)
      qdmax = max(qdmax, float(np.abs(s.qd).max()))
      held = held + 1 if S.stable(sim, s) else 0
      if held >= 100:
        t_stand = round((k - 99) * 0.02, 2)
        break
    res.append((t_stand, [e for _, e in sup.events if "->" in e], round(qdmax, 1)))
  ok = sum(r[0] is not None for r in res)
  print(os.path.basename(p), f"stood {ok}/{len(res)}", "t", [r[0] for r in res])
  for r in res:
    if r[0] is None:
      print("   fail:", r[1], "max qd", r[2])
