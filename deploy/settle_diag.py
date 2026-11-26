"""Why does SETTLE time out? Logs the settle errors (gravity, joint rms, max qd) through a boot trial.
usage: python settle_diag.py TRACKER SETTLE_P CLIP_P SETTLE_S CLIP_S --scenario prone --seed 0 [--soft U] [--terrain H]"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S

ap = argparse.ArgumentParser()
ap.add_argument("u", nargs=5)
ap.add_argument("--scenario", default="prone")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--soft", type=float, default=None)
ap.add_argument("--terrain", type=float, default=None)
a = ap.parse_args()
S.SOFT, S.TERRAIN, S.HUMAN_TILT = a.soft, a.terrain, 180.0
S.UNIFIED = {"tracker": a.u[0], "routines": [(a.u[1], a.u[2]), (a.u[3], a.u[4])], "dance_clip": None}
rng = np.random.default_rng(a.seed)
S.TERRAIN_SEED = a.seed
sim = S.Sim(rng)
qpos, settle = S.initial_pose(a.scenario, rng)
sim.set_state(qpos)
if settle:
  sim.settle_passive(1.5)
sup = S.make_supervisor(None, {})
best = (9, 9)
for k in range(int(10 / 0.02)):
  s = sim.state()
  sim.apply(sup.step(s))
  if sup.mode == "SETTLE":
    eg, eq = sup.cur.err(s)
    qd, qdt = np.abs(s.qd).max(), np.abs(sim.d.qvel[6:]).max()
    if k % 25 == 0:
      worst = np.argsort(-np.abs(s.q - sup.cur.q_t))[:3]
      print(f"t {k*0.02:5.2f} {sup.cur.clip['name']:12s} eg {eg:.3f} eq {eq:.3f} qd {qd:.2f} qd_true {qdt:.2f} worst joints {worst} {np.round((s.q - sup.cur.q_t)[worst], 2)} z {sim.pelvis_z:.3f}")
  elif k % 25 == 0:
    print(f"t {k*0.02:5.2f} mode {sup.mode}")
print(sup.events)
