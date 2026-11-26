"""Real robot vs simulation, from the same state.

Takes a real run log (run_g1.py --log), finds where the get-up tracking started, puts the CPU
MuJoCo robot in exactly that state (joint angles + body orientation) and runs the same tracker on
the same reference. Saves both trajectories (q, qd, policy target, orientation) for comparison.

Usage: python real_vs_sim.py LIVE.npz TRACKER.onnx OUT.npz [--soft U] [--seeds 3]
"""
import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S  # noqa: E402
from g1_stack import projected_gravity  # noqa: E402
from unified import RefTimeline, TrackerPolicy, load_clip  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("live")
ap.add_argument("tracker")
ap.add_argument("out")
ap.add_argument("--soft", type=float, default=None)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--mass", type=float, default=0.0, help="extra torso payload [kg]")
ap.add_argument("--clip", default="motions/getup_supine.npz")
a = ap.parse_args()

z = np.load(a.live)
sup, q, qd, tgt, quat = z["sup"], z["q"], z["qd"], z["tgt"], z["quat"]
k0 = int(np.argmax(sup == 2))
n = int((sup[k0:] == 2).sum())
clip = load_clip(a.clip)
quat0 = quat[k0] / np.linalg.norm(quat[k0])


def timeline(q0, r0):
  tl = RefTimeline()
  tl.reset_pose(q0, r0, "b")
  tl.append_blend(clip["q"][0], clip["quat"][0], 1.0)
  tl.append_clip(clip)
  return tl


ref = timeline(q[k0], quat0)
out = {"real_q": q[k0:k0 + n], "real_qd": qd[k0:k0 + n], "real_tgt": tgt[k0:k0 + n],
       "real_grav": np.array([projected_gravity(x / np.linalg.norm(x)) for x in quat[k0:k0 + n]]),
       "ref_q": ref.q[:n + 150], "ref_quat": ref.quat[:n + 150]}
S.SOFT = a.soft
S.EXTRA_MASS = a.mass
sims = []
for seed in range(a.seeds):
  sim = S.Sim(np.random.default_rng(seed))
  qp = np.zeros(36)
  qp[3:7] = quat0
  qp[7:] = q[k0]
  qp[2] = 0.3
  sim.set_state(qp)
  # lower the body until it rests on the floor in that pose (hold the joints in place)
  sim.settle_passive(1.0, kp=60, kd=3, target=q[k0].copy())
  s = sim.state()
  tl = timeline(s.q.astype(float), s.quat.astype(float))
  tp = TrackerPolicy(a.tracker)
  tp.start(s, tl)
  rec = {k: [] for k in ("q", "qd", "tgt", "grav", "qpos", "tau")}
  for k in range(n + 150):
    s = sim.state()
    t_ = tp.act(s, tl)
    tl.advance()
    rec["q"].append(s.q.copy()); rec["qd"].append(s.qd.copy()); rec["tgt"].append(t_.copy())
    rec["grav"].append(projected_gravity(s.quat)); rec["qpos"].append(sim.d.qpos.copy())
    sim.apply(t_)
    rec["tau"].append(sim.d.actuator_force[np.argsort(sim.act_j)].copy())
  sims.append({k: np.array(v) for k, v in rec.items()})
for k in sims[0]:
  out[f"sim_{k}"] = np.stack([s_[k] for s_ in sims])
np.savez(a.out, **out)
up = -out["sim_grav"][:, :, 2]
print(f"real track ticks {n}; sim final uprightness per seed: {np.round(up[:, -1], 2)}")
