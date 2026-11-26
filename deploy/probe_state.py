"""READ-ONLY robot check: subscribes to rt/lowstate (never publishes rt/lowcmd), prints body
orientation, joint readings vs limits, motor temperatures, remote keys, and times every policy.

Usage: python probe_state.py --iface eth0 [--unified TRACKER SETTLE CLIP] [--getup getup.onnx]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_stack import JOINT_NAMES, NJ, GetupPolicy, RobotState, projected_gravity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iface", default="eth0")
ap.add_argument("--unified", nargs=3, default=None)
ap.add_argument("--getup", default=None)
ap.add_argument("--seconds", type=float, default=3.0)
a = ap.parse_args()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_  # noqa: E402

P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "g1_joint_params.json")))
lo, hi = np.array(P["q_lo"]), np.array(P["q_hi"])
ChannelFactoryInitialize(0, a.iface)
msgs = []
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(lambda m: msgs.append((time.time(), m)), 10)
t0 = time.time()
while time.time() - t0 < a.seconds:
  time.sleep(0.05)
if not msgs:
  print("NO rt/lowstate received on", a.iface)
  sys.exit(1)
ts = np.array([t for t, _ in msgs])
s = msgs[-1][1]
q = np.array([s.motor_state[i].q for i in range(NJ)])
qd = np.array([s.motor_state[i].dq for i in range(NJ)])
quat = np.array(s.imu_state.quaternion, dtype=np.float64)
gyro = np.array(s.imu_state.gyroscope, dtype=np.float64)
rpy = np.degrees(np.array(s.imu_state.rpy))
g = projected_gravity(quat / np.linalg.norm(quat))
tilt = float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))
temps = [list(s.motor_state[i].temperature) for i in range(NJ)]
print(f"lowstate: {len(msgs)} msgs in {a.seconds:.1f}s ({len(msgs) / a.seconds:.0f} Hz), mode_machine={s.mode_machine}, mode_pr={s.mode_pr}")
print(f"imu quat(wxyz)={np.round(quat, 3)}  rpy(deg)={np.round(rpy, 1)}  tilt from upright={tilt:.0f} deg  projected_gravity={np.round(g, 2)}")
print(f"gyro max {np.abs(gyro).max():.2f} rad/s, joint vel max {np.abs(qd).max():.2f} rad/s")
out = [(JOINT_NAMES[i], round(float(q[i]), 3)) for i in range(NJ) if q[i] < lo[i] - 0.05 or q[i] > hi[i] + 0.05]
print("joints outside limits (>0.05 rad):", out if out else "none")
print("max motor temperature:", max(max(t) for t in temps), "C")
raw = bytes(s.wireless_remote)
print("remote keys raw:", raw[2:4].hex() if len(raw) >= 4 else "n/a")
rs = RobotState(quat=quat / np.linalg.norm(quat), gyro=gyro, q=q, qd=qd)
if a.unified:
  from unified import RefTimeline, SettleRoutine, TrackerPolicy, load_clip

  r = SettleRoutine(a.unified[1], load_clip(a.unified[2]))
  eg, eq = r.err(rs)
  print(f"settle target error: orientation {eg:.2f} (0=match, 4=opposite), joints rms {eq:.2f} rad")
  tp = TrackerPolicy(a.unified[0])
  tl = RefTimeline()
  tl.reset_clip(r.clip)
  tp.start(rs, tl)
  for name, f in (("settle", lambda: r.settle.act(rs)), ("tracker", lambda: tp.act(rs, tl))):
    f()
    t1 = time.time()
    for _ in range(50):
      out = f()
    dt = (time.time() - t1) / 50 * 1000
    print(f"{name} inference: {dt:.1f} ms/step (budget 20 ms), finite={bool(np.all(np.isfinite(out)))}")
if a.getup:
  gp = GetupPolicy(a.getup)
  gp.act(rs)
  t1 = time.time()
  for _ in range(50):
    gp.act(rs)
  print(f"v3 inference: {(time.time() - t1) / 50 * 1000:.1f} ms/step")
print("read-only probe done (no commands were sent)")
