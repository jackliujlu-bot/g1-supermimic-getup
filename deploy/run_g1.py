"""Run the dance tracker + auto get-up on a Unitree G1 (29 DoF), or dry-run it in MuJoCo.

  Real robot:  python run_g1.py --backend real --iface enp3s0 --getup getup.onnx --dance a.onnx b.onnx
  Dry run:     python run_g1.py --backend sim --getup getup.onnx --dance a.onnx --sim-start supine

Keys (terminal, press then Enter):   1..9 = start dance N   s = stop dance (stand)
                                      p = pause in place (hold current pose, stiff)
                                      q / x = DAMPING (soft e-stop, motors go limp-damped)
Remote (real only): R1 = next dance, L1 = stop dance, L2+B = DAMPING.

Safety layer (always on):
  * starts in DAMPING; 'g' (or remote START) engages the policy with a 1.0 s gain ramp
  * stale state (> 0.1 s) or loop overrun x5 in a row -> DAMPING
  * if GETUP has not reached a stable stand within --getup-timeout s -> DAMPING
    (stops endless attempts that could damage the robot); press 'g' to retry
  * joint targets clipped to the URDF limits minus 0.02 rad
"""

from __future__ import annotations

import argparse
import os
import queue
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from g1_stack import (DEFAULT_Q, KD, KP, NJ, DancePolicy, GetupPolicy, HumanGetup, RobotState, Supervisor,  # noqa: E402
                      projected_gravity)

DT = 0.02
# G1 29-DoF position limits from the robot model (g1_joint_params.json), minus a margin
from g1_stack import P as _P  # noqa: E402

# Sanity clip only. The policies were trained WITHOUT clipping their PD targets to the joint range and
# rely on targets somewhat beyond it (e.g. ankle roll/pitch for balance; the joint itself still stops at
# its limit and the torque stays within the motor limit). Clipping at the range removed that balance
# torque and made standing shaky, so only absurd targets are clipped here.
Q_LO = np.array(_P["q_lo"]) - 1.6
Q_HI = np.array(_P["q_hi"]) + 1.6
DAMP_KD = 3.0


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class SimBackend:
  """MuJoCo stand-in for the robot (same interface as RealBackend)."""

  def __init__(self, start: str, seed: int = 0):
    import sim2sim as S

    self.S = S
    rng = np.random.default_rng(seed)
    self.sim = S.Sim(rng, randomize=True)
    q, settle = S.initial_pose(start, rng)
    self.sim.set_state(q)
    if settle:
      self.sim.settle_passive(1.5)
    self.cmd = ("damp", None, None, None)

  def read(self):
    return self.sim.state(), time.time(), 0

  def write(self, q_tgt, kp, kd):
    q = self.sim.d.qpos[7:]
    qd = self.sim.d.qvel[6:]
    for _ in range(4):
      q = self.sim.d.qpos[7:]
      qd = self.sim.d.qvel[6:]
      tau = self.sim.clip_torque(kp * (q_tgt - q) - kd * qd, qd)
      self.sim.d.ctrl[:] = tau[self.sim.act_j]
      self.S.mujoco.mj_step(self.sim.m, self.sim.d)

  def remote_keys(self):
    return 0


class RealBackend:
  """unitree_sdk2py low-level interface (rt/lowstate, rt/lowcmd)."""

  def __init__(self, iface: str):
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    ChannelFactoryInitialize(0, iface)
    # Takeover sequence validated on this robot by the earlier soup7 deployment:
    #   native "ai" service -> request its damping -> release it -> lowcmd channel silent -> our damping frames
    import unitree_sdk2py.g1.loco.g1_loco_client as _lc

    _lc.LOCO_SERVICE_NAME = "loco"  # current G1 firmware (C++ SDK uses "loco"; older python SDK says "sport")
    LocoClient = _lc.LocoClient

    self.msc = MotionSwitcherClient()
    self.msc.SetTimeout(5.0)
    self.msc.Init()
    self.loco = LocoClient()
    self.loco.SetTimeout(3.0)
    self.loco.Init()
    self.released_ai = False
    _, res = self.msc.CheckMode()
    name = (res or {}).get("name", "")
    print(f"[real] native service before takeover: '{name}'")
    if name:
      if name != "ai":
        raise RuntimeError(f"unexpected native service '{name}', refusing to take over")
      ok = False
      for _ in range(10):
        self.loco.Damp()
        time.sleep(0.2)
        code, fsm = self.loco.GetFsmId()
        print(f"[real] native damp request -> fsm query code={code} fsm={fsm}")
        if code == 0 and fsm == 1:
          ok = True
          break
      if not ok:
        # releasing the native controller makes the motors passive: harmless only if the robot is lying
        tilt = self._tilt_now(LowState_, ChannelSubscriber)
        print(f"[real] native damping not confirmed; body tilt {tilt:.0f} deg")
        if tilt < 60:
          raise RuntimeError("native damping not confirmed and robot is not lying down: refusing to release")
      print("[real] native service in damping, releasing it")
      self.msc.ReleaseMode()
      for _ in range(30):
        time.sleep(0.1)
        _, res = self.msc.CheckMode()
        if not (res or {}).get("name"):
          break
      else:
        raise RuntimeError("native service did not release")
      self.released_ai = True
    # nobody else may be publishing motor commands
    seen = []
    probe = ChannelSubscriber("rt/lowcmd", LowCmd_)
    probe.Init(lambda m: seen.append(1), 10)
    time.sleep(1.0)
    probe.Close()
    if seen:
      raise RuntimeError(f"rt/lowcmd not silent ({len(seen)} msgs in 1 s): another controller is running")
    self.crc = CRC()
    self.cmd = unitree_hg_msg_dds__LowCmd_()
    self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    self.pub.Init()
    self.state = None
    self.stamp = 0.0
    self.sub = ChannelSubscriber("rt/lowstate", LowState_)
    self.sub.Init(self._on_state, 10)
    t0 = time.time()
    while self.state is None:
      if time.time() - t0 > 5:
        raise RuntimeError("no rt/lowstate received - check network interface / robot power")
      time.sleep(0.01)
    self.mode_machine = self.state.mode_machine

  @staticmethod
  def _tilt_now(LowState_, ChannelSubscriber):
    got = []
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(lambda m: got.append(m), 10)
    t0 = time.time()
    while not got and time.time() - t0 < 2:
      time.sleep(0.01)
    sub.Close()
    if not got:
      raise RuntimeError("no rt/lowstate")
    q = np.array(got[-1].imu_state.quaternion, dtype=np.float64)
    g = projected_gravity(q / np.linalg.norm(q))
    return float(np.degrees(np.arccos(np.clip(-g[2], -1, 1))))

  def _on_state(self, msg):
    self.state = msg
    self.stamp = time.time()

  def read(self):
    s = self.state
    q = np.array([s.motor_state[i].q for i in range(NJ)])
    qd = np.array([s.motor_state[i].dq for i in range(NJ)])
    quat = np.array(s.imu_state.quaternion, dtype=np.float64)  # w x y z
    gyro = np.array(s.imu_state.gyroscope, dtype=np.float64)
    return RobotState(quat=quat / np.linalg.norm(quat), gyro=gyro, q=q, qd=qd), self.stamp, 0

  def write(self, q_tgt, kp, kd):
    c = self.cmd
    c.mode_pr = 0
    c.mode_machine = self.mode_machine
    for i in range(NJ):
      mc = c.motor_cmd[i]
      mc.mode = 1
      mc.q = float(q_tgt[i])
      mc.dq = 0.0
      mc.kp = float(kp[i])
      mc.kd = float(kd[i])
      mc.tau = 0.0
    c.crc = self.crc.Crc(c)
    self.pub.Write(c)

  def remote_keys(self) -> int:
    raw = bytes(self.state.wireless_remote)
    if len(raw) < 4 or raw[0] != 0x55 or raw[1] != 0x51:  # remote frame not valid
      return 0
    return struct.unpack("<H", raw[2:4])[0]

  def shutdown(self, q_now):
    """0.5 s of damping frames, then hand the robot back to the native service (in damping)."""
    for _ in range(25):
      self.write(q_now, np.zeros(NJ), np.full(NJ, DAMP_KD))
      time.sleep(0.02)
    if self.released_ai:
      print("[real] restoring native 'ai' service:", self.msc.SelectMode("ai"))
      time.sleep(1.0)
      print("[real] native damp:", self.loco.Damp())


# Unitree wireless remote bit layout (keys uint16)
R1, L1, START, SELECT, R2, L2, F1, F2, A, B, X, Y, UP, RIGHT, DOWN, LEFT = (1 << i for i in range(16))


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------


def stdin_reader(qu: queue.Queue):
  for line in sys.stdin:
    qu.put(line.strip().lower())


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--backend", choices=("real", "sim"), required=True)
  ap.add_argument("--iface", default="eth0")
  ap.add_argument("--getup", default=None, help="v3 PPO get-up ONNX (required without --unified; "
                  "with --unified only used if --v3 is given)")
  ap.add_argument("--v3", action="store_true", help="unified mode: also use the v3 policy (fallback / upright boot)")
  ap.add_argument("--dance", nargs="*", default=[])
  ap.add_argument("--getup-timeout", type=float, default=30.0)
  ap.add_argument("--fallback", choices=("v3", "damp"), default=None,
                  help="what to do when the human-style get-up fails: v3 PPO get-up, or damping "
                       "(default: damp on the real robot, v3 in sim)")
  ap.add_argument("--human-tilt", type=float, default=None,
                  help="use the human-style get-up only within this angle of its lying pose (180 = always)")
  ap.add_argument("--human", nargs="+", metavar="SETTLE TRACK CLIP", default=None,
                  help="human-style get-up routines, triples of: settle policy ONNX, tracker ONNX of the "
                       "get-up clip, the clip .npz (e.g. one triple for supine, one for prone)")
  ap.add_argument("--unified", nargs="+", default=None, metavar="TRACKER_ONNX SETTLE_ONNX GETUP_CLIP",
                  help="RECOMMENDED: one SuperMimic tracker for get-up, standing and dancing: TRACKER_ONNX then "
                       "one or more pairs SETTLE_ONNX GETUP_CLIP (e.g. a face-down and a face-up get-up; the "
                       "pair whose lying pose is closest to the robot's body orientation is used). "
                       "--dance then takes motion clips (.npz) instead of per-dance ONNX files")
  ap.add_argument("--sim-start", default="supine")
  ap.add_argument("--script", default="", help="sim only: scripted keys 't:key,t:key' e.g. '0:g,12:1'")
  ap.add_argument("--duration", type=float, default=0, help="sim only: stop after N s")
  ap.add_argument("--diag", action="store_true", help="print supervisor diagnostics every second")
  ap.add_argument("--log", default="", help="save every RUN tick (q, qd, quat, gyro, target, mode) to this .npz")
  a = ap.parse_args()
  if a.fallback is None:
    a.fallback = "damp" if a.backend == "real" else "v3"
  if a.human_tilt is None:  # unified stack: human-style get-up from every fallen pose by default
    a.human_tilt = 180.0 if a.unified else 45.0

  if a.unified:
    from unified import build_unified, load_clip, unified_pairs
    dances = {str(i + 1): p for i, p in enumerate(a.dance)}
    for k, p in dances.items():
      print(f"[dance {k}] {os.path.basename(p)}  {len(load_clip(p)['q']) * DT:.1f}s")
  else:
    dances = {str(i + 1): DancePolicy(p) for i, p in enumerate(a.dance)}
    for k, p in zip(dances, a.dance):
      print(f"[dance {k}] {os.path.basename(p)}  {dances[k].num_frames * DT:.1f}s")

  def make_sup():
    if a.unified:
      return build_unified(a.getup if a.v3 else None, a.unified[0], unified_pairs(a.unified), dances, dt=DT,
                           human_tilt_deg=a.human_tilt, fallback=a.fallback)
    human = HumanGetup(*a.human, max_tilt_deg=a.human_tilt) if a.human else None
    return Supervisor(GetupPolicy(a.getup), dances, dt=DT, human=human)

  sup = make_sup()
  be = RealBackend(a.iface) if a.backend == "real" else SimBackend(a.sim_start)

  keys: queue.Queue = queue.Queue()
  script = []
  if a.backend == "sim" and a.script:
    script = sorted((float(t), k) for t, k in (x.split(":") for x in a.script.split(",")))
  else:
    threading.Thread(target=stdin_reader, args=(keys,), daemon=True).start()

  print("[run] DAMPING. press 'g'+Enter (or remote START / A) to engage. R2 or L2+B = damping.")
  import signal

  def _term(*_):
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, _term)
  log = {"t": [], "mode": [], "sup": [], "q": [], "qd": [], "quat": [], "gyro": [], "tgt": []}
  s = None
  try:
    _loop(a, be, sup, make_sup, dances, keys, script, log)
  except KeyboardInterrupt:
    print("[run] stop requested")
  finally:
    if a.log:
      np.savez(a.log, **{k: np.array(v) for k, v in log.items()})
      print("[run] log ->", a.log)
    if a.backend == "real":
      s, _, _ = be.read()
      be.shutdown(s.q)
      print("[run] damped and handed back; exit")


def _loop(a, be, sup, make_sup, dances, keys, script, log):
  mode = "DAMP"
  engaged_t = 0.0
  getup_since = None
  overruns = 0
  last_keys = 0
  dance_cycle = 0
  t_start = time.time()
  sim_t = 0.0
  while True:
    tick = time.time()
    now = sim_t if a.backend == "sim" else tick - t_start
    s, stamp, _ = be.read()
    # ---- inputs
    cmds = []
    while not keys.empty():
      cmds.append(keys.get())
    while script and script[0][0] <= now:
      cmds.append(script.pop(0)[1])
    if a.backend == "real":
      rk = be.remote_keys()
      pressed = rk & ~last_keys
      last_keys = rk
      if ((rk & L2) and (rk & B)) or (rk & R2):  # R2 = emergency damping (same as soup7 deploy)
        if mode != "DAMP":
          cmds.append("x")
      if pressed & (START | A):
        cmds.append("g")
      if pressed & R1 and dances:
        dance_cycle = dance_cycle % len(dances) + 1
        cmds.append(str(dance_cycle))
      if pressed & L1:
        cmds.append("s")
    for c in cmds:
      if c in ("q", "x"):
        mode = "DAMP"
        print(f"[{now:6.2f}] DAMPING")
      elif c == "g" and mode == "DAMP":
        mode = "RUN"
        engaged_t = now
        sup = make_sup()
        getup_since = now
        print(f"[{now:6.2f}] ENGAGED (get-up/stand policy)")
      elif c in dances and mode == "RUN":
        ok = sup.request_dance(c, s)
        print(f"[{now:6.2f}] dance {c}: {'started' if ok else 'refused (not standing stably yet)'}")
        if not ok and hasattr(sup, "diag"):
          print("         ", sup.diag(s))
      elif c == "s" and mode == "RUN":
        if sup.stop_dance():
          print(f"[{now:6.2f}] dance stopped -> stand")
    # ---- watchdogs
    if a.backend == "real" and mode == "RUN" and tick - stamp > 0.1:
      print("[safety] stale lowstate -> DAMPING")
      mode = "DAMP"
    # ---- control
    if mode == "RUN":
      n_ev = len(sup.events)
      tgt = sup.step(s)
      if getattr(sup, "wants_damp", False):
        print(f"[{now:6.2f}] human-style get-up failed ({sup.events[-1][1]}) -> DAMPING (fallback=damp)")
        mode = "DAMP"
        tgt = s.q.copy()
      if not np.all(np.isfinite(tgt)):
        print(f"[safety] non-finite policy output -> DAMPING")
        mode = "DAMP"
        tgt = s.q.copy()
      tgt = np.clip(tgt, Q_LO, Q_HI)
      for _, ev in sup.events[n_ev:]:
        if not ev.startswith("ref:"):
          print(f"[{now:6.2f}] {ev}")
      if a.diag and hasattr(sup, "diag") and abs(now - round(now)) < DT / 2:
        print(f"[{now:6.2f}] diag {sup.diag(s)}")
      if sup.recovering():  # getting up (any stage): must reach a stable stand within the timeout
        if sup.stable_ticks > 0:
          getup_since = now
        elif getup_since is not None and now - getup_since > a.getup_timeout:
          print(f"[safety] no stable stand for {a.getup_timeout}s -> DAMPING")
          mode = "DAMP"
      else:
        getup_since = now
      if mode == "RUN":
        ramp = min(1.0, (now - engaged_t) / 1.0)
        be.write(tgt, KP * ramp, KD * max(ramp, 0.3))
      if a.log:
        for k, v in (("t", now), ("mode", 1), ("sup", {"SETTLE": 1, "TRACK": 2, "GETUP": 3, "DANCE": 4}.get(sup.mode, 0)),
                     ("q", s.q), ("qd", s.qd), ("quat", s.quat), ("gyro", s.gyro), ("tgt", tgt)):
          log[k].append(v)
    if mode == "DAMP":
      be.write(s.q, np.zeros(NJ), np.full(NJ, DAMP_KD))
    # ---- timing
    if a.backend == "sim":
      sim_t += DT
      if a.duration and sim_t >= a.duration:
        break
    else:
      el = time.time() - tick
      overruns = overruns + 1 if el > DT else 0
      if overruns >= 5:
        print("[safety] control loop overrun -> DAMPING")
        mode = "DAMP"
        overruns = 0
      time.sleep(max(0.0, DT - el))
  if a.backend == "sim":
    sim = be.sim
    print(f"[sim] end: mode={mode}/{sup.mode} pelvis_z={sim.pelvis_z:.3f} events={sup.events}")
  return sup


if __name__ == "__main__":
  main()
