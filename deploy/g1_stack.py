"""Shared G1 control stack: SuperMimic dance tracker + get-up policy + supervisor.

Used unchanged by sim2sim (MuJoCo CPU) and by the real robot (unitree_sdk2py).
Everything runs at 50 Hz on numpy + onnxruntime. The low-level PD loop
(tau = kp (q* - q) - kd qd) runs on the motor drivers (real) or in the simulator.

Inputs per control tick (`RobotState`): pelvis IMU quaternion (wxyz, world), pelvis
gyro (body frame, rad/s), joint positions and velocities in mjlab / Unitree-SDK order.
Output: joint position targets q* (29) + kp, kd arrays.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "g1_joint_params.json")))
JOINT_NAMES: list[str] = P["joint_names"]
NJ = len(JOINT_NAMES)
KP = np.array(P["kp"], dtype=np.float32)
KD = np.array(P["kd"], dtype=np.float32)
ACTION_SCALE = np.array(P["action_scale"], dtype=np.float32)
DEFAULT_Q = np.array(P["default_q"], dtype=np.float32)


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------


def quat_conj(q):
  return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])


def quat_rotate_inv(q, v):
  qv = np.concatenate([[0.0], v])
  return quat_mul(quat_mul(quat_conj(q), qv), q)[1:]


def projected_gravity(q):
  return quat_rotate_inv(q, np.array([0.0, 0.0, -1.0])).astype(np.float32)


def yaw_of(q):
  w, x, y, z = q
  return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
  return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


# ---------------------------------------------------------------------------
# observation history (mjlab layout: term-major, oldest -> newest)
# ---------------------------------------------------------------------------


class TermHistory:
  TERMS = ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel", "actions")

  def __init__(self, history: int):
    self.h = history
    self.buf: dict[str, np.ndarray] | None = None

  def reset(self):
    self.buf = None

  def push(self, terms: dict[str, np.ndarray]) -> np.ndarray:
    if self.buf is None:  # mjlab fills an uninitialised buffer with the first frame
      self.buf = {k: np.repeat(terms[k][None], self.h, axis=0) for k in self.TERMS}
    else:
      for k in self.TERMS:
        self.buf[k] = np.concatenate([self.buf[k][1:], terms[k][None]], axis=0)
    return np.concatenate([self.buf[k].reshape(-1) for k in self.TERMS]).astype(np.float32)


@dataclass
class RobotState:
  quat: np.ndarray  # pelvis wxyz
  gyro: np.ndarray  # pelvis body frame
  q: np.ndarray
  qd: np.ndarray


def sensor_terms(s: RobotState, last_action: np.ndarray) -> dict[str, np.ndarray]:
  return {
    "projected_gravity": projected_gravity(s.quat),
    "base_ang_vel": s.gyro.astype(np.float32),
    "joint_pos": (s.q - DEFAULT_Q).astype(np.float32),
    "joint_vel": s.qd.astype(np.float32),
    "actions": last_action.astype(np.float32),
  }


def _session(path):
  so = ort.SessionOptions()
  so.intra_op_num_threads = 2
  return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


# ---------------------------------------------------------------------------
# policies
# ---------------------------------------------------------------------------


class GetupPolicy:
  """PPO get-up / stand policy (mjlab rsl_rl ONNX export: input 'obs')."""

  def __init__(self, onnx_path: str, history: int = 5):
    self.sess = _session(onnx_path)
    self.inp = self.sess.get_inputs()[0].name
    self.hist = TermHistory(history)
    self.last_action = np.zeros(NJ, np.float32)

  def reset(self, last_action: np.ndarray | None = None):
    self.hist.reset()
    self.last_action = np.zeros(NJ, np.float32) if last_action is None else last_action.astype(np.float32)

  def act(self, s: RobotState) -> np.ndarray:
    obs = self.hist.push(sensor_terms(s, self.last_action))
    a = self.sess.run(None, {self.inp: obs[None]})[0][0]
    self.last_action = a.astype(np.float32)
    return DEFAULT_Q + ACTION_SCALE * a


class DancePolicy:
  """SuperMimic tracker for one clip (tools/export_dance.py ONNX:
  inputs obs[1,H*D], anchor_quat[1,4], time_step[1,1] -> actions[1,29], ref_joint_pos[1,29])."""

  def __init__(self, onnx_path: str):
    self.sess = _session(onnx_path)
    meta = self.sess.get_modelmeta().custom_metadata_map
    self.history = int(meta.get("history_length", 10))
    self.num_frames = int(meta["num_frames"])
    self.ref_yaw0 = float(meta["ref_yaw0"])
    self.hist = TermHistory(self.history)
    self.last_action = np.zeros(NJ, np.float32)
    self.t = 0
    self.yaw_off = np.array([1.0, 0, 0, 0])

  def start(self, s: RobotState, last_action: np.ndarray | None = None):
    """Align the clip's start heading with the robot's current heading."""
    self.t = 0
    self.hist.reset()
    self.last_action = np.zeros(NJ, np.float32) if last_action is None else last_action.astype(np.float32)
    # robot quat expressed in the motion's world frame: q_m = yaw(ref0 - robot0) * q_robot
    self.yaw_off = yaw_quat(self.ref_yaw0 - yaw_of(s.quat))

  @property
  def done(self) -> bool:
    return self.t >= self.num_frames - 1

  def act(self, s: RobotState) -> np.ndarray:
    obs = self.hist.push(sensor_terms(s, self.last_action))
    qa = quat_mul(self.yaw_off, s.quat).astype(np.float32)
    out = self.sess.run(None, {"obs": obs[None], "anchor_quat": qa[None],
                               "time_step": np.array([[self.t]], np.float32)})
    a = out[0][0]
    self.ref_q = out[1][0]
    self.last_action = a.astype(np.float32)
    self.t = min(self.t + 1, self.num_frames - 1)
    return DEFAULT_Q + ACTION_SCALE * a


# ---------------------------------------------------------------------------
# supervisor
# ---------------------------------------------------------------------------


class HumanRoutine:
  """One human get-up: settle policy -> lying start pose of a human clip -> tracker plays the clip."""

  def __init__(self, settle_onnx: str, track_onnx: str, target_npz: str):
    self.settle = GetupPolicy(settle_onnx)
    self.track = DancePolicy(track_onnx)
    z = np.load(target_npz)
    self.q_t = z["joint_pos"][0].astype(np.float64)
    self.g_t = quat_rotate_inv(z["root_quat_w"][0], np.array([0.0, 0.0, -1.0]))
    self.name = os.path.basename(track_onnx)

  def settle_err(self, s: RobotState):
    eg = float(np.sum((projected_gravity(s.quat) - self.g_t) ** 2))
    eq = float(np.sqrt(np.mean((s.q - self.q_t) ** 2)))
    return eg, eq

  def is_settled(self, s: RobotState) -> bool:
    eg, eq = self.settle_err(s)
    return eg < 0.05 and eq < 0.15 and np.abs(s.qd).max() < 1.0


class HumanGetup:
  """Set of human get-up routines (e.g. from supine and from prone). On entry the routine whose
  lying orientation is closest to the robot's current body orientation is chosen, so the robot
  never has to roll over more than ~90 degrees before getting up."""

  def __init__(self, *args, max_tilt_deg: float = 45.0):
    assert len(args) % 3 == 0 and args, "HumanGetup(settle, track, clip, [settle, track, clip, ...])"
    self.routines = [HumanRoutine(*args[i:i + 3]) for i in range(0, len(args), 3)]
    self.cur = self.routines[0]
    # only used when the body already lies within this angle of a routine's lying orientation;
    # other poses (prone, kneeling, ...) are handled by the PPO get-up policy
    self.max_eg = 2.0 - 2.0 * np.cos(np.radians(max_tilt_deg))

  def eligible(self, s: RobotState) -> bool:
    return min(r.settle_err(s)[0] for r in self.routines) < self.max_eg

  def choose(self, s: RobotState) -> "HumanRoutine":
    self.cur = min(self.routines, key=lambda r: r.settle_err(s)[0])
    return self.cur

  @property
  def settle(self):
    return self.cur.settle

  @property
  def track(self):
    return self.cur.track

  def is_settled(self, s: RobotState) -> bool:
    return self.cur.is_settled(s)


class Supervisor:
  """GETUP (also holds the standing pose) <-> DANCE.

  * Boot: always GETUP. Lying -> it gets up; already standing -> it just stands.
  * DANCE is entered only on request and only from a stable stand.
  * Fall detection in DANCE: torso tilt > `fall_tilt_deg` for `fall_hold` ticks,
    or a large unexpected tilt rate while tilted -> hand control to GETUP.
  * After a dance ends -> GETUP (standing hold).
  """

  def __init__(self, getup: GetupPolicy, dances: dict[str, DancePolicy], dt: float = 0.02,
               fall_tilt_deg: float = 50.0, fall_hold: int = 3, human: HumanGetup | None = None,
               lying_tilt_deg: float = 60.0, settle_timeout_s: float = 8.0):
    self.getup, self.dances, self.dt = getup, dances, dt
    self.human = human
    self.cos_lying = np.cos(np.radians(lying_tilt_deg))
    self.settle_timeout = int(settle_timeout_s / dt)
    self._mode_ticks = 0
    self._settled_ticks = 0
    self._track_bad = 0
    self.mode = "GETUP"
    self._boot = True
    self._lying_ticks = 0
    self._fallback = False  # human path failed -> stay with the PPO get-up until standing again
    self.active: DancePolicy | None = None
    self.cos_fall = np.cos(np.radians(fall_tilt_deg))
    self.fall_hold = fall_hold
    self._tilted = 0
    self.stable_ticks = 0
    self.events: list[tuple[float, str]] = []
    self.time = 0.0
    self.last_target = DEFAULT_Q.copy()
    self.getup.reset()

  def upright_cos(self, s: RobotState) -> float:
    return float(-projected_gravity(s.quat)[2])

  def is_stable_stand(self, s: RobotState) -> bool:
    return (self.upright_cos(s) > 0.95 and np.abs(s.gyro).max() < 0.6
            and np.abs(s.q - DEFAULT_Q).max() < 0.35 and np.abs(s.qd).max() < 1.5)

  def recovering(self) -> bool:
    return self.mode != "DANCE" and self.stable_ticks == 0

  def stop_dance(self) -> bool:
    if self.mode != "DANCE":
      return False
    self.mode = "GETUP"
    self.getup.reset(last_action=self.active.last_action)
    return True

  def request_dance(self, name: str, s: RobotState) -> bool:
    if self.mode != "GETUP" or self.stable_ticks < int(1.0 / self.dt):
      return False
    self.active = self.dances[name]
    self.active.start(s, last_action=self.getup.last_action)
    self.mode = "DANCE"
    self.events.append((self.time, f"dance:{name}"))
    return True

  def _enter(self, mode: str, s: RobotState, last_action, why: str):
    self.events.append((self.time, f"{why}->{mode}"))
    if why in ("settle_timeout", "track_diverged", "not_supine"):
      self._fallback = True
    self.mode = mode
    self._mode_ticks = 0
    self.stable_ticks = 0
    if mode == "GETUP":
      self.getup.reset(last_action=last_action)
    elif mode == "SETTLE":
      self._settled_ticks = 0
      self.human.choose(s)
      self.events.append((self.time, f"routine:{self.human.cur.name}"))
      self.human.settle.reset(last_action=last_action)
    elif mode == "HTRACK":
      self._track_bad = 0
      self.human.track.start(s, last_action=last_action)

  def _down_entry(self, s: RobotState, last_action, why: str):
    """Fallen / lying: human-style path when available, else the PPO get-up (v3)."""
    if self.human is not None:
      self._enter("SETTLE", s, last_action, why)
    else:
      self._enter("GETUP", s, last_action, why)

  def step(self, s: RobotState) -> np.ndarray:
    self.time += self.dt
    self._mode_ticks += 1
    if self._boot:
      self._boot = False
      if self.human is not None and self.upright_cos(s) < self.cos_lying and self.human.eligible(s):
        self._enter("SETTLE", s, None, "boot_lying")
    if self.mode == "SETTLE":
      self._settled_ticks = self._settled_ticks + 1 if self.human.is_settled(s) else 0
      if self._mode_ticks == int(1.0 / self.dt) and not self.human.eligible(s):
        self._enter("GETUP", s, self.human.settle.last_action, "not_supine")
      elif self._settled_ticks >= 15:
        self._enter("HTRACK", s, self.human.settle.last_action, "settled")
      elif self._mode_ticks > self.settle_timeout:
        self._enter("GETUP", s, self.human.settle.last_action, "settle_timeout")
    elif self.mode == "HTRACK":
      tr = self.human.track
      if tr.t > 0:
        bad = float(np.sqrt(np.mean((s.q - tr.ref_q) ** 2))) > 0.5
        self._track_bad = self._track_bad + 1 if bad else 0
      if self._track_bad >= 25:
        self._enter("GETUP", s, tr.last_action, "track_diverged")
      elif tr.done:
        self._enter("GETUP", s, tr.last_action, "human_getup_done")
    if self.mode == "SETTLE":
      tgt = self.human.settle.act(s)
      self.last_target = tgt
      return tgt
    if self.mode == "HTRACK":
      tgt = self.human.track.act(s)
      self.last_target = tgt
      return tgt
    if self.mode == "DANCE":
      c = self.upright_cos(s)
      self._tilted = self._tilted + 1 if c < self.cos_fall else 0
      if self._tilted >= self.fall_hold:
        self._down_entry(s, self.active.last_action, "fall_detected")
        if self.mode != "DANCE" and self.mode != "GETUP":
          return self.step_mode_only(s)
      elif self.active.done:
        self.events.append((self.time, "dance_done->GETUP"))
        self.mode = "GETUP"
        self.getup.reset(last_action=self.active.last_action)
    if self.mode == "DANCE":
      tgt = self.active.act(s)
    else:
      # knocked over while standing in GETUP: lying still for 0.5 s -> human-style path
      lying = (self.human is not None and not self._fallback and self.upright_cos(s) < self.cos_lying
               and self.human.eligible(s) and np.abs(s.gyro).max() < 0.5)
      self._lying_ticks = self._lying_ticks + 1 if lying else 0
      if self._lying_ticks >= 25:
        self._lying_ticks = 0
        self._enter("SETTLE", s, self.getup.last_action, "lying")
        return self.step_mode_only(s)
      tgt = self.getup.act(s)
      self.stable_ticks = self.stable_ticks + 1 if self.is_stable_stand(s) else 0
      if self.stable_ticks >= int(1.0 / self.dt):
        self._fallback = False
    self.last_target = tgt
    return tgt

  def step_mode_only(self, s: RobotState) -> np.ndarray:
    pol = self.human.settle if self.mode == "SETTLE" else self.human.track
    tgt = pol.act(s)
    self.last_target = tgt
    return tgt
