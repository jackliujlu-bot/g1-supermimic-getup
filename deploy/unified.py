"""One tracker for everything after the robot is down: get up (human clip) -> stand -> dance -> stand.

The SuperMimic tracker (`tracker.onnx`, reference motion as an input) follows a single continuous
reference timeline. Get-up clip, standing hold, smooth blends and dances are stitched into that
timeline, so normal operation never switches networks:

    SETTLE (small PPO: any fallen pose -> lying start pose of the human get-up clip)
      -> TRACK  [get-up clip | hold stand | blend | dance | blend | hold stand | ...]

The PPO get-up policy (v3) is only a fallback: poses the human get-up does not cover, settle
timeout, or the robot diverging from the reference. When v3 has the robot standing again the
tracker takes over with a blend from the current pose to the standing pose.
"""
from __future__ import annotations

import os

import numpy as np

from g1_stack import (ACTION_SCALE, DEFAULT_Q, JOINT_NAMES, NJ, GetupPolicy, RobotState, TermHistory, _session,
                      projected_gravity, quat_mul, quat_rotate_inv, sensor_terms, yaw_of, yaw_quat)

STRIDE, NGOAL = 5, 10


def load_clip(path: str) -> dict:
  z = np.load(path)
  return {"q": z["joint_pos"].astype(np.float64), "qd": z["joint_vel"].astype(np.float64),
          "quat": z["root_quat_w"].astype(np.float64), "name": os.path.basename(path).rsplit(".", 1)[0]}


def _slerp(a, b, u):
  d = float(np.dot(a, b))
  if d < 0:
    b, d = -b, -d
  if d > 0.9995:
    r = a + u * (b - a)
    return r / np.linalg.norm(r)
  th = np.arccos(d)
  return (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)


def _tilt_deg(quat) -> float:
  return float(np.degrees(np.arccos(np.clip(-projected_gravity(quat)[2], -1, 1))))


class RefTimeline:
  """Reference frames at 50 Hz: joint pos/vel and pelvis orientation (motion frame) + a label."""

  def __init__(self):
    self.q = np.zeros((0, NJ))
    self.qd = np.zeros((0, NJ))
    self.quat = np.zeros((0, 4))
    self.label: list[str] = []
    self.t = 0

  # ---- building
  def reset_pose(self, q, quat, label="stand"):
    self.q, self.qd, self.quat = q[None].copy(), np.zeros((1, NJ)), quat[None].copy()
    self.label, self.t = [label], 0

  def reset_clip(self, clip):
    self.q, self.qd, self.quat = clip["q"].copy(), clip["qd"].copy(), clip["quat"].copy()
    self.label, self.t = [clip["name"]] * len(clip["q"]), 0

  def truncate(self):
    """Drop everything after the current frame (to re-plan from here)."""
    k = self.t + 1
    self.q, self.qd, self.quat, self.label = self.q[:k], self.qd[:k], self.quat[:k], self.label[:k]
    self.qd[-1] = 0.0

  def _append(self, q, qd, quat, label):
    self.q = np.concatenate([self.q, q])
    self.qd = np.concatenate([self.qd, qd])
    self.quat = np.concatenate([self.quat, quat])
    self.label += [label] * len(q)

  def append_clip(self, clip, label=None):
    """Append a clip, rotated about z so it starts with the heading the timeline ends with."""
    off = yaw_quat(yaw_of(self.quat[-1]) - yaw_of(clip["quat"][0]))
    quat = np.array([quat_mul(off, x) for x in clip["quat"]])
    self._append(clip["q"], clip["qd"], quat, label or clip["name"])

  def append_blend(self, q_to, quat_to, seconds=1.0, label="blend"):
    """Smooth (smoothstep) blend from the last frame to pose q_to; the target orientation keeps the
    current heading and takes only roll/pitch from quat_to."""
    n = max(2, int(round(seconds / 0.02)))
    q0, r0 = self.q[-1], self.quat[-1]
    r1 = quat_mul(yaw_quat(yaw_of(r0) - yaw_of(quat_to)), quat_to)
    u = np.arange(1, n + 1) / n
    s = u * u * (3 - 2 * u)
    q = q0 + s[:, None] * (q_to - q0)
    qd = np.gradient(np.concatenate([q0[None], q]), 0.02, axis=0)[1:]
    qd[-1] = 0.0
    quat = np.array([_slerp(r0, r1, x) for x in s])
    self._append(q, qd, quat, label)

  def compact(self, keep=60):
    if self.t > 1500:
      k = self.t - keep
      self.q, self.qd, self.quat, self.label = self.q[k:], self.qd[k:], self.quat[k:], self.label[k:]
      self.t -= k

  # ---- playing
  @property
  def at_end(self) -> bool:
    return self.t >= len(self.q) - 1

  @property
  def cur_label(self) -> str:
    return self.label[self.t]

  def window(self):
    idx = np.minimum(self.t + STRIDE * np.arange(NGOAL), len(self.q) - 1)
    return self.q[idx], self.qd[idx], self.quat[idx]

  def advance(self):
    self.t = min(self.t + 1, len(self.q) - 1)
    self.compact()


class TrackerPolicy:
  """tracker.onnx (tools/export_tracker.py): inputs obs, goal_q, goal_qd, goal_quat, anchor_quat."""

  def __init__(self, onnx_path: str):
    self.sess = _session(onnx_path)
    meta = self.sess.get_modelmeta().custom_metadata_map
    self.hist = TermHistory(int(meta.get("history_length", 10)))
    self.last_action = np.zeros(NJ, np.float32)
    self.yaw_off = np.array([1.0, 0, 0, 0])

  def start(self, s: RobotState, tl: RefTimeline, last_action=None):
    """Align the timeline's current heading with the robot's heading (motion frame = yaw_off * world)."""
    self.hist.reset()
    self.last_action = np.zeros(NJ, np.float32) if last_action is None else last_action.astype(np.float32)
    self.yaw_off = yaw_quat(yaw_of(tl.quat[tl.t]) - yaw_of(s.quat))

  def act(self, s: RobotState, tl: RefTimeline) -> np.ndarray:
    obs = self.hist.push(sensor_terms(s, self.last_action))
    gq, gqd, gquat = tl.window()
    aq = quat_mul(self.yaw_off, s.quat)
    a = self.sess.run(None, {"obs": obs[None], "goal_q": gq[None].astype(np.float32),
                             "goal_qd": gqd[None].astype(np.float32),
                             "goal_quat": gquat[None].astype(np.float32),
                             "anchor_quat": aq[None].astype(np.float32)})[0][0]
    self.last_action = a.astype(np.float32)
    return DEFAULT_Q + ACTION_SCALE * a


class SettleRoutine:
  def __init__(self, settle_onnx: str, getup_clip: dict):
    self.settle = GetupPolicy(settle_onnx)
    self.clip = getup_clip
    self.q_t = getup_clip["q"][0]
    self.g_t = quat_rotate_inv(getup_clip["quat"][0], np.array([0.0, 0.0, -1.0]))

  def err(self, s: RobotState):
    return (float(np.sum((projected_gravity(s.quat) - self.g_t) ** 2)),
            float(np.sqrt(np.mean((s.q - self.q_t) ** 2))))

  def settled(self, s: RobotState, tol: float = 0.30) -> bool:
    """Body orientation at the lying start pose, robot at rest, joints within `tol` rad rms.
    (Real robot 2026-11-24: rolled over fine but stopped at rms 0.23 with the waist yaw 0.9 rad off;
    the 1 s blend at the start of the get-up reference absorbs such a residual.)"""
    eg, eq = self.err(s)
    return eg < 0.05 and eq < tol and np.abs(s.qd).max() < 1.0


class UnifiedSupervisor:
  """Modes: SETTLE -> TRACK (one tracker, one reference timeline).

  getup=None (default on the robot): no PPO get-up policy at all. Boot standing -> the tracker blends
  to the standing pose; boot lying / sitting / kneeling -> SETTLE -> human get-up; any failure ->
  damping (wants_damp). With a getup policy given, it is used as the fallback / for upright boots."""

  def __init__(self, getup: GetupPolicy, tracker: TrackerPolicy, routines: list[SettleRoutine],
               dances: dict[str, dict], stand_q: np.ndarray | None = None, dt: float = 0.02,
               human_tilt_deg: float = 180.0, fall_tilt_deg: float = 50.0, fall_hold: int = 3,
               blend_s: float = 1.0, settle_timeout_s: float = 8.0, diverge_rms: float = 0.5,
               fallback: str = "v3", settle_tol: float = 0.30):
    self.getup, self.tracker, self.routines, self.dances, self.dt = getup, tracker, routines, dances, dt
    self.stand_q = routines[0].clip["q"][-1].copy() if stand_q is None else stand_q
    self.stand_quat = routines[0].clip["quat"][-1].copy()
    self.max_eg = 2.0 - 2.0 * np.cos(np.radians(human_tilt_deg))
    self.cos_fall = np.cos(np.radians(fall_tilt_deg))
    self.fall_hold, self.blend_s = fall_hold, blend_s
    self.settle_timeout = int(settle_timeout_s / dt)
    self.diverge_rms = diverge_rms
    self.fallback, self.settle_tol = fallback, settle_tol
    self.wants_damp = False  # fallback="damp": the controller should switch to damping
    self.tl = RefTimeline()
    self.cur: SettleRoutine = routines[0]
    self.mode = "GETUP"
    self.events: list[tuple[float, str]] = []
    self.time = 0.0
    self.stable_ticks = 0
    self._ticks = 0
    self._settled = 0
    self._near, self._hold, self._hold_from = 0, None, None
    self._attempts = 1  # the boot get-up is attempt 1
    self.max_attempts = 4
    self._tilted = 0
    self._bad = 0
    self._lying = 0
    self._fallback = False
    self._boot = True
    self.last_target = DEFAULT_Q.copy()
    if self.getup is not None:
      self.getup.reset()
    self._restands = []

  # ---- helpers
  def upright_cos(self, s):
    return float(-projected_gravity(s.quat)[2])

  def is_stable_stand(self, s, ref_q):
    return (self.upright_cos(s) > 0.95 and np.abs(s.gyro).max() < 0.6
            and np.abs(s.q - ref_q).max() < 0.35 and np.abs(s.qd).max() < 1.5)

  def eligible(self, s):
    return min(r.err(s)[0] for r in self.routines) < self.max_eg

  @property
  def idle(self) -> bool:
    """Standing still, ready for a dance."""
    return self.mode == "TRACK" and self.tl.at_end and self.tl.cur_label == "stand" and self.stable_ticks > 0

  def _enter(self, mode, s, last_action, why, alternate=False):
    if mode == "GETUP" and (self.getup is None or
                            (self.fallback == "damp" and why in ("settle_timeout", "not_human_pose", "diverged"))):
      self.events.append((self.time, f"{why}->DAMP (fallback=damp)"))
      self.wants_damp = True
      self.mode = "FAILED"
      return
    self.events.append((self.time, f"{why}->{mode}"))
    if why in ("settle_timeout", "not_human_pose", "diverged"):
      self._fallback = True
    self.mode, self._ticks, self.stable_ticks = mode, 0, 0
    if mode == "GETUP":
      self.getup.reset(last_action=last_action)
    elif mode == "SETTLE":
      pool = [r for r in self.routines if r is not self.cur] if (alternate and len(self.routines) > 1) else self.routines
      self.cur = min(pool, key=lambda r: r.err(s)[0])
      self._settled = 0
      self._near, self._hold = 0, None
      self.cur.settle.reset(last_action=last_action)

  def _start_track_getup(self, s, last_action):
    """Reference: 1 s blend from the robot's actual pose to the clip's first frame, then the clip."""
    self.tl.reset_pose(s.q.astype(np.float64), s.quat.astype(np.float64), label="getup_blend")
    self.tl.append_blend(self.cur.clip["q"][0], self.cur.clip["quat"][0], self.blend_s, label="getup_blend")
    self.tl.append_clip(self.cur.clip)
    if np.abs(self.cur.clip["q"][-1] - self.stand_q).max() > 0.05:
      # get-up clips end in different human standing poses: blend into the one common standing pose
      # (dances start from it and the stand check compares against it)
      self.tl.append_blend(self.stand_q, self.stand_quat, self.blend_s, label="stand")
    else:
      self.tl.label[-1] = "stand"  # last frame of the get-up clip is the standing hold
    self.tracker.start(s, self.tl, last_action)
    self._bad = self._tilted = 0

  def _start_track_stand(self, s, last_action):
    """From wherever the robot is (standing, after v3): blend from the current pose to the stand pose."""
    q_now = s.q.astype(np.float64)
    self.tl.reset_pose(q_now, s.quat.astype(np.float64), label="blend")
    self.tl.append_blend(self.stand_q, self.stand_quat, self.blend_s, label="stand")
    self.tracker.start(s, self.tl, last_action)
    self._bad = self._tilted = 0

  def _down(self, s, last_action, why):
    """Fell / get-up failed. A fresh fall starts a new series of attempts; a failed attempt (settle stuck,
    get-up clip diverged) retries with the OTHER routine (supine <-> prone: e.g. arm trapped under the
    body on a soft uneven floor, sim2sim 2026-11-26), up to max_attempts, then the fallback (damp)."""
    if why == "fall_detected":
      self._attempts = 0
    if self.eligible(s) and self._attempts < self.max_attempts:
      retry = why != "fall_detected" and self._attempts > 0
      self._attempts += 1
      self._enter("SETTLE", s, last_action, f"retry_{why}" if retry else why, alternate=retry)
    else:
      self._enter("GETUP", s, last_action, why)

  def recovering(self) -> bool:
    """Getting up (settle, v3, the get-up clip or the blend into standing): the watchdog applies."""
    if self.mode in ("SETTLE", "GETUP"):
      return True
    lab = self.tl.cur_label
    return lab.startswith("getup") or (lab == "stand" and self.tl.at_end and self.stable_ticks == 0)

  def diag(self, s: RobotState) -> str:
    ref = self.tl.q[self.tl.t] if len(self.tl.q) else DEFAULT_Q
    return (f"mode={self.mode} ref={self.tl.cur_label if len(self.tl.q) else '-'} end={self.tl.at_end if len(self.tl.q) else '-'} "
            f"stable={self.stable_ticks} up={self.upright_cos(s):.3f} gyro={np.abs(s.gyro).max():.2f} "
            f"dq={np.abs(s.q - ref).max():.2f} qd={np.abs(s.qd).max():.2f}")

  # ---- commands
  def request_dance(self, name: str, s: RobotState) -> bool:
    if not self.idle or self.stable_ticks < int(1.0 / self.dt):
      return False
    clip = self.dances[name]
    self.tl.truncate()
    self.tl.append_blend(clip["q"][0], clip["quat"][0], self.blend_s, label="blend")
    self.tl.append_clip(clip, label="dance:" + name)
    self.tl.append_blend(self.stand_q, self.stand_quat, self.blend_s, label="stand")
    self.events.append((self.time, f"dance:{name}"))
    return True

  def stop_dance(self) -> bool:
    if self.mode != "TRACK" or not self.tl.cur_label.startswith("dance:"):
      return False
    self.tl.truncate()
    self.tl.append_blend(self.stand_q, self.stand_quat, self.blend_s, label="stand")
    self.events.append((self.time, "dance_stopped"))
    return True

  # ---- control step
  def step(self, s: RobotState) -> np.ndarray:
    self.time += self.dt
    self._ticks += 1
    if self._boot:
      self._boot = False
      if self.getup is None:
        knees = [i for i, n in enumerate(JOINT_NAMES) if "knee" in n]
        standing = self.upright_cos(s) > 0.9 and np.all(s.q[knees] < 1.0)
        if standing:
          self.events.append((self.time, "boot_standing->TRACK:stand"))
          self.mode, self._ticks = "TRACK", 0
          self._start_track_stand(s, None)
        else:
          self._enter("SETTLE", s, None, "boot_not_standing")
      elif self.upright_cos(s) < np.cos(np.radians(60)) and self.eligible(s):
        self._enter("SETTLE", s, None, "boot_lying")

    if self.mode == "SETTLE":
      # once the start pose is reached, stop the settle policy and hold that pose with plain PD: on a
      # soft mat the policy keeps wiggling (2-6 rad/s, sim2sim 2026-11-25) and never comes to rest
      eg, eq = self.cur.err(s)
      near = eg < 0.05 and eq < self.settle_tol
      self._near = self._near + 1 if near else 0
      if self._hold is None and self._near >= 10:
        self._hold = 0
        self._hold_from = self.last_target.copy()
        self.events.append((self.time, "settle_pose->HOLD"))
      elif self._hold is not None and (eg > 0.10 or eq > self.settle_tol + 0.10):
        self._hold = None
        self._near = 0
        self.events.append((self.time, "hold_drift->SETTLE"))
      self._settled = self._settled + 1 if (self._hold is not None and self._hold >= 15
                                            and self.cur.settled(s, self.settle_tol)) else 0
      if self._ticks == int(1.0 / self.dt) and not self.eligible(s):
        self._enter("GETUP", s, self.cur.settle.last_action, "not_human_pose")
      elif self._settled >= 15:
        self.events.append((self.time, f"settled->TRACK:{self.cur.clip['name']}"))
        self.mode, self._ticks = "TRACK", 0
        self._start_track_getup(s, self.cur.settle.last_action)
      elif self._ticks > self.settle_timeout:
        self._down(s, self.cur.settle.last_action, "settle_timeout")

    elif self.mode == "TRACK":
      ref_up = _tilt_deg(self.tl.quat[self.tl.t]) < 30
      self._tilted = self._tilted + 1 if (ref_up and self.upright_cos(s) < self.cos_fall) else 0
      bad = float(np.sqrt(np.mean((s.q - self.tl.q[self.tl.t]) ** 2))) > self.diverge_rms
      self._bad = self._bad + 1 if bad else 0
      # during the get-up clip the robot may lag the reference a little: require a longer tilt
      hold = self.fall_hold if not self.tl.cur_label.startswith("getup") else 25
      if self._tilted >= hold:
        self._fallback = False
        self._down(s, self.tracker.last_action, "fall_detected")
      elif self._bad >= 25:
        if self.upright_cos(s) < self.cos_fall:
          self._down(s, self.tracker.last_action, "diverged")
        elif self.getup is None:
          # still upright but far off the reference: re-plan from the current pose to the stand pose
          # (twice within 5 s -> give up and damp)
          self._restands = [x for x in self._restands if self.time - x < 5.0] + [self.time]
          if len(self._restands) > 2:
            self._enter("GETUP", s, self.tracker.last_action, "diverged")
          else:
            self.events.append((self.time, "diverged_upright->TRACK:restand"))
            self._start_track_stand(s, self.tracker.last_action)
        else:
          self._enter("GETUP", s, self.tracker.last_action, "diverged")

    elif self.mode == "GETUP":
      lying = (not self._fallback and self.upright_cos(s) < np.cos(np.radians(60)) and self.eligible(s)
               and np.abs(s.gyro).max() < 0.5)
      self._lying = self._lying + 1 if lying else 0
      if self._lying >= 25:
        self._lying = 0
        self._enter("SETTLE", s, self.getup.last_action, "lying")
      elif self.stable_ticks >= int(1.0 / self.dt):
        self._fallback = False
        self.events.append((self.time, "stood->TRACK:stand"))
        self.mode, self._ticks = "TRACK", 0
        self._start_track_stand(s, self.getup.last_action)

    # ---- act
    if self.mode == "FAILED":
      self.last_target = s.q.copy()
      return self.last_target
    if self.mode == "SETTLE":
      if self._hold is None:
        tgt = self.cur.settle.act(s)
      else:  # PD hold of the clip's first frame, blended in over 0.3 s
        w = min(1.0, (self._hold + 1) / 15)
        tgt = (1 - w) * self._hold_from + w * self.cur.q_t
        self._hold += 1
    elif self.mode == "TRACK":
      tgt = self.tracker.act(s, self.tl)
      ref_q = self.tl.q[self.tl.t]
      holding = self.tl.at_end and self.tl.cur_label == "stand"
      # the tracker keeps making small balance corrections while standing: looser velocity limits
      # than for v3 and a leaky counter, so one twitchy tick does not reset "standing stably"
      steady = (holding and self.upright_cos(s) > 0.95 and np.abs(s.gyro).max() < 1.0
                and np.abs(s.q - ref_q).max() < 0.35 and np.abs(s.qd).max() < 3.0)
      self.stable_ticks = self.stable_ticks + 1 if steady else (max(0, self.stable_ticks - 5) if holding else 0)
      if self.tl.label[self.tl.t] != self.tl.label[min(self.tl.t + 1, len(self.tl.label) - 1)]:
        self.events.append((self.time, f"ref:{self.tl.label[min(self.tl.t + 1, len(self.tl.label) - 1)]}"))
      self.tl.advance()
    else:
      tgt = self.getup.act(s)
      self.stable_ticks = self.stable_ticks + 1 if self.is_stable_stand(s, DEFAULT_Q) else 0
    self.last_target = tgt
    return tgt


def unified_pairs(args):
  """CLI list TRACKER SETTLE CLIP [SETTLE CLIP ...] -> [(settle, clip), ...]"""
  rest = list(args[1:])
  if len(rest) < 2 or len(rest) % 2:
    raise SystemExit("--unified: TRACKER_ONNX then pairs of SETTLE_ONNX GETUP_CLIP")
  return [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]


# standing pose after a get-up / at boot: the end pose of the LAFAN prone get-up clip. (The BONES
# getup_supine end pose made the tracker shake on a soft mat: 16 deg joint swing vs 4, sim2sim 2026-11-26.)
STAND_CLIP = os.environ.get("G1_STAND_CLIP") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions", "lafan_prone.npz")


def build_unified(getup_onnx, tracker_onnx, routines, dance_clips, stand_clip=STAND_CLIP, **kw):
  """routines: list of (settle_onnx, getup_clip_npz); dance_clips: dict name -> clip npz path.
  stand_clip: the common standing pose is this clip's last frame (the BONES get-up ends in a neutral
  stance, the LAFAN ones in splayed human stances); None = the first routine's clip end."""
  sup = UnifiedSupervisor(GetupPolicy(getup_onnx) if getup_onnx else None, TrackerPolicy(tracker_onnx),
                          [SettleRoutine(a, load_clip(b)) for a, b in routines],
                          {k: load_clip(v) for k, v in dance_clips.items()}, **kw)
  if stand_clip and os.path.exists(stand_clip) and "stand_q" not in kw:
    c = load_clip(stand_clip)
    sup.stand_q, sup.stand_quat = c["q"][-1].copy(), c["quat"][-1].copy()
  # warm up every ONNX session (the first run is slow: 157 ms on the G1's Orin)
  s0 = RobotState(quat=np.array([1.0, 0, 0, 0]), gyro=np.zeros(3), q=DEFAULT_Q.copy(), qd=np.zeros(NJ))
  if sup.getup is not None:
    sup.getup.act(s0); sup.getup.reset()
  for r in sup.routines:
    r.settle.act(s0); r.settle.reset()
  tl = RefTimeline(); tl.reset_pose(DEFAULT_Q.astype(np.float64), np.array([1.0, 0, 0, 0]))
  sup.tracker.start(s0, tl); sup.tracker.act(s0, tl)
  return sup
