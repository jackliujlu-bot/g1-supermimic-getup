"""Export a SuperMimic checkpoint + one motion clip to a self-contained deploy ONNX.

Inputs : obs [1, H*D] (term-major proprio history, oldest->newest),
         anchor_quat [1, 4] (robot pelvis wxyz, already yaw-aligned to the clip frame),
         time_step [1, 1]
Outputs: actions [1, 29], ref_joint_pos [1, 29]
Metadata: history_length, num_frames, ref_yaw0, joint_names, clip.

Also checks parity against the official SuperMimic exporter (policy_motion.onnx) on the
env's own goal observations while the actor tracks the clip in mjlab.

Usage: python export_dance.py RUN_DIR CLIP.npz OUT.onnx [--steps 300]
"""
import argparse
import json
import math
import os

import mjlab.tasks  # noqa: F401
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_inv, quat_mul

import supermimic.tasks  # noqa: F401
from supermimic.representations import motion_anchor_window, motion_goal_window
from supermimic.runners.exporter import _SingleEncoderOnnx, export_actor_as_onnx
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor, build_world_model
from supermimic.utils.helpers import quat_to_rot6d

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("clip")
ap.add_argument("out")
ap.add_argument("--steps", type=int, default=300)
ap.add_argument("--history", type=int, default=10)
a = ap.parse_args()
dev = "cuda:0"

env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
env_cfg.scene.num_envs = 1
env_cfg.commands["motion"].motion_file = a.clip
env_cfg.commands["motion"].sampling_mode = "start"
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev))
rcfg = SuperMimicRunnerCfg(actor_obs="proprio", history_length=a.history)
wm = build_world_model(rcfg, env, dev)
actor = build_actor(rcfg, wm, dev)
actor.load_state_dict(torch.load(os.path.join(a.run_dir, "actor.pt"), map_location=dev))
actor.eval()
term_widths = wm.representation.proprio_term_widths(wm.num_actions)

motion = env.unwrapped.command_manager.get_term("motion").motion
T = motion.time_step_total
ts = torch.arange(T, device=dev)
cmd_table = motion_goal_window(motion, ts)  # [T, N*2J]
_, tq = motion_anchor_window(motion, ts)  # [T, N, 4]
ref_q = motion.joint_pos.clone()
w0, x0, y0, z0 = motion.root_quat_w[0].tolist()
ref_yaw0 = math.atan2(2 * (w0 * z0 + x0 * y0), 1 - 2 * (y0 * y0 + z0 * z0))


class DanceOnnx(nn.Module):
  def __init__(self):
    super().__init__()
    self.pi = _SingleEncoderOnnx(actor, "motion", term_widths)
    self.register_buffer("cmd", cmd_table.detach().cpu())
    self.register_buffer("tq", tq.detach().cpu())
    self.register_buffer("refq", ref_q.detach().cpu())
    self.T = T

  def forward(self, obs, anchor_quat, time_step):
    t = torch.clamp(time_step.long().squeeze(-1), 0, self.T - 1)
    tq_t = self.tq[t]  # [1, N, 4]
    n = tq_t.shape[1]
    rq = anchor_quat[:, None, :].expand(-1, n, 4).reshape(-1, 4)
    rel = quat_mul(quat_inv(rq), tq_t.reshape(-1, 4))
    rot6 = quat_to_rot6d(rel).reshape(anchor_quat.shape[0], -1)
    x = torch.cat([obs, self.cmd[t], rot6], dim=-1)
    return self.pi(x), self.refq[t]


m = DanceOnnx().cpu().eval()
hist = m.pi.hist_size
torch.onnx.export(m, (torch.zeros(1, hist), torch.tensor([[1.0, 0, 0, 0]]), torch.zeros(1, 1)), a.out,
                  input_names=["obs", "anchor_quat", "time_step"], output_names=["actions", "ref_joint_pos"],
                  opset_version=17, dynamo=False)
mo = onnx.load(a.out)
meta = {"history_length": str(a.history), "num_frames": str(T), "ref_yaw0": f"{ref_yaw0:.8f}",
        "joint_names": json.dumps(list(env.unwrapped.scene["robot"].joint_names)), "clip": os.path.basename(a.clip)}
for k, v in meta.items():
  p = mo.metadata_props.add()
  p.key, p.value = k, v
onnx.save(mo, a.out)
print(f"exported {a.out}: {T} frames ({T/50:.1f}s), hist {hist}")

# ---------------- parity vs the official exporter ----------------
offdir = os.path.join(os.path.dirname(os.path.abspath(a.out)), "_official")
export_actor_as_onnx(actor.cpu(), env.unwrapped, offdir, "parity", term_widths)
actor.to(dev)
off = ort.InferenceSession(os.path.join(offdir, "policy_motion.onnx"), providers=["CPUExecutionProvider"])
mine = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
gm = env.unwrapped.observation_manager
goal_terms = gm.active_terms["goal"]
widths = [gm.group_obs_term_dim["goal"][i][0] for i in range(len(goal_terms))]
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deploy"))
from g1_stack import TermHistory, RobotState, sensor_terms, DEFAULT_Q, ACTION_SCALE  # noqa: E402
from mjlab.envs import mdp as mjmdp  # noqa: E402

env.reset()
robot = env.unwrapped.scene["robot"]
cmdterm = env.unwrapped.command_manager.get_term("motion")
h = TermHistory(a.history)
last = np.zeros(29, np.float32)
diffs, errs = [], []
for k in range(min(a.steps, T - 2)):
  d = robot.data
  s = RobotState(quat=d.root_link_quat_w[0].cpu().numpy().astype(np.float64),
                 gyro=mjmdp.builtin_sensor(env.unwrapped, "robot/imu_ang_vel")[0].cpu().numpy(),
                 q=d.joint_pos[0].cpu().numpy(), qd=d.joint_vel[0].cpu().numpy())
  ob = h.push(sensor_terms(s, last))
  goal = gm.compute_group("goal")[0].cpu().numpy()
  parts = np.split(goal, np.cumsum(widths)[:-1])
  g_actor = np.concatenate([p for n, p in zip(goal_terms, parts) if n != "motion_anchor_pos_b"])
  t = int(cmdterm.time_steps[0])
  a_off = off.run(None, {"obs": np.concatenate([ob, g_actor])[None].astype(np.float32),
                         "time_step": np.array([[t]], np.float32)})[0][0]
  # the env's motion is in the env frame; anchor quat = robot quat (same frame, no yaw offset needed)
  a_me = mine.run(None, {"obs": ob[None], "anchor_quat": s.quat[None].astype(np.float32),
                         "time_step": np.array([[t]], np.float32)})[0][0]
  diffs.append(float(np.abs(a_off - a_me).max()))
  last = a_me.astype(np.float32)
  act = torch.tensor(a_me, device=dev)[None]
  _, _, dones, _ = env.step(act)
  errs.append(float(torch.norm(cmdterm.body_pos_w - cmdterm.robot_body_pos_w, dim=-1).mean()))
  if bool(dones[0]):
    print(f"episode ended at step {k}")
    break
print(f"parity max|a_official - a_deploy| = {max(diffs):.2e}; mean body pos err {np.mean(errs):.3f} m over {len(errs)} steps")
