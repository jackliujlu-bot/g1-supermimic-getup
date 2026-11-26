"""Export a G1-Getup checkpoint to ONNX and verify obs/ONNX parity with the deploy stack.

Usage: python export_getup.py CHECKPOINT.pt OUT_DIR
Parity: runs the mjlab play env (no noise), feeds the deploy stack's GetupPolicy with raw
robot state and compares its joint targets to the torch policy's on the env's own obs.
"""
import os
import sys
from dataclasses import asdict

import numpy as np
import torch

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(W, "getup"))
sys.path.insert(0, os.path.join(W, "deploy"))
import g1getup  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs import mdp as mjmdp  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # noqa: E402

ckpt, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
dev = "cuda:0"
cfg = load_env_cfg("G1-Getup", play=True)
cfg.scene.num_envs = 8
cfg.events.pop("push", None)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev))
agent = load_rl_cfg("G1-Getup")
runner = MjlabOnPolicyRunner(env, asdict(agent), out, dev)
runner.load(ckpt)
runner.export_policy_to_onnx(out, "getup.onnx")
policy = runner.get_inference_policy(device=dev)
print("exported", os.path.join(out, "getup.onnx"))

from g1_stack import ACTION_SCALE, DEFAULT_Q, GetupPolicy, RobotState  # noqa: E402

gp = GetupPolicy(os.path.join(out, "getup.onnx"))
robot = env.unwrapped.scene["robot"]
obs = env.get_observations()
errs = []
gp.reset()
for k in range(150):
  with torch.no_grad():
    a_t = policy(obs)
  d = robot.data
  s = RobotState(quat=d.root_link_quat_w[0].cpu().numpy().astype(np.float64),
                 gyro=mjmdp.builtin_sensor(env.unwrapped, "robot/imu_ang_vel")[0].cpu().numpy(),
                 q=d.joint_pos[0].cpu().numpy(), qd=d.joint_vel[0].cpu().numpy())
  tgt = gp.act(s)
  tgt_t = DEFAULT_Q + ACTION_SCALE * a_t[0].cpu().numpy()
  errs.append(float(np.abs(tgt - tgt_t).max()))
  obs, _, dones, _ = env.step(a_t)
  if bool(dones[0]):
    gp.reset()
print(f"parity max |dq*| over 150 steps: {max(errs):.2e} (median {np.median(errs):.2e})")
