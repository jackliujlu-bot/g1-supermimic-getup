"""Export the SuperMimic tracker with the reference motion as an INPUT (no clip baked in), so the
deploy stack can stitch get-up, standing and dances into one continuous reference timeline.

Inputs : obs [1, H*D]        term-major proprio history (oldest -> newest)
         goal_q [1, 10, 29]  reference joint pos at t, t+5, ..., t+45 (clamped at the timeline end)
         goal_qd [1, 10, 29] reference joint vel at the same frames
         goal_quat [1, 10, 4] reference pelvis quat (wxyz) at the same frames, motion frame
         anchor_quat [1, 4]  robot pelvis quat (wxyz) expressed in the motion frame
Outputs: actions [1, 29]

Usage: python export_tracker.py RUN_DIR ANY_CLIP.npz OUT.onnx
"""
import argparse
import json
import os

import mjlab.tasks  # noqa: F401
import onnx
import torch
import torch.nn as nn
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_inv, quat_mul

import supermimic.tasks  # noqa: F401
from supermimic.representations import GOAL_FRAME_STRIDE, NUM_GOAL_FRAMES
from supermimic.runners.exporter import _SingleEncoderOnnx
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor, build_world_model
from supermimic.utils.helpers import quat_to_rot6d

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("clip")
ap.add_argument("out")
ap.add_argument("--history", type=int, default=10)
a = ap.parse_args()
dev = "cuda:0"
env_cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
env_cfg.scene.num_envs = 1
env_cfg.commands["motion"].motion_file = a.clip
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev))
rcfg = SuperMimicRunnerCfg(actor_obs="proprio", history_length=a.history)
wm = build_world_model(rcfg, env, dev)
actor = build_actor(rcfg, wm, dev)
actor.load_state_dict(torch.load(os.path.join(a.run_dir, "actor.pt"), map_location=dev))
actor.eval()
term_widths = wm.representation.proprio_term_widths(wm.num_actions)


class TrackerOnnx(nn.Module):
  def __init__(self):
    super().__init__()
    self.pi = _SingleEncoderOnnx(actor.cpu(), "motion", term_widths)

  def forward(self, obs, goal_q, goal_qd, goal_quat, anchor_quat):
    b, n, _ = goal_q.shape
    cmd = torch.cat([goal_q, goal_qd], dim=-1).reshape(b, -1)
    rq = anchor_quat[:, None, :].expand(-1, n, 4).reshape(-1, 4)
    rel = quat_mul(quat_inv(rq), goal_quat.reshape(-1, 4))
    rot6 = quat_to_rot6d(rel).reshape(b, -1)
    return self.pi(torch.cat([obs, cmd, rot6], dim=-1))


m = TrackerOnnx().cpu().eval()
hist = m.pi.hist_size
N = NUM_GOAL_FRAMES
torch.onnx.export(m, (torch.zeros(1, hist), torch.zeros(1, N, 29), torch.zeros(1, N, 29),
                      torch.tensor([[[1.0, 0, 0, 0]] * N]), torch.tensor([[1.0, 0, 0, 0]])), a.out,
                  input_names=["obs", "goal_q", "goal_qd", "goal_quat", "anchor_quat"], output_names=["actions"],
                  opset_version=17, dynamo=False)
mo = onnx.load(a.out)
meta = {"history_length": str(a.history), "num_goal_frames": str(N), "goal_stride": str(GOAL_FRAME_STRIDE),
        "joint_names": json.dumps(list(env.unwrapped.scene["robot"].joint_names))}
for k, v in meta.items():
  p = mo.metadata_props.add()
  p.key, p.value = k, v
onnx.save(mo, a.out)
print(f"exported {a.out}: hist {hist}, goal frames {N} stride {GOAL_FRAME_STRIDE}")
