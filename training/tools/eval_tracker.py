"""Batch evaluation of a SuperMimic checkpoint in mjlab (deploy-path actor, sensor noise on).

Each env plays one clip from frame 0 to its end; a fall = the env's anchor_pos/anchor_ori
termination fires before the clip ends. Reports success rate and mean errors.

Usage: python eval_tracker.py RUN_DIR CLIP_LIST.txt [--envs 1024] [--rounds 2] [--out res.json]
"""
import argparse
import json
import os

import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.runners.exporter import _SingleEncoderOnnx
from supermimic.runners.supermimic_runner import SuperMimicRunnerCfg, build_actor, build_world_model

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("clips")
ap.add_argument("--envs", type=int, default=1024)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--history", type=int, default=10)
ap.add_argument("--out", default="")
a = ap.parse_args()
dev = "cuda:0"
files = [l.strip() for l in open(a.clips) if l.strip()]

cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=True)
cfg.scene.num_envs = a.envs
cfg.commands["motion"].motion_file = files
cfg.commands["motion"].sampling_mode = "start"
cfg.observations["actor"].enable_corruption = False  # noise added below (training spec)
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=dev))
rcfg = SuperMimicRunnerCfg(actor_obs="proprio", history_length=a.history)
wm = build_world_model(rcfg, env, dev)
actor = build_actor(rcfg, wm, dev)
actor.load_state_dict(torch.load(os.path.join(a.run_dir, "actor.pt"), map_location=dev))
actor.eval()
tw = wm.representation.proprio_term_widths(wm.num_actions)
pi = _SingleEncoderOnnx(actor, "motion", tw).to(dev).eval()
from supermimic.representations import PROPRIO_NOISE_HALF_WIDTHS

half = torch.cat([torch.full((w,), PROPRIO_NOISE_HALF_WIDTHS[n], device=dev)
                  for n, w in zip(PROPRIO_NOISE_HALF_WIDTHS, tw)])
U = env.unwrapped
om = U.observation_manager
goal_terms = om.active_terms["goal"]
gw = [om.group_obs_term_dim["goal"][i][0] for i in range(len(goal_terms))]
keep = torch.cat([torch.full((w,), n != "motion_anchor_pos_b", dtype=torch.bool) for n, w in zip(goal_terms, gw)]).to(dev)
cmd = U.command_manager.get_term("motion")
motion = cmd.motion
ends = motion.motion_ends
tm = U.termination_manager

res = {"ok": 0, "fail": 0, "body_pos": [], "joint_pos": [], "per_clip": {}}
for rnd in range(a.rounds):
  env.reset()
  start_clip = torch.searchsorted(ends, cmd.time_steps, right=True).clamp(max=len(ends) - 1)
  done = torch.zeros(a.envs, dtype=torch.bool, device=dev)
  failed = torch.zeros_like(done)
  H = a.history
  hist = None
  err_b = torch.zeros(a.envs, device=dev)
  err_j = torch.zeros(a.envs, device=dev)
  n = torch.zeros(a.envs, device=dev)
  for step in range(int(ends.diff().max().item() if len(ends) > 1 else ends[0].item()) + 5):
    o = om.compute_group("actor")
    o = o + (torch.rand_like(o) * 2 - 1) * half
    if hist is None:
      hist = o[:, None].repeat(1, H, 1)
    else:
      hist = torch.cat([hist[:, 1:], o[:, None]], dim=1)
    # term-major flatten
    parts, i0 = [], 0
    for w in tw:
      parts.append(hist[:, :, i0:i0 + w].reshape(a.envs, -1))
      i0 += w
    g = om.compute_group("goal")[:, keep]
    with torch.no_grad():
      act = pi(torch.cat(parts + [g], dim=-1))
    _, _, dones, _ = env.step(act)
    live = ~done
    err_b += live * torch.norm(cmd.body_pos_w - cmd.robot_body_pos_w, dim=-1).mean(-1)
    err_j += live * torch.norm(cmd.joint_pos - U.scene["robot"].data.joint_pos, dim=-1)
    n += live
    term_fail = torch.zeros_like(done)
    for name in ("anchor_pos", "anchor_ori"):
      if name in tm.active_terms:
        term_fail |= tm.get_term(name).bool()
    newly = dones.bool() & live
    failed |= newly & term_fail
    done |= newly
    if bool(done.all()):
      break
    # envs that finished keep running (auto-reset) but are ignored; reset history for them
    if newly.any():
      hist[newly] = o[newly][:, None].repeat(1, H, 1)
  for e in range(a.envs):
    c = motion.motion_names[int(start_clip[e])]
    d = res["per_clip"].setdefault(c, [0, 0])
    d[0] += int(not failed[e])
    d[1] += 1
  res["ok"] += int((~failed).sum())
  res["fail"] += int(failed.sum())
  res["body_pos"].append(float((err_b / n.clamp(min=1)).mean()))
  res["joint_pos"].append(float((err_j / n.clamp(min=1)).mean()))
  print(f"round {rnd}: success {(~failed).float().mean():.4f}, body pos err {res['body_pos'][-1]:.4f} m", flush=True)
tot = res["ok"] + res["fail"]
summary = {"episodes": tot, "success_rate": res["ok"] / tot, "body_pos_err_m": float(np.mean(res["body_pos"])),
           "joint_pos_err_rad": float(np.mean(res["joint_pos"])),
           "failed_clips": sorted([c for c, (o, t) in res["per_clip"].items() if o < t])}
print("SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "failed_clips"}), "n_failed_clips", len(summary["failed_clips"]))
if a.out:
  json.dump(summary, open(a.out, "w"), indent=1)
