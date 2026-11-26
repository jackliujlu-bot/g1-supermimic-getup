"""Time env.step of the SuperMimic G1 env. Usage: bench_env.py MOTION_GLOB NUM_ENVS [STEPS]"""
import sys
import time

import mjlab.tasks  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

import supermimic.tasks  # noqa: F401
from supermimic.scripts.train_utils import create_env

motion, n = sys.argv[1], int(sys.argv[2])
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 50
cfg = load_env_cfg("SuperMimic-GenCoord-G1", play=False)
cfg.commands["motion"].motion_file = motion
t0 = time.time()
env = create_env(cfg, n, "cuda:0")
print(f"build {time.time()-t0:.1f}s; sim dt={cfg.sim.mujoco.timestep} decim={cfg.decimation} "
      f"iters={cfg.sim.mujoco.iterations} ls={cfg.sim.mujoco.ls_iterations} "
      f"nconmax={getattr(cfg.sim,'nconmax',None)} njmax={getattr(cfg.sim,'njmax',None)}", flush=True)
env.reset()
act = torch.zeros(n, env.num_actions, device="cuda:0")
for phase in ("warm", "timed"):
  torch.cuda.synchronize()
  t0 = time.time()
  for _ in range(steps if phase == "timed" else 10):
    env.step(act)
  torch.cuda.synchronize()
  dt = time.time() - t0
  if phase == "timed":
    print(f"{n} envs: {dt/steps*1000:.1f} ms/step, {n*steps/dt:.0f} steps/s", flush=True)
