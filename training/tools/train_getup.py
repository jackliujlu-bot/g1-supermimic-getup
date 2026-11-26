"""Train the G1 get-up policy with mjlab's RSL-RL PPO: same CLI as mjlab/scripts/train.py.

Example: CUDA_VISIBLE_DEVICES=2 python train_getup.py G1-Getup --env.scene.num-envs 4096
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "getup"))
import g1getup  # noqa: E402,F401  (registers G1-Getup)
from mjlab.scripts.train import main  # noqa: E402

if __name__ == "__main__":
  main()
