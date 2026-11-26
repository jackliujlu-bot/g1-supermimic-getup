"""Parity: TrackerPolicy + RefTimeline (reference as input) vs the per-clip DancePolicy ONNX."""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim2sim as S
from g1_stack import DancePolicy
from unified import RefTimeline, TrackerPolicy, load_clip

clip, dance_onnx, tracker_onnx = sys.argv[1:4]
sim = S.Sim(np.random.default_rng(0))
qpos, _ = S.initial_pose("standing", np.random.default_rng(0))
sim.set_state(qpos)
dp, tp, tl = DancePolicy(dance_onnx), TrackerPolicy(tracker_onnx), RefTimeline()
tl.reset_clip(load_clip(clip))
s = sim.state()
dp.start(s); tp.start(s, tl)
d = []
for k in range(300):
  s = sim.state()
  a1 = dp.act(s); a2 = tp.act(s, tl); tl.advance()
  d.append(np.abs(a1 - a2).max())
  sim.apply(a1)
print(f"parity max |target diff| over {len(d)} steps: {max(d):.2e} (frames {dp.num_frames} vs {len(tl.q)})")
