"""Plot the main training curves of the v10 lineage from parse_train_log.py CSVs.

Usage: python plot_curves.py CSV_DIR OUT.png
Runs are chained on one x axis (cumulative iterations): v3_scratch -> v7 -> v8 (to 1214) -> v10.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

d = sys.argv[1]
chain = [("v3_scratch", None), ("v7_delay_still", None), ("v8_long", 1214), ("v10", None)]
side = [("v9_umax2", "v8_long", 1214)]
metrics = [("Mean reward", "平均回报"), ("Metrics/motion/error_joint_pos", "关节误差 (rad)"),
           ("Metrics/motion/error_body_pos", "身体位置误差 (m)"), ("Mean episode length", "平均回合长度")]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
fig, axes = plt.subplots(len(metrics), 1, figsize=(11, 3 * len(metrics)), sharex=True)
off, starts = 0, {}
for name, stop in chain:
  p = os.path.join(d, f"{name}.csv")
  if not os.path.exists(p):
    continue
  df = pd.read_csv(p)
  if stop:
    df = df[df["iter"] - df["iter"].min() <= stop]
  x = df["iter"] - df["iter"].min() + off
  starts[name] = off
  for ax, (k, lab) in zip(axes, metrics):
    if k in df:
      ax.plot(x, df[k].rolling(20, min_periods=1).mean(), lw=1, label=name)
  off = int(x.max()) + 1
for name, parent, at in side:
  p = os.path.join(d, f"{name}.csv")
  if os.path.exists(p) and parent in starts:
    df = pd.read_csv(p)
    x = df["iter"] - df["iter"].min() + starts[parent] + at
    for ax, (k, lab) in zip(axes, metrics):
      if k in df:
        ax.plot(x, df[k].rolling(20, min_periods=1).mean(), lw=1, ls="--", color="grey", label=f"{name}（弃用）")
for ax, (k, lab) in zip(axes, metrics):
  ax.set_ylabel(lab)
  ax.grid(alpha=0.3)
  for s in starts.values():
    ax.axvline(s, color="k", lw=0.5, alpha=0.3)
axes[0].legend(loc="lower right", fontsize=8)
axes[-1].set_xlabel("累计训练轮数（1 轮 = 10 次 PPO 迭代；竖线 = 换下一阶段）")
fig.suptitle("v10 训练链路：从零训练 → v7（延迟）→ v8（老师）→ v10")
fig.tight_layout()
fig.savefig(sys.argv[2], dpi=110)
print("->", sys.argv[2])
