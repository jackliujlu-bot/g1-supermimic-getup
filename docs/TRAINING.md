# 复现步骤

所有命令都是原样记录的。脚本里的工作区路径写成了 `W=/path/to/workspace`，复现时改成你自己的目录。
硬件：一张 48 GB 显卡（与他人共用），4096 个并行环境。日志里的"1 轮" = 10 次 PPO 迭代（`--runner.max-iterations 100000` 显示为 `/10000`）。

## 0. 环境

```bash
mkdir ws && cd ws
git clone https://github.com/HybridRobotics/supermimic.git && git -C supermimic checkout 5fb9229
git clone https://github.com/HybridRobotics/mjlab_research.git && git -C mjlab_research checkout 379c55e
cd supermimic && uv sync        # 两个仓库都不需要修改
# 本仓库的 training/tools 放到 $W/tools，training/getup 放到 $W/getup
```

## 1. 数据

### BONES-SEED（约 274.5 h）

```bash
# 120 Hz CSV -> 50 Hz npz，同时算每段的统计量（最低点、支撑时长、关节速度……）
python -I tools/convert_bones.py --csv-root <BONES-SEED>/g1/csv --out data/bones50 --workers 48
# 过滤：去掉靠道具支撑、关节 > 35 rad/s、短于 1 s 的片段 -> 134,603 段 / 274.5 h
python tools/filter_bones.py data/bones50/stats.csv <BONES-SEED>/metadata/seed_metadata_v004.csv data/bones50/keep.csv
# 修正穿地并打包
python -I tools/fix_and_pack.py --clips data/bones50/clips --list data/bones50/keep.csv --out data/pack_main --workers 32
```

### LAFAN1 起身动作

```bash
python tools/convert_lafan.py data/lafan_g1 data/lafan50          # G1 重定向 CSV -> 50 Hz，列出所有起身片段
python tools/make_lafan_getups.py data/lafan50/full data/lafan_getups
#   -> lafan_prone.npz、lafan_supine.npz，以及放慢 1.25 倍的 *_x1.25.npz
```

### 训练用动作包 pack_v3（137,403 段 / 281 h）

```bash
L=data/lafan_getups
python -I tools/pack_finetune.py --base data/pack_main --out data/pack_v3 \
  --clip $L/lafan_prone.npz:600 --clip $L/lafan_prone_x1.25.npz:600 \
  --clip $L/lafan_supine.npz:600 --clip $L/lafan_supine_x1.25.npz:600 \
  --clip deploy/motions/getup_supine.npz:400
```

## 2. 追踪器训练（v10 链路）

`tools/train_v3.sh NAME GPU MAX_ITERS [额外参数]` 会启动 `finetune_tracker.py`（任务 `SuperMimic-GenCoord-G1-Robust`，4096 环境，`pack_v3`，本体感觉观测 + 10 帧历史），并在每次保存 checkpoint 时拷贝一份到 `snapshots/NAME/`。默认 `G1R_TERRAIN=1`。

训练环境变量（`tools/robust_task.py`）：

| 变量 | 含义 | 默认 |
|---|---|---|
| `G1R_TERRAIN=1` | 不平地面（45% 平地，其余 1–2.5 cm 起伏） | train_v3.sh 里为 1 |
| `G1R_HARD_PROB` | 硬地比例，其余是软垫 | 0.35 |
| `G1R_UMAX` | 软垫最大软度（**不要超过 1**，v9 用 2 时约 28% 摔倒的机器人会穿地） | 1.0 |
| `G1R_DELAY=N` | 控制延迟 0..N 个物理步（每步 5 ms） | 关 |
| `G1R_STILL=w` | 参考动作静止时的关节速度惩罚 | 关 |
| `G1R_ACTION_RATE=w` | 动作变化率惩罚权重 | -0.1 |

```bash
cd $W
# v3_scratch：从零训练，10,000 轮，11-25 11:36 -> 20:19（8 h 43 min）
tools/train_v3.sh v3_scratch 0 100000

# v7：+ 0-20 ms 延迟 + 站立防抖 + 自己做 BC 老师，800 轮（26 min）
S=$PWD/snapshots/v3_scratch/final
G1R_DELAY=4 G1R_STILL=-0.05 tools/train_v3.sh v7_delay_still 0 8000 \
  --init $S --init-critic $S/critic.pt --teacher $S --bc-weight 0.1

# v8：基线追踪器做 BC 老师，跑到第 1214 轮（39 min）后停掉
S=$PWD/snapshots/v7_delay_still/final
G1R_DELAY=4 G1R_STILL=-0.05 tools/train_v3.sh v8_long 0 30000 \
  --init $S --init-critic $S/critic.pt --teacher $PWD/final/tracker --bc-weight 0.05

# v10：从 v8 第 1214 轮接着练，3000 轮（1 h 37 min）——最终版
S=$PWD/snapshots/v8_long/1125_2307_log1214
G1R_DELAY=4 G1R_STILL=-0.05 tools/train_v3.sh v10 0 30000 \
  --init $S --init-critic $S/critic.pt --teacher $PWD/final/tracker --bc-weight 0.05

# 导出 ONNX（参考动作作为输入，可以把起身、站立、跳舞拼成一条时间线）
python tools/export_tracker.py snapshots/v10/final data/lafan_getups/lafan_prone.npz exports/release_v10/tracker.onnx
```

基线 `final/tracker`：11-23 用作者默认任务 `SuperMimic-GenCoord-G1` 在 `pack_main` 上训练，约 3.3–3.8 h。

## 3. SETTLE 策略（每种起身动作一个）

```bash
for n in prone supine; do
  # 阶段 A：硬地，4500 轮
  G1SETTLE_CLIP=$W/data/lafan_getups/lafan_$n.npz WANDB_MODE=offline \
    python -u tools/train_getup.py G1-Settle --env.scene.num-envs 4096 --agent.max-iterations 4500
  # 阶段 B：软垫 + 不平地面，从 model_4499 接着练 2000 轮 -> model_6498
  G1SETTLE_SOFT=1 G1SETTLE_TERRAIN=1 G1SETTLE_CLIP=$W/data/lafan_getups/lafan_$n.npz WANDB_MODE=offline \
    python -u tools/train_getup.py G1-Settle --env.scene.num-envs 4096 --agent.max-iterations 2000 \
    --agent.resume True --agent.load-run <阶段A的run目录> --agent.load-checkpoint model_4499.pt
  python tools/export_getup.py logs/rsl_rl/g1_getup/<run>/model_6498.pt exports/settle2_lafan_$n
done
```

## 4. 评测

```bash
# 完整评测：94 支舞（mjlab）+ CPU MuJoCo 里 4 种地面 × 5 种躺姿 × 6 次 + 3 支舞 × 3 种地面的跳舞+推倒
# 留出测试：S2S_SEED_BASE=5000 S2S_VARY_TERRAIN=1
S2S_SEED_BASE=5000 S2S_VARY_TERRAIN=1 \
SETTLE_PRONE=$W/exports/settle2_lafan_prone/getup.onnx SETTLE_SUPINE=$W/exports/settle2_lafan_supine/getup.onnx \
  bash tools/eval_v3.sh snapshots/v10/final exports/ckeval/full_v10_final 0
python tools/swing_summary.py exports/ckeval/full_v10_final     # 站立摆动幅度

# 只看追踪器跟踪起身动作和站立抖动
bash tools/clip_eval.sh exports/ckeval/v10/tracker.onnx exports/ckeval/v10
# 渲染演示视频
MUJOCO_GL=egl python deploy/render_demo.py out.mp4 TRACKER SETTLE_P CLIP_P SETTLE_S CLIP_S \
  --cells "仰卧·硬地:supine:hard:5001" "俯卧·软垫:prone:soft:5002"
```

## 5. 版本历史

| 版本 | 时间 | 内容 | 结果 |
|---|---|---|---|
| base（main） | 11-23 | 作者默认配置，BONES-SEED | 跳舞最准（0.66），起身和软垫不行 |
| ft3000 / ft_v2 | 11-25 上午 | base + 软垫等随机化微调 | 起身 87/96 |
| v3_ft / **v3_scratch** | 11-25 11:36 | pack_v3：微调 / **从零** | 从零版起身 96/96，作为后续起点 |
| v4_ft | 11-25 19:42 | 加不平地面微调 | 变差，淘汰 |
| v5_ft | 11-25 20:19 | ft3000 平地微调 | 未评测（已有更好的路线） |
| v6_still | 11-25 21:40 | 只加防抖惩罚（150 轮） | 无效 |
| v7 | 11-25 21:55 | + 0–20 ms 延迟 | 软垫好、硬地差 |
| v8 | 11-25 22:29 | + 基线做老师 | 换站姿后四种地面都过关 |
| v9_umax2 | 11-25 23:10 | 软垫范围 ×2 | 软垫 0/30，停掉 |
| **v10** | 11-25 23:56 | 接 v8 继续 | 最终版 |

部署端的关键修改（`deploy/unified.py`，2026-11-25/26）：

1. SETTLE 到位后改用 PD 保持姿势再交给追踪器（软垫上 settle 策略一直晃，导致超时）。
2. 站姿改成 LAFAN 俯卧起身的最后一帧（BONES 站姿在软垫上抖 16°）。
3. 失败后换另一套起身路线重试，最多 4 次。
4. 评测时限从 25 s 放宽到 40 s（重试需要时间）。
