# 数据说明：从哪来、什么格式、在哪里用

本仓库不含动作数据（许可原因见 README）。这份文档说明数据怎么得到、怎么处理、格式是什么，以及训练、评测、部署各用了哪些数据。

## 1. 总览

```
BONES-SEED  g1/csv（142,220 段，120 Hz）
  │ convert_bones.py   120→50 Hz、cm→m、外旋 xyz 欧拉角→四元数、度→弧度；算每段统计
  ▼
data/bones50/clips/<日期>/<片段名>.npz   +  stats.csv
  │ filter_bones.py    去掉靠道具支撑、关节 > 35 rad/s、短于 1 s 的片段
  ▼
data/bones50/keep.csv（134,603 段 / 274.5 h）
  │ fix_and_pack.py    修正穿地，拼接成一个大数组
  ▼
data/pack_main/  ─────────────────────────────►  基线追踪器（作者默认任务）
  │
  │ pack_finetune.py   在末尾追加 2,800 段起身动作副本（重复采样）
  ▼
data/pack_v3/（137,403 段 / 281 h）────────────►  从零训练 → v7 → v8 → v10

LAFAN1  G1 重定向 CSV（30 Hz，fallAndGetUp*）
  │ convert_lafan.py   30→50 Hz，切出所有起身片段并分类（仰卧/俯卧/侧卧、是否手撑地）
  ▼
data/lafan50/full/*.npz + segments.csv
  │ make_lafan_getups.py  切出选定的两段，修正穿地，末尾加 1 s 站立，另存放慢 1.25 倍版本
  ▼
data/lafan_getups/lafan_{prone,supine}[_x1.25].npz
  ├─► pack_v3 里的起身副本（各 600 份）
  ├─► SETTLE 策略训练的目标姿势（第一帧）
  └─► 部署：deploy/motions/lafan_{prone,supine}.npz
```

## 2. 怎么拿到数据

**方法 A：自己生成。** 公开仓库的读者只能用这个方法。

1. 在 Hugging Face 同意 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) 的条款，下载 `g1.tar.gz`（约 23 GB）和 `metadata/seed_metadata_v004.csv`，解压出 `g1/csv/`。
2. 获取 LAFAN1 重定向到 G1 的 CSV（`fallAndGetUp1_subject1/4/5`、`fallAndGetUp2_subject2/3`、`fallAndGetUp3_subject1`），放到 `data/lafan_g1/`。
3. 按 [TRAINING.md](TRAINING.md) 第 1 步运行脚本（全量转换时我们用了 48 个进程）。

**方法 B：从私有数据仓库下载。** 只适用于有权限的人。

```bash
gh release download v10 -R <私有数据仓库> -p 'pack_v3*' -p 'SHA256SUMS_pack_v3'
sha256sum -c SHA256SUMS_pack_v3
cat pack_v3.tar.zst.part* | zstd -d | tar x          # -> pack_v3/（14 GB）
python training/tools/slice_pack.py pack_v3 pack_main   # pack_main = 前 134,603 段
```

起身动作在私有数据仓库的 `data/lafan_getups/` 和 `motions/` 里。

## 3. 格式

### 单个片段（`.npz`）

转换后的片段、起身动作、部署用的舞蹈都是这个格式。字段是 mjlab `MotionLoader` 使用的键：

| 键 | 形状 | 含义 |
|---|---|---|
| `fps` | 标量 | 50 |
| `root_pos_w` | (T, 3) | 骨盆位置，米，世界坐标 |
| `root_quat_w` | (T, 4) | 骨盆姿态四元数，**wxyz** 顺序，相邻帧保持同一半球 |
| `root_lin_vel_w` / `root_ang_vel_w` | (T, 3) | 骨盆线速度 / 角速度（世界坐标） |
| `joint_pos` / `joint_vel` | (T, 29) | 关节角（弧度）/ 关节速度，顺序见下 |
| `joint_names` | (29,) | 关节名 |

29 个关节的顺序（打包数据、追踪器、部署代码全都用这个顺序）：

```
 0-5   左腿  hip_pitch hip_roll hip_yaw knee ankle_pitch ankle_roll
 6-11  右腿  同上
12-14  腰    waist_yaw waist_roll waist_pitch
15-21  左臂  shoulder_pitch shoulder_roll shoulder_yaw elbow wrist_roll wrist_pitch wrist_yaw
22-28  右臂  同上
```

### 打包数据（`pack_main/`、`pack_v3/`）

所有片段按顺序首尾拼接，每个字段是一个大数组：

| 文件 | 内容 |
|---|---|
| `joint_pos.npy`、`joint_vel.npy` | (总帧数, 29) float32 |
| `root_pos_w.npy`、`root_quat_w.npy`、`root_lin_vel_w.npy`、`root_ang_vel_w.npy` | (总帧数, 3 或 4) |
| `motion_ends.npy` | (片段数,) int64，每个片段**结束帧的累计位置**：第 i 段是 `[ends[i-1], ends[i])`，第 0 段从 0 开始 |
| `motion_names.json` | 片段名列表，和 `motion_ends` 一一对应 |
| `fps.npy`、`joint_names.json` | 50 和关节名 |
| `lift_stats.csv`（只有 pack_main） | 每段为修正穿地抬高了多少 |

片段名规则：
* BONES：`<动作名>_R_<编号>__A<演员号>`，带 `_M` 后缀的是左右镜像。
* 追加的起身副本：`<原名>__ft0000` … `__ft0599`。

取出第 i 段：

```python
import json, numpy as np
d = "data/pack_v3"
ends = np.load(f"{d}/motion_ends.npy"); names = json.load(open(f"{d}/motion_names.json"))
jp = np.load(f"{d}/joint_pos.npy", mmap_mode="r")      # 14 GB 的包要用 mmap 读
i = names.index("lafan_prone__ft0000")
s, e = (0 if i == 0 else ends[i - 1]), ends[i]
clip_q = jp[s:e]                                         # (T, 29)
print(names[i], (e - s) / 50, "s")
```

## 4. 每一步用了什么数据

| 环节 | 用到的数据 | 怎么用 |
|---|---|---|
| 基线追踪器（11-23） | `pack_main` | `train.py SuperMimic-GenCoord-G1 --motion-file data/pack_main`，所有片段一起训练 |
| v3_scratch / v7 / v8 / v10 | `pack_v3` | `train_v3.sh` 里写死 `--motion-file ../data/pack_v3`。训练时从整个包里抽片段和起始帧，按 SuperMimic 默认的采样方式（日志里的 `sampling_entropy`、`sampling_top1_prob` 就是采样分布的指标）。起身动作各复制 600 份，在 281 h 里约占 2%，但会被频繁抽到 |
| BC 老师（v7–v10） | 同上 | 老师和学生看同一批数据，只多一个模仿老师动作的损失 |
| SETTLE 策略 | `lafan_{prone,supine}.npz` | 环境变量 `G1SETTLE_CLIP` 指定。用**第一帧**的身体朝向和关节角作为目标姿势（奖励 `settle_orient`、`settle_pose`、`settle_still`）。初始状态是随机摔倒和从 BONES 贴地姿势里采样 |
| 舞蹈评测（mjlab） | [training/eval_lists/eval_dances.txt](../training/eval_lists/eval_dances.txt)（94 段） | `eval_tracker.py` 每段从第 0 帧播放到结束，跑 3 轮，统计成功率和关节误差 |
| 混合动作评测 | [training/eval_lists/eval_clips.txt](../training/eval_lists/eval_clips.txt)（494 段） | 同上，早期阶段用 |
| sim2sim 起身评测 | `lafan_{prone,supine}.npz` | `sim2sim.py --unified TRACKER SETTLE_P lafan_prone.npz SETTLE_S lafan_supine.npz` |
| sim2sim 跳舞 + 推倒 | `deploy/motions/` 的 3 支舞 | `--dance motions/<舞>.npz` |
| 部署（真机） | `deploy/motions/lafan_{prone,supine}.npz` | 第一帧 = SETTLE 的目标；整段 = 起身参考动作；`lafan_prone` 最后一帧 = 统一站姿（`G1_STAND_CLIP` 可以改） |

评测清单里的路径是相对工作区 `$W` 的（`data/bones50/clips/...`），使用前要先生成 `data/bones50/clips`，或者把路径改成你的位置。

## 5. 部署需要的 6 个动作文件

| 文件 | 来源 | 用途 |
|---|---|---|
| `lafan_prone.npz` | LAFAN1 `fallAndGetUp2_subject2` 28.0–35.0 s | 俯卧起身；最后一帧是站姿 |
| `lafan_supine.npz` | LAFAN1 `fallAndGetUp2_subject2` 12.0–18.6 s | 仰卧起身 |
| `dance_hiphop_hip_hop_iv_R_001__A320.npz` | BONES-SEED | 测试舞蹈 |
| `dance_latino_jive_sequence_R_002__A312.npz` | BONES-SEED | 测试舞蹈 |
| `victory_dance_macarena_change_direction_R_001__A321.npz` | BONES-SEED | 测试舞蹈 |
| `getup_supine.npz` | BONES-SEED 起身片段 | 旧站姿（已不用）；pack_v3 里有 400 份副本 |

只测起身的话，只需要前两个。

## 6. 注意事项

* `convert_bones.py` 的欧拉角顺序必须是外旋 `xyz`，用别的顺序时跳跃片段会漂起来或陷进地里（见 [PITFALLS.md](PITFALLS.md) 第 2 条）。
* 打包后的片段已经修正过穿地。自己新加片段时也要先跑 `fix_and_pack.py`，或者用 `make_lafan_getups.py` 里的同一套方法。
* `pack_finetune.py` 会检查关节顺序和原包是否一致，不一致会直接报错。
* 新动作上真机前，先用 `clip_track_test.py` 在仿真里跟踪一遍，再跑 sim2sim。
