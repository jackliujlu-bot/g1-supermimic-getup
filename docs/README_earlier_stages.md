# SuperMimic（单张 4090 从零训练）+ 摔倒自动起身 — G1 部署包

* 真机部署：看 [deploy/README.md](deploy/README.md)。
* 训练服务器上的完整工作区：`gpu4090:/path/to/workspace/`

## 结论

推荐使用“全程跟踪器 + 真人式起身”方案（见下文同名章节）。按 `deploy/README.md` 的流程上真机：先吊架，再落地，再在软垫上测起身。下表是最初方案（v3 起身 + 每支舞一个 ONNX）的验收结果。

| 项目 | 结果 |
|---|---|
| 舞蹈跟踪（SuperMimic，80k 步，单张 4090 约 3.3 h） | 94 支舞 × 3 轮 = 3072 次：**成功率 99.84%**；混合 494 段动作 2048 次：99.3–99.8% |
| 部署链路一致性 | 自写的部署 ONNX 与官方导出器动作差 ≤ 6e-6；起身策略部署观测与训练观测动作差 ≤ 1.3e-6 |
| 起身（sim2sim，60 次） | 仰卧 / 俯卧 / 侧躺 / 坐地 / 随机姿态摔落 / 开机即站立，各 10 次：**60/60**；平均 1.85 s 站稳，最慢 5.4 s |
| 跳舞 + 中途被推倒 + 自动起身（sim2sim） | 3 支舞 × 5 次 × (不推 / 900 N 推 0.2 s)：**30/30**；被推倒的 15 次全部检测到摔倒并在摔倒后约 4–6 s 重新站稳 |
| 控制器空跑（`run_g1.py --backend sim`） | 躺着开机 → 自动起身 → 跳舞 1 → 站 → 跳舞 2 → 站 → 急停阻尼，全流程通过 |

sim2sim 设置：CPU MuJoCo（与训练用的 MuJoCo-Warp 不同实现），使用部署代码本身，带 1 拍动作延迟、IMU/编码器噪声、PD 增益 ±10%、摩擦 0.5–1.1、电机转矩-转速曲线（额定扭矩到 1/3 空载转速后线性降到 0）。

演示视频：[reports/sim2sim_demo.mp4](reports/sim2sim_demo.mp4)（仰卧起身、俯卧起身、跳舞中被推倒再起身）。

## 做了什么

### 1. 数据（BONES-SEED G1）
* 从 `g1.tar.gz` 已解压的 142,220 段 CSV 转换：120 Hz → 50 Hz，根位置 cm→m，根姿态为外旋 xyz 欧拉角（度），关节角度→弧度。欧拉顺序用正运动学验证过：只有 `xyz` 时跳跃片段脚贴地（误差 2 cm），其他顺序漂浮或穿地 15–60 cm。
* 过滤（得到 134,603 段 / 274.5 h，论文为 129.5k 段 / 264.6 h）：去掉需要道具支撑的（站在箱子/梯子/台阶上 > 0.5 s、坐椅子/凳子/床）、关节速度 > 35 rad/s、时长 < 1 s 的片段。**保留**躺、坐地、跪、爬等贴地动作，起身需要它们。
* 穿地修正：用 G1 碰撞几何体（胶囊/球）精确算最低点，逐帧把根抬高（平滑），处理后所有片段穿地 ≤ 5 mm。

### 2. SuperMimic 跟踪器
* 官方代码 `HybridRobotics/supermimic` + `mjlab_research`，未改算法，任务 `SuperMimic-GenCoord-G1`，4096 环境。
* 为能上真机：`--runner.actor-obs proprio --runner.history-length 10`。默认的 `rep_state` 会把仿真全状态喂给 actor，真机拿不到。
* 环境问题：服务器驱动只支持 CUDA 12.9，`uv sync` 默认装的 torch(cu132) / warp(cu13) 会**静默退回 CPU**（慢约 400 倍）。已换成 `torch 2.14.1+cu126` 与 `warp-lang 1.19.0+cu12`。之后**不要再执行 `uv sync`**，否则会被还原。
* 每支舞单独导出 ONNX（`tools/export_dance.py`）：参考动作表已打包进模型，输入只有“本体感知历史 + 骨盆 IMU 四元数 + 帧号”。开始跳舞时用当前朝向对齐参考动作的初始偏航角。

### 3. 起身（自己设计的部分）
* 方法：单独的 PPO 策略（mjlab / rsl_rl），观测与跟踪器相同，都是真机可得的量（重力投影、陀螺仪、关节位置/速度、上一帧动作，5 帧历史），输出关节目标，最终收敛到跟踪器的默认站姿。
* 初始状态：45% 来自 BONES-SEED 真实贴地姿态（躺/坐/跪/爬，共 20 万个），25% 任意姿态从 1 m 高摔下，15% 站立被强推（学习“摔倒过程中”接管），15% 直接站立（开机已站着、或刚从跳舞切过来）。
* 训练分三阶段：
  * v1：前 4000 次迭代对躯干施加逐渐减为 0 的向上助力（课程学习）。
  * v2：电机改为转矩-转速曲线模型，并加强平滑项。
  * v3：加入关节速度（< 65% 空载转速）和扭矩（< 85% 额定）的余量惩罚。最终采用 v3。
* 切换逻辑（`deploy/g1_stack.py`）：躯干倾角 > 50° 连续 3 拍 → 判定摔倒，切到起身策略；只有在站稳 ≥ 1 s 后才允许开始跳舞；开机一律先运行起身策略（躺着就起身，站着就保持站立）。

## 风险与上机注意（诚实版）
* 起身动作偏快偏猛：平均约 2 s 站起，峰值关节速度约为电机空载转速的 66%（最高 92%），约 6% 的控制拍达到扭矩上限。仿真里在限值内，但真机电机发热和电池压降仿真没有建模。**第一次务必在软垫上、带保护绳测试**，并观察电机温度。
* 被推倒时的冲击（地面反力峰值可达体重的 20–35 倍）来自摔倒本身，不是策略造成的；做摔倒测试请铺垫子。
* 跟踪失败最多的类型：hip-hop shuffle 快速步伐、单脚连续跳、坐地起立的个别片段。新舞蹈上机前请先跑 sim2sim。
* sim2sim 仍然是仿真。真机还会遇到 IMU 偏置、关节间隙、接触模型差异等问题。代码里的安全层（阻尼急停、kp 斜坡、看门狗、起身超时）是兜底措施。

## 紧凑起身（中间尝试，已被下文方案取代）

起因：v3 起身时手臂和腿张开幅度大、占地大，窄空间里或表演时周围有人会有危险。

评估：`training/tools/getup_eval.sh`，数据在 `reports/compact_getup/`，对比视频 `reports/compare_v3_vs_v4A.mp4`（左 v3 / 右 v4A）。
* 测试内容：空旷场地、两侧墙过道、四面墙方盒（墙在躺姿表面外 15 / 30 cm）。
* 指标：扫过面积（起身全过程身体投影的凸包）、手臂/腿相对站姿的最大张开角。

| 版本 | 方法 | 成功 | 扫过面积 | 手臂张开 | 腿张开 | 方盒 15 cm 碰墙 |
|---|---|---|---|---|---|---|
| v3 | 当前部署版 | 30/30 | 1.16 m² | 1.76 rad | 1.77 rad | 20% |
| v4A-16000 | + 真人起身姿态先验（BONES-SEED 166 段）+ 张开 / 占地惩罚 | 59/60 | 0.94 m² | 1.38 | 0.99 | 20% |
| v4B | 先验权重 5 | 训练中崩溃（停在地上“摆姿势”刷奖励） | | | | |
| v5 | 站立时也给先验 | 起身变慢，跑动 2.5 m | | | | |
| v6 | + 虚拟墙终止 | 29/30 | 1.13 m² | 1.27 | 1.17 | 20% |
| 直接跟踪真人起身片段 | SuperMimic 跟踪 `stand_up_lying*` | 30/30（标准起始姿态） | — | 更大 | — | 俯卧版关节超速 1.9×，不能用 |

结论：奖励塑形能把关节张开压下 20–45%，但窄空间碰墙率没变。候选策略 `deploy/policies/getup_compact_candidate.onnx`（v4A-16000）在 1 个随机姿态下起不来，所以没有替换 v3。

## 全程跟踪器 + 真人式起身（推荐方案）

用法和文件说明见 [deploy/README.md](deploy/README.md)，全流程视频见 `reports/unified_flow.mp4`。

**结构**：倒地后先运行一个小的归位策略（约 2 秒），之后起身、站立、跳舞都由同一个 SuperMimic 跟踪器完成。
* 跟踪器权重和跳舞用的完全相同，没有改过。
* 区别在导出方式：参考动作改为输入，部署时把真人起身片段、站立姿势、1 秒平滑过渡和舞蹈拼接成一条连续的参考时间线。
* v3 只在失败时兜底。

| 项目（仿真，部署代码本身） | 旧方案（v3 + 每支舞一个 ONNX） | 全程跟踪器 |
|---|---|---|
| 新导出的跟踪器与旧的单舞 ONNX 输出一致性 | — | 最大差 1.5e-6 rad |
| 5 种倒地姿势 × 10 次 + 开机站着 × 10 次 | 60/60 | **60/60**。50 次倒地中，45 次全程真人式起身，另 5 次由 v3 兜底后同样站起 |
| 从躺下到站稳 | 约 2 s | 约 8–11 s（真人约 4.5 s，片段慢放了 1.6 倍） |
| 3 支舞 × 5 次，不推 / 900 N 推倒 | 30/30 | **30/30**（被推倒的 15 次全部自动起身） |
| 切换瞬间关节目标跳变（最大） | 0.98 rad（跳舞结束切回 v3） | **0.29 rad**（正常动作 p99 为 0.67） |
| 站立 60 s，不推 | 不倒，漂移 < 1 cm | 不倒，漂移 1–3 cm |
| 站立 60 s，每 4–6 s 被推一次（300 N，0.1 s） | **5 次里 5 次都倒了**（之后自己起身） | **5/5 不倒** |
| 四面墙方盒（墙在躺姿外 15 cm）重碰墙 | 4/25 | 8/25 |

仍需注意：
* 起身更慢，窄空间里碰墙比 v3 多：归位时要翻身调整到仰躺姿势，比 v3 多占地方。
* 约 1/10 的真人式起身中途跟偏，由 v3 兜底完成。
* 修复了一个旧问题：`run_g1.py` 原来把关节目标裁剪到关节限位内 0.02 rad，削掉了跟踪器站立和跳舞时需要的脚踝力矩。这个问题对旧方案上真机同样存在，只是旧的 sim2sim 测试不经过这层裁剪。

尝试过程（结论以上表为准）：
* 先让 v3 加上真人姿态先验，并对张开幅度和占地扣分（v4A）：张开幅度降低 20–45%，但窄空间碰墙没有改善。
* 再试"归位 → 跟踪真人起身 → v3 站立"三段式：可行，但切换时有突跳，而且站立用的是 v3 的姿势，和舞蹈起始姿势差得较远。
* 最终改为全程跟踪器。
* 俯卧起身片段：数据集里 8 段全部测过，直接跟踪时扭矩打满 25–43%，不能用。所以俯卧时也是先归位成仰躺，再起身。

## 复现（在 gpu4090 上）
```bash
W=/path/to/workspace; cd $W
# 数据
supermimic/.venv/bin/python -I tools/convert_bones.py --csv-root <seed>/g1/csv --out data/bones50 --workers 48
supermimic/.venv/bin/python -I tools/fix_and_pack.py --clips data/bones50/clips --list data/bones50/keep.csv --out data/pack_main
supermimic/.venv/bin/python -I tools/build_floor_pool.py --pack data/pack_main --meta <seed>/metadata/seed_metadata_v004.csv --out data/floor_pool.npz
# 跟踪器
cd supermimic && CUDA_VISIBLE_DEVICES=1 .venv/bin/python src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 \
  --num-envs 4096 --motion-file $W/data/pack_main --runner.actor-obs proprio --runner.history-length 10
# 起身（v1 → v2 → v3，后两个阶段都从上一阶段的检查点 resume）
supermimic/.venv/bin/python tools/train_getup.py G1-Getup --env.scene.num-envs 4096
supermimic/.venv/bin/python tools/train_getup.py G1-Getup-DC        --agent.resume True --agent.load-run <v1> --agent.load-checkpoint model_4000.pt --agent.max-iterations 6000
supermimic/.venv/bin/python tools/train_getup.py G1-Getup-DC-Gentle --agent.resume True --agent.load-run <v2> --agent.load-checkpoint model_5500.pt --agent.max-iterations 5000
```
最终检查点：`final/tracker/`（跟踪器，80k 步）、`final/getup_model_10000.pt`（起身 v3），对应 ONNX 在 `final/policies/`。
