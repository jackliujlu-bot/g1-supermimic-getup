# G1 部署：SuperMimic 舞蹈跟踪 + 真人式摔倒起身

本目录包含上真机需要的全部文件，不依赖训练代码。可以直接拷到 G1 的机载电脑上运行，也可以拷到用网线连接 G1 的外部电脑上运行。

## 推荐方案：全程跟踪器（`--unified`）

起身、站立、跳舞都由同一个 SuperMimic 跟踪器完成，正常流程里不切换网络，只更换参考动作。

| 文件 | 作用 |
|---|---|
| `unified.py` | 参考动作时间线（起身片段、站立、过渡、舞蹈拼接成连续的一条）和状态机 |
| `g1_stack.py` | 观测历史、ONNX 策略封装、旧方案的状态机（两种方案共用这份代码） |
| `run_g1.py` | 真机主程序（unitree_sdk2py 低层接口，50 Hz），带安全层；`--backend sim` 可以在 MuJoCo 里空跑同一套控制循环 |
| `sim2sim.py` / `hold_test.py` / `switch_test.py` | 验收测试（CPU MuJoCo）。仿真里加了噪声、1 拍延迟、PD 增益 ±10%、摩擦随机和电机转矩-转速曲线 |
| `policies/unified/tracker.onnx` | SuperMimic 跟踪器，参考动作作为输入。权重和跳舞用的完全相同 |
| `policies/unified/settle.onnx` | 归位策略：从任意倒地姿势调整到真人起身的起始躺姿 |
| `policies/getup.onnx` | v3 起身策略，只在失败时兜底 |
| `motions/getup_supine.npz` | 真人仰卧起身动作（BONES-SEED，慢放 1.6 倍，从屈膝仰躺开始），最后一帧就是站立姿势 |
| `motions/*.npz` | 舞蹈动作。新增舞蹈只需要放一个 `.npz` 文件，不用重新导出 ONNX |

### 运行逻辑

```
上电 ─► 阻尼（电机放软）─按 g / 遥控 START─►
   躺着 ─► 归位（settle.onnx，约 2 s）─► 跟踪器照真人起身（约 6 s）─► 跟踪器站立
   站着 / 坐着 ─► v3 先站稳 ─► 用 1 s 过渡到跟踪器站立
跟踪器站立，站稳 ≥ 1 s 后按 1..9 / 遥控 R1
   ─► 1 s 过渡到舞蹈第一帧 ─► 跳舞 ─► 1 s 过渡回站立（按 s / 遥控 L1 可以中途停止）
跳舞或站立中摔倒（躯干倾角 > 50°，连续 3 拍）─► 归位 ─► 照真人起身 ─► 站立
兜底：归位超过 8 s、跟踪偏离参考动作太大 ─► v3 起身 ─► 站稳后回到跟踪器
任意时刻：q / x / 遥控 L2+B ─► 阻尼
```

### 吊架上测试

```bash
python run_g1.py --backend real --iface enp3s0 --getup policies/getup.onnx \
  --unified policies/unified/tracker.onnx policies/unified/settle.onnx motions/getup_supine.npz \
  --dance motions/victory_dance_macarena_change_direction_R_001__A321.npz \
          motions/dance_hiphop_hip_hop_iv_R_001__A320.npz \
          motions/dance_latino_jive_sequence_R_002__A312.npz
```

不连机器人、先在仿真里把整套流程走一遍：

```bash
python run_g1.py --backend sim --sim-start supine --getup policies/getup.onnx \
  --unified policies/unified/tracker.onnx policies/unified/settle.onnx motions/getup_supine.npz \
  --dance motions/victory_dance_macarena_change_direction_R_001__A321.npz \
  --script "0:g,16:1" --duration 45 --diag
```

### 新舞蹈上机前

```bash
python sim2sim.py --getup policies/getup.onnx \
  --unified policies/unified/tracker.onnx policies/unified/settle.onnx motions/getup_supine.npz \
  --human-tilt 180 --scenarios standing --trials 5 --dance motions/<新舞蹈>.npz
```

注意：如果舞蹈里有躯干倾斜超过 50° 的动作（下腰、趴地），会被误判成摔倒。这类舞蹈要先调整摔倒判定再上机。

## 安全层（`run_g1.py` 内，始终开启）

* 启动时先进入阻尼模式；接管时 kp 在 1 s 内从 0 线性升到额定值，防止猛冲。
* 策略输出出现 NaN 或无穷 → 阻尼。关节目标只做宽松的异常值裁剪（关节限位外 1.6 rad）。
  * 原因：策略训练时没有裁剪 PD 目标，站立和跳舞时会把脚踝目标推到限位之外一些，以此产生平衡力矩。
  * 关节本身仍然停在机械限位，力矩也受电机限制。
  * 以前按限位内 0.02 rad 裁剪，会让跟踪器站立明显发晃，已修正。
* `rt/lowstate` 超过 0.1 s 没有更新，或控制循环连续 5 拍超时 → 阻尼。
* 起身（任意阶段）连续 30 s（`--getup-timeout`）没有站稳 → 阻尼，不让它反复硬起，保护电机。按 `g` 可以重试。

## 真机步骤

1. 安装依赖：`pip install onnxruntime numpy`，以及 [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)。
2. 电脑网口和 G1 设在同一网段（默认 192.168.123.x），确认 `--iface` 填的网卡名。
3. **第一次务必挂在吊架上**。机器人进入调试 / 零力矩模式（遥控 L2+R2），程序启动时会自动释放高层运控。
4. 吊架测试：输入 `g` 回车，应该缓慢进入站姿。检查没有异响、没有关节抖动后，输入 `x` 回车进入阻尼。
5. 落地站立：吊架放低让脚着地，保留保护绳。`g` → 稳定站立 → `1` 跳舞。
6. 起身测试：让机器人平躺在软垫上（仰卧、俯卧、侧卧都要试），保护绳放松但不脱钩，然后 `g` → 自动起身。
7. 摔倒恢复：跳舞过程中由保护人员用保护绳轻拉使它失衡，确认能自动起身。

## 旧方案（仍然可用）

不加 `--unified` 时就是原来的方案：v3 起身和站立，每支舞一个 ONNX（`policies/dance_*.onnx`）。
加 `--human SETTLE TRACK CLIP` 后，仰躺时会改走真人式起身。

```bash
python run_g1.py --backend real --iface enp3s0 --getup policies/getup.onnx --dance policies/dance_macarena.onnx
```

旧方案导出新舞蹈的方法：在训练服务器上运行 `python tools/export_dance.py <检查点目录> <clip.npz> policies/dance_xxx.onnx`。
