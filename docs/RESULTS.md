# Qwen3.5-4B 单卡结果

## 结论

- 冻结协议 v1.1：**19/20，样本成功率 95%**。
- 达到本项目预先声明的“95%以上（`>=95%`）”门槛。
- 没有达到“严格大于95%”口径；20题时该口径要求20/20。
- 这是冻结20题上的功能性 pilot，不可外推为总体成功率的95%统计保证。

## 运行配置

- 模型：`Qwen3.5-4B-text-only-v1`，zero-shot，未训练、未微调。
- 硬件：单张 NVIDIA B300 SXM6 AC。
- 解码：greedy，`do_sample=false`，thinking关闭，每轮最多生成8 token。
- 动作：`UP / DOWN / LEFT / RIGHT`，每次固定移动1米。
- 主指标：在题目给定的、最多5步的预算内到达目标坐标。
- 数据 SHA256：`a09e7c2e3c9249125d7b610fe326fc3b09bf9618b144e2f7d54a51e267b0d75e`。
- 执行源码提交：`3eaafbb52d93c6a014fc8edf98cba3ab070daaac`。

## 分组结果

### 按方向

| 方向 | 成功/总数 |
|---|---:|
| 右 | 5/5 |
| 左 | 5/5 |
| 上 | 5/5 |
| 下 | 4/5 |

### 按最短动作长度

| 距离/最短步数 | 成功/总数 |
|---:|---:|
| 1 | 4/4 |
| 2 | 4/4 |
| 3 | 4/4 |
| 4 | 3/4 |
| 5 | 4/4 |

## 唯一失败样本

- `down-4`：动作次数不超过4次，请到达下方4米处。
  - 目标：`(0, -4)`，预算：4步。
  - 输出：`DOWN -> UP -> DOWN -> DOWN`。
  - 终点：`(0, -2)`，因预算耗尽失败。

## 协议修订透明度

- 首轮 v1 为18/20。prompt 显示了坐标，却没有声明方向对应的坐标正负号。
- v1.1 只补充通用映射 `UP=(0,+1), DOWN=(0,-1), LEFT=(-1,0), RIGHT=(+1,0)`；数据、模型、解码、预算和评分不变。
- v1 原始逐题结果仍保留在 `results/qwen3.5-4b-zero-shot-v1-under-specified-coordinates.json`，没有用 v1.1 覆盖。

## 证据文件

- `results/qwen3.5-4b-zero-shot-v1.1.json`：最终逐题原始输出、动作与轨迹。
- `results/qwen3.5-4b-zero-shot-v1.1.status.json`：执行提交与持久化结果哈希。
- `results/qwen3.5-4b-zero-shot-v1-under-specified-coordinates.json`：首轮18/20结果。
- `results/oracle.json`：环境和评分链的20/20 oracle 自检。
