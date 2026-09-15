# Intent Alignment Text Toy

一个用于结题快速验证的、可复现的文字交互 toy 环境。它测试 Qwen3.5-4B 能否把中文位移意图转换成逐步动作，并在不超过5步的动作预算内到达目标。

示例：

```text
用户意图：在5步以内向右走3米。
Agent: RIGHT  -> 环境坐标 (1, 0)
Agent: RIGHT  -> 环境坐标 (2, 0)
Agent: RIGHT  -> 环境坐标 (3, 0)，成功并自动结束
```

## 冻结口径

- 二维文字网格，起点 `(0, 0)`。
- 每个 `UP / DOWN / LEFT / RIGHT` 动作移动1米；分别对应 `(0,+1) / (0,-1) / (-1,0) / (+1,0)`。
- 20道固定测试题，动作预算均不超过5步。
- 到达目标即成功；非法输出或预算耗尽仍未到达则失败。
- 样本成功率达到95%以上等价于至少 `19/20`。
- 使用 Qwen3.5-4B zero-shot greedy decoding，不做训练或微调。

完整协议见 [`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md)。

## 实测结果

Qwen3.5-4B 在单张 NVIDIA B300 上按冻结 v1.1 协议取得 **19/20（95%）**，达到“95%以上”的样本门槛。完整逐题结果与首轮18/20记录见 [`docs/RESULTS.md`](docs/RESULTS.md)。

## 本地验证环境

核心环境只依赖 Python 标准库：

```bash
python3 -m pip install -e '.[test]'
pytest -q
python3 scripts/run_oracle.py
```

## Qwen3.5-4B 单卡评测

需要支持 Qwen3.5 的 `torch + transformers` 运行环境：

```bash
python3 -m pip install -e .
python3 scripts/run_qwen.py \
  --model /path/to/Qwen3.5-4B-text-only \
  --output results/qwen3.5-4b-zero-shot.json
```

结果 JSON 会保存冻结数据哈希、模型配置哈希、运行时版本、每题原始输出、动作轨迹、终点和汇总成功率。

## 结论边界

这是20题的功能性 pilot，不是 Minecraft 仿真，也不等价于合同中的相对“认知偏差”收敛证明。即使样本为20/20，也只应写作“在该冻结20题 toy 测试集上成功率为100%”。
