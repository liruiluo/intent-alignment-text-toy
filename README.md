# Intent Alignment Text Toy

一个用于结题快速验证的、可复现的文字交互 Agent 环境。它测试 Qwen3.5-4B 能否把中文位移意图转换成逐步动作，并在不超过 5 步的动作预算内到达目标。

```text
用户意图：在5步以内向右走3米。
Agent: RIGHT  -> 环境坐标 (1, 0)
Agent: RIGHT  -> 环境坐标 (2, 0)
Agent: RIGHT  -> 环境坐标 (3, 0)，成功并自动结束
```

## Dense SPHERE 三臂单卡实测（2026-09-17）

在 veRL `v0.9.0`（commit `483b8a009ba3a97563edee3a19887e4862b8094a`）上，已完成 SPHERE 及两个消融实验。三个实验臂复用同一张现有 B300 顺序运行，每臂都只训练 **1 个 optimizer/global update**：

| 实验臂 | 作用层 / α | 训练采样 | 更新数 | 训练后 test100 |
|---|---|---:|---:|---:|
| SPHERE | 最后一层 / 自适应 α | 800 trajectories | **1** | **100/100** |
| All hidden expert layers | 全部 32 层 / 自适应 α | 800 trajectories | **1** | **100/100** |
| 固定 α 消融 | 最后一层 / `α=1` | 800 trajectories | **1** | **100/100** |

三臂的 worker evidence、梯度组装、checkpoint、参数变化和 pipeline exit 均通过审计；没有申请新卡。训练前已经是 100/100，所以这里证明的是三种实现真实执行并保持能力，不是单步训练提点，也不是三者优劣排序。

- 实验协议：[`docs/SPHERE_ONE_STEP_PROTOCOL.md`](docs/SPHERE_ONE_STEP_PROTOCOL.md)
- 三臂结果与轻量证据：[`results/qwen3.5-4b-sphere-one-step-v1/`](results/qwen3.5-4b-sphere-one-step-v1/)
- 一键核验：`PYTHONPATH=src python3 scripts/verify_sphere_published_results.py`
- Windows 单文件演示：[`showcase/intent-alignment-showcase.html`](showcase/intent-alignment-showcase.html)

## v2 单卡实测结果

| 指标 | 结果 |
|---|---:|
| 训练数据 | 400 条 |
| 独立测试数据 | 100 条 |
| 训练前 test100 | **100 / 100（100%）** |
| veRL 训练 | **1 个 GRPO optimizer/global update** |
| 训练采样 | 800 trajectories；400 组 × 2 |
| 有组内奖励差异的训练组 | 66 |
| Actor gradient norm | 6.470837 |
| 训练后 test100 | **100 / 100（100%）** |
| 成功率变化 | **0 个百分点** |

结论：Qwen3.5-4B 在冻结 test100 上训练前已达到满分；完成一次真实参数更新后仍保持 100%。这次实验验证了单卡 veRL AgentLoop 的训练链和能力保持，**没有证明单步训练提高了成功率**。

- 完整结果说明：[`docs/RESULTS.md`](docs/RESULTS.md)
- 冻结协议：[`docs/VERL_ONE_STEP_PROTOCOL.md`](docs/VERL_ONE_STEP_PROTOCOL.md)
- 可审计结果：[`results/qwen3.5-4b-verl-one-step-v2/`](results/qwen3.5-4b-verl-one-step-v2/)
- Windows 单文件演示：[`showcase/intent-alignment-showcase.html`](showcase/intent-alignment-showcase.html)

## 实验设计

- 动作空间：`UP / DOWN / LEFT / RIGHT`，每个动作固定移动 1 米。
- 数据：400 条训练题、100 条独立测试题；训练和测试的中文模板族不重合。
- 语义覆盖：4 个方向 × 1–5 米，共 20 个语义格；每题满足 `distance <= max_steps <= 5`。
- Agent：模型每轮只输出一个动作，环境执行并返回新坐标，再进入下一轮。
- 模型：完整 Qwen3.5-4B，thinking 关闭。
- 训练：veRL v0.9.0（commit `483b8a009ba3a97563edee3a19887e4862b8094a`）原生 `AgentLoopBase`，单张 NVIDIA B300，GRPO。
- 评测：训练前后都在同一冻结 test100 上做 greedy paired evaluation；成功门槛为至少 `95/100`。

## 快速自检

核心环境只依赖 Python 标准库；测试额外需要 pytest：

```bash
python3 -m pip install -e '.[test]'
pytest -q
python3 scripts/run_oracle.py \
  --dataset data/intent_alignment_test_100.jsonl
```

生成冻结数据与 veRL parquet：

```bash
PYTHONPATH=src python3 scripts/generate_dataset.py --output-dir data
PYTHONPATH=src python3 scripts/prepare_verl_data.py \
  --output-dir /path/to/generated-data
```

Qwen3.5-4B 单卡评测：

```bash
PYTHONPATH=src python3 scripts/run_qwen.py \
  --model /path/to/Qwen3.5-4B \
  --dataset data/intent_alignment_test_100.jsonl \
  --prompt-contract v2-agent-loop \
  --batch-size 100 \
  --max-new-tokens 8 \
  --output /path/to/test100.json
```

单步 veRL 训练入口：

```bash
VERL_ROOT=/path/to/verl-v0.9.0 \
MODEL_PATH=/path/to/Qwen3.5-4B \
TRAIN_PARQUET=/path/to/train.parquet \
TEST_PARQUET=/path/to/test.parquet \
RUN_DIR=/path/to/run \
PYTHON_BIN=/path/to/python \
CUDA_HOME=/path/to/cuda-13 \
  bash scripts/run_verl_one_step.sh
```

训练入口会保存 veRL resume shard，并额外保存可直接评测的 Hugging Face 权重到 `global_step_1/actor/huggingface`。veRL 可能把原始分片 BF16 checkpoint 导出为单文件 FP32 `model.safetensors`；参数审计按 tensor key 和数值比较，不把序列化文件名变化当成学习证据。

公开结果包还包含完整的 800 行 rollout ledger、去除机器本地路径的完整训练日志、训练合同/解析配置/launcher 以及 checkpoint 文件清单。`scripts/verify_published_results.py` 会重新回放 200 条评测轨迹、重算 rollout/group 统计、从日志读取实际 update 与梯度，并核对所有公开 SHA256。

```bash
PYTHONPATH=src python3 scripts/audit_parameter_update.py \
  --base-model /path/to/Qwen3.5-4B \
  --trained-model /path/to/global_step_1/actor/huggingface \
  --output /path/to/parameter-evidence.json

PYTHONPATH=src python3 scripts/compare_results.py \
  --before /path/to/before-test100.json \
  --after /path/to/after-test100.json \
  --parameter-evidence /path/to/parameter-evidence.json \
  --training-evidence /path/to/training-evidence.json \
  --rollout /path/to/rollouts-step1.jsonl \
  --training-log /path/to/training-log.redacted.txt \
  --checkpoint-inventory /path/to/checkpoint-inventory.json \
  --run-contract /path/to/run-contract.redacted.json \
  --resolved-config /path/to/resolved-config.redacted.yaml \
  --launcher /path/to/launch.redacted.sh \
  --output /path/to/paired-comparison.json

PYTHONPATH=src python3 scripts/verify_published_results.py
```

## 历史 v1.1 pilot

仓库保留最初 20 题 zero-shot pilot 的 19/20（95%）记录，用于审计提示词修订历史：

- 协议：[`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md)
- 结果：[`results/qwen3.5-4b-zero-shot-v1.1.json`](results/qwen3.5-4b-zero-shot-v1.1.json)

当前 HTML 已升级为 v2 正式结果。它只离线回放真实 test100 轨迹，不会在浏览器中运行模型。

## 结论边界

这是文字 toy 环境，不是 Minecraft 仿真。test100 成功率只描述冻结测试集，不能外推为所有中文表达上的总体成功率。训练前已经 100%，因此本轮只能报告能力保持和真实参数更新，不能报告成功率提升。
