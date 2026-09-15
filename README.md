# Intent Alignment Text Toy

一个用于结题快速验证的、可复现的文字交互 Agent 环境。它测试 Qwen3.5-4B 能否把中文位移意图转换成逐步动作，并在不超过 5 步的动作预算内到达目标。

```text
用户意图：在5步以内向右走3米。
Agent: RIGHT  -> 环境坐标 (1, 0)
Agent: RIGHT  -> 环境坐标 (2, 0)
Agent: RIGHT  -> 环境坐标 (3, 0)，成功并自动结束
```

## 当前 v2 实验

- 动作空间：`UP / DOWN / LEFT / RIGHT`，每个动作固定移动 1 米。
- 数据：400 条训练题、100 条独立测试题；训练和测试的中文模板族不重合。
- 语义覆盖：4 个方向 × 1–5 米，共 20 个语义格；每题预算满足 `distance <= max_steps <= 5`。
- Agent：模型每轮只输出一个动作，环境执行并返回新坐标，再进入下一轮。
- 模型：Qwen3.5-4B，thinking 关闭。
- 训练：veRL v0.9.0（commit `483b8a009ba3a97563edee3a19887e4862b8094a`）原生 `AgentLoopBase`，单张 GPU，GRPO，恰好 1 个 optimizer update。
- 评测：训练前后都在同一冻结 test100 上做 greedy paired evaluation；成功门槛为至少 `95/100`。

冻结训练与评测合同见 [`docs/VERL_ONE_STEP_PROTOCOL.md`](docs/VERL_ONE_STEP_PROTOCOL.md)。最终结果会写入 [`docs/RESULTS.md`](docs/RESULTS.md)。

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

训练入口会保存 veRL resume shard，并额外保存可直接评测的 Hugging Face 权重到 `global_step_1/actor/huggingface`。

## 历史 v1.1 pilot

仓库仍保留最初 20 题 zero-shot pilot、19/20（95%）结果和旧版 Windows HTML，便于审计提示词修订历史：

- 协议：[`docs/TEST_PROTOCOL.md`](docs/TEST_PROTOCOL.md)
- 结果：[`results/qwen3.5-4b-zero-shot-v1.1.json`](results/qwen3.5-4b-zero-shot-v1.1.json)
- 旧演示：[`showcase/intent-alignment-showcase.html`](showcase/intent-alignment-showcase.html)

v2 正式运行结束后，HTML 会替换为训练前后 100 题结果。HTML 只离线回放真实结果，不会在浏览器中运行模型。

## 结论边界

这是文字 toy 环境，不是 Minecraft 仿真。test100 成功率只描述冻结测试集；它不构成面对所有表达时总体成功率的统计保证。若训练前已达到 100%，训练后仍为 100%，正确结论是“单步训练后保持 100%，且参数确实更新”，而不是“训练带来提点”。
