# Qwen3.5-4B 单卡 veRL 单步训练结果

## 最新补充：Dense SPHERE 三臂（2026-09-17）

在相同 400/100 数据、Qwen3.5-4B、veRL v0.9.0 和单张现有 B300 上，SPHERE 最后一层自适应 α、全部 32 层自适应 α、最后一层固定 `α=1` 三个实验臂均完成恰好 1 个 optimizer/global update，训练后均为 **100/100**。完整协议与轻量证据见 [`SPHERE_ONE_STEP_PROTOCOL.md`](SPHERE_ONE_STEP_PROTOCOL.md) 和 [`../results/qwen3.5-4b-sphere-one-step-v1/`](../results/qwen3.5-4b-sphere-one-step-v1/)。

这不改变下文普通 GRPO 基线的结论：训练前已经满分，因此结果证明实现执行、参数更新和能力保持，不证明成功率提升。


## 结论

- 训练前冻结 test100：**100/100（100%）**。
- 使用 veRL v0.9.0 在一张现有 NVIDIA B300 上完成 **1 个 GRPO optimizer/global update**。
- 训练后同一冻结 test100：**100/100（100%）**。
- 成功率变化：**0 个百分点**。
- 参数审计确认 policy 权重发生真实数值变化。

因此，本轮证明的是：单卡原生 veRL AgentLoop 可以完成一次端到端训练更新，并在该冻结测试集上保持满分。训练前已饱和，本轮数据不支持“训练提高成功率”的结论。

## 冻结设置

| 项目 | 设置 |
|---|---|
| 模型 | 完整 Qwen3.5-4B |
| 环境 | 文字二维移动 AgentLoop |
| 动作 | `UP / DOWN / LEFT / RIGHT`，每步固定 1 米 |
| 训练集 / 测试集 | 400 / 100，模板族不重合 |
| 最大动作预算 | 每题不超过 5 步 |
| 显式 reasoning | 关闭，`enable_thinking=false` |
| 框架 | veRL v0.9.0，commit `483b8a009ba3a97563edee3a19887e4862b8094a` |
| 算法 | GRPO |
| 硬件 | 1 × NVIDIA B300（复用现有 allocation；未申请新卡） |
| 评测 | Greedy，训练前后使用同一冻结 test100 |

执行源码提交：`100f0f4c117779f941cf38f61ef095f58c2ea1a6`。

## 单步训练硬证据

- `global_step = 1`
- optimizer update 目标与日志实测均为 1
- 800 条 trajectory
- 400 个 GRPO group，每组 2 条 rollout
- 66 个 mixed-reward group
- reward 取值为 `{0, 1}`
- `actor/grad_norm = 6.470837116241455`
- checkpoint 清单同时包含 model、optimizer、extra、data 和 Hugging Face model，并记录逐文件大小与 SHA256
- 单步训练阶段耗时约 785.6 秒

## 参数变化证据

veRL 把两分片 BF16 base checkpoint 导出为单文件 FP32 `model.safetensors`，同时进行了已知的配置规范化。因此，不能用“文件名或整文件 hash 不同”直接代替参数变化证明。

语义审计结果：

- 训练前后文本 policy core 的 426 个 tensor key 完全一致；
- 固定检查第 0、1、8、16、24、31 层的 input LayerNorm，共 6 个 probe；
- 6/6 probe 在 FP32 数值上发生变化；
- 重新量化到 BF16 后仍有 8 个值发生变化；
- `weights_changed = true`；
- `effective_bfloat16_weights_changed = true`。

base 中 15 个 MTP tensor 因导出配置把 `mtp_num_hidden_layers` 从 1 规范化为 0 而排除；视觉 tensor 前后数量均为 297。完整差异和 probe hash 见 [`parameter-evidence.json`](../results/qwen3.5-4b-verl-one-step-v2/parameter-evidence.json)。

## 配对评测

| 阶段 | 成功数 | 成功率 | 达到 95% 门槛 |
|---|---:|---:|---:|
| 训练前 | 100 / 100 | 100% | 是 |
| 训练后 | 100 / 100 | 100% | 是 |
| 变化 | 0 | 0 pp | — |

逐题配对结果：

- improved：0
- regressed：0
- unchanged success：100
- unchanged failure：0

## 结果文件

可审计副本位于 [`results/qwen3.5-4b-verl-one-step-v2/`](../results/qwen3.5-4b-verl-one-step-v2/)：

- `before-test100.json`：训练前 100 题逐步轨迹；
- `after-test100.json`：训练后 100 题逐步轨迹；
- `paired-comparison.json`：逐题配对结果和身份校验；
- `training-evidence.json`：global step、实际 optimizer update、数据/源码/runtime 和审计文件绑定；
- `rollouts-step1.jsonl`：完整 800 条训练 trajectory，可重算 400 组和 66 个 mixed-reward group；
- `training-log.redacted.txt`：去除机器本地标识、ANSI 控制码和无语义行尾空白的完整训练日志，可读回 `training/global_step=1` 和 `actor/grad_norm`；
- `checkpoint-inventory.json`：`global_step_1` 的 model、optimizer、data、extra 与 HF export 逐文件大小/SHA256；
- `run-contract.redacted.json`、`resolved-config.redacted.yaml`、`launch.redacted.sh`：可公开核验的精确运行合同、解析配置与入口；
- `parameter-evidence.json`：tensor 级参数变化证据，并绑定训练前后被评测权重的文件大小与 SHA256；
- `training-complete.json`：训练完成回执；
- `result-status.json`：最终状态摘要；
- `artifact-manifest.json`：原始 run artifact 与公开副本的 SHA256。

公开副本替换机器本地路径、hostname、Pod IP 和平台任务标识，并去除 ANSI 控制码与无语义行尾空白；任务输出、指标、训练遥测、rollout ledger 和 tensor probe 数值保持不变。原始 artifact SHA 与公开副本 SHA 分栏记录，避免把不可获取的原始 hash 当成公开文件 hash。

可在仓库根目录运行 `PYTHONPATH=src python3 scripts/verify_published_results.py`，重新回放全部评测轨迹并核对训练日志、rollout、checkpoint 清单、参数 probe 和 manifest。

## 工程排障记录

- 早期启动脚本把已声明为 `readonly` 的 `CUDA_HOME` 再作为命令前缀变量赋值，Bash 在创建日志重定向前退出。改为提前 `export` 后，同一现有单卡正常完成训练。
- veRL 导出的 Hugging Face checkpoint 是合法的单文件 `model.safetensors`。早期后处理器错误地只接受 `model.safetensors.index.json`；恢复审计支持两种合法序列化格式，未重跑训练。

## 结论边界

- 这是文字 toy 环境，不包含 Minecraft 视觉、物理、控制延迟或真实游戏状态。
- 100/100 是冻结 test100 的样本结果，不是面对所有中文意图的总体统计保证。
- 训练前后都为 100%，因此没有可观测的成功率提升；本轮价值是训练链、参数更新和能力保持的实证。
