# v2 冻结协议：400/100 + veRL 单卡单步训练

## 1. 目标

验证 Qwen3.5-4B 在文字交互 AgentLoop 中能否理解中文位移意图，逐轮输出固定 1 米动作，并在最多 5 步内到达目标；随后用 veRL 在单卡上完成恰好 1 个 optimizer update，并在同一冻结 test100 上比较训练前后结果。

## 2. 环境与动作

- 起点固定为 `(0, 0)`。
- 动作只有 `UP / DOWN / LEFT / RIGHT`。
- 每次动作固定移动 1 米，分别对应 `(0,+1) / (0,-1) / (-1,0) / (+1,0)`。
- 模型每轮只输出一个动作；环境执行后返回当前位置、目标、已用步数和剩余预算。
- 到达目标立即成功；非法动作立即失败；预算耗尽仍未到达则失败。
- thinking 关闭。Qwen chat template 可以包含禁用思考的空 scaffold，模型不能生成自由 reasoning。

## 3. 冻结数据

- `data/intent_alignment_train_400.jsonl`：400 条。
- `data/intent_alignment_test_100.jsonl`：100 条。
- 4 个方向 × 1–5 米 = 20 个语义格。
- 每个语义格有 20 条训练题、5 条测试题。
- 训练使用 20 个模板族，测试使用另外 5 个模板族，模板族完全不重合。
- 每题满足 `distance <= max_steps <= 5`。
- 数据由 `scripts/generate_dataset.py` 确定性生成，哈希由 `data/dataset_manifest.json` 冻结。

## 4. 原生 veRL AgentLoop

- veRL：v0.9.0，commit `483b8a009ba3a97563edee3a19887e4862b8094a`。
- 运行依赖包含该版本 `requirements.txt` 固定的 `TransferQueue==0.1.8`；入口在启动 Ray 前校验版本。
- rollout 使用 veRL 原生支持的 async SGLang backend（冻结为 `sglang==0.5.12`），Qwen3.5 hybrid GDN 使用 Triton attention backend，sampling backend 使用 PyTorch。
- B300 JIT 使用 CUDA 13 toolkit，并显式加入同一 Python runtime 的 CUDA 13 动态库目录。
- 模型使用完整 `Qwen3.5-4B` checkpoint。
- Ray 临时目录使用 `/tmp` 下的短 run-scoped 路径，避免 Linux AF_UNIX socket 路径超限。
- 自定义类只实现环境 wrapper：`IntentAlignmentAgentLoop(AgentLoopBase)`。
- 交互链为 `policy action -> env.step -> observation -> 下一轮 policy`。
- policy token 的 response mask 为 1；环境 observation token 的 response mask 为 0。
- 每条 trajectory 的终点 reward 为成功 `1.0`、失败 `0.0`。
- 不复制、不按领域分叉 veRL 的 shared rollout。

## 5. 单步训练合同

- 算法：GRPO。
- 单张 GPU，`n_gpus_per_node=1`。
- `train_batch_size=400`。
- `rollout.n=2`，共采样 800 条 trajectory。
- `ppo_epochs=1`。
- 配置值 `ppo_mini_batch_size=400`；veRL v0.9.0 在该路径中按 `rollout.n` 扩为 800，与完整 trajectory batch 相等。
- `total_training_steps=1`，因此只执行 1 个 optimizer/global update。
- 保存 `model / optimizer / extra / hf_model`；可评测权重位于 `global_step_1/actor/huggingface`。

运行完成必须同时核验：

1. `global_step_1` 存在且只有一次 optimizer update；
2. actor gradient norm 为正且有限；
3. 800 条 trajectory 形成 400 个两样本 GRPO group；
4. 至少一个 group 内同时出现 reward 0 和 1；
5. checkpoint 包含 model、optimizer、extra、data 和 Hugging Face model；
6. tensor 级语义审计确认相同 policy core key 上发生数值变化，并且至少一个变化在重新量化为 BF16 后仍存在。

veRL 导出的 Hugging Face 权重可以是分片 index，也可以是单文件 `model.safetensors`。文件数量、文件名、dtype 或整文件 SHA 变化只描述序列化差异，不能单独作为参数学习证据。

## 6. 训练前后评测

训练前和训练后使用相同：

- 冻结 test100 数据及 SHA256；
- `v2-agent-loop` prompt contract；
- greedy decoding：`do_sample=false`、`max_new_tokens=8`；
- thinking 关闭；
- batch size 100；
- tokenizer 语义指纹；
- Python、PyTorch、Transformers 和 GPU 型号。

`scripts/compare_results.py` 逐项 fail closed：重新执行每条 `raw_outputs -> env.step` 轨迹，核对 protocol、数据、task identity、tokenizer、model/runtime、完整权重 manifest、固定 tensor probes，并从公开 rollout ledger 与训练日志重算 800 trajectories、400×2 groups、66 mixed groups、`global_step=1` 和正有限梯度。它还核对 checkpoint inventory、运行合同、解析配置和 launcher 的公开 SHA256。已冻结的 veRL 导出配置规范化允许通过，其他模型语义漂移一律拒绝。

训练采用 veRL 原生增量 token context；greedy 评测用完整语义对话重建，并保留每轮原始 assistant 文本。二者的环境语义与对话角色一致，不声称每个历史模板 token 完全相同。

## 7. 已完成正式运行

正式 run 使用现有 1×B300 allocation，没有申请新卡：

- 执行源码：`100f0f4c117779f941cf38f61ef095f58c2ea1a6`
- `global_step = 1`
- 800 trajectories / 400 groups / 66 mixed-reward groups
- `actor/grad_norm = 6.470837116241455`
- 训练前：100/100
- 训练后：100/100
- 成功率变化：0
- policy core 426 个 tensor key 一致；6/6 FP32 probes 改变；8 个 probe 值在 BF16 下仍改变

完整结果见 [`docs/RESULTS.md`](RESULTS.md) 和 [`results/qwen3.5-4b-verl-one-step-v2/`](../results/qwen3.5-4b-verl-one-step-v2/)。

## 8. 报告边界

- test100 只证明冻结 toy 测试集上的表现。
- 训练前已经 100/100，训练后仍为 100/100；报告“保持 100%，参数发生更新”。
- 本实验不接入 Minecraft，不证明真实 Minecraft 操作能力。
