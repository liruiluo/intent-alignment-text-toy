# Qwen3.5-4B Dense SPHERE：三臂单步结果

## 结论

三种冻结实现都在 **同一张现有 NVIDIA B300** 上顺序完成了恰好 **1 个 veRL optimizer/global update**，随后在同一份冻结 test100 上达到 **100/100**，均超过 `95/100` 门槛。

| 实验臂 | SPHERE 目标 | α | mixed groups | `||g_ppo||` | `||g_sphere||` | λ | 训练后 test100 |
|---|---|---:|---:|---:|---:|---:|---:|
| `sphere_dense_last_adaptive_v1` | 最后一层（layer 31） | 自适应 | 61 | 6.248408 | 613,299,136 | 1.0188e-11 | **100/100** |
| `sphere_dense_all_hidden_adaptive_v1` | 全部 32 个隐藏层 | 自适应 | 60 | 6.293316 | 633,554,944 | 9.9333e-12 | **100/100** |
| `sphere_dense_last_fixed_alpha_v1` | 最后一层（layer 31） | 固定 1 | 67 | 7.009164 | 616,071,552 | 1.1377e-11 | **100/100** |

每个实验臂都满足：

- 400 条训练题，`rollout.n=2`，共 800 trajectories / 400 GRPO groups；
- `global_step=1` 且 `optimizer_updates_actual=1`；
- SPHERE 梯度经 `g_final = g_ppo + lambda * g_sphere` 进入唯一一次 optimizer step；
- post-cast SPHERE 梯度增量非零，全张量组装残差为 0；
- checkpoint `global_step_1` 完整，FP32 与 BF16 有效参数变化均成立；
- 训练完成后才运行冻结 test100；pipeline exit code 为 0。

## 运行身份

- 模型：Qwen3.5-4B（dense；未使用 MoE）
- 框架：veRL `v0.9.0`，commit `483b8a009ba3a97563edee3a19887e4862b8094a`
- 环境：文字 Agent Loop；动作 `UP / DOWN / LEFT / RIGHT`，每次固定移动 1 米
- 数据：400 train / 100 test；每题最多 5 步
- 硬件：复用同一张现有 NVIDIA B300，三个实验臂顺序运行；未申请新卡
- 显式 reasoning：关闭（`thinking=false`）

## 可审计文件

- [`summary.json`](summary.json)：三臂统一摘要与精确 source/public SHA256；
- 每个实验臂目录中的：
  - `result-status.json`
  - `training-evidence.json`
  - `training-complete.json`
  - `worker-evidence.json`
  - `checkpoint-inventory.json`
  - `parameter-evidence.json`
  - `after-test100.json`（100 条逐题 Agent 轨迹）
  - `pipeline-exit.json`
  - `run-contract.json`（隐私字段已替换，模型/数据/源码/veRL/硬件合同保留）
  - `artifact-manifest.json`
- [`publication-manifest.json`](publication-manifest.json)：发布文件全集 SHA256。

本仓库不提交约 71 GB 的 checkpoint；其文件级清单与 SHA256 已保存在每臂 `checkpoint-inventory.json`。运行：

```bash
PYTHONPATH=src python3 scripts/verify_sphere_published_results.py
```

即可重新核对发布哈希、三臂身份、单步训练、800 trajectories、SPHERE 梯度进入 optimizer、参数变化和 100/100 逐题结果。

## 结论边界

训练前 Qwen3.5-4B 已在该 toy test100 上达到 100/100，因此这三次实验说明的是：**三种 dense-SPHERE 实现都真实参与了一次参数更新，并保持了该文字环境能力**。它们不能证明单步训练带来提点，也不能据此比较三种实现的优劣，更不能外推到 Minecraft 或更复杂环境。
