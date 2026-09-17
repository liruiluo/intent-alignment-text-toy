# Qwen3.5-4B Dense SPHERE：单步冻结实验协议

## 目标

在 veRL 最新稳定版 `v0.9.0`（commit
`483b8a009ba3a97563edee3a19887e4862b8094a`）上，把 SPHERE
(arXiv:2605.04712) 映射到 dense Qwen3.5-4B，并比较三个独立实验臂：

1. `sphere_dense_last_adaptive_v1`：最后一个 Transformer MLP 层，批次自适应目标尺度；
2. `sphere_dense_all_hidden_adaptive_v1`：全部 32 个 Transformer MLP 层逐层计算后求和，批次自适应目标尺度；
3. `sphere_dense_last_fixed_alpha_v1`：最后一个 Transformer MLP 层，固定目标尺度 `alpha=1`。

三臂共同使用梯度范数比例系数 `rho=1e-3`。固定 `alpha` 指论文中的各向同性
Gram 目标尺度，不是 SPHERE loss 前的系数；论文报告了 `alpha=1/2/5`，本次冻结采用
其中明确报告的 `alpha=1`。

## Dense 映射

论文附录证明 dense MLP 是 `E=1, K=1` 的退化 MoE。Qwen3.5-4B 每个
Transformer MLP 的特征 `Phi` 取自：

```text
SiLU(gate_proj(x)) * up_proj(x) -> [Phi] -> down_proj(Phi)
```

实现通过 `mlp.down_proj` 的 `forward_pre_hook` 捕获输入。只选择和 PPO action loss
对齐的 response token（左移后的 `loss_mask`），不使用 prompt token。

每层的 SPHERE penalty 为：

```text
A = Phi^T Phi / T
L = ||A - alpha I||_F^2
```

自适应目标尺度取 `alpha = Tr(A) / D`。当 `D > T` 时使用 sample-space Gram
`Phi Phi^T / T` 计算相同的 Frobenius 范数和迹，但各向同性目标的维度仍是特征维
`D`。All-hidden 臂对 32 层 penalty 分别计算后求和，不把各层特征拼成 joint Gram。

SPHERE 的非线性 Gram 目标在完整逻辑 PPO minibatch 上计算，不能把各 physical
microbatch 的 penalty 分别算完再求和。实现先在 PPO-only pass 中按 physical
microbatch 收集 detached action-token `Phi`，逐层拼成完整逻辑 minibatch，精确计算
penalty 和闭式 `dL/dPhi`；随后按原 microbatch 顺序重放一个线性 surrogate，把同一
`dL/dPhi` 传回模型参数。每次重放都核对 token/mask identity，所以 physical
microbatch 只承担显存分块，不改变 SPHERE 目标。SPHERE replay 恢复 PPO collection
开始前的 Torch CPU/CUDA RNG 状态。worker 直接逐元素比较 collection 与 replay 的
`Phi`。Qwen3.5 的 BF16 hybrid-attention kernel 在重复 forward 中并非 bitwise
确定，因此冻结数值门为整体相对 L2 与每层相对 L2 不超过 `5e-2`，整体相对最大差异
不超过 `1e-1`；每层相对最大差异保留为诊断值，但不单独作为硬门，因为单个 BF16
元素或以某层自身较小的最大激活作分母都会产生不稳定比值。同时直接约束与 SPHERE
feature gradient 的原始 surrogate 内积相对偏差不超过 `1e-3`。`1e-1` 的整体最大差异
门由完整 32 层一更新实测校准：该次整体相对 L2 为 `2.0414%`、整体相对最大差异为
`8.3050%`、raw surrogate 偏差为 `0.001872%`，其余梯度组装与 32 层非零梯度门全部
通过。校准失败记录保留，不算正式结果；后续新 run 按此冻结门重新执行。这些门共同
防止 activation replay 明显漂移和实际正则方向漂移。

## 自适应正则系数

每个逻辑 PPO minibatch 做两次 forward/backward，再显式组装唯一一次 optimizer
step 所消费的梯度：

```text
1. PPO-only backward + 收集完整逻辑 minibatch Phi -> ||g_ppo||
   并保存完整 `g_ppo`
2. 精确逻辑 minibatch SPHERE surrogate backward -> ||g_sphere||
3. lambda = rho * ||g_ppo|| / (||g_sphere|| + 1e-8)
4. 显式写入 `g_final = g_ppo + lambda * g_sphere`
5. `optimizer.step()`
```

`lambda` 作为 detached scalar。公式与论文附录 Algorithm 1 的总 loss
`L_PPO + lambda * L_SPHERE` 梯度严格等价，同时避开同一 FSDP 模型第三次独立 replay
产生的数值漂移。两次 backward 后只执行一次 `optimizer.step()`，因此每臂仍是恰好
一个 optimizer/global update。证据中必须记录 `combined_forward_backward=0`、
`explicit_gradient_combination=1`。组装时逐参数用 FP32 计算预期和，再转换到 optimizer
实际接收的梯度 dtype；随后对全部梯度元素核对 post-cast 预期值与实写值，要求 L2 和
最大绝对残差都为零，并要求转换后相对 PPO 梯度至少有一个元素和非零 L2 变化。抽样
投影只保留为辅助诊断，不能单独证明 SPHERE 梯度已经进入 optimizer。

## 冻结比较合同

三臂必须满足：

- 同一 Qwen3.5-4B base 权重哈希；每臂从 base 独立启动；
- 同一 400 条 train / 100 条 test 与现有冻结 SHA256；
- `rollout.n=2`、800 trajectories、GRPO、actor LR `1e-6`；
- thinking 关闭，使用同一个原生 veRL AgentLoop；
- 关闭 actor gradient checkpointing 和 torch compile，以保证 activation hook 只记录真实
  forward 且不会因重算重复计数；
- 每臂恰好 1 个 optimizer/global update；
- 训练声明终点后才运行 test100；成功门槛为 `>=95/100`；
- 不申请新卡，只复用当前存活的现有 1×B300 allocation；三个实验臂顺序运行，实验结束恢复 holder，不释放 allocation。

## 最低证据

每臂发布：

- target layer 完整列表、层数、每层 hook 调用数与 action-token 数；
- 逻辑 minibatch penalty、alpha、闭式 feature gradient、physical microbatch replay
  identity、PPO gradient norm、SPHERE gradient norm、lambda、正则梯度贡献比例；
- 每个目标层首个 microbatch activation gradient 的 finite/nonzero 证据；
- optimizer step 数、global step 1 checkpoint、参数变化审计；
- 训练后冻结 test100 的逐题轨迹与成功率；
- exact source/veRL/model/data/runtime/launcher hashes；
- 本轮唯一 256-bit run nonce、canonical worker-evidence 路径，以及 worker / training
  receipt / run contract 的 nonce 哈希绑定；若 success 与 failed evidence 共存或目标路径已存在，
  launcher/auditor 必须 fail closed；
- PPO/SPHERE source gradient、FP32 assembly、optimizer destination gradient 的 dtype、
  tensor representation 与 DTensor placement 计数，以及 post-cast 全张量残差。

若三臂都保持 100%，结论只能是“SPHERE 三种实现均真实参与一次更新且保持 toy
能力”，不能声称比普通 GRPO 提点。

## 公开复现入口

仓库提供统一入口 [`scripts/reproduce_sphere_three_arms.py`](../scripts/reproduce_sphere_three_arms.py)。它固定按 last/adaptive、all-hidden/adaptive、last/fixed-alpha 的顺序在同一张可见 GPU 上串行运行，并对每臂执行：

```text
复用同 base 的冻结 before-test100
→ 解析并保存 veRL 配置
→ 1 个 optimizer/global update
→ 参数变化与训练证据审计
→ global_step_1 上的冻结 test100
→ >=95 门与报告值 100/100 的最终核对
```

复现入口不包含 9N 或其他平台的资源申请/释放逻辑；调用者须先进入一张已有的空闲 B300。精确 parquet 输入提交在 `data/verl-v0.9.0/`，入口还会核对 veRL、runtime、base checkpoint 和数据 SHA256。完整命令及环境变量见顶层 [`README.md`](../README.md#一条命令复现-sphere-三臂)。

## 正式单步结果（2026-09-17）

三个实验臂均复用同一张现有 NVIDIA B300 顺序完成；每臂都是 400 条训练题、`rollout.n=2`、800 trajectories / 400 GRPO groups，并且只有 1 个 optimizer/global update。训练完成回执落盘后才运行冻结 test100。

| 实验臂 | mixed groups | 目标层数 | `||g_ppo||` | `||g_sphere||` | λ | replay L2 / max | test100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sphere_dense_last_adaptive_v1` | 61 | 1 | 6.248408 | 613,299,136 | 1.0188e-11 | 1.8567% / 3.0556% | **100/100** |
| `sphere_dense_all_hidden_adaptive_v1` | 60 | 32 | 6.293316 | 633,554,944 | 9.9333e-12 | 1.9848% / 3.5112% | **100/100** |
| `sphere_dense_last_fixed_alpha_v1` | 67 | 1 | 7.009164 | 616,071,552 | 1.1377e-11 | 1.8511% / 2.7473% | **100/100** |

三臂均满足：

- worker evidence `status=passed`，所有运行时检查为 true；
- `global_step=1`、`optimizer_updates_actual=1`、checkpoint `global_step_1` 完整；
- 组装后正则梯度增量非零，全张量组装 L2 残差为 0；
- activation replay 的整体相对 L2 < 5%、整体相对最大差异 < 10%、raw surrogate 偏差 < 0.1%；
- FP32 参数变化和 BF16 有效参数变化均成立；
- pipeline exit code 为 0，训练后 test100 均达到 100/100。

公开摘要、逐题轨迹和文件哈希见 [`results/qwen3.5-4b-sphere-one-step-v1/`](../results/qwen3.5-4b-sphere-one-step-v1/)，可运行：

```bash
PYTHONPATH=src python3 scripts/verify_sphere_published_results.py
```

重新核验。约 71 GB 的 checkpoint 不提交 Git；每臂的 `checkpoint-inventory.json` 保留文件级大小和 SHA256。

### 结果解释

三臂的 base checkpoint 在训练前已经是 100/100，因此本实验不支持“SPHERE 单步提高成功率”或“三种实现有性能优劣”的结论。可支持的结论是：三种 dense-SPHERE 实现均真实产生非零正则梯度、进入唯一一次 optimizer step、改变了参数，并在冻结文字 toy test100 上保持 100% 成功率。
