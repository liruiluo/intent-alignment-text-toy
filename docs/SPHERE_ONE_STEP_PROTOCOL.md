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
microbatch 只承担显存分块，不改变 SPHERE 目标。

## 自适应正则系数

每个逻辑 PPO minibatch 做三次无参数更新的 forward/backward，再做唯一一次 optimizer
step：

```text
1. PPO-only backward + 收集完整逻辑 minibatch Phi -> ||g_ppo||
2. 精确逻辑 minibatch SPHERE surrogate backward -> ||g_sphere||
3. lambda = rho * ||g_ppo|| / (||g_sphere|| + 1e-8)
4. (PPO + lambda*SPHERE) backward -> optimizer.step()
```

`lambda` 作为 detached scalar。三次 backward 不等于三次训练；只有第 4 行之后执行
一次 `optimizer.step()`，因此每臂仍是恰好一个 optimizer/global update。

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
- exact source/veRL/model/data/runtime/launcher hashes。

若三臂都保持 100%，结论只能是“SPHERE 三种实现均真实参与一次更新且保持 toy
能力”，不能声称比普通 GRPO 提点。
