# Qwen3.5-4B + veRL 单步训练结果

本目录保存冻结 run 的可公开审计副本：训练前/后 test100、配对比较、完整 rollout ledger、去除机器本地标识的训练日志、checkpoint 清单、运行合同，以及参数变化证据和完成状态。

- 原始 run：`intent-v2-verl090-sglang0512-grpo-step1-qwen35-4b-full-b300-20260916-run3`
- 执行源码：`100f0f4c117779f941cf38f61ef095f58c2ea1a6`
- 训练前：`100/100`
- 训练后：`100/100`
- veRL：v0.9.0，GRPO，恰好 1 个 optimizer/global update
- 采样：800 trajectories，400 组 × 2，66 个 mixed-reward group
- 参数证据：6/6 个冻结 FP32 probe 发生变化；重新量化到 BF16 后仍有 8 个值变化

`artifact-manifest.json` 分开记录原始 run artifact 与本目录公开副本的 SHA256。公开副本替换机器本地路径、hostname、Pod IP/平台任务标识，并去除 ANSI 控制码与无语义行尾空白；不改任务输出、指标、完整 rollout、训练遥测或 tensor probe 数值。

在仓库根目录执行：

```bash
PYTHONPATH=src python3 scripts/verify_published_results.py
```

验证器会回放训练前后 200 条轨迹，重算训练 rollout/group 统计，从训练日志确认实际 1 个 update 与梯度，核对 checkpoint 与被评测权重的 SHA256，并检查公开包没有机器本地标识。
