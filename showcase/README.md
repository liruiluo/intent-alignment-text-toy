# Windows 离线演示

成品：[`intent-alignment-showcase.html`](intent-alignment-showcase.html)

## 打开方式

1. 把这个 HTML 文件复制到 Windows。
2. 双击，用 Microsoft Edge 或 Google Chrome 打开。
3. 点击右上角全屏按钮，或按 `F` 进入全屏演示。

它是单文件离线页面，不需要 Python、服务器、网络或额外素材。

## 建议的 3 分钟展示顺序

1. 在“现场回放”选择 `test-right-3-f03`，播放“限 5 步向右 3 米”的真实训练后轨迹：`RIGHT → RIGHT → RIGHT`。
2. 强调这是 Agent Loop：每轮模型只输出一个动作，环境执行后再返回新观察。
3. 打开“训练结果”，展示 400 条训练数据、100 条独立测试数据，以及训练前后均为 `100/100`。
4. 打开“实验方法”，说明使用 veRL v0.9.0、单张 B300、GRPO、恰好 1 个 optimizer update。
5. 展示参数证据：800 trajectories、66 个 mixed-reward group、actor grad norm 6.4708、6/6 probe 变化、BF16 下仍有 8 个值变化。

## 汇报口径

推荐说：

> 我们实现了一个不依赖 Minecraft 的文字交互 Agent 环境。Qwen3.5-4B 每轮观察状态并输出一个固定 1 米动作。冻结数据为 400 条训练题和 100 条独立测试题。训练前模型已经做到 100/100；随后在现有单张 B300 上用 veRL 完成一个真实 GRPO optimizer update，训练后仍为 100/100，参数审计确认权重发生变化。

不要说“单步训练提高了成功率”：训练前后都是 100%，本轮证明的是训练链打通、参数真实更新和能力保持。

## 自检

浏览器打开：

```text
intent-alignment-showcase.html?selftest=1
```

页面会在右下角隐藏节点写入 `PASS` 或 `FAIL:...`。自动化测试还会校验：

- HTML 内嵌数据与冻结训练后 test100 完全一致；
- 训练前后均为 100/100；
- 训练证据为 1 update / 800 trajectories / 66 mixed groups；
- 参数证据为 6 个 probe 全部变化，BF16 有效变化数为 8；
- 页面无外部 CSS/JS/网络依赖；
- 页面不包含机器本地路径、hostname、Pod IP 或平台任务标识。
