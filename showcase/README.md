# Windows 离线演示

成品：[`intent-alignment-showcase.html`](intent-alignment-showcase.html)

## 打开方式

1. 把这个 HTML 文件复制到 Windows。
2. 双击，用 Microsoft Edge 或 Google Chrome 打开。
3. 点击右上角全屏按钮，或按 `F` 进入全屏演示。

它是单文件离线页面，不需要 Python、服务器、网络或额外素材。

## 建议的 3 分钟展示顺序

1. 在“现场回放”选择 `right-3`，播放“在5步以内向右走3米”。
2. 选择“唯一失败题”，展示 `DOWN -> UP -> DOWN -> DOWN` 为什么无法在4步内恢复。
3. 打开“20题结果”，展示 `19/20 = 95%` 以及四方向分组结果。
4. 打开“实验方法”，说明这是逐轮 `Observation -> Policy -> Action -> Env Step` 的 Agent Loop；本次 `enable_thinking=false`。

## 快捷键

- `Space`：播放 / 暂停
- `→`：执行一个完整 Agent cycle
- `R`：重置当前题
- `F`：全屏

## 证据口径

页面嵌入并回放 `results/qwen3.5-4b-zero-shot-v1.1.json` 中的冻结实测轨迹。浏览器不会伪装实时运行 Qwen3.5-4B；该边界也直接写在页面里。
