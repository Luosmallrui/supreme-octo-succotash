# E2 托管模型小试（2026-11-26）

> 已被全量结果取代，见 `summary_e2_api_2026-11-26.md`。小试高估了 gpt-5.4-mini direct 的位置效应。

评审模型 `gpt-5.4-mini`（接口返回 `gpt-5.4-mini-2026-03-17`），OpenAI 官方 Chat Completions。候选与布局取自 `manifest_qwen3-8b.jsonl`：随机抽 25 个完整区组（`--blocks 25 --sample-seed 0`，9 个 both_correct_pair、16 个 one_correct_pair），只跑 select，每种模式 100 次调用。

- direct：`reasoning_effort=none`，top-5 logprobs，选择取编号分布的 argmax（与本地预填 direct 一致）。捕获的概率质量最低 0.929。实际输出的编号与 argmax 一致的占 76%。
- think：`reasoning_effort=medium`，取回复里最后一个 `\boxed{k}`。100 次全部正常结束，无需强制收尾。推理平均 1,386 token，最长 13,313。
- 每次调用平均输入 9,257 token，缓存命中可忽略。两种模式均无报错。

同 25 个区组上与 Qwen3-8B 的对比（差值单位为百分点，95% CI 按题整群重采样）：

| 评审 | 模式 | 分层 | 位置效应：选 A | 编号效应：选 A | 准确率 | 首位占比 |
|---|---|---|---:|---:|---:|---:|
| Qwen3-8B | direct | one_correct | +56.2 [+34.4, +75.0] | −12.5 [−25.0, −3.1] | 61% | 56% |
| gpt-5.4-mini | direct | one_correct | +28.1 [+9.4, +50.0] | −3.1 [−28.1, +21.9] | 79% | 21% |
| Qwen3-8B | direct | both_correct | +50.0 [+22.2, +77.8] | −5.6 [−22.2, +11.1] | | |
| gpt-5.4-mini | direct | both_correct | +0.0 [−27.8, +33.3] | +33.3 [+11.1, +55.6] | | |
| Qwen3-8B | think | one_correct | +56.2 [+37.5, +75.0] | +6.2 [−12.5, +25.0] | 95% | 58% |
| gpt-5.4-mini | think | one_correct | +21.9 [+3.1, +40.6] | −3.1 [−21.9, +15.6] | 94% | 27% |
| Qwen3-8B | think | both_correct | +66.7 [+44.4, +88.9] | +22.2 [−5.6, +50.0] | | |
| gpt-5.4-mini | think | both_correct | +27.8 [+0.0, +61.1] | +16.7 [−11.1, +44.4] | | |

准确率与首位占比按模式合并两个分层（各 100 次）；随机选择的首位占比为 12.5%。区组只有 25 个，区间很宽，只用于检查流程和估计效应量级，不进论文。

完整报告：`report_e2_gpt-5.4-mini_{direct,think}_pilot.md`。原始结果：`e2_gpt-5.4-mini_{direct,think}.jsonl`。全量运行使用同一设置时，会跳过这些已保存的行。
