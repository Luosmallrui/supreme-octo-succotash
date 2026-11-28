# Mistral-Small-3.2-24B：推荐 system prompt 对编号效应的影响

模型卡（mistralai/Mistral-Small-3.2-24B-Instruct-2506，Apache-2.0）推荐的 `SYSTEM_PROMPT.txt` 按其 README 的 `load_system_prompt` 填入模型名与日期（2026-11-28），存于 `system_prompt.txt`。模型的模板不会自动加入它，DeepInfra 也不会插入（只发 "Hi" 时提示为 4 个 token）。

运行：`api_select.py --exp e2 --tasks select --blocks 40 --system-file .../system_prompt.txt`，其余设置与原 Mistral 运行相同（OpenRouter → DeepInfra，预填，贪心编号，`--lenient --follow-up`）。共 160 次调用，$0.13。对比 `compare_mistral_system.md`（`e2_compare.py --reference mistral-small-3.2-24b`，原运行取同一批区组：`hosted_same_blocks.jsonl`）。

| | 区组 | 位置效应（无 → 有 system） | 编号 1 效应（无 → 有 system） | 编号效应配对差 | 位置效应配对差 |
|---|---:|---|---|---|---|
| A 正确、B 错误 | 25 | +44.0 → +58.0 | +4.0 → +22.0 | +18.0 [+2.0, +32.0] | +14.0 [-4.0, +32.0] |
| A、B 均正确 | 15 | +53.3 → +46.7 | +13.3 → +46.7 | +33.3 [+13.3, +53.3] | -6.7 [-33.3, +20.0] |

加入推荐 system prompt 后，Mistral 更偏好打印为 "Solution 1" 的候选；位置效应没有可检测的变化。与 Llama 的情形（`../llama_system_probe/report.md`：去掉默认 system 轮才出现编号偏好）方向相反，但结论一致：编号偏好随与任务无关的 system 轮变化，位置效应不变。

本运行和原 Mistral 运行都已按修正后的解析器重跑列表式回复（见 `api_select.py` 中 `LABEL_LIST` 与 `--redo-lists`）。
