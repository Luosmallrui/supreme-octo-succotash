# E2 开源模型评审（经 OpenRouter）：2026-11-27

审稿意见的头号问题是：物理槽位偏差会不会只是 Qwen3 小模型自评的特例。为此，用另外三个家族的开源模型和一个更大的 Qwen3，在同一套 E2 布局上做评审。布局取自 Qwen3-8B 的 E2 清单，每个模型 976 次调用：944 次 select，32 次 lookup。

## 运行设置

- **入口与路由：** 用 `api_select.py --exp e2 --mode direct`，`--base-url https://openrouter.ai/api/v1 --provider deepinfra`，所有调用都固定由 DeepInfra 提供。按提供商列表，权重为 fp8 或 bf16。
- **预填：** 使用 `--prefill --prefill-style plain`，与 Qwen3 相同的预填作为最后一条 assistant 消息发送。
  - Llama-3.1-8B 和 Mistral-Small 会接着预填往下写，与本地 Qwen3 一致。
  - Gemma-3 和 Qwen3-32B 会另起一条回复，但开头就是编号。
- **读出：** 实测 OpenRouter 只返回最后一个生成 token 的 logprobs，流式也一样。CoreWeave、Parasail、NextBit、AkashML、DeepInfra 都试过。因此无法读到编号分布，选择取贪心解码（temperature 0）的编号，最多 16 个 token。
- **解析：** `--lenient` 接受三种写法：开头的整数、未闭合的 `\boxed{k` 或 `{k}`、以及 "Solution 3 is correct" 这类声明。"{1,2,3}" 这类列表不算选择。
- **追问：** `--follow-up`：读不出编号时，再追问一轮，只要求 `\boxed{k}`，这类调用记为 forced。
- **Qwen3-32B：** 加 `--extra-body '{"reasoning": {"enabled": false}}'` 关闭思考。

| 模型 | 调用 | 报错 | 追问 | 仍无编号 | 输入 token | 费用 |
|---|---:|---:|---:|---:|---:|---:|
| meta-llama/llama-3.1-8b-instruct | 976 | 0 | 0 | 0 | 9.80M | $0.20 |
| google/gemma-3-12b-it | 976 | 0 | 46 | 34 | 10.94M | $0.55 |
| google/gemma-3-27b-it | 976 | 0 | 0 | 0 | 10.52M | $1.02 |
| mistralai/mistral-small-3.2-24b-instruct | 976 | 0 | 89 | 1 | 10.60M | $1.01 |
| qwen/qwen3-32b（非思考） | 976 | 0 | 0 | 0 | 10.23M | $0.83 |

Mistral 的 89 次追问大多是它列出了多个解答，例如 "{1,2,3,5,6,7,8}"。共 $3.60，另有约 $0.07 的探测和小试。

## 结果

数字均来自 `compare_e2_judges.md`（`e2_compare.py`）。原有各行与重算前逐字一致。单位为百分点，区间按原题整群重采样 5000 次。

| 评审 | 准确率 % | 首位效应：一对一错 | 首位效应：两个都对 | 编号 1 效应：一对一错 | 编号 1 效应：两个都对 | lookup |
|---|---:|---|---|---|---|---:|
| Qwen3-8B（自评，分布读出） | 70.2 | +63.5 [+56.2, +70.4] | +63.7 [+54.7, +72.6] | −9.6 [−13.5, −5.8] | −0.5 [−5.2, +4.2] | 2/32 |
| Qwen3-32B | 64.7 | +58.5 [+50.4, +66.5] | +52.4 [+43.4, +61.8] | −9.2 [−14.2, −4.6] | −0.5 [−6.1, +5.2] | 1/32 |
| Mistral-Small-3.2-24B | 80.6 | +53.5 [+45.4, +61.2] | +59.9 [+51.4, +67.9] | +6.5 [+1.2, +11.9] | +17.5 [+11.3, +24.1] | 2/32 |
| Llama-3.1-8B | 70.0 | +43.1 [+35.4, +50.4] | +51.9 [+42.5, +61.3] | +24.6 [+18.1, +31.2] | +25.5 [+17.9, +33.0] | 0/32 |
| Gemma-3-27B | 77.6 | +26.2 [+18.5, +34.2] | +41.5 [+32.1, +50.5] | −3.1 [−9.2, +2.7] | +2.8 [−3.8, +9.4] | 3/32 |
| Gemma-3-12B | 68.3 | +13.8 [+7.7, +20.0] | +18.4 [+11.8, +25.5] | −6.2 [−12.7, +0.4] | +1.4 [−4.7, +8.0] | 3/32 |
| DeepSeek-V4.1-Flash（已有） | 80.2 | +21.5 [+13.5, +29.6] | +11.8 [+4.7, +19.3] | +21.5 [+15.8, +28.1] | +46.7 [+38.7, +55.2] | 9/32 |

**结论：**
- **物理槽位偏差不是 Qwen 小模型自评的特例。** 三个其他家族都有明显的首位效应。Mistral-24B 的首位效应与 Qwen 相当，Llama-8B 次之，Gemma 较弱。Qwen3-32B 评别人的候选，与 Qwen3-8B 自评没有可分辨的差别：配对差 −5.0 [−12.7, +2.7]。
- **也不是选择器太弱的副产品。** Mistral 的准确率与 DeepSeek 相当（80.6% 对 80.2%），首位效应却是它的约 2.5 倍。
- **编号偏好因家族而异：** Llama 和 Mistral 有，Gemma 和 Qwen 没有。
- **lookup：** 能说出某个位置印着哪个编号的，只有 GPT 评审。

**读法差异：** Qwen3 行是编号分布上的 argmax；这里的开源模型行是贪心编号，Gemma 和 Qwen3-32B 还是另起一轮回复后给出的编号。贪心编号就是分布的 argmax，所以 E2 用的“选择”指标可以直接比较。不过分布层面的指标，例如分配给 A 的概率，无法计算。

## 未纳入或未完成

- **Llama-3.3-70B：** DeepInfra 上严重限流，小试中单次调用 40–1000 秒，有 2/8 次在重试后失败。剩余额度也不够，所以没有跑完整 E2。
- **GLM（SiliconFlow，`--lenient --follow-up`，不带预填）：**
  - GLM-4-32B-0414 已完成（2026-11-27/28，976/976，并发 2，约 4.5 小时，10.9M 输入 token，按 ¥1.89/M 约 ¥21）。SiliconFlow 不返回 logprobs，预填也不可靠，所以由模型自己写 `oxed{k}`，与 GPT 评审相同；82 次追问，11 次仍无编号。结果：首位效应 +18.8 [+13.1, +24.6]（一对一错）、+17.5 [+9.0, +25.9]（两个都对），编号 1 效应 −5.8、−0.5（区间含 0），准确率 74.5%，lookup 4/32。与 Qwen3-8B 的首位效应配对差为 −44.6 [−52.7, −36.2]。加入后，lookup 与首位效应的 Spearman 相关为 −0.82（p = 0.002，n = 11），见 `report_lookup_carrier.md`。
  - GLM-4-9B-0414 在 130 行时停止：38 行写的是题目答案（如 `\boxed{236}`）而不是解答编号，追问后仍然如此。同规模的其他家族已有 Llama-3.1-8B 覆盖，所以不再继续。

## 复现

```bash
cd gonogo
python api_select.py --exp e2 --model meta-llama/llama-3.1-8b-instruct --tag llama-3.1-8b --mode direct \
    --base-url https://openrouter.ai/api/v1 --key-file ../openrouter.txt --provider deepinfra --prefill \
    --prefill-style plain --effort omit --forced-effort omit --logprobs off --max-tokens-param max_tokens \
    --max-tokens 16 --temperature 0 --lenient --follow-up --concurrency 8 --max-in-m 12
python e2_analyze.py --input out/e2_position_label/api/e2_llama-3.1-8b_direct.jsonl \
    --output out/e2_position_label/api/report_e2_llama-3.1-8b_direct.md
```

其他模型只需换 `--model` 和 `--tag`；Qwen3-32B 另加 `--extra-body '{"reasoning": {"enabled": false}}'`。对比表的命令见 `e2_compare.py` 的说明，输入为 `评审:模式:文件`。
