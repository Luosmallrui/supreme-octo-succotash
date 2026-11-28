# 预注册确认：qwen3-8b-dryrun 评 qwen3-8b 的候选

由 `confirm_rules.py` 生成，计划见 `out/e6_duels/prereg_llama.md`。差值单位为百分点；“校正区间”为对 8 项检验做 Bonferroni 校正后的 99.375% 区间（按原题整群重采样 20000 次），落在预测一侧的阈值之外即为确认。

## 数据

- 列表调用：6224 次，覆盖 443 个集合；编号概率质量均值 1.000，低于 0.5 的 0 次。
- 列表方法可评估的集合 443，答案对决 443，全候选对决 130（N = 8）。
- 等 token 预算下的双序对数：中位数 5；预算规则的 token 为 3 顺序投票的 0.87 倍。

## 登记的检验

| 编号 | 主张 | 比较 | 差值 | 95% 区间 | 校正区间 | 阈值 | 集合 / 原题 | 结论 |
|---|---|---|---:|---|---|---|---|---|
| L1 | calibration by the slot prior hurts (N >= 8) | pride1 − raw1 | -6.2 | [-9.9, -2.6] | [-11.4, -1.2] | < +0 | 310 / 153 | 确认 |
| L2 | cyclic-shift voting collapses (N >= 16) | cycsum − vote3 | -21.4 | [-31.2, -11.7] | [-35.0, -8.0] | < +0 | 140 / 105 | 确认 |
| L3 | listwise posterior repairs cyclic voting | ours-cyc − cycsum | +14.9 | [+7.3, +22.2] | [+4.3, +25.1] | > +0 | 403 / 164 | 确认 |
| D1 | duel posterior beats the listwise posterior | duel_re − list_ours3 | +5.6 | [+2.1, +9.5] | [+0.7, +11.3] | > +0 | 443 / 164 | 确认 |
| D2 | duel posterior beats majority voting | duel_re − majority | +6.1 | [+2.7, +9.8] | [+1.5, +11.5] | > +0 | 443 / 164 | 确认 |
| D3 | label-free EM beats majority voting | em_shared − majority | +4.3 | [+1.8, +6.8] | [+1.0, +8.0] | > +0 | 443 / 164 | 确认 |
| D4 | label-free EM within 3 pp of supervised | em_shared − duel_re | -1.8 | [-4.0, +0.0] | [-5.0, +0.5] | > -3 | 443 / 164 | 未确认 |
| C1 | count-free duels beat the per-candidate vote | budget − vote3c | +12.7 | [+5.1, +20.4] | [+2.1, +23.8] | > +0 | 130 / 130 | 确认 |

确认 7 / 8 项。

## 描述性准确率（%）

| 来源 | 方法 | 准确率 | 集合 |
|---|---|---:|---:|
| list | majority | 74.1 | 443 |
| list | raw1 | 63.4 | 443 |
| list | pride1 | 59.1 | 443 |
| list | vote3 | 68.2 | 443 |
| list | cycsum | 58.3 | 403 |
| list | ours-cyc | 73.2 | 403 |
| list | ours3 | 74.5 | 443 |
| duel | majority | 74.1 | 443 |
| duel | list_ours3 | 74.5 | 443 |
| duel | duel_prereg | 80.1 | 443 |
| duel | duel_re | 80.1 | 443 |
| duel | em_group | 78.8 | 443 |
| duel | em_shared | 78.3 | 443 |
| allcand | random | 57.3 | 130 |
| allcand | majority | 69.7 | 130 |
| allcand | raw1 | 56.9 | 130 |
| allcand | vote3 | 61.5 | 130 |
| allcand | vote3c | 58.5 | 130 |
| allcand | budget | 71.2 | 130 |
| allcand | full56 | 71.5 | 130 |
| allcand | knockout14 | 74.0 | 130 |

EM（共享对决参数）：δ = 2.14，τ² = 5.98，σ²ε = 0.75；γ 按组 = comp-4 2.20，comp-8 12.00，math500-4 12.00，math500-8 0.41，math500-16 0.54，math500-32 0.19。
