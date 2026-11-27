# 位置感知的贝叶斯选择（闭式后验）· qwen3-4b · direct

由 `bayes_select.py` 生成。方法、推导和变体见脚本说明与 `method_bayes_select.md`。参数（α、β、γ）只在其他 4 折的题目上估计（5 折交叉拟合，折与 `e1_analyze.py` 相同）。数值为准确率 (%)，单次调用的方法对 3 个随机顺序取平均。

## 各组准确率

| 组 | N | 集合 | majority | raw1 | pride1 | ours-debias1 | ours1 | vote3 | ours3 | cycsum | ours-debias-cyc | ours-cyc | ours1-1beta | ours1-bcrp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 33 | 53.8 | 65.7 | 68.7 | 66.7 | 63.6 | 72.7 | 60.6 | 78.8 | 69.7 | 69.7 | 63.6 | 62.6 |
| comp | 8 | 29 | 55.5 | 52.9 | 48.3 | 47.1 | 58.6 | 55.2 | 58.6 | 65.5 | 55.2 | 55.2 | 58.6 | 58.6 |
| math500 | 4 | 100 | 69.0 | 69.3 | 70.7 | 67.7 | 70.7 | 71.0 | 69.0 | 71.0 | 70.0 | 70.0 | 70.3 | 69.7 |
| math500 | 8 | 100 | 78.0 | 66.0 | 64.0 | 65.0 | 77.0 | 68.0 | 75.0 | 60.0 | 72.0 | 74.0 | 79.0 | 76.3 |
| math500 | 16 | 100 | 81.0 | 67.7 | 63.0 | 72.3 | 81.0 | 71.0 | 81.0 | 58.0 | 79.0 | 81.0 | 82.0 | 79.7 |
| math500 | 32 | 80 | 83.1 | 70.8 | 65.4 | 70.4 | 84.2 | 85.0 | 83.8 | — | — | — | 83.8 | 76.0 |

列说明：`majority` majority vote (0 calls)；`raw1` raw selector, 1 call；`pride1` divide by slot prior (PriDe), 1 call；`ours-debias1` ours without count prior, 1 call；`ours1` ours, 1 call；`vote3` vote over 3 orderings；`ours3` ours, 3 orderings；`cycsum` summed probability over N cyclic shifts；`ours-debias-cyc` ours without count prior, N cyclic shifts；`ours-cyc` ours, N cyclic shifts；`ours1-1beta` ablation: one beta for all slots, 1 call；`ours1-bcrp` ablation: Beta/CRP count prior, 1 call。

## 合并所有组的配对差（按原题整群重采样）

| 比较 | 差值 (pp) [95% CI] | 集合数 |
|---|---|---:|
| ours1 − majority | +1.3 [-0.2, +2.8] | 442 |
| ours1 − raw1 | +8.4 [+3.5, +13.2] | 442 |
| ours-debias1 − raw1 | +0.1 [-3.3, +3.1] | 442 |
| ours-debias1 − pride1 | +2.3 [-1.4, +6.0] | 442 |
| ours3 − vote3 | +2.5 [-2.3, +7.3] | 442 |
| ours3 − majority | +0.2 [-2.3, +2.6] | 442 |
| ours-cyc − cycsum | +10.0 [+3.8, +16.2] | 402 |
| ours-debias-cyc − cycsum | +9.0 [+3.0, +15.0] | 402 |
| ours-cyc − majority | +0.3 [-2.8, +3.3] | 402 |
| ours1 − ours1-1beta | -0.5 [-1.6, +0.5] | 442 |

## 拟合参数（5 折平均）

β 按槽位区间（首位、第 2 位、中间、末位）估计；π_s 为所有候选都错时的槽位先验 softmax(α)；D_s 为单次调用中该槽位区分“此处候选正确”与“全错”的 KL 散度（闭式），衡量该槽位的信息量。

| 组 | N | γ | β 首位 | β 第 2 位 | β 中间 | β 末位 | π 首位 | D 首位 | D 中间（均值） | D 末位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 0.80 | 2.40 | 0.85 | 0.43 | 0.18 | 0.38 | 0.519 | 0.015 | 0.002 |
| comp | 8 | 0.70 | 1.93 | 1.03 | -0.16 | -0.95 | 0.57 | 0.270 | 0.001 | 0.009 |
| math500 | 4 | 0.79 | 1.79 | 1.03 | 0.39 | 0.09 | 0.51 | 0.279 | 0.011 | 0.000 |
| math500 | 8 | 0.46 | 1.20 | 0.44 | -0.58 | -2.58 | 0.43 | 0.165 | 0.007 | 0.094 |
| math500 | 16 | 0.27 | 1.04 | 0.03 | -1.33 | -1.36 | 0.54 | 0.113 | 0.010 | 0.026 |
| math500 | 32 | 0.12 | 1.08 | 1.53 | -0.97 | -2.52 | 0.59 | 0.109 | 0.003 | 0.032 |
