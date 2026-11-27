# 位置感知的贝叶斯选择（闭式后验）· qwen3-8b · think

由 `bayes_select.py` 生成。方法、推导和变体见脚本说明与 `method_bayes_select.md`。参数（α、β、γ）只在其他 4 折的题目上估计（5 折交叉拟合，折与 `e1_analyze.py` 相同）。数值为准确率 (%)，单次调用的方法对 3 个随机顺序取平均。

## 各组准确率

| 组 | N | 集合 | majority | raw1 | pride1 | ours-debias1 | ours1 | vote3 | ours3 | cycsum | ours-debias-cyc | ours-cyc | ours1-1beta | ours1-bcrp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 33 | 55.3 | 75.8 | 74.7 | 72.7 | 75.8 | 69.7 | 72.7 | — | — | — | 75.8 | 75.8 |
| comp | 8 | 30 | 48.2 | 76.7 | 74.4 | 72.2 | 73.3 | 76.7 | 73.3 | — | — | — | 75.6 | 72.2 |
| math500 | 4 | 100 | 75.8 | 93.0 | 93.3 | 90.3 | 92.7 | 94.0 | 93.0 | — | — | — | 92.0 | 92.3 |
| math500 | 8 | 100 | 76.2 | 90.3 | 89.3 | 86.0 | 89.0 | 92.0 | 94.0 | — | — | — | 89.7 | 89.0 |
| math500 | 16 | 100 | 81.0 | 89.7 | 88.0 | 87.3 | 89.7 | 89.0 | 91.0 | — | — | — | 89.0 | 89.7 |
| math500 | 32 | 80 | 78.1 | 89.6 | 86.2 | 85.0 | 90.8 | 91.2 | 91.2 | — | — | — | 91.2 | 90.4 |

列说明：`majority` majority vote (0 calls)；`raw1` raw selector, 1 call；`pride1` divide by slot prior (PriDe), 1 call；`ours-debias1` ours without count prior, 1 call；`ours1` ours, 1 call；`vote3` vote over 3 orderings；`ours3` ours, 3 orderings；`cycsum` summed probability over N cyclic shifts；`ours-debias-cyc` ours without count prior, N cyclic shifts；`ours-cyc` ours, N cyclic shifts；`ours1-1beta` ablation: one beta for all slots, 1 call；`ours1-bcrp` ablation: Beta/CRP count prior, 1 call。

## 合并所有组的配对差（按原题整群重采样）

| 比较 | 差值 (pp) [95% CI] | 集合数 |
|---|---|---:|
| ours1 − majority | +14.2 [+9.8, +19.1] | 443 |
| ours1 − raw1 | -0.4 [-1.0, +0.2] | 443 |
| ours-debias1 − raw1 | -3.5 [-4.9, -2.2] | 443 |
| ours-debias1 − pride1 | -2.1 [-3.6, -0.7] | 443 |
| ours3 − vote3 | +0.7 [-0.7, +2.3] | 443 |
| ours3 − majority | +15.5 [+10.9, +20.6] | 443 |
| ours1 − ours1-1beta | -0.1 [-0.8, +0.6] | 443 |

## 拟合参数（5 折平均）

β 按槽位区间（首位、第 2 位、中间、末位）估计；π_s 为所有候选都错时的槽位先验 softmax(α)；D_s 为单次调用中该槽位区分“此处候选正确”与“全错”的 KL 散度（闭式），衡量该槽位的信息量。

| 组 | N | γ | β 首位 | β 第 2 位 | β 中间 | β 末位 | π 首位 | D 首位 | D 中间（均值） | D 末位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 0.70 | 2.26 | 1.54 | 0.98 | 1.91 | 0.36 | 0.500 | 0.113 | 0.434 |
| comp | 8 | 0.36 | 3.51 | 5.16 | 1.53 | 1.89 | 0.37 | 0.770 | 0.171 | 0.426 |
| math500 | 4 | 1.26 | 3.93 | 4.14 | 3.04 | 1.13 | 0.34 | 0.892 | 1.011 | 0.140 |
| math500 | 8 | 0.43 | 5.00 | 5.11 | 1.62 | 1.66 | 0.36 | 0.962 | 0.217 | 0.306 |
| math500 | 16 | 0.28 | 5.55 | 4.07 | 2.15 | -2.06 | 0.36 | 0.972 | 0.244 | 0.054 |
| math500 | 32 | 0.14 | 5.43 | 3.77 | 2.34 | 3.01 | 0.58 | 0.521 | 0.096 | 0.350 |
