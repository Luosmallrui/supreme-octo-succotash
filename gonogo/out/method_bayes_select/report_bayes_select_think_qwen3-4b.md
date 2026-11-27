# 位置感知的贝叶斯选择（闭式后验）· qwen3-4b · think

由 `bayes_select.py` 生成。方法、推导和变体见脚本说明与 `method_bayes_select.md`。参数（α、β、γ）只在其他 4 折的题目上估计（5 折交叉拟合，折与 `e1_analyze.py` 相同）。数值为准确率 (%)，单次调用的方法对 3 个随机顺序取平均。

## 各组准确率

| 组 | N | 集合 | majority | raw1 | pride1 | ours-debias1 | ours1 | vote3 | ours3 | cycsum | ours-debias-cyc | ours-cyc | ours1-1beta | ours1-bcrp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 33 | 53.8 | 76.8 | 76.8 | 73.7 | 72.7 | 72.7 | 69.7 | — | — | — | 73.7 | 71.7 |
| comp | 8 | 29 | 55.5 | 71.3 | 72.4 | 70.1 | 67.8 | 75.9 | 79.3 | — | — | — | 69.0 | 69.0 |
| math500 | 4 | 100 | 69.0 | 88.0 | 87.7 | 84.3 | 88.0 | 89.0 | 89.0 | — | — | — | 87.7 | 88.0 |
| math500 | 8 | 100 | 78.0 | 86.3 | 86.3 | 83.7 | 86.7 | 88.0 | 88.0 | — | — | — | 85.3 | 84.7 |
| math500 | 16 | 100 | 81.0 | 85.7 | 85.0 | 82.0 | 86.7 | 90.0 | 84.0 | — | — | — | 85.3 | 87.3 |
| math500 | 32 | 80 | 83.1 | 92.1 | 89.2 | 90.8 | 90.8 | 93.8 | 92.5 | — | — | — | 92.1 | 82.5 |

列说明：`majority` majority vote (0 calls)；`raw1` raw selector, 1 call；`pride1` divide by slot prior (PriDe), 1 call；`ours-debias1` ours without count prior, 1 call；`ours1` ours, 1 call；`vote3` vote over 3 orderings；`ours3` ours, 3 orderings；`cycsum` summed probability over N cyclic shifts；`ours-debias-cyc` ours without count prior, N cyclic shifts；`ours-cyc` ours, N cyclic shifts；`ours1-1beta` ablation: one beta for all slots, 1 call；`ours1-bcrp` ablation: Beta/CRP count prior, 1 call。

## 合并所有组的配对差（按原题整群重采样）

| 比较 | 差值 (pp) [95% CI] | 集合数 |
|---|---|---:|
| ours1 − majority | +11.2 [+7.1, +15.6] | 442 |
| ours1 − raw1 | -0.5 [-1.6, +0.5] | 442 |
| ours-debias1 − raw1 | -2.8 [-4.1, -1.5] | 442 |
| ours-debias1 − pride1 | -2.1 [-3.5, -0.8] | 442 |
| ours3 − vote3 | -1.6 [-3.5, +0.2] | 442 |
| ours3 − majority | +11.9 [+7.1, +16.8] | 442 |
| ours1 − ours1-1beta | +0.3 [-0.7, +1.2] | 442 |

## 拟合参数（5 折平均）

β 按槽位区间（首位、第 2 位、中间、末位）估计；π_s 为所有候选都错时的槽位先验 softmax(α)；D_s 为单次调用中该槽位区分“此处候选正确”与“全错”的 KL 散度（闭式），衡量该槽位的信息量。

| 组 | N | γ | β 首位 | β 第 2 位 | β 中间 | β 末位 | π 首位 | D 首位 | D 中间（均值） | D 末位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 0.80 | 2.70 | 2.66 | 1.56 | 0.49 | 0.36 | 0.626 | 0.278 | 0.030 |
| comp | 8 | 0.70 | 4.02 | 1.78 | 1.50 | 0.93 | 0.28 | 1.039 | 0.168 | 0.077 |
| math500 | 4 | 0.79 | 3.68 | 2.27 | 1.58 | 1.39 | 0.29 | 0.966 | 0.294 | 0.228 |
| math500 | 8 | 0.46 | 4.46 | 2.56 | 1.67 | 1.20 | 0.38 | 0.878 | 0.153 | 0.140 |
| math500 | 16 | 0.27 | 4.77 | 2.57 | 1.59 | 3.90 | 0.43 | 0.775 | 0.066 | 0.627 |
| math500 | 32 | 0.12 | 7.19 | 5.66 | 2.42 | 2.80 | 0.37 | 0.985 | 0.189 | 0.248 |
