# 位置感知的贝叶斯选择（闭式后验）· qwen3-8b · direct

由 `bayes_select.py` 生成。方法、推导和变体见脚本说明与 `method_bayes_select.md`。参数（α、β、γ）只在其他 4 折的题目上估计（5 折交叉拟合，折与 `e1_analyze.py` 相同）。数值为准确率 (%)，单次调用的方法对 3 个随机顺序取平均。

## 各组准确率

| 组 | N | 集合 | majority | raw1 | pride1 | ours-debias1 | ours1 | vote3 | ours3 | cycsum | ours-debias-cyc | ours-cyc | ours1-1beta | ours1-bcrp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 33 | 55.3 | 55.6 | 58.6 | 55.6 | 58.6 | 57.6 | 60.6 | 63.6 | 63.6 | 63.6 | 58.6 | 56.6 |
| comp | 8 | 30 | 48.2 | 54.4 | 51.1 | 50.0 | 54.4 | 60.0 | 60.0 | 50.0 | 50.0 | 53.3 | 55.6 | 51.1 |
| math500 | 4 | 100 | 75.8 | 72.0 | 71.7 | 67.3 | 78.0 | 72.0 | 79.0 | 76.0 | 79.0 | 81.0 | 79.0 | 78.0 |
| math500 | 8 | 100 | 76.2 | 57.7 | 52.7 | 62.3 | 75.7 | 62.0 | 72.0 | 52.0 | 56.0 | 66.0 | 75.0 | 75.7 |
| math500 | 16 | 100 | 81.0 | 63.7 | 57.0 | 63.7 | 81.7 | 73.0 | 80.0 | 49.0 | 74.0 | 79.0 | 81.0 | 81.0 |
| math500 | 32 | 80 | 78.1 | 65.8 | 57.5 | 67.1 | 78.3 | 72.5 | 76.2 | — | — | — | 77.9 | 77.9 |

列说明：`majority` majority vote (0 calls)；`raw1` raw selector, 1 call；`pride1` divide by slot prior (PriDe), 1 call；`ours-debias1` ours without count prior, 1 call；`ours1` ours, 1 call；`vote3` vote over 3 orderings；`ours3` ours, 3 orderings；`cycsum` summed probability over N cyclic shifts；`ours-debias-cyc` ours without count prior, N cyclic shifts；`ours-cyc` ours, N cyclic shifts；`ours1-1beta` ablation: one beta for all slots, 1 call；`ours1-bcrp` ablation: Beta/CRP count prior, 1 call。

## 合并所有组的配对差（按原题整群重采样）

| 比较 | 差值 (pp) [95% CI] | 集合数 |
|---|---|---:|
| ours1 − majority | +1.2 [+0.1, +2.4] | 443 |
| ours1 − raw1 | +12.0 [+7.0, +16.9] | 443 |
| ours-debias1 − raw1 | -0.1 [-4.0, +3.9] | 443 |
| ours-debias1 − pride1 | +4.1 [-1.0, +9.5] | 443 |
| ours3 − vote3 | +6.3 [+0.9, +11.6] | 443 |
| ours3 − majority | +0.4 [-1.7, +2.5] | 443 |
| ours-cyc − cycsum | +14.9 [+7.5, +22.2] | 403 |
| ours-debias-cyc − cycsum | +10.2 [+2.8, +17.3] | 403 |
| ours-cyc − majority | -0.7 [-3.2, +1.9] | 403 |
| ours1 − ours1-1beta | +0.1 [-0.7, +0.8] | 443 |

## 拟合参数（5 折平均）

β 按槽位区间（首位、第 2 位、中间、末位）估计；π_s 为所有候选都错时的槽位先验 softmax(α)；D_s 为单次调用中该槽位区分“此处候选正确”与“全错”的 KL 散度（闭式），衡量该槽位的信息量。

| 组 | N | γ | β 首位 | β 第 2 位 | β 中间 | β 末位 | π 首位 | D 首位 | D 中间（均值） | D 末位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 0.70 | 1.33 | 0.37 | 1.01 | -0.89 | 0.48 | 0.185 | 0.109 | 0.030 |
| comp | 8 | 0.36 | 1.06 | 0.96 | 0.83 | 0.52 | 0.57 | 0.110 | 0.032 | 0.010 |
| math500 | 4 | 1.26 | 1.37 | 0.78 | 0.46 | 0.74 | 0.47 | 0.199 | 0.024 | 0.038 |
| math500 | 8 | 0.43 | -0.01 | 0.29 | -1.11 | -0.74 | 0.52 | 0.000 | 0.021 | 0.015 |
| math500 | 16 | 0.28 | 0.78 | -0.25 | -0.96 | 0.35 | 0.37 | 0.075 | 0.010 | 0.006 |
| math500 | 32 | 0.14 | 0.88 | 0.09 | -1.05 | 0.33 | 0.40 | 0.095 | 0.005 | 0.006 |
