# 位置感知的贝叶斯选择（闭式后验）· deepseek-flash（Qwen3-8B 候选，预填读出） · direct

由 `bayes_select.py` 生成。方法、推导和变体见脚本说明与 `method_bayes_select.md`。参数（α、β、γ）只在其他 4 折的题目上估计（5 折交叉拟合，折与 `e1_analyze.py` 相同）。数值为准确率 (%)，单次调用的方法对 3 个随机顺序取平均。

## 各组准确率

| 组 | N | 集合 | majority | raw1 | pride1 | ours-debias1 | ours1 | vote3 | ours3 | cycsum | ours-debias-cyc | ours-cyc | ours1-1beta | ours1-bcrp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 33 | 55.3 | 61.6 | 66.7 | 64.6 | 63.6 | 72.7 | 69.7 | 69.7 | 69.7 | 69.7 | 65.7 | 63.6 |
| comp | 8 | 30 | 48.2 | 54.4 | 57.8 | 55.6 | 53.3 | 50.0 | 56.7 | 63.3 | 66.7 | 66.7 | 55.6 | 52.2 |
| math500 | 4 | 100 | 75.8 | 74.3 | 77.3 | 75.0 | 80.0 | 75.0 | 79.0 | 82.0 | 79.0 | 81.0 | 80.0 | 80.3 |
| math500 | 8 | 100 | 76.2 | 77.3 | 73.7 | 67.3 | 77.0 | 81.0 | 82.0 | 76.0 | 77.0 | 80.0 | 78.0 | 76.3 |
| math500 | 16 | 100 | 81.0 | 78.3 | 74.0 | 67.0 | 81.0 | 80.0 | 80.0 | — | — | — | 81.3 | 79.3 |
| math500 | 32 | 80 | 78.1 | 77.5 | 67.1 | 63.3 | 80.8 | 81.2 | 81.2 | — | — | — | 82.1 | 81.2 |

列说明：`majority` majority vote (0 calls)；`raw1` raw selector, 1 call；`pride1` divide by slot prior (PriDe), 1 call；`ours-debias1` ours without count prior, 1 call；`ours1` ours, 1 call；`vote3` vote over 3 orderings；`ours3` ours, 3 orderings；`cycsum` summed probability over N cyclic shifts；`ours-debias-cyc` ours without count prior, N cyclic shifts；`ours-cyc` ours, N cyclic shifts；`ours1-1beta` ablation: one beta for all slots, 1 call；`ours1-bcrp` ablation: Beta/CRP count prior, 1 call。

## 合并所有组的配对差（按原题整群重采样）

| 比较 | 差值 (pp) [95% CI] | 集合数 |
|---|---|---:|
| ours1 − majority | +2.6 [+0.8, +4.6] | 443 |
| ours1 − raw1 | +2.5 [-1.3, +5.9] | 443 |
| ours-debias1 − raw1 | -6.9 [-9.7, -4.1] | 443 |
| ours-debias1 − pride1 | -4.5 [-7.7, -1.5] | 443 |
| ours3 − vote3 | +1.4 [-1.5, +4.1] | 443 |
| ours3 − majority | +4.0 [+1.0, +7.1] | 443 |
| ours-cyc − cycsum | +1.5 [-2.3, +5.5] | 263 |
| ours-debias-cyc − cycsum | -0.4 [-3.1, +2.3] | 263 |
| ours-cyc − majority | +7.4 [+2.8, +12.1] | 263 |
| ours1 − ours1-1beta | -0.8 [-1.5, -0.2] | 443 |

## 拟合参数（5 折平均）

β 按槽位区间（首位、第 2 位、中间、末位）估计；π_s 为所有候选都错时的槽位先验 softmax(α)；D_s 为单次调用中该槽位区分“此处候选正确”与“全错”的 KL 散度（闭式），衡量该槽位的信息量。

| 组 | N | γ | β 首位 | β 第 2 位 | β 中间 | β 末位 | π 首位 | D 首位 | D 中间（均值） | D 末位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| comp | 4 | 0.70 | 1.39 | 0.85 | 1.35 | 0.94 | 0.61 | 0.154 | 0.149 | 0.037 |
| comp | 8 | 0.36 | 1.24 | 1.74 | 0.81 | 1.13 | 0.49 | 0.164 | 0.037 | 0.043 |
| math500 | 4 | 1.26 | 1.43 | 1.57 | 0.80 | 0.64 | 0.62 | 0.160 | 0.052 | 0.026 |
| math500 | 8 | 0.43 | 1.35 | 1.14 | 0.80 | 0.34 | 0.46 | 0.197 | 0.033 | 0.005 |
| math500 | 16 | 0.28 | 1.64 | 0.68 | 0.79 | 0.57 | 0.34 | 0.312 | 0.019 | 0.009 |
| math500 | 32 | 0.14 | 1.36 | 0.77 | 0.10 | 0.68 | 0.32 | 0.223 | 0.000 | 0.004 |
