
## qwen3-8b: behaviour

| mode | N | inst | acc | chance(A) | first | front¼ | mid½ | back¼ | last | front−mid [95%CI] | back−mid [95%CI] | honest gap | perm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 4 | 150 | 0.355 | 0.250 | 0.720 | 0.720 | 0.253 | 0.193 | 0.193 | +46.7 [+39.0,+54.3] | -6.0 [-11.0,-1.0] | 52.7 | 0.001 |
| A | 8 | 124 | 0.310 | 0.125 | 0.798 | 0.500 | 0.232 | 0.278 | 0.298 | +26.8 [+21.6,+31.7] | +4.6 [-0.4,+9.9] | 29.0 | 0.001 |
| A | 16 | 63 | 0.197 | 0.062 | 0.714 | 0.234 | 0.135 | 0.286 | 0.429 | +9.9 [+5.0,+14.5] | +15.1 [+9.5,+21.8] | 17.9 | 0.001 |
| A | 32 | 13 | 0.308 | 0.031 | 0.692 | 0.269 | 0.264 | 0.433 | 0.538 | +0.5 [-15.9,+16.8] | +16.8 [+4.3,+32.7] | 0.9 | 0.001 |
| B | 4 | 150 | 0.355 | nan | 0.713 | 0.713 | 0.223 | 0.260 | 0.260 | +49.0 [+41.3,+56.3] | +3.7 [-0.7,+8.3] | 53.3 | 0.001 |
| B | 8 | 124 | 0.170 | nan | 0.540 | 0.343 | 0.093 | 0.153 | 0.218 | +25.0 [+20.0,+30.0] | +6.0 [+3.4,+9.1] | 25.8 | 0.001 |
| B | 16 | 63 | 0.106 | nan | 0.413 | 0.163 | 0.079 | 0.103 | 0.159 | +8.3 [+5.6,+11.5] | +2.4 [+0.4,+4.8] | 8.3 | 0.001 |
| B | 32 | 13 | 0.197 | nan | 0.538 | 0.269 | 0.173 | 0.173 | 0.308 | +9.6 [+1.9,+21.2] | +0.0 [-2.9,+2.9] | 8.2 | 0.001 |
| C | 4 | 150 | 0.355 | nan | 0.440 | 0.440 | 0.200 | 0.580 | 0.580 | +24.0 [+17.7,+31.0] | +38.0 [+30.7,+46.0] | 36.7 | 0.001 |
| C | 8 | 124 | 0.154 | nan | 0.282 | 0.185 | 0.071 | 0.290 | 0.476 | +11.5 [+7.5,+15.9] | +22.0 [+17.1,+26.8] | 22.6 | 0.001 |
| C | 16 | 63 | 0.089 | nan | 0.143 | 0.091 | 0.063 | 0.139 | 0.365 | +2.8 [+0.4,+6.0] | +7.5 [+4.8,+10.3] | 7.5 | 0.001 |
| C | 32 | 13 | 0.159 | nan | 0.154 | 0.154 | 0.154 | 0.173 | 0.308 | +0.0 [+0.0,+0.0] | +1.9 [+0.0,+4.8] | 1.9 | 0.038 |
| T | 4 | 40 | 0.556 | 0.250 | 0.650 | 0.650 | 0.537 | 0.500 | 0.500 | +11.3 [+0.0,+22.5] | -3.7 [-18.8,+11.2] | 5.0 | 0.098 |
| T | 8 | 30 | 0.412 | 0.125 | 0.633 | 0.550 | 0.342 | 0.417 | 0.467 | +20.8 [+10.8,+31.7] | +7.5 [-4.2,+18.3] | 18.3 | 0.023 |

Choice distribution in mode A (share of prompts choosing each position-quartile; uniform = 0.25), and unparseable rate:

- N=4: all picks [0.613, 0.132, 0.153, 0.102] | wrong picks [0.672, 0.106, 0.14, 0.083] | unparseable 0.000
- N=8: all picks [0.675, 0.08, 0.09, 0.155] | wrong picks [0.798, 0.039, 0.038, 0.124] | unparseable 0.000
- N=16: all picks [0.61, 0.035, 0.101, 0.254] | wrong picks [0.687, 0.01, 0.075, 0.227] | unparseable 0.000
- N=32: all picks [0.553, 0.163, 0.075, 0.209] | wrong picks [0.701, 0.125, 0.028, 0.146] | unparseable 0.000

Mode A choice probability P(correct index) from first-token logprobs (N ≤ 8; lower-variance than greedy):

- N=4: per-position P(correct) = [0.669, 0.272, 0.272, 0.184]; front−mid +39.7 [+33.4,+45.6], back−mid -8.8 [-12.4,-5.2]
- N=8: per-position P(correct) = [0.737, 0.191, 0.219, 0.211, 0.25, 0.263, 0.264, 0.297]; front−mid +22.8 [+18.2,+27.3], back−mid +4.5 [+0.9,+8.2]

Majority effect in mode A (wrong candidates often share an answer):

- N=4: picks the plurality WRONG answer 0.540 (uniform-random would be 0.482); acc when wrong answers have a plurality cluster ≥2: 0.326 (n=396) vs all-distinct: 0.412 (n=204)
- N=8: picks the plurality WRONG answer 0.495 (uniform-random would be 0.523); acc when wrong answers have a plurality cluster ≥2: 0.304 (n=920) vs all-distinct: 0.389 (n=72)
- N=16: picks the plurality WRONG answer 0.500 (uniform-random would be 0.604); acc when wrong answers have a plurality cluster ≥2: 0.197 (n=1008) vs all-distinct: nan (n=0)
- N=32: picks the plurality WRONG answer 0.505 (uniform-random would be 0.750); acc when wrong answers have a plurality cluster ≥2: 0.308 (n=416) vs all-distinct: nan (n=0)

Prior vs content (conditional logit, free slot intercepts; β_bin = weight on is-correct in each position bin; LR test of equal β; 95% CI by problem bootstrap):

- mode A N=4 (n=600): β by bin = 0.42 [-0.16,1.04], 1.24 [0.68,1.90], 0.91 [0.44,1.42], 1.14 [0.54,1.85]; edge−mid -0.29 [-0.76,+0.20]; LR p=0.125; slot prior logits (slot1=0) first/mid/last = 0.00/-1.71/-2.09
- mode A N=8 (n=992): β by bin = 0.74 [0.09,1.36], 2.40 [1.61,3.68], 2.68 [2.02,3.41], 1.58 [0.92,2.32]; edge−mid -1.38 [-2.06,-0.78]; LR p=0.000; slot prior logits (slot1=0) first/mid/last = 0.00/-3.27/-2.25
- mode A N=16 (n=1008): β by bin = 0.88 [-0.55,2.08], 3.39 [2.18,15.41], 1.86 [0.93,2.66], 1.58 [0.87,2.39]; edge−mid -1.39 [-7.49,-0.54]; LR p=0.000; slot prior logits (slot1=0) first/mid/last = 0.00/-4.40/-1.62
- mode A N=32 (n=416): β by bin = 1.48 [-3.10,4.21], 1.48 [0.21,11.00], 2.38 [0.15,15.04], 1.88 [-0.51,5.05]; edge−mid -0.25 [-7.92,+1.24]; LR p=0.375; slot prior logits (slot1=0) first/mid/last = 0.00/-5.29/-4.54
- mode T N=4 (n=103): β by bin = 4.14 [2.96,35.02], 3.95 [2.39,36.77], 2.36 [0.83,31.15], 2.11 [-0.20,21.71]; edge−mid -0.03 [-17.26,+16.50]; LR p=0.366; slot prior logits (slot1=0) first/mid/last = 0.00/-0.36/-0.01
- mode T N=8 (n=143): β by bin = 2.86 [1.94,17.07], 4.09 [3.08,20.04], 18.55 [16.75,21.07], 4.20 [3.10,5.93]; edge−mid -7.79 [-15.49,-6.03]; LR p=0.056; slot prior logits (slot1=0) first/mid/last = 0.00/-7.19/-2.08

Mode A accuracy split by source (math500 / comp):

- N=4 math500: inst=118 acc=0.350 front=0.712 mid=0.250 back=0.186
- N=4 comp: inst=32 acc=0.375 front=0.750 mid=0.266 back=0.219
- N=8 math500: inst=97 acc=0.321 front=0.510 mid=0.235 back=0.304
- N=8 comp: inst=27 acc=0.273 front=0.463 mid=0.222 back=0.185
- N=16 math500: inst=60 acc=0.206 front=0.242 mid=0.142 back=0.300
- N=16 comp: inst=3 acc=0.021 front=0.083 mid=0.000 back=0.000
- N=32 math500: inst=13 acc=0.308 front=0.269 mid=0.264 back=0.433

## qwen3-8b: attention (head-mean, query = instruction rows)

- mode A N=4 (prompts=160): layers1-2 slot share front/mid/back per-slot = 0.101/0.130/0.639 (uniform 0.250); r(attn on correct block, acc) across positions: L1=-0.45 L2=-0.45 | full-sample acc curve: L1=-0.49 L2=-0.49 | best layer 30 (+0.93)
    slot-attention profile vs slot-pick share (Spearman across slots): L1=-1.00 L2=-1.00 mid-layers=-0.40 deep=+0.15 | best L25 (+0.20)
- mode A N=8 (prompts=320): layers1-2 slot share front/mid/back per-slot = 0.037/0.063/0.337 (uniform 0.125); r(attn on correct block, acc) across positions: L1=-0.12 L2=-0.12 | full-sample acc curve: L1=-0.07 L2=-0.08 | best layer 27 (+0.92)
    slot-attention profile vs slot-pick share (Spearman across slots): L1=+0.14 L2=+0.07 mid-layers=+0.73 deep=+0.65 | best L8 (+0.81)
- mode A N=16 (prompts=480): layers1-2 slot share front/mid/back per-slot = 0.015/0.030/0.174 (uniform 0.062); r(attn on correct block, acc) across positions: L1=+0.29 L2=+0.29 | full-sample acc curve: L1=+0.38 L2=+0.38 | best layer 27 (+0.95)
    slot-attention profile vs slot-pick share (Spearman across slots): L1=+0.61 L2=+0.60 mid-layers=+0.78 deep=+0.77 | best L19 (+0.87)
- mode A N=32 (prompts=416): layers1-2 slot share front/mid/back per-slot = 0.008/0.015/0.087 (uniform 0.031); r(attn on correct block, acc) across positions: L1=+0.44 L2=+0.44 | full-sample acc curve: L1=+0.44 L2=+0.44 | best layer 25 (+0.73)
    slot-attention profile vs slot-pick share (Spearman across slots): L1=+0.28 L2=+0.45 mid-layers=+0.37 deep=+0.36 | best L10 (+0.53)
- mode C N=4 (prompts=160): layers1-2 slot share front/mid/back per-slot = 0.104/0.132/0.633 (uniform 0.250); r(attn on correct block, acc) across positions: L1=+0.83 L2=+0.83 | full-sample acc curve: L1=+0.75 L2=+0.76 | best layer 10 (+1.00)
- mode C N=8 (prompts=320): layers1-2 slot share front/mid/back per-slot = 0.039/0.064/0.333 (uniform 0.125); r(attn on correct block, acc) across positions: L1=+0.93 L2=+0.93 | full-sample acc curve: L1=+0.84 L2=+0.84 | best layer 10 (+1.00)
- mode C N=16 (prompts=480): layers1-2 slot share front/mid/back per-slot = 0.016/0.031/0.172 (uniform 0.062); r(attn on correct block, acc) across positions: L1=+0.94 L2=+0.94 | full-sample acc curve: L1=+0.93 L2=+0.93 | best layer 19 (+0.98)
- mode C N=32 (prompts=416): layers1-2 slot share front/mid/back per-slot = 0.008/0.015/0.086 (uniform 0.031); r(attn on correct block, acc) across positions: L1=+0.96 L2=+0.96 | full-sample acc curve: L1=+0.96 L2=+0.96 | best layer 8 (+0.96)

### Choice model (mode A, conditional logit over candidate blocks)

All log-liks are 5-fold cross-validated (folds by problem). Features: position basis (pos = first, last, linear, quadratic in relative position); content (cor = is-correct + answer-agreement with other candidates + log length), is-correct indicator (cor), attention share at layer l (attn, z-scored within prompt). ΔLL/prompt = gain from adding attn to pos+cor. top1 = argmax-attention matches choice.

expl = share of the positional log-lik gain (pos+cor vs cor) recovered by replacing position dummies with attention (cor+attn vs cor).

| N | prompts | LL cor | LL pos | LL pos+cor | best layer | ΔLL/prompt (best) | ΔLL L1 | ΔLL L2 | top1 L1 | top1 L2 | top1 best | chance | expl L1 | expl L2 | expl best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 160 | -1.378 | -1.062 | -1.055 | 14 | 0.006 | -0.005 | -0.004 | 0.075 | 0.075 | 0.375 (L27) | 0.250 | 0.37 | 0.38 | 0.38 (L2) |
| 8 | 320 | -2.037 | -1.525 | -1.488 | 8 | 0.022 | -0.002 | 0.006 | 0.106 | 0.106 | 0.256 (L35) | 0.125 | -0.01 | -0.01 | 0.41 (L24) |
| 16 | 480 | -2.805 | -1.416 | -1.433 | 1 | 0.009 | 0.009 | -0.052 | 0.108 | 0.108 | 0.208 (L24) | 0.062 | -0.00 | -0.00 | 0.39 (L24) |
| 32 | 416 | -3.052 | -2.497 | -1.911 | 16 | 0.494 | -0.100 | 0.008 | 0.022 | 0.022 | 0.075 (L24) | 0.031 | -0.01 | 0.01 | 0.24 (L18) |