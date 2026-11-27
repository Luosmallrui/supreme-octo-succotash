# E1 (qwen3-8b): positional prior and single-call calibration


(no e1_direct results yet)


(no e1_think results yet)


## api-deepseek-flash-prefill-direct mode (3360 calls)

model deepseek-flash; label distribution from API logprobs. errors 0.005; forced ending 0.00; no index even after forcing 0.005; mean reasoning tokens 0

### Slot prior (random orderings): P(pick slot | slot holds a wrong candidate)

| group | N | calls | slot 1 | slot 2 | middle | last | ratio slot1 / middle | mean P(slot 1) |
|---|---|---|---|---|---|---|---|---|
| comp | 4 | 99 | 0.500 | 0.170 | 0.093 | 0.000 | 5.4 | 0.614 |
| comp | 8 | 90 | 0.491 | 0.038 | 0.043 | 0.000 | 11.4 | 0.492 |
| math500 | 4 | 300 | 0.459 | 0.055 | 0.053 | 0.023 | 8.6 | 0.613 |
| math500 | 8 | 300 | 0.415 | 0.070 | 0.030 | 0.040 | 14.0 | 0.532 |
| math500 | 16 | 300 | 0.380 | 0.054 | 0.015 | 0.021 | 25.4 | 0.474 |
| math500 | 32 | 240 | 0.434 | 0.048 | 0.009 | 0.000 | 46.7 | 0.513 |

### Selection accuracy by method (mixed sets: ≥1 correct and ≥1 wrong candidate)

Δ = method − raw single call, paired bootstrap over problems. calibrated = divide the decision distribution by the slot prior estimated on held-out problems (5-fold, folds by problem).

| group | N | sets | random pick | majority vote | raw (1 call) | calibrated (1 call) [Δ, 95%CI] | calib. m=5 | calib. m=10 | calib. m=20 | maj-first order (1 call) [Δ] | vote over K orders (K calls) [Δ] | cyclic perm. (N calls) [Δ] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comp | 4 | 33 | 42.4 | 55.3 | 61.6 | 66.7 [+5.1 [-4.0,+13.1]] | 65.7 | 65.7 | 66.6 | 57.6 [-4.0 [-14.1,+5.1]] | 72.7 (K=3) [+11.1 [+4.0,+19.2]] | 69.7 [+8.1 [-2.0,+18.2]] |
| comp | 8 | 30 | 38.8 | 48.2 | 54.4 | 57.8 [+3.3 [-1.1,+8.9]] | 57.7 | 57.8 | 57.4 | 56.7 [+2.2 [-7.8,+13.3]] | 50.0 (K=3) [-4.4 [-11.1,+2.2]] | 63.3 [+8.9 [-5.6,+24.4]] |
| math500 | 4 | 100 | 57.5 | 75.8 | 74.3 | 77.3 [+3.0 [-0.3,+6.7]] | 77.3 | 77.3 | 77.4 | 81.0 [+6.7 [+0.3,+13.0]] | 75.0 (K=3) [+0.7 [-2.7,+4.0]] | 82.0 [+7.7 [-0.3,+15.3]] |
| math500 | 8 | 100 | 62.9 | 76.2 | 77.3 | 73.7 [-3.7 [-8.0,+0.7]] | 73.5 | 74.1 | 74.1 | 80.0 [+2.7 [-2.7,+7.7]] | 81.0 (K=3) [+3.7 [+0.3,+7.0]] | 76.0 [-1.3 [-7.0,+4.0]] |
| math500 | 16 | 100 | 65.0 | 81.0 | 78.3 | 74.0 [-4.3 [-8.0,-0.7]] | 74.7 | 75.5 | 74.6 | 84.0 [+5.7 [+0.0,+11.3]] | 80.0 (K=3) [+1.7 [-2.0,+5.0]] | n/a |
| math500 | 32 | 80 | 68.9 | 78.1 | 77.5 | 67.1 [-10.4 [-19.2,-2.5]] | 67.0 | 67.1 | 66.6 | 82.5 [+5.0 [-1.2,+10.8]] | 81.2 (K=3) [+3.8 [+0.0,+7.9]] | n/a |