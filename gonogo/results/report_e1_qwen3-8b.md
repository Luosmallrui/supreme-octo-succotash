# E1 (qwen3-8b): positional prior and single-call calibration


## direct mode (6224 calls)

### Slot prior (random orderings): P(pick slot | slot holds a wrong candidate)

| group | N | calls | slot 1 | slot 2 | middle | last | ratio slot1 / middle | mean P(slot 1) |
|---|---|---|---|---|---|---|---|---|
| comp | 4 | 99 | 0.536 | 0.094 | 0.078 | 0.091 | 6.8 | 0.569 |
| comp | 8 | 90 | 0.545 | 0.019 | 0.024 | 0.051 | 22.5 | 0.572 |
| math500 | 4 | 300 | 0.398 | 0.055 | 0.101 | 0.053 | 4.0 | 0.556 |
| math500 | 8 | 300 | 0.708 | 0.020 | 0.059 | 0.104 | 12.1 | 0.627 |
| math500 | 16 | 300 | 0.519 | 0.009 | 0.030 | 0.094 | 17.3 | 0.583 |
| math500 | 32 | 240 | 0.605 | 0.012 | 0.013 | 0.074 | 45.8 | 0.650 |

### Selection accuracy by method (mixed sets: ≥1 correct and ≥1 wrong candidate)

Δ = method − raw single call, paired bootstrap over problems. calibrated = divide the decision distribution by the slot prior estimated on held-out problems (5-fold, folds by problem).

| group | N | sets | random pick | majority vote | raw (1 call) | calibrated (1 call) [Δ, 95%CI] | calib. m=5 | calib. m=10 | calib. m=20 | maj-first order (1 call) [Δ] | vote over K orders (K calls) [Δ] | cyclic perm. (N calls) [Δ] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comp | 4 | 33 | 42.4 | 55.3 | 55.6 | 58.6 [+3.0 [-4.0,+10.1]] | 58.3 | 57.8 | 57.7 | 57.6 [+2.0 [-10.1,+13.1]] | 57.6 (K=3) [+2.0 [-4.0,+8.1]] | 63.6 [+8.1 [-1.0,+17.2]] |
| comp | 8 | 30 | 38.8 | 48.2 | 54.4 | 51.1 [-3.3 [-12.2,+4.4]] | 50.8 | 51.6 | 51.2 | 46.7 [-7.8 [-21.1,+4.4]] | 60.0 (K=3) [+5.6 [-2.2,+15.6]] | 50.0 [-4.4 [-16.7,+7.8]] |
| math500 | 4 | 100 | 57.5 | 75.8 | 72.0 | 71.7 [-0.3 [-2.7,+2.0]] | 72.7 | 71.9 | 71.4 | 78.0 [+6.0 [-0.3,+13.0]] | 72.0 (K=3) [+0.0 [-3.7,+3.3]] | 76.0 [+4.0 [-2.3,+10.0]] |
| math500 | 8 | 100 | 62.9 | 76.2 | 57.7 | 52.7 [-5.0 [-9.7,-0.3]] | 52.7 | 52.7 | 52.6 | 63.0 [+5.3 [-2.0,+12.7]] | 62.0 (K=3) [+4.3 [+0.0,+8.7]] | 52.0 [-5.7 [-13.7,+2.3]] |
| math500 | 16 | 100 | 65.0 | 81.0 | 63.7 | 57.0 [-6.7 [-12.0,-1.3]] | 54.2 | 54.7 | 55.6 | 69.0 [+5.3 [-2.0,+12.3]] | 73.0 (K=3) [+9.3 [+4.7,+14.0]] | 49.0 [-14.7 [-24.3,-5.0]] |
| math500 | 32 | 80 | 68.9 | 78.1 | 65.8 | 57.5 [-8.3 [-15.8,-1.7]] | 56.3 | 56.9 | 57.5 | 62.5 [-3.3 [-15.4,+7.9]] | 72.5 (K=3) [+6.7 [+1.7,+11.7]] | 55.0 [-9.2 [-25.8,+6.7]] |

## think mode (1772 calls)

forced ending 0.06 (budget hit 0.03); mean max-probability of the decision distribution when not forced 0.987

### Slot prior (random orderings): P(pick slot | slot holds a wrong candidate)

| group | N | calls | slot 1 | slot 2 | middle | last | ratio slot1 / middle | mean P(slot 1) |
|---|---|---|---|---|---|---|---|---|
| comp | 4 | 99 | 0.179 | 0.075 | 0.077 | 0.091 | 2.3 | 0.411 |
| comp | 8 | 90 | 0.164 | 0.000 | 0.024 | 0.068 | 6.7 | 0.423 |
| math500 | 4 | 300 | 0.083 | 0.008 | 0.008 | 0.060 | 10.1 | 0.474 |
| math500 | 8 | 300 | 0.123 | 0.010 | 0.018 | 0.032 | 6.6 | 0.619 |
| math500 | 16 | 300 | 0.139 | 0.027 | 0.010 | 0.021 | 14.5 | 0.642 |
| math500 | 32 | 240 | 0.276 | 0.036 | 0.002 | 0.000 | n/a (middle≈0) | 0.737 |

### Selection accuracy by method (mixed sets: ≥1 correct and ≥1 wrong candidate)

Δ = method − raw single call, paired bootstrap over problems. calibrated = divide the decision distribution by the slot prior estimated on held-out problems (5-fold, folds by problem).

| group | N | sets | random pick | majority vote | raw (1 call) | calibrated (1 call) [Δ, 95%CI] | calib. m=5 | calib. m=10 | calib. m=20 | maj-first order (1 call) [Δ] | vote over K orders (K calls) [Δ] | cyclic perm. (N calls) [Δ] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comp | 4 | 33 | 42.4 | 55.3 | 75.8 | 74.7 [-1.0 [-3.0,+0.0]] | 74.8 | 74.9 | 74.8 | 78.8 [+3.0 [-4.0,+11.1]] | 69.7 (K=3) [-6.1 [-11.1,-1.0]] | n/a |
| comp | 8 | 30 | 38.8 | 48.2 | 76.7 | 74.4 [-2.2 [-5.6,+0.0]] | 71.7 | 73.2 | 73.7 | 63.3 [-13.3 [-24.4,-4.4]] | 76.7 (K=3) [+0.0 [-6.7,+7.8]] | n/a |
| math500 | 4 | 100 | 57.5 | 75.8 | 93.0 | 93.3 [+0.3 [+0.0,+1.0]] | 91.7 | 93.1 | 93.1 | 93.0 [+0.0 [-4.7,+4.7]] | 94.0 (K=3) [+1.0 [-1.0,+3.0]] | n/a |
| math500 | 8 | 100 | 62.9 | 76.2 | 90.3 | 89.3 [-1.0 [-2.3,+0.0]] | 86.8 | 87.3 | 88.8 | 91.0 [+0.7 [-2.7,+4.0]] | 92.0 (K=3) [+1.7 [-0.3,+3.7]] | n/a |
| math500 | 16 | 100 | 65.0 | 81.0 | 89.7 | 88.0 [-1.7 [-3.0,-0.3]] | 85.0 | 85.7 | 86.5 | 90.0 [+0.3 [-5.0,+5.3]] | 89.0 (K=3) [-0.7 [-3.3,+1.7]] | n/a |
| math500 | 32 | 80 | 68.9 | 78.1 | 89.6 | 86.2 [-3.3 [-7.1,-0.4]] | 80.0 | 82.7 | 84.8 | 90.0 [+0.4 [-3.3,+4.2]] | 91.2 (K=3) [+1.7 [+0.0,+3.8]] | n/a |