# E1 (qwen3-4b): positional prior and single-call calibration


## direct mode (6212 calls)

### Slot prior (random orderings): P(pick slot | slot holds a wrong candidate)

| group | N | calls | slot 1 | slot 2 | middle | last | ratio slot1 / middle | mean P(slot 1) |
|---|---|---|---|---|---|---|---|---|
| comp | 4 | 99 | 0.309 | 0.208 | 0.119 | 0.080 | 2.6 | 0.541 |
| comp | 8 | 87 | 0.660 | 0.078 | 0.022 | 0.020 | 29.5 | 0.689 |
| math500 | 4 | 300 | 0.438 | 0.137 | 0.093 | 0.048 | 4.7 | 0.623 |
| math500 | 8 | 300 | 0.523 | 0.071 | 0.053 | 0.087 | 9.8 | 0.672 |
| math500 | 16 | 300 | 0.691 | 0.040 | 0.019 | 0.048 | 36.0 | 0.790 |
| math500 | 32 | 240 | 0.700 | 0.000 | 0.008 | 0.039 | 86.3 | 0.815 |

### Selection accuracy by method (mixed sets: ≥1 correct and ≥1 wrong candidate)

Δ = method − raw single call, paired bootstrap over problems. calibrated = divide the decision distribution by the slot prior estimated on held-out problems (5-fold, folds by problem).

| group | N | sets | random pick | majority vote | raw (1 call) | calibrated (1 call) [Δ, 95%CI] | calib. m=5 | calib. m=10 | calib. m=20 | maj-first order (1 call) [Δ] | vote over K orders (K calls) [Δ] | cyclic perm. (N calls) [Δ] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comp | 4 | 33 | 43.9 | 53.8 | 65.7 | 68.7 [+3.0 [-2.0,+9.1]] | 66.7 | 68.5 | 69.3 | 63.6 [-2.0 [-13.1,+8.1]] | 72.7 (K=3) [+7.1 [+0.0,+14.1]] | 78.8 [+13.1 [+2.0,+25.3]] |
| comp | 8 | 29 | 40.9 | 55.5 | 52.9 | 48.3 [-4.6 [-12.6,+2.3]] | 45.7 | 45.6 | 47.8 | 62.1 [+9.2 [-9.2,+27.6]] | 55.2 (K=3) [+2.3 [-3.4,+8.0]] | 65.5 [+12.6 [-6.9,+29.9]] |
| math500 | 4 | 100 | 55.8 | 69.0 | 69.3 | 70.7 [+1.3 [-1.7,+4.3]] | 70.4 | 69.9 | 70.2 | 69.0 [-0.3 [-7.0,+6.7]] | 71.0 (K=3) [+1.7 [-1.7,+5.0]] | 71.0 [+1.7 [-4.3,+8.0]] |
| math500 | 8 | 100 | 63.0 | 78.0 | 66.0 | 64.0 [-2.0 [-5.0,+1.0]] | 62.3 | 62.9 | 63.4 | 66.0 [+0.0 [-8.0,+7.7]] | 68.0 (K=3) [+2.0 [-2.3,+6.3]] | 60.0 [-6.0 [-15.3,+3.0]] |
| math500 | 16 | 100 | 66.6 | 81.0 | 67.7 | 63.0 [-4.7 [-9.0,-0.7]] | 61.3 | 62.0 | 62.5 | 68.0 [+0.3 [-8.0,+8.7]] | 71.0 (K=3) [+3.3 [-1.3,+8.0]] | 58.0 [-9.7 [-18.7,-0.3]] |
| math500 | 32 | 80 | 68.8 | 83.1 | 70.8 | 65.4 [-5.4 [-10.8,-0.0]] | 61.8 | 63.3 | 64.1 | 57.5 [-13.3 [-27.5,-0.4]] | 85.0 (K=3) [+14.2 [+8.3,+20.4]] | 55.0 [-16.7 [-32.5,-1.7]] |

## think mode (1768 calls)

forced ending 0.09 (budget hit 0.01); mean max-probability of the decision distribution when not forced 0.989

### Slot prior (random orderings): P(pick slot | slot holds a wrong candidate)

| group | N | calls | slot 1 | slot 2 | middle | last | ratio slot1 / middle | mean P(slot 1) |
|---|---|---|---|---|---|---|---|---|
| comp | 4 | 99 | 0.182 | 0.038 | 0.050 | 0.140 | 3.6 | 0.455 |
| comp | 8 | 87 | 0.180 | 0.078 | 0.042 | 0.060 | 4.3 | 0.478 |
| math500 | 4 | 300 | 0.102 | 0.082 | 0.061 | 0.040 | 1.7 | 0.489 |
| math500 | 8 | 300 | 0.174 | 0.079 | 0.026 | 0.029 | 6.8 | 0.641 |
| math500 | 16 | 300 | 0.234 | 0.110 | 0.015 | 0.000 | 16.1 | 0.707 |
| math500 | 32 | 240 | 0.157 | 0.014 | 0.003 | 0.000 | 48.4 | 0.726 |

### Selection accuracy by method (mixed sets: ≥1 correct and ≥1 wrong candidate)

Δ = method − raw single call, paired bootstrap over problems. calibrated = divide the decision distribution by the slot prior estimated on held-out problems (5-fold, folds by problem).

| group | N | sets | random pick | majority vote | raw (1 call) | calibrated (1 call) [Δ, 95%CI] | calib. m=5 | calib. m=10 | calib. m=20 | maj-first order (1 call) [Δ] | vote over K orders (K calls) [Δ] | cyclic perm. (N calls) [Δ] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| comp | 4 | 33 | 43.9 | 53.8 | 76.8 | 76.8 [+0.0 [+0.0,+0.0]] | 76.7 | 76.8 | 76.8 | 81.8 [+5.1 [-5.1,+15.2]] | 72.7 (K=3) [-4.0 [-10.1,+1.0]] | n/a |
| comp | 8 | 29 | 40.9 | 55.5 | 71.3 | 72.4 [+1.1 [+0.0,+3.4]] | 69.3 | 71.8 | 72.4 | 72.4 [+1.1 [-11.5,+13.8]] | 75.9 (K=3) [+4.6 [-2.3,+11.5]] | n/a |
| math500 | 4 | 100 | 55.8 | 69.0 | 88.0 | 87.7 [-0.3 [-1.0,+0.0]] | 87.5 | 87.8 | 87.8 | 89.0 [+1.0 [-4.7,+6.3]] | 89.0 (K=3) [+1.0 [-1.3,+3.3]] | n/a |
| math500 | 8 | 100 | 63.0 | 78.0 | 86.7 | 86.3 [-0.3 [-1.3,+0.7]] | 84.3 | 85.2 | 86.5 | 92.0 [+5.3 [+0.7,+10.0]] | 88.0 (K=3) [+1.3 [-1.3,+4.3]] | n/a |
| math500 | 16 | 100 | 66.6 | 81.0 | 86.0 | 85.0 [-1.0 [-2.7,+0.3]] | 83.8 | 84.6 | 84.6 | 86.0 [+0.0 [-4.7,+4.0]] | 90.0 (K=3) [+4.0 [+1.3,+7.0]] | n/a |
| math500 | 32 | 80 | 68.8 | 83.1 | 92.5 | 89.2 [-3.3 [-5.4,-1.2]] | 86.3 | 87.6 | 88.5 | 90.0 [-2.5 [-7.9,+2.1]] | 95.0 (K=3) [+2.5 [+0.0,+5.0]] | n/a |