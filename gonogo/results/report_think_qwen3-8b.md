# qwen3-8b: thinking-mode position sweep (selT2)

## Behaviour

| N | problems | finished | acc (all) | acc (finished) | front−mid all [95%CI] | back−mid all [95%CI] | front−mid finished [95%CI] | back−mid finished [95%CI] | trunc front−mid | lift edge−mid [95%CI] | perm p (all) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 50 | 0.85 | 0.695 | 0.813 | +10.0 [+0.0,+21.0] | +4.0 [-8.0,+16.0] | +7.8 [-2.6,+18.0] | +2.8 [-9.7,+14.5] | -4.0 [-14.0,+6.0] | -0.4 [-10.9,+10.3] | 0.282 |
| 8 | 50 | 0.88 | 0.620 | 0.709 | +11.5 [+3.0,+20.0] | -1.5 [-9.5,+6.5] | +10.5 [+2.3,+19.2] | -0.9 [-8.0,+5.7] | -3.0 [-8.0,+2.0] | +2.1 [-4.7,+9.2] | 0.000 |
| 16 | 28 | 0.95 | 0.525 | 0.550 | +15.2 [+7.6,+22.3] | +8.9 [+2.2,+15.6] | +14.8 [+6.9,+22.6] | +8.9 [+1.8,+16.1] | -1.8 [-4.9,+1.3] | +6.6 [+1.7,+11.8] | 0.049 |

lift = P(pick slot k | correct at k) − P(pick slot k | k wrong), finished runs only; edge = first+last quarter.

- N=4: acc|finished by position [0.86, 0.75, 0.82, 0.81]
  truncation by position [0.12, 0.12, 0.2, 0.14]
  lift by slot [0.72, 0.7, 0.8, 0.77] | P(pick k | k wrong) [0.142, 0.047, 0.023, 0.039]
- N=8: acc|finished by position [0.87, 0.7, 0.66, 0.67, 0.65, 0.76, 0.65, 0.7]
  truncation by position [0.08, 0.12, 0.12, 0.1, 0.2, 0.1, 0.2, 0.08]
  lift by slot [0.73, 0.65, 0.64, 0.65, 0.6, 0.74, 0.64, 0.68] | P(pick k | k wrong) [0.141, 0.052, 0.02, 0.02, 0.055, 0.02, 0.01, 0.016]
- N=16: acc|finished by position [0.7, 0.67, 0.67, 0.52, 0.54, 0.57, 0.46, 0.52, 0.4, 0.52, 0.48, 0.43, 0.59, 0.54, 0.63, 0.56]
  truncation by position [0.04, 0.04, 0.04, 0.04, 0.0, 0.0, 0.07, 0.11, 0.11, 0.04, 0.11, 0.0, 0.04, 0.07, 0.04, 0.04]
  lift by slot [0.43, 0.55, 0.64, 0.5, 0.53, 0.57, 0.45, 0.52, 0.4, 0.51, 0.48, 0.43, 0.59, 0.54, 0.62, 0.54] | P(pick k | k wrong) [0.27, 0.115, 0.022, 0.02, 0.005, 0.003, 0.007, 0.0, 0.002, 0.005, 0.002, 0.003, 0.002, 0.002, 0.008, 0.012]

## Attention from reasoning tokens (680 sequences; head mean; shares normalised over candidate blocks)

early = first quarter of the reasoning trace; all = whole trace; dec = token predicting the chosen index.

- N=4 (120 seqs, 30 problems):
    [early] L1-2 F/M/B×N = 0.57/0.76/1.91; L13-24 F/M/B×N = 1.32/0.77/1.14; L25-36 F/M/B×N = 1.98/0.61/0.80 | r(correct-block attn, acc|fin) across positions: L1 -0.10 L2 -0.10, best L35 +0.86
    [all] L1-2 F/M/B×N = 0.68/0.85/1.61; L13-24 F/M/B×N = 1.32/0.77/1.14; L25-36 F/M/B×N = 2.01/0.61/0.77 | r(correct-block attn, acc|fin) across positions: L1 -0.12 L2 -0.13, best L22 +0.87
    [dec] L1-2 F/M/B×N = 0.84/1.00/1.17; L13-24 F/M/B×N = 1.54/0.75/0.96; L25-36 F/M/B×N = 2.09/0.59/0.73 | r(correct-block attn, acc|fin) across positions: L1 -0.84 L2 -0.22, best L36 +0.91
    choice model (finished, n=101): CV LL/prompt cor -0.751, pos+cor -0.843 (position gain -0.093)
    [early] ΔLL from attention: L1 -0.025 L2 -0.044 best L3 +0.031 | expl L1 nan L2 nan best n/a | top1 L1 0.23 L2 0.22 best 0.36 (chance 0.25)
    [dec] ΔLL from attention: L1 -0.015 L2 +0.027 best L2 +0.027 | expl L1 nan L2 nan best n/a | top1 L1 0.24 L2 0.24 best 0.46 (chance 0.25)
- N=8 (240 seqs, 30 problems):
    [early] L1-2 F/M/B×N = 0.51/0.74/2.02; L13-24 F/M/B×N = 1.15/0.79/1.28; L25-36 F/M/B×N = 1.51/0.77/0.95 | r(correct-block attn, acc|fin) across positions: L1 -0.30 L2 -0.27, best L25 +0.86
    [all] L1-2 F/M/B×N = 0.60/0.84/1.71; L13-24 F/M/B×N = 1.11/0.80/1.28; L25-36 F/M/B×N = 1.50/0.77/0.96 | r(correct-block attn, acc|fin) across positions: L1 -0.36 L2 -0.29, best L31 +0.85
    [dec] L1-2 F/M/B×N = 0.67/0.94/1.46; L13-24 F/M/B×N = 1.36/0.77/1.10; L25-36 F/M/B×N = 1.52/0.75/0.97 | r(correct-block attn, acc|fin) across positions: L1 -0.28 L2 -0.36, best L23 +0.91
    choice model (finished, n=201): CV LL/prompt cor -1.433, pos+cor -1.354 (position gain +0.079)
    [early] ΔLL from attention: L1 -0.028 L2 -0.032 best L32 +0.028 | expl L1 0.16 L2 0.15 best L21 0.26 | top1 L1 0.09 L2 0.09 best 0.24 (chance 0.12)
    [dec] ΔLL from attention: L1 -0.010 L2 -0.008 best L26 +0.027 | expl L1 -0.13 L2 0.51 best L25 0.77 | top1 L1 0.14 L2 0.11 best 0.29 (chance 0.12)
- N=16 (320 seqs, 20 problems):
    [early] L1-2 F/M/B×N = 0.41/0.69/2.21; L13-24 F/M/B×N = 0.97/0.79/1.46; L25-36 F/M/B×N = 1.14/0.92/1.03 | r(correct-block attn, acc|fin) across positions: L1 +0.21 L2 +0.15, best L8 +0.45
    [all] L1-2 F/M/B×N = 0.47/0.82/1.89; L13-24 F/M/B×N = 0.92/0.82/1.44; L25-36 F/M/B×N = 1.12/0.93/1.03 | r(correct-block attn, acc|fin) across positions: L1 +0.12 L2 +0.10, best L8 +0.53
    [dec] L1-2 F/M/B×N = 0.47/0.94/1.66; L13-24 F/M/B×N = 1.19/0.80/1.20; L25-36 F/M/B×N = 1.17/0.93/0.98 | r(correct-block attn, acc|fin) across positions: L1 -0.18 L2 +0.07, best L8 +0.67
    choice model (finished, n=308): CV LL/prompt cor -1.812, pos+cor -1.337 (position gain +0.475)
    [early] ΔLL from attention: L1 -0.001 L2 -0.006 best L5 +0.032 | expl L1 0.07 L2 0.05 best L21 0.31 | top1 L1 0.05 L2 0.05 best 0.25 (chance 0.06)
    [dec] ΔLL from attention: L1 -0.003 L2 +0.009 best L4 +0.029 | expl L1 0.13 L2 0.12 best L14 0.50 | top1 L1 0.06 L2 0.09 best 0.26 (chance 0.06)