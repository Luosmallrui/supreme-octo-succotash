# Experiment collaboration and runbook

GitHub `main` is the shared source for code and results. The GPU server has two project directories. `~/attnbasin_gonogo` is the original run directory where E3 v1 is currently running. `~/supreme-octo-succotash/gonogo` is the Git working copy for later experiments. Outputs do not sync automatically between them.

## Start of each work session

Run `git pull --ff-only origin main` in the local repository. On the GPU server, update the Git working copy with:

```bash
cd ~/supreme-octo-succotash
git status --short --branch
git pull --ff-only origin main
```

Commit and push changes from the working copy where they were made. Pull remote updates before pushing, and resolve any conflicts. The collaborators share one GPU. Check `tmux ls`, `nvidia-smi`, and the queue logs before starting a job.

## Current job and result locations

At 17:09 China Standard Time on November 26, 2026, `attention_e3` was running the final E3 v1 step, `e3_think_qwen3-8b`, in the original directory. About 2,832 of 3,514 requests had been saved. Use the server logs for the current status.

```bash
ssh gpu
cd ~/attnbasin_gonogo
tmux ls
tail -12 logs/queue11.log
tail -5 logs/e3_think_qwen3-8b.log
nvidia-smi
```

Queue 11 writes raw results and reports to `~/attnbasin_gonogo/out/e3_readout/`. Step logs are in `~/attnbasin_gonogo/logs/e3_*.log`. Its completion marker is `QUEUE11_DONE` in `logs/queue11.log`. Wait for the writer to finish before copying the active JSONL file.

| Experiment | Entry point | Raw results and reports | Queue log |
|---|---|---|---|
| E1 candidate selection and calibration | `e1_run.py`, `e1_analyze.py` | `gonogo/out/e1_*`, `gonogo/results/report_e1_*.md` | `gonogo/logs/queue.log`, `queue9.log` |
| E2 physical order and printed labels | `run_queue10.sh` | `gonogo/out/e2_position_label/` | `gonogo/logs/queue10.log` |
| E3 v1, currently finishing | `run_queue11.sh` | Original directory `out/e3_readout/` | Original directory `logs/queue11.log` |
| E3 v2 | `run_queue12.sh` | Git working copy `gonogo/out/e3v2_readout/` | Git working copy `gonogo/logs/queue12.log` |
| E4 attention and position intervention | `run_queue13.sh` | Git working copy `gonogo/out/e4_mask_position/` | Git working copy `gonogo/logs/queue13.log` |
| E5 decoy in slot 1, E6 order-balanced answer duels | `run_queue14.sh` | Git working copy `gonogo/out/e5_decoy/`, `gonogo/out/e6_duels/` | Git working copy `gonogo/logs/queue14.log` |
| Qwen3 cross-judging (E2) and bidirectional E4 conditions | `run_queue15.sh` | Git working copy `gonogo/out/e2_position_label/cross/`, `gonogo/out/e4_mask_position/e4_bidir_*` | Git working copy `gonogo/logs/queue15.log` |
| Other judges (E2, E4), label→answer lookup (E7), sampled thinking readouts (E3) | `run_queue16.sh` | Git working copy `gonogo/out/e7_label_answer/`, `gonogo/out/e2_position_label/cross/`, `gonogo/out/e4_mask_position/judges/`, `gonogo/out/e3v2_readout/e3_think_sampled_*` | Git working copy `gonogo/logs/queue16.log` |

Figures and summary reports are in `gonogo/results/`. The `run_queue*.sh` scripts specify inputs, outputs, and parameters for each queue.

## After E3 v1 finishes in the original directory

Confirm `QUEUE11_DONE`, then copy its final data to the Git working copy. These commands copy E3 v1 data and its queue log. They leave the newer code in the Git working copy intact.

```bash
grep QUEUE11_DONE ~/attnbasin_gonogo/logs/queue11.log
rsync -a ~/attnbasin_gonogo/out/e3_readout/ ~/supreme-octo-succotash/gonogo/out/e3_readout/
mkdir -p ~/supreme-octo-succotash/gonogo/logs
cp ~/attnbasin_gonogo/logs/queue11.log ~/supreme-octo-succotash/gonogo/logs/
```

E3 v2 reuses the complete E3 v1 generations. Once the data is in place, start queue 12 from the Git working copy. Start each queue only once. Queue 12 checks before using the GPU that both `out/e3_readout/e3_think_*.jsonl` files contain every request in their manifests, and stops with a message if one is incomplete. Set `E3V2_ALLOW_PARTIAL_REUSE=1` only to regenerate the missing rows on purpose.

```bash
cd ~/supreme-octo-succotash/gonogo
tmux new-session -d -s attention_e3v2 'bash run_queue12.sh >>logs/queue12.log 2>&1'
tail -f logs/queue12.log
```

After queue 12 reports `QUEUE12_DONE`, start queue 13. Queue 13 runs a self-test for each model before the main experiment.

```bash
cd ~/supreme-octo-succotash/gonogo
grep QUEUE12_DONE logs/queue12.log
tmux new-session -d -s attention_e4 'bash run_queue13.sh >>logs/queue13.log 2>&1'
tail -f logs/queue13.log
```

Queue 13 reports `QUEUE13_DONE` on success. For failed steps, inspect `logs/e3v2_*.log` or `logs/e4_*.log`, fix the problem, and follow the relevant script's resume behavior.

## Queue 14: E5 decoy and E6 answer duels

Both experiments use the E1 sets and candidates already in `gonogo/out/` (`e1_sets_<model>.jsonl`, `graded_<model>.jsonl`) and the E1 pre-filled readout without reasoning. Design and pre-registered success criteria: `gonogo/out/method_bayes_select/run_designs_2026-11-27.md`. Expected GPU time per model: E6 10–15 minutes (3,962 two-candidate prompts for Qwen3-8B), E5 20–30 minutes (about 2,600 prompts). The queue waits until GPU 0 is free.

Pull the code, then run a short smoke test that checks vLLM and the readout on 16 prompts:

```bash
cd ~/supreme-octo-succotash && git pull --ff-only origin main
cd gonogo
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/pylib" ~/venvs/vllm/bin/python3 e6_run.py --model qwen3-8b --limit 16 --out /tmp/e6_smoke.jsonl
head -c 600 /tmp/e6_smoke.jsonl
```

Each row should have `p` with two probabilities that sum to about 1 and `p_mass` close to 1. Then start the queue:

```bash
tmux new-session -d -s attention_e56 'bash run_queue14.sh >>logs/queue14.log 2>&1'
tail -f logs/queue14.log
```

The queue verifies each output and reports `QUEUE14_DONE`. Step logs are `logs/e6_<model>.log` and `logs/e5_<model>.log`. Outputs are written as `.tmp` files and renamed only after the check passes. Then commit and push the results:

```bash
cd ~/supreme-octo-succotash
git add gonogo/out/e5_decoy gonogo/out/e6_duels gonogo/logs/queue14.log gonogo/logs/e5_*.log gonogo/logs/e6_*.log
git commit -m 'Add E5 decoy and E6 duel results'
git pull --rebase origin main
git push origin main
```

The analyses (`e5_analyze.py`, `e6_analyze.py`) run locally after the results are pulled.

## Queue 15: cross-judging inside Qwen3 and bidirectional inter-candidate attention

Reviewer follow-ups, all without reasoning on the E2 layouts. Qwen3-4B judges Qwen3-8B's candidates and Qwen3-8B judges Qwen3-4B's (`e2_run.py --judge`), which separates self- from other-judging within one family. E4 gains two conditions in which candidate blocks attend to each other in both directions (PINE-style), with sequential and with shared position ids (`e4_run.py --conditions bidir:seq,bidir:shared`); the self-test checks that the shared-position version is order-invariant. The queue starts after queue 14 has finished and takes about 2.5 hours.

```bash
cd ~/supreme-octo-succotash && git pull --ff-only origin main
cd gonogo
tmux new-session -d -s attention_q15 'bash run_queue15.sh >>logs/queue15.log 2>&1'
tail -f logs/queue15.log
```

To also run a larger judge on Qwen3-8B's candidates, start the queue with its model path, for example `LARGE_JUDGE=/data/home/202412012250/models/Qwen3-14B`. Qwen3-32B in bf16 needs an 80 GB GPU. The queue reports `QUEUE15_DONE`; step logs are `logs/e2_cross_*.log`, `logs/e4_bidir_*.log` and `logs/e4_bidir_selftest_*.log`. Then commit:

```bash
cd ~/supreme-octo-succotash
git add gonogo/out/e2_position_label/cross gonogo/out/e4_mask_position/e4_bidir_*.jsonl \
    gonogo/out/e4_mask_position/report_e4_with_bidir_*.md gonogo/logs/queue15.log gonogo/logs/e2_cross_*.log gonogo/logs/e4_bidir_*.log
git commit -m 'Add Qwen3 cross-judging and bidirectional E4 results'
git pull --rebase origin main
git push origin main
```

## Queue 16: other model families, label binding, sampled thinking readouts

Reviewer follow-ups (`run_queue16.sh`). They run after queue 15, without reasoning unless noted:
1. **E7:** the label→answer lookup (`e7_label_answer.py`) asks for the final answer of the candidate printed as "Solution L".
2. **E2 crossed design:** run with judges from other families and a larger Qwen3 on Qwen3-8B's layouts.
3. **E4 intervention:** the 2×2 plus the bidirectional conditions, on one Llama-architecture judge.
4. **E3 thinking-mode readouts:** label, answer and solo, rerun with Qwen3's recommended sampling instead of greedy decoding.

Expected GPU time:
- E7: a few minutes per model.
- E2: about 15 minutes per 8–12B judge, about 40 minutes for Qwen3-32B.
- E4: about 3 hours for an 8B judge.
- E3 resample: 1–2 hours per model.

Download the judges first, about 105 GB in total (Qwen3-32B in bf16 needs an 80 GB GPU). The `modelscope` CLI is `pip install modelscope`; Llama and Gemma may ask you to accept their licence on the model page first.

```bash
df -h /data/home/202412012250/models
cd /data/home/202412012250/models
modelscope download --model LLM-Research/Meta-Llama-3.1-8B-Instruct --local_dir Llama-3.1-8B-Instruct
modelscope download --model LLM-Research/gemma-3-12b-it --local_dir gemma-3-12b-it
modelscope download --model Qwen/Qwen3-32B --local_dir Qwen3-32B
```

Then start the queue. Every judge is optional: leave a path out of `JUDGES`, or unset `E4_JUDGE`, to skip it. `E3_SAMPLE=0` skips the resampling.

```bash
cd ~/supreme-octo-succotash && git pull --ff-only origin main
cd gonogo
M=/data/home/202412012250/models
tmux new-session -d -s attention_q16 "JUDGES='$M/Llama-3.1-8B-Instruct $M/gemma-3-12b-it $M/Qwen3-32B' E4_JUDGE=$M/Llama-3.1-8B-Instruct bash run_queue16.sh >>logs/queue16.log 2>&1"
tail -f logs/queue16.log
```

The queue reports `QUEUE16_DONE`. Step logs are `logs/e7_*.log`, `logs/e2_cross_*.log`, `logs/e4_selftest_*.log`, `logs/e4_*.log` and `logs/e3_think_sampled_*.log`. If the E4 selftest fails, the queue stops before the E4 run; send the selftest log. Then commit:

```bash
cd ~/supreme-octo-succotash
git add gonogo/out/e7_label_answer gonogo/out/e2_position_label/cross gonogo/out/e4_mask_position/judges \
    gonogo/out/e4_mask_position/report_e4_checks.md gonogo/out/e3v2_readout/e3_think_sampled_*.jsonl \
    gonogo/out/e3v2_readout/report_e3_think_sampled_*.md gonogo/out/e3v2_readout/report_e3_checks.md \
    gonogo/logs/queue16.log gonogo/logs/e7_*.log gonogo/logs/e2_cross_*.log gonogo/logs/e4_*.log gonogo/logs/e3_think_sampled_*.log
git commit -m 'Add queue 16: other judges, label lookup, sampled thinking readouts'
git pull --rebase origin main
git push origin main
```

## Queue 17: count-free duel matrices (route-2 check)

Full order-balanced duel matrices over individual candidates (no grouping by answer) on the 130 E1 sets with N = 8, 7,280 two-candidate prompts per model (`e6_run.py --all-candidates --n 8`). Sparse duel schedules are simulated offline from these matrices. The queue waits for queue 16 and for GPU 0; about 15–25 minutes per model.

```bash
cd ~/supreme-octo-succotash && git pull --ff-only origin main
cd gonogo
tmux new-session -d -s attention_q17 'bash run_queue17.sh >>logs/queue17.log 2>&1'
```

It reports `QUEUE17_DONE`. Then commit:

```bash
cd ~/supreme-octo-succotash
git add gonogo/out/e6_duels/e6_allcand_N8_*.jsonl gonogo/logs/queue17.log gonogo/logs/e6_allcand_*.log
git commit -m 'Add count-free duel matrices (queue 17)'
git pull --rebase origin main
git push origin main
```

## Queue 18: Llama-3.1-8B duels for the pre-registered confirmation

Llama-3.1-8B-Instruct judges Qwen3-8B's natural sets through order-balanced duels: 3,962 answer duels (as E6) and 7,280 all-candidate duels on the N = 8 sets (as queue 17). These data confirm the selection rules on a second family, so **do not analyse them** until the analysis plan `gonogo/out/e6_duels/prereg_llama.md` is committed. About 30–60 minutes on GPU 0. Set `LLAMA` to the local model directory used for the Llama E4 run.

```bash
cd ~/supreme-octo-succotash && git pull --ff-only origin main
cd gonogo
LLAMA=/path/to/Llama-3.1-8B-Instruct tmux new-session -d -s attention_q18 'bash run_queue18.sh >>logs/queue18.log 2>&1'
```

It reports `QUEUE18_DONE`. Then commit:

```bash
cd ~/supreme-octo-succotash
git add gonogo/out/e6_duels/e6_llama-3.1-8b_on_qwen3-8b.jsonl gonogo/out/e6_duels/e6_allcand_N8_llama-3.1-8b_on_qwen3-8b.jsonl \
    gonogo/logs/queue18.log gonogo/logs/e6_llama-*.log
git commit -m 'Add Llama-3.1-8B duels (queue 18)'
git pull --rebase origin main
git push origin main
```

## Queue 19: Llama-3.1-8B listwise selection on the natural sets

The listwise half of the same confirmation: Llama-3.1-8B-Instruct selects on Qwen3-8B's 443 natural sets (E1 design, direct; 6,224 calls: three random orderings, the majority-first ordering and the cyclic shifts). `e1_run.py --judge` reads the labels from single label tokens, since Llama 3 encodes 1–32 as one token each. A 16-call smoke test on N ≥ 16 sets runs first and must read label tokens with at least 0.9 mean probability mass. The queue waits for queue 18 and GPU 0; about 1.5–3 hours. As for queue 18, do not analyse the output yourself; the analysis is `confirm_rules.py` as registered in `gonogo/out/e6_duels/prereg_llama.md`.

```bash
cd ~/supreme-octo-succotash && git pull --ff-only origin main
cd gonogo
LLAMA=/path/to/Llama-3.1-8B-Instruct tmux new-session -d -s attention_q19 'bash run_queue19.sh >>logs/queue19.log 2>&1'
```

It reports `QUEUE19_DONE`. Then commit:

```bash
cd ~/supreme-octo-succotash
git add gonogo/out/e1_direct_llama-3.1-8b_on_qwen3-8b.jsonl gonogo/out/e1_smoke_llama-3.1-8b.jsonl \
    gonogo/logs/queue19.log gonogo/logs/e1_llama-3.1-8b.log
git commit -m 'Add Llama-3.1-8B listwise natural-set calls (queue 19)'
git pull --rebase origin main
git push origin main
```

## Sync new results

In the GPU Git working copy, review, commit, and push the completed outputs. Raw results can take several minutes to push.

```bash
cd ~/supreme-octo-succotash
git status --short
git add gonogo/out/e3_readout gonogo/out/e3v2_readout gonogo/out/e4_mask_position gonogo/logs
git commit -m 'Update experiment results'
git pull --rebase origin main
git push origin main
```

Then run `git pull --ff-only origin main` locally. Check `git status` before committing and add only code and results you have reviewed. Do not commit a JSONL file while a job is still writing it.
