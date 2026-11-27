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
