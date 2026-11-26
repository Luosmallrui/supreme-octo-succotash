# 实验协作与运行记录

本仓库的代码与结果通过 GitHub `main` 共享。服务器有两个目录。`~/attnbasin_gonogo` 是此前建立的运行目录，正在执行 E3 第一版。`~/supreme-octo-succotash/gonogo` 是 Git 工作副本，后续实验从这里启动。两个目录的输出互不自动同步。

## 每次开始工作

在本地仓库执行 `git pull --ff-only origin main`。在 GPU 的 Git 工作副本执行下面的命令。

```bash
cd ~/supreme-octo-succotash
git status --short --branch
git pull --ff-only origin main
```

修改代码或报告后，在修改所在的工作副本提交并推送。推送前再次拉取远端更新，遇到冲突时先处理冲突。共同使用一张 GPU，启动前检查 `tmux ls`、`nvidia-smi` 和队列日志，避免两人启动同一队列。

## 当前运行与结果位置

截至 2026-11-26 17:09（北京时间），旧目录中的 `attention_e3` 正在运行 `run_queue11.sh` 的最后一步 `e3_think_qwen3-8b`。已保存约 2,832/3,514 条。运行进度以服务器日志为准。

```bash
ssh gpu
cd ~/attnbasin_gonogo
tmux ls
tail -12 logs/queue11.log
tail -5 logs/e3_think_qwen3-8b.log
nvidia-smi
```

队列 11 的原始输出和报告在 `~/attnbasin_gonogo/out/e3_readout/`，每一步的日志在 `~/attnbasin_gonogo/logs/e3_*.log`。成功标记为 `logs/queue11.log` 中的 `QUEUE11_DONE`。运行中的 JSONL 会继续增长，复制前先核对写入已结束。

| 实验 | 执行脚本 | 原始结果及报告 | 队列日志 |
|---|---|---|---|
| E1 候选集选择、校准 | `e1_run.py`、`e1_analyze.py` | `gonogo/out/e1_*`、`gonogo/results/report_e1_*.md` | `gonogo/logs/queue.log`、`queue9.log` |
| E2 顺序与打印编号 | `run_queue10.sh` | `gonogo/out/e2_position_label/` | `gonogo/logs/queue10.log` |
| E3 第一版，正在收尾 | `run_queue11.sh` | 旧目录 `out/e3_readout/` | 旧目录 `logs/queue11.log` |
| E3 第二版 | `run_queue12.sh` | Git 工作副本 `gonogo/out/e3v2_readout/` | Git 工作副本 `gonogo/logs/queue12.log` |
| E4 注意力与位置干预 | `run_queue13.sh` | Git 工作副本 `gonogo/out/e4_mask_position/` | Git 工作副本 `gonogo/logs/queue13.log` |

图和汇总报告放在 `gonogo/results/`。各队列步骤的具体输入、输出和参数以相应 `run_queue*.sh` 为准。

## 旧目录的 E3 完成后

先确认 `QUEUE11_DONE`，再把最终结果复制到 Git 工作副本。这个复制只针对 E3 第一版的数据和日志，保留工作副本中的新版代码。

```bash
grep QUEUE11_DONE ~/attnbasin_gonogo/logs/queue11.log
rsync -a ~/attnbasin_gonogo/out/e3_readout/ ~/supreme-octo-succotash/gonogo/out/e3_readout/
mkdir -p ~/supreme-octo-succotash/gonogo/logs
cp ~/attnbasin_gonogo/logs/queue11.log ~/supreme-octo-succotash/gonogo/logs/
```

E3 第二版复用第一版的完整生成文本。准备好数据后，从 Git 工作副本启动队列 12。确认队列 12 成功后再启动队列 13，队列 13 会先运行两个模型的自检。每个队列只启动一次。

```bash
cd ~/supreme-octo-succotash/gonogo
tmux new-session -d -s attention_e3v2 'bash run_queue12.sh >>logs/queue12.log 2>&1'
tail -f logs/queue12.log
```

队列 12 出现成功标记 `QUEUE12_DONE` 后，运行以下命令。

```bash
cd ~/supreme-octo-succotash/gonogo
grep QUEUE12_DONE logs/queue12.log
tmux new-session -d -s attention_e4 'bash run_queue13.sh >>logs/queue13.log 2>&1'
tail -f logs/queue13.log
```

队列 13 的成功标记为 `QUEUE13_DONE`。步骤失败时查看同目录的 `logs/e3v2_*.log` 或 `logs/e4_*.log`，修复后再按脚本的续跑逻辑重启。

## 同步新增结果

在 GPU 的 Git 工作副本中查看变化、提交并推送。原始结果较大，推送可能需要数分钟。

```bash
cd ~/supreme-octo-succotash
git status --short
git add gonogo/out/e3_readout gonogo/out/e3v2_readout gonogo/out/e4_mask_position gonogo/logs
git commit -m 'Update experiment results'
git pull --rebase origin main
git push origin main
```

本地随后运行 `git pull --ff-only origin main`。同事在推送前检查 `git status`，只提交自己确认过的结果和代码；数据写入期间不要提交尚未完成的 JSONL。
