# 多候选选择中的位置偏置（Attention Basin 续作 · go/no-go 实验）

研究问题：LLM 在一个提示里读入 N 个采样解并选出最佳解时，选择在多大程度上由候选的**位置**决定、这一偏置由什么承载（物理位置、打印编号、注意力分配），以及常见纠偏方法是否有效。模型为 Qwen3-4B / Qwen3-8B，题目为 MATH500 与 AIME 2024/2025、HMMT 2025。选题背景见 `Attention Basin 续作选题：多候选聚合场景.md`。

## 目录

| 路径 | 内容 |
|---|---|
| `gonogo/data/problems.json` | 590 道题及标准答案 |
| `gonogo/out/` | 每次运行的**原始输出**（候选解、选择结果、注意力、各实验的清单与报告），是原始数据的唯一存放位置 |
| `gonogo/results/` | 由分析脚本从 `out/` 生成的汇总表、`summary.pkl` 与图 |
| `gonogo/logs/` | GPU 服务器上的运行日志 |
| `gonogo/pylib/` | 服务器上使用的 math-verify 及依赖；队列脚本通过 `PYTHONPATH=pylib` 引用 |
| `gonogo/run_queue*.sh` | 历次 GPU 队列，记录每批数据的产生方式 |

## 实验与脚本

| 实验 | 脚本 | 主要输出 |
|---|---|---|
| 候选生成与判分 | `gen_candidates.py` → `grade_candidates.py` | `out/graded_{model}.jsonl` |
| 位置扫描（唯一正确解依次移过每个位置；模式 A/B/C 与思考模式） | `build_select.py` → `run_select.py` | `out/specs*.jsonl`、`out/sel*.jsonl` |
| 注意力（指令行 / 推理 token 到各候选块） | `attn_capture.py`、`attn_think.py` | `out/attn{A,C,T}_*.pkl` |
| 扫描与注意力分析 | `analyze.py`、`analyze_think.py`、`plot*.py` | `results/report_tables.md`、`results/report_think_*.md`、`results/fig*` |
| E1 自然候选集：槽位先验、校准、多顺序投票 | `e1_sets.py` → `e1_run.py`（托管模型：`e1_api.py`）→ `e1_analyze.py` | `out/e1_*`、`results/report_e1_*.md` |
| E2 物理位置 × 打印编号 | `e2_design.py` → `e2_run.py` → `e2_analyze.py`（`run_queue10.sh`） | `out/e2_position_label/` |
| E1 / E2 托管模型评审（OpenAI 兼容接口；direct 取 top-5 编号 logprobs，think 开推理） | `api_select.py --exp e1\|e2`（密钥读 `$OPENAI_API_KEY` 或 `--key-file`，不进仓库） → `e1_analyze.py` / `e2_analyze.py`；跨评审对比 `e2_compare.py` | `out/e1_api-*_qwen3-8b.jsonl`、`out/e2_position_label/api/` |
| E3 读出形式（编号 / 引文 / 答案），协议 v2 | `e3_design.py` → `e3_run.py` → `e3_analyze.py`（`run_queue12.sh`） | `out/e3v2_readout/` |
| E4 注意力隔离 × 共享位置编号（2×2 因果干预） | `e4_run.py` → `e4_analyze.py`（`run_queue13.sh`） | `out/e4_mask_position/` |

E3 第一版（`out/e3_readout/`）因输出格式和长度预算问题丢失了大部分响应，协议 v2 的改动写在 `e3_run.py` 文件头；第一版的事后复核见 `results/e3_readout/report_e3_three_completed_rescore.md`。

E4 每个模型先运行 `--selftest`（显式因果掩码与原生前向一致；“隔离 + 共享”条件对候选顺序严格不变），自检失败时队列停止。

## 运行

在 GPU 服务器上使用 `~/venvs/vllm/bin/python3` 并设置 `PYTHONPATH=$PWD/pylib`；队列脚本会等待 GPU 0 空闲。分析脚本默认从 `out/` 读取原始数据（`--data`），把报告写入 `results/`（`--dir`）。

## 仓库清理（2026-11-26）

以下内容已从当前版本中移除，仍保留在提交 `7e4b3f6` 的历史中：

- `gonogo.zip`（整个文件夹的压缩包）、全部 `__pycache__/`、`*.lock`、`.e3-submit.*/` 暂存副本、`out/e3_dispatch/` 一次性提交脚本；
- `results/` 中与 `out/` 逐字节相同的原始数据和报告副本（约 245 MB）；
- `out/cands_*.jsonl`（`graded_*.jsonl` 含相同行并附判分）、`out/e1_think_*.jsonl.gen.jsonl`（生成检查点，全部生成已在 `e1_think_*.jsonl` 中）；
- 冒烟测试输出与日志、模型下载日志。

由于历史中仍含这些大文件，完整克隆仍约 1 GB。只需要代码时可用：
`git clone --filter=blob:limit=1m <repo>`。
