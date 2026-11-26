"""Compare E2 position and printed-label effects across judges (local Qwen3 runs and hosted API judges).

Each input is judge:mode:path. Effects use the same block definition as e2_analyze.py: within a complete
2x2 block, average over the other factor and take on-minus-off. Judge differences are paired on the blocks
both judges completed. 95% intervals resample whole problems.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from e2_analyze import cluster_mean

METRICS = ["position: choose A", "position: correct", "label: choose A", "label: correct"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def block_effects(rows):
    """{(stratum, pid, set_id, probe_slot, a, b): [4 effects]} over complete blocks of select rows."""
    blocks = defaultdict(dict)
    for r in rows:
        if r["task"] == "select" and "error" not in r:
            key = (r["stratum"], r["pid"], r["set_id"], r["probe_slot"], r["a"], r["b"])
            blocks[key][r["a_first"], r["a_label1"]] = r
    out = {}
    for key, cells in blocks.items():
        if len(cells) != 4:
            continue
        o = np.empty((2, 2, 2))
        for (first, label1), r in cells.items():
            o[int(first), int(label1)] = [r["choice_candidate"] == r["a"], r["correct"]]
        pos = o.mean(axis=1)
        lab = o.mean(axis=0)
        out[key] = np.concatenate([pos[1] - pos[0], lab[1] - lab[0]])
    return out


def fmt(est, lo, hi, scale=100, signed=True):
    f = "+.1f" if signed else ".1f"
    return f"{scale * est:{f}} [{scale * lo:{f}}, {scale * hi:{f}}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", help="judge:mode:path")
    ap.add_argument("--reference", default="qwen3-8b", help="judge that the others are paired against")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = {}
    for spec in args.inputs:
        judge, mode, path = spec.split(":", 2)
        runs[judge, mode] = load(path)
    modes = sorted({m for _, m in runs}, key=["direct", "think"].index)
    judges = list(dict.fromkeys(j for j, _ in runs))
    effects = {k: block_effects(v) for k, v in runs.items()}
    strata = ["one_correct_pair", "both_correct_pair"]
    lines = ["# E2 跨评审模型对比", "",
             "候选与布局相同（Qwen3-8B 的 E2 清单）。效应定义同 `e2_analyze.py`：完整四格区组内对另一因素等权平均，"
             "取“条件成立减不成立”，单位为百分点。括号内为按原题整群重采样的 95% 区间。"
             f"与 {args.reference} 的差值只用两者都完成的区组配对计算。", ""]
    lines += ["## 总览（select 任务）", "",
              "| 评审 | 模式 | 调用 | 报错 | 强制收尾 | 未给出编号 | 准确率 % | 选首位 % | lookup 成功 % |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in modes:
        for judge in judges:
            rows = runs.get((judge, mode))
            if rows is None:
                continue
            sel = [r for r in rows if r["task"] == "select" and "error" not in r]
            look = [r for r in rows if r["task"] == "lookup" and "error" not in r]
            pids = [r["pid"] for r in sel]
            acc = cluster_mean(pids, [[r["correct"]] for r in sel], args.B, rng)
            first = cluster_mean(pids, [[r["choice_slot"] == 1] for r in sel], args.B, rng)
            lk = (f"{100 * np.mean([r['correct'] for r in look]):.1f}（{len(look)} 次）" if look else "—")
            lines.append(f"| {judge} | {mode} | {len(rows)} | {sum('error' in r for r in rows)} | "
                         f"{sum(bool(r.get('forced')) for r in rows)} | "
                         f"{sum(r.get('choice_label') is None for r in rows if 'error' not in r)} | "
                         f"{fmt(acc[0][0], acc[1][0], acc[2][0], signed=False)} | "
                         f"{fmt(first[0][0], first[1][0], first[2][0], signed=False)} | {lk} |")
    lines += ["", "E2 的首位总是放 A 或 B，而 A、B 是挑出来的候选，所以“选首位”不是干净的位置先验，只作参考"
              "（纯随机选择时为 12.5%）。位置效应看下面的配对差值。think 行若调用不满 976 次，只统计已完成的完整区组。", ""]
    for stratum in strata:
        lines += [f"## {stratum}", "",
                  "A 正确、B 错误。" if stratum == "one_correct_pair" else "A、B 均正确。", "",
                  "| 评审 | 模式 | 区组 | " + " | ".join(METRICS) + " |",
                  "|---|---|---:|" + "---:|" * len(METRICS)]
        for mode in modes:
            for judge in judges:
                eff = {k: v for k, v in effects.get((judge, mode), {}).items() if k[0] == stratum}
                if not eff:
                    continue
                pids = [k[1] for k in eff]
                est, lo, hi = cluster_mean(pids, list(eff.values()), args.B, rng)
                lines.append(f"| {judge} | {mode} | {len(eff)} | "
                             + " | ".join(fmt(est[i], lo[i], hi[i]) for i in range(len(METRICS))) + " |")
        lines += ["", f"与 {args.reference} 的配对差值（该评审减 {args.reference}）：", "",
                  "| 评审 | 模式 | 配对区组 | " + " | ".join(METRICS) + " |",
                  "|---|---|---:|" + "---:|" * len(METRICS)]
        for mode in modes:
            ref = {k: v for k, v in effects.get((args.reference, mode), {}).items() if k[0] == stratum}
            for judge in judges:
                if judge == args.reference:
                    continue
                eff = effects.get((judge, mode), {})
                keys = [k for k in ref if k in eff]
                if not keys:
                    continue
                est, lo, hi = cluster_mean([k[1] for k in keys], [eff[k] - ref[k] for k in keys], args.B, rng)
                lines.append(f"| {judge} | {mode} | {len(keys)} | "
                             + " | ".join(fmt(est[i], lo[i], hi[i]) for i in range(len(METRICS))) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
