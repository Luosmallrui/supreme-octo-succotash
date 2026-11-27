"""Analyze the crossed physical-position and printed-label experiment (E2)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def cluster_mean(pids, values, n_boot, rng):
    """Estimate a row-weighted mean, resampling whole problems with replacement."""
    values = np.asarray(values, dtype=float)
    estimate = values.mean(axis=0)
    _, groups = np.unique(pids, return_inverse=True)
    counts = np.bincount(groups)
    if len(counts) < 2:
        return estimate, np.full(estimate.shape, np.nan), np.full(estimate.shape, np.nan)
    totals = np.zeros((len(counts), len(estimate)))
    np.add.at(totals, groups, values)
    samples = []
    for start in range(0, n_boot, 128):
        draw = rng.integers(len(counts), size=(min(128, n_boot - start), len(counts)))
        samples.append(totals[draw].sum(axis=1) / counts[draw].sum(axis=1)[:, None])
    lo, hi = np.percentile(np.concatenate(samples), [2.5, 97.5], axis=0)
    return estimate, lo, hi


def interval(lo, hi, signed=False):
    if np.isnan(lo):
        return "仅一道题"
    fmt = "+.1f" if signed else ".1f"
    return f"[{100 * lo:{fmt}}, {100 * hi:{fmt}}]"


def selection_report(rows, stratum, n_boot, rng):
    blocks = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["pid"], r["set_id"], r["probe_slot"], r["a"], r["b"])
        blocks[key][r["a_first"], r["a_label1"]].append(r)
    pids, values = [], []
    missing = duplicate = forced = 0
    for key, cells in blocks.items():
        if any(len(cell) != 1 for cell in cells.values()):
            duplicate += 1
            continue
        if len(cells) != 4:
            missing += 1
            continue
        # Axes are physical position, printed label, then outcome.
        outcomes = np.empty((2, 2, 4))
        for (first, label1), (r,) in cells.items():
            choice = r["choice_candidate"]
            forced += r["forced"]
            outcomes[int(first), int(label1)] = [
                choice == r["a"], r["correct"],
                choice is not None and choice not in (r["a"], r["b"]), choice is None,
            ]
        margins = []
        for axis in (0, 1):
            off, on = outcomes.mean(axis=1 - axis)
            margins.extend((on, off, on - off))
        pids.append(key[0])
        values.append(np.array(margins).ravel())
    lines = [f"完整四格 {len(pids)}/{len(blocks)} 个区组，覆盖 {len(set(pids))} 道题。"
             f"排除含重复格子 {duplicate} 组，其余缺格 {missing} 组。"]
    if not pids:
        return lines
    est, lo, hi = (x.reshape(2, 3, 4) for x in cluster_mean(pids, values, n_boot, rng))
    lines += ["", "| 对比（A 条件减 B 条件） | 指标 | A 条件 % | B 条件 % | 差值（百分点） | 95% CI |",
              "|---|---|---:|---:|---:|---|"]
    metrics = [(0, "选择 A"), (1, "选中正确答案")]
    if stratum == "both_correct_pair":
        metrics.append((2, "选中 A/B 之外的候选"))
    for factor, name in enumerate(["A 在首位 vs B 在首位", "A 编号为 1 vs B 编号为 1"]):
        for metric, label in metrics:
            on, off, diff = 100 * est[factor, :, metric]
            ci = interval(lo[factor, 2, metric], hi[factor, 2, metric], signed=True)
            lines.append(f"| {name} | {label} | {on:.1f} | {off:.1f} | {diff:+.1f} | {ci} |")
    lines.append(f"\n完整区组中未能映射到候选的选择占 {100 * est[0, :2, 3].mean():.1f}%。"
                 f"强制收尾 {forced}/{4 * len(pids)} 次。")
    return lines


def lookup_report(rows, n_boot, rng):
    if not rows:
        return ["无 lookup 记录。"]
    pids = [r["pid"] for r in rows]
    passed = sum(r["correct"] for r in rows)
    est, lo, hi = cluster_mean(pids, [[r["correct"]] for r in rows], n_boot, rng)
    ci = f"{interval(lo[0], hi[0])}%" if np.isfinite(lo[0]) else "仅一道题，无法估计"
    return [f"{len(set(pids))} 道题，{passed}/{len(rows)} 次按指令选中目标，"
            f"成功率 {100 * est[0]:.1f}%（95% CI {ci}）。"
            f"强制收尾 {sum(r['forced'] for r in rows)}/{len(rows)} 次。"]


def final_rows(path):
    """One row per request: API runs append a row per attempt, so keep the last successful one.

    Rows with an "error" field (hosted-judge API failures) are left out; a request with no successful row is
    therefore missing, which drops its block as incomplete. Local runs have one row per request and no errors.
    """
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            rows[r.get("request_id", f"line {n}")] = (n, r)
    return sorted(rows.values(), key=lambda x: x[0])


def report(path, n_boot, seed):
    groups = defaultdict(lambda: defaultdict(list))
    for n, r in final_rows(path):
        if r["task"] not in ("select", "lookup"):
            raise ValueError(f"Unknown task on line {n}: {r['task']}")
        if r["task"] == "select" and any(type(r[k]) is not bool for k in ("a_first", "a_label1")):
            raise ValueError(f"Expected two Boolean treatment indicators on line {n}")
        groups[r["model"], r["mode"], r["N"], r["stratum"]][r["task"]].append(r)
    if not groups:
        raise ValueError(f"No results in {path}")
    lines = ["# E2 物理位置与显示编号", "",
             "每个区组固定原题、候选集、A/B 候选和比较位置。两个因素各取两种状态。"
             "对另一因素等权平均后计算配对差值，完整区组等权。"
             "未选出有效候选的响应按未选中 A、回答错误计，强制收尾的响应按最终选择计。"
             f"95% 区间按原题整群重采样 {n_boot} 次。"]
    rng = np.random.default_rng(seed)
    for (model, mode, count, stratum), tasks in sorted(groups.items()):
        lines += ["", f"## {model} · {mode} · N={count} · {stratum}", "", "### 候选选择", ""]
        if stratum == "one_correct_pair":
            lines += ["A 正确，B 错误。", ""]
        elif stratum == "both_correct_pair":
            lines += ["A 和 B 均正确。选中其他候选按候选身份统计。", ""]
        lines.extend(selection_report(tasks["select"], stratum, n_boot, rng))
        lines += ["", "### Lookup 指令执行", ""]
        lines.extend(lookup_report(tasks["lookup"], n_boot, rng))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="E2 result JSONL")
    parser.add_argument("--output", type=Path, help="Markdown report (default: stdout)")
    parser.add_argument("--B", type=int, default=5000, help="number of bootstrap draws")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.B < 1:
        parser.error("--B must be positive")
    result = report(args.input, args.B, args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")
