"""Analyze E4: position and label effects under the 2x2 mask x position-id intervention."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from e2_analyze import cluster_mean

CONDITIONS = [("causal", "seq"), ("isolated", "seq"), ("causal", "shared"), ("isolated", "shared"),
              ("bidir", "seq"), ("bidir", "shared")]
NAMES = {("causal", "seq"): "标准（因果 + 顺序位置）", ("isolated", "seq"): "隔离候选块",
         ("causal", "shared"): "共享位置编号", ("isolated", "shared"): "隔离 + 共享（顺序无关）",
         ("bidir", "seq"): "双向跨候选注意力", ("bidir", "shared"): "双向 + 共享（顺序无关）"}
# Per-block metrics; effects are "A condition minus B condition", averaging over the other factor.
METRICS = ["首位效应 · 选择 A (pp)", "编号效应 · 选择 A (pp)", "首位效应 · P(A) (pp)", "编号效应 · P(A) (pp)",
           "首位效应 · 正确率 (pp)", "四格平均正确率 (%)", "选第 1 位的比例 (%)", "数字概率质量 (%)"]


def load(paths):
    blocks = defaultdict(lambda: defaultdict(dict))
    meta = set()
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                meta.add((r["model"] if "judge" not in r else f"{r['judge']} 评 {r['model']} 的布局", r["align"]))
                key = (r["pid"], r["set_id"], r["probe_slot"], r["a"], r["b"])
                cell = (r["a_first"], r["a_label1"])
                cond = (r["mask"], r["pos"])
                if cell in blocks[cond, r["stratum"]][key]:
                    raise ValueError(f"Duplicate {r['request_id']} {cond}")
                blocks[cond, r["stratum"]][key][cell] = r
    if len(meta) != 1:
        raise ValueError(f"Expected one model and alignment, found {meta}")
    return meta.pop(), blocks


def block_metrics(cells):
    """Vector of METRICS for one complete 2x2 block (None if incomplete)."""
    if len(cells) != 4:
        return None
    out = np.zeros((2, 2, 5))
    for (first, lab1), r in cells.items():
        a_label = r["labels"][r["order"].index(r["a"])]
        out[int(first), int(lab1)] = [r["choice_candidate"] == r["a"], r["p_label"][a_label - 1], r["correct"],
                                      r["choice_slot"] == 1, r["p_mass"]]
    pos = out[1].mean(0) - out[0].mean(0)       # A first minus B first
    lab = out[:, 1].mean(0) - out[:, 0].mean(0)  # A labelled 1 minus B labelled 1
    mean = out.mean((0, 1))
    return np.array([pos[0], lab[0], pos[1], lab[1], pos[2], mean[2], mean[3], mean[4]])


def summarize(blocks, n_boot, rng):
    vals = {k: m for k, cells in blocks.items() if (m := block_metrics(cells)) is not None}
    if not vals:
        return None, {}
    pids = [k[0] for k in vals]
    est, lo, hi = cluster_mean(pids, list(vals.values()), n_boot, rng)
    return (est, lo, hi, len(vals), len(set(pids))), vals


def fmt(est, lo, hi):
    ci = f"[{100 * lo:+.1f}, {100 * hi:+.1f}]" if np.isfinite(lo) else "—"
    return f"{100 * est:+.1f} {ci}"


def report(paths, n_boot, seed):
    (model, align), blocks = load(paths)
    present = [c for c in CONDITIONS if any(cond == c for cond, _ in blocks)]
    rng = np.random.default_rng(seed)
    lines = ["# E4 注意力隔离 × 位置编号（2×2 干预）", "",
             f"模型 {model}，direct 读出（预填 \"The best solution is Solution \"，在 8 个打印编号上的下一词分布）。"
             f"布局沿用 E2。共享位置编号的对齐方式：{align}。效应为 A 条件减 B 条件，对另一因素等权平均；"
             f"P(A) 为分配给 A 编号的概率。区间按原题整群重采样 {n_boot} 次。"
             "“隔离 + 共享”与“双向 + 共享”条件下首位效应按构造应为 0（检验项）。"]
    strata = sorted({s for _, s in blocks})
    for stratum in strata:
        lines += ["", f"## {stratum}", "", "| 条件 | 区组/题数 | " + " | ".join(METRICS) + " |",
                  "|---|---|" + "---|" * len(METRICS)]
        per = {}
        for cond in present:
            summary, vals = summarize(blocks.get((cond, stratum), {}), n_boot, rng)
            per[cond] = vals
            if summary is None:
                lines.append(f"| {NAMES[cond]} | 0 | " + " | ".join("—" for _ in METRICS) + " |")
                continue
            est, lo, hi, nb, npid = summary
            cells = [fmt(est[i], lo[i], hi[i]) if i < 5 else f"{100 * est[i]:.1f}" for i in range(len(METRICS))]
            lines.append(f"| {NAMES[cond]} | {nb}/{npid} | " + " | ".join(cells) + " |")
        base = per[CONDITIONS[0]]
        lines += ["", "### 相对标准条件的配对差（同一区组）", "",
                  "| 条件 − 标准 | 区组数 | 首位效应 · 选择 A (pp) | 首位效应 · P(A) (pp) | 四格平均正确率 (pp) |",
                  "|---|---|---|---|---|"]
        for cond in present[1:]:
            keys = [k for k in per[cond] if k in base]
            if not keys:
                continue
            diff = [per[cond][k][[0, 2, 5]] - base[k][[0, 2, 5]] for k in keys]
            est, lo, hi = cluster_mean([k[0] for k in keys], diff, n_boot, rng)
            lines.append(f"| {NAMES[cond]} | {len(keys)} | " +
                         " | ".join(fmt(est[i], lo[i], hi[i]) for i in range(3)) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, nargs="+", required=True, help="one or more E4 outputs of one model")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report(args.input, args.B, args.seed), encoding="utf-8")
