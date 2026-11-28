"""E7 control: is the label -> answer lookup reading labels, or naming a likely answer?

A reply can match the answer of the candidate printed "Solution L" without reading labels, if that answer is the
set's majority answer or the correct one (the model solves the problem itself). This keeps only queries whose target
candidate holds a wrong minority answer (neither a most frequent answer of the set nor a correct one) and reports,
per judge, how often the reply matches the target, the candidate in slot L, the slot-1 candidate, the majority answer
and a correct answer. Answers are compared after norm_key, as in e7_label_answer.py. Intervals: 95% bootstrap over
problems, 5000 draws.

  python e7_control.py --data results   (E1 sets; writes out/e7_label_answer/report_e7_control.md)
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from common import norm_key

HERE = Path(__file__).resolve().parent
RUNS = (("qwen3-8b", "out/e7_label_answer/e7_qwen3-8b_on_qwen3-8b.jsonl"),
        ("qwen3-4b", "out/e7_label_answer/e7_qwen3-4b_on_qwen3-4b.jsonl"))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def boot(pids, vals, n_boot=5000, seed=0):
    ids = sorted(set(pids))
    ix = {p: i for i, p in enumerate(ids)}
    g = np.array([ix[p] for p in pids])
    v = np.asarray(vals, float)
    tot, cnt = np.bincount(g, v, len(ids)), np.bincount(g, None, len(ids))
    d = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))
    lo, hi = np.percentile(tot[d].sum(1) / cnt[d].sum(1), [2.5, 97.5])
    return f"{100 * v.mean():.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "results", help="folder with e1_sets_<model>.jsonl")
    ap.add_argument("--output", type=Path, default=HERE / "out/e7_label_answer/report_e7_control.md")
    args = ap.parse_args()
    lines = ["# E7 对照：目标候选持有错误的少数答案时", "",
             "由 `e7_control.py` 生成。只保留目标候选（印着 Solution L）的答案既不是该集合出现最多的答案、也不是正确答案的查询，"
             "此时回复与目标一致只能来自读取编号，不能来自自己解题或报多数答案。答案经 `norm_key` 归一化后比较。区间按原题整群重采样 5000 次。", "",
             "| 评审 | 查询 | 原题 | 与目标一致 (%) | 随机候选 (%) | 第 L 个位置的候选 (%) | 第 1 个位置的候选 (%) | 等于多数答案 (%) | 等于正确答案 (%) |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for model, path in RUNS:
        sets = {s["set_id"]: s for s in load(args.data / f"e1_sets_{model}.jsonl")}
        sub = []
        for r in load(HERE / path):
            s = sets[r["set_id"]]
            cnt = collections.Counter(s["key"].values())
            top = max(cnt.values())
            maj = {norm_key(a) for a, c in cnt.items() if c == top}
            cor = {norm_key(s["key"][k]) for k in s["key"] if s["cor"][k]}
            target = norm_key(s["key"][str(r["k_label"])])
            if target in maj or target in cor:
                continue
            a = norm_key(r["answer"]) if r.get("answer") is not None else None
            sub.append((r, a in maj, a in cor))
        rs = [x[0] for x in sub]
        lines.append(f"| {model} | {len(rs)} | {len({r['pid'] for r in rs})} | {boot([r['pid'] for r in rs], [r['match_label'] for r in rs])} | "
                     f"{100 * np.mean([r['chance_label'] for r in rs]):.1f} | {100 * np.mean([r['match_ordinal'] for r in rs]):.1f} | "
                     f"{100 * np.mean([r['match_first'] for r in rs]):.1f} | {100 * np.mean([x[1] for x in sub]):.1f} | "
                     f"{100 * np.mean([x[2] for x in sub]):.1f} |")
    lines += ["", "第 L 个位置与第 1 个位置的候选可能答案相同，多数答案也可能与它们相同，所以各列之和可超过 100%。", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
