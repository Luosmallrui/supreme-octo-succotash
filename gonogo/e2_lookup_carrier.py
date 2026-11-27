"""Does the ability to bind positions to printed labels track which carrier dominates? (E2, all judges, no reasoning)

Reads compare_e2_judges.md (e2_compare.py) and, for every judge in direct mode on Qwen3-8B's layouts, takes
  lookup   correct answers to "which printed label is at position k?" (of 32),
  slot     the effect of listing A first on P(choose A), mean of the two strata (A correct/B wrong, both correct),
  label    the effect of printing A as "Solution 1", same average,
  accuracy over all selection prompts.
It reports Spearman rank correlations across judges (exact two-sided p from all permutations is too many for n = 10;
the p-values are scipy's t approximation) and the sign of slot - label per judge.

  python e2_lookup_carrier.py   (writes out/e2_position_label/api/report_lookup_carrier.md)
"""
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
SRC = HERE / "out/e2_position_label/api/compare_e2_judges.md"
OUT = HERE / "out/e2_position_label/api/report_lookup_carrier.md"
NUM = r"([+-]?\d+\.\d)"


def parse(text):
    overview, effects, stratum = {}, {}, None
    for line in text.splitlines():
        if line.startswith("## one_correct_pair"):
            stratum = "one"
        elif line.startswith("## both_correct_pair"):
            stratum = "both"
        elif line.startswith("与 qwen3-8b 的配对差值"):
            stratum = None                      # skip the paired-difference tables
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("| ") else []
        if len(cells) == 9 and cells[1] == "direct" and "次" in cells[8]:
            acc = float(re.match(NUM, cells[6]).group(1))
            look = round(float(re.match(r"([\d.]+)", cells[8]).group(1)) * 32 / 100)
            overview[cells[0]] = (acc, look)
        elif len(cells) == 7 and cells[1] == "direct" and stratum:
            pos, lab = (float(re.match(NUM, cells[k]).group(1)) for k in (3, 5))
            effects.setdefault(cells[0], {})[stratum] = (pos, lab)
    return overview, effects


def main():
    overview, effects = parse(SRC.read_text(encoding="utf-8"))
    rows = []
    for judge, (acc, look) in overview.items():
        e = effects[judge]
        slot = (e["one"][0] + e["both"][0]) / 2
        label = (e["one"][1] + e["both"][1]) / 2
        rows.append((judge, look, slot, label, acc))
    rows.sort(key=lambda r: r[1])
    look, slot, label, acc = (np.array([r[k] for r in rows], float) for k in (1, 2, 3, 4))
    lines = ["# 绑定能力与偏差载体（E2，direct，Qwen3-8B 的布局）", "",
             "由 `e2_lookup_carrier.py` 从 `compare_e2_judges.md` 生成。lookup 为 32 次“第 k 个位置印的是哪个编号”中答对的次数；"
             "slot、label 为两种区组（A 对 B 错、两个都对）的效应平均值（百分点）。Spearman 相关跨评审计算，p 值为 scipy 的 t 近似，n 较小，只作描述。", "",
             "| 评审 | lookup | slot | label | slot − label | 准确率 % |", "|---|---:|---:|---:|---:|---:|"]
    for judge, lk, s, lb, a in rows:
        lines.append(f"| {judge} | {lk} | {s:+.1f} | {lb:+.1f} | {s - lb:+.1f} | {a:.1f} |")
    lines += ["", "| 相关 | Spearman ρ | p | n |", "|---|---:|---:|---:|"]
    for name, x, y in [("lookup 与 slot", look, slot), ("lookup 与 label", look, label),
                       ("lookup 与 slot − label", look, slot - label), ("lookup 与准确率", look, acc),
                       ("准确率与 slot", acc, slot), ("准确率与 slot − label", acc, slot - label)]:
        r, p = spearmanr(x, y)
        lines.append(f"| {name} | {r:+.2f} | {p:.3f} | {len(x)} |")
    lines += ["", f"slot > label 的评审：{sum(slot > label)} 个，lookup 为 {int(look[slot > label].min())}–{int(look[slot > label].max())}；"
              f"slot ≤ label 的评审：{sum(slot <= label)} 个，lookup 为 {int(look[slot <= label].min())}–{int(look[slot <= label].max())}。"]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
