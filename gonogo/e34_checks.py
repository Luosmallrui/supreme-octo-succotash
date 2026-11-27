"""Checks on the E3 v2 and E4 results that e3_analyze.py and e4_analyze.py do not compute.

E4  Per condition: accuracy against picking a uniformly random candidate, how often A and B are chosen (with A
    correct and B wrong, their difference measures whether the choice still follows content), the most frequent
    printed label, and the distribution of the chosen slot.
E3  Thinking mode: accuracy of the answer readout (numbered headers) with A (correct) or B (wrong) in the first
    slot, each against the same model solving the problem alone. For every readout: the share of traces that reach
    the token budget, and how repetitive their last 6,000 characters are (zlib-compressed / raw size; finished
    traces are about 0.3, repetition loops far lower), with the first-slot effect split by forced ending.
    Both modes: what the quotation replies that match no candidate contain.

Intervals: 95% percentile bootstrap over problems (cluster_mean from e2_analyze), 5000 draws, fixed seeds.

  python e34_checks.py   (reads out/e4_mask_position/e4_*.jsonl and out/e3v2_readout/e3_*.jsonl; writes
                          out/e4_mask_position/report_e4_checks.md and out/e3v2_readout/report_e3_checks.md)
"""
import argparse
import json
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from e2_analyze import cluster_mean

HERE = Path(__file__).resolve().parent
MODELS = ("qwen3-8b", "qwen3-4b")
CONDITIONS = [("causal", "seq", "标准"), ("isolated", "seq", "隔离候选块"), ("causal", "shared", "共享位置编号"),
              ("isolated", "shared", "隔离 + 共享"), ("bidir", "seq", "双向跨候选注意力"),
              ("bidir", "shared", "双向 + 共享")]   # bidir rows come from e4_bidir_<model>.jsonl when present
STRATA = ("one_correct_pair", "both_correct_pair")


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ci(est, lo, hi, signed=True):
    f = "+.1f" if signed else ".1f"
    return f"{100 * est:{f}} [{100 * lo:{f}}, {100 * hi:{f}}]"


def e4_report(B):
    lines = ["# E4 补充检查：干预后选择是否仍跟随内容", "",
             "由 `e34_checks.py` 生成。one_correct 中 A 正确、B 错误，“选 A − 选 B”衡量选择是否仍跟随内容；"
             "随机正确率为在 8 个候选中均匀随机选一个的期望正确率。区间按原题整群重采样 5000 次。", ""]
    sources = []
    for model in MODELS:
        files = [HERE / f"out/e4_mask_position/e4_{model}.jsonl", HERE / f"out/e4_mask_position/e4_bidir_{model}.jsonl"]
        sources.append((model, model, [f for f in files if f.exists()]))
    for f in sorted((HERE / "out/e4_mask_position/judges").glob("e4_*_on_*.jsonl")):   # other judges (queue 16)
        judge, cand = f.stem[3:].split("_on_")
        sources.append((f"{judge} 评 {cand} 的布局", judge, [f]))
    for title, model, files in sources:
        rows = [r for f in files for r in load(f)]
        for stratum in STRATA:
            lines += [f"## {title} · {stratum}", "",
                      "| 条件 | 行数 | 正确率 (%) | 随机正确率 (%) | 正确率 − 随机 (pp) | 选 A (%) | 选 B (%) | 选 A − 选 B (pp) "
                      "| 最常选的打印编号 | 其占比 (%) | 各槽位被选比例 (%)，槽位 1→8 |",
                      "|---|---:|---|---:|---|---:|---:|---|---:|---:|---|"]
            for mask, pos, name in CONDITIONS:
                rs = [r for r in rows if (r["mask"], r["pos"]) == (mask, pos) and r["stratum"] == stratum]
                if not rs:
                    continue
                pids = [r["pid"] for r in rs]
                chance = [np.mean([r["cor"][str(k)] for k in r["order"]]) for r in rs]
                vals = [[r["correct"], r["correct"] - c, r["choice_candidate"] == r["a"],
                         r["choice_candidate"] == r["b"],
                         (r["choice_candidate"] == r["a"]) - (r["choice_candidate"] == r["b"])]
                        for r, c in zip(rs, chance)]
                seed = zlib.crc32(f"e4|{model}|{stratum}|{mask}|{pos}".encode())
                est, lo, hi = cluster_mean(pids, vals, B, np.random.default_rng(seed))
                label, n_label = Counter(r["choice_label"] for r in rs).most_common(1)[0]
                slots = np.bincount([r["choice_slot"] for r in rs], minlength=9)[1:] / len(rs)
                lines.append(f"| {name} | {len(rs)} | {ci(est[0], lo[0], hi[0], False)} | {100 * np.mean(chance):.1f} | "
                             f"{ci(est[1], lo[1], hi[1])} | {100 * est[2]:.1f} | {100 * est[3]:.1f} | "
                             f"{ci(est[4], lo[4], hi[4])} | {label} | {100 * n_label / len(rs):.1f} | "
                             + " ".join(f"{100 * x:.0f}" for x in slots) + " |")
            lines.append("")
    return "\n".join(lines) + "\n"


def tail_ratio(text):
    raw = text[-6000:].encode()
    return len(zlib.compress(raw)) / max(1, len(raw))


def e3_report(B):
    lines = ["# E3 v2 补充检查：独立解题对照、推理循环与短句读出", "",
             "由 `e34_checks.py` 生成。E3 的 think 模式沿用 v1 的贪心解码（温度 0）和 12,288 token 预算，以便复用 v1 生成。"
             "区间按原题整群重采样 5000 次。", ""]
    variants = [(m, s) for m in MODELS for s in ("", "_sampled")
                if (HERE / f"out/e3v2_readout/e3_think{s}_{m}.jsonl").exists()]
    for model, suffix in variants:   # "_sampled": queue 16 rerun with Qwen3's recommended sampling (e3_run.py --sample)
        rows = load(HERE / f"out/e3v2_readout/e3_think{suffix}_{model}.jsonl")
        solo = {r["pid"]: r["correct"] for r in rows if r["readout"] == "solo_answer"}
        title = f"{model} · think" + (" · 采样重跑（温度 0.6）" if suffix else " · 贪心")
        lines += [f"## {title}", "", "### 带候选解答 vs 独立解题（one_correct，answer 读出，numbered 标题）", "",
                  "| 首位候选 | 题数 | 带候选正确率 (%) | 独立解题正确率 (%) | 差值 (pp) |", "|---|---:|---:|---:|---|"]
        for first, name in ((True, "A（正确）"), (False, "B（错误）")):
            rs = [r for r in rows if r["readout"] == "answer" and r["header"] == "numbered"
                  and r["stratum"] == "one_correct_pair" and r["a_first"] == first and r["pid"] in solo]
            pids = [r["pid"] for r in rs]
            vals = [[r["correct"], solo[r["pid"]], r["correct"] - solo[r["pid"]]] for r in rs]
            seed = zlib.crc32(f"e3solo|{model}{suffix}|{first}".encode())
            est, lo, hi = cluster_mean(pids, vals, B, np.random.default_rng(seed))
            lines.append(f"| {name} | {len(set(pids))} | {100 * est[0]:.1f} | {100 * est[1]:.1f} | "
                         f"{ci(est[2], lo[2], hi[2])} |")
        lines += ["", "### 触及预算与重复循环", "",
                  "| 读出 | 标题 | 调用 | 触及预算 | 占比 (%) | 末尾压缩比中位数（触及预算） | 末尾压缩比中位数（正常结束且长于 6,000 字符） |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        groups = defaultdict(list)
        for r in rows:
            groups[r["readout"], r["header"]].append(r)
        for (readout, header), rs in sorted(groups.items()):
            capped = [r for r in rs if r["finish"] == "length"]
            done = [r for r in rs if r["finish"] != "length" and len(r["gen"]) > 6000]
            med = lambda xs: f"{np.median([tail_ratio(r['gen']) for r in xs]):.2f}" if xs else "—"
            lines.append(f"| {readout} | {header} | {len(rs)} | {len(capped)} | {100 * len(capped) / len(rs):.1f} | "
                         f"{med(capped)} | {med(done)} |")
        lines += ["", "### label 读出按是否强制收尾拆分（行级，未按区组配对）", "",
                  "| 分层 | 强制收尾 | 行数 | P(选 A｜A 首位) (%) | P(选 A｜B 首位) (%) | 差值 (pp) | 正确率 (%) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        label = [r for r in rows if r["readout"] == "label" and r["header"] == "numbered"]
        for stratum in STRATA:
            for forced in (False, True):
                rs = [r for r in label if r["stratum"] == stratum and bool(r["forced"]) == forced]
                a1 = np.mean([r["choice_candidate"] == r["a"] for r in rs if r["a_first"]])
                b1 = np.mean([r["choice_candidate"] == r["a"] for r in rs if not r["a_first"]])
                lines.append(f"| {stratum} | {'是' if forced else '否'} | {len(rs)} | {100 * a1:.1f} | {100 * b1:.1f} | "
                             f"{100 * (a1 - b1):+.1f} | {100 * np.mean([r['correct'] for r in rs]):.1f} |")
        lines.append("")
    lines += ["## 短句读出（quote）未匹配的回复", "",
              "未匹配指既非逐字引用、也无法按 20 字符片段归属到某个候选（宽松规则）。“以编号开头”指回复以 "
              "“Solution k” 开头（可带 markdown），即给出编号而不是引用原文；“短答案”指不超过 25 个字符的回复。", "",
              "| 模型 | 模式 | 调用 | 未匹配 | 其中以编号开头 | 其中短答案 | 其他 |", "|---|---|---:|---:|---:|---:|---:|"]
    starts_label = re.compile(r"\W*solution\s*\d", re.I)
    for model in MODELS:
        for mode in ("direct", "think"):
            rs = [r for r in load(HERE / f"out/e3v2_readout/e3_{mode}_{model}.jsonl") if r["readout"] == "quote"]
            bad = [r for r in rs if r.get("choice_candidate") is None]
            finals = [(r.get("final") or "").strip() for r in bad]
            lab = sum(bool(starts_label.match(f)) for f in finals)
            short = sum(not starts_label.match(f) and len(f) <= 25 for f in finals)
            lines.append(f"| {model} | {mode} | {len(rs)} | {len(bad)} | {lab} | {short} | {len(bad) - lab - short} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--B", type=int, default=5000)
    args = ap.parse_args()
    for path, text in ((HERE / "out/e4_mask_position/report_e4_checks.md", e4_report(args.B)),
                       (HERE / "out/e3v2_readout/report_e3_checks.md", e3_report(args.B))):
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
