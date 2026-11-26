"""Paired analysis of E3 readouts, candidate headings, and question-only controls."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from e2_analyze import cluster_mean


READOUTS = ("label", "quote", "answer")
METRICS = ("选择 A", "正确率", "未匹配率")


def load(path):
    layouts = defaultdict(lambda: defaultdict(dict))
    controls = defaultdict(list)
    quality = defaultdict(Counter)
    ids = set()
    runs = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            runs.add((r["model"], r["mode"]))
            rid = r["request_id"]
            if rid in ids:
                raise ValueError(f"{path}: duplicate request {rid}")
            ids.add(rid)
            kind, header = r["readout"], r["header"]
            qc = quality[kind, header]
            qc["calls"] += 1
            qc["unparsed"] += not r["parsed"]
            qc["length"] += r["finish"] == "length"
            correct = bool(r["parsed"] and r["correct"])
            if kind in ("solo_answer", "lookup_quote"):
                controls[kind].append((r["pid"], correct))
                continue
            if kind not in READOUTS or header not in ("numbered", "plain", "none"):
                raise ValueError(f"{rid}: unknown readout or heading")
            a, b = r["a"], r["b"]
            first, label1 = r["a_first"], r["a_label1"]
            order, labels = r["order"], r["labels"]
            first_two = (order[0], order[r["probe_slot"] - 1])
            pair_label = (labels[order.index(a)], labels[order.index(b)])
            if (type(first) is not bool or type(label1) is not bool
                    or first_two != ((a, b) if first else (b, a)) or pair_label[0 if label1 else 1] != 1):
                raise ValueError(f"{rid}: position or label contradicts assigned treatment")
            # Removing the two manipulated positions leaves a constant background.
            # Only match contrasts across files/readouts when the full layout agrees.
            background = tuple((k, n) for k, n in zip(order, labels) if k not in (a, b))
            key = (r["pid"], r["set_id"], r["N"], r["probe_slot"], a, b, background, max(pair_label))
            cells = layouts[r["stratum"], kind, header][key]
            if (first, label1) in cells:
                raise ValueError(f"{rid}: duplicate treatment in the same block")
            chosen = r["choice_candidate"]
            matched = bool(r["parsed"] and chosen is not None)
            if kind == "answer":
                cells[first, label1] = (0.0, correct, not r["parsed"])
            else:
                cells[first, label1] = (matched and chosen == a, matched and correct, not matched)
    if len(runs) != 1:
        raise ValueError(f"{path}: expected results from one model and mode, found {runs}")
    return runs.pop(), layouts, controls, quality


def physical_means(blocks, label_states=(False, True)):
    """Each block yields [B first, A first] means, with equal weight per requested label."""
    out = {}
    for key, cells in blocks.items():
        if all((first, lab) in cells for first in (False, True) for lab in label_states):
            out[key] = np.array([[cells[first, lab] for lab in label_states]
                                 for first in (False, True)], dtype=float).mean(axis=1)
    return out


def percent(x):
    return f"{100*x:.1f}"


class Report:
    def __init__(self, draws):
        self.draws = draws
        self.rng = np.random.default_rng(0)
        self.lines = ["# E3 候选位置与输出形式", "",
                      "首位效应为 A 在首位时的指标减去 B 在首位时的指标。编号四格中两个编号条件等权，"
                      "每个完整区组权重相同。无法识别的输出计为错误且未选 A。"
                      f"区间为按原题配对重采样 {draws} 次的 95% 百分位区间。"]

    def section(self, heading):
        self.lines.extend(["", heading, ""])

    def comparison(self, label, pids, left, right):
        if not len(pids):
            self.lines.append(f"| {label} | 0 | 0 | — | — | — | — |")
            return
        left, right = np.asarray(left), np.asarray(right)
        mean, lo, hi = cluster_mean(pids, (left - right)[:, None], self.draws, self.rng)
        ci = f"[{100*lo[0]:+.1f}, {100*hi[0]:+.1f}]" if np.isfinite(lo[0]) else "仅一道题"
        self.lines.append(f"| {label} | {len(pids)} | {len(set(pids))} | {percent(left.mean())} | "
                          f"{percent(right.mean())} | {100*mean[0]:+.1f} | {ci} |")

    def table(self, left, right):
        self.lines.extend([f"| 对比 | 配对数 | 题数 | {left} | {right} | 差值 (pp) | 95% CI (pp) |",
                           "|---|---:|---:|---:|---:|---:|---|"])

    def physical(self, name, blocks, metric):
        self.comparison(name, [k[0] for k in blocks], [v[1, metric] for v in blocks.values()],
                        [v[0, metric] for v in blocks.values()])

    def paired(self, name, left, right, metric, effect=True):
        keys = [k for k in left if k in right]
        def values(data):
            if effect:
                return [data[k][1, metric] - data[k][0, metric] for k in keys]
            return [data[k][:, metric].mean() for k in keys]
        self.comparison(name, [k[0] for k in keys], values(left), values(right))


def report(primary, paired, n_boot):
    (model, mode), layouts, controls, quality = primary
    full = {key: physical_means(blocks) for key, blocks in layouts.items() if key[2] == "numbered"}
    rpt = Report(n_boot)
    rpt.section(f"## {model} · {mode} · 输出质量")
    rpt.lines.extend(["| 输出形式 | 标题 | 调用数 | 未解析或未匹配数 (%) | 触及长度上限数 |",
                      "|---|---|---:|---:|---:|"])
    for (kind, header), qc in sorted(quality.items()):
        rpt.lines.append(f"| {kind} | {header} | {qc['calls']} | {qc['unparsed']} "
                         f"({percent(qc['unparsed']/qc['calls'])}%) | {qc['length']} |")

    for stratum in sorted({key[0] for key in layouts}):
        rpt.section(f"## {stratum} · 编号四格")
        rpt.table("A 首位 (%)", "B 首位 (%)")
        for kind in READOUTS:
            key = (stratum, kind, "numbered")
            if key not in full:
                continue
            blocks = full[key]
            used = f"（完整 {len(blocks)}/{len(layouts[key])}）"
            for i in ((1,) if kind == "answer" else (0, 1, 2)):
                rpt.physical(f"{kind} · {METRICS[i]}{used}", blocks, i)
        quote, label = (full.get((stratum, kind, "numbered"), {}) for kind in ("quote", "label"))
        if quote and label:
            rpt.section("### 同布局的短句与编号读出")
            rpt.table("quote", "label")
            for i in (0, 1, 2):
                rpt.paired(f"{METRICS[i]}首位效应 (pp)", quote, label, i)
                rpt.paired(f"四格平均{METRICS[i]} (%)", quote, label, i, effect=False)

    rpt.section("## 短句查找与独立解题")
    for kind in ("lookup_quote", "solo_answer"):
        trials = controls.get(kind, [])
        if trials:
            pids, outcomes = zip(*trials)
            mean, lo, hi = cluster_mean(pids, np.asarray(outcomes)[:, None], n_boot, rpt.rng)
            ci = (f"[{percent(lo[0])}, {percent(hi[0])}]%" if np.isfinite(lo[0]) else "仅一道题")
            rpt.lines.append(f"- {kind}：{sum(outcomes)}/{len(outcomes)} 次正确（{percent(mean[0])}%），"
                             f"{len(set(pids))} 道题，95% CI {ci}。")
    solo = dict(controls.get("solo_answer", []))
    if len(solo) != len(controls.get("solo_answer", [])):
        raise ValueError("Expected exactly one solo_answer per problem")
    if solo:
        rpt.section("### 同题的带候选解答与独立解题")
        rpt.table("answer (%)", "solo (%)")
        for (stratum, kind, _), blocks in sorted(full.items()):
            if kind != "answer":
                continue
            by_pid = defaultdict(list)
            for key, v in blocks.items():
                if key[0] in solo:
                    by_pid[key[0]].append(v[:, 1].mean())
            pids = list(by_pid)
            rpt.comparison(stratum, pids, [np.mean(by_pid[p]) for p in pids], [solo[p] for p in pids])

    rpt.section("## 一正一误时的标题对照")
    rpt.lines.extend(["固定 A 的底层编号为 1，plain 与 none 隐去打印编号。每个区组配对比较两种物理顺序。", ""])
    headers = {(kind, header): physical_means(blocks, (True,)) for (st, kind, header), blocks in layouts.items()
               if st == "one_correct_pair"}
    heading_kinds = sorted({kind for kind, h in headers if h in ("plain", "none")})
    rpt.table("A 首位 (%)", "B 首位 (%)")
    for kind in heading_kinds:
        for h in ("numbered", "plain", "none"):
            rpt.physical(f"{kind} · {h} 正确率", headers.get((kind, h), {}), 1)
    if heading_kinds:
        rpt.section("### 相同布局下的正确率首位效应之差")
        rpt.table("该标题效应 (pp)", "numbered 效应 (pp)")
        for kind in heading_kinds:
            for h in ("plain", "none"):
                rpt.paired(f"{kind} · {h} − numbered", headers.get((kind, h), {}),
                           headers.get((kind, "numbered"), {}), 1)

    if paired:
        (other_model, other_mode), other_layouts, _, _ = paired
        if model != other_model or {mode, other_mode} != {"direct", "think"}:
            raise ValueError("Paired file must contain the other mode for the same model")
        rpt.section("## 同布局的 Think − Direct 首位效应")
        rpt.table("Think 效应 (pp)", "Direct 效应 (pp)")
        for key, primary_blocks in sorted(full.items()):
            other_blocks = physical_means(other_layouts.get(key, {}))
            think, direct = (primary_blocks, other_blocks) if mode == "think" else (other_blocks, primary_blocks)
            stratum, kind, _ = key
            for i in ((1,) if kind == "answer" else (0, 1, 2)):
                rpt.paired(f"{stratum} · {kind} · {METRICS[i]}", think, direct, i)
        if heading_kinds:
            rpt.section("### 固定 A 编号为 1 时，各标题下正确率首位效应的模式差")
            rpt.table("Think 效应 (pp)", "Direct 效应 (pp)")
            for kind in heading_kinds:
                for h in ("numbered", "plain", "none"):
                    primary_blocks = headers.get((kind, h), {})
                    other = physical_means(other_layouts.get(("one_correct_pair", kind, h), {}), (True,))
                    think, direct = (primary_blocks, other) if mode == "think" else (other, primary_blocks)
                    rpt.paired(f"{kind} · {h}", think, direct, 1)
    return "\n".join(rpt.lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paired-input", type=Path)
    parser.add_argument("--B", type=int, default=5000)
    args = parser.parse_args()
    if args.B < 1:
        parser.error("--B must be positive")
    result = report(load(args.input), load(args.paired_input) if args.paired_input else None, args.B)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")
