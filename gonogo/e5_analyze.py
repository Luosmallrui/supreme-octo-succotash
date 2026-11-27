"""E5 analysis: does a decoy in slot 1 absorb primacy, and does selection among the real candidates improve?

Conditioning on "the decoy is wrong" under the softmax channel of bayes_select.py is exact renormalisation over the
real slots (a softmax is unchanged by dropping an alternative whose logit does not depend on the hypothesis), so each
decoy call is turned into an N-slot call over the real candidates and scored with the same closed form.

Reported per condition (control = E1 direct, same orderings, no decoy; blank; wrong decoy):
  - probability mass and argmax share of the decoy;
  - slot prior among real candidates: P(real slot j chosen | its candidate is wrong), first real slot vs middle;
  - accuracy of the raw choice among real slots, and of the closed-form posterior with 1 and 3 calls;
  - paired differences against the control on the same sets (problem bootstrap).
The wrong decoy is also split by whether its final answer appears among the real candidates.

  python e5_analyze.py --model qwen3-8b --output out/e5_decoy/report_e5_qwen3-8b.md
"""
import argparse
import collections
from pathlib import Path

import numpy as np

import bayes_select as B

HERE = Path(__file__).resolve().parent


def real_call(r):
    """Decoy call -> call over the real candidates (renormalised), keeping the ordering metadata."""
    p = np.asarray(r["p"][1:], float)
    p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
    return dict(set_id=r["set_id"], order_name=r["order_name"], order=r["order"], p=p.tolist(), p_decoy=r["p"][0],
                decoy_key=r.get("decoy_key"), decoy_key_in_set=r.get("decoy_key_in_set"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--decoy", type=Path, help="default out/e5_decoy/e5_<model>.jsonl")
    ap.add_argument("--data", type=Path, default=HERE / "results", help="E1 sets and direct calls (control)")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    sets = {s["set_id"]: s for s in B.load(args.data / f"e1_sets_{args.model}.jsonl")}
    control = {sid: [r for r in rs if r["order_name"].startswith("r")]
               for sid, rs in B.load_calls(args.data / f"e1_direct_{args.model}.jsonl").items()}
    cond = {"control": control}
    raw = collections.defaultdict(list)
    for r in B.load(args.decoy or HERE / f"out/e5_decoy/e5_{args.model}.jsonl"):
        raw[r["decoy"]].append(r)
    for kind, rs in raw.items():
        cond[kind] = collections.defaultdict(list)
        for r in rs:
            cond[kind][r["set_id"]].append(real_call(r))

    per_set = collections.defaultdict(dict)
    prior_stats = collections.defaultdict(lambda: collections.defaultdict(list))
    for g, N in sorted({(B.group(s), s["N"]) for s in sets.values()}):
        S = [s for s in sets.values() if B.group(s) == g and s["N"] == N and len(control.get(s["set_id"], [])) == 3]
        if len({s["pid"] for s in S}) < 5:
            continue
        fold = B.fold_of(S)
        for name, calls in cond.items():
            Sc = [s for s in S if len(calls.get(s["set_id"], [])) == 3]
            # slot prior among real candidates (argmax over real slots), random orderings
            fa, nw = np.zeros(N), np.zeros(N)
            for s in Sc:
                for r in calls[s["set_id"]]:
                    ch = int(np.argmax(r["p"]))
                    for i, k in enumerate(r["order"]):
                        if not s["cor"][str(k)]:
                            nw[i] += 1
                            fa[i] += ch == i
            fa = fa / np.maximum(nw, 1)
            prior_stats[name][g, N] = (fa[0], fa[1:-1].mean() if N > 2 else np.nan, fa[-1])
            for f in range(5):
                train = [s for s in Sc if fold[s["pid"]] != f]
                test = [s for s in Sc if fold[s["pid"]] == f]
                if not train or not test:
                    continue
                tc = [(r, [s["cor"][str(k)] for k in r["order"]]) for s in train for r in calls[s["set_id"]]]
                alpha, beta = B.fit_channel(tc, N)
                gamma = B.fit_prior(train, "linear")[0]
                for s in test:
                    R = calls[s["set_id"]]
                    lin = B.prior_linear(s, gamma)
                    ev = [B.call_evidence(r, s, alpha, beta) for r in R]
                    d = per_set[s["set_id"]]
                    d.update(pid=s["pid"], group=(g, N), majority=B.accuracy(dict(B.counts(s)), s))
                    d[f"{name}:raw1"] = np.mean([float(s["cor"][str(r["order"][int(np.argmax(r["p"]))])]) for r in R])
                    d[f"{name}:ours1"] = np.mean([B.accuracy(B.fused(lin, [e]), s) for e in ev])
                    d[f"{name}:ours3"] = B.accuracy(B.fused(lin, ev), s)
                    if name != "control":
                        d[f"{name}:p_decoy"] = np.mean([r["p_decoy"] for r in R])

    names = list(cond)
    lines = [f"# E5 首位诱饵 · {args.model} · direct", "",
             "由 `e5_analyze.py` 生成。诱饵调用去掉诱饵后在真实候选上重新归一化（在 softmax 通道下这正是对“诱饵错误”的条件化），"
             "再用 `bayes_select.py` 的闭式后验。control 为同一批顺序上无诱饵的 E1 direct 调用。参数 5 折交叉拟合。", "",
             "## 诱饵吸走的概率", ""]
    for kind, rs in raw.items():
        lines.append(f"- {kind}：{len(rs)} 次调用，诱饵平均概率 {np.mean([r['p'][0] for r in rs]):.3f}，"
                     f"诱饵为 argmax 的比例 {np.mean([r['chose_decoy'] for r in rs]):.3f}。")
    lines += ["", "## 真实候选之间的槽位先验：P(该位被选 | 该位候选错误)", "",
              "| 组 | N | " + " | ".join(f"{n} 第 1 个真实位 / 中间 / 末位" for n in names) + " |",
              "|---|---:|" + "---|" * len(names)]
    for key in sorted(prior_stats["control"]):
        cells = []
        for n in names:
            a, m, z = prior_stats[n].get(key, (np.nan,) * 3)
            cells.append(f"{a:.3f} / {m:.3f} / {z:.3f}（比值 {a / m:.1f}）" if m > 0 else f"{a:.3f} / {m:.3f} / {z:.3f}")
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    cols = ["majority"] + [f"{n}:{m}" for n in names for m in ("raw1", "ours1", "ours3")]
    lines += ["", "## 准确率 (%)", "", "| 组 | N | " + " | ".join(cols) + " |", "|---|---:|" + "---:|" * len(cols)]
    for key in sorted({v["group"] for v in per_set.values() if "group" in v}):
        here = [v for v in per_set.values() if v.get("group") == key]
        lines.append(f"| {key[0]} | {key[1]} | " + " | ".join(
            f"{100 * np.mean([v[c] for v in here if c in v]):.1f}" if any(c in v for v in here) else "—" for c in cols) + " |")
    rng = np.random.default_rng(0)
    lines += ["", "## 合并所有组的配对差（与 control 同一批集合，按原题整群重采样）", "",
              "| 比较 | 差值 (pp) [95% CI] | 集合数 |", "|---|---|---:|"]
    for n in names:
        if n == "control":
            continue
        for m in ("raw1", "ours1", "ours3"):
            est, k = B.pooled_diff(per_set, f"{n}:{m}", f"control:{m}", args.B, rng)
            lines.append(f"| {n}:{m} − control:{m} | {est} | {k} |")
        est, k = B.pooled_diff(per_set, f"{n}:ours3", "majority", args.B, rng)
        lines.append(f"| {n}:ours3 − majority | {est} | {k} |")
    if "wrong" in raw:
        lines += ["", "## 错误诱饵按其答案是否出现在集合中拆分", "",
                  "| 诱饵答案在集合中 | 调用 | 诱饵平均概率 | 真实候选中选到与诱饵同答案的比例 | 真实候选上原始准确率 |", "|---|---:|---:|---:|---:|"]
        for flag in (True, False):
            rs = [r for r in raw["wrong"] if r["decoy_key_in_set"] is flag]
            if not rs:
                continue
            same, acc = [], []
            for r in rs:
                s = sets[r["set_id"]]
                k = r["order"][int(np.argmax(r["p"][1:]))]
                same.append(s["key"][str(k)] == r["decoy_key"])
                acc.append(s["cor"][str(k)])
            lines.append(f"| {'是' if flag else '否'} | {len(rs)} | {np.mean([r['p'][0] for r in rs]):.3f} | "
                         f"{np.mean(same):.3f} | {100 * np.mean(acc):.1f} |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
