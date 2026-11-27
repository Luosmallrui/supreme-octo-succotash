"""E6 analysis: closed-form posterior from order-balanced answer duels, against majority vote and listwise selection.

Model. For the duel with answer a shown first and b second, logit p_ab = h + delta (z_a - z_b) + e, where h is the
first-position advantage and z marks the correct answer. With both orders, d_ab = (logit p_ab - logit p_ba) / 2
cancels h exactly, and d_ab ~ N(delta (z_a - z_b), sigma^2).

Objective and closed form. Maximising P(a* = a | counts, duels) (Bayes decision under 0-1 loss) gives
    score(a) = gamma n_a + (delta / sigma^2) sum_{b != a} d_ab,      a_hat = argmax_a score(a),
because the delta^2 term is the same under every hypothesis (each has K-1 pairs involving the correct answer).
The least-squares (Borda) sum of d_ab is thus the sufficient statistic of the duels. delta and sigma^2 come from OLS of
d_ab on (z_a - z_b) on held-out problems; gamma as in bayes_select.py. 5-fold cross-fitting by problem.

Compared with: majority vote; duels without counts (gamma = 0); duels read in one order only (h not cancelled);
listwise E1 selection on the same sets (raw single call, vote over 3 orderings, closed-form listwise posterior with
1 and 3 calls, from bayes_select.py). Token cost is reported per set.

  python e6_analyze.py --model qwen3-8b --output out/e6_duels/report_e6_qwen3-8b.md
  python e6_analyze.py --model qwen3-8b --label deepseek-flash \n      --duels out/e6_duels/e6_api-deepseek-flash-prefill_qwen3-8b.jsonl \n      --calls out/e1_api-deepseek-flash-prefill-direct_qwen3-8b.jsonl   (hosted judge, api_select.py --exp e6)
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

import bayes_select as B

HERE = Path(__file__).resolve().parent
EPS = 1e-6


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


def load_duels(path):
    """Duel rows, one per set. API files (api_select.py --exp e6) hold one row per distinct prompt with the sets
    that use it under "sets", and may hold failed attempts: keep the last successful row per request and expand."""
    last = {}
    for r in B.load(path):
        if "error" not in r and r.get("p") is not None:
            last[r["request_id"]] = r
    rows = []
    for r in last.values():
        for s in r.get("sets") or [None]:
            rows.append(r if s is None else dict({k: v for k, v in r.items() if k != "sets"}, **s))
    return rows


def duel_table(rows):
    """set_id -> {(a, b): d_ab, ...} with both orders present, plus the first-position logits and token counts."""
    first = collections.defaultdict(dict)
    tokens = collections.Counter()
    for r in rows:
        p1 = r["p"][0] / max(r["p"][0] + r["p"][1], EPS)
        first[r["set_id"]][r["key_a"], r["key_b"]] = logit(p1)
        tokens[r["set_id"]] += r["n_prompt"]
    d, h = {}, {}
    for sid, lo in first.items():
        d[sid] = {(a, b): (v - lo[b, a]) / 2 for (a, b), v in lo.items() if (b, a) in lo}
        h[sid] = [(v + lo[b, a]) / 2 for (a, b), v in lo.items() if (b, a) in lo and a < b]
    return first, d, h, tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--duels", type=Path, help="default out/e6_duels/e6_<model>.jsonl")
    ap.add_argument("--data", type=Path, default=HERE / "results", help="E1 sets and direct calls")
    ap.add_argument("--calls", type=Path, help="listwise E1 calls of the same judge (default <data>/e1_direct_<model>.jsonl)")
    ap.add_argument("--label", help="judge name in the report title (default: the model)")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    rows = load_duels(args.duels or HERE / f"out/e6_duels/e6_{args.model}.jsonl")
    sets = {s["set_id"]: s for s in B.load(args.data / f"e1_sets_{args.model}.jsonl")}
    calls = B.load_calls(args.calls or args.data / f"e1_direct_{args.model}.jsonl")
    first, d, h, tokens = duel_table(rows)

    per_set, stats = {}, collections.defaultdict(list)
    for g, N in sorted({(B.group(s), s["N"]) for s in sets.values()}):
        S = [s for s in sets.values() if B.group(s) == g and s["N"] == N and s["set_id"] in d
             and sum(r["order_name"].startswith("r") for r in calls[s["set_id"]]) == 3]
        if len({s["pid"] for s in S}) < 5:
            continue
        fold = B.fold_of(S)
        for f in range(5):
            train = [s for s in S if fold[s["pid"]] != f]
            test = [s for s in S if fold[s["pid"]] == f]
            cor = lambda s, a: a in B.correct_keys(s)
            xs = np.array([cor(s, a) - cor(s, b) for s in train for (a, b) in d[s["set_id"]]], float)
            ys = np.array([v for s in train for v in d[s["set_id"]].values()], float)
            delta = float((xs * ys).sum() / max((xs ** 2).sum(), EPS))
            sigma2 = float(np.mean((ys - delta * xs) ** 2))
            gamma = B.fit_prior(train, "linear")[0]
            tc = [(r, [s["cor"][str(k)] for k in r["order"]]) for s in train for r in calls[s["set_id"]]
                  if r["order_name"].startswith("r")]
            alpha, beta = B.fit_channel(tc, N)
            stats[g, N].append((delta, sigma2, gamma))
            for s in test:
                sid = s["set_id"]
                keys = list(B.counts(s))
                borda = {a: sum(v for (x, _), v in d[sid].items() if x == a) for a in keys}
                one = {a: 0.0 for a in keys}          # one order only: the alphabetically first answer is shown first
                for (a, b), v in first[sid].items():
                    if a < b:
                        one[a] += v
                        one[b] -= v
                lin = B.prior_linear(s, gamma)
                R = [r for r in calls[sid] if r["order_name"].startswith("r")]
                ev = [B.call_evidence(r, s, alpha, beta) for r in R]
                votes, mass = collections.Counter(), collections.Counter()
                for r in R:
                    k = s["key"][str(r["order"][int(np.argmax(r["p"]))])]
                    votes[k] += 1
                    mass[k] += max(r["p"])
                best = max(votes, key=lambda a: (votes[a], mass[a]))
                res = {"majority": B.accuracy(dict(B.counts(s)), s),
                       "duel-only": B.accuracy(borda, s) if len(keys) > 1 else 1.0 * (keys[0] in B.correct_keys(s)),
                       "duel-one-order": B.accuracy(one, s),
                       "duel-ours": B.accuracy({a: lin[a] + delta / sigma2 * borda[a] for a in keys}, s),
                       "list-raw1": np.mean([float(s["cor"][str(r["order"][int(np.argmax(r["p"]))])]) for r in R]),
                       "list-vote3": float(best in B.correct_keys(s)),
                       "list-ours1": np.mean([B.accuracy(B.fused(lin, [e]), s) for e in ev]),
                       "list-ours3": B.accuracy(B.fused(lin, ev), s),
                       "duel-tokens": tokens[sid], "list3-tokens": sum(r["n_prompt"] for r in R)}
                per_set[sid] = dict(res, pid=s["pid"], group=(g, N))

    methods = ["majority", "duel-only", "duel-one-order", "duel-ours", "list-raw1", "list-vote3", "list-ours1", "list-ours3"]
    title = args.model if not args.label else f"{args.label} 评 {args.model} 的候选"
    lines = [f"# E6 位置平衡的答案对决 · {title} · direct", "",
             "由 `e6_analyze.py` 生成。`duel-ours` 为闭式后验 γn_a + (δ/σ²)Σ_b d_ab；`duel-only` 为 γ=0；`duel-one-order` 只用一种顺序"
             "（首位优势 h 未消去）；`list-*` 为同一批集合上的 E1 列表式选择（`bayes_select.py` 的方法）。参数 5 折交叉拟合。数值为准确率 (%)。", "",
             "## 各组准确率", "",
             "| 组 | N | 集合 | " + " | ".join(methods) + " | 对决 token / 3 次列表 token |",
             "|---|---:|---:|" + "---:|" * (len(methods) + 1)]
    for (g, N) in sorted({v["group"] for v in per_set.values()}):
        here = [v for v in per_set.values() if v["group"] == (g, N)]
        ratio = sum(v["duel-tokens"] for v in here) / sum(v["list3-tokens"] for v in here)
        lines.append(f"| {g} | {N} | {len(here)} | " + " | ".join(f"{100 * np.mean([v[m] for v in here]):.1f}" for m in methods)
                     + f" | {ratio:.2f} |")
    allv = list(per_set.values())
    ratio = sum(v["duel-tokens"] for v in allv) / sum(v["list3-tokens"] for v in allv)
    lines.append(f"| 合并 | — | {len(allv)} | " + " | ".join(f"{100 * np.mean([v[m] for v in allv]):.1f}" for m in methods)
                 + f" | {ratio:.2f} |")
    rng = np.random.default_rng(0)
    lines += ["", "## 合并所有组的配对差（按原题整群重采样）", "", "| 比较 | 差值 (pp) [95% CI] | 集合数 |", "|---|---|---:|"]
    for a, b in [("duel-ours", "majority"), ("duel-ours", "list-ours3"), ("duel-ours", "list-vote3"), ("duel-ours", "list-ours1"),
                 ("duel-only", "list-raw1"), ("duel-only", "duel-one-order"), ("duel-ours", "duel-only")]:
        est, n = B.pooled_diff(per_set, a, b, args.B, rng)
        lines.append(f"| {a} − {b} | {est} | {n} |")
    pc = [r["p"][0] / (r["p"][0] + r["p"][1]) if r["cor_a"] else r["p"][1] / (r["p"][0] + r["p"][1])
          for r in rows if r["cor_a"] != r["cor_b"] and r["p"][0] + r["p"][1] > 0]
    hs = [x for v in h.values() for x in v]
    lines += ["", "## 对决本身", "",
              f"- 一对一错的对决：{len(pc)} 次，正确答案的平均选择概率 {np.mean(pc):.3f}（两种顺序合并），"
              f"选择概率 > 0.5 的比例 {np.mean(np.array(pc) > 0.5):.3f}。",
              f"- 首位优势 h（两种顺序 logit 的平均，题内两两）：均值 {np.mean(hs):+.2f}，中位数 {np.median(hs):+.2f}"
              f"（logit 单位；正值表示偏向先出现的答案）。",
              "", "## 拟合参数（5 折平均）", "", "| 组 | N | δ | σ² | δ/σ² | γ |", "|---|---:|---:|---:|---:|---:|"]
    for (g, N), v in sorted(stats.items()):
        de, s2, ga = np.mean(v, 0)
        lines.append(f"| {g} | {N} | {de:.2f} | {s2:.2f} | {de / s2:.2f} | {ga:.2f} |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
