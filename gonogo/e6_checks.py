"""Checks on the E6 duel posterior that e6_analyze.py does not compute.

1. The pre-registered posterior (parameters per source x N, as in e6_analyze.py) against majority vote, with the 95%
   interval and a Bonferroni interval for the 8 pre-registered settings (H1 DeepSeek listwise; E5 wrong and blank
   decoy for 2 models; E6 for 2 models; H3 DeepSeek duels).
2. Simple duel heuristics: majority vote with ties broken by the duel Borda sum; Copeland (most duels won, by the sign
   of d_ab); PRP-Allpair (Qin et al., 2024): a win in both orders scores 1, a win in one order 0.5, ties at random.
3. Prediction P5 (N >= 8 only); majority tied vs. unique; how often the posterior overrules a unique majority.
4. Sensitivity: duel parameters (delta, sigma^2) pooled over all groups with gamma per group; all parameters pooled.
5. Exploratory, chosen after seeing the data: an answer-level random effect in the duel model,
       d_ab = s_a - s_b + e_ab,  s_a = delta z_a + u_a,  u_a ~ N(0, tau^2),  e_ab ~ N(0, s2e).
   Marginalising u on the complete graph (D^T D = K I orthogonal to 1, K answers) gives the closed form
       score(a) = gamma n_a + delta / (s2e + K tau^2) * sum_b d_ab,
   which is the pre-registered rule when tau^2 = 0. The pre-registered fit uses sigma^2 = s2e + 2 tau^2, so for K > 2
   it over-weights the duels by (s2e + K tau^2) / (s2e + 2 tau^2). tau^2 is the mean product r_ab r_ac of residuals
   sharing an answer (b != c).

Same folds, seeds and bootstrap as e6_analyze.py (5000 draws over problems; the Bonferroni interval uses 20000).

  python e6_checks.py --data results   (writes out/e6_duels/report_e6_checks.md; adds the hosted judges in API
                                        when their duel and E1 files exist)
"""
import argparse
import collections
from itertools import combinations
from pathlib import Path

import numpy as np

import bayes_select as B
from e6_analyze import EPS, duel_table, load_duels

HERE = Path(__file__).resolve().parent
MODELS = ("qwen3-8b", "qwen3-4b")
API = (("deepseek-flash 评 qwen3-8b 的候选", "qwen3-8b", "out/e6_duels/e6_api-deepseek-flash-prefill_qwen3-8b.jsonl",
        "out/e1_api-deepseek-flash-prefill-direct_qwen3-8b.jsonl"),)  # included when both files exist
SETTINGS = 8


def fit_duel(train, d):
    cor = lambda s, a: a in B.correct_keys(s)
    xs = np.array([cor(s, a) - cor(s, b) for s in train for (a, b) in d[s["set_id"]]], float)
    ys = np.array([v for s in train for v in d[s["set_id"]].values()], float)
    delta = float((xs * ys).sum() / max((xs ** 2).sum(), EPS))
    sigma2 = float(np.mean((ys - delta * xs) ** 2))
    prods = []
    for s in train:
        dd = d[s["set_id"]]
        r = {(a, b): v - delta * (cor(s, a) - cor(s, b)) for (a, b), v in dd.items()}
        for a in sorted({a for a, _ in dd}):
            others = sorted(b for (x, b) in r if x == a)
            prods += [r[a, b] * r[a, c] for b, c in combinations(others, 2)]
    tau2 = max(float(np.mean(prods)) if prods else 0.0, 0.0)
    return delta, sigma2, tau2, max(sigma2 - 2 * tau2, 1e-3)


def evaluate(model, duels, calls_path, data):
    rows = load_duels(duels)
    sets = {s["set_id"]: s for s in B.load(data / f"e1_sets_{model}.jsonl")}
    calls = B.load_calls(calls_path)
    first, d, _, tokens = duel_table(rows)
    cor = lambda s, a: a in B.correct_keys(s)
    groups = sorted({(B.group(s), s["N"]) for s in sets.values()})
    allS = {gN: [s for s in sets.values() if (B.group(s), s["N"]) == gN and s["set_id"] in d
                 and sum(r["order_name"].startswith("r") for r in calls[s["set_id"]]) == 3] for gN in groups}
    folds = {gN: B.fold_of(S) for gN, S in allS.items() if len({s["pid"] for s in S}) >= 5}
    per_set, params, over = {}, collections.defaultdict(list), collections.Counter()
    for gN, fold in folds.items():
        for f in range(5):
            train = [s for s in allS[gN] if fold[s["pid"]] != f]
            test = [s for s in allS[gN] if fold[s["pid"]] == f]
            pooled = [s for g2, f2 in folds.items() for s in allS[g2] if f2[s["pid"]] != f]
            delta, sigma2, tau2, s2e = fit_duel(train, d)
            pdelta, psigma2, _, _ = fit_duel(pooled, d)
            gamma = B.fit_prior(train, "linear")[0]
            pgamma = B.fit_prior(pooled, "linear")[0]
            params[gN].append((delta, sigma2, tau2, s2e))
            for s in test:
                sid = s["set_id"]
                c = B.counts(s)
                keys = list(c)
                borda = {a: sum(v for (x, _), v in d[sid].items() if x == a) for a in keys}
                wins = {a: sum(v > 0 for (x, _), v in d[sid].items() if x == a) for a in keys}
                fo = first[sid]
                prp = {a: sum(((fo[a, b] > 0) + (fo[b, a] < 0)) / 2 for b in keys if b != a and (a, b) in fo and (b, a) in fo)
                       for a in keys}
                top = max(c.values())
                tied = [a for a in keys if c[a] == top]
                post = lambda g, w: {a: g * c[a] + w * borda[a] for a in keys}
                ours = post(gamma, delta / sigma2)
                ahat = max(ours, key=ours.get)
                if len(tied) == 1 and ahat != tied[0]:
                    over["n"] += 1
                    over["ours_right"] += cor(s, ahat)
                    over["majority_right"] += cor(s, tied[0])
                R = [r for r in calls[sid] if r["order_name"].startswith("r")]
                per_set[sid] = dict(
                    pid=s["pid"], group=gN, N=s["N"], tied=len(tied) > 1,
                    majority=B.accuracy(dict(c), s), ours=B.accuracy(ours, s),
                    maj_tb=B.accuracy({a: c[a] + (1e-3 * np.tanh(borda[a]) if a in tied else 0) for a in keys}, s),
                    copeland=B.accuracy({a: wins[a] + 1e-3 * c[a] for a in keys}, s),
                    prp=B.accuracy(prp, s),
                    pooled_duel=B.accuracy(post(gamma, pdelta / psigma2), s),
                    pooled_all=B.accuracy(post(pgamma, pdelta / psigma2), s),
                    re=B.accuracy(post(gamma, delta / (s2e + len(keys) * tau2)), s),
                    K=len(keys), dtok=tokens[sid], ltok=sum(r["n_prompt"] for r in R))
    return per_set, params, over


def bonferroni(per_set, a, b, k, draws=20000):
    ids = list(per_set)
    pids = sorted({per_set[i]["pid"] for i in ids})
    ix = {p: j for j, p in enumerate(pids)}
    g = np.array([ix[per_set[i]["pid"]] for i in ids])
    v = np.array([per_set[i][a] - per_set[i][b] for i in ids], float)
    tot, cnt = np.bincount(g, v, len(pids)), np.bincount(g, None, len(pids)).astype(float)
    draw = np.random.default_rng(0).integers(0, len(pids), (draws, len(pids)))
    bs = tot[draw].sum(1) / cnt[draw].sum(1)
    lo, hi = np.percentile(bs, [100 * 0.025 / k, 100 - 100 * 0.025 / k])
    return f"{100 * v.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", float((bs <= 0).mean())


def report(data, n_boot):
    lines = ["# E6 补充检查：对决后验的稳健性、简单启发式与答案层面的随机效应", "",
             "由 `e6_checks.py` 生成。折、种子与 `e6_analyze.py` 相同；区间按原题整群重采样 5000 次。"
             "“事先登记”指 `e6_analyze.py` 的 duel-ours（参数按来源 × N 分组）。第 5 节的随机效应版本是看过数据后提出的，只作探索。", ""]
    judges = [(m, m, HERE / f"out/e6_duels/e6_{m}.jsonl", data / f"e1_direct_{m}.jsonl") for m in MODELS]
    judges += [(name, m, HERE / d, HERE / c) for name, m, d, c in API if (HERE / d).exists() and (HERE / c).exists()]
    for name, model, duels, calls in judges:
        ps, params, over = evaluate(model, duels, calls, data)
        rng = np.random.default_rng(0)
        diff = lambda sub, a, b: B.pooled_diff(sub, a, b, n_boot, rng)[0]
        text, p_one = bonferroni(ps, "ours", "majority", SETTINGS)
        sub = lambda f: {k: v for k, v in ps.items() if f(v)}
        lines += [f"## {name}", "", "| 比较 | 差值 (pp) [95% CI] | 集合数 |", "|---|---|---:|",
                  f"| 1. 事先登记 − 多数投票 | {diff(ps, 'ours', 'majority')} | {len(ps)} |",
                  f"| 1. 同上，Bonferroni 区间（{SETTINGS} 个事先登记的设置，{100 - 5 / SETTINGS:.1f}%） | {text}，单侧 bootstrap p = {p_one:.4f} | {len(ps)} |",
                  f"| 2. 多数投票 + 对决 Borda 打破平局 − 多数投票 | {diff(ps, 'maj_tb', 'majority')} | {len(ps)} |",
                  f"| 2. 事先登记 − 多数投票 + 对决打破平局 | {diff(ps, 'ours', 'maj_tb')} | {len(ps)} |",
                  f"| 2. Copeland（赢的对决数）− 多数投票 | {diff(ps, 'copeland', 'majority')} | {len(ps)} |",
                  f"| 2. PRP-Allpair（两种顺序都赢计 1 分，只赢一种计 0.5 分）− 多数投票 | {diff(ps, 'prp', 'majority')} | {len(ps)} |",
                  f"| 2. 事先登记 − PRP-Allpair | {diff(ps, 'ours', 'prp')} | {len(ps)} |"]
        big = sub(lambda v: v["N"] >= 8)
        tied, untied = sub(lambda v: v["tied"]), sub(lambda v: not v["tied"])
        lines += [f"| 3. P5：N ≥ 8，事先登记 − 多数投票 | {diff(big, 'ours', 'majority')} | {len(big)} |",
                  f"| 3. 多数票并列的集合 | {diff(tied, 'ours', 'majority')} | {len(tied)} |",
                  f"| 3. 多数票唯一的集合 | {diff(untied, 'ours', 'majority')} | {len(untied)} |",
                  f"| 4. 对决参数跨组合并（γ 仍分组）− 多数投票 | {diff(ps, 'pooled_duel', 'majority')} | {len(ps)} |",
                  f"| 4. 全部参数跨组合并 − 多数投票 | {diff(ps, 'pooled_all', 'majority')} | {len(ps)} |",
                  f"| 5. 随机效应版本 − 多数投票（探索） | {diff(ps, 're', 'majority')} | {len(ps)} |",
                  f"| 5. 随机效应版本 − 事先登记（探索） | {diff(ps, 're', 'ours')} | {len(ps)} |", ""]
        acc = ", ".join(f"{m} {100 * np.mean([v[m] for v in ps.values()]):.1f}"
                        for m in ("majority", "ours", "maj_tb", "copeland", "prp", "pooled_duel", "re"))
        lines += [f"合并准确率 (%)：{acc}。", ""]
        ratio = sum(v["dtok"] for v in ps.values()) / sum(v["ltok"] for v in ps.values())
        lines += [f"多数票唯一时，事先登记的后验推翻多数票 {over['n']} 次：其中后验正确 {over['ours_right']} 次，"
                  f"多数票正确 {over['majority_right']} 次。合并所有组，对决的输入 token 为 3 次列表式调用的 {ratio:.2f} 倍。", "",
                  "| 组 | N | 平均答案数 K | δ | σ²（总残差） | τ²（答案层面） | s2e（对决层面） | 事先登记相对随机效应的过度加权（平均 K 下） "
                  "| 多数投票 (%) | 事先登记 (%) | 随机效应 (%) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for gN, v in sorted(params.items()):
            de, s2, t2, s2e = np.mean(v, 0)
            here = [x for x in ps.values() if x["group"] == gN]
            mK = np.mean([x["K"] for x in here])
            acc = lambda m: f"{100 * np.mean([x[m] for x in here]):.1f}"
            lines.append(f"| {gN[0]} | {gN[1]} | {mK:.1f} | {de:.2f} | {s2:.2f} | {t2:.2f} | {s2e:.2f} | "
                         f"×{(s2e + mK * t2) / s2:.2f} | {acc('majority')} | {acc('ours')} | {acc('re')} |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "results", help="E1 sets and direct calls")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--output", type=Path, default=HERE / "out/e6_duels/report_e6_checks.md")
    args = ap.parse_args()
    args.output.write_text(report(args.data, args.B), encoding="utf-8")
    print(f"wrote {args.output}")
