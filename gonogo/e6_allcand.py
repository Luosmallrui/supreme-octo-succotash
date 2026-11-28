"""Count-free duels over individual candidates (queue 17) against listwise selection on the same N=8 sets.

Gate (7-day plan, day 2): can a duel schedule that costs no more input tokens than the vote over 3 orderings beat it,
when candidates are not grouped by final answer (the count-free regime, e.g. code or open-ended outputs)?

Data. out/e6_duels/e6_allcand_N8_<model>.jsonl (e6_run.py --all-candidates --n 8): for every ordered pair of the 8
candidates, P(Solution 1) and P(Solution 2) in a two-candidate prompt. With L_ab = logit P(first wins | a first),
d_ab = (L_ab - L_ba) / 2 is the order-balanced preference for a over b and h_ab = (L_ab + L_ba) / 2 the first-slot
advantage. Listwise calls: the 3 random orderings of E1 (<data>/e1_direct_<model>.jsonl).

Count-free methods use only the selector's outputs: no answer keys, no counts.
  listwise  raw1 (one call, mean over the 3 orderings); vote3c (vote over 3 orderings per candidate, ties by summed
            top probability); soft3 (summed label probability per candidate)
  duels     full56 (all 56 ordered pairs, Borda sum of d); copeland56; full28 (one random order per pair, minus the
            cross-fitted mean first-slot advantage); knockout14 (random bracket of 7 balanced matches); knockout7
            (one order per match, debiased); champ14 (king of the hill, balanced); rand-m (m random balanced pairs,
            least-squares scores on the observed graph); budget (as many random balanced pairs as the 3-order vote's
            tokens allow); active5 (4 disjoint balanced pairs, then the two winners with the largest least-squares
            score meet: 10 calls); budget1 (as many single-order duels as the tokens allow, debiased); hybrid (one listwise call, then a balanced duel between its top two); hybrid3 (soft3 top
            two, then a balanced duel)
Reference (uses answer keys): vote3 (the paper's 3-order vote, votes pooled by answer), majority.
Randomised schedules are averaged over SCHED draws per set (fixed seeds). Accuracy is the expected accuracy of the
argmax with ties broken at random. Intervals: 95% bootstrap over problems, 5000 draws.

  python e6_allcand.py --data results   (writes out/e6_duels/report_e6_allcand.md)
"""
import argparse
import collections
import itertools
import zlib
from pathlib import Path

import numpy as np

import bayes_select as B

HERE = Path(__file__).resolve().parent
MODELS = ("qwen3-8b", "qwen3-4b")
EPS = 1e-6
SCHED = 200


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return float(np.log(p / (1 - p)))


def load_matrix(path):
    """set_id -> {(a, b): (L_ab, input tokens)} for the ordered pair with a shown first."""
    by = collections.defaultdict(dict)
    for r in B.load(path):
        if "error" in r or r.get("p") is None:
            continue
        p = r["p"]
        by[r["set_id"]][r["k_a"], r["k_b"]] = (logit(p[0] / max(p[0] + p[1], EPS)), r["n_prompt"])
    return by


def acc(scores, cor):
    top = max(scores.values())
    tied = [k for k, v in scores.items() if v >= top - 1e-9]
    return float(np.mean([cor[k] for k in tied]))


def ls_scores(ks, pairs, d):
    """Least-squares scores s minimising sum over observed pairs of (d_ab - s_a + s_b)^2 (minimum-norm solution)."""
    ix = {k: i for i, k in enumerate(ks)}
    Lap, rhs = np.zeros((len(ks), len(ks))), np.zeros(len(ks))
    for a, b in pairs:
        i, j = ix[a], ix[b]
        Lap[i, i] += 1; Lap[j, j] += 1; Lap[i, j] -= 1; Lap[j, i] -= 1
        rhs[i] += d[a, b]; rhs[j] -= d[a, b]
    s = np.linalg.pinv(Lap) @ rhs
    return {k: float(s[ix[k]]) for k in ks}


def match(a, b, pref, rng):
    """Winner of one match; pref > 0 favours a, ties at random."""
    v = pref(a, b)
    return a if v > 0 else b if v < 0 else (a if rng.random() < 0.5 else b)


def knockout(ks, pref, rng):
    field = list(rng.permutation(ks))
    while len(field) > 1:
        field = [match(field[i], field[i + 1], pref, rng) for i in range(0, len(field), 2)]
    return field[0]


def champion(ks, pref, rng):
    order = list(rng.permutation(ks))
    champ = order[0]
    for c in order[1:]:
        champ = match(champ, c, pref, rng)
    return champ


def evaluate(model, data, matrix_path):
    M = load_matrix(matrix_path)
    sets = {s["set_id"]: s for s in B.load(data / f"e1_sets_{model}.jsonl") if s["set_id"] in M}
    calls = B.load_calls(data / f"e1_direct_{model}.jsonl")
    S = [s for s in sets.values() if sum(r["order_name"].startswith("r") for r in calls[s["set_id"]]) == 3]
    fold = B.fold_of(S)
    h_all = {}   # set_id -> mean first-slot advantage over its pairs
    for s in S:
        m = M[s["set_id"]]
        h_all[s["set_id"]] = float(np.mean([(m[a, b][0] + m[b, a][0]) / 2
                                            for a, b in itertools.combinations(sorted(int(k) for k in s["key"]), 2)]))
    per_set, cost, diag = {}, collections.defaultdict(list), collections.defaultdict(list)
    for s in S:
        sid = s["set_id"]
        m = M[sid]
        ks = sorted(int(k) for k in s["key"])
        cor = {k: bool(s["cor"][str(k)]) for k in ks}
        h_hat = float(np.mean([h_all[t["set_id"]] for t in S if fold[t["pid"]] != fold[s["pid"]]]))   # cross-fitted
        L = {ab: v[0] for ab, v in m.items()}
        tok = {ab: v[1] for ab, v in m.items()}
        d = {(a, b): (L[a, b] - L[b, a]) / 2 for a in ks for b in ks if a != b}
        upairs = list(itertools.combinations(ks, 2))
        rng = np.random.default_rng(zlib.crc32(f"allcand|{model}|{sid}".encode()))
        R = [r for r in calls[sid] if r["order_name"].startswith("r")]
        t3 = sum(r["n_prompt"] for r in R)
        res = dict(pid=s["pid"], group=B.group(s))
        res["oracle"] = float(any(cor.values()))
        res["random"] = float(np.mean(list(cor.values())))
        res["majority"] = B.accuracy(dict(B.counts(s)), s)
        # listwise, count-free
        res["raw1"] = float(np.mean([cor[r["order"][int(np.argmax(r["p"]))]] for r in R]))
        votes, mass, soft = collections.Counter(), collections.Counter(), collections.Counter()
        for r in R:
            p = B.dist(r)
            k = r["order"][int(np.argmax(p))]
            votes[k] += 1
            mass[k] += float(p.max())
            for i, kk in enumerate(r["order"]):
                soft[kk] += float(p[i])
        res["vote3c"] = acc({k: votes[k] + 1e-3 * mass[k] for k in ks}, cor)
        res["soft3"] = acc({k: soft[k] for k in ks}, cor)
        kv, km = collections.Counter(), collections.Counter()   # the paper's vote3: votes pooled by answer key
        for r in R:
            p = B.dist(r)
            a = s["key"][str(r["order"][int(np.argmax(p))])]
            kv[a] += 1
            km[a] += float(p.max())
        res["vote3"] = float(max(kv, key=lambda a: (kv[a], km[a])) in B.correct_keys(s))
        # duels, count-free
        res["full56"] = acc({a: sum(d[a, b] for b in ks if b != a) for a in ks}, cor)
        res["copeland56"] = acc({a: sum(d[a, b] > 0 for b in ks if b != a) + 1e-3 * np.tanh(sum(d[a, b] for b in ks if b != a))
                                 for a in ks}, cor)
        bal = lambda a, b: d[a, b]
        one = lambda o: (lambda a, b: (L[a, b] - h_hat) if o[a, b] else -(L[b, a] - h_hat))
        f28, k14, k7, c14, hyb, hyb3 = [], [], [], [], [], []
        for _ in range(SCHED):
            o = {}
            for a, b in upairs:
                first = rng.random() < 0.5
                o[a, b], o[b, a] = first, not first
            sc = {a: 0.0 for a in ks}
            for a, b in upairs:
                v = one(o)(a, b)
                sc[a] += v; sc[b] -= v
            f28.append(acc(sc, cor))
            k14.append(cor[knockout(ks, bal, rng)])
            k7.append(cor[knockout(ks, one(o), rng)])
            c14.append(cor[champion(ks, bal, rng)])
        res["full28"], res["knockout14"], res["knockout7"], res["champ14"] = map(np.mean, (f28, k14, k7, c14))
        for r in R:
            p = B.dist(r)
            top2 = [r["order"][i] for i in np.argsort(-p)[:2]]
            hyb.append(cor[match(top2[0], top2[1], bal, rng)])
        res["hybrid"] = float(np.mean(hyb))
        top2 = sorted(ks, key=lambda k: -soft[k])[:2]
        res["hybrid3"] = float(cor[match(top2[0], top2[1], bal, rng)])
        pair_tok = {(a, b): tok[a, b] + tok[b, a] for a, b in upairs}
        m_budget = int(t3 // np.mean(list(pair_tok.values())))
        for mm in (4, 7, 14):
            res[f"rand{mm}"] = float(np.mean([acc(ls_scores(ks, [upairs[i] for i in rng.choice(len(upairs), mm, replace=False)], d), cor)
                                              for _ in range(SCHED)]))
        bud, btok = [], []
        for _ in range(SCHED):
            chosen = [upairs[i] for i in rng.choice(len(upairs), min(m_budget, len(upairs)), replace=False)]
            bud.append(acc(ls_scores(ks, chosen, d), cor) if chosen else res["random"])
            btok.append(sum(pair_tok[p] for p in chosen))
        res["budget"] = float(np.mean(bud))
        act, b1 = [], []
        m1 = int(t3 // np.mean(list(tok.values())))
        for _ in range(SCHED):
            perm = list(rng.permutation(ks))   # active5: 4 disjoint pairs, then the two widest winners meet
            first = [(perm[i], perm[i + 1]) for i in range(0, 8, 2)]
            sc = ls_scores(ks, first, d)
            win = sorted({max(p, key=lambda k: sc[k]) for p in first}, key=lambda k: -sc[k])[:2]
            act.append(cor[match(win[0], win[1], bal, rng)])
            chosen = [upairs[i] for i in rng.choice(len(upairs), min(m1, len(upairs)), replace=False)]
            dd = {}
            for a, b in chosen:   # one random order per pair, minus the cross-fitted first-slot advantage
                v = (L[a, b] - h_hat) if rng.random() < 0.5 else -(L[b, a] - h_hat)
                dd[a, b] = v
            b1.append(acc(ls_scores(ks, chosen, dd), cor))
        res["active5"], res["budget1"] = float(np.mean(act)), float(np.mean(b1))
        per_set[sid] = res
        mean_duel = float(np.mean(list(tok.values())))
        cost["vote3"].append(t3); cost["vote3c"].append(t3); cost["soft3"].append(t3); cost["raw1"].append(t3 / 3)
        cost["full56"].append(sum(tok.values())); cost["copeland56"].append(sum(tok.values()))
        cost["full28"].append(28 * mean_duel); cost["knockout14"].append(14 * mean_duel); cost["champ14"].append(14 * mean_duel)
        cost["knockout7"].append(7 * mean_duel); cost["hybrid"].append(t3 / 3 + 2 * mean_duel)
        cost["hybrid3"].append(t3 + 2 * mean_duel)
        for mm in (4, 7, 14):
            cost[f"rand{mm}"].append(2 * mm * mean_duel)
        cost["budget"].append(float(np.mean(btok)))
        cost["active5"].append(10 * mean_duel); cost["budget1"].append(m1 * mean_duel)
        cost["majority"].append(0); cost["oracle"].append(0); cost["random"].append(0)
        diag["m_budget"].append(m_budget)
        diag["h"].append(h_all[sid])
        diag["first_wins"].append(np.mean([L[a, b] > 0 for a in ks for b in ks if a != b]))
        diag["dc"] += [d[a, b] for a in ks for b in ks if a != b and cor[a] and not cor[b]]
        diag["dsame"] += [d[a, b] for a in ks for b in ks if a < b and cor[a] == cor[b]]
    return per_set, cost, diag, len(S)


CALLS = dict(oracle="—", random="0", majority="0", raw1="1 listwise", vote3="3 listwise", vote3c="3 listwise",
             soft3="3 listwise", full56="56 duels", copeland56="56 duels", full28="28 duels", knockout14="14 duels",
             champ14="14 duels", knockout7="7 duels", rand4="8 duels", rand7="14 duels", rand14="28 duels",
             budget="token-matched", active5="10 duels", budget1="token-matched, 1 order", hybrid="1 listwise + 2 duels", hybrid3="3 listwise + 2 duels")
NAMES = dict(oracle="至少一个正确（上限）", random="随机选一个", majority="多数投票（用答案分组，参照）",
             raw1="单次列表调用", vote3="3 顺序投票（按答案合并票，论文口径）", vote3c="3 顺序投票（按候选计票，不分组）",
             soft3="3 顺序概率求和（不分组）", full56="完整矩阵 Borda（双序）", copeland56="完整矩阵 Copeland（双序）",
             full28="完整矩阵，每对单序 + 交叉拟合去偏", knockout14="淘汰赛，每场双序", champ14="擂台赛，每场双序",
             knockout7="淘汰赛，每场单序 + 去偏", rand4="随机 4 对（双序）+ 最小二乘", rand7="随机 7 对（双序）+ 最小二乘",
             rand14="随机 14 对（双序）+ 最小二乘", budget="与 3 顺序投票等 token 的随机双序对 + 最小二乘",
             active5="两轮主动：4 组不相交双序对，取优势最大的 2 个胜者再对决",
             budget1="与 3 顺序投票等 token 的随机单序对决 + 去偏 + 最小二乘",
             hybrid="1 次列表调用取前 2，再双序对决", hybrid3="3 顺序概率求和取前 2，再双序对决")
ORDER = ["oracle", "random", "majority", "raw1", "vote3", "vote3c", "soft3", "full56", "copeland56", "full28",
         "knockout14", "champ14", "knockout7", "rand4", "rand7", "rand14", "budget", "active5", "budget1", "hybrid", "hybrid3"]
COUNT_FREE_DUELS = ["full56", "copeland56", "full28", "knockout14", "champ14", "knockout7", "rand4", "rand7", "rand14",
                    "budget", "active5", "budget1", "hybrid", "hybrid3", "soft3", "vote3c", "raw1"]


def report(data, n_boot):
    lines = ["# 不分组的候选级对决 vs 列表式选择（N = 8，队列 17）", "",
             "由 `e6_allcand.py` 生成。每个候选单独作为一个单位，不按最终答案分组、不用票数（代码等无法按答案聚类的场景）。"
             "对决：两候选提示，读 Solution 1/2 的概率；d_ab = (L_ab − L_ba)/2 抵消首位优势。随机调度在每个集合上取 "
             f"{SCHED} 次抽样的平均；正确率是 argmax 的期望正确率（并列时随机）。区间按原题整群重采样 {n_boot} 次。"
             "token 为输入 token（不计前缀缓存）。", ""]
    for model in MODELS:
        path = HERE / f"out/e6_duels/e6_allcand_N8_{model}.jsonl"
        if not path.exists():
            continue
        ps, cost, diag, n = evaluate(model, data, path)
        t3 = np.mean(cost["vote3"])
        groups = [("math500", "MATH500"), ("comp", "竞赛题")]
        lines += [f"## {model}（{n} 个集合）", "",
                  "| 方法 | 调用 | 输入 token / 集合 | 相对 3 顺序投票 | 全部 (%) | MATH500 (%) | 竞赛题 (%) |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for mth in ORDER:
            vals = [ps[k][mth] for k in ps]
            by = {g: np.mean([ps[k][mth] for k in ps if ps[k]["group"] == g]) for g, _ in groups}
            c = np.mean(cost[mth])
            lines.append(f"| {NAMES[mth]} | {CALLS[mth]} | {c:,.0f} | {c / t3:.2f}× | {100 * np.mean(vals):.1f} | "
                         f"{100 * by['math500']:.1f} | {100 * by['comp']:.1f} |")
        rng = np.random.default_rng(0)
        lines += ["", "### 配对差（pp，按原题整群重采样）", "",
                  "| 方法 | − 3 顺序投票（论文口径） | − 3 顺序投票（不分组） | − 单次列表调用 | − 多数投票（需答案分组） | 相对 token |",
                  "|---|---|---|---|---|---:|"]
        for mth in COUNT_FREE_DUELS:
            cells = [B.pooled_diff(ps, mth, ref, n_boot, rng)[0] for ref in ("vote3", "vote3c", "raw1", "majority")]
            lines.append(f"| {NAMES[mth]} | " + " | ".join(cells) + f" | {np.mean(cost[mth]) / t3:.2f}× |")
        for g, gname in groups:
            sub = {k: v for k, v in ps.items() if v["group"] == g}
            cells = [f"{NAMES[mth]}: {B.pooled_diff(sub, mth, 'vote3', n_boot, rng)[0]}" for mth in ("full56", "budget", "active5", "budget1", "knockout7")]
            lines.append(f"| {gname}，− 3 顺序投票（论文口径） | " + "；".join(cells) + " | | | | |")
        dc, ds = np.array(diag["dc"]), np.array(diag["dsame"])
        lines += ["", "### 诊断", "",
                  f"- 等 token 预算下可做的双序对数：中位数 {np.median(diag['m_budget']):.0f}（范围 {min(diag['m_budget'])}–{max(diag['m_budget'])}，完整为 28）。",
                  f"- 两候选提示的首位优势 h（logit）：集合均值 {np.mean(diag['h']):+.2f}；单序时第一个获胜的比例 {100 * np.mean(diag['first_wins']):.1f}%。",
                  f"- 正确对错误的 d：均值 {dc.mean():+.2f}，d > 0 的比例 {100 * (dc > 0).mean():.1f}%（{len(dc)} 对）；同为对或同为错的 |d| 均值 {np.abs(ds).mean():.2f}。",
                  ""]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "results", help="E1 sets and direct calls")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--output", type=Path, default=HERE / "out/e6_duels/report_e6_allcand.md")
    args = ap.parse_args()
    text = report(args.data, args.B)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
