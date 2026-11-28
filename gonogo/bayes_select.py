"""Position-aware Bayes selection: the closed-form posterior over final answers, evaluated on the E1 natural sets.

Objective. Choose the final answer a that maximises the posterior probability of being correct (Bayes decision under
0-1 loss, i.e. maximal expected accuracy), given the answer counts of the N sampled candidates and M selector calls
under orderings pi_1..pi_M.

Model.
  Counts   candidates are i.i.d. draws; each is correct with probability theta, otherwise it takes a wrong answer
           with probability ~q0 per value. Then log P(n | a* = a) = gamma * n_a + const, gamma = log(theta/((1-theta) q0)).
  Selector conditional logit over slots (the choice model of the paper, Sec. 4):
               P(slot s | z) = softmax_s(alpha_s + beta_s * z_s),  z_s = 1 if the candidate in slot s is correct.
           alpha_s is the slot prior, beta_s the slot's sensitivity to correctness (beta_1 is small: primacy is not
           additive in the log-odds, which is why dividing out a prior fails).
  Evidence the selector's full next-token distribution p_m over slots; the score uses the expected log-likelihood
           of a choice drawn from p_m, i.e. it minimises sum_m KL(p_m || P(. | a)).

Closed-form solution. With pi_s = softmax(alpha)_s and S_m(a) the slots holding answer a in call m,
  score(a) = gamma * n_a + sum_m [ sum_{s in S_m(a)} p_ms * beta_s - log(1 + sum_{s in S_m(a)} pi_s (exp(beta_s) - 1)) ]
  a_hat    = argmax_a score(a).
To first order in beta the bracket is sum_{s in S_m(a)} beta_s (p_ms - pi_s): observed minus position-expected mass,
weighted by slot sensitivity. Special cases: beta -> 0 gives majority vote; gamma = 0 gives position-debiased selection.

Fitting. alpha, beta (one beta per slot bin: first, second, middle, last) by maximum likelihood on the random-order
calls of held-out problems; gamma by maximum likelihood of the correct answer under the count prior on the same
problems. 5-fold cross-fitting by problem, folds as in e1_analyze.py.

Comparisons: majority vote; raw single call; slot-prior division (PriDe); vote over the 3 random orderings; summed
probability over the N cyclic shifts (all from e1_analyze.py). Variants: no count prior (debiasing only); a single
beta for all slots; a hierarchical Beta / Chinese-restaurant count prior. Pooled differences are paired by problem.

  python bayes_select.py --model qwen3-8b --mode direct   (reads results/e1_{mode}_{model}.jsonl, e1_sets_{model}.jsonl)
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp

HERE = Path(__file__).resolve().parent


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def group(s):
    return "math500" if s["source"] == "math500" else "comp"


def slot_bins(N):
    b = np.full(N, 2)
    b[0], b[1], b[-1] = 0, 1, 3
    return b


def dist(r):
    p = np.asarray(r["p"], float)
    return p / p.sum()


def fit_channel(train, N, per_bin=True, ridge=1e-2):
    """Maximum-likelihood conditional logit with soft targets p (small ridge for stability)."""
    P = np.array([dist(r) for r, _ in train])
    Z = np.array([z for _, z in train], float)
    bidx = slot_bins(N) if per_bin else np.zeros(N, int)
    nb = int(bidx.max()) + 1

    def nll(x):
        a = np.concatenate([[0.0], x[:N - 1]])
        U = a[None] + x[N - 1:][bidx][None] * Z
        L = U - logsumexp(U, axis=1, keepdims=True)
        G = np.exp(L) - P
        gb = np.zeros(nb)
        np.add.at(gb, bidx, (G * Z).sum(0))
        return -(P * L).sum() + ridge * (x ** 2).sum(), np.concatenate([G.sum(0)[1:], gb]) + 2 * ridge * x

    x = minimize(nll, np.zeros(N - 1 + nb), jac=True, method="L-BFGS-B").x
    return np.concatenate([[0.0], x[:N - 1]]), x[N - 1:][bidx]


def call_evidence(r, s_, alpha, beta):
    """Per-call closed-form score of every answer present (log-likelihood ratio against 'no candidate correct')."""
    p = dist(r)
    lse0 = logsumexp(alpha)
    slots = collections.defaultdict(list)
    for i, k in enumerate(r["order"]):
        slots[s_["key"][str(k)]].append(i)
    out = {}
    for key, S in slots.items():
        S = np.array(S)
        u = alpha.copy()
        u[S] += beta[S]
        out[key] = float((p[S] * beta[S]).sum() - (logsumexp(u) - lse0))
    return out


def counts(s_):
    return collections.Counter(s_["key"].values())


def prior_linear(s_, gamma):
    return {a: gamma * n for a, n in counts(s_).items()}


def prior_beta_crp(s_, h):
    a0, b0, kappa = np.exp(h)
    c = counts(s_)
    N = sum(c.values())
    return {a: betaln(a0 + n, b0 + N - n) - gammaln(n) - gammaln(kappa + N - n) for a, n in c.items()}


def correct_keys(s_):
    return {s_["key"][k] for k in s_["key"] if s_["cor"][k]}


def fit_prior(train_sets, kind):
    def nll(h):
        tot = 0.0
        for s_ in train_sets:
            pr = prior_linear(s_, h[0]) if kind == "linear" else prior_beta_crp(s_, h)
            keys = list(pr)
            v = np.array([pr[k] for k in keys])
            lp = v - logsumexp(v)
            cor = correct_keys(s_)
            tot -= logsumexp([lp[i] for i, k in enumerate(keys) if k in cor])
        return tot
    return minimize(nll, [1.0] if kind == "linear" else [0.0, 0.0, 0.0], method="Nelder-Mead").x


def accuracy(scores, s_):
    """Expected accuracy of argmax, ties broken uniformly at random."""
    top = max(scores.values())
    tied = [a for a, v in scores.items() if v >= top - 1e-9]
    cor = correct_keys(s_)
    return sum(a in cor for a in tied) / len(tied)


def fused(prior, evidences):
    tot = dict(prior)
    for e in evidences:
        for k, v in e.items():
            tot[k] += v
    return tot


def kl_slot(alpha, beta):
    """Discriminability of one slot: KL(P(.|candidate there correct) || P(.|all wrong)), closed form."""
    pi = np.exp(alpha - logsumexp(alpha))
    Zr = 1 + pi * (np.exp(beta) - 1)
    return pi * np.exp(beta) / Zr * beta - np.log(Zr)


METHODS = [  # (column, description)
    ("majority", "majority vote (0 calls)"),
    ("raw1", "raw selector, 1 call"),
    ("pride1", "divide by slot prior (PriDe), 1 call"),
    ("ours-debias1", "ours without count prior, 1 call"),
    ("ours1", "ours, 1 call"),
    ("vote3", "vote over 3 orderings"),
    ("ours3", "ours, 3 orderings"),
    ("cycsum", "summed probability over N cyclic shifts"),
    ("ours-debias-cyc", "ours without count prior, N cyclic shifts"),
    ("ours-cyc", "ours, N cyclic shifts"),
    ("ours1-1beta", "ablation: one beta for all slots, 1 call"),
    ("ours1-bcrp", "ablation: Beta/CRP count prior, 1 call"),
]
PAIRS = [("ours1", "majority"), ("ours1", "raw1"), ("ours-debias1", "raw1"), ("ours-debias1", "pride1"),
         ("ours3", "vote3"), ("ours3", "majority"), ("ours-cyc", "cycsum"), ("ours-debias-cyc", "cycsum"),
         ("ours-cyc", "majority"), ("ours1", "ours1-1beta")]


def load_calls(path):
    """set_id -> calls with a label distribution; API files keep the last successful row per ordering."""
    last = {}
    for r in load(path):
        if "error" not in r and r.get("p") is not None:
            last[r["set_id"], r["order_name"]] = r
    calls = collections.defaultdict(list)
    for r in last.values():
        calls[r["set_id"]].append(r)
    return calls


def fold_of(S):
    """Problem -> fold (0..4), as e1_analyze.py."""
    pids = sorted({s["pid"] for s in S})
    return {p: i % 5 for i, p in enumerate(np.random.default_rng(1).permutation(pids))}


def pooled_diff(per_set, a, b, B, rng):
    """Mean of per-set (a - b) and its 95% interval, resampling problems; returns (text, n_sets)."""
    ids = [k for k, d in per_set.items() if a in d and b in d]
    pid_list = sorted({per_set[k]["pid"] for k in ids})
    ix = {p: j for j, p in enumerate(pid_list)}
    g = np.array([ix[per_set[k]["pid"]] for k in ids])
    v = np.array([per_set[k][a] - per_set[k][b] for k in ids], float)
    tot, cnt = np.bincount(g, v, len(pid_list)), np.bincount(g, None, len(pid_list)).astype(float)
    bs = []
    for start in range(0, B, 500):
        d = rng.integers(0, len(pid_list), (min(500, B - start), len(pid_list)))
        bs.append(tot[d].sum(1) / cnt[d].sum(1))
    lo, hi = np.percentile(np.concatenate(bs), [2.5, 97.5])
    return f"{100 * v.mean():+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]", len(ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="candidate generator (and selector, for the local runs)")
    ap.add_argument("--mode", choices=("direct", "think"), required=True)
    ap.add_argument("--data", type=Path, default=HERE / "results")
    ap.add_argument("--calls", type=Path, help="selector calls (default <data>/e1_<mode>_<model>.jsonl); "
                    "hosted judges: out/e1_api-<tag>-<mode>_qwen3-8b.jsonl")
    ap.add_argument("--sets", type=Path, help="E1 sets (default <data>/e1_sets_<model>.jsonl)")
    ap.add_argument("--label", help="selector name in the report title (default: model)")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    sets = {s["set_id"]: s for s in load(args.sets or args.data / f"e1_sets_{args.model}.jsonl")}
    calls = load_calls(args.calls or args.data / f"e1_{args.mode}_{args.model}.jsonl")
    per_set, params, rows = listwise_per_set(sets, calls)
    report(args, per_set, params, rows)


def listwise_per_set(sets, calls):
    """Cross-fitted accuracy of every listwise method on every set: (per_set, params, rows)."""
    per_set = {}          # set_id -> {method: accuracy}
    params = collections.defaultdict(list)
    rows = []
    for g, N in sorted({(group(s), s["N"]) for s in sets.values()}):
        S = [s for s in sets.values() if group(s) == g and s["N"] == N
             and sum(r["order_name"].startswith("r") for r in calls[s["set_id"]]) == 3]
        if len({s["pid"] for s in S}) < 5:
            continue
        fold = fold_of(S)
        for f in range(5):
            train = [s for s in S if fold[s["pid"]] != f]
            test = [s for s in S if fold[s["pid"]] == f]
            tc = [(r, [s["cor"][str(k)] for k in r["order"]]) for s in train for r in calls[s["set_id"]]
                  if r["order_name"].startswith("r")]
            slot_prior = np.mean([dist(r) for r, _ in tc], 0)
            alpha, beta = fit_channel(tc, N)
            alpha1, beta1 = fit_channel(tc, N, per_bin=False)
            gamma = fit_prior(train, "linear")[0]
            h = fit_prior(train, "bcrp")
            params[g, N].append((alpha, beta, gamma))
            for s in test:
                R = [r for r in calls[s["set_id"]] if r["order_name"].startswith("r")]
                C = [r for r in calls[s["set_id"]] if r["order_name"].startswith("c")]
                zero = {k: 0.0 for k in counts(s)}
                lin = prior_linear(s, gamma)
                ev = [call_evidence(r, s, alpha, beta) for r in R]
                res = {"majority": accuracy(dict(counts(s)), s),
                       "raw1": np.mean([float(s["cor"][str(r["order"][int(np.argmax(r["p"]))])]) for r in R]),
                       "pride1": np.mean([float(s["cor"][str(r["order"][int(np.argmax(dist(r) / np.maximum(slot_prior, 1e-9)))])])
                                          for r in R]),
                       "ours-debias1": np.mean([accuracy(fused(zero, [e]), s) for e in ev]),
                       "ours1": np.mean([accuracy(fused(lin, [e]), s) for e in ev]),
                       "ours3": accuracy(fused(lin, ev), s),
                       "ours1-1beta": np.mean([accuracy(fused(lin, [call_evidence(r, s, alpha1, beta1)]), s) for r in R]),
                       "ours1-bcrp": np.mean([accuracy(fused(prior_beta_crp(s, h), [e]), s) for e in ev])}
                votes, mass = collections.Counter(), collections.Counter()
                for r in R:
                    k = s["key"][str(r["order"][int(np.argmax(r["p"]))])]
                    votes[k] += 1
                    mass[k] += max(r["p"])
                best = max(votes, key=lambda a: (votes[a], mass[a]))
                res["vote3"] = float(best in correct_keys(s))
                if len(C) == N:
                    tot = collections.Counter()
                    for r in C:
                        for i, k in enumerate(r["order"]):
                            tot[k] += r["p"][i]
                    res["cycsum"] = float(s["cor"][str(max(tot, key=tot.get))])
                    evc = [call_evidence(r, s, alpha, beta) for r in C]
                    res["ours-debias-cyc"] = accuracy(fused(zero, evc), s)
                    res["ours-cyc"] = accuracy(fused(lin, evc), s)
                per_set[s["set_id"]] = dict(res, pid=s["pid"], group=(g, N))
        here = [per_set[s["set_id"]] for s in S]
        rows.append((g, N, len(S), {m: 100 * np.mean([x[m] for x in here]) for m, _ in METHODS if all(m in x for x in here)}))
    return per_set, params, rows


def report(args, per_set, params, rows):
    rng = np.random.default_rng(0)

    def pooled(a, b):
        return pooled_diff(per_set, a, b, args.B, rng)

    lines = [f"# 位置感知的贝叶斯选择（闭式后验）· {args.label or args.model} · {args.mode}", "",
             "由 `bayes_select.py` 生成。方法、推导和变体见脚本说明与 `method_bayes_select.md`。参数（α、β、γ）只在其他 4 折的题目上估计"
             "（5 折交叉拟合，折与 `e1_analyze.py` 相同）。数值为准确率 (%)，单次调用的方法对 3 个随机顺序取平均。", "",
             "## 各组准确率", "",
             "| 组 | N | 集合 | " + " | ".join(m for m, _ in METHODS) + " |",
             "|---|---:|---:|" + "---:|" * len(METHODS)]
    for g, N, n, r in rows:
        lines.append(f"| {g} | {N} | {n} | " + " | ".join(f"{r[m]:.1f}" if m in r else "—" for m, _ in METHODS) + " |")
    lines += ["", "列说明：" + "；".join(f"`{m}` {d}" for m, d in METHODS) + "。", "",
              "## 合并所有组的配对差（按原题整群重采样）", "",
              "| 比较 | 差值 (pp) [95% CI] | 集合数 |", "|---|---|---:|"]
    for a, b in PAIRS:
        if any(a in d and b in d for d in per_set.values()):
            est, n = pooled(a, b)
            lines.append(f"| {a} − {b} | {est} | {n} |")
    lines += ["", "## 拟合参数（5 折平均）", "",
              "β 按槽位区间（首位、第 2 位、中间、末位）估计；π_s 为所有候选都错时的槽位先验 softmax(α)；"
              "D_s 为单次调用中该槽位区分“此处候选正确”与“全错”的 KL 散度（闭式），衡量该槽位的信息量。", "",
              "| 组 | N | γ | β 首位 | β 第 2 位 | β 中间 | β 末位 | π 首位 | D 首位 | D 中间（均值） | D 末位 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (g, N), ps in sorted(params.items()):
        al = np.mean([p[0] for p in ps], 0)
        be = np.mean([p[1] for p in ps], 0)
        ga = np.mean([p[2] for p in ps])
        pi = np.exp(al - logsumexp(al))
        D = kl_slot(al, be)
        lines.append(f"| {g} | {N} | {ga:.2f} | {be[0]:.2f} | {be[1]:.2f} | {be[2]:.2f} | {be[-1]:.2f} | {pi[0]:.2f} | "
                     f"{D[0]:.3f} | {D[2:-1].mean() if N > 3 else D[2]:.3f} | {D[-1]:.3f} |")
    text = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
