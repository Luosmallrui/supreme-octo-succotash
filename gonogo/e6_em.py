"""E6 without correctness labels: EM over the latent correct answer for the order-balanced duel posterior.

Model (as e6_checks.py, item 5). In a set with K distinct answers, counts n_a and order-balanced duel statistics
d_ab = (logit p_ab - logit p_ba) / 2 on all K(K-1)/2 pairs,
    prior     P(a* = a | n) = softmax_a(gamma n_a),
    duels     d_ab = s_a - s_b + e_ab,  s_a = delta z_a + u_a,  u_a ~ N(0, tau2),  e_ab ~ N(0, s2e),  z_a = 1[a = a*].
On the complete graph D^T D = K I on vectors orthogonal to 1, so the duels enter only through the Borda vector
b_a = sum_b d_ab (sum_a b_a = 0) and the cycle energy C = ||d||^2 - ||b||^2 / K, and with V = s2e + K tau2,
    log P(d | a* = a) = -(||b||^2/K - 2 delta b_a + delta^2 (K-1)) / (2V) - C / (2 s2e)
                        - (K-1)/2 log V - (K(K-1)/2 - K + 1)/2 log s2e + const,
    P(a* = a | n, d)  = softmax_a(gamma n_a + delta b_a / V).                       (closed-form posterior)
EM treats a* as missing: the E-step is the posterior above; the M-step maximises the expected complete-data
log-likelihood over gamma (1-D, concave) and over (delta, tau2, s2e) (3-D, numerical). It uses no correctness
labels, so it is fitted on the evaluation sets themselves (transductive), with no folds and no held-out problems.
Variants: parameters per group (source x N), or (delta, tau2, s2e) shared by all groups with gamma per group.
Several starts; the run with the highest marginal likelihood is kept.

Compared with the supervised, 5-fold cross-fitted rules of e6_analyze.py / e6_checks.py: majority vote, the
listwise posterior over three orderings (list-ours3), the pre-registered duel posterior (tau2 = 0) and the
answer-level (random-effect) duel posterior. Pooled paired differences resample problems (5000 draws).

  python e6_em.py --data results   (writes out/e6_duels/report_e6_em.md)
"""
import argparse
import collections
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax

import bayes_select as B
from e6_analyze import duel_table, load_duels
from e6_checks import fit_duel

HERE = Path(__file__).resolve().parent
JUDGES = (("qwen3-8b", "qwen3-8b", "out/e6_duels/e6_qwen3-8b.jsonl", None),
          ("qwen3-4b", "qwen3-4b", "out/e6_duels/e6_qwen3-4b.jsonl", None),
          ("deepseek-flash", "qwen3-8b", "out/e6_duels/e6_api-deepseek-flash-prefill_qwen3-8b.jsonl",
           "out/e1_api-deepseek-flash-prefill-direct_qwen3-8b.jsonl"))
STARTS = [(0.5, 1.0, 1.0), (2.0, 1.0, 1.0), (2.0, 5.0, 1.0), (5.0, 5.0, 2.0)]   # (delta, tau2, s2e)


def stats(s, dd):
    keys = list(B.counts(s))
    K = len(keys)
    b = np.array([sum(v for (x, _), v in dd.items() if x == a) for a in keys])
    sq = 0.5 * sum(v * v for v in dd.values())
    return dict(keys=keys, K=K, n=np.array([B.counts(s)[a] for a in keys], float), b=b, bb=float(b @ b),
                C=max(sq - float(b @ b) / K, 0.0), M=K * (K - 1) // 2)


def duel_ll(st, delta, tau2, s2e):
    """log P(d | a* = a) for every answer a of one set (up to a constant)."""
    K = st["K"]
    if K < 2:
        return np.zeros(K)
    V = s2e + K * tau2
    return (-(st["bb"] / K - 2 * delta * st["b"] + delta ** 2 * (K - 1)) / (2 * V) - st["C"] / (2 * s2e)
            - (K - 1) / 2 * np.log(V) - (st["M"] - K + 1) / 2 * np.log(s2e))


def em(sets_by_group, shared, iters=400, tol=1e-8):
    """EM over the latent correct answer. Returns gamma per group, duel parameters per group, marginal log-lik."""
    groups = list(sets_by_group)
    best = None
    for d0, t0, e0 in STARTS:
        gam = {g: 0.5 for g in groups}
        th = {g: np.log([d0, t0, e0]) for g in groups} if not shared else {"*": np.log([d0, t0, e0])}
        last = -np.inf
        for _ in range(iters):
            W, ll = {}, 0.0
            for g in groups:
                p = th["*" if shared else g]
                delta, tau2, s2e = np.exp(p)
                for i, st in enumerate(sets_by_group[g]):
                    lp = gam[g] * st["n"] + duel_ll(st, delta, tau2, s2e)
                    ll += logsumexp(lp) - logsumexp(gam[g] * st["n"])
                    W[g, i] = softmax(lp)
            for g in groups:                                     # M-step: gamma (1-D, concave)
                sts = sets_by_group[g]
                f = lambda x: -sum(W[g, i] @ (x * st["n"]) - logsumexp(x * st["n"]) for i, st in enumerate(sts))
                gam[g] = minimize_scalar(f, bounds=(-3, 12), method="bounded").x
            for key in th:                                       # M-step: duel parameters (delta, tau2, s2e)
                members = [(g, i, st) for g in groups if shared or g == key for i, st in enumerate(sets_by_group[g])]

                def negq(x):
                    delta, tau2, s2e = np.exp(x)
                    return -sum(W[g, i] @ duel_ll(st, delta, tau2, s2e) for g, i, st in members if st["K"] > 1)
                th[key] = minimize(negq, th[key], method="L-BFGS-B", bounds=[(-6, 4), (-8, 5), (-8, 5)]).x
            if ll - last < tol:
                break
            last = ll
        if best is None or ll > best[2]:
            best = (dict(gam), {k: np.exp(v) for k, v in th.items()}, ll)
    return best


def evaluate(name, model, duels, calls_path, data, n_boot):
    sets = {s["set_id"]: s for s in B.load(data / f"e1_sets_{model}.jsonl")}
    calls = B.load_calls(calls_path)
    _, d, _, _ = duel_table(load_duels(duels))
    groups = sorted({(B.group(s), s["N"]) for s in sets.values()})
    S = {gN: [s for s in sets.values() if (B.group(s), s["N"]) == gN and s["set_id"] in d
              and sum(r["order_name"].startswith("r") for r in calls[s["set_id"]]) == 3] for gN in groups}
    S = {gN: v for gN, v in S.items() if len({s["pid"] for s in v}) >= 5}
    per_set, sup_params = {}, {}
    for gN, ss in S.items():                                     # supervised, cross-fitted (as e6_analyze.py)
        fold = B.fold_of(ss)
        for f in range(5):
            train = [s for s in ss if fold[s["pid"]] != f]
            test = [s for s in ss if fold[s["pid"]] == f]
            delta, sigma2, tau2, s2e = fit_duel(train, d)
            gamma = B.fit_prior(train, "linear")[0]
            tc = [(r, [s["cor"][str(k)] for k in r["order"]]) for s in train for r in calls[s["set_id"]]
                  if r["order_name"].startswith("r")]
            alpha, beta = B.fit_channel(tc, gN[1])
            for s in test:
                st = stats(s, d[s["set_id"]])
                lin = B.prior_linear(s, gamma)
                ev = [B.call_evidence(r, s, alpha, beta) for r in calls[s["set_id"]] if r["order_name"].startswith("r")]
                post = lambda w: B.accuracy(dict(zip(st["keys"], gamma * st["n"] + w * st["b"])), s)
                per_set[s["set_id"]] = dict(pid=s["pid"], group=gN, majority=B.accuracy(dict(B.counts(s)), s),
                                            list_ours3=B.accuracy(B.fused(lin, ev), s),
                                            duel_prereg=post(delta / sigma2),
                                            duel_re=post(delta / (s2e + st["K"] * tau2)))
        dl, s2, t2, e2 = fit_duel(ss, d)                         # all-data supervised fit, for parameter recovery
        sup_params[gN] = (B.fit_prior(ss, "linear")[0], dl, t2, e2)
    st_by_group = {gN: [stats(s, d[s["set_id"]]) for s in ss] for gN, ss in S.items()}
    fits = {}
    for label, shared in (("em_group", False), ("em_shared", True)):
        gam, th, ll = em(st_by_group, shared)
        fits[label] = (gam, th, ll)
        for gN, ss in S.items():
            delta, tau2, s2e = th["*" if shared else gN]
            for s, st in zip(ss, st_by_group[gN]):
                score = gam[gN] * st["n"] + delta * st["b"] / (s2e + st["K"] * tau2)
                per_set[s["set_id"]][label] = B.accuracy(dict(zip(st["keys"], score)), s)
    return per_set, sup_params, fits


def report(data, n_boot):
    methods = ["majority", "list_ours3", "duel_prereg", "duel_re", "em_group", "em_shared"]
    names = {"majority": "多数投票", "list_ours3": "列表式闭式后验（3 顺序）", "duel_prereg": "对决后验（事先登记，τ²=0）",
             "duel_re": "对决后验 + 答案层面随机效应（有监督）", "em_group": "EM，参数按组（无标签）",
             "em_shared": "EM，对决参数共享、γ 按组（无标签）"}
    lines = ["# E6 无标签参数估计：对潜在正确答案做 EM", "",
             "由 `e6_em.py` 生成。模型与 `e6_checks.py` 第 5 项相同（答案层面随机效应）。EM 不使用任何正确性标签，直接在评估集合上拟合"
             "（转导式，无交叉拟合、无留出题）；有监督各行与 `e6_analyze.py` 相同，按题 5 折交叉拟合。数值为准确率 (%)；"
             "差值按原题整群重采样 5000 次。", ""]
    for name, model, duels, calls in JUDGES:
        calls_path = HERE / calls if calls else data / f"e1_direct_{model}.jsonl"
        ps, sup, fits = evaluate(name, model, HERE / duels, calls_path, data, n_boot)
        rng = np.random.default_rng(0)
        lines += [f"## {name}（评 {model} 的候选）", "", "| 方法 | 准确率 % |", "|---|---:|"]
        for m in methods:
            lines.append(f"| {names[m]} | {100 * np.mean([v[m] for v in ps.values()]):.1f} |")
        lines += ["", "| 比较 | 差值 (pp) [95% CI] | 集合数 |", "|---|---|---:|"]
        for a, b in [("duel_re", "majority"), ("duel_re", "list_ours3"), ("em_group", "majority"),
                     ("em_group", "list_ours3"), ("em_group", "duel_re"), ("em_shared", "majority"),
                     ("em_shared", "list_ours3"), ("em_shared", "duel_re"), ("em_shared", "duel_prereg")]:
            est, n = B.pooled_diff(ps, a, b, n_boot, rng)
            lines.append(f"| {names[a]} − {names[b]} | {est} | {n} |")
        lines += ["", "参数（有监督：全部集合、用标签；EM：无标签）。δ/V 为平均答案数 K 下对 Borda 和的权重。", "",
                  "| 组 | N | γ 有监督 | γ EM(按组) | δ 有监督 | δ EM(按组) | τ² 有监督 | τ² EM(按组) | s2e 有监督 | s2e EM(按组) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        gam_g, th_g, ll_g = fits["em_group"]
        for gN, (g_s, d_s, t_s, e_s) in sorted(sup.items()):
            d_e, t_e, e_e = th_g[gN]
            lines.append(f"| {gN[0]} | {gN[1]} | {g_s:.2f} | {gam_g[gN]:.2f} | {d_s:.2f} | {d_e:.2f} | {t_s:.2f} | {t_e:.2f} | "
                         f"{e_s:.2f} | {e_e:.2f} |")
        gam_s, th_s, ll_s = fits["em_shared"]
        d_e, t_e, e_e = th_s["*"]
        lines += ["", f"共享参数 EM：δ = {d_e:.2f}，τ² = {t_e:.2f}，s2e = {e_e:.2f}；γ 按组 = "
                  + "，".join(f"{g[0]}-{g[1]} {v:.2f}" for g, v in sorted(gam_s.items()))
                  + f"。边际对数似然：按组 {ll_g:.1f}，共享 {ll_s:.1f}。", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=HERE / "results", help="E1 sets and direct calls")
    ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--output", type=Path, default=HERE / "out/e6_duels/report_e6_em.md")
    args = ap.parse_args()
    args.output.write_text(report(args.data, args.B), encoding="utf-8")
    print(f"wrote {args.output}")
