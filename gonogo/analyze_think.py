"""Thinking-mode analysis: position sweep behaviour + attention from reasoning tokens (attn_think.py).

Go criterion for the thinking version of the idea: the position effect on *content use* (finished runs)
is predicted by shallow-layer attention from early reasoning tokens to the candidate blocks.
"""
import argparse
import collections
import json
import os
import pickle

import numpy as np
from scipy import stats
from scipy.optimize import minimize

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results")
ap.add_argument("--model", default="qwen3-8b")
ap.add_argument("--sel", default="selT2")
ap.add_argument("--attn", default="attnT")
ap.add_argument("--B", type=int, default=2000)
args = ap.parse_args()
rng = np.random.default_rng(0)
Ns = [4, 8, 16]


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def fmb(N):
    q = max(1, N // 4)
    F = list(range(1, q + 1))
    Bk = list(range(N - q + 1, N + 1))
    return F, [p for p in range(1, N + 1) if p not in F and p not in Bk], Bk


def boot_groups(rows, f, B):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["pid"]].append(r)
    ks = list(g)
    vals = [f([x for i in rng.integers(0, len(ks), len(ks)) for x in g[ks[i]]]) for _ in range(B)]
    return np.nanpercentile(vals, [2.5, 97.5])


def mean_at(rows, key, poss, cond=lambda r: True):
    v = [key(r) for r in rows if r["pos"] in poss and cond(r)]
    return float(np.mean(v)) if v else np.nan


acc = lambda r: float(bool(r["correct"]))          # noqa: E731
trunc = lambda r: float(r["chosen"] is None)        # noqa: E731
done = lambda r: r["chosen"] is not None            # noqa: E731

rows = load_jsonl(os.path.join(args.dir, f"{args.sel}_{args.model}.jsonl"))
KEYS = json.load(open(os.path.join(args.dir, f"keys_{args.model}.json")))
rep = [f"# {args.model}: thinking-mode position sweep ({args.sel})\n"]
summ = {}

# ------------------------------------------------------------------ behaviour
rep.append("## Behaviour\n")
rep.append("| N | problems | finished | acc (all) | acc (finished) | front−mid all [95%CI] | back−mid all [95%CI] | "
           "front−mid finished [95%CI] | back−mid finished [95%CI] | trunc front−mid | lift edge−mid [95%CI] | perm p (all) |")
rep.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for N in Ns:
    rs = [r for r in rows if r["N"] == N]
    if not rs:
        continue
    F, M, Bk = fmb(N)
    out = {}
    for nm, key, cond in [("all", acc, lambda r: True), ("fin", acc, done), ("trunc", trunc, lambda r: True)]:
        fm = lambda S, key=key, cond=cond: mean_at(S, key, F, cond) - mean_at(S, key, M, cond)   # noqa: E731
        bm = lambda S, key=key, cond=cond: mean_at(S, key, Bk, cond) - mean_at(S, key, M, cond)  # noqa: E731
        out[nm] = dict(curve=[mean_at(rs, key, [p], cond) for p in range(1, N + 1)], fm=fm(rs), bm=bm(rs),
                       fm_ci=boot_groups(rs, fm, args.B).tolist(), bm_ci=boot_groups(rs, bm, args.B).tolist())
    fin = [r for r in rs if done(r)]

    def lift(S):
        hit = np.array([np.mean([r["chosen"] == s for r in S if r["pos"] == s]) if any(r["pos"] == s for r in S) else np.nan
                        for s in range(1, N + 1)])
        fa = np.array([np.mean([r["chosen"] == s for r in S if r["pos"] != s]) if any(r["pos"] != s for r in S) else np.nan
                       for s in range(1, N + 1)])
        return hit - fa, fa
    edge = [p - 1 for p in F + Bk]
    mid = [p - 1 for p in M]
    em = lambda S: np.nanmean(lift(S)[0][edge]) - np.nanmean(lift(S)[0][mid])  # noqa: E731
    L, FA = lift(fin)
    em_ci = boot_groups(fin, em, min(args.B, 1000))
    # within-problem permutation test on acc(all): permute position labels inside each (pid, draw)
    d = collections.defaultdict(dict)
    for r in rs:
        d[(r["pid"], r["draw"])][r["pos"]] = acc(r)
    Mx = np.array([[v[p] for p in range(1, N + 1)] for v in d.values() if len(v) == N])
    obs = Mx.mean(0).var()
    perm_p = (1 + sum(np.array([row[rng.permutation(N)] for row in Mx]).mean(0).var() >= obs for _ in range(2000))) / 2001
    summ[N] = dict(out, lift=L.tolist(), fa=FA.tolist(), edge_mid=float(em(fin)), edge_mid_ci=em_ci.tolist(),
                   n_prob=len({r["pid"] for r in rs}), fin_rate=len(fin) / len(rs), perm_p=float(perm_p))
    o = out
    rep.append(f"| {N} | {summ[N]['n_prob']} | {summ[N]['fin_rate']:.2f} | {np.mean([acc(r) for r in rs]):.3f} | "
               f"{np.mean([acc(r) for r in fin]):.3f} | "
               f"{100*o['all']['fm']:+.1f} [{100*o['all']['fm_ci'][0]:+.1f},{100*o['all']['fm_ci'][1]:+.1f}] | "
               f"{100*o['all']['bm']:+.1f} [{100*o['all']['bm_ci'][0]:+.1f},{100*o['all']['bm_ci'][1]:+.1f}] | "
               f"{100*o['fin']['fm']:+.1f} [{100*o['fin']['fm_ci'][0]:+.1f},{100*o['fin']['fm_ci'][1]:+.1f}] | "
               f"{100*o['fin']['bm']:+.1f} [{100*o['fin']['bm_ci'][0]:+.1f},{100*o['fin']['bm_ci'][1]:+.1f}] | "
               f"{100*o['trunc']['fm']:+.1f} [{100*o['trunc']['fm_ci'][0]:+.1f},{100*o['trunc']['fm_ci'][1]:+.1f}] | "
               f"{100*summ[N]['edge_mid']:+.1f} [{100*em_ci[0]:+.1f},{100*em_ci[1]:+.1f}] | {perm_p:.3f} |")
rep.append("\nlift = P(pick slot k | correct at k) − P(pick slot k | k wrong), finished runs only; edge = first+last quarter.\n")
for N in summ:
    rep.append(f"- N={N}: acc|finished by position {np.round(summ[N]['fin']['curve'], 2).tolist()}")
    rep.append(f"  truncation by position {np.round(summ[N]['trunc']['curve'], 2).tolist()}")
    rep.append(f"  lift by slot {np.round(summ[N]['lift'], 2).tolist()} | P(pick k | k wrong) {np.round(summ[N]['fa'], 3).tolist()}")

# ------------------------------------------------------------------ attention from reasoning tokens
ap_ = os.path.join(args.dir, f"{args.attn}_{args.model}.pkl")
attn_summ = {}
if os.path.exists(ap_):
    recs = pickle.load(open(ap_, "rb"))
    NL = recs[0]["bins"].shape[0]
    groups = [("L1-2", slice(0, 2)), (f"L{NL//3+1}-{2*NL//3}", slice(NL // 3, 2 * NL // 3)), (f"L{2*NL//3+1}-{NL}", slice(2 * NL // 3, NL))]

    def shares(rec, which):
        """[L, N] attention share over candidate blocks (header+content), normalised over blocks."""
        names = rec["names"]
        idx = {n: i for i, n in enumerate(names)}
        N = rec["N"]
        if which == "early":
            a = rec["bins"][:, 0]
        elif which == "all":
            a = np.nanmean(rec["bins"], 1)
        else:
            a = rec["dec"]
        v = np.stack([a[:, idx[f"h{k}"]] + a[:, idx[f"c{k}"]] for k in range(1, N + 1)], 1)
        return v / np.maximum(v.sum(1, keepdims=True), 1e-12)

    def fit_clogit(X, y):
        P, N, F = X.shape
        Xy = X[np.arange(P), y]

        def nll(beta):
            u = X @ beta
            u = u - u.max(1, keepdims=True)
            pr = np.exp(u)
            pr /= pr.sum(1, keepdims=True)
            return -np.log(pr[np.arange(P), y] + 1e-300).sum(), -(Xy - np.einsum("pn,pnf->pf", pr, X)).sum(0)
        return minimize(nll, np.zeros(F), jac=True, method="L-BFGS-B").x

    rep.append(f"\n## Attention from reasoning tokens ({len(recs)} sequences; head mean; shares normalised over candidate blocks)\n")
    rep.append("early = first quarter of the reasoning trace; all = whole trace; dec = token predicting the chosen index.\n")
    for N in Ns:
        rr = [r for r in recs if r["N"] == N]
        if not rr:
            continue
        res = {}
        line = [f"- N={N} ({len(rr)} seqs, {len({r['pid'] for r in rr})} problems):"]
        for which in ["early", "all", "dec"]:
            use = [r for r in rr if which != "dec" or r["dec"] is not None]
            if not use:
                continue
            S = np.stack([shares(r, which) for r in use])           # P, L, N
            slot = np.nanmean(S, 0)                                   # L, N
            # attention on the correct block by position vs acc|finished by position (same sequences)
            cb = np.full((NL, N), np.nan)
            accp = np.full(N, np.nan)
            for p in range(1, N + 1):
                ii = [i for i, r in enumerate(use) if r["pos"] == p]
                if ii:
                    cb[:, p - 1] = np.nanmean(S[ii, :, p - 1], 0)
                fin_p = [bool(use[i]["correct"]) for i in ii if use[i]["chosen"] is not None]
                accp[p - 1] = np.mean(fin_p) if fin_p else np.nan
            ok = np.isfinite(accp)
            rho = np.array([stats.pearsonr(cb[l, ok], accp[ok])[0] if ok.sum() > 2 and np.nanstd(accp[ok]) > 0 else np.nan
                            for l in range(NL)])
            res[which] = dict(slot=slot.tolist(), rho=rho.tolist(), acc_by_pos=accp.tolist(), corr_block=cb.tolist())
            F_, M_, B_ = fmb(N)
            prof = "; ".join(f"{g} F/M/B×N = {N*slot[sl].mean(0)[np.array(F_)-1].mean():.2f}/{N*slot[sl].mean(0)[np.array(M_)-1].mean():.2f}/"
                             f"{N*slot[sl].mean(0)[np.array(B_)-1].mean():.2f}" for g, sl in groups)
            line.append(f"    [{which}] {prof} | r(correct-block attn, acc|fin) across positions: L1 {rho[0]:+.2f} L2 {rho[1]:+.2f}, "
                        f"best L{int(np.nanargmax(rho))+1} {np.nanmax(rho):+.2f}" if np.isfinite(rho).any() else f"    [{which}] {prof}")
        rep += line

        # choice model over slots, finished runs: position basis + content (+ attention at layer l)
        data = [r for r in rr if r["chosen"] is not None and r["dec"] is not None]
        if len(data) < 20:
            attn_summ[N] = res
            continue
        y = np.array([r["chosen"] - 1 for r in data])
        P = len(data)
        rr_ = np.arange(N) / (N - 1)
        POS = np.broadcast_to(np.stack([rr_ == 0, rr_ == 1, rr_, rr_ ** 2], 1).astype(float), (P, N, 4))
        COR = np.zeros((P, N, 3))
        sels = {(r["N"], r["pid"], r["draw"], r["pos"]): r for r in rows}
        for i, r in enumerate(data):
            COR[i, r["pos"] - 1, 0] = 1
            ks = sels[(r["N"], r["pid"], r["draw"], r["pos"])]["ks"]
            info = [KEYS[f"{r['pid']}|{k}"] for k in ks]
            cnt = collections.Counter(x[0] for x in info)
            COR[i, :, 1] = [(cnt[x[0]] - 1) / (N - 1) for x in info]
            ln = np.log([max(x[2], 1) for x in info])
            COR[i, :, 2] = (ln - ln.mean()) / (ln.std() + 1e-9)
        pids = sorted({r["pid"] for r in data})
        fold_of = {pid: i % 5 for i, pid in enumerate(np.random.default_rng(1).permutation(pids))}
        fold = np.array([fold_of[r["pid"]] for r in data])

        def cv_ll(X):
            X = np.ascontiguousarray(X)
            tot = 0.0
            for f in range(5):
                tr, te = fold != f, fold == f
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                beta = fit_clogit(X[tr], y[tr])
                u = X[te] @ beta
                u = u - u.max(1, keepdims=True)
                tot += (u - np.log(np.exp(u).sum(1, keepdims=True)))[np.arange(te.sum()), y[te]].sum()
            return tot / P
        ll_cor, ll_pc = cv_ll(COR), cv_ll(np.concatenate([POS, COR], 2))
        rep.append(f"    choice model (finished, n={P}): CV LL/prompt cor {ll_cor:.3f}, pos+cor {ll_pc:.3f} (position gain {ll_pc-ll_cor:+.3f})")
        for which in ["early", "dec"]:
            Sall = np.stack([shares(r, which) for r in data])          # P, L, N
            dll, expl, top1 = np.zeros(NL), np.full(NL, np.nan), np.zeros(NL)
            for l in range(NL):
                A = Sall[:, l]
                Az = ((A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9))[:, :, None]
                dll[l] = cv_ll(np.concatenate([POS, COR, Az], 2)) - ll_pc
                if ll_pc - ll_cor > 0.005:
                    expl[l] = (cv_ll(np.concatenate([COR, Az], 2)) - ll_cor) / (ll_pc - ll_cor)
                top1[l] = np.mean(A.argmax(1) == y)
            res[f"choice_{which}"] = dict(dll=dll.tolist(), expl=expl.tolist(), top1=top1.tolist(), ll_cor=ll_cor, ll_pc=ll_pc)
            bl = int(np.argmax(dll))
            rep.append(f"    [{which}] ΔLL from attention: L1 {dll[0]:+.3f} L2 {dll[1]:+.3f} best L{bl+1} {dll[bl]:+.3f} | "
                       f"expl L1 {expl[0]:.2f} L2 {expl[1]:.2f} best " +
                       (f"L{int(np.nanargmax(expl))+1} {np.nanmax(expl):.2f}" if np.isfinite(expl).any() else "n/a") +
                       f" | top1 L1 {top1[0]:.2f} L2 {top1[1]:.2f} best {top1.max():.2f} (chance {1/N:.2f})")
        attn_summ[N] = res
else:
    rep.append(f"\n(no {args.attn}_{args.model}.pkl yet)")

open(os.path.join(args.dir, f"report_think_{args.model}.md"), "w").write("\n".join(rep))
pickle.dump(dict(behaviour=summ, attn=attn_summ), open(os.path.join(args.dir, f"think_summary_{args.model}.pkl"), "wb"))
print("\n".join(rep))
