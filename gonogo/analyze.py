"""Step 6: go/no-go analysis.

Behaviour : accuracy vs position of the correct candidate (modes A/B/C), honest gap
            estimates (split-half), paired-bootstrap contrasts, within-instance
            permutation test.
Attention : block attention share vs slot, per layer; does it track the accuracy
            curve (across positions) and the model's choice (per prompt, beyond
            position and correctness; conditional logit)?
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
ap.add_argument("--models", default="qwen3-8b,qwen3-4b")
ap.add_argument("--dir", default="results", help="where reports and summaries are written")
ap.add_argument("--data", default="out", help="raw run outputs that are read (sel*, keys, attn*, e1_*)")
ap.add_argument("--B", type=int, default=2000)
args = ap.parse_args()
rng = np.random.default_rng(0)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def middle(N):
    q = max(1, N // 4)
    return list(range(q + 1, N - q + 1))


def front(N):
    return list(range(1, max(1, N // 4) + 1))


def back(N):
    return list(range(N - max(1, N // 4) + 1, N + 1))


def instance_matrix(rows, N):
    """rows of one (model, mode, N) -> matrix [instances, N] of correctness, list of keys."""
    d = collections.defaultdict(dict)
    for r in rows:
        d[(r["pid"], r["draw"])][r["pos"]] = float(bool(r["correct"]))
    keys = [k for k, v in d.items() if len(v) == N]
    M = np.array([[d[k][p] for p in range(1, N + 1)] for k in keys])
    return M, keys


def contrasts(M, N):
    f, m, b = [np.array(x) - 1 for x in (front(N), middle(N), back(N))]
    return dict(first=M[:, 0].mean(), last=M[:, -1].mean(), front=M[:, f].mean(),
                mid=M[:, m].mean(), back=M[:, b].mean())


def boot_ci(M, fn, B):
    n = len(M)
    vals = [fn(M[rng.integers(0, n, n)]) for _ in range(B)]
    return np.percentile(vals, [2.5, 97.5])


def honest_gap(M, keys, N):
    """Choose best/worst position-bin on draw 0, measure on draw 1 and vice versa."""
    nb = min(N, 4)
    bins = np.array_split(np.arange(N), nb)
    draws = np.array([k[1] for k in keys])
    out = []
    par = draws % 2
    if len(set(par)) < 2:  # single draw: split instances in halves instead
        par = np.arange(len(M)) % 2
    for a, b in ((0, 1), (1, 0)):
        Ma, Mb = M[par == a], M[par == b]
        if len(Ma) == 0 or len(Mb) == 0:
            continue
        ma = np.array([Ma[:, bb].mean() for bb in bins])
        mb = np.array([Mb[:, bb].mean() for bb in bins])
        out.append(mb[ma.argmax()] - mb[ma.argmin()])
    return float(np.mean(out)) if out else float("nan")


def perm_test(M, n_perm=2000):
    """Within-instance permutation of position labels; statistic = var of position means."""
    obs = M.mean(0).var()
    cnt = 0
    for _ in range(n_perm):
        P = np.array([row[rng.permutation(len(row))] for row in M])
        cnt += P.mean(0).var() >= obs
    return (cnt + 1) / (n_perm + 1)


# ---------------------------------------------------------------- behaviour
report = []
summary = {}
for model in args.models.split(","):
    files = sorted(f for f in os.listdir(args.data) if f.startswith("sel") and f.endswith(f"_{model}.jsonl"))
    if not files:
        continue
    rows = []
    for f in files:
        rs = load_jsonl(os.path.join(args.data, f))
        if f.startswith("selT"):  # thinking-mode selection: keep separate from direct-answer mode A
            for r in rs:
                r["mode"] = "T"
        rows += rs
    report.append(f"\n## {model}: behaviour\n")
    report.append("| mode | N | inst | acc | chance(A) | first | front¼ | mid½ | back¼ | last | front−mid [95%CI] | back−mid [95%CI] | honest gap | perm p |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mode in ["A", "B", "C", "T"]:
        for N in [4, 8, 16, 32]:
            rr = [r for r in rows if r["mode"] == mode and r["N"] == N]
            if not rr:
                continue
            M, keys = instance_matrix(rr, N)
            c = contrasts(M, N)
            f, m, b = [np.array(x) - 1 for x in (front(N), middle(N), back(N))]
            ci_f = boot_ci(M, lambda X: X[:, f].mean() - X[:, m].mean(), args.B)
            ci_b = boot_ci(M, lambda X: X[:, b].mean() - X[:, m].mean(), args.B)
            hg = honest_gap(M, keys, N)
            pp = perm_test(M, 1000)
            summary[(model, mode, N)] = dict(curve=M.mean(0).tolist(), n=len(M), **c, honest_gap=hg, perm_p=pp,
                                             ci_f=ci_f.tolist(), ci_b=ci_b.tolist())
            report.append(
                f"| {mode} | {N} | {len(M)} | {M.mean():.3f} | {1/N if mode in ('A', 'T') else float('nan'):.3f} | {c['first']:.3f} | {c['front']:.3f} | "
                f"{c['mid']:.3f} | {c['back']:.3f} | {c['last']:.3f} | {100*(c['front']-c['mid']):+.1f} [{100*ci_f[0]:+.1f},{100*ci_f[1]:+.1f}] | "
                f"{100*(c['back']-c['mid']):+.1f} [{100*ci_b[0]:+.1f},{100*ci_b[1]:+.1f}] | {100*hg:.1f} | {pp:.3f} |")
    # positional choice prior in mode A (which slot gets picked, regardless of correctness)
    report.append("\nChoice distribution in mode A (share of prompts choosing each position-quartile; uniform = 0.25), and unparseable rate:\n")
    for N in [4, 8, 16, 32]:
        rr = [r for r in rows if r["mode"] == "A" and r["N"] == N]
        if not rr:
            continue
        ch = np.array([r["chosen"] for r in rr if r["chosen"] is not None])
        bins = np.array_split(np.arange(1, N + 1), 4)
        share = [np.isin(ch, bb).mean() for bb in bins]
        # choices among WRONG picks only (pure positional prior)
        wr = np.array([r["chosen"] for r in rr if r["chosen"] is not None and r["chosen"] != r["pos"]])
        wshare = [np.isin(wr, bb).mean() for bb in bins] if len(wr) else [np.nan] * 4
        none = np.mean([r["chosen"] is None for r in rr])
        report.append(f"- N={N}: all picks {np.round(share,3).tolist()} | wrong picks {np.round(wshare,3).tolist()} | unparseable {none:.3f}")
    # exact choice probability from first-token logprobs (single-digit indices, N <= 8)
    report.append("\nMode A choice probability P(correct index) from first-token logprobs (N ≤ 8; lower-variance than greedy):\n")
    for N in [4, 8]:
        rr = [r for r in rows if r["mode"] == "A" and r["N"] == N and "lp1" in r]
        if not rr:
            continue
        d = collections.defaultdict(dict)
        for r in rr:
            d[(r["pid"], r["draw"])][r["pos"]] = float(np.exp(r["lp1"].get(str(r["pos"]), -50)))
        keys = [k for k, v in d.items() if len(v) == N]
        Mp = np.array([[d[k][p_] for p_ in range(1, N + 1)] for k in keys])
        f, m, b = [np.array(x) - 1 for x in (front(N), middle(N), back(N))]
        ci_f = boot_ci(Mp, lambda X: X[:, f].mean() - X[:, m].mean(), args.B)
        ci_b = boot_ci(Mp, lambda X: X[:, b].mean() - X[:, m].mean(), args.B)
        summary[(model, "Aprob", N)] = dict(curve=Mp.mean(0).tolist(), n=len(Mp))
        report.append(f"- N={N}: per-position P(correct) = {np.round(Mp.mean(0), 3).tolist()}; front−mid {100*(Mp[:, f].mean()-Mp[:, m].mean()):+.1f} "
                      f"[{100*ci_f[0]:+.1f},{100*ci_f[1]:+.1f}], back−mid {100*(Mp[:, b].mean()-Mp[:, m].mean()):+.1f} [{100*ci_b[0]:+.1f},{100*ci_b[1]:+.1f}]")
    # majority effect: does the model side with the largest answer cluster?
    kp = os.path.join(args.data, f"keys_{model}.json")
    if os.path.exists(kp):
        K = json.load(open(kp))
        report.append("\nMajority effect in mode A (wrong candidates often share an answer):\n")
        for N in [4, 8, 16, 32]:
            rr = [r for r in rows if r["mode"] == "A" and r["N"] == N and r["chosen"] is not None]
            if not rr:
                continue
            pick_plur, exp_plur, acc_split = [], [], {True: [], False: []}
            for r in rr:
                keys_ = [K[f"{r['pid']}|{k}"][0] for k in r["ks"]]
                wrong = [k for i, k in enumerate(keys_) if i != r["pos"] - 1]
                top, topc = collections.Counter(wrong).most_common(1)[0]
                pick_plur.append(keys_[r["chosen"] - 1] == top and r["chosen"] != r["pos"])
                exp_plur.append(topc / N)
                acc_split[topc >= 2].append(r["correct"])
            report.append(f"- N={N}: picks the plurality WRONG answer {np.mean(pick_plur):.3f} (uniform-random would be {np.mean(exp_plur):.3f}); "
                          f"acc when wrong answers have a plurality cluster ≥2: {np.mean(acc_split[True]) if acc_split[True] else float('nan'):.3f} "
                          f"(n={len(acc_split[True])}) vs all-distinct: {np.mean(acc_split[False]) if acc_split[False] else float('nan'):.3f} (n={len(acc_split[False])})")
        # prior vs content: is the position effect a correctness-blind slot prior, or does position change how much
        # the candidate's correctness counts?  Conditional logit with free slot intercepts + per-position-bin weight on
        # is-correct + answer-agreement term.  edge−mid < 0 means edge content is used LESS, not more.
        report.append("\nPrior vs content (conditional logit, free slot intercepts; β_bin = weight on is-correct in each position bin; "
                      "LR test of equal β; 95% CI by problem bootstrap):\n")
        prior_content = {}
        for mode in ["A", "T"]:
            for N in [4, 8, 16, 32]:
                rr = [r for r in rows if r["mode"] == mode and r["N"] == N and r.get("chosen") is not None]
                if len(rr) < 40:
                    continue
                nb = min(N, 4)
                Bm = np.eye(nb)[np.minimum(nb - 1, np.arange(N) * nb // N)]
                cor = np.array([[float(s == r["pos"]) for s in range(1, N + 1)] for r in rr])
                agr = np.array([[np.log(c[k]) for k in ks] for ks in
                                ([K[f"{r['pid']}|{k}"][0] for k in r["ks"]] for r in rr) for c in [collections.Counter(ks)]])
                Y = np.array([r["chosen"] - 1 for r in rr])
                G = np.array([r["pid"] for r in rr])

                def fit_pc(idx, inter):
                    nbeta = nb if inter else 1

                    def nll(th):
                        a = np.concatenate([[0.], th[:N - 1]])
                        beta = Bm @ th[N - 1:N - 1 + nbeta] if inter else np.full(N, th[N - 1])
                        U = a[None] + cor[idx] * beta[None] + th[-1] * agr[idx]
                        U = U - U.max(1, keepdims=True)
                        return -(U[np.arange(len(idx)), Y[idx]] - np.log(np.exp(U).sum(1))).sum()
                    res = minimize(nll, np.zeros(N - 1 + nbeta + 1), method="L-BFGS-B")
                    return res.fun, res.x[N - 1:N - 1 + nbeta], res.x[:N - 1]
                allidx = np.arange(len(rr))
                l0, _, _ = fit_pc(allidx, False)
                l1, b1, a1 = fit_pc(allidx, True)
                p_lr = stats.chi2.sf(2 * (l0 - l1), nb - 1)
                upids = np.unique(G)
                byp = {p_: np.where(G == p_)[0] for p_ in upids}
                bs = np.array([fit_pc(np.concatenate([byp[p_] for p_ in rng.choice(upids, len(upids))]), True)[1]
                               for _ in range(200)])
                em = lambda b: b[..., [0, -1]].mean(-1) - b[..., 1:-1].mean(-1)
                lo, hi = np.percentile(bs, [2.5, 97.5], 0)
                elo, ehi = np.percentile(em(bs), [2.5, 97.5])
                prior_content[(mode, N)] = dict(beta=b1.tolist(), lo=lo.tolist(), hi=hi.tolist(), edge_mid=float(em(b1)),
                                                edge_ci=[float(elo), float(ehi)], p_lr=float(p_lr), slot_prior=[0.] + a1.tolist())
                report.append(f"- mode {mode} N={N} (n={len(rr)}): β by bin = " +
                              ", ".join(f"{b:.2f} [{l:.2f},{h:.2f}]" for b, l, h in zip(b1, lo, hi)) +
                              f"; edge−mid {em(b1):+.2f} [{elo:+.2f},{ehi:+.2f}]; LR p={p_lr:.3f}; "
                              f"slot prior logits (slot1=0) first/mid/last = 0.00/{np.mean(a1[:-1]) if N > 2 else 0:.2f}/{a1[-1]:.2f}")
        summary[(model, "prior_content")] = prior_content
    # by source
    report.append("\nMode A accuracy split by source (math500 / comp):\n")
    for N in [4, 8, 16, 32]:
        for src in ["math500", "comp"]:
            rr = [r for r in rows if r["mode"] == "A" and r["N"] == N and ((r["source"] == "math500") == (src == "math500"))]
            if not rr:
                continue
            M, keys = instance_matrix(rr, N)
            c = contrasts(M, N)
            report.append(f"- N={N} {src}: inst={len(M)} acc={M.mean():.3f} front={c['front']:.3f} mid={c['mid']:.3f} back={c['back']:.3f}")

# ---------------------------------------------------------------- attention


def block_shares(rec, layer, which="inst", part="block"):
    """Return attention share over the N candidate blocks (sums to 1) for one prompt."""
    names = rec["names"]
    a = rec[which][layer]
    N = rec["N"]
    idx = {n: i for i, n in enumerate(names)}
    if part == "block":
        v = np.array([a[idx[f"h{k}"]] + a[idx[f"c{k}"]] for k in range(1, N + 1)])
    elif part == "content":
        v = np.array([a[idx[f"c{k}"]] for k in range(1, N + 1)])
    elif part == "header":
        v = np.array([a[idx[f"h{k}"]] for k in range(1, N + 1)])
    return v / max(v.sum(), 1e-12)


def fit_clogit(X, y):
    """Conditional logit. X: [P, N, F] features, y: [P] chosen index. Returns (beta, loglik)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    P, N, F = X.shape
    Xy = X[np.arange(P), y]

    def nll(beta):
        u = X @ beta
        u = u - u.max(1, keepdims=True)
        pr = np.exp(u)
        pr /= pr.sum(1, keepdims=True)
        tot = -np.log(pr[np.arange(P), y] + 1e-300).sum()
        g = -(Xy - np.einsum("pn,pnf->pf", pr, X)).sum(0)
        return tot, g

    res = minimize(nll, np.zeros(F), jac=True, method="L-BFGS-B")
    return res.x, -res.fun


attn_summary = {}
for model in args.models.split(","):
    sp = os.path.join(args.data, f"sel_{model}.jsonl")
    kp = os.path.join(args.data, f"keys_{model}.json")
    KEYS = json.load(open(kp)) if os.path.exists(kp) else None
    recs = []
    for tag in ("attnA", "attnC", "attn"):
        p = os.path.join(args.data, f"{tag}_{model}.pkl")
        if os.path.exists(p):
            recs += pickle.load(open(p, "rb"))
    if not recs or not os.path.exists(sp):
        continue
    sel = {(r["N"], r["pid"], r["draw"], r["pos"], r["mode"]): r
           for f in sorted(os.listdir(args.data))
           if f.startswith("sel") and not f.startswith("selT") and f.endswith(f"_{model}.jsonl")
           for r in load_jsonl(os.path.join(args.data, f))}
    NL = recs[0]["inst"].shape[0]
    report.append(f"\n## {model}: attention (head-mean, query = instruction rows)\n")
    for mode in ["A", "C"]:
        beh_mode = "A" if mode == "A" else "C"
        for N in [4, 8, 16, 32]:
            rr = [r for r in recs if r["mode"] == mode and r["N"] == N]
            if not rr:
                continue
            # slot profile & correct-block share vs position, per layer
            slot = np.zeros((NL, N))
            corr_by_pos = np.zeros((NL, N))
            cnt = np.zeros(N)
            for r in rr:
                for l in range(NL):
                    s = block_shares(r, l)
                    slot[l] += s
                    corr_by_pos[l, r["pos"] - 1] += s[r["pos"] - 1]
                cnt[r["pos"] - 1] += 1
            slot /= len(rr)
            corr_by_pos /= np.maximum(cnt, 1)
            # behaviour curve on the same instances (draw 0 subset)
            inst_keys = {(r["pid"], r["draw"]) for r in rr}
            beh = [sel[(N, k[0], k[1], pos, beh_mode)]["correct"] for k in inst_keys for pos in range(1, N + 1)
                   if (N, k[0], k[1], pos, beh_mode) in sel]
            M = np.array(beh, dtype=float).reshape(len(inst_keys), N) if len(beh) == len(inst_keys) * N else None
            acc_curve = M.mean(0) if M is not None else None
            # full-sample curve (both draws)
            full_curve = np.array(summary.get((model, beh_mode, N), {}).get("curve", [np.nan] * N))
            rho = np.full(NL, np.nan)
            rho_full = np.full(NL, np.nan)
            if acc_curve is not None and N >= 4:
                for l in range(NL):
                    rho[l] = stats.pearsonr(corr_by_pos[l], acc_curve)[0] if acc_curve.std() > 0 else np.nan
                    rho_full[l] = stats.pearsonr(corr_by_pos[l], full_curve)[0] if np.nanstd(full_curve) > 0 else np.nan
            attn_summary[(model, mode, N)] = dict(slot=slot.tolist(), corr_by_pos=corr_by_pos.tolist(),
                                                  acc_curve=None if acc_curve is None else acc_curve.tolist(),
                                                  rho=rho.tolist(), rho_full=rho_full.tolist())
            shallow = slice(0, 2)
            sh = slot[shallow].mean(0)
            f, m, b = [np.array(x) - 1 for x in (front(N), middle(N), back(N))]
            report.append(f"- mode {mode} N={N} (prompts={len(rr)}): layers1-2 slot share front/mid/back per-slot = "
                          f"{sh[f].mean():.3f}/{sh[m].mean():.3f}/{sh[b].mean():.3f} (uniform {1/N:.3f}); "
                          f"r(attn on correct block, acc) across positions: L1={rho[0]:+.2f} L2={rho[1]:+.2f} "
                          f"| full-sample acc curve: L1={rho_full[0]:+.2f} L2={rho_full[1]:+.2f} "
                          f"| best layer {int(np.nanargmax(rho_full))+1} ({np.nanmax(rho_full):+.2f})")
            # does the slot attention profile track the slot *pick* profile (correctness-blind prior)?
            if beh_mode == "A":
                picks = np.zeros(N)
                for r in rr:
                    s_ = sel.get((N, r["pid"], r["draw"], r["pos"], "A"))
                    if s_ is not None and s_["chosen"] is not None:
                        picks[s_["chosen"] - 1] += 1
                picks /= max(picks.sum(), 1)
                rp = np.array([stats.spearmanr(slot[l], picks)[0] for l in range(NL)])
                attn_summary[(model, mode, N)]["rho_pick"] = rp.tolist()
                attn_summary[(model, mode, N)]["pick_share"] = picks.tolist()
                report.append(f"    slot-attention profile vs slot-pick share (Spearman across slots): L1={rp[0]:+.2f} L2={rp[1]:+.2f} "
                              f"mid-layers={np.nanmean(rp[NL//3:2*NL//3]):+.2f} deep={np.nanmean(rp[2*NL//3:]):+.2f} | best L{int(np.nanargmax(rp))+1} ({np.nanmax(rp):+.2f})")

    # per-prompt choice model (mode A): which block does the model pick?
    report.append("\n### Choice model (mode A, conditional logit over candidate blocks)\n")
    report.append("All log-liks are 5-fold cross-validated (folds by problem). Features: position basis (pos = first, last, linear, quadratic in relative position); content (cor = is-correct + answer-agreement with other candidates + log length), is-correct indicator (cor), attention share at layer l (attn, z-scored within prompt). "
                  "ΔLL/prompt = gain from adding attn to pos+cor. top1 = argmax-attention matches choice.\n")
    report.append("expl = share of the positional log-lik gain (pos+cor vs cor) recovered by replacing position dummies with attention (cor+attn vs cor).\n")
    report.append("| N | prompts | LL cor | LL pos | LL pos+cor | best layer | ΔLL/prompt (best) | ΔLL L1 | ΔLL L2 | top1 L1 | top1 L2 | top1 best | chance | expl L1 | expl L2 | expl best |")
    report.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    choice_curves = {}
    for N in [4, 8, 16, 32]:
        rr = [r for r in recs if r["mode"] == "A" and r["N"] == N]
        data = []
        for r in rr:
            s = sel.get((N, r["pid"], r["draw"], r["pos"], "A"))
            if s is None or s["chosen"] is None:
                continue
            data.append((r, s["chosen"] - 1))
        if len(data) < 20:
            continue
        rr_ = np.arange(N) / (N - 1)
        pos_d = np.stack([rr_ == 0, rr_ == 1, rr_, rr_ ** 2], 1).astype(float)  # first, last, linear, quadratic

        y = np.array([c for _, c in data])
        P = len(data)
        POS = np.broadcast_to(pos_d, (P, N, pos_d.shape[1]))
        # content features per candidate: is-correct, answer agreement with the other candidates, length (z within prompt)
        COR = np.zeros((P, N, 3 if KEYS else 1))
        for i, (r, _) in enumerate(data):
            COR[i, r["pos"] - 1, 0] = 1
            if KEYS:
                ks = sel[(N, r["pid"], r["draw"], r["pos"], "A")]["ks"]
                info = [KEYS[f"{r['pid']}|{k}"] for k in ks]
                keys_ = [x[0] for x in info]
                cnt = collections.Counter(keys_)
                COR[i, :, 1] = [(cnt[k] - 1) / (N - 1) for k in keys_]
                ln = np.log([x[2] for x in info])
                COR[i, :, 2] = (ln - ln.mean()) / (ln.std() + 1e-9)
        # 5-fold cross-validated log-lik, folds split by problem id (no leakage across positions)
        pids = sorted({r["pid"] for r, _ in data})
        fold_of = {pid: i % 5 for i, pid in enumerate(np.random.default_rng(1).permutation(pids))}
        fold = np.array([fold_of[r["pid"]] for r, _ in data])

        def cv_ll(X):
            tot = 0.0
            for f in range(5):
                tr, te = fold != f, fold == f
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                beta, _ = fit_clogit(X[tr], y[tr])
                u = X[te] @ beta
                u = u - u.max(1, keepdims=True)
                lp = u - np.log(np.exp(u).sum(1, keepdims=True))
                tot += lp[np.arange(te.sum()), y[te]].sum()
            return tot

        ll_pos = cv_ll(np.ascontiguousarray(POS))
        ll_cor = cv_ll(COR)
        ll_pc = cv_ll(np.concatenate([POS, COR], 2))
        dll, top1, expl = np.zeros(NL), np.zeros(NL), np.zeros(NL)
        for l in range(NL):
            A = np.stack([block_shares(r, l) for r, _ in data])
            Az = ((A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9))[:, :, None]
            ll = cv_ll(np.concatenate([POS, COR, Az], 2))
            ll_ca = cv_ll(np.concatenate([COR, Az], 2))
            dll[l] = (ll - ll_pc) / P
            expl[l] = (ll_ca - ll_cor) / (ll_pc - ll_cor) if (ll_pc - ll_cor) / P > 0.005 else np.nan
            top1[l] = np.mean(A.argmax(1) == y)
        choice_curves[N] = dict(dll=dll.tolist(), top1=top1.tolist(), expl=expl.tolist(),
                                ll_pos=ll_pos / P, ll_cor=ll_cor / P, ll_pc=ll_pc / P)
        bl = int(np.argmax(dll))
        report.append(f"| {N} | {P} | {ll_cor/P:.3f} | {ll_pos/P:.3f} | {ll_pc/P:.3f} | {bl+1} | {dll[bl]:.3f} | {dll[0]:.3f} | {dll[1]:.3f} | "
                      f"{top1[0]:.3f} | {top1[1]:.3f} | {top1.max():.3f} (L{int(top1.argmax())+1}) | {1/N:.3f} | "
                      f"{expl[0]:.2f} | {expl[1]:.2f} | " + (f"{np.nanmax(expl):.2f} (L{int(np.nanargmax(expl))+1})" if np.isfinite(expl).any() else "n/a (no CV position effect)") + " |")
    attn_summary[(model, "choice")] = choice_curves

os.makedirs(args.dir, exist_ok=True)
open(os.path.join(args.dir, "report_tables.md"), "w").write("\n".join(report))
pickle.dump(dict(summary=summary, attn=attn_summary), open(os.path.join(args.dir, "summary.pkl"), "wb"))
print("\n".join(report))
