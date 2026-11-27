"""bayes_select_active.py -- offline simulation of adaptive 'primacy-as-probe' ordering on E1 direct data (all N cyclic shifts were run).

At each step the posterior over answers (closed form: count prior + per-call evidence) picks which candidate to put
first; the call with that candidate first is the cyclic shift starting with it. Compared against random cyclic
shifts and random orderings with the same closed-form fusion, and majority vote. Cross-fitted parameters.
"""
import collections, itertools, sys
import numpy as np
from scipy.special import logsumexp

import bayes_select as B

# usage: python bayes_select_active.py qwen3-8b > out/method_bayes_select/report_active_qwen3-8b.txt
MODEL = sys.argv[1]
sets = {s["set_id"]: s for s in B.load(B.HERE / "results" / f"e1_sets_{MODEL}.jsonl")}
calls = collections.defaultdict(list)
for r in B.load(B.HERE / "results" / f"e1_direct_{MODEL}.jsonl"):
    calls[r["set_id"]].append(r)

BUDGETS = (1, 2, 3)
rng = np.random.default_rng(0)
per = {}
for g, N in sorted({(B.group(s), s["N"]) for s in sets.values()}):
    S = [s for s in sets.values() if B.group(s) == g and s["N"] == N
         and sum(r["order_name"].startswith("c") for r in calls[s["set_id"]]) == N]
    if not S:
        continue
    pids = sorted({s["pid"] for s in S})
    fold = {p: i % 5 for i, p in enumerate(np.random.default_rng(1).permutation(pids))}
    for f in range(5):
        train = [s for s in S if fold[s["pid"]] != f]
        test = [s for s in S if fold[s["pid"]] == f]
        # fit on random-order calls of training problems (all problems with E1 data in this group)
        trainall = [s for s in sets.values() if B.group(s) == g and s["N"] == N and fold.get(s["pid"], -1) != f
                    and s["pid"] in fold]
        tc = [(r, [s["cor"][str(k)] for k in r["order"]]) for s in trainall for r in calls[s["set_id"]]
              if r["order_name"].startswith("r")]
        alpha, beta = B.fit_channel(tc, N)
        gamma = B.fit_prior(train, "linear")[0]
        for s in test:
            first = {r["order"][0]: r for r in calls[s["set_id"]] if r["order_name"].startswith("c")}
            R = [r for r in calls[s["set_id"]] if r["order_name"].startswith("r")]
            prior = B.prior_linear(s, gamma)
            ev_first = {k: B.call_evidence(r, s, alpha, beta) for k, r in first.items()}
            res = {"majority": B.accuracy(dict(B.counts(s)), s)}
            for M in BUDGETS:
                # random orderings (r0..r2): average over subsets of size M
                subs = list(itertools.combinations(range(len(R)), M))
                ev_r = [B.call_evidence(r, s, alpha, beta) for r in R]
                res[f"rand-order M={M}"] = np.mean([B.accuracy(B.fused(prior, [ev_r[i] for i in sub]), s) for sub in subs])
                # random cyclic shifts
                ks = list(first)
                res[f"rand-shift M={M}"] = np.mean([B.accuracy(B.fused(prior, [ev_first[k] for k in rng.choice(ks, M, replace=False)]), s)
                                                    for _ in range(30)])
                # adaptive: probe (put first) a candidate of the answer whose posterior is closest to 1/2
                for rule in ("uncertain", "leader", "runnerup"):
                    used, evs = set(), []
                    for _ in range(M):
                        sc = B.fused(prior, evs)
                        keys = list(sc)
                        post = np.exp(np.array([sc[k] for k in keys]) - logsumexp([sc[k] for k in keys]))
                        order = np.argsort(-post)
                        if rule == "uncertain":
                            rank = sorted(range(len(keys)), key=lambda i: abs(post[i] - 0.5))
                        elif rule == "leader":
                            rank = list(order)
                        else:
                            rank = list(order[1:]) + list(order[:1])
                        pick = None
                        for i in rank:
                            cands = [int(c) for c in s["key"] if s["key"][c] == keys[i] and int(c) not in used]
                            if cands:
                                pick = cands[0]
                                break
                        if pick is None:
                            break
                        used.add(pick)
                        evs.append(ev_first[pick])
                    res[f"active-{rule} M={M}"] = B.accuracy(B.fused(prior, evs), s)
            per[s["set_id"]] = dict(res, pid=s["pid"])

methods = ["majority"] + [f"{m} M={M}" for M in BUDGETS for m in ("rand-order", "rand-shift", "active-uncertain", "active-leader", "active-runnerup")]
print(MODEL, "sets with all cyclic shifts:", len(per))
for m in methods:
    print(f"  {m:28s} {100*np.mean([d[m] for d in per.values()]):.1f}")
rng2 = np.random.default_rng(0)
def pooled(a, b):
    ids = list(per)
    pl = sorted({per[i]["pid"] for i in ids}); ix = {p: j for j, p in enumerate(pl)}
    g = np.array([ix[per[i]["pid"]] for i in ids]); v = np.array([per[i][a] - per[i][b] for i in ids])
    tot, cnt = np.bincount(g, v, len(pl)), np.bincount(g, None, len(pl)).astype(float)
    d = rng2.integers(0, len(pl), (4000, len(pl)))
    bs = tot[d].sum(1) / cnt[d].sum(1)
    return f"{100*v.mean():+.1f} [{100*np.percentile(bs,2.5):+.1f}, {100*np.percentile(bs,97.5):+.1f}]"
for M in BUDGETS:
    for rule in ("uncertain", "leader", "runnerup"):
        print(f"  active-{rule} - rand-shift M={M}: {pooled(f'active-{rule} M={M}', f'rand-shift M={M}')} | vs rand-order: {pooled(f'active-{rule} M={M}', f'rand-order M={M}')} | vs majority: {pooled(f'active-{rule} M={M}', 'majority')}")
