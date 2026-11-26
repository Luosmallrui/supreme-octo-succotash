"""E1 analysis: positional prior in generative selection and single-call calibration.

Per mode (direct / think) and (source group, N):
  slot prior    P(pick slot k | candidate in slot k is wrong), random orderings only
  methods       majority vote (no LLM call) | raw single call | calibrated single call (slot prior estimated on
                held-out problems, 5-fold cross-fitting; also from only m problems) | majority-answer-first ordering
                (1 call) | answer vote over K random orderings (K calls) | balanced cyclic permutation (N calls, direct)
  CIs           paired bootstrap over problems for each method minus raw
"""
import argparse
import collections
import json
import os
import pickle

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="results", help="where reports and summaries are written")
ap.add_argument("--data", default="out", help="raw run outputs that are read (sel*, keys, attn*, e1_*)")
ap.add_argument("--model", default="qwen3-8b")
ap.add_argument("--B", type=int, default=2000)
ap.add_argument("--few", default="5,10,20", help="numbers of calibration problems for the few-problem prior")
args = ap.parse_args()
rng = np.random.default_rng(0)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


SETS = {s["set_id"]: s for s in load_jsonl(os.path.join(args.data, f"e1_sets_{args.model}.jsonl"))}
rep = [f"# E1 ({args.model}): positional prior and single-call calibration\n"]
summary = {}


def group(s):
    return "math500" if s["source"] == "math500" else "comp"


def cand_correct(s, k):
    return bool(s["cor"][str(k)])


def maj_acc(s):
    """Expected accuracy of majority vote over answer keys (ties broken uniformly at random)."""
    cnt = collections.Counter(s["key"].values())
    top = max(cnt.values())
    tied = [a for a, c in cnt.items() if c == top]
    cor_keys = {s["key"][k] for k in s["key"] if s["cor"][k]}
    return sum(a in cor_keys for a in tied) / len(tied)


def pick_correct(s, order, p, prior=None):
    q = np.asarray(p, float)
    if prior is not None:
        q = q / np.maximum(prior, 1e-9)
    return cand_correct(s, order[int(q.argmax())])


def boot_diff(by_pid, B):
    """by_pid: pid -> list of (method value - raw value); mean and 95% CI over problems."""
    pids = list(by_pid)
    allv = [x for v in by_pid.values() for x in v]
    bs = [np.mean([x for i in rng.integers(0, len(pids), len(pids)) for x in by_pid[pids[i]]]) for _ in range(B)]
    return float(np.mean(allv)), np.percentile(bs, [2.5, 97.5])


api_modes = sorted(f[len("e1_"):-len(f"_{args.model}.jsonl")] for f in os.listdir(args.data)
                   if f.startswith("e1_api-") and f.endswith(f"_{args.model}.jsonl"))
for mode in ["direct", "think"] + api_modes:
    fp = os.path.join(args.data, f"e1_{mode}_{args.model}.jsonl")
    if not os.path.exists(fp):
        rep.append(f"\n(no e1_{mode} results yet)\n")
        continue
    rows = load_jsonl(fp)
    has_p = all(r.get("p") is not None for r in rows)
    rep.append(f"\n## {mode} mode ({len(rows)} calls)\n")
    if mode.startswith("api-"):
        # selector is a different model judging Qwen3-8B candidates; label logprobs only if the API returned them
        ok = list({(r["set_id"], r["order_name"]): r for r in rows if "error" not in r}.values())
        has_p = bool(ok) and all(r.get("p") is not None for r in ok)
        readout = "label distribution from API logprobs" if has_p else "choice-only"
        rep.append(f"model {rows[0].get('api_model')}; {readout}. errors {np.mean(['error' in r for r in rows]):.3f}; "
                   f"forced ending {np.mean([bool(r.get('forced')) for r in rows]):.2f}; no index even after forcing "
                   f"{np.mean([r['choice'] is None for r in rows]):.3f}; mean reasoning tokens "
                   f"{np.mean([r.get('n_reason') or 0 for r in rows]):.0f}\n")
        rows = ok
    if mode == "think":
        fr = np.mean([r["forced"] for r in rows])
        tr = np.mean([r["finish"] == "length" for r in rows])
        sharp = np.mean([max(r["p"]) for r in rows if not r["forced"]])
        rep.append(f"forced ending {fr:.2f} (budget hit {tr:.2f}); mean max-probability of the decision distribution "
                   f"when not forced {sharp:.3f}\n")
    by = collections.defaultdict(list)
    for r in rows:
        by[(group(SETS[r["set_id"]]), r["N"])].append(r)

    # ---------- slot prior ----------
    rep.append("### Slot prior (random orderings): P(pick slot | slot holds a wrong candidate)\n")
    rep.append("| group | N | calls | slot 1 | slot 2 | middle | last | ratio slot1 / middle | mean P(slot 1) |")
    rep.append("|---|---|---|---|---|---|---|---|---|")
    for (g, N), rr in sorted(by.items()):
        rs = [r for r in rr if r["order_name"].startswith("r")]
        fa = np.zeros(N)
        nw = np.zeros(N)
        for r in rs:
            s = SETS[r["set_id"]]
            for i, k in enumerate(r["order"]):
                if not cand_correct(s, k):
                    nw[i] += 1
                    fa[i] += r["choice"] == i + 1
        fa = fa / np.maximum(nw, 1)
        mid = fa[1:-1].mean() if N > 2 else np.nan
        P1 = np.mean([r["p"][0] for r in rs]) if has_p else np.mean([r["choice"] == 1 for r in rs])
        summary[(mode, g, N, "fa")] = fa.tolist()
        if has_p:
            summary[(mode, g, N, "meanp")] = np.mean([r["p"] for r in rs], 0).tolist()
        rep.append(f"| {g} | {N} | {len(rs)} | {fa[0]:.3f} | {fa[1]:.3f} | {mid:.3f} | {fa[-1]:.3f} | "
                   + (f"{fa[0]/mid:.1f}" if mid >= 0.002 else "n/a (middle≈0)") + f" | {P1:.3f} |")

    # ---------- methods ----------
    rep.append("\n### Selection accuracy by method (mixed sets: ≥1 correct and ≥1 wrong candidate)\n")
    rep.append("Δ = method − raw single call, paired bootstrap over problems. calibrated = divide the decision distribution by "
               "the slot prior estimated on held-out problems (5-fold, folds by problem).\n")
    hdr = "| group | N | sets | random pick | majority vote | raw (1 call) | calibrated (1 call) [Δ, 95%CI] | " \
          + " | ".join(f"calib. m={m}" for m in args.few.split(",")) + \
          " | maj-first order (1 call) [Δ] | vote over K orders (K calls) [Δ] | cyclic perm. (N calls) [Δ] |"
    rep.append(hdr)
    rep.append("|" + "---|" * (hdr.count("|") - 1))
    for (g, N), rr in sorted(by.items()):
        sets_here = sorted({r["set_id"] for r in rr})
        pids = sorted({SETS[x]["pid"] for x in sets_here})
        fold = {p: i % 5 for i, p in enumerate(np.random.default_rng(1).permutation(pids))}
        rcalls = collections.defaultdict(list)
        for r in rr:
            rcalls[r["set_id"]].append(r)
        # priors per fold, from random-order calls of the other folds
        prior = {}
        for f in (range(5) if has_p else []):
            tr = [r for r in rr if r["order_name"].startswith("r") and fold[SETS[r["set_id"]]["pid"]] != f]
            prior[f] = np.mean([r["p"] for r in tr], 0)
        per = collections.defaultdict(lambda: collections.defaultdict(list))   # method -> pid -> values (per set)
        few = [int(x) for x in args.few.split(",")]
        for sid in sets_here:
            s = SETS[sid]
            pid = s["pid"]
            cs = rcalls[sid]
            R = [r for r in cs if r["order_name"].startswith("r")]
            if not R:
                continue
            raw = np.mean([r["correct"] for r in R])
            per["random pick"][pid].append(s["n_correct"] / N)
            per["majority vote"][pid].append(maj_acc(s))
            per["raw"][pid].append(raw)
            if has_p:
                per["calibrated"][pid].append(np.mean([pick_correct(s, r["order"], r["p"], prior[fold[pid]]) for r in R]))
            for m in (few if has_p else []):
                accs = []
                others = [p for p in pids if fold[p] != fold[pid]]
                for _ in range(10):
                    sub = set(rng.choice(others, min(m, len(others)), replace=False))
                    pr = np.mean([r["p"] for r in rr if r["order_name"].startswith("r") and SETS[r["set_id"]]["pid"] in sub], 0)
                    accs.append(np.mean([pick_correct(s, r["order"], r["p"], pr) for r in R]))
                per[f"calib m={m}"][pid].append(np.mean(accs))
            M = [r for r in cs if r["order_name"] == "maj"]
            if M:
                per["maj-first"][pid].append(float(M[0]["correct"]))
            # vote over K random orderings: vote on the chosen answer, ties broken by summed probability
            votes, mass = collections.Counter(), collections.Counter()
            for r in R:
                if r["choice"] is None:
                    continue
                k = r["order"][r["choice"] - 1]
                votes[s["key"][str(k)]] += 1
                mass[s["key"][str(k)]] += r["p"][r["choice"] - 1] if has_p else 0.0
            if votes:
                best = max(votes, key=lambda a: (votes[a], mass[a]))
                cor_keys = {s["key"][k] for k in s["key"] if s["cor"][k]}
                per["vote K"][pid].append(float(best in cor_keys))
            C = [r for r in cs if r["order_name"].startswith("c")]
            if len(C) == N:
                tot = collections.Counter()
                for r in C:
                    for i, k in enumerate(r["order"]):
                        tot[k] += r["p"][i]
                per["cyclic"][pid].append(float(cand_correct(s, max(tot, key=tot.get))))
                per["raw (cyc sets)"][pid].append(raw)

        def m_(name):
            v = [x for xs in per[name].values() for x in xs]
            return np.mean(v) if v else np.nan

        def d_(name, base="raw"):
            common = [p for p in per[name] if p in per[base]]
            if not common:
                return "n/a"
            dd = {p: [a - b for a, b in zip(per[name][p], per[base][p])] for p in common}
            mu, ci = boot_diff(dd, args.B)
            return f"{100*mu:+.1f} [{100*ci[0]:+.1f},{100*ci[1]:+.1f}]"
        res = {k: m_(k) for k in per}
        summary[(mode, g, N, "acc")] = res
        Kn = len([x for x in rcalls[sets_here[0]] if x["order_name"].startswith("r")])
        rep.append(f"| {g} | {N} | {len(sets_here)} | {100*res['random pick']:.1f} | {100*res['majority vote']:.1f} | "
                   f"{100*res['raw']:.1f} | {100*res.get('calibrated', np.nan):.1f} [{d_('calibrated')}] | "
                   + " | ".join(f"{100*res.get(f'calib m={m}', np.nan):.1f}" for m in few) + " | "
                   + (f"{100*res['maj-first']:.1f} [{d_('maj-first')}]" if "maj-first" in per else "n/a") + " | "
                   + (f"{100*res['vote K']:.1f} (K={Kn}) [{d_('vote K')}]" if "vote K" in per else "n/a") + " | "
                   + (f"{100*res['cyclic']:.1f} [{d_('cyclic', 'raw (cyc sets)')}]" if "cyclic" in per else "n/a") + " |")

open(os.path.join(args.dir, f"report_e1_{args.model}.md"), "w").write("\n".join(rep))
pickle.dump(summary, open(os.path.join(args.dir, f"e1_summary_{args.model}.pkl"), "wb"))
print("\n".join(rep))
