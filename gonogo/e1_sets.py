"""E1 step 1: candidate sets for the positional-prior / calibration experiment.

Natural setting (not the single-correct position sweep): for each problem and N we draw a random subset of
N sampled candidates (conditioned on containing at least one correct and one wrong final answer), then
several orderings of the same subset:
  r0..r{K-1}   random orderings (raw single call; K-call vote / averaging baselines)
  maj          candidates sorted by answer frequency (most common answer first): "bias-aware ordering"
  c0..c{N-1}   cyclic shifts of r0 (balanced permutation baseline), only for --cyc_sets sets per problem
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, dump_jsonl, load_jsonl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3-8b")
ap.add_argument("--plan", default="math500:4:100:1,math500:8:100:1,math500:16:100:1,math500:32:40:2,comp:4:40:1,comp:8:40:1",
                help="source:N:max_problems:sets_per_problem")
ap.add_argument("--K", type=int, default=3, help="random orderings per set")
ap.add_argument("--cyc_sets", type=int, default=1, help="sets per problem that also get all N cyclic shifts")
ap.add_argument("--max_prompt", type=int, default=28000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
probs = {p["pid"]: p for p in json.load(open(os.path.join(here, "data/problems.json")))}
tok = AutoTokenizer.from_pretrained(MODELS[args.model])
by = collections.defaultdict(list)
for r in load_jsonl(os.path.join(here, f"out/graded_{args.model}.jsonl")):
    if r["key"] is not None:          # drop truncated candidates (no final answer): trivially rejectable
        by[r["pid"]].append(r)
rng = random.Random(args.seed)
sets, summary = [], {}
for item in args.plan.split(","):
    src, N, cap, S = item.split(":")
    N, cap, S = int(N), int(cap), int(S)
    elig = []
    for pid, rs in sorted(by.items()):
        source = probs[pid]["source"]
        if (src == "math500") != (source == "math500"):
            continue
        plen = len(tok(probs[pid]["problem"])["input_ids"])
        lmax = min(4096, (args.max_prompt - plen - 300 - 8 * N) // N)
        pool = [r for r in rs if r["n_tokens"] <= lmax]
        if len(pool) >= N and any(r["correct"] for r in pool) and any(not r["correct"] for r in pool):
            elig.append((pid, pool))
    rng.shuffle(elig)
    elig = elig[:cap]
    n_sets = 0
    for pid, pool in elig:
        for s in range(S):
            for _ in range(200):
                sub = rng.sample(pool, N)
                if any(r["correct"] for r in sub) and any(not r["correct"] for r in sub):
                    break
            else:
                continue
            ks = [r["k"] for r in sub]
            orders = {}
            for i in range(args.K):
                o = ks[:]
                rng.shuffle(o)
                orders[f"r{i}"] = o
            cnt = collections.Counter(r["key"] for r in sub)
            ans = {r["k"]: r["key"] for r in sub}
            rand_rank = {k: rng.random() for k in ks}
            orders["maj"] = sorted(ks, key=lambda k: (-cnt[ans[k]], ans[k], rand_rank[k]))
            if s < args.cyc_sets:
                for c in range(N):
                    orders[f"c{c}"] = orders["r0"][c:] + orders["r0"][:c]
            sets.append(dict(set_id=f"{src}-N{N}-{pid}-s{s}", pid=pid, source=probs[pid]["source"], N=N, ks=ks,
                             cor={str(r["k"]): bool(r["correct"]) for r in sub}, key={str(r["k"]): r["key"] for r in sub},
                             n_correct=sum(r["correct"] for r in sub), orders=orders))
            n_sets += 1
    summary[f"{src}:N{N}"] = dict(eligible_problems_used=len(elig), sets=n_sets)
print(json.dumps(summary, indent=1))
print(f"{len(sets)} sets")
dump_jsonl(sets, args.out)
