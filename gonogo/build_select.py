"""Step 3: build position-sweep specs.

For each eligible problem and each N, draw R candidate sets of 1 correct + (N-1) wrong
candidates. The wrong candidates keep a fixed relative order and the correct one is
moved through every position 1..N (Lost-in-the-Middle style sweep).
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, dump_jsonl, load_jsonl, norm_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--R", type=int, default=2)
ap.add_argument("--Ns", default="4,8,16,32")
ap.add_argument("--cap_math", default="4:150,8:120,16:90,32:60", help="max MATH500 problems per N")
ap.add_argument("--cap_comp", default="", help="max AIME/HMMT problems per N (default: all of them)")
ap.add_argument("--max_prompt", type=int, default=38000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--draw_offset", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

from transformers import AutoTokenizer  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
probs = {p["pid"]: p for p in json.load(open(os.path.join(here, "data/problems.json")))}
tok = AutoTokenizer.from_pretrained(MODELS[args.model])
rows = load_jsonl(os.path.join(here, f"out/graded_{args.model}.jsonl"))
by = collections.defaultdict(list)
for r in rows:
    by[r["pid"]].append(r)

caps = {int(a): int(b) for a, b in (x.split(":") for x in args.cap_math.split(","))}
capc = {int(a): int(b) for a, b in (x.split(":") for x in args.cap_comp.split(","))} if args.cap_comp else {}
rng = random.Random(args.seed)
specs = []
summary = {}
for N in [int(x) for x in args.Ns.split(",")]:
    elig = []
    for pid, rs in sorted(by.items()):
        p = probs[pid]
        plen = len(tok(p["problem"])["input_ids"])
        lmax = min(4096, (args.max_prompt - plen - 200 - 8 * N) // N)
        gk = norm_key(p["answer"])
        cor = [r for r in rs if r["correct"] is True and r["n_tokens"] <= lmax]
        wro = [r for r in rs if r["correct"] is False and r["n_tokens"] <= lmax and r["key"] != gk]
        if len(cor) >= 1 and len(wro) >= N - 1:
            elig.append((pid, cor, wro))
    math = [e for e in elig if probs[e[0]]["source"] == "math500"]
    comp = [e for e in elig if probs[e[0]]["source"] != "math500"]
    rng.shuffle(math)
    if capc:  # only reshuffle when capping, so earlier spec files stay reproducible
        rng.shuffle(comp)
    chosen = comp[: capc.get(N, 10**9)] + math[: caps.get(N, 10**9)]
    summary[N] = dict(elig_math=len(math), elig_comp=len(comp), used=len(chosen))
    for pid, cor, wro in chosen:
        for d in range(args.R):
            c = rng.choice(cor)
            W = rng.sample(wro, N - 1)
            for pos in range(1, N + 1):
                order = W[: pos - 1] + [c] + W[pos - 1 :]
                specs.append(dict(model=args.model, N=N, draw=d + args.draw_offset, pid=pid, pos=pos,
                                  source=probs[pid]["source"],
                                  ks=[r["k"] for r in order],
                                  n_distinct_wrong=len({r["key"] for r in W})))
print(json.dumps(summary, indent=1))
print(f"{len(specs)} specs")
dump_jsonl(specs, args.out)
