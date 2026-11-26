"""Step 2: grade candidates with math-verify and print feasibility statistics."""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump_jsonl, last_boxed, load_jsonl, norm_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

PROBS = {p["pid"]: p for p in json.load(open(os.path.join(os.path.dirname(__file__), "data/problems.json")))}


def grade(pair):
    pred, gold = pair
    if pred is None:
        return None
    from math_verify import parse, verify
    try:
        g = parse("$" + gold + "$")
        p = parse("\\boxed{" + pred + "}")
        return bool(verify(g, p))
    except Exception:
        return False


if __name__ == "__main__":
    rows = load_jsonl(args.inp)
    for r in rows:
        r["pred"] = last_boxed(r["text"]) if r["finish"] == "stop" else None
    with mp.Pool(48) as pool:
        res = pool.map(grade, [(r["pred"], PROBS[r["pid"]]["answer"]) for r in rows], chunksize=64)
    for r, c in zip(rows, res):
        r["correct"] = c
        r["key"] = norm_key(r["pred"])
    dump_jsonl(rows, args.out)

    by = collections.defaultdict(list)
    for r in rows:
        by[r["pid"]].append(r)
    stats = collections.defaultdict(collections.Counter)
    for pid, rs in by.items():
        src = PROBS[pid]["source"]
        src = "math500" if src == "math500" else "comp"
        nc = sum(r["correct"] is True for r in rs)
        nw = sum(r["correct"] is False for r in rs)
        stats[src]["problems"] += 1
        stats[src]["samples"] += len(rs)
        stats[src]["valid"] += nc + nw
        stats[src]["correct"] += nc
        if nc >= 1:
            for N in (4, 8, 16, 32):
                if nw >= N - 1:
                    stats[src][f"elig_N{N}"] += 1
    for src, s in stats.items():
        print(src, dict(s), f"sample_acc={s['correct']/max(1,s['valid']):.3f}")
