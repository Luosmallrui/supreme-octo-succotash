"""E6: order-balanced answer duels on the E1 natural sets (direct readout).

For each E1 set, every distinct final answer is represented by one candidate (the lowest candidate id with that
answer). Every ordered pair of answers (a, b), a != b, is shown as a two-candidate selection prompt with the E1
template (Solution 1 = representative of a, Solution 2 = representative of b), and the pre-filled readout gives
P(label 1) and P(label 2). Showing both orders lets the analysis cancel the first-position advantage exactly:
d_ab = (logit p_ab - logit p_ba) / 2 (see e6_analyze.py).

  python e6_run.py --model qwen3-8b --out out/e6_duels/e6_qwen3-8b.jsonl
  python e6_run.py --model qwen3-8b --dry_run        # build prompts only (no vLLM, no tokenizer)
"""
import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, PREFILL, build_chat, build_user_content, dump_jsonl, load_jsonl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def representatives(s):
    """{answer key: lowest candidate id with that key} for one E1 set."""
    rep = {}
    for k in sorted(int(x) for x in s["key"]):
        rep.setdefault(s["key"][str(k)], k)
    return rep


def duel_calls(sets, problems, cands):
    calls = []
    for s in sets:
        rep = representatives(s)
        for a, b in itertools.permutations(sorted(rep), 2):
            ka, kb = rep[a], rep[b]
            content, _ = build_user_content(problems[s["pid"]], [cands[s["pid"], ka], cands[s["pid"], kb]], "A")
            calls.append(dict(set_id=s["set_id"], pid=s["pid"], source=s["source"], N=s["N"], n_answers=len(rep),
                              request_id=f"{s['set_id']}/{ka}-{kb}", key_a=a, key_b=b, k_a=ka, k_b=kb,
                              cor_a=bool(s["cor"][str(ka)]), cor_b=bool(s["cor"][str(kb)]),
                              n_a=sum(v == a for v in s["key"].values()), n_b=sum(v == b for v in s["key"].values()),
                              content=content))
    return calls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--sets", help="E1 sets (default out/e1_sets_<model>.jsonl)")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first K calls only")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    sets = load_jsonl(args.sets or os.path.join(HERE, f"out/e1_sets_{args.model}.jsonl"))
    problems = {p["pid"]: p["problem"] for p in json.load(open(os.path.join(HERE, "data/problems.json"), encoding="utf-8"))}
    cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(os.path.join(HERE, f"out/graded_{args.model}.jsonl"))}
    calls = duel_calls(sets, problems, cands)
    if args.limit:
        calls = calls[:args.limit]
    print(f"{len(calls)} duel calls over {len(sets)} sets; ~{sum(len(c['content']) for c in calls) / 4e6:.1f}M prompt "
          f"tokens (chars/4)", flush=True)
    if args.dry_run:
        print(calls[0]["content"][:400] + "\n...\n" + calls[0]["content"][-300:])
        return
    if not args.out:
        ap.error("--out is required unless --dry_run")
    out_path = os.path.abspath(args.out)

    from vllm import LLM
    from label_readout import score
    llm = LLM(model=MODELS[args.model], max_model_len=40960, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0)
    tok = llm.get_tokenizer()
    t0 = time.time()
    prefixes = [build_chat(tok, c["content"], enable_thinking=False, prefill=PREFILL["A"]) for c in calls]
    res = score(llm, prefixes, [2] * len(calls), args.chunk,
                log=lambda m: print(f"{m} {time.time() - t0:.0f}s", flush=True))
    rows = []
    for c, (p, mass, n_prompt) in zip(calls, res):
        row = {k: v for k, v in c.items() if k != "content"}
        row.update(model=args.model, mode="direct", p=[round(x, 6) for x in p], p_mass=round(mass, 4), n_prompt=n_prompt)
        rows.append(row)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump_jsonl(rows, out_path)
    corr = [r for r in rows if r["cor_a"] != r["cor_b"]]
    if corr:
        pc = sum(r["p"][0] if r["cor_a"] else r["p"][1] for r in corr) / len(corr)
        print(f"mean P(correct) over {len(corr)} correct-vs-wrong duels: {pc:.3f}; mean P(first): "
              f"{sum(r['p'][0] for r in rows) / len(rows):.3f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
