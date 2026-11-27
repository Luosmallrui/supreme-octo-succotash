"""E5: sacrificial anchor. A decoy block is put first in the E1 random orderings (direct readout).

For each E1 set and each random ordering r0..r2, the N real candidates keep their order but move to slots 2..N+1
(printed "Solution 2".."Solution N+1"), and slot 1 holds a decoy:
  wrong   a wrong sampled solution of the same problem that is not in the set, preferring one whose final answer
          differs from every answer in the set (lowest candidate id); sets without any such solution are skipped
  blank   "(This solution is intentionally left blank.)"
The pre-filled readout gives the distribution over the N+1 printed labels; index 0 is the decoy. The control is the
existing E1 direct run on the same orderings without a decoy (out/e1_direct_<model>.jsonl).

  python e5_run.py --model qwen3-8b --out out/e5_decoy/e5_qwen3-8b.jsonl
  python e5_run.py --model qwen3-8b --dry_run
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS, PREFILL, build_chat, build_user_content, dump_jsonl, load_jsonl  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BLANK = "(This solution is intentionally left blank.)"


def wrong_decoy(s, pool):
    """Lowest-id wrong, non-truncated solution of s's problem outside the set; prefer an answer not in the set."""
    in_set = {int(k) for k in s["key"]}
    answers = set(s["key"].values())
    wrong = [r for r in pool.get(s["pid"], []) if r["key"] is not None and not r["correct"] and r["k"] not in in_set]
    if not wrong:
        return None
    return min(wrong, key=lambda r: (r["key"] in answers, r["k"]))


def decoy_calls(sets, problems, cands, pool, decoys):
    calls, skipped = [], 0
    for s in sets:
        dw = wrong_decoy(s, pool)
        for name in sorted(n for n in s["orders"] if n.startswith("r")):
            order = s["orders"][name]
            for kind in decoys:
                if kind == "wrong" and dw is None:
                    skipped += 1
                    continue
                text = BLANK if kind == "blank" else dw["text"]
                content, _ = build_user_content(problems[s["pid"]], [text] + [cands[s["pid"], k] for k in order], "A")
                calls.append(dict(set_id=s["set_id"], pid=s["pid"], source=s["source"], N=s["N"], n_correct=s["n_correct"],
                                  order_name=name, order=order, decoy=kind,
                                  decoy_k=None if kind == "blank" else dw["k"],
                                  decoy_key=None if kind == "blank" else dw["key"],
                                  decoy_key_in_set=None if kind == "blank" else dw["key"] in set(s["key"].values()),
                                  request_id=f"{s['set_id']}/{name}/{kind}", content=content))
    return calls, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--sets", help="E1 sets (default out/e1_sets_<model>.jsonl)")
    ap.add_argument("--decoys", default="wrong,blank")
    ap.add_argument("--max_prompt", type=int, default=28500, help="drop prompts longer than this (as e1_run.py)")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    sets = load_jsonl(args.sets or os.path.join(HERE, f"out/e1_sets_{args.model}.jsonl"))
    problems = {p["pid"]: p["problem"] for p in json.load(open(os.path.join(HERE, "data/problems.json"), encoding="utf-8"))}
    graded = load_jsonl(os.path.join(HERE, f"out/graded_{args.model}.jsonl"))
    cands = {(r["pid"], r["k"]): r["text"] for r in graded}
    pool = {}
    for r in graded:
        pool.setdefault(r["pid"], []).append(r)
    calls, skipped = decoy_calls(sets, problems, cands, pool, args.decoys.split(","))
    if args.limit:
        calls = calls[:args.limit]
    n_in = sum(c["decoy_key_in_set"] is True for c in calls)
    print(f"{len(calls)} decoy calls over {len(sets)} sets ({skipped} wrong-decoy calls skipped: no wrong solution "
          f"outside the set; {n_in} wrong decoys share an answer with the set); ~{sum(len(c['content']) for c in calls) / 4e6:.1f}M "
          f"prompt tokens (chars/4)", flush=True)
    if args.dry_run:
        c = calls[0]
        print(c["content"][:600] + "\n...\n" + c["content"][-250:])
        return
    if not args.out:
        ap.error("--out is required unless --dry_run")
    out_path = os.path.abspath(args.out)

    from vllm import LLM
    from label_readout import score
    llm = LLM(model=MODELS[args.model], max_model_len=40960, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0)
    tok = llm.get_tokenizer()
    prefixes = [build_chat(tok, c["content"], enable_thinking=False, prefill=PREFILL["A"]) for c in calls]
    n_tok = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in prefixes]
    keep = [i for i, n in enumerate(n_tok) if n <= args.max_prompt]
    print(f"{len(calls) - len(keep)} calls dropped as longer than {args.max_prompt} tokens", flush=True)
    calls = [calls[i] for i in keep]
    prefixes = [prefixes[i] for i in keep]
    t0 = time.time()
    res = score(llm, prefixes, [c["N"] + 1 for c in calls], args.chunk,
                log=lambda m: print(f"{m} {time.time() - t0:.0f}s", flush=True))
    rows = []
    for c, (p, mass, n_prompt) in zip(calls, res):
        row = {k: v for k, v in c.items() if k != "content"}
        choice = max(range(len(p)), key=lambda i: p[i]) if mass > 0 else None
        row.update(model=args.model, mode="direct", p=[round(x, 6) for x in p], p_mass=round(mass, 4),
                   choice_slot=None if choice is None else choice + 1, chose_decoy=choice == 0, n_prompt=n_prompt)
        rows.append(row)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dump_jsonl(rows, out_path)
    for kind in sorted({r["decoy"] for r in rows}):
        rs = [r for r in rows if r["decoy"] == kind]
        print(f"{kind}: {len(rs)} calls, mean P(decoy) {sum(r['p'][0] for r in rs) / len(rs):.3f}, "
              f"decoy chosen {sum(r['chose_decoy'] for r in rs) / len(rs):.3f}", flush=True)
    print(f"done in {time.time() - t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
