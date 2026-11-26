"""Step 4: run selection (A) / aggregation with headers (B) / aggregation without headers (C)."""
import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MODELS, PREFILL, build_chat, build_user_content, dump_jsonl, last_boxed, load_jsonl,  # noqa: E402
                    parse_index, parse_prefilled_boxed)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--specs", required=True)
ap.add_argument("--modes", default="A,B,C")
ap.add_argument("--think", action="store_true", help="thinking-mode selection")
ap.add_argument("--max_think_tokens", type=int, default=6144)
ap.add_argument("--filter", default="", help="python expr on spec s, e.g. \"s['draw']==0 and s['N']<=16\"")
ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the first K specs")
ap.add_argument("--out", required=True)
args = ap.parse_args()

here = os.path.dirname(os.path.abspath(__file__))
PROBS = {p["pid"]: p for p in json.load(open(os.path.join(here, "data/problems.json")))}


def grade(pair):
    pred, gold = pair
    if pred is None:
        return False
    from math_verify import parse, verify
    try:
        return bool(verify(parse("$" + gold + "$"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    specs = load_jsonl(args.specs)
    if args.filter:
        specs = [s for s in specs if eval(args.filter, {}, {"s": s})]
    if args.limit:
        specs = specs[: args.limit]
    cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(os.path.join(here, f"out/graded_{args.model}.jsonl"))}
    modes = args.modes.split(",")

    llm = LLM(model=MODELS[args.model], max_model_len=40960, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0)
    tok = llm.get_tokenizer()

    # order so that prompts sharing prefixes are adjacent: group -> (A,B per pos) -> (C per pos)
    groups = collections.defaultdict(list)
    for s in specs:
        groups[(s["N"], s["draw"], s["pid"])].append(s)
    reqs = []
    for g, ss in groups.items():
        ss.sort(key=lambda s: s["pos"])
        hdr_modes = [m for m in modes if m in ("A", "B")]
        for s in ss:
            for m in hdr_modes:
                reqs.append((s, m))
        if "C" in modes:
            for s in ss:
                reqs.append((s, "C"))

    prompts, sps = [], []
    for s, m in reqs:
        texts = [cands[(s["pid"], k)] for k in s["ks"]]
        content, _ = build_user_content(PROBS[s["pid"]]["problem"], texts, m)
        if args.think:
            prompts.append(build_chat(tok, content, enable_thinking=True))
            sps.append(SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_think_tokens, seed=0))
        else:
            prompts.append(build_chat(tok, content, enable_thinking=False, prefill=PREFILL[m]))
            sps.append(SamplingParams(temperature=0.0, max_tokens=4 if m == "A" else 48,
                                      logprobs=20 if m == "A" else None))
    print(f"{len(prompts)} requests", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, sps)
    print(f"generation done in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for (s, m), o in zip(reqs, outs):
        text = o.outputs[0].text
        ans_text = text.split("</think>")[-1] if args.think else text
        r = dict(s, mode=m, out=text if not args.think else ans_text[-2000:], n_prompt=len(o.prompt_token_ids),
                 n_out=len(o.outputs[0].token_ids), finish=o.outputs[0].finish_reason)
        if args.think:
            r["gen"] = text  # full reasoning trace, needed for reasoning-token attention capture
        if args.think and "</think>" not in text:
            ans_text = None
        if m == "A":
            r["chosen"] = parse_index(ans_text, s["N"]) if ans_text is not None else None
            r["correct"] = r["chosen"] == s["pos"]
            lps = o.outputs[0].logprobs
            if lps:  # first-token distribution over candidate digits (exact choice prob when N <= 9)
                r["lp1"] = {v.decoded_token: round(v.logprob, 4) for v in lps[0].values()}
        else:
            if ans_text is None:
                r["pred"] = None
            elif args.think:
                r["pred"] = last_boxed(ans_text)
            else:
                r["pred"] = parse_prefilled_boxed(ans_text)
        rows.append(r)
    bc = [r for r in rows if r["mode"] != "A"]
    with mp.Pool(32) as pool:
        res = pool.map(grade, [(r["pred"], PROBS[r["pid"]]["answer"]) for r in bc], chunksize=32)
    for r, c in zip(bc, res):
        r["correct"] = c
    dump_jsonl(rows, args.out)

    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["N"], r["mode"])].append(r["correct"])
    for k in sorted(agg):
        v = agg[k]
        print(k, f"n={len(v)} acc={sum(v)/len(v):.3f}")
