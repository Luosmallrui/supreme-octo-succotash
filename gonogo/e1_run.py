"""E1 step 2: run selection over candidate sets and record the full decision distribution over indices.

mode direct: assistant turn pre-filled with "The best solution is Solution "; distribution read at that point.
mode think:  thinking-mode generation, then the distribution is read at the chosen-index position of the final
             \\boxed{...}. Runs that hit the budget, or finish without a boxed index in 1..N, get a forced
             ending ("The best solution is Solution ") and are flagged forced=True.

P(index = k) for k = 1..N: first-token logprobs give P(first digit d); for N >= 10 a second pass on
prefix+d gives P(next digit | d), so P(10d+x) = P(d) P(x|d) and P(d alone) = P(d) (1 - sum_x P(x|d)).
Tokenizers that encode every label 1..N as one token (Llama 3: "12" is a single token) are read in one pass from
the label tokens themselves (readout="label tokens"); the digit-by-digit reading would split "12" in a way the model
never sees.

--judge PATH   another local model selects among --model's candidates (direct mode only); its prompts use its own
               chat template (for Llama-3.1 including the default system turn), with a doubled BOS removed.
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (MODELS, PREFILL, build_chat, build_user_content, dump_jsonl, last_boxed, load_jsonl,  # noqa: E402
                    parse_index, vllm_prompt)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3-8b")
ap.add_argument("--sets", required=True)
ap.add_argument("--mode", choices=["direct", "think"], required=True)
ap.add_argument("--orders", default="all", help="'all', or comma list of order-name prefixes, e.g. r,maj")
ap.add_argument("--max_think_tokens", type=int, default=12288)
ap.add_argument("--max_prompt", type=int, default=28500)
ap.add_argument("--filter", default="", help="python expr on set s")
ap.add_argument("--limit", type=int, default=0, help="smoke test: keep only the first K calls")
ap.add_argument("--gen_from", default="", help="think mode: reuse generations saved in this .gen.jsonl (skip generation)")
ap.add_argument("--chunk", type=int, default=8, help="calls per scoring chunk (second-digit pass reuses the prefix cache)")
ap.add_argument("--out", required=True)
ap.add_argument("--judge", default="", help="local model path that selects instead of --model (direct mode only)")
args = ap.parse_args()
if args.judge and args.mode != "direct":
    ap.error("--judge is for --mode direct")

here = os.path.dirname(os.path.abspath(__file__))
PROBS = {p["pid"]: p for p in json.load(open(os.path.join(here, "data/problems.json")))}
FORCE = "The best solution is Solution "


def digit_probs(lp):
    """vLLM logprob dict -> {digit string: prob} for single-digit tokens."""
    out = {}
    for v in lp.values():
        t = (v.decoded_token or "").strip()
        if len(t) == 1 and t.isascii() and t.isdigit():
            out[t] = out.get(t, 0.0) + math.exp(v.logprob)
    return out


def label_token_probs(lp, n):
    """vLLM logprob dict -> list of P(label k), k = 1..n, for tokenizers with one token per label."""
    p = [0.0] * n
    for v in lp.values():
        t = (v.decoded_token or "").strip()
        if t.isascii() and t.isdigit() and not t.startswith("0") and 1 <= int(t) <= n:
            p[int(t) - 1] += math.exp(v.logprob)
    return p


def single_token_labels(tok, n):
    """True if every label 1..n is encoded as one token on its own."""
    return all(len(tok.encode(str(k), add_special_tokens=False)) == 1 for k in range(1, n + 1))


if __name__ == "__main__":
    from vllm import LLM, SamplingParams

    sets = load_jsonl(args.sets)
    if args.filter:
        sets = [s for s in sets if eval(args.filter, {}, {"s": s})]
    cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(os.path.join(here, f"out/graded_{args.model}.jsonl"))}
    llm = LLM(model=args.judge or MODELS[args.model], max_model_len=40960, gpu_memory_utilization=0.92,
              enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0, max_logprobs=64)
    tok = llm.get_tokenizer()
    judge = os.path.basename(os.path.normpath(args.judge)).lower() if args.judge else None
    one_pass = single_token_labels(tok, max(s["N"] for s in sets))
    print(f"label readout: {'one pass over label tokens' if one_pass else 'digit by digit'}", flush=True)

    calls = []
    for s in sets:
        for name, order in s["orders"].items():
            if args.orders != "all" and not any(name.startswith(p) for p in args.orders.split(",")):
                continue
            content, _ = build_user_content(PROBS[s["pid"]]["problem"], [cands[(s["pid"], k)] for k in order], "A")
            prompt = vllm_prompt(tok, build_chat(tok, content, enable_thinking=(args.mode == "think"),
                                                prefill="" if args.mode == "think" else PREFILL["A"]))
            calls.append(dict(s=s, name=name, order=order, prompt=prompt))
    if args.limit:
        calls = calls[: args.limit]
    n_tok = [len(tok(c["prompt"], add_special_tokens=False)["input_ids"]) for c in calls]
    too_long = sum(n > args.max_prompt for n in n_tok)
    calls = [c for c, n in zip(calls, n_tok) if n <= args.max_prompt]
    print(f"{len(calls)} calls ({too_long} dropped as longer than {args.max_prompt} tokens)", flush=True)
    t0 = time.time()

    # ---- generation (think) ----
    if args.mode == "think":
        if args.gen_from:
            saved = {(g["set_id"], g["order_name"]): g for g in load_jsonl(args.gen_from)}
            calls = [c for c in calls if (c["s"]["set_id"], c["name"]) in saved]
            for c in calls:
                g = saved[(c["s"]["set_id"], c["name"])]
                c.update(gen=g["gen"], n_out=g["n_out"], finish=g["finish"])
            print(f"loaded {len(calls)} saved generations from {args.gen_from}", flush=True)
        else:
            sp = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=args.max_think_tokens, seed=0)
            outs = llm.generate([c["prompt"] for c in calls], sp)
            for c, o in zip(calls, outs):
                c.update(gen=o.outputs[0].text, n_out=len(o.outputs[0].token_ids), finish=o.outputs[0].finish_reason)
            # checkpoint before scoring, so a scoring failure does not lose hours of generation
            dump_jsonl([dict(set_id=c["s"]["set_id"], order_name=c["name"], gen=c["gen"], n_out=c["n_out"], finish=c["finish"])
                        for c in calls], args.out + ".gen.jsonl")
        for c in calls:
            gen = c["gen"]
            N = c["s"]["N"]
            j = gen.find("</think>")
            choice_gen, prefix, forced = None, None, True
            if j >= 0:
                ans = gen[j:]
                choice_gen = parse_index(ans, N)
                b = ans.rfind("\\boxed{")
                bx = last_boxed(ans)
                if b >= 0 and bx is not None and re.fullmatch(r"\s*\d+\s*", bx) and 1 <= int(bx) <= N:
                    prefix = c["prompt"] + gen[: j + b + len("\\boxed{")]
                    forced = False
                else:
                    prefix = c["prompt"] + gen + "\n\n" + FORCE
            else:
                prefix = c["prompt"] + gen + "\n</think>\n\n" + FORCE
            c.update(choice_gen=choice_gen, prefix=prefix, forced=forced)
        print(f"generation done in {time.time()-t0:.0f}s", flush=True)
    else:
        for c in calls:
            c.update(prefix=c["prompt"], forced=False, choice_gen=None, n_out=0, finish=None)

    # ---- decision distribution over indices ----
    # scoring in small chunks: the second-digit prompts (prefix + d) then hit the prefix cache of the first pass
    n_max = max(c["s"]["N"] for c in calls)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, logprobs=min(64, max(20, n_max + 8)) if one_pass else 20)
    p1, p2, plab, n_prompt = {}, {}, {}, {}
    for c0 in range(0, len(calls), args.chunk):
        idx = list(range(c0, min(c0 + args.chunk, len(calls))))
        o1 = llm.generate([calls[i]["prefix"] for i in idx], sp1, use_tqdm=False)
        for i, o in zip(idx, o1):
            if one_pass:
                plab[i] = label_token_probs(o.outputs[0].logprobs[0], calls[i]["s"]["N"])
            else:
                p1[i] = digit_probs(o.outputs[0].logprobs[0])
            n_prompt[i] = len(o.prompt_token_ids)
        second = [] if one_pass else [(i, d) for i in idx for d in range(1, calls[i]["s"]["N"] // 10 + 1)]
        if second:
            o2 = llm.generate([calls[i]["prefix"] + str(d) for i, d in second], sp1, use_tqdm=False)
            for (i, d), o in zip(second, o2):
                p2[(i, d)] = digit_probs(o.outputs[0].logprobs[0])
        if (c0 // args.chunk) % 25 == 0:
            print(f"scored {idx[-1]+1}/{len(calls)} {time.time()-t0:.0f}s", flush=True)
    print(f"scoring done in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for i, c in enumerate(calls):
        s, N = c["s"], c["s"]["N"]
        if one_pass:
            p = plab[i]
        else:
            first = p1[i]
            p = [0.0] * N
            for d in range(1, 10):
                pd = first.get(str(d), 0.0)
                if d > N or pd == 0:
                    continue
                nxt = p2.get((i, d), {})
                for x in range(10):
                    if 10 * d + x <= N:
                        p[10 * d + x - 1] += pd * nxt.get(str(x), 0.0)
                p[d - 1] += pd * (1 - sum(nxt.values()))
        mass = sum(p)
        choice_p = max(range(N), key=lambda k: p[k]) + 1 if mass > 0 else None
        choice = c["choice_gen"] if (c["choice_gen"] is not None and not c["forced"]) else choice_p
        k_chosen = c["order"][choice - 1] if choice else None
        row = dict(set_id=s["set_id"], pid=s["pid"], source=s["source"], N=N, n_correct=s["n_correct"], order_name=c["name"],
                   order=c["order"], mode=args.mode, p=[round(x / mass, 6) if mass > 0 else 0.0 for x in p], p_mass=round(mass, 4),
                   choice=choice, choice_p=choice_p, choice_gen=c["choice_gen"], forced=c["forced"],
                   correct=bool(k_chosen is not None and s["cor"][str(k_chosen)]), n_out=c["n_out"], finish=c["finish"],
                   n_prompt=n_prompt[i])
        if args.mode == "think":
            row["gen"] = c["gen"]
        if judge:   # new fields only for other judges, so Qwen3 rows keep their format
            row.update(judge=judge, readout="label tokens" if one_pass else "digits")
        rows.append(row)
    dump_jsonl(rows, args.out)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["source"] == "math500", r["N"], r["order_name"][0] if r["order_name"][0] in "rc" else r["order_name"])].append(r)
    for k in sorted(agg):
        v = agg[k]
        print(("math500" if k[0] else "comp"), f"N={k[1]} order={k[2]}: n={len(v)} acc={sum(r['correct'] for r in v)/len(v):.3f} "
              f"forced={sum(r['forced'] for r in v)/len(v):.2f} mass={sum(r['p_mass'] for r in v)/len(v):.3f}", flush=True)
