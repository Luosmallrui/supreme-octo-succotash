"""Step 1: sample N candidate solutions per problem with vLLM (non-thinking mode)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GEN_SUFFIX, MODELS, dump_jsonl, last_boxed, norm_key  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(MODELS))
ap.add_argument("--sources", default="math500,aime24,aime25,hmmt25")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--n_comp", type=int, default=32, help="samples for competition sets (aime/hmmt)")
ap.add_argument("--max_tokens", type=int, default=2048)
ap.add_argument("--max_tokens_comp", type=int, default=4096)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--first", type=int, default=8, help="adaptive: screening samples per problem")
args = ap.parse_args()

from vllm import LLM, SamplingParams  # noqa: E402

probs = json.load(open(os.path.join(os.path.dirname(__file__), "data/problems.json")))
probs = [p for p in probs if p["source"] in args.sources.split(",")]
if args.limit:
    probs = probs[: args.limit]
print(f"{len(probs)} problems", flush=True)

llm = LLM(model=MODELS[args.model], max_model_len=8192, gpu_memory_utilization=0.92,
          enable_prefix_caching=True, seed=0)
tok = llm.get_tokenizer()



def is_correct(text, finish, gold):
    if finish != "stop":
        return False
    pred = last_boxed(text)
    if pred is None:
        return False
    if norm_key(pred) == norm_key(gold):
        return True
    try:
        from math_verify import parse, verify
        return bool(verify(parse("$" + gold + "$"), parse("\\boxed{" + pred + "}")))
    except Exception:
        return False


def sample(ps, n, mt, seed, k0):
    sp = SamplingParams(n=n, temperature=0.7, top_p=0.8, top_k=20, max_tokens=mt)
    prompts = [tok.apply_chat_template([{"role": "user", "content": p["problem"] + GEN_SUFFIX}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
               for p in ps]
    out = []
    for p, o in zip(ps, llm.generate(prompts, sp)):
        for k, c in enumerate(o.outputs):
            out.append(dict(pid=p["pid"], k=k0 + k, text=c.text, n_tokens=len(c.token_ids), finish=c.finish_reason))
    return out


rows = []
t0 = time.time()
for group in ["math500", "comp"]:
    ps = [p for p in probs if (p["source"] == "math500") == (group == "math500")]
    if not ps:
        continue
    n = args.n if group == "math500" else args.n_comp
    mt = args.max_tokens if group == "math500" else args.max_tokens_comp
    first = sample(ps, args.first, mt, 1234, 0)
    rows += first
    gold = {p["pid"]: p["answer"] for p in ps}
    allc = {p["pid"]: True for p in ps}
    for r in first:
        if not is_correct(r["text"], r["finish"], gold[r["pid"]]):
            allc[r["pid"]] = False
    hard = [p for p in ps if not allc[p["pid"]]]
    print(f"[{group}] stage1 {len(ps)}x{args.first}; {len(hard)} problems not all-correct; {time.time()-t0:.0f}s", flush=True)
    if hard and n > args.first:
        rows += sample(hard, n - args.first, mt, 4321, args.first)
    print(f"[{group}] stage2 {len(hard)}x{n-args.first}; elapsed {time.time()-t0:.0f}s", flush=True)

dump_jsonl(rows, args.out)
print(f"wrote {len(rows)} rows -> {args.out}", flush=True)
