"""E1 on hosted models (SiliconFlow, OpenAI-compatible chat API). Choice-only: this API returns no logprobs.

Same candidate sets and orderings as the local Qwen3-8B run (random orderings r* and majority-answer-first 'maj').
Calls are shuffled (fixed seed) so a partial run is balanced across N; every result is appended as soon as it
returns, and a restart skips (set_id, order_name) pairs already in --out. Token caps stop new submissions.
The key is read from $SILICONFLOW_API_KEY and never printed.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_user_content, last_boxed, load_jsonl  # noqa: E402
import re  # noqa: E402

FORCE = "The best solution is Solution "


def strict_index(text, N):
    """Only a boxed integer in 1..N counts (no 'first number in the text' fallback)."""
    b = last_boxed(text or "")
    if b is not None and re.fullmatch(r"\s*(?:\\text\{)?\s*(?:Solution\s*)?(\d+)\s*\}?\s*", b):
        k = int(re.search(r"\d+", b).group())
        return k if 1 <= k <= N else None
    return None

ap = argparse.ArgumentParser()
ap.add_argument("--api_model", required=True, help="e.g. Qwen/Qwen3-32B, deepseek-ai/DeepSeek-R1")
ap.add_argument("--tag", required=True, help="short name used in the output mode, e.g. qwen3-32b")
ap.add_argument("--sets", default="out/e1_sets_qwen3-8b.jsonl")
ap.add_argument("--cands", default="out/graded_qwen3-8b.jsonl")
ap.add_argument("--orders", default="r,maj")
ap.add_argument("--thinking_budget", type=int, default=12288)
ap.add_argument("--no_thinking_params", action="store_true", help="do not send enable_thinking / thinking_budget")
ap.add_argument("--max_tokens", type=int, default=2048, help="answer tokens (excludes reasoning on SiliconFlow)")
ap.add_argument("--concurrency", type=int, default=16)
ap.add_argument("--filter", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max_in_m", type=float, default=60.0, help="stop submitting after this many million prompt tokens")
ap.add_argument("--max_out_m", type=float, default=20.0, help="stop submitting after this many million completion tokens")
ap.add_argument("--out", required=True)
args = ap.parse_args()

BASE = "https://api.siliconflow.cn/v1/chat/completions"
KEY = os.environ.get("SILICONFLOW_API_KEY", "")
if not KEY:
    sys.exit("SILICONFLOW_API_KEY is not set")
here = os.path.dirname(os.path.abspath(__file__))
PROBS = {p["pid"]: p for p in json.load(open(os.path.join(here, "data/problems.json")))}
sets = load_jsonl(os.path.join(here, args.sets))
if args.filter:
    sets = [s for s in sets if eval(args.filter, {}, {"s": s})]
cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(os.path.join(here, args.cands))}

calls = [(s, name, order) for s in sets for name, order in s["orders"].items()
         if any(name.startswith(p) for p in args.orders.split(","))]
random.Random(0).shuffle(calls)
done = set()
if os.path.exists(args.out):
    done = {(r["set_id"], r["order_name"]) for r in load_jsonl(args.out) if "error" not in r}   # error rows are retried
calls = [c for c in calls if (c[0]["set_id"], c[1]) not in done]
if args.limit:
    calls = calls[: args.limit]
print(f"{len(calls)} calls to run ({len(done)} already done) on {args.api_model}", flush=True)

lock = threading.Lock()
tot = {"in": 0, "out": 0, "n": 0, "err": 0}
stop = threading.Event()


def post(body):
    data = json.dumps(body).encode()
    last = "http error"
    for attempt in range(8):
        req = urllib.request.Request(BASE, data=data, method="POST",
                                     headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=2400) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:200]
            if e.code in (429, 500, 502, 503, 504):
                last = f"{e.code} {msg}"
                time.sleep(min(300, 15 * 2 ** attempt))
                continue
            return {"error": f"{e.code} {msg}"}
        except Exception as e:  # timeouts, connection resets
            time.sleep(min(300, 15 * 2 ** attempt))
            last = repr(e)
    return {"error": f"gave up after retries: {last}"}


def run(c):
    try:
        return run_one(c)
    except Exception as e:   # record and keep going; the row is retried on the next start
        s, name, order = c
        row = dict(set_id=s["set_id"], pid=s["pid"], source=s["source"], N=s["N"], order_name=name, order=order,
                   mode=f"api-{args.tag}", api_model=args.api_model, p=None, choice=None, correct=False, error=f"exception: {e!r}")
        with lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            tot["err"] += 1
        return row


def run_one(c):
    if stop.is_set():
        return None
    s, name, order = c
    content, _ = build_user_content(PROBS[s["pid"]]["problem"], [cands[(s["pid"], k)] for k in order], "A")
    body = dict(model=args.api_model, messages=[{"role": "user", "content": content}], max_tokens=args.max_tokens,
                temperature=0.6, top_p=0.95, stream=False)
    if not args.no_thinking_params:
        body.update(enable_thinking=True, thinking_budget=args.thinking_budget)
    t0 = time.time()
    js = post(body)
    N = s["N"]
    row = dict(set_id=s["set_id"], pid=s["pid"], source=s["source"], N=N, n_correct=s["n_correct"], order_name=name,
               order=order, mode=f"api-{args.tag}", api_model=args.api_model, p=None, latency=round(time.time() - t0, 1))
    if "error" in js:
        row.update(error=js["error"], choice=None, correct=False)
    else:
        ch = js["choices"][0]
        msg = ch["message"]
        ans = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        choice = strict_index(ans, N)
        u = js.get("usage", {})
        forced, f_in = False, 0
        if choice is None:
            # forced ending: hand the reasoning + answer back as a pre-filled assistant turn and read the index
            pre = "<think>\n" + reasoning + "\n</think>\n\n" + ans + "\n\n" + FORCE
            fb = dict(model=args.api_model, max_tokens=4, temperature=0.0, stream=False,
                      messages=[{"role": "user", "content": content}, {"role": "assistant", "content": pre}])
            if not args.no_thinking_params:
                fb["enable_thinking"] = False
            fj = post(fb)
            if "error" not in fj:
                cont = fj["choices"][0]["message"].get("content") or ""
                m = re.match(r"\s*(\d+)", cont)
                if m and 1 <= int(m.group(1)) <= N:
                    choice = int(m.group(1))
                f_in = (fj.get("usage") or {}).get("prompt_tokens") or 0
            forced = True
        row.update(choice=choice, forced=forced, correct=bool(choice is not None and s["cor"][str(order[choice - 1])]),
                   finish=ch.get("finish_reason"), n_prompt=(u.get("prompt_tokens") or 0) + f_in, n_out=u.get("completion_tokens"),
                   n_reason=(u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                   content=ans[-2000:], reasoning=reasoning)
    with lock:
        with open(args.out, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tot["n"] += 1
        tot["err"] += "error" in row
        tot["in"] += row.get("n_prompt") or 0
        tot["out"] += row.get("n_out") or 0
        if tot["in"] > args.max_in_m * 1e6 or tot["out"] > args.max_out_m * 1e6:
            if not stop.is_set():
                print(f"token cap reached (in {tot['in']/1e6:.1f}M, out {tot['out']/1e6:.1f}M): no new submissions", flush=True)
            stop.set()
    return row


t_start = time.time()
with cf.ThreadPoolExecutor(args.concurrency) as ex:
    futs = [ex.submit(run, c) for c in calls]
    for i, f in enumerate(cf.as_completed(futs), 1):
        f.result()
        if i % 25 == 0 or i == len(futs):
            el = time.time() - t_start
            print(f"{i}/{len(futs)} {el/60:.0f} min | errors {tot['err']} | tokens in {tot['in']/1e6:.2f}M out {tot['out']/1e6:.2f}M"
                  f" | ETA {el/i*(len(futs)-i)/60:.0f} min", flush=True)
print("DONE", flush=True)
