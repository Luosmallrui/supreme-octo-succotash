"""E1 / E2 candidate selection with a hosted judge on an OpenAI-compatible Chat Completions API.

The judge sees the same Qwen3-8B candidate sets, orderings and user prompts as the local runs. Most hosted
chat APIs do not accept an assistant pre-fill, so by default the judge writes \boxed{k} itself.

direct  Reasoning off where the model allows it (--effort none). The prompt already asks for only \\boxed{k}.
        When the API returns logprobs, the label distribution is read at the first digit token inside the
        reply's \\boxed{} and the choice is its argmax, as in the local pre-filled runs. Otherwise the choice
        is the parsed reply. The distribution covers only the --top-logprobs most likely tokens (OpenAI
        allows at most 5 here), renormalised; p_mass records the raw mass captured. For N >= 10 it is
        also approximate because a leading "1" token can start 10-19.
think   Reasoning on (--effort, default medium). The choice is the last \\boxed{k} in the visible reply.
        A reply without one gets one follow-up turn that asks only for the label (forced=True). The hidden
        reasoning is not carried into that turn.
--prefill  (direct only; DeepSeek beta "chat prefix completion" at <base-url>/beta) pre-fills the assistant
        turn exactly as the local Qwen3 runs do ("The best solution is Solution ", or "The printed label is
        Solution " for E2 lookup) and reads the label distribution at the first generated token.

Tested providers: OpenAI (default --base-url) and DeepSeek (--base-url https://api.deepseek.com
--key-file deepseek.txt --top-logprobs 20 --max-tokens-param max_tokens). Without a pre-fill, DeepSeek in
non-thinking mode writes a full visible analysis before \boxed{k}, so use --prefill for its direct mode. A quota or balance error (429 insufficient quota, 402) stops the
whole run; error rows are retried on the next start.

Rows are appended as they return. A restart skips saved request IDs, retries rows saved with an error,
and refuses to mix rows produced with different API settings. The key comes from --key-file, else from
the variable named by --key-env (default OPENAI_API_KEY); it is never printed or written. Token caps stop new submissions.

Outputs are read by the existing analyses:
  E1  out/e1_api-<tag>-<mode>_qwen3-8b.jsonl  -> e1_analyze.py --data out (mode "api-<tag>-<mode>")
  E2  out/e2_position_label/api/e2_<tag>_<mode>.jsonl -> e2_analyze.py --input <file> (model = <tag>)
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import PREFILL, build_user_content, last_boxed  # noqa: E402
from e2_run import make_result as e2_make_result, prefill as e2_prefill, user_prompt as e2_user_prompt  # noqa: E402

LABEL = re.compile(r"\s*(?:\\text\{)?\s*(?:Solution\s*)?#?\s*(\d+)\s*\}?\s*")
TOKEN_LABEL = re.compile(r"\s*(\d+)\s*\}?\s*")
FOLLOW_UP = ("Reply with only your final answer to the question above in the form \\boxed{{k}}, "
             "where k is an integer from 1 to {n}.")
DEFAULTS = {"direct": dict(effort="none", max_tokens=64, logprobs=True),
            "think": dict(effort="medium", max_tokens=32768, logprobs=False)}
RETRY = (408, 409, 429, 500, 502, 503, 504)
QUOTA = ("insufficient_quota", "no credits remaining", "exceeded your current quota")  # 429s that never clear



def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def boxed_label(text, n):
    """Label in the last \\boxed{} of text if it is an integer in 1..n (no bare-number fallback)."""
    b = last_boxed(text or "")
    if b is not None:
        m = LABEL.fullmatch(b)
        if m and 1 <= int(m.group(1)) <= n:
            return int(m.group(1))
    return None


def label_distribution(tokens, text, n):
    """(p over labels 1..n, raw mass) at the first digit token inside the last \\boxed{} of text, or (None, None)."""
    if not tokens or "".join(t["token"] for t in tokens) != text:
        return None, None
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None, None
    digit = re.compile(r"\d").search(text, start)
    if digit is None:
        return None, None
    pos = 0
    for t in tokens:
        end = pos + len(t["token"])
        if end > digit.start():
            break
        pos = end
    alts = {a["token"]: a["logprob"] for a in t.get("top_logprobs") or []}
    alts.setdefault(t["token"], t["logprob"])
    p = [0.0] * n
    for tok, lp in alts.items():
        m = TOKEN_LABEL.fullmatch(tok)
        if m and 1 <= int(m.group(1)) <= n:
            p[int(m.group(1)) - 1] += math.exp(lp)
    mass = sum(p)
    if not math.isfinite(mass) or mass <= 0:
        return None, None
    return [x / mass for x in p], mass


def first_token_distribution(tokens, n):
    """(p over labels 1..n, raw mass) at the first non-blank generated token after a pre-fill, or (None, None)."""
    t = next((t for t in tokens or [] if t["token"].strip()), None)
    if t is None:
        return None, None
    alts = {a["token"]: a["logprob"] for a in t.get("top_logprobs") or []}
    alts.setdefault(t["token"], t["logprob"])
    p = [0.0] * n
    for tok, lp in alts.items():
        m = TOKEN_LABEL.fullmatch(tok)
        if m and 1 <= int(m.group(1)) <= n:
            p[int(m.group(1)) - 1] += math.exp(lp)
    mass = sum(p)
    if not math.isfinite(mass) or mass <= 0:
        return None, None
    return [x / mass for x in p], mass


def read_key(key_file, key_env):
    """The key file wins, so a key set for another provider in the environment is never sent here."""
    key = Path(key_file).read_text(encoding="utf-8").strip() if key_file else os.environ.get(key_env, "").strip()
    if not key:
        sys.exit(f"Pass --key-file or set {key_env}")
    return key


class Client:
    def __init__(self, base_url, key, timeout, halt):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.beta_url = base_url.rstrip("/") + "/beta/chat/completions"  # DeepSeek chat prefix completion
        self.key = key
        self.timeout = timeout
        self.halt = halt  # set when the run stops; waiting retries give up at once

    def post(self, body, beta=False):
        url = self.beta_url if beta else self.url
        data = json.dumps(body).encode()
        last = "no attempt"
        for attempt in range(8):
            if self.halt.is_set():
                return {"error": f"run stopped before retry: {last}", "fatal": True}
            req = urllib.request.Request(url, data=data, method="POST",
                                         headers={"Authorization": "Bearer " + self.key,
                                                  "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:300]
                if e.code == 402 or (e.code == 429 and any(q in msg.lower() for q in QUOTA)):
                    return {"error": f"{e.code} {msg}", "status": e.code, "fatal": True}
                if e.code not in RETRY:
                    return {"error": f"{e.code} {msg}", "status": e.code}
                last = f"{e.code} {msg}"
                wait = e.headers.get("retry-after")
                self.halt.wait(min(300, float(wait) if wait and wait.replace(".", "").isdigit() else 10 * 2 ** attempt))
            except Exception as e:  # timeouts, connection resets
                last = repr(e)
                self.halt.wait(min(300, 10 * 2 ** attempt))
        return {"error": f"gave up after retries: {last}"}


def build_calls(args, probs, cands):
    """List of (request_id, n, prompt, meta) in submission order."""
    if args.exp == "e2":
        rows = load_jsonl(args.input)
        if args.cand_model:
            rows = [r for r in rows if r["model"] == args.cand_model]
        rows = [r for r in rows if r["task"] in args.tasks.split(",")]
        if args.blocks:
            blocks = defaultdict(list)
            for r in rows:
                key = (r["set_id"], r["stratum"], r["task"], r["probe_slot"], r["a"], r["b"], r["lookup_slot"])
                blocks[key].append(r)
            keys = sorted(blocks)
            random.Random(args.sample_seed).shuffle(keys)
            rows = [r for k in keys[:args.blocks] for r in blocks[k]]
        calls = [(r["request_id"], r["N"], e2_user_prompt(r, probs[r["pid"]], cands), r) for r in rows]
    else:
        sets = load_jsonl(args.input)
        if args.filter:
            sets = [s for s in sets if eval(args.filter, {}, {"s": s})]
        calls = []
        for s in sets:
            for name, order in s["orders"].items():
                if any(name.startswith(p) for p in args.orders.split(",")):
                    content, _ = build_user_content(probs[s["pid"]], [cands[(s["pid"], k)] for k in order], "A")
                    calls.append((f"{s['set_id']}/{name}", s["N"], content, dict(s=s, name=name, order=order)))
    random.Random(0).shuffle(calls)
    if args.limit:
        calls = calls[:args.limit]
    return calls


def request_body(args, messages, effort, max_tokens, logprobs):
    body = {"model": args.model, "messages": messages, args.max_tokens_param: max_tokens}
    if effort != "omit":
        body["reasoning_effort"] = effort
    if args.temperature is not None:
        body["temperature"] = args.temperature
    if logprobs:
        body.update(logprobs=True, top_logprobs=args.top_logprobs)
    return body


def ask(args, client, prompt, n, prefix=None):
    """One selection: returns dict with choice, choice_gen, p, mass, forced, usage and text fields.

    With prefix, the assistant turn is pre-filled (DeepSeek beta prefix completion) and the label is read at
    the first generated token, as in the local pre-filled runs."""
    t0 = time.time()
    messages = [{"role": "user", "content": prompt}]
    if prefix is not None:
        messages.append({"role": "assistant", "content": prefix, "prefix": True})
    js = client.post(request_body(args, messages, args.effort, args.max_tokens, args.logprobs),
                     beta=prefix is not None)
    out = dict(latency=None, forced=False, n_prompt=0, n_cached=0, n_out=0, n_reason=0)
    if "error" in js:
        out.update(error=js["error"], fatal=js.get("fatal", False), latency=round(time.time() - t0, 1))
        return out
    ch = js["choices"][0]
    text = ch["message"].get("content") or ""
    lp = (ch.get("logprobs") or {}).get("content")
    if prefix is not None:
        m = re.match(r"\s*(\d+)", text)
        choice_gen = int(m.group(1)) if m and 1 <= int(m.group(1)) <= n else None
        p, mass = first_token_distribution(lp, n) if args.logprobs else (None, None)
    else:
        choice_gen = boxed_label(text, n)
        p, mass = label_distribution(lp, text, n) if args.logprobs else (None, None)
    usage = [js.get("usage") or {}]
    forced_text = None
    if p is not None:
        choice = p.index(max(p)) + 1
    else:
        choice = choice_gen
    if choice is None and args.mode == "think":
        follow = FOLLOW_UP.format(n=n)
        if text.strip():
            messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": text},
                        {"role": "user", "content": follow}]
        else:
            messages = [{"role": "user", "content": prompt + "\n\n" + follow}]
        fj = client.post(request_body(args, messages, args.forced_effort, args.forced_max_tokens, False))
        if "error" in fj:  # saved as an error row, so the call is retried on the next start
            out["error"] = out["forced_error"] = fj["error"]
            out["fatal"] = fj.get("fatal", False)
        else:
            forced_text = fj["choices"][0]["message"].get("content") or ""
            choice = boxed_label(forced_text, n)
            usage.append(fj.get("usage") or {})
        out["forced"] = True
    for u in usage:
        out["n_prompt"] += u.get("prompt_tokens") or 0
        out["n_cached"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        out["n_out"] += u.get("completion_tokens") or 0
        out["n_reason"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    out.update(choice=choice, choice_gen=choice_gen, p=p, mass=mass, finish=ch.get("finish_reason"),
               resolved_model=js.get("model"), content=text[-4000:], forced_content=forced_text,
               latency=round(time.time() - t0, 1))
    return out


def make_row(args, settings, meta, got):
    common = dict(api_model=args.model, api_settings=settings, resolved_model=got.get("resolved_model"),
                  date=datetime.date.today().isoformat(), forced=got["forced"], finish=got.get("finish"),
                  n_prompt=got["n_prompt"], n_cached=got["n_cached"], n_out=got["n_out"], n_reason=got["n_reason"],
                  latency=got["latency"], content=got.get("content"), forced_content=got.get("forced_content"))
    for k in ("error", "forced_error"):
        if k in got:
            common[k] = got[k]
    choice = got.get("choice")
    if args.exp == "e2":
        row = dict(meta, model=args.tag, cand_model=meta["model"])
        meta_out = dict(common, choice_gen=got.get("choice_gen"), p_label=got.get("p"), p_mass=got.get("mass"))
        if choice is None:
            return dict(row, mode=args.mode, choice_label=None, choice_slot=None, choice_candidate=None,
                        correct=False, **meta_out)
        return e2_make_result(row, args.mode, choice, **meta_out)
    s, order = meta["s"], meta["order"]
    return dict(request_id=f"{s['set_id']}/{meta['name']}", set_id=s["set_id"], pid=s["pid"], source=s["source"],
                N=s["N"], n_correct=s["n_correct"], order_name=meta["name"], order=order,
                mode=f"api-{args.tag}-{args.mode}", p=got.get("p"), p_mass=got.get("mass"), choice=choice,
                choice_gen=got.get("choice_gen"),
                correct=bool(choice is not None and s["cor"][str(order[choice - 1])]), **common)


def load_done(path, settings):
    """Request IDs already saved without error; repairs a torn final line; refuses mixed settings."""
    done = set()
    if not path.exists():
        return done
    raw = path.read_bytes()
    head, _, tail = raw.rpartition(b"\n")
    if tail.strip():  # last write has no newline: keep it if it parses, else drop it
        try:
            json.loads(tail)
            raw += b"\n"
        except ValueError:
            raw = head + b"\n" if head else b""
            print("Discarded incomplete final row", flush=True)
        path.write_bytes(raw)
    for line in raw.split(b"\n"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("api_settings") != settings:
            sys.exit(f"{path} holds rows made with other API settings: {r.get('api_settings')}")
        if "error" not in r:
            done.add(r["request_id"])
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", choices=("e1", "e2"), required=True)
    ap.add_argument("--model", required=True, help="API model ID, e.g. gpt-5.4-mini")
    ap.add_argument("--tag", help="short judge name in outputs (default: the model ID)")
    ap.add_argument("--mode", choices=("direct", "think"), required=True)
    ap.add_argument("--input", type=Path, help="E2 manifest or E1 sets (default per --exp)")
    ap.add_argument("--cands", type=Path, default=HERE / "out/graded_qwen3-8b.jsonl")
    ap.add_argument("--cand-model", default="qwen3-8b", help="E2: manifest rows for this candidate generator")
    ap.add_argument("--tasks", default="select,lookup", help="E2 tasks")
    ap.add_argument("--blocks", type=int, default=0, help="E2: run only this many random complete blocks")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--orders", default="r,maj", help="E1 ordering-name prefixes")
    ap.add_argument("--filter", default="", help="E1 set filter, a Python expression over s")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--effort", help="reasoning_effort, or 'omit' for models without it (default: none / medium)")
    ap.add_argument("--max-tokens", type=int, help="max_completion_tokens, reasoning included (default 64 / 32768)")
    ap.add_argument("--logprobs", choices=("on", "off"), help="request logprobs (default on for direct)")
    ap.add_argument("--top-logprobs", type=int, default=5, help="alternatives per token (OpenAI maximum: 5)")
    ap.add_argument("--max-tokens-param", choices=("max_completion_tokens", "max_tokens"),
                    default="max_completion_tokens", help="DeepSeek honours only max_tokens")
    ap.add_argument("--temperature", type=float, help="omit by default; several reasoning models reject it")
    ap.add_argument("--forced-effort", default="none", help="reasoning_effort for the think-mode follow-up turn")
    ap.add_argument("--forced-max-tokens", type=int, default=256)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--key-file", help="file holding the API key (takes precedence over --key-env)")
    ap.add_argument("--key-env", default="OPENAI_API_KEY", help="environment variable holding the key")
    ap.add_argument("--prefill", action="store_true",
                    help="direct only: pre-fill the answer via DeepSeek beta prefix completion (tag gets -prefill)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--max-in-m", type=float, default=5.0, help="stop submitting after this many M prompt tokens")
    ap.add_argument("--max-out-m", type=float, default=2.0, help="stop submitting after this many M output tokens")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the call count and prompt size, call nothing")
    args = ap.parse_args()
    if args.prefill and args.mode != "direct":
        ap.error("--prefill is for --mode direct")
    d = DEFAULTS[args.mode]
    args.effort = args.effort or d["effort"]
    args.max_tokens = args.max_tokens or (4 if args.prefill else d["max_tokens"])
    args.logprobs = d["logprobs"] if args.logprobs is None else args.logprobs == "on"
    args.tag = args.tag or (args.model + ("-prefill" if args.prefill else ""))
    if args.input is None:
        args.input = HERE / ("out/e2_position_label/manifest_qwen3-8b.jsonl" if args.exp == "e2"
                             else "out/e1_sets_qwen3-8b.jsonl")
    if args.out is None:
        args.out = HERE / (f"out/e2_position_label/api/e2_{args.tag}_{args.mode}.jsonl" if args.exp == "e2"
                           else f"out/e1_api-{args.tag}-{args.mode}_qwen3-8b.jsonl")
    settings = dict(base_url=args.base_url, model=args.model, mode=args.mode, effort=args.effort,
                    max_tokens=args.max_tokens, logprobs=args.logprobs and args.top_logprobs,
                    temperature=args.temperature,
                    forced_effort=args.forced_effort, cands=args.cands.name)
    if args.prefill:  # only added when used, so rows saved before these options existed still match
        settings["prefill"] = True
    if args.max_tokens_param != "max_completion_tokens":
        settings["max_tokens_param"] = args.max_tokens_param

    probs = {p["pid"]: p["problem"] for p in json.loads((HERE / "data/problems.json").read_text(encoding="utf-8"))}
    cands = {(r["pid"], r["k"]): r["text"] for r in load_jsonl(args.cands)}
    calls = build_calls(args, probs, cands)
    if len({c[0] for c in calls}) != len(calls):
        sys.exit("duplicate request IDs in the selected calls")
    done = load_done(args.out, settings)
    todo = [c for c in calls if c[0] not in done]
    chars = sum(len(c[2]) for c in todo)
    print(f"{args.exp} {args.mode} {args.model}: {len(todo)} calls to run, {len(calls) - len(todo)} already saved; "
          f"~{chars / 4 / 1e6:.2f}M prompt tokens (chars/4) -> {args.out}", flush=True)
    if args.dry_run or not todo:
        return

    stop = threading.Event()
    client = Client(args.base_url, read_key(args.key_file, args.key_env), args.timeout, stop)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(args.out) + ".lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"{lock_path} exists: another run is writing {args.out} (delete the lock if it is stale)")
    lock = threading.Lock()
    tot = dict(n=0, err=0, ok=0, inp=0, out=0, reason=0, cached=0, forced=0, invalid=0)

    def run(call):
        if stop.is_set():
            return
        rid, n, prompt, meta = call
        try:
            prefix = None
            if args.prefill:
                prefix = e2_prefill(meta["task"]) if args.exp == "e2" else PREFILL["A"]
            got = ask(args, client, prompt, n, prefix)
        except Exception as e:  # recorded and retried on the next start
            got = dict(error=f"exception: {e!r}", forced=False, n_prompt=0, n_cached=0, n_out=0, n_reason=0,
                       latency=None)
        row = make_row(args, settings, meta, got)
        with lock:
            with args.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            tot["n"] += 1
            tot["err"] += "error" in row
            tot["ok"] += "error" not in row
            tot["inp"] += row["n_prompt"]
            tot["out"] += row["n_out"]
            tot["reason"] += row["n_reason"]
            tot["cached"] += row["n_cached"]
            tot["forced"] += bool(row["forced"])
            tot["invalid"] += "error" not in row and row.get("choice", row.get("choice_label")) is None
            if got.get("fatal") and not stop.is_set():
                print(f"account cannot make more calls, stopping (error rows are retried on restart): "
                      f"{row.get('error') or row.get('forced_error')}", flush=True)
                stop.set()
            if tot["err"] >= 5 and tot["ok"] == 0 and not stop.is_set():
                print(f"first {tot['err']} calls all failed, stopping: {row.get('error')}", flush=True)
                stop.set()
            if (tot["inp"] > args.max_in_m * 1e6 or tot["out"] > args.max_out_m * 1e6) and not stop.is_set():
                print(f"token cap reached (in {tot['inp'] / 1e6:.2f}M, out {tot['out'] / 1e6:.2f}M): "
                      "no new submissions", flush=True)
                stop.set()

    t0 = time.time()
    try:
        with cf.ThreadPoolExecutor(args.concurrency) as ex:
            futs = [ex.submit(run, c) for c in todo]
            for i, f in enumerate(cf.as_completed(futs), 1):
                f.result()
                if i % 10 == 0 or i == len(futs):
                    el = time.time() - t0
                    print(f"{i}/{len(futs)} {el / 60:.1f} min | errors {tot['err']} forced {tot['forced']} "
                          f"no-choice {tot['invalid']} | tokens in {tot['inp'] / 1e6:.2f}M "
                          f"(cached {tot['cached'] / 1e6:.2f}M) out {tot['out'] / 1e6:.3f}M "
                          f"(reasoning {tot['reason'] / 1e6:.3f}M)", flush=True)
    finally:
        os.close(lock_fd)
        lock_path.unlink()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
