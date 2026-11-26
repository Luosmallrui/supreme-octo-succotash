"""Run the E2 crossed position/printed-label experiment on a Qwen3 model."""
import argparse
import fcntl
import json
import math
import os
import re
from pathlib import Path

from common import MODELS, PREFILL, build_chat, build_user_content, last_boxed, load_jsonl


HERE = Path(__file__).resolve().parent
THINK_BUDGET = 12288


def user_prompt(row, problem, candidates):
    order, labels, n = row["order"], row["labels"], row["N"]
    if n != 8 or sorted(labels) != list(range(1, n + 1)) or len(set(order)) != n or len(order) != n:
        raise ValueError(f"Invalid E2 layout for {row['request_id']}")
    texts = [candidates[(row["pid"], k)] for k in order]
    content, spans = build_user_content(problem, texts, "A")
    if row["task"] == "select":
        instruction = "Which solution is correct? Return the printed Solution label of the best solution."
    elif row["task"] == "lookup" and row["lookup_slot"] in range(1, n + 1):
        instruction = (f"Count the candidate solutions from top to bottom, starting at 1. "
                       f"What printed Solution label appears on the candidate at position {row['lookup_slot']}?")
    else:
        raise ValueError(f"Unknown task or invalid lookup slot in {row['request_id']}")
    instruction += (f" Labels may appear in any order. Reply with only the printed number in the form "
                    f"\\boxed{{k}}, where k is an integer from 1 to {n}.")
    parts = []
    for name, start, end in spans:
        if name.startswith("h"):
            parts.append(f"Solution {labels[int(name[1:]) - 1]}:\n")
        elif name == "inst":
            parts.append(instruction)
        else:
            parts.append(content[start:end])
    return "".join(parts)


def prefill(task):
    return PREFILL["A"] if task == "select" else "The printed label is Solution "


def parse_think(text, n):
    split = text.partition("</think>")
    boxed = last_boxed(split[2]) if split[1] else None
    if boxed is not None and re.fullmatch(r"[1-9]", boxed) and int(boxed) <= n:
        return int(boxed)
    return None


def label_distribution(logprobs, n):
    p = [0.0] * n
    for obj in logprobs.values():
        token = (obj.decoded_token or "").strip()
        if re.fullmatch(r"[1-9]", token) and int(token) <= n:
            p[int(token) - 1] += math.exp(obj.logprob)
    mass = sum(p)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("No finite positive probability mass on any printed label")
    return [x / mass for x in p], mass


def score_labels(llm, prompts, n, params):
    outs = llm.generate(prompts, params, use_tqdm=False)
    return [label_distribution(o.outputs[0].logprobs[0], n) for o in outs]


def make_result(row, mode, label, **meta):
    slot = row["labels"].index(label) + 1
    candidate = row["order"][slot - 1]
    correct = (slot == row["lookup_slot"] if row["task"] == "lookup"
               else bool(row["cor"][str(candidate)]))
    return dict(row, mode=mode, choice_label=label, choice_slot=slot,
                choice_candidate=candidate, correct=correct, **meta)


def run_batch(rows, mode, probs, cands, llm, tok, sampling_params, scorer):
    think = mode == "think"
    prompts = [
        build_chat(tok, user_prompt(r, probs[r["pid"]], cands), enable_thinking=think,
                   prefill="" if think else prefill(r["task"]))
        for r in rows
    ]
    if not think:
        distributions = score_labels(llm, prompts, 8, scorer)
        return [make_result(r, mode, p.index(max(p)) + 1, p_label=p, p_mass=mass,
                            forced=False, finish="scored", n_out=0)
                for r, (p, mass) in zip(rows, distributions, strict=True)]

    params = [sampling_params(temperature=0.6, top_p=0.95, top_k=20, seed=r["seed"],
                              max_tokens=THINK_BUDGET) for r in rows]
    generated = [o.outputs[0] for o in llm.generate(prompts, params, use_tqdm=False)]
    parsed = [parse_think(g.text, 8) for g in generated]
    fallback = {}
    missing = [i for i, label in enumerate(parsed) if label is None]
    if missing:
        forced_prompts = []
        for i in missing:
            text = generated[i].text
            closing = "\n</think>" if "</think>" not in text else ""
            forced_prompts.append(prompts[i] + text + closing + "\n\n" + prefill(rows[i]["task"]))
        fallback = dict(zip(missing, score_labels(llm, forced_prompts, 8, scorer), strict=True))

    results = []
    for i, (r, g, label) in enumerate(zip(rows, generated, parsed, strict=True)):
        p, mass = fallback.get(i, (None, None))
        choice = label if label is not None else p.index(max(p)) + 1
        results.append(make_result(r, mode, choice, choice_gen=label, p_label=p, p_mass=mass,
                                   gen=g.text, forced=label is None, finish=g.finish_reason,
                                   n_out=len(g.token_ids)))
    return results


def recover_results(handle, wanted, mode):
    """Validate saved rows, and repair a torn final write before appending."""
    handle.seek(0)
    done = set()
    while True:
        at = handle.tell()
        line = handle.readline()
        if not line:
            break
        try:
            result = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if not line.endswith(b"\n"):
                handle.seek(at)
                handle.truncate()
                print("Discarded incomplete final row", flush=True)
                break
            raise
        key = result["request_id"]
        if (key in done or key not in wanted or result.get("mode") != mode
                or any(result.get(k) != v for k, v in wanted[key].items())):
            raise ValueError(f"Output does not match this manifest/mode, or duplicates {key}")
        done.add(key)
        if not line.endswith(b"\n"):
            handle.write(b"\n")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--mode", choices=("direct", "think"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    rows = [r for r in load_jsonl(args.manifest) if r["model"] == args.model]
    wanted = {r["request_id"]: r for r in rows}
    if not rows or len(wanted) != len(rows):
        raise ValueError("Manifest has no rows for the model or contains duplicate request IDs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a+b") as sink:
        fcntl.flock(sink, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = recover_results(sink, wanted, args.mode)
        pending = [r for r in rows if r["request_id"] not in done]
        print(f"{args.model} {args.mode}: {len(done)}/{len(rows)} already saved", flush=True)
        if not pending:
            return
        probs = {p["pid"]: p["problem"] for p in json.loads((HERE / "data/problems.json").read_text())}
        cands = {(r["pid"], r["k"]): r["text"]
                 for r in load_jsonl(HERE / "out" / f"graded_{args.model}.jsonl")}
        from vllm import LLM, SamplingParams

        llm = LLM(model=MODELS[args.model], max_model_len=40960, gpu_memory_utilization=0.92,
                  enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0)
        scorer = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        tok = llm.get_tokenizer()
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            batch = run_batch(chunk, args.mode, probs, cands, llm, tok, SamplingParams, scorer)
            for r in batch:
                sink.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
            sink.flush()
            os.fsync(sink.fileno())
            print(f"Saved {len(done) + start + len(batch)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
