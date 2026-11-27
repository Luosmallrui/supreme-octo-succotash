"""E3 readout, heading, and question-only controls on a fixed candidate layout.

Protocol v2 (2026-11-26). Changes from v1, which lost most responses to formatting rather than to the model:
  direct  The assistant turn is pre-filled with "FINAL:" (as E1/E2 pre-fill the index), so the readout is
          immediate instead of a 256-token start of a solution. The response is the first non-empty line.
  think   Same prompt, budget and greedy decoding as v1, so v1 generations can be reused (--reuse). If the
          trace has no usable final line, the run gets a forced ending ("</think>" if needed, then "FINAL:")
          and is flagged forced=True, as in E1/E2.
  parse   "FINAL:" may carry markdown (**FINAL: 3**); a label may be written "Solution 3" or \\boxed{3};
          math answers and labels fall back to the last \\boxed{} after the trace (parse_rule="boxed").
  quote   Verbatim matching ignores whitespace, "$", "*" and "`". Paraphrased quotes are attributed by
          distinctive 20-character shingles (quote_match="shingle"); analyses report strict and lenient.
"""
import argparse
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from common import MODELS, build_chat, last_boxed, load_jsonl


HERE = Path(__file__).resolve().parent
BUDGET = {"direct": 128, "think": 12288}
FORCED_BUDGET = 128
PREFILL = "FINAL:"
FORMAT = '\nGive the response on the final line in the format FINAL: <response>.'
FINAL = re.compile(r"FINAL\s*(?:\*\*|__)?\s*[:：]")
SHINGLE = 20


def whitespace(text):
    return " ".join(text.split())


def user_prompt(row, problem, candidates):
    problem_line = f"Problem:\n{problem.strip()}"
    readout = row["readout"]
    if readout == "solo_answer":
        return problem_line + "\n\nSolve the problem and return its final mathematical answer." + FORMAT

    order, labels, n = row["order"], row["labels"], row["N"]
    if len(order) != n or len(set(order)) != n or sorted(labels) != list(range(1, n + 1)):
        raise ValueError(f"Invalid layout for {row['request_id']}")
    chunks = [problem_line,
              f"Below are {n} candidate solutions to this problem. Some of them may be incorrect."]
    for k, label in zip(order, labels):
        if row["header"] == "numbered":
            header = f"Solution {label}:\n"
        elif row["header"] == "plain":
            header = "-----\n"
        elif row["header"] == "none":
            header = ""
        else:
            raise ValueError(f"Invalid header for {row['request_id']}")
        text = candidates[row["pid"], k].strip()
        chunks.append(header + text)
    if readout == "label" and row["header"] == "numbered":
        instruction = ("Choose the best solution. Return only its printed Solution number. "
                       "The printed numbers may appear in any order.")
    elif readout == "quote":
        instruction = ("Choose the best solution. Copy a short passage verbatim from its text that uniquely "
                       "identifies it (aim for 6 to 16 words). Include some of its working, not just its final answer.")
    elif readout == "lookup_quote":
        cue = whitespace(row["anchor_cues"][str(row["lookup_candidate"])])
        instruction = (f"Locate the solution that contains this passage verbatim:\n{cue}\n"
                       "Copy a different, non-overlapping passage verbatim from that solution that uniquely "
                       "identifies it (aim for 6 to 16 words). Include some of its working, not just its final answer.")
    elif readout == "answer":
        instruction = "Use the candidate solutions to determine the final mathematical answer to the problem."
    else:
        raise ValueError(f"Unsupported readout/heading for {row['request_id']}")
    return "\n\n".join(chunks + [instruction + FORMAT])


# ---------------------------------------------------------------------------
# Extracting the response
# ---------------------------------------------------------------------------

def first_line(text):
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def clean(text):
    """Strip markdown emphasis and stray punctuation around a one-line response."""
    if text is None:
        return None
    text = text.strip()
    for _ in range(3):
        before = text
        for mark in ("**", "__", "`"):
            if text.startswith(mark):
                text = text[len(mark):]
            if text.endswith(mark):
                text = text[:-len(mark)]
        text = text.strip().rstrip(".").strip()
        if text == before:
            break
    return text or None


def answer_part(generated):
    """Text after the reasoning trace, or None when the trace never closed."""
    if "</think>" in generated:
        return generated.rsplit("</think>", 1)[1]
    if "<think>" in generated:
        return None
    return generated


def final_line(text):
    """Response after the last FINAL marker (rest of its line, else the next non-empty line)."""
    matches = list(FINAL.finditer(text))
    if not matches:
        return None
    return clean(first_line(text[matches[-1].end():]) or "")


def extract(generated, mode, readout):
    """(response, parse_rule) from a generation, or (None, None) when a forced ending is needed.

    direct: the generation continues the pre-filled "FINAL:", so the response is its first line.
    think:  last FINAL line after </think>; labels and math answers may instead end in \\boxed{}.
    """
    if mode == "direct":
        line = first_line(generated)
        if line is not None and FINAL.match(line):
            line = line[FINAL.match(line).end():]
        return clean(line), "prefill"
    tail = answer_part(generated)
    if tail is None:
        return None, None
    final = final_line(tail)
    if final is not None:
        return final, "final"
    if readout in ("label", "answer", "solo_answer") and last_boxed(tail) is not None:
        return clean(last_boxed(tail)), "boxed"
    return None, None


def forced_suffix(generated):
    return ("" if "</think>" in generated else "\n</think>") + "\n\n" + PREFILL


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

LABEL = re.compile(r"(?:\\boxed\{\s*)?(?:\\text\{\s*)?(?:Solution\s*)?#?\s*(\d+)\s*\}?\s*\}?")


def parse_label(final, labels):
    text = final.replace("$", "").replace("*", "").strip()
    m = LABEL.fullmatch(text)
    if not m:
        return None
    number = int(m.group(1))
    return number if number in labels else None


@lru_cache(maxsize=1024)
def parse_gold(gold):
    from math_verify import parse
    parsed = parse("$" + gold + "$")
    if not parsed:
        raise ValueError(f"Unparseable reference answer {gold!r}")
    return parsed


def math_text(final):
    """The math expression in a response: last \\boxed{} if present, else the response without $ wrappers."""
    boxed = last_boxed(final)
    text = (boxed if boxed is not None else final).strip()
    for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
            break
    return text


def grade_math(final, gold):
    from math_verify import parse, verify
    expected = parse_gold(gold)
    text = math_text(final)
    try:
        candidate = parse(r"\boxed{" + text + "}")
        return bool(candidate), bool(candidate and verify(expected, candidate))
    except Exception:
        # Preserve the generation and count a failed parse as a failed trial.
        return False, False


QUOTE_WRAPPERS = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("«", "»"), ("`", "`")]


def squash(text):
    """Case-, whitespace- and markup-insensitive form used for verbatim matching."""
    return re.sub(r"[\s$*`]+", "", text).lower()


def shingles(text):
    return {text[i:i + SHINGLE] for i in range(len(text) - SHINGLE + 1)}


def match_quote(final, row, candidates):
    """(slot index, None, match kind) or (None, error, None).

    exact:   the squashed excerpt occurs in exactly one displayed candidate.
    shingle: >= 50% of the excerpt's 20-char shingles occur in some candidate, >= 10 shingles occur in exactly
             one candidate, and one candidate holds >= 90% of those distinctive shingles. Thresholds were fixed on
             synthetic two-word-deletion paraphrases of displayed 8B candidates (attributed 50%, precision 0.998)
             before any v2 output existed.
    For lookup_quote, shingles inside the cue are discarded first, and an exact match must lie mostly outside it.
    """
    excerpt = whitespace(final)
    for left, right in QUOTE_WRAPPERS:
        if len(excerpt) >= 2 and excerpt.startswith(left) and excerpt.endswith(right):
            excerpt = excerpt[1:-1].strip()
            break
    q = squash(excerpt)
    if len(q) < SHINGLE or len(excerpt.split()) < 4:
        return None, "too_short", None
    bodies = [squash(candidates[row["pid"], k]) for k in row["order"]]
    cue = None
    if row["readout"] == "lookup_quote":
        cue = squash(row["anchor_cues"][str(row["lookup_candidate"])])
        if not cue or cue not in bodies[row["order"].index(row["lookup_candidate"])]:
            raise ValueError(f"Missing lookup cue in original text for {row['request_id']}")
    own = shingles(q)
    if cue:
        own -= shingles(cue)
        if len(own) < 0.5 * max(1, len(q) - SHINGLE + 1):
            return None, "overlaps_cue", None
    hits = [i for i, body in enumerate(bodies) if q in body]
    if len(hits) == 1:
        return hits[0], None, "exact"
    if len(hits) > 1:
        return None, "ambiguous", None
    found = distinctive = 0
    owner = [0] * len(bodies)
    for s in own:
        where = [i for i, body in enumerate(bodies) if s in body]
        found += bool(where)
        if len(where) == 1:
            distinctive += 1
            owner[where[0]] += 1
    if not own or found < 0.5 * len(own) or distinctive < 10:
        return None, "absent", None
    best = max(range(len(bodies)), key=owner.__getitem__)
    if owner[best] < 0.9 * distinctive:
        return None, "ambiguous", None
    return best, None, "shingle"


def result(row, mode, gen, finish, n_out, n_prompt, candidates, final, rule, forced, forced_gen=None,
           grader=grade_math):
    candidate = slot = label = match = None
    parsed = correct = False
    error = "missing_final" if final is None else None
    if final is not None:
        if row["readout"] == "label":
            number = parse_label(final, row["labels"])
            error = "invalid_label"
            if number is not None:
                slot = row["labels"].index(number)
                candidate = row["order"][slot]
                error = None
        elif row["readout"] in ("quote", "lookup_quote"):
            slot, error, match = match_quote(final, row, candidates)
            if slot is not None:
                candidate = row["order"][slot]
        else:
            parsed, correct = grader(final, row["gold"])
            if not parsed:
                error = "invalid_math_answer"
        if slot is not None:
            parsed = True
            label = row["labels"][slot]
            correct = (candidate == row["lookup_candidate"] if row["readout"] == "lookup_quote"
                       else bool(row["cor"][str(candidate)]))
    return dict(row, mode=mode, protocol=2, choice_candidate=candidate,
                choice_slot=slot + 1 if slot is not None else None, choice_label=label, correct=correct,
                parsed=parsed, parse_error=error, parse_rule=rule, quote_match=match, forced=forced,
                final=final, gen=gen, forced_gen=forced_gen, finish=finish, n_out=n_out, n_prompt=n_prompt,
                max_tokens=BUDGET[mode])


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def load_reuse(path, wanted):
    """v1 think generations keyed by request ID (same prompt, budget and greedy decoding)."""
    out = {}
    for r in load_jsonl(path):
        key = r["request_id"]
        if key not in wanted or r.get("mode") != "think":
            continue
        if any(r.get(k) != v for k, v in wanted[key].items()):
            raise ValueError(f"Reused row {key} differs from the manifest")
        out[key] = dict(gen=r["gen"], finish=r["finish"], n_out=r["n_out"], n_prompt=r["n_prompt"])
    return out


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
            saved = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if not line.endswith(b"\n"):
                handle.seek(at)
                handle.truncate()
                print("Discarded incomplete final row", flush=True)
                break
            raise
        key = saved["request_id"]
        if (key in done or key not in wanted or saved.get("mode") != mode or saved.get("protocol") != 2
                or any(saved.get(k) != v for k, v in wanted[key].items())):
            raise ValueError(f"Output does not match this manifest/mode/protocol, or duplicates {key}")
        done.add(key)
        if not line.endswith(b"\n"):
            handle.write(b"\n")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--mode", choices=BUDGET, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse", type=Path, help="think mode: v1 result file whose generations are reused")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sample", action="store_true",
                    help="think mode: Qwen3's recommended sampling (temperature 0.6, top-p 0.95, top-k 20) with each "
                         "row's seed instead of greedy decoding; rows are marked decoding=sampled")
    ap.add_argument("--readouts", help="comma list: run only these readouts, e.g. label,answer,solo_answer")
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    if args.sample and (args.mode != "think" or args.reuse):
        ap.error("--sample is for think mode and cannot reuse greedy v1 generations")
    if args.reuse and args.mode != "think":
        ap.error("--reuse only applies to think mode (direct prompts changed in protocol v2)")
    import fcntl

    rows = [r for r in load_jsonl(args.manifest) if r["model"] == args.model
            and (not args.readouts or r["readout"] in args.readouts.split(","))]
    wanted = {r["request_id"]: r for r in rows}
    if not rows or len(wanted) != len(rows):
        raise ValueError("No rows for this model or duplicate request IDs")
    reuse = load_reuse(args.reuse, wanted) if args.reuse and args.reuse.exists() else {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a+b") as sink:
        fcntl.flock(sink, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = recover_results(sink, wanted, args.mode)
        pending = [r for r in rows if r["request_id"] not in done]
        n_reuse = sum(r["request_id"] in reuse for r in pending)
        print(f"{args.model} {args.mode}: {len(done)}/{len(rows)} already saved, "
              f"{n_reuse} pending rows reuse v1 generations", flush=True)
        if not pending:
            return

        sys.path.insert(0, str(HERE / "pylib"))
        probs = {p["pid"]: p["problem"] for p in json.loads((HERE / "data/problems.json").read_text())}
        cands = {(r["pid"], r["k"]): r["text"]
                 for r in load_jsonl(HERE / "out" / f"graded_{args.model}.jsonl")}
        # Resolve math grading before allocating the GPU.
        for gold in {r["gold"] for r in pending if r["readout"] in ("answer", "solo_answer")}:
            parse_gold(gold)
        from vllm import LLM, SamplingParams

        llm = LLM(model=MODELS[args.model], max_model_len=40960, gpu_memory_utilization=0.92,
                  enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0)
        tok = llm.get_tokenizer()
        sampling = SamplingParams(temperature=0, max_tokens=BUDGET[args.mode])
        forcing = SamplingParams(temperature=0, max_tokens=FORCED_BUDGET)
        think = args.mode == "think"
        # Reused rows first: they only need the (cheap) forced endings.
        pending.sort(key=lambda r: r["request_id"] not in reuse)
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            prompts = [build_chat(tok, user_prompt(r, probs[r["pid"]], cands), enable_thinking=think,
                                  prefill="" if think else PREFILL)
                       for r in chunk]
            fresh = [i for i, r in enumerate(chunk) if r["request_id"] not in reuse]
            gens = {i: dict(reuse[r["request_id"]]) for i, r in enumerate(chunk) if r["request_id"] in reuse}
            if fresh:
                params = ([SamplingParams(temperature=0.6, top_p=0.95, top_k=20, seed=chunk[i]["seed"],
                                          max_tokens=BUDGET[args.mode]) for i in fresh]
                          if args.sample else sampling)
                outs = llm.generate([prompts[i] for i in fresh], params, use_tqdm=False)
                for i, o in zip(fresh, outs, strict=True):
                    g = o.outputs[0]
                    gens[i] = dict(gen=g.text, finish=g.finish_reason, n_out=len(g.token_ids),
                                   n_prompt=len(o.prompt_token_ids))
            parsed = {i: extract(gens[i]["gen"], args.mode, chunk[i]["readout"]) for i in range(len(chunk))}
            need = [i for i in range(len(chunk)) if think and parsed[i][0] is None]
            forced_gen = {}
            if need:
                outs = llm.generate([prompts[i] + gens[i]["gen"] + forced_suffix(gens[i]["gen"]) for i in need],
                                    forcing, use_tqdm=False)
                for i, o in zip(need, outs, strict=True):
                    forced_gen[i] = o.outputs[0].text
                    parsed[i] = (extract(forced_gen[i], "direct", chunk[i]["readout"])[0], "forced")
            for i, r in enumerate(chunk):
                g = gens[i]
                final, rule = parsed[i]
                record = result(r, args.mode, g["gen"], g["finish"], g["n_out"], g["n_prompt"], cands,
                                final, rule, forced=i in forced_gen, forced_gen=forced_gen.get(i))
                if args.sample:
                    record["decoding"] = "sampled"
                sink.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
            sink.flush()
            os.fsync(sink.fileno())
            print(f"Saved {len(done) + start + len(chunk)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
